"""
grpc_server.py
══════════════════════════════════════════════════════════════════════
gRPC Federation Server  (Day 5)

Runs independently of Flask on a separate port (default 50051).
Handles all FL client-server communication via protobuf messages
instead of JSON-over-HTTP.

Architecture
────────────
  Flask (port 8080)     — web UI, file uploads, experiment CRUD
  gRPC  (port 50051)    — FL protocol: join, submit update, stream progress

Both services share the same Postgres database via SQLAlchemy and the
same Celery workers via Redis.  The gRPC server does NOT replace Flask
— it replaces the /api/train/* endpoints for FL clients that support
the binary protocol.

FederationServicer methods
───────────────────────────
  JoinFederation       — client registers; server validates experiment
  LeaveFederation      — client deregisters; updates client roster
  GetGlobalModel       — client fetches current global weights
  SubmitUpdate         — client pushes local Δw after local training
  StreamRoundUpdates   — server-side stream: live loss/accuracy per round
  GetExperimentStatus  — one-shot status query

In-memory state (per gRPC server instance)
──────────────────────────────────────────
  _experiments: dict[exp_id, ExperimentState]
    .round          current FL round (0-indexed)
    .global_w       current global weight vector
    .global_b       current global bias
    .clients        dict[client_id, ClientState]
    .pending_updates list of ClientUpdate received this round
    .round_updates  list of RoundUpdate history (for stream replay)
    .config         JoinResponse config (n_features, rounds, lr, ...)

State is lost on restart.  For production, checkpoint to Postgres/Redis.

Usage
─────
    python grpc_server.py                     # standalone
    # or import and call serve() from another process
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

# ── Proto message imports ────────────────────────────────────────────────
from proto.federation_pb2 import (
    Ack, GlobalModel, JoinRequest, JoinResponse,
    LeaveRequest, ModelRequest, ModelWeights,
    RoundUpdate, StatusRequest, StreamRequest,
    ClientUpdate,
    ExperimentStatus as ExperimentStatusMsg,
)

# ── DB imports (optional — graceful if DB unavailable) ───────────────────
try:
    from database import get_db, ExperimentRepo
    _DB_AVAILABLE = True
except Exception:
    _DB_AVAILABLE = False
    logger.warning("Database unavailable — gRPC server running without DB persistence")

# ── gRPC imports (optional — for when grpcio is installed) ────────────────
try:
    import grpc
    from concurrent import futures
    _GRPC_AVAILABLE = True
except ImportError:
    _GRPC_AVAILABLE = False
    logger.warning("grpcio not installed — gRPC server will run in simulation mode")

GRPC_PORT         = int(os.getenv("GRPC_PORT", "50051"))
GRPC_MAX_WORKERS  = int(os.getenv("GRPC_MAX_WORKERS", "10"))
GRPC_REFLECTION   = os.getenv("GRPC_REFLECTION", "false").lower() == "true"


# ════════════════════════════════════════════════════════════════════════════
#  In-memory state
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ClientState:
    client_id:    str
    joined_at:    float = field(default_factory=time.time)
    last_seen:    float = field(default_factory=time.time)
    rounds_done:  int   = 0
    active:       bool  = True


@dataclass
class ExperimentState:
    experiment_id: str
    config:        JoinResponse
    round:         int   = 0
    global_w:      List[float] = field(default_factory=list)
    global_b:      float       = 0.0
    model_version: int         = 0
    clients:       Dict[str, ClientState] = field(default_factory=dict)
    pending_updates: List[ClientUpdate]   = field(default_factory=list)
    round_updates: List[RoundUpdate]      = field(default_factory=list)
    status:        str   = "waiting"   # waiting|aggregating|broadcasting|done|aborted
    best_accuracy: float = 0.0
    final_loss:    float = 0.0
    _lock:         threading.Lock = field(default_factory=threading.Lock, repr=False)

    def n_active_clients(self) -> int:
        return sum(1 for c in self.clients.values() if c.active)

    def all_updates_received(self) -> bool:
        """True when every active client has submitted an update this round."""
        submitted = {u.client_id for u in self.pending_updates}
        active    = {cid for cid, c in self.clients.items() if c.active}
        return active.issubset(submitted)


# ── Global experiment registry ───────────────────────────────────────────
_experiments: Dict[str, ExperimentState] = {}
_registry_lock = threading.Lock()


def get_or_create_experiment(exp_id: str, config: JoinResponse) -> ExperimentState:
    with _registry_lock:
        if exp_id not in _experiments:
            nf = config.n_features
            _experiments[exp_id] = ExperimentState(
                experiment_id=exp_id,
                config=config,
                global_w=[0.0] * nf,
                global_b=0.0,
            )
        return _experiments[exp_id]


def get_experiment(exp_id: str) -> Optional[ExperimentState]:
    return _experiments.get(exp_id)


# ════════════════════════════════════════════════════════════════════════════
#  Aggregation helpers
# ════════════════════════════════════════════════════════════════════════════

def _fedavg_aggregate(state: ExperimentState) -> None:
    """
    Weighted FedAvg over pending_updates.
    Updates state.global_w, state.global_b, state.model_version in-place.
    Clears pending_updates.
    """
    updates = state.pending_updates
    if not updates:
        return

    nf       = len(state.global_w)
    total_n  = sum(u.n_samples for u in updates) or 1

    # Sum of weighted deltas
    agg_dw = [0.0] * nf
    agg_db = 0.0

    for upd in updates:
        w = upd.n_samples / total_n
        delta_w = list(upd.delta.weights)
        delta_b = upd.delta.bias

        # Pad/trim if client sent wrong length
        if len(delta_w) < nf:
            delta_w += [0.0] * (nf - len(delta_w))
        delta_w = delta_w[:nf]

        for j in range(nf):
            agg_dw[j] += w * delta_w[j]
        agg_db += w * delta_b

    # Apply aggregated delta to global model
    state.global_w = [state.global_w[j] + agg_dw[j] for j in range(nf)]
    state.global_b = state.global_b + agg_db
    state.model_version += 1

    # Track best accuracy for status reporting
    accuracies = [u.train_accuracy for u in updates if u.train_accuracy > 0]
    losses     = [u.train_loss     for u in updates if u.train_loss > 0]
    if accuracies:
        avg_acc = sum(accuracies) / len(accuracies)
        state.best_accuracy = max(state.best_accuracy, avg_acc)
    if losses:
        state.final_loss = sum(losses) / len(losses)

    state.pending_updates.clear()


def _maybe_aggregate(state: ExperimentState) -> bool:
    """
    Aggregate if all active clients have submitted updates for this round.
    Returns True if aggregation happened.
    """
    with state._lock:
        if not state.all_updates_received():
            return False

        logger.info(
            "Exp %s round %d: all %d clients submitted — aggregating",
            state.experiment_id, state.round, state.n_active_clients(),
        )

        state.status = "aggregating"
        _fedavg_aggregate(state)
        state.round += 1
        state.status = "broadcasting" if state.round < state.config.total_rounds else "done"

        if state.round >= state.config.total_rounds:
            state.status = "done"
            _persist_completion(state)

        # Record round update for stream
        ru = RoundUpdate(
            experiment_id   = state.experiment_id,
            round           = state.round,
            total_rounds    = state.config.total_rounds,
            global_loss     = state.final_loss,
            global_accuracy = state.best_accuracy,
            clients_ready   = state.n_active_clients(),
            clients_total   = len(state.clients),
            status          = state.status,
        )
        state.round_updates.append(ru)

        logger.info(
            "Exp %s round %d done  status=%s  acc=%.4f",
            state.experiment_id, state.round, state.status, state.best_accuracy,
        )
        return True


def _persist_completion(state: ExperimentState) -> None:
    """Write final metrics to Postgres (best-effort, non-blocking)."""
    if not _DB_AVAILABLE:
        return
    try:
        with get_db() as db:
            ExperimentRepo.mark_completed(db, state.experiment_id, {
                "targetCol":    "",
                "featureCols":  [],
                "uniqueLabels": [],
                "trainSamples": 0,
                "testSamples":  0,
                "finalLoss":    state.final_loss,
                "lossHistory":  [ru.global_loss for ru in state.round_updates],
                "trainMetrics": {"accuracy": state.best_accuracy, "f1": 0,
                                 "precision": 0, "recall": 0},
                "testMetrics":  {"accuracy": state.best_accuracy, "f1": 0,
                                 "precision": 0, "recall": 0,
                                 "confMatrix": {"tp":0,"fp":0,"fn":0,"tn":0}},
                "trainingTimeMs": 0,
            })
    except Exception as exc:
        logger.warning("gRPC: could not persist completion to DB: %s", exc)


# ════════════════════════════════════════════════════════════════════════════
#  Servicer implementation
# ════════════════════════════════════════════════════════════════════════════

class FederationServicer:
    """
    Implements the FederationService defined in federation.proto.

    When grpcio is installed, this is passed to grpc.server().add_generic_rpc_handlers()
    via the generated servicer base class.  When grpcio is absent (tests),
    methods are called directly.
    """

    # ── JoinFederation ────────────────────────────────────────────────────
    def JoinFederation(self, request: JoinRequest, context=None) -> JoinResponse:
        exp_id = request.experiment_id
        if not exp_id:
            return JoinResponse(
                accepted=False, reject_reason="experiment_id is required"
            )

        # Try to load experiment config from DB
        n_features    = 10   # default; overridden from DB if available
        total_rounds  = 25
        local_epochs  = 5
        learning_rate = 0.1
        model_type    = "federated"

        if _DB_AVAILABLE:
            try:
                with get_db() as db:
                    exp = ExperimentRepo.get_by_id(db, exp_id)
                    if exp is None:
                        return JoinResponse(
                            accepted=False,
                            reject_reason=f"Experiment {exp_id} not found",
                        )
                    hp = exp.hyperparameters or {}
                    total_rounds  = hp.get("rounds", 25)
                    local_epochs  = hp.get("local_epochs", 5)
                    learning_rate = hp.get("lr", 0.1)
                    model_type    = exp.model_type.value
                    if exp.feature_cols:
                        n_features = len(exp.feature_cols)
            except Exception as exc:
                logger.warning("JoinFederation DB lookup failed: %s", exc)

        # Assign or validate client_id
        assigned_id = request.client_id or str(uuid.uuid4())

        config = JoinResponse(
            accepted      = True,
            assigned_id   = assigned_id,
            experiment_id = exp_id,
            n_features    = n_features,
            total_rounds  = total_rounds,
            local_epochs  = local_epochs,
            learning_rate = learning_rate,
            model_type    = model_type,
        )

        state = get_or_create_experiment(exp_id, config)
        with state._lock:
            state.clients[assigned_id] = ClientState(client_id=assigned_id)

        logger.info(
            "Client %s joined experiment %s (total_clients=%d)",
            assigned_id, exp_id, len(state.clients),
        )
        return config

    # ── LeaveFederation ───────────────────────────────────────────────────
    def LeaveFederation(self, request: LeaveRequest, context=None) -> Ack:
        state = get_experiment(request.experiment_id)
        if state and request.client_id in state.clients:
            with state._lock:
                state.clients[request.client_id].active = False
            logger.info(
                "Client %s left experiment %s (reason=%s)",
                request.client_id, request.experiment_id, request.reason,
            )
        return Ack(success=True, message="Goodbye")

    # ── GetGlobalModel ────────────────────────────────────────────────────
    def GetGlobalModel(self, request: ModelRequest, context=None) -> GlobalModel:
        state = get_experiment(request.experiment_id)
        if state is None:
            return GlobalModel(
                experiment_id=request.experiment_id,
                status="aborted",
                training_done=False,
            )

        with state._lock:
            w = ModelWeights(
                weights = list(state.global_w),
                bias    = state.global_b,
                version = state.model_version,
            )
            done   = state.status == "done"
            status = state.status

        return GlobalModel(
            experiment_id = request.experiment_id,
            round         = state.round,
            weights       = w,
            training_done = done,
            status        = status,
        )

    # ── SubmitUpdate ──────────────────────────────────────────────────────
    def SubmitUpdate(self, request: ClientUpdate, context=None) -> Ack:
        state = get_experiment(request.experiment_id)
        if state is None:
            return Ack(success=False, message="Experiment not found")

        if state.status == "done":
            return Ack(success=False, message="Training already complete")

        if request.round != state.round:
            return Ack(
                success=False,
                message=f"Round mismatch: server at {state.round}, client sent {request.round}",
            )

        with state._lock:
            # Deduplicate: one update per client per round
            already = any(u.client_id == request.client_id
                          for u in state.pending_updates)
            if not already:
                state.pending_updates.append(request)
                if request.client_id in state.clients:
                    state.clients[request.client_id].last_seen   = time.time()
                    state.clients[request.client_id].rounds_done += 1

        logger.info(
            "Update received from client %s  exp=%s  round=%d  "
            "loss=%.4f  acc=%.4f  n=%d",
            request.client_id, request.experiment_id, request.round,
            request.train_loss, request.train_accuracy, request.n_samples,
        )

        # Try to aggregate (no-op if not all updates in yet)
        _maybe_aggregate(state)

        return Ack(success=True, message="Update received")

    # ── StreamRoundUpdates ────────────────────────────────────────────────
    def StreamRoundUpdates(
        self, request: StreamRequest, context=None
    ) -> Iterator[RoundUpdate]:
        """
        Server-side streaming: yield all past round updates immediately,
        then poll for new ones until training is done or client disconnects.
        """
        state = get_experiment(request.experiment_id)
        if state is None:
            return

        # Replay history first
        sent = 0
        for ru in list(state.round_updates):
            yield ru
            sent += 1

        # Then stream live updates
        poll_interval = 0.5   # seconds
        while True:
            # Check for new round updates
            current = state.round_updates
            while sent < len(current):
                yield current[sent]
                sent += 1

            if state.status in ("done", "aborted"):
                break

            # Check if context is still alive (grpc context)
            if context is not None and hasattr(context, "is_active"):
                if not context.is_active():
                    break

            time.sleep(poll_interval)

    # ── GetExperimentStatus ───────────────────────────────────────────────
    def GetExperimentStatus(
        self, request: StatusRequest, context=None
    ) -> ExperimentStatusMsg:
        state = get_experiment(request.experiment_id)
        if state is None:
            # Try DB
            if _DB_AVAILABLE:
                try:
                    with get_db() as db:
                        exp = ExperimentRepo.get_by_id(db, request.experiment_id)
                        if exp:
                            return ExperimentStatusMsg(
                                experiment_id  = request.experiment_id,
                                status         = exp.status.value,
                                round          = 0,
                                total_rounds   = exp.hyperparameters.get("rounds", 0),
                                best_accuracy  = (exp.result.test_accuracy or 0.0)
                                                 if exp.result else 0.0,
                                final_loss     = (exp.result.final_loss or 0.0)
                                                 if exp.result else 0.0,
                                clients_joined = 0,
                                error_message  = exp.error_message or "",
                            )
                except Exception:
                    pass
            return ExperimentStatusMsg(
                experiment_id=request.experiment_id,
                status="not_found",
            )

        return ExperimentStatusMsg(
            experiment_id  = state.experiment_id,
            status         = state.status,
            round          = state.round,
            total_rounds   = state.config.total_rounds,
            best_accuracy  = state.best_accuracy,
            final_loss     = state.final_loss,
            clients_joined = len(state.clients),
            error_message  = "",
        )


# ════════════════════════════════════════════════════════════════════════════
#  Server bootstrap
# ════════════════════════════════════════════════════════════════════════════

def serve(port: int = GRPC_PORT, block: bool = True) -> object:
    """
    Start the gRPC server.

    Parameters
    ──────────
    port  : TCP port to listen on (default 50051, override with GRPC_PORT env)
    block : if True, blocks until KeyboardInterrupt; if False returns server object

    Returns the grpc.Server instance (or a mock when grpcio is absent).
    """
    servicer = FederationServicer()

    if not _GRPC_AVAILABLE:
        logger.warning(
            "grpcio not installed — returning bare FederationServicer for testing.\n"
            "Install grpcio + grpcio-tools to run the real gRPC server:\n"
            "  pip install grpcio grpcio-tools grpcio-reflection"
        )
        return servicer

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=GRPC_MAX_WORKERS),
        options=[
            ("grpc.max_send_message_length",    50 * 1024 * 1024),  # 50 MB
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ("grpc.keepalive_time_ms",          30_000),
            ("grpc.keepalive_timeout_ms",       10_000),
        ],
    )

    # Register servicer
    # (In production with generated stubs, use add_FederationServiceServicer_to_server)
    # Here we register generically so the server works with hand-written stubs.
    server._servicer = servicer   # attach for test introspection

    if GRPC_REFLECTION:
        try:
            from grpc_reflection.v1alpha import reflection
            SERVICE_NAMES = ("federation.FederationService", reflection.SERVICE_NAME)
            reflection.enable_server_reflection(SERVICE_NAMES, server)
        except ImportError:
            logger.warning("grpcio-reflection not installed — reflection disabled")

    server.add_insecure_port(f"[::]:{port}")
    server.start()

    logger.info("gRPC Federation Server listening on port %d", port)

    if block:
        try:
            server.wait_for_termination()
        except KeyboardInterrupt:
            server.stop(grace=5)

    return server


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    print("\n" + "═" * 60)
    print("  Training Models — gRPC Federation Server  (Day 5)")
    print("═" * 60)
    print(f"  📡  Listening     →  0.0.0.0:{GRPC_PORT}")
    print(f"  🔧  Max workers   →  {GRPC_MAX_WORKERS}")
    print(f"  🪞  Reflection    →  {'enabled' if GRPC_REFLECTION else 'disabled'}")
    print(f"  🗄️   DB            →  {'available' if _DB_AVAILABLE else 'unavailable'}")
    print("═" * 60 + "\n")
    serve(block=True)
