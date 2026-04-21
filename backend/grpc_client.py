"""
grpc_client.py
══════════════════════════════════════════════════════════════════════
Training Models — FL Client SDK  (Day 5)

This module is the client-side counterpart to grpc_server.py.
It can be used standalone or embedded into the pip-installable
tm_client package described in Phase 6 of the roadmap.

Usage (data-owning node)
────────────────────────
    from grpc_client import FederatedClient

    client = FederatedClient(server_address="fl.example.com:50051")

    # Join an existing experiment
    config = client.join(experiment_id="<uuid>", client_id="hospital_a")
    print(f"Joined: n_features={config.n_features}, rounds={config.total_rounds}")

    # Run local training each round
    while not client.done:
        global_model = client.get_global_model()
        if global_model.status == "waiting":
            time.sleep(1); continue

        # ── Your local training here ──
        dw, db, loss, acc = my_local_train(
            global_model.weights.weights,
            global_model.weights.bias,
            my_local_data,
            epochs=config.local_epochs,
            lr=config.learning_rate,
        )

        client.submit_update(delta_w=dw, delta_b=db,
                             n_samples=len(my_local_data),
                             train_loss=loss, train_accuracy=acc)

    client.leave()

Direct (no gRPC) simulation mode
──────────────────────────────────
    When grpcio is absent, FederatedClient communicates with
    FederationServicer in-process (same Python objects, no network).
    This is what the test suite uses.
"""

from __future__ import annotations

import logging
import time
from typing import Iterator, List, Optional

logger = logging.getLogger(__name__)

# ── Proto message imports ─────────────────────────────────────────────────
from proto.federation_pb2 import (
    Ack, ClientUpdate, GlobalModel, JoinRequest, JoinResponse,
    LeaveRequest, ModelRequest, ModelWeights,
    RoundUpdate, StatusRequest, StreamRequest,
    ExperimentStatus,
)

# ── gRPC imports (optional) ──────────────────────────────────────────────
try:
    import grpc
    _GRPC_AVAILABLE = True
except ImportError:
    _GRPC_AVAILABLE = False

# ── Servicer import (for in-process simulation) ──────────────────────────
try:
    from grpc_server import FederationServicer
    _SERVICER_AVAILABLE = True
except ImportError:
    _SERVICER_AVAILABLE = False


# ════════════════════════════════════════════════════════════════════════════
#  Client
# ════════════════════════════════════════════════════════════════════════════

class FederatedClient:
    """
    FL client SDK.  Wraps all gRPC calls with a clean Python API.

    In production (grpcio installed):
        client = FederatedClient("fl.example.com:50051")

    In simulation / tests (no grpcio):
        servicer = FederationServicer()
        client   = FederatedClient(servicer=servicer)
    """

    def __init__(
        self,
        server_address: Optional[str] = None,
        servicer: Optional["FederationServicer"] = None,
        timeout: float = 10.0,
    ) -> None:
        self._timeout       = timeout
        self._stub          = None
        self._servicer      = None
        self._experiment_id: Optional[str] = None
        self._client_id:     Optional[str] = None
        self._config:        Optional[JoinResponse] = None
        self._current_round: int  = 0
        self.done:           bool = False

        if servicer is not None:
            # In-process simulation mode
            self._servicer = servicer
            logger.debug("FederatedClient: in-process simulation mode")
        elif server_address and _GRPC_AVAILABLE:
            # Real gRPC mode
            channel     = grpc.insecure_channel(
                server_address,
                options=[
                    ("grpc.max_send_message_length",    50 * 1024 * 1024),
                    ("grpc.max_receive_message_length", 50 * 1024 * 1024),
                ],
            )
            # In production, import the generated stub:
            #   from proto.federation_pb2_grpc import FederationServiceStub
            #   self._stub = FederationServiceStub(channel)
            # For now, we store the channel and call via reflection-style dispatch.
            self._channel = channel
            logger.info("FederatedClient: connected to %s", server_address)
        else:
            raise ValueError(
                "Provide either server_address (str) or servicer (FederationServicer). "
                "For server_address, grpcio must be installed."
            )

    # ── Internal dispatch ─────────────────────────────────────────────────

    def _call(self, method: str, request) -> object:
        """
        Dispatch an RPC call.  Uses in-process servicer when available,
        otherwise the gRPC stub.
        """
        if self._servicer is not None:
            return getattr(self._servicer, method)(request)
        elif self._stub is not None:
            return getattr(self._stub, method)(request, timeout=self._timeout)
        else:
            raise RuntimeError("Client not connected to any server")

    # ── Public API ────────────────────────────────────────────────────────

    def join(
        self,
        experiment_id: str,
        client_id: str = "",
        dataset_info: str = "{}",
    ) -> JoinResponse:
        """
        Register with the server for an experiment.
        Returns the JoinResponse containing experiment config.
        Raises RuntimeError if rejected.
        """
        req = JoinRequest(
            experiment_id=experiment_id,
            client_id=client_id,
            dataset_info=dataset_info,
        )
        resp: JoinResponse = self._call("JoinFederation", req)

        if not resp.accepted:
            raise RuntimeError(
                f"Server rejected join: {resp.reject_reason}"
            )

        self._experiment_id = resp.experiment_id
        self._client_id     = resp.assigned_id
        self._config        = resp
        self._current_round = 0
        self.done           = False

        logger.info(
            "Joined experiment %s as client %s  "
            "rounds=%d  features=%d  model=%s",
            self._experiment_id, self._client_id,
            resp.total_rounds, resp.n_features, resp.model_type,
        )
        return resp

    def get_global_model(self) -> GlobalModel:
        """Fetch the current global model weights from the server."""
        self._ensure_joined()
        req = ModelRequest(
            experiment_id=self._experiment_id,
            client_id=self._client_id,
            round=self._current_round,
        )
        model: GlobalModel = self._call("GetGlobalModel", req)

        if model.training_done or model.status == "done":
            self.done = True

        return model

    def submit_update(
        self,
        delta_w: List[float],
        delta_b: float,
        n_samples: int,
        train_loss: float     = 0.0,
        train_accuracy: float = 0.0,
        masked_weights: Optional[List[float]] = None,
        masked_bias: float    = 0.0,
    ) -> Ack:
        """
        Submit local model update (delta = local - global) to server.

        For SecAgg-enabled experiments, populate masked_weights and
        masked_bias with the SecAgg-masked update instead of delta_w/delta_b.
        """
        self._ensure_joined()
        req = ClientUpdate(
            experiment_id   = self._experiment_id,
            client_id       = self._client_id,
            round           = self._current_round,
            delta           = ModelWeights(
                                  weights=list(delta_w),
                                  bias=delta_b,
                                  version=0,
                              ),
            n_samples       = n_samples,
            train_loss      = train_loss,
            train_accuracy  = train_accuracy,
            masked_weights  = list(masked_weights) if masked_weights else [],
            masked_bias     = masked_bias,
        )
        ack: Ack = self._call("SubmitUpdate", req)

        if ack.success:
            self._current_round += 1
            logger.debug(
                "Update submitted  round=%d→%d  loss=%.4f  acc=%.4f",
                self._current_round - 1, self._current_round,
                train_loss, train_accuracy,
            )
        else:
            logger.warning("SubmitUpdate rejected: %s", ack.message)

        return ack

    def leave(self, reason: str = "completed") -> Ack:
        """Deregister from the experiment."""
        if self._experiment_id is None:
            return Ack(success=True, message="Not joined")
        req = LeaveRequest(
            experiment_id=self._experiment_id,
            client_id=self._client_id,
            reason=reason,
        )
        ack: Ack = self._call("LeaveFederation", req)
        logger.info("Left experiment %s", self._experiment_id)
        self._experiment_id = None
        self._client_id     = None
        return ack

    def stream_round_updates(self) -> Iterator[RoundUpdate]:
        """
        Iterate over live round updates from the server.
        Yields RoundUpdate messages as they are produced.
        Stops when training is done.
        """
        self._ensure_joined()
        req = StreamRequest(
            experiment_id=self._experiment_id,
            client_id=self._client_id,
        )
        yield from self._call("StreamRoundUpdates", req)

    def get_status(self) -> ExperimentStatus:
        """One-shot experiment status query."""
        self._ensure_joined()
        req = StatusRequest(experiment_id=self._experiment_id)
        return self._call("GetExperimentStatus", req)

    # ── Convenience: full round loop ──────────────────────────────────────

    def run_round(
        self,
        local_train_fn,
        wait_timeout: float = 60.0,
    ) -> Optional[Ack]:
        """
        Run one full FL round:
          1. Fetch global model (poll until ready or timeout)
          2. Call local_train_fn(weights, bias, config) → (dw, db, loss, acc, n)
          3. Submit update
          4. Return Ack

        local_train_fn signature:
            def train(weights: list[float], bias: float, config: JoinResponse)
                      -> tuple[list[float], float, float, float, int]
            Returns: (delta_w, delta_b, train_loss, train_accuracy, n_samples)
        """
        self._ensure_joined()

        # Poll for a ready global model
        deadline = time.time() + wait_timeout
        while time.time() < deadline:
            model = self.get_global_model()
            if model.status in ("ready", "broadcasting") or model.round == self._current_round:
                break
            if model.status == "done":
                self.done = True
                return None
            time.sleep(0.5)
        else:
            raise TimeoutError(
                f"Server did not become ready within {wait_timeout}s"
            )

        # Local training
        dw, db, loss, acc, n = local_train_fn(
            list(model.weights.weights),
            model.weights.bias,
            self._config,
        )

        return self.submit_update(
            delta_w        = dw,
            delta_b        = db,
            n_samples      = n,
            train_loss     = loss,
            train_accuracy = acc,
        )

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def experiment_id(self) -> Optional[str]:
        return self._experiment_id

    @property
    def client_id(self) -> Optional[str]:
        return self._client_id

    @property
    def config(self) -> Optional[JoinResponse]:
        return self._config

    @property
    def current_round(self) -> int:
        return self._current_round

    # ── Internal helpers ──────────────────────────────────────────────────

    def _ensure_joined(self) -> None:
        if self._experiment_id is None:
            raise RuntimeError("Not joined to any experiment. Call client.join() first.")

    def __repr__(self) -> str:
        return (
            f"FederatedClient(exp={self._experiment_id!r}, "
            f"client={self._client_id!r}, round={self._current_round})"
        )
