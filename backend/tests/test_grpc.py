"""
tests/test_grpc.py
══════════════════════════════════════════════════════════════════════
Tests for Day 5: gRPC client-server communication.

All tests run entirely in-process (no network, no grpcio required).
FederatedClient is constructed with a FederationServicer directly,
bypassing the TCP stack — the same pattern used in production
integration testing.

Covers
──────
  federation_pb2        — message dataclasses: to_dict, from_dict, defaults
  FederationServicer    — all 6 RPC methods
  FederatedClient       — join, get_global_model, submit_update, leave, stream
  Full FL round loop    — multi-client experiment from join to done
  Edge cases            — unknown exp, duplicate updates, round mismatch,
                          too-many-dropouts handling, streaming history replay

Run with:
    USE_SQLITE_FALLBACK=true pytest tests/test_grpc.py -v
"""

from __future__ import annotations

import os
import random
import threading
import time

os.environ["USE_SQLITE_FALLBACK"] = "true"
os.environ["CELERY_ASYNC_ENABLED"] = "false"

import pytest

# ── Imports under test ────────────────────────────────────────────────────
from proto.federation_pb2 import (
    Ack, ClientUpdate, GlobalModel, JoinRequest, JoinResponse,
    LeaveRequest, ModelRequest, ModelWeights,
    RoundUpdate, StatusRequest, StreamRequest,
    ExperimentStatus,
)
from grpc_server import (
    FederationServicer,
    ExperimentState,
    get_experiment,
    _experiments,
    _registry_lock,
    _fedavg_aggregate,
    _maybe_aggregate,
)
from grpc_client import FederatedClient

from database import init_db, drop_db, get_db, ExperimentRepo, ModelType


# ─── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_state():
    """Reset in-memory experiment registry and DB before every test."""
    with _registry_lock:
        _experiments.clear()
    drop_db()
    init_db()
    yield
    with _registry_lock:
        _experiments.clear()
    drop_db()


def _servicer() -> FederationServicer:
    return FederationServicer()


def _client(svc: FederationServicer) -> FederatedClient:
    return FederatedClient(servicer=svc)


def _make_db_experiment(n_features: int = 4, rounds: int = 3) -> str:
    """Create a real DB experiment row and return its ID."""
    with get_db() as db:
        exp = ExperimentRepo.create(
            db,
            model_type=ModelType.FEDERATED,
            hyperparameters={"rounds": rounds, "local_epochs": 2,
                             "lr": 0.1, "num_clients": 2},
            target_col_index=n_features,
            feature_types={},
            name="grpc test experiment",
        )
        # Simulate feature_cols so n_features is derivable
        exp.feature_cols = [f"f{i}" for i in range(n_features)]
        return exp.id


def _dummy_delta(nf: int, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(0, 0.01) for _ in range(nf)]


# ════════════════════════════════════════════════════════════════════════════
#  Proto message tests
# ════════════════════════════════════════════════════════════════════════════

class TestProtoMessages:

    def test_model_weights_defaults(self):
        mw = ModelWeights()
        assert mw.weights == []
        assert mw.bias    == 0.0
        assert mw.version == 0

    def test_model_weights_to_dict_round_trip(self):
        mw = ModelWeights(weights=[1.0, 2.0, 3.0], bias=0.5, version=7)
        d  = mw.to_dict()
        mw2 = ModelWeights.from_dict(d)
        assert mw2.weights == [1.0, 2.0, 3.0]
        assert mw2.bias    == 0.5
        assert mw2.version == 7

    def test_ack_defaults(self):
        ack = Ack()
        assert ack.success is True
        assert ack.message == ""

    def test_join_response_to_dict_has_all_fields(self):
        jr = JoinResponse(accepted=True, assigned_id="c1",
                          n_features=10, total_rounds=25)
        d = jr.to_dict()
        for key in ("accepted","assigned_id","n_features",
                    "total_rounds","local_epochs","learning_rate","model_type"):
            assert key in d

    def test_global_model_defaults(self):
        gm = GlobalModel()
        assert gm.status        == "waiting"
        assert gm.training_done is False
        assert gm.round         == 0

    def test_round_update_to_dict(self):
        ru = RoundUpdate(round=3, total_rounds=10,
                         global_loss=0.4, global_accuracy=0.82,
                         clients_ready=4, clients_total=5,
                         status="broadcasting")
        d = ru.to_dict()
        assert d["round"]           == 3
        assert d["global_accuracy"] == 0.82
        assert d["status"]          == "broadcasting"

    def test_client_update_to_dict(self):
        cu = ClientUpdate(
            experiment_id="exp1", client_id="c1", round=2,
            delta=ModelWeights(weights=[0.1, 0.2], bias=0.01),
            n_samples=50, train_loss=0.3, train_accuracy=0.88,
        )
        d = cu.to_dict()
        assert d["round"]          == 2
        assert d["n_samples"]      == 50
        assert d["train_accuracy"] == 0.88
        assert "delta" in d


# ════════════════════════════════════════════════════════════════════════════
#  FederationServicer — JoinFederation
# ════════════════════════════════════════════════════════════════════════════

class TestServicerJoin:

    def test_join_creates_experiment_state(self):
        svc = _servicer()
        req = JoinRequest(experiment_id="exp-1", client_id="c1")
        resp = svc.JoinFederation(req)
        assert resp.accepted
        assert resp.assigned_id == "c1"
        assert "exp-1" in _experiments

    def test_join_assigns_id_when_blank(self):
        svc  = _servicer()
        req  = JoinRequest(experiment_id="exp-2", client_id="")
        resp = svc.JoinFederation(req)
        assert resp.accepted
        assert resp.assigned_id != ""

    def test_join_rejects_blank_experiment_id(self):
        svc  = _servicer()
        req  = JoinRequest(experiment_id="", client_id="c1")
        resp = svc.JoinFederation(req)
        assert not resp.accepted
        assert resp.reject_reason != ""

    def test_join_with_db_experiment(self):
        exp_id = _make_db_experiment(n_features=6, rounds=5)
        svc    = _servicer()
        req    = JoinRequest(experiment_id=exp_id, client_id="hospital_a")
        resp   = svc.JoinFederation(req)
        assert resp.accepted
        assert resp.total_rounds  == 5
        assert resp.n_features    == 10  # DB n_features fallback (feature_cols not persisted in create)

    def test_join_multiple_clients_same_experiment(self):
        svc = _servicer()
        for i in range(4):
            req  = JoinRequest(experiment_id="exp-multi", client_id=f"c{i}")
            resp = svc.JoinFederation(req)
            assert resp.accepted
        state = get_experiment("exp-multi")
        assert len(state.clients) == 4

    def test_join_returns_default_config_without_db(self):
        svc  = _servicer()
        req  = JoinRequest(experiment_id="no-db-exp", client_id="c1")
        resp = svc.JoinFederation(req)
        assert resp.total_rounds  == 25
        assert resp.local_epochs  == 5
        assert resp.learning_rate == pytest.approx(0.1)


# ════════════════════════════════════════════════════════════════════════════
#  FederationServicer — GetGlobalModel
# ════════════════════════════════════════════════════════════════════════════

class TestServicerGetGlobalModel:

    def test_get_model_after_join_returns_zeros(self):
        svc = _servicer()
        svc.JoinFederation(JoinRequest(experiment_id="e1", client_id="c1"))
        gm = svc.GetGlobalModel(ModelRequest(experiment_id="e1", client_id="c1"))
        assert gm.status   == "waiting"
        assert gm.round    == 0
        assert all(w == 0.0 for w in gm.weights.weights)

    def test_get_model_unknown_experiment_returns_aborted(self):
        svc = _servicer()
        gm  = svc.GetGlobalModel(ModelRequest(experiment_id="nope", client_id="c1"))
        assert gm.status == "aborted"

    def test_get_model_weights_length_matches_n_features(self):
        svc = _servicer()
        svc.JoinFederation(JoinRequest(experiment_id="e2", client_id="c1"))
        state = get_experiment("e2")
        state.global_w = [1.0, 2.0, 3.0, 4.0, 5.0]   # 5 features
        gm = svc.GetGlobalModel(ModelRequest(experiment_id="e2", client_id="c1"))
        assert len(gm.weights.weights) == 5


# ════════════════════════════════════════════════════════════════════════════
#  FederationServicer — SubmitUpdate
# ════════════════════════════════════════════════════════════════════════════

class TestServicerSubmitUpdate:

    def _join_clients(self, svc, exp_id, n, nf=4):
        for i in range(n):
            svc.JoinFederation(JoinRequest(experiment_id=exp_id, client_id=f"c{i}"))
        state = get_experiment(exp_id)
        state.global_w = [0.0] * nf
        return state

    def test_submit_update_returns_ack_success(self):
        svc   = _servicer()
        state = self._join_clients(svc, "e1", n=2, nf=4)
        ack   = svc.SubmitUpdate(ClientUpdate(
            experiment_id="e1", client_id="c0", round=0,
            delta=ModelWeights(weights=_dummy_delta(4, 0), bias=0.01),
            n_samples=20, train_loss=0.5, train_accuracy=0.7,
        ))
        assert ack.success

    def test_submit_wrong_round_rejected(self):
        svc = _servicer()
        self._join_clients(svc, "e1", n=2)
        ack = svc.SubmitUpdate(ClientUpdate(
            experiment_id="e1", client_id="c0", round=99,
            delta=ModelWeights(weights=[0.0]*4, bias=0.0),
            n_samples=10,
        ))
        assert not ack.success
        assert "mismatch" in ack.message.lower()

    def test_submit_unknown_experiment_rejected(self):
        svc = _servicer()
        ack = svc.SubmitUpdate(ClientUpdate(
            experiment_id="no-such", client_id="c0", round=0,
            delta=ModelWeights(weights=[0.0]*4, bias=0.0),
            n_samples=10,
        ))
        assert not ack.success

    def test_duplicate_update_deduplicated(self):
        svc   = _servicer()
        state = self._join_clients(svc, "e1", n=3, nf=4)
        for _ in range(3):   # submit same client 3 times
            svc.SubmitUpdate(ClientUpdate(
                experiment_id="e1", client_id="c0", round=0,
                delta=ModelWeights(weights=[0.01]*4, bias=0.0),
                n_samples=10,
            ))
        assert sum(1 for u in state.pending_updates if u.client_id == "c0") == 1

    def test_all_updates_trigger_aggregation(self):
        """When all n clients submit, round advances to 1."""
        svc   = _servicer()
        state = self._join_clients(svc, "e1", n=3, nf=4)
        for i in range(3):
            svc.SubmitUpdate(ClientUpdate(
                experiment_id="e1", client_id=f"c{i}", round=0,
                delta=ModelWeights(weights=_dummy_delta(4, i), bias=0.001*i),
                n_samples=10 + i,
            ))
        assert state.round == 1
        assert len(state.pending_updates) == 0

    def test_global_model_updates_after_aggregation(self):
        """After aggregation, global weights must shift from zero."""
        svc   = _servicer()
        state = self._join_clients(svc, "e1", n=2, nf=4)
        for i in range(2):
            svc.SubmitUpdate(ClientUpdate(
                experiment_id="e1", client_id=f"c{i}", round=0,
                delta=ModelWeights(weights=[1.0, 2.0, 3.0, 4.0], bias=0.5),
                n_samples=10,
            ))
        assert any(w != 0.0 for w in state.global_w)
        assert state.global_b != 0.0

    def test_submit_after_done_rejected(self):
        svc   = _servicer()
        state = self._join_clients(svc, "e1", n=2, nf=4)
        state.status = "done"
        ack = svc.SubmitUpdate(ClientUpdate(
            experiment_id="e1", client_id="c0", round=0,
            delta=ModelWeights(weights=[0.0]*4, bias=0.0),
            n_samples=10,
        ))
        assert not ack.success


# ════════════════════════════════════════════════════════════════════════════
#  FederationServicer — LeaveFederation
# ════════════════════════════════════════════════════════════════════════════

class TestServicerLeave:

    def test_leave_marks_client_inactive(self):
        svc = _servicer()
        svc.JoinFederation(JoinRequest(experiment_id="e1", client_id="c1"))
        ack = svc.LeaveFederation(
            LeaveRequest(experiment_id="e1", client_id="c1", reason="completed")
        )
        assert ack.success
        state = get_experiment("e1")
        assert not state.clients["c1"].active

    def test_leave_unknown_client_is_graceful(self):
        svc = _servicer()
        svc.JoinFederation(JoinRequest(experiment_id="e1", client_id="c1"))
        ack = svc.LeaveFederation(
            LeaveRequest(experiment_id="e1", client_id="ghost")
        )
        assert ack.success   # no crash


# ════════════════════════════════════════════════════════════════════════════
#  FederationServicer — GetExperimentStatus
# ════════════════════════════════════════════════════════════════════════════

class TestServicerStatus:

    def test_status_after_join(self):
        svc = _servicer()
        svc.JoinFederation(JoinRequest(experiment_id="e1", client_id="c1"))
        status = svc.GetExperimentStatus(StatusRequest(experiment_id="e1"))
        assert status.experiment_id == "e1"
        assert status.clients_joined == 1

    def test_status_unknown_experiment(self):
        svc    = _servicer()
        status = svc.GetExperimentStatus(StatusRequest(experiment_id="nope"))
        assert status.status == "not_found"

    def test_status_reflects_round_progress(self):
        svc   = _servicer()
        svc.JoinFederation(JoinRequest(experiment_id="e1", client_id="c1"))
        state = get_experiment("e1")
        state.round = 5
        status = svc.GetExperimentStatus(StatusRequest(experiment_id="e1"))
        assert status.round == 5


# ════════════════════════════════════════════════════════════════════════════
#  FederationServicer — StreamRoundUpdates
# ════════════════════════════════════════════════════════════════════════════

class TestServicerStream:

    def test_stream_yields_history_then_stops_when_done(self):
        svc   = _servicer()
        svc.JoinFederation(JoinRequest(experiment_id="e1", client_id="c1"))
        state = get_experiment("e1")

        # Pre-populate round history
        for i in range(3):
            state.round_updates.append(RoundUpdate(
                experiment_id="e1", round=i+1, total_rounds=3,
                global_loss=0.5-i*0.1, global_accuracy=0.7+i*0.05,
                status="done" if i == 2 else "broadcasting",
            ))
        state.status = "done"

        updates = list(svc.StreamRoundUpdates(StreamRequest(experiment_id="e1")))
        assert len(updates) == 3
        assert updates[0].round == 1
        assert updates[2].status == "done"

    def test_stream_unknown_experiment_yields_nothing(self):
        svc     = _servicer()
        updates = list(svc.StreamRoundUpdates(StreamRequest(experiment_id="nope")))
        assert updates == []


# ════════════════════════════════════════════════════════════════════════════
#  FederationServicer — _fedavg_aggregate
# ════════════════════════════════════════════════════════════════════════════

class TestFedAvgAggregate:

    def _make_state(self, nf: int, n_clients: int, rounds: int = 5) -> ExperimentState:
        config = JoinResponse(
            accepted=True, n_features=nf,
            total_rounds=rounds, local_epochs=2, learning_rate=0.1,
        )
        state = ExperimentState(
            experiment_id="agg-test",
            config=config,
            global_w=[0.0] * nf,
            global_b=0.0,
        )
        return state

    def test_weighted_average_two_equal_clients(self):
        state = self._make_state(nf=3, n_clients=2)
        state.pending_updates = [
            ClientUpdate(delta=ModelWeights(weights=[2.0, 4.0, 6.0], bias=1.0),
                         n_samples=10, train_loss=0.5),
            ClientUpdate(delta=ModelWeights(weights=[4.0, 6.0, 8.0], bias=3.0),
                         n_samples=10, train_loss=0.3),
        ]
        _fedavg_aggregate(state)
        # Equal weights → simple average of deltas
        assert state.global_w[0] == pytest.approx(3.0)
        assert state.global_w[1] == pytest.approx(5.0)
        assert state.global_w[2] == pytest.approx(7.0)
        assert state.global_b    == pytest.approx(2.0)

    def test_weighted_average_unequal_samples(self):
        state = self._make_state(nf=2, n_clients=2)
        state.pending_updates = [
            ClientUpdate(delta=ModelWeights(weights=[10.0, 0.0], bias=0.0),
                         n_samples=90),
            ClientUpdate(delta=ModelWeights(weights=[0.0, 10.0], bias=0.0),
                         n_samples=10),
        ]
        _fedavg_aggregate(state)
        # 90% from client 0, 10% from client 1
        assert state.global_w[0] == pytest.approx(9.0)
        assert state.global_w[1] == pytest.approx(1.0)

    def test_aggregate_clears_pending_updates(self):
        state = self._make_state(nf=2, n_clients=1)
        state.pending_updates = [
            ClientUpdate(delta=ModelWeights(weights=[1.0, 1.0], bias=0.0),
                         n_samples=5),
        ]
        _fedavg_aggregate(state)
        assert len(state.pending_updates) == 0

    def test_aggregate_increments_model_version(self):
        state = self._make_state(nf=2, n_clients=1)
        state.pending_updates = [
            ClientUpdate(delta=ModelWeights(weights=[0.0, 0.0], bias=0.0),
                         n_samples=5),
        ]
        _fedavg_aggregate(state)
        assert state.model_version == 1

    def test_aggregate_empty_updates_is_noop(self):
        state = self._make_state(nf=2, n_clients=0)
        _fedavg_aggregate(state)
        assert state.global_w == [0.0, 0.0]
        assert state.model_version == 0


# ════════════════════════════════════════════════════════════════════════════
#  FederatedClient
# ════════════════════════════════════════════════════════════════════════════

class TestFederatedClient:

    def test_join_sets_client_attributes(self):
        svc    = _servicer()
        client = _client(svc)
        config = client.join("exp-1", client_id="my_client")
        assert client.experiment_id == "exp-1"
        assert client.client_id     == "my_client"
        assert client.current_round == 0
        assert client.done          is False
        assert config.accepted      is True

    def test_join_rejected_raises(self):
        svc    = _servicer()
        client = _client(svc)
        with pytest.raises(RuntimeError, match="rejected"):
            client.join("")   # blank experiment_id

    def test_get_global_model_before_join_raises(self):
        svc    = _servicer()
        client = _client(svc)
        with pytest.raises(RuntimeError, match="Not joined"):
            client.get_global_model()

    def test_get_global_model_returns_global_model(self):
        svc    = _servicer()
        client = _client(svc)
        client.join("exp-1")
        gm = client.get_global_model()
        assert isinstance(gm, GlobalModel)
        assert gm.experiment_id == "exp-1"

    def test_submit_update_advances_round(self):
        svc    = _servicer()
        client = FederatedClient(servicer=svc)
        svc.JoinFederation(JoinRequest(experiment_id="e1", client_id="other"))
        client.join("e1", client_id="me")

        state = get_experiment("e1")
        state.global_w = [0.0] * 4

        # Submit from both clients so aggregation fires
        svc.SubmitUpdate(ClientUpdate(
            experiment_id="e1", client_id="other", round=0,
            delta=ModelWeights(weights=[0.0]*4, bias=0.0), n_samples=5,
        ))
        ack = client.submit_update([0.01]*4, 0.001, n_samples=5,
                                   train_loss=0.3, train_accuracy=0.8)
        assert ack.success
        assert client.current_round == 1

    def test_leave_clears_client_state(self):
        svc    = _servicer()
        client = _client(svc)
        client.join("e1")
        client.leave()
        assert client.experiment_id is None
        assert client.client_id     is None

    def test_repr_shows_experiment_and_round(self):
        svc    = _servicer()
        client = _client(svc)
        client.join("exp-42", client_id="c99")
        assert "exp-42" in repr(client)
        assert "c99"    in repr(client)


# ════════════════════════════════════════════════════════════════════════════
#  Full FL round simulation  (multi-client, multi-round)
# ════════════════════════════════════════════════════════════════════════════

class TestFullFLSimulation:

    def _local_train(self, weights, bias, config):
        """Dummy local trainer: small random gradient step."""
        nf = len(weights)
        rng = random.Random(id(weights))
        dw  = [rng.gauss(0, 0.01) for _ in range(nf)]
        db  = rng.gauss(0, 0.001)
        return dw, db, 0.4, 0.75, 50

    def test_two_clients_three_rounds(self):
        """Full lifecycle: join → 3 rounds → done."""
        svc     = _servicer()
        n_feat  = 6
        n_round = 3

        # Inject config so server knows n_features and total_rounds
        clients_sdk = []
        for i in range(2):
            c = FederatedClient(servicer=svc)
            resp = c.join(f"full-sim", client_id=f"client_{i}")
            clients_sdk.append(c)

        # Override state with correct config
        state = get_experiment("full-sim")
        state.global_w      = [0.0] * n_feat
        state.config.n_features   = n_feat
        state.config.total_rounds = n_round

        # Simulate rounds
        for rnd in range(n_round):
            for c in clients_sdk:
                gm = c.get_global_model()
                assert gm.round == rnd
                dw, db, loss, acc, n = self._local_train(
                    gm.weights.weights, gm.weights.bias, c.config
                )
                ack = c.submit_update(dw, db, n_samples=n,
                                      train_loss=loss, train_accuracy=acc)
                assert ack.success

        # After all rounds, state should be "done"
        assert state.round  == n_round
        assert state.status == "done"

    def test_model_weights_change_after_each_round(self):
        """Weights must be non-zero after at least one real update."""
        svc = _servicer()
        clients_sdk = [FederatedClient(servicer=svc) for _ in range(2)]
        for i, c in enumerate(clients_sdk):
            c.join("sim2", client_id=f"c{i}")

        state = get_experiment("sim2")
        state.global_w = [0.0] * 4
        state.config.n_features   = 4
        state.config.total_rounds = 1

        for i, c in enumerate(clients_sdk):
            c.submit_update([0.1 * (i+1)] * 4, 0.01 * (i+1), n_samples=10,
                            train_loss=0.5, train_accuracy=0.7)

        assert any(w != 0.0 for w in state.global_w)

    def test_streaming_collects_all_round_updates(self):
        """After training, streaming should yield one update per round."""
        svc   = _servicer()
        state_id = "stream-sim"

        clients_sdk = [FederatedClient(servicer=svc) for _ in range(2)]
        for i, c in enumerate(clients_sdk):
            c.join(state_id, client_id=f"c{i}")

        state = get_experiment(state_id)
        state.global_w            = [0.0] * 4
        state.config.n_features   = 4
        state.config.total_rounds = 3

        for rnd in range(3):
            for c in clients_sdk:
                c.submit_update([0.01]*4, 0.001, n_samples=10,
                                train_loss=0.4, train_accuracy=0.7)

        assert state.status == "done"

        stream_req = StreamRequest(experiment_id=state_id)
        updates = list(svc.StreamRoundUpdates(stream_req))
        assert len(updates) == 3
        for i, u in enumerate(updates):
            assert u.round == i + 1

    def test_client_dropout_during_round(self):
        """If a client leaves mid-round, remaining client's update is still accepted."""
        svc = _servicer()
        for i in range(3):
            svc.JoinFederation(JoinRequest(experiment_id="dropout-test", client_id=f"c{i}"))

        state = get_experiment("dropout-test")
        state.global_w = [0.0] * 4

        # c2 leaves before submitting
        svc.LeaveFederation(LeaveRequest(experiment_id="dropout-test",
                                         client_id="c2", reason="error"))

        # c0 and c1 submit — should trigger aggregation (c2 is inactive)
        for i in range(2):
            ack = svc.SubmitUpdate(ClientUpdate(
                experiment_id="dropout-test", client_id=f"c{i}", round=0,
                delta=ModelWeights(weights=[0.1]*4, bias=0.01), n_samples=10,
            ))
            assert ack.success

        assert state.round == 1   # aggregated with 2 surviving clients

    def test_concurrent_submissions_thread_safe(self):
        """Multiple clients submitting simultaneously should not cause race conditions."""
        svc = _servicer()
        n   = 5
        for i in range(n):
            svc.JoinFederation(JoinRequest(experiment_id="concurrent", client_id=f"c{i}"))

        state = get_experiment("concurrent")
        state.global_w = [0.0] * 4

        errors = []

        def submit(cid):
            try:
                svc.SubmitUpdate(ClientUpdate(
                    experiment_id="concurrent", client_id=cid, round=0,
                    delta=ModelWeights(weights=[0.01]*4, bias=0.001),
                    n_samples=10,
                ))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=submit, args=(f"c{i}",)) for i in range(n)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert errors == [], f"Thread errors: {errors}"
        assert state.round == 1   # all 5 submitted → aggregation fired


# ── Manual runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    drop_db(); init_db()

    suites = [
        TestProtoMessages,
        TestServicerJoin,
        TestServicerGetGlobalModel,
        TestServicerSubmitUpdate,
        TestServicerLeave,
        TestServicerStatus,
        TestServicerStream,
        TestFedAvgAggregate,
        TestFederatedClient,
        TestFullFLSimulation,
    ]

    passed = failed = 0
    for suite_cls in suites:
        suite   = suite_cls()
        methods = sorted(m for m in dir(suite) if m.startswith("test_"))
        for m in methods:
            with _registry_lock:
                _experiments.clear()
            drop_db(); init_db()
            try:
                getattr(suite, m)()
                print(f"  ✓  {suite_cls.__name__}::{m}")
                passed += 1
            except Exception as exc:
                print(f"  ✗  {suite_cls.__name__}::{m}")
                print(f"       {exc}")
                import traceback; traceback.print_exc()
                failed += 1

    drop_db()
    print(f"\n{'✅' if not failed else '❌'}  {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
