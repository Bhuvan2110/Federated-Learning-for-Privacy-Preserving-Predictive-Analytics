# -*- coding: utf-8 -*-
# Generated Python stubs for federation.proto
# Equivalent to running: python -m grpc_tools.protoc -I proto --python_out=proto proto/federation.proto
#
# These stubs are committed to the repo so consumers don't need grpc_tools
# at runtime.  Regenerate with:
#   pip install grpcio-tools
#   python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. federation.proto
#
# The hand-written versions below are API-compatible with the grpcio library
# and are used by grpc_server.py and grpc_client.py.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


# ── ModelWeights ──────────────────────────────────────────────────────────

@dataclass
class ModelWeights:
    weights: List[float] = field(default_factory=list)
    bias:    float        = 0.0
    version: int          = 0

    def to_dict(self) -> dict:
        return {"weights": list(self.weights), "bias": self.bias, "version": self.version}

    @classmethod
    def from_dict(cls, d: dict) -> "ModelWeights":
        return cls(weights=list(d.get("weights", [])),
                   bias=float(d.get("bias", 0.0)),
                   version=int(d.get("version", 0)))


# ── Ack ───────────────────────────────────────────────────────────────────

@dataclass
class Ack:
    success: bool  = True
    message: str   = ""

    def to_dict(self) -> dict:
        return {"success": self.success, "message": self.message}


# ── JoinRequest / JoinResponse ────────────────────────────────────────────

@dataclass
class JoinRequest:
    experiment_id: str = ""
    client_id:     str = ""
    dataset_info:  str = ""   # JSON string

    def to_dict(self) -> dict:
        return {"experiment_id": self.experiment_id,
                "client_id":     self.client_id,
                "dataset_info":  self.dataset_info}


@dataclass
class JoinResponse:
    accepted:      bool  = False
    assigned_id:   str   = ""
    experiment_id: str   = ""
    n_features:    int   = 0
    total_rounds:  int   = 0
    local_epochs:  int   = 5
    learning_rate: float = 0.1
    model_type:    str   = "federated"
    reject_reason: str   = ""

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# ── ModelRequest / GlobalModel ────────────────────────────────────────────

@dataclass
class ModelRequest:
    experiment_id: str = ""
    client_id:     str = ""
    round:         int = 0

    def to_dict(self) -> dict:
        return {"experiment_id": self.experiment_id,
                "client_id": self.client_id, "round": self.round}


@dataclass
class GlobalModel:
    experiment_id: str          = ""
    round:         int          = 0
    weights:       ModelWeights = field(default_factory=ModelWeights)
    training_done: bool         = False
    status:        str          = "waiting"   # waiting|ready|done|aborted

    def to_dict(self) -> dict:
        return {"experiment_id": self.experiment_id,
                "round":         self.round,
                "weights":       self.weights.to_dict(),
                "training_done": self.training_done,
                "status":        self.status}


# ── ClientUpdate ──────────────────────────────────────────────────────────

@dataclass
class ClientUpdate:
    experiment_id:   str          = ""
    client_id:       str          = ""
    round:           int          = 0
    delta:           ModelWeights = field(default_factory=ModelWeights)
    n_samples:       int          = 0
    train_loss:      float        = 0.0
    train_accuracy:  float        = 0.0
    masked_weights:  List[float]  = field(default_factory=list)
    masked_bias:     float        = 0.0

    def to_dict(self) -> dict:
        return {
            "experiment_id":  self.experiment_id,
            "client_id":      self.client_id,
            "round":          self.round,
            "delta":          self.delta.to_dict(),
            "n_samples":      self.n_samples,
            "train_loss":     self.train_loss,
            "train_accuracy": self.train_accuracy,
            "masked_weights": list(self.masked_weights),
            "masked_bias":    self.masked_bias,
        }


# ── LeaveRequest ──────────────────────────────────────────────────────────

@dataclass
class LeaveRequest:
    experiment_id: str = ""
    client_id:     str = ""
    reason:        str = "completed"

    def to_dict(self) -> dict:
        return {"experiment_id": self.experiment_id,
                "client_id": self.client_id, "reason": self.reason}


# ── StreamRequest / RoundUpdate ───────────────────────────────────────────

@dataclass
class StreamRequest:
    experiment_id: str = ""
    client_id:     str = ""

    def to_dict(self) -> dict:
        return {"experiment_id": self.experiment_id, "client_id": self.client_id}


@dataclass
class RoundUpdate:
    experiment_id:   str   = ""
    round:           int   = 0
    total_rounds:    int   = 0
    global_loss:     float = 0.0
    global_accuracy: float = 0.0
    clients_ready:   int   = 0
    clients_total:   int   = 0
    status:          str   = "aggregating"

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# ── StatusRequest / ExperimentStatus ─────────────────────────────────────

@dataclass
class StatusRequest:
    experiment_id: str = ""

    def to_dict(self) -> dict:
        return {"experiment_id": self.experiment_id}


@dataclass
class ExperimentStatus:
    experiment_id:  str   = ""
    status:         str   = "pending"
    round:          int   = 0
    total_rounds:   int   = 0
    best_accuracy:  float = 0.0
    final_loss:     float = 0.0
    clients_joined: int   = 0
    error_message:  str   = ""

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}
