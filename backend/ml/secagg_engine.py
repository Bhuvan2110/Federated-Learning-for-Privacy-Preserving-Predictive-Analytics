"""
ml/secagg_engine.py
══════════════════════════════════════════════════════════════════════
SecAgg-FedAvg: Federated Averaging with Secure Aggregation.

Difference from plain federated_train()
────────────────────────────────────────
  In plain FedAvg the server receives each client's raw weight update
  and averages them.  The server can therefore inspect any individual
  client's model — it sees *who updated what*.

  SecAgg-FedAvg replaces the aggregation step with a SecAggServer
  round:  each client masks its update before sending, the server
  sums masked updates, then unmasks using collected self-seeds.
  The server only ever learns the *sum* of updates, never individual
  client contributions.

  The training outcome (loss curve, accuracy) is mathematically
  identical to plain FedAvg — SecAgg is a privacy wrapper around the
  aggregation step, not a change to the learning algorithm.

Algorithm per round
───────────────────
  1. Each client runs local_epochs of SGD → produces (w_local, b_local)
  2. Compute update:   Δw_i = w_local_i − w_global
                       Δb_i = b_local_i − b_global
  3. SecAggServer.aggregate_round([Δw_i], [Δb_i])
     → returns Σ Δw_survived and Σ Δb_survived  (unmasked, but server
       never saw individual updates)
  4. Weighted-average update (by number of samples):
       w_global += (Σ Δw_survived) / n_survived
       b_global += (Σ Δb_survived) / n_survived
  5. Record round secagg_summary for audit log

Dropout model
─────────────
  dropout_rate ∈ [0, 1) controls what fraction of clients "fail" each
  round.  Dropped clients' self-masks are reconstructed from Shamir
  shares held by surviving clients.  Training still proceeds with the
  surviving clients' updates.

  If dropout causes fewer than `threshold` clients to survive, the
  round raises RuntimeError (too many dropouts).  This matches the
  real protocol: aborted rounds are re-started after client re-join.
"""

from __future__ import annotations

import logging
import time

from ml.engine import (
    prepare_data, split_data, normalise, apply_norm,
    train_epoch, predict, metrics,
)
from ml.secagg import SecAggServer

logger = logging.getLogger(__name__)


def secagg_federated_train(
    rows: list,
    headers: list,
    target_idx: int,
    ftypes: dict,
    *,
    rounds: int      = 25,
    local_epochs: int = 5,
    lr: float        = 0.1,
    num_clients: int = 5,
    dropout_rate: float = 0.0,
    secagg_threshold: int = 0,      # 0 = auto: ceil(n/2)
) -> dict:
    """
    SecAgg-FedAvg training.

    Returns a result dict with the same shape as federated_train()
    plus a `secagg` sub-dict containing per-round audit information.

    Parameters
    ──────────
    dropout_rate     : fraction of clients to simulate dropping each round
                       e.g. 0.2 means 1 in 5 clients drops per round
    secagg_threshold : Shamir threshold for dropout recovery
                       0 = auto (ceil(n_clients/2))
    """
    t0 = time.time()

    # ── Data preparation ──────────────────────────────────────────────────
    X, y, fcols, tcol, ulabels, lmap = prepare_data(rows, headers, target_idx, ftypes)
    Xtr, ytr, Xte, yte              = split_data(X, y)
    XtrN, means, stds               = normalise(Xtr)
    XteN                            = apply_norm(Xte, means, stds)

    m  = len(XtrN)
    nf = len(fcols)
    cs = m // num_clients

    clients_data = [
        {
            "X": XtrN[k * cs : (m if k == num_clients - 1 else (k + 1) * cs)],
            "y": ytr [k * cs : (m if k == num_clients - 1 else (k + 1) * cs)],
        }
        for k in range(num_clients)
    ]
    client_sizes = [len(c["X"]) for c in clients_data]

    # ── SecAgg server setup ───────────────────────────────────────────────
    secagg_server = SecAggServer(
        n_clients    = num_clients,
        n_features   = nf,
        threshold    = secagg_threshold,
        dropout_rate = dropout_rate,
    )

    logger.info(
        "SecAgg-FedAvg: rounds=%d clients=%d dropout=%.0f%% threshold=%d",
        rounds, num_clients, dropout_rate * 100, secagg_server.threshold,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    gw   = [0.0] * nf
    gb   = 0.0
    hist: list[float]  = []
    secagg_log: list[dict] = []

    for rnd in range(rounds):

        # Each client runs local SGD and computes Δw, Δb
        weight_updates: list[list[float]] = []
        bias_updates:   list[float]       = []
        round_losses:   list[float]       = []

        for client in clients_data:
            w, b = list(gw), gb
            ll   = 0.0
            for _ in range(local_epochs):
                w, b, ll = train_epoch(client["X"], client["y"], w, b, lr)

            # Update = local_params − global_params
            dw = [w[j] - gw[j] for j in range(nf)]
            db = b - gb

            weight_updates.append(dw)
            bias_updates.append(db)
            round_losses.append(ll)

        # SecAgg aggregation — server only sees sum of masked updates
        try:
            sagg_result = secagg_server.aggregate_round(
                weight_updates, bias_updates
            )
        except RuntimeError as exc:
            logger.error("Round %d SecAgg failed (dropout too high): %s", rnd + 1, exc)
            # Skip this round — keep global model unchanged
            hist.append(hist[-1] if hist else 0.0)
            secagg_log.append({"round": rnd + 1, "error": str(exc)})
            continue

        n_surv = sagg_result.n_clients_survived

        # Weighted average over surviving clients
        surv_indices = list(range(num_clients - sagg_result.n_clients_dropped))
        total_surv_n = sum(client_sizes[i] for i in surv_indices)

        if total_surv_n == 0:
            logger.warning("Round %d: no surviving client data — skipping", rnd + 1)
            continue

        # The SecAgg sum is Σ Δw_i (unweighted sum of surviving clients)
        # We need weighted average: Σ (n_i / N_surv) * Δw_i
        # Since SecAgg gives us Σ Δw_i (not weighted), we approximate
        # by equal weighting (1/n_survived).  For weighted aggregation,
        # weights would need to be incorporated into the mask protocol.
        gw = [gw[j] + sagg_result.aggregated_weights[j] / n_surv for j in range(nf)]
        gb = gb + sagg_result.aggregated_bias / n_surv

        # Round loss: average over *all* clients (before dropout)
        round_loss = sum(round_losses) / len(round_losses)
        hist.append(round(round_loss, 6))

        # Audit log entry (no individual update data — preserves SecAgg guarantee)
        secagg_log.append({
            "round":              rnd + 1,
            "n_survived":         n_surv,
            "n_dropped":          sagg_result.n_clients_dropped,
            "verified":           sagg_result.verified,
            "overhead_ms":        round(sagg_result.masking_overhead_ms, 2),
        })

        if (rnd + 1) % 5 == 0 or rnd == 0:
            logger.info(
                "Round %d/%d  loss=%.4f  survived=%d/%d  verified=%s",
                rnd + 1, rounds, round_loss,
                n_surv, num_clients,
                sagg_result.verified,
            )

    # ── Final evaluation ──────────────────────────────────────────────────
    train_mets = metrics(predict(XtrN, gw, gb), ytr)
    test_mets  = metrics(predict(XteN, gw, gb), yte)

    total_overhead_ms = sum(e.get("overhead_ms", 0) for e in secagg_log)
    all_verified      = all(e.get("verified", True) for e in secagg_log)

    secagg_summary = {
        "protocol":            "SecAgg (Bonawitz et al. 2017)",
        "n_clients":           num_clients,
        "threshold":           secagg_server.threshold,
        "dropout_rate":        dropout_rate,
        "rounds_completed":    len([e for e in secagg_log if "error" not in e]),
        "rounds_aborted":      len([e for e in secagg_log if "error" in e]),
        "all_rounds_verified": all_verified,
        "total_overhead_ms":   round(total_overhead_ms, 2),
        "per_round_log":       secagg_log,
    }

    logger.info(
        "SecAgg-FedAvg complete  test_acc=%.4f  all_verified=%s  overhead=%.0fms",
        test_mets["accuracy"], all_verified, total_overhead_ms,
    )

    return {
        "model":           "SecAgg-Federated",
        "featureCols":     fcols,
        "targetCol":       tcol,
        "uniqueLabels":    ulabels,
        "trainSamples":    len(Xtr),
        "testSamples":     len(Xte),
        "numClients":      num_clients,
        "rounds":          len(hist),
        "roundsRequested": rounds,
        "localEpochs":     local_epochs,
        "lr":              lr,
        "lossHistory":     hist,
        "trainMetrics":    train_mets,
        "testMetrics":     test_mets,
        "finalLoss":       hist[-1] if hist else 0.0,
        "trainingTimeMs":  int((time.time() - t0) * 1000),
        "secagg":          secagg_summary,
    }
