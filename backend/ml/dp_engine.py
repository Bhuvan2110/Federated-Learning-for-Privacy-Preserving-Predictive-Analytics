"""
ml/dp_engine.py
══════════════════════════════════════════════════════════════════════
Differentially Private Federated Averaging (DP-FedAvg).

Algorithm (per round)
─────────────────────
  1. Server broadcasts global model (w_global, b_global) to all clients.
  2. Each client runs `local_epochs` of SGD on its local data.
  3. Each client computes the gradient update:
         Δw_k  =  w_k_local − w_global
         Δb_k  =  b_k_local − b_global
  4. Server clips each update to L2 norm ≤ C  (bounds sensitivity).
  5. Server sums clipped updates and adds Gaussian noise σ·N(0,I):
         Δw_noised  =  Σ_k clip(Δw_k, C)  +  N(0, σ²·C²·I)
         Δb_noised  =  Σ_k clip(Δb_k, C)  +  N(0, σ²·C²)
  6. Server averages: w_global += Δw_noised / num_clients
  7. MomentsAccountant records one step per round.

This is equivalent to the central DP model: the server sees only
the noised aggregate, never individual updates.

Parameters
──────────
  epsilon          (ε)  privacy budget — total allowed spend
  delta            (δ)  failure probability, typically 1e-5 or 1/N²
  clip_threshold   (C)  L2 norm bound for gradient clipping
  noise_multiplier (z)  σ = z·C; if None, auto-computed from ε,δ,T
"""

from __future__ import annotations

import time
import logging

from ml.engine import (
    prepare_data, split_data, normalise, apply_norm,
    train_epoch, predict, metrics,
)
from ml.privacy import (
    GradientClipper,
    GaussianMechanism,
    MomentsAccountant,
    PrivacyBudget,
    recommended_noise_multiplier,
)

logger = logging.getLogger(__name__)


def dp_federated_train(
    rows: list,
    headers: list,
    target_idx: int,
    ftypes: dict,
    *,
    # FL hyperparameters
    rounds: int = 25,
    local_epochs: int = 5,
    lr: float = 0.1,
    num_clients: int = 5,
    # DP hyperparameters
    target_epsilon: float = 2.0,
    delta: float = 1e-5,
    clip_threshold: float = 1.0,
    noise_multiplier: float | None = None,
) -> dict:
    """
    DP-FedAvg: differentially private federated averaging.

    Returns a result dict with the same shape as federated_train()
    plus a `privacy` sub-dict containing the full budget accounting.

    Raises PrivacyBudget.BudgetExhaustedError mid-run if the
    accumulated epsilon exceeds target_epsilon.
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

    # Partition data across clients
    clients = [
        {
            "X": XtrN[k * cs : (m if k == num_clients - 1 else (k + 1) * cs)],
            "y": ytr [k * cs : (m if k == num_clients - 1 else (k + 1) * cs)],
        }
        for k in range(num_clients)
    ]

    # ── DP setup ──────────────────────────────────────────────────────────
    # Sampling rate: fraction of training data seen per round per client
    # (approximation: one client's data / total data)
    sampling_rate = max(cs, 1) / max(m, 1)

    # Auto-compute noise_multiplier if not provided
    if noise_multiplier is None:
        noise_multiplier = recommended_noise_multiplier(
            target_epsilon=target_epsilon,
            delta=delta,
            num_rounds=rounds,
            sampling_rate=sampling_rate,
        )
        logger.info(
            "DP auto-calibrated noise_multiplier=%.4f for ε=%.2f, δ=%.2e, T=%d",
            noise_multiplier, target_epsilon, delta, rounds,
        )

    sigma = noise_multiplier * clip_threshold

    clipper = GradientClipper(clip_threshold=clip_threshold)
    budget  = PrivacyBudget(
        target_epsilon=target_epsilon,
        delta=delta,
        noise_multiplier=noise_multiplier,
        sampling_rate=sampling_rate,
    )

    logger.info(
        "DP-FedAvg: rounds=%d clients=%d C=%.2f z=%.4f σ=%.4f "
        "ε_target=%.2f δ=%.2e",
        rounds, num_clients, clip_threshold, noise_multiplier, sigma,
        target_epsilon, delta,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    gw   = [0.0] * nf
    gb   = 0.0
    hist = []                     # loss per round
    eps_hist = []                 # epsilon per round (from budget tracker)

    for rnd in range(rounds):

        # 1. Collect local updates from each client
        client_dw: list[list[float]] = []
        client_db: list[float]       = []
        round_losses: list[float]    = []

        for client in clients:
            w, b = list(gw), gb
            ll   = 0.0
            for _ in range(local_epochs):
                w, b, ll = train_epoch(client["X"], client["y"], w, b, lr)

            # Gradient update = local params − global params
            dw = [w[j] - gw[j] for j in range(nf)]
            db = b - gb

            client_dw.append(dw)
            client_db.append(db)
            round_losses.append(ll)

        # 2. Clip each client's weight update, then sum
        clipped_dw_sum = clipper.clip_and_sum(client_dw)

        # Clip scalar bias update per client and sum
        clipped_db_sum = sum(
            clipper.clip([db])[0] for db in client_db
        )

        # 3. Add calibrated Gaussian noise  (σ = noise_multiplier * C)
        #    Noise scale is σ (already incorporates clipping threshold C)
        noised_dw = [
            clipped_dw_sum[j] + sigma * _gauss()
            for j in range(nf)
        ]
        noised_db = clipped_db_sum + sigma * _gauss()

        # 4. Average and apply to global model
        gw = [gw[j] + noised_dw[j] / num_clients for j in range(nf)]
        gb = gb + noised_db / num_clients

        # 5. Record loss and privacy spend
        round_loss = sum(round_losses) / len(round_losses)
        hist.append(round(round_loss, 6))

        try:
            eps = budget.consume_round()
        except PrivacyBudget.BudgetExhaustedError as exc:
            logger.warning("Round %d: %s — stopping early", rnd + 1, exc)
            eps_hist.append(round(budget.current_epsilon, 6))
            break

        eps_hist.append(round(eps, 6))

        if (rnd + 1) % 5 == 0:
            logger.info(
                "Round %d/%d  loss=%.4f  ε=%.4f  remaining=%.4f",
                rnd + 1, rounds,
                round_loss,
                eps,
                budget.remaining_epsilon,
            )

    # ── Final evaluation ──────────────────────────────────────────────────
    train_metrics = metrics(predict(XtrN, gw, gb), ytr)
    test_metrics  = metrics(predict(XteN, gw, gb), yte)

    privacy_summary = budget.to_dict()
    privacy_summary["noise_sigma"]    = round(sigma, 6)
    privacy_summary["clip_threshold"] = clip_threshold
    privacy_summary["epsilon_history"] = eps_hist

    logger.info(
        "DP-FedAvg complete  test_acc=%.4f  ε_final=%.4f  δ=%.2e  rounds=%d",
        test_metrics["accuracy"],
        budget.current_epsilon,
        delta,
        budget.rounds_consumed,
    )

    return {
        "model":            "DP-Federated",
        "featureCols":      fcols,
        "targetCol":        tcol,
        "uniqueLabels":     ulabels,
        "trainSamples":     len(Xtr),
        "testSamples":      len(Xte),
        "numClients":       num_clients,
        "rounds":           budget.rounds_consumed,   # may be < requested if budget exhausted
        "roundsRequested":  rounds,
        "localEpochs":      local_epochs,
        "lr":               lr,
        "lossHistory":      hist,
        "trainMetrics":     train_metrics,
        "testMetrics":      test_metrics,
        "finalLoss":        hist[-1] if hist else 0.0,
        "trainingTimeMs":   int((time.time() - t0) * 1000),
        # DP accounting
        "privacy":          privacy_summary,
    }


def _gauss() -> float:
    """Standard normal sample (mean=0, std=1)."""
    import random
    return random.gauss(0.0, 1.0)
