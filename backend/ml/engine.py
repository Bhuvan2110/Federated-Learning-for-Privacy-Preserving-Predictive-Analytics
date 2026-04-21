"""
ml/engine.py
══════════════════════════════════════════════════════════════════════
Pure-Python ML engine — zero external ML dependencies.

Extracted from app.py so that:
  • Flask routes can import it directly (synchronous path)
  • Celery workers can import it without pulling in Flask (async path)
  • Unit tests can test ML logic in isolation

Contains:
  - Data preparation & normalisation
  - Logistic regression training (central + federated FedAvg)
  - Metrics calculation (accuracy, precision, recall, F1, confusion matrix)
"""

import math
import random
import time


# ── Math primitives ───────────────────────────────────────────────────────

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, x))))


def dot(a: list, b: list) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


# ── Normalisation ─────────────────────────────────────────────────────────

def normalise(X: list) -> tuple:
    """Z-score normalise feature matrix. Returns (X_norm, means, stds)."""
    if not X or not X[0]:
        return X, [], []
    n = len(X[0])
    means, stds = [], []
    for j in range(n):
        col = [r[j] for r in X]
        m   = sum(col) / len(col)
        s   = math.sqrt(sum((v - m) ** 2 for v in col) / len(col)) or 1.0
        means.append(m)
        stds.append(s)
    return [[(r[j] - means[j]) / stds[j] for j in range(n)] for r in X], means, stds


def apply_norm(X: list, means: list, stds: list) -> list:
    return [[(r[j] - means[j]) / stds[j] for j in range(len(r))] for r in X]


# ── Data preparation ──────────────────────────────────────────────────────

def prepare_data(rows: list, headers: list, target_idx: int, ftypes: dict) -> tuple:
    """
    Parse raw CSV rows into feature matrix X and label vector y.

    Returns: (X, y, feature_cols, target_col, unique_labels, label_map)
    """
    tcol  = headers[target_idx]
    fcols = [
        h for i, h in enumerate(headers)
        if i != target_idx and ftypes.get(h, "numeric") != "ignore"
    ]
    if not fcols:
        raise ValueError("No feature columns selected.")

    def enc(v: str, t: str) -> float:
        if t == "numeric":
            return float(v)
        if t == "binary":
            return 1.0 if str(v).strip().lower() in ("1", "true", "yes", "y", "1.0") else 0.0
        # categorical → hash bucketing
        return float(hash(str(v)) % 10_000) / 10_000.0

    clean = []
    for row in rows:
        try:
            x = [enc(row[h], ftypes.get(h, "numeric")) for h in fcols]
            t = str(row.get(tcol, "")).strip()
            if t:
                clean.append((x, t))
        except Exception:
            pass

    if len(clean) < 10:
        raise ValueError(f"Need ≥10 clean rows, got {len(clean)}.")

    labels = [r[1] for r in clean]
    unique = sorted(set(labels))

    if len(unique) == 2:
        lmap = {l: i for i, l in enumerate(unique)}
    else:
        freq = {l: labels.count(l) for l in unique}
        top  = max(freq, key=freq.get)
        lmap = {l: (1 if l == top else 0) for l in unique}

    return (
        [r[0] for r in clean],
        [lmap[r[1]] for r in clean],
        fcols, tcol, unique, lmap,
    )


def split_data(X: list, y: list, ratio: float = 0.2) -> tuple:
    """Shuffle-split into train / test sets."""
    idx = list(range(len(X)))
    random.shuffle(idx)
    cut = int(len(X) * (1 - ratio))
    tr, te = idx[:cut], idx[cut:]
    return (
        [X[i] for i in tr], [y[i] for i in tr],
        [X[i] for i in te], [y[i] for i in te],
    )


# ── Logistic regression ───────────────────────────────────────────────────

def train_epoch(X: list, y: list, w: list, b: float, lr: float) -> tuple:
    """One gradient descent epoch. Returns (w, b, loss)."""
    m, nf = len(X), len(w)
    dw    = [0.0] * nf
    db    = 0.0
    loss  = 0.0
    for i in range(m):
        p   = sigmoid(dot(X[i], w) + b)
        err = p - y[i]
        loss += -(y[i] * math.log(p + 1e-10) + (1 - y[i]) * math.log(1 - p + 1e-10))
        for j in range(nf):
            dw[j] += err * X[i][j]
        db += err
    w_new = [w[j] - (lr * dw[j]) / m for j in range(nf)]
    return w_new, b - (lr * db) / m, loss / m


def predict(X: list, w: list, b: float) -> list:
    return [1 if sigmoid(dot(x, w) + b) >= 0.5 else 0 for x in X]


# ── Metrics ───────────────────────────────────────────────────────────────

def metrics(yp: list, yt: list) -> dict:
    tp = fp = fn = tn = 0
    for p, t in zip(yp, yt):
        if   t == 1 and p == 1: tp += 1
        elif t == 0 and p == 1: fp += 1
        elif t == 1 and p == 0: fn += 1
        else:                   tn += 1
    n    = len(yt)
    acc  = (tp + tn) / n if n else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0
    return {
        "accuracy":   round(acc,  4),
        "precision":  round(prec, 4),
        "recall":     round(rec,  4),
        "f1":         round(f1,   4),
        "confMatrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


# ── Training entry points ─────────────────────────────────────────────────

def central_train(
    rows: list, headers: list, target_idx: int,
    ftypes: dict, epochs: int = 100, lr: float = 0.1,
) -> dict:
    """
    Centralised logistic regression training.
    Returns a result dict compatible with the REST API and Celery task.
    """
    t0 = time.time()
    X, y, fcols, tcol, ulabels, lmap = prepare_data(rows, headers, target_idx, ftypes)
    Xtr, ytr, Xte, yte              = split_data(X, y)
    XtrN, means, stds               = normalise(Xtr)
    XteN                            = apply_norm(Xte, means, stds)

    w    = [0.0] * len(fcols)
    b    = 0.0
    hist = []
    for _ in range(epochs):
        w, b, loss = train_epoch(XtrN, ytr, w, b, lr)
        hist.append(round(loss, 6))

    return {
        "model":           "Central",
        "featureCols":     fcols,
        "targetCol":       tcol,
        "uniqueLabels":    ulabels,
        "trainSamples":    len(Xtr),
        "testSamples":     len(Xte),
        "epochs":          epochs,
        "lr":              lr,
        "lossHistory":     hist,
        "trainMetrics":    metrics(predict(XtrN, w, b), ytr),
        "testMetrics":     metrics(predict(XteN, w, b), yte),
        "finalLoss":       hist[-1] if hist else 0,
        "trainingTimeMs":  int((time.time() - t0) * 1000),
    }


def federated_train(
    rows: list, headers: list, target_idx: int,
    ftypes: dict, rounds: int = 25, local_epochs: int = 5,
    lr: float = 0.1, num_clients: int = 5,
) -> dict:
    """
    Federated Averaging (FedAvg — McMahan et al. 2017).
    Data is partitioned across simulated clients; server aggregates
    weighted averages of client model updates each round.
    Returns a result dict compatible with the REST API and Celery task.
    """
    t0 = time.time()
    X, y, fcols, tcol, ulabels, lmap = prepare_data(rows, headers, target_idx, ftypes)
    Xtr, ytr, Xte, yte              = split_data(X, y)
    XtrN, means, stds               = normalise(Xtr)
    XteN                            = apply_norm(Xte, means, stds)

    m   = len(XtrN)
    nf  = len(fcols)
    cs  = m // num_clients

    # Partition training data across clients
    clients = [
        {
            "X": XtrN[k * cs : (m if k == num_clients - 1 else (k + 1) * cs)],
            "y": ytr [k * cs : (m if k == num_clients - 1 else (k + 1) * cs)],
        }
        for k in range(num_clients)
    ]

    gw   = [0.0] * nf
    gb   = 0.0
    hist = []

    for _ in range(rounds):
        round_results = []
        for client in clients:
            w, b = list(gw), gb
            ll   = 0.0
            for _ in range(local_epochs):
                w, b, ll = train_epoch(client["X"], client["y"], w, b, lr)
            round_results.append({"w": w, "b": b, "loss": ll, "n": len(client["X"])})

        # Weighted FedAvg aggregation
        total_n = sum(r["n"] for r in round_results)
        gw = [
            sum(r["w"][j] * r["n"] / total_n for r in round_results)
            for j in range(nf)
        ]
        gb   = sum(r["b"] * r["n"] / total_n for r in round_results)
        hist.append(round(
            sum(r["loss"] * r["n"] for r in round_results) / total_n, 6
        ))

    return {
        "model":           "Federated",
        "featureCols":     fcols,
        "targetCol":       tcol,
        "uniqueLabels":    ulabels,
        "trainSamples":    len(Xtr),
        "testSamples":     len(Xte),
        "numClients":      num_clients,
        "rounds":          rounds,
        "localEpochs":     local_epochs,
        "lr":              lr,
        "lossHistory":     hist,
        "trainMetrics":    metrics(predict(XtrN, gw, gb), ytr),
        "testMetrics":     metrics(predict(XteN, gw, gb), yte),
        "finalLoss":       hist[-1] if hist else 0,
        "trainingTimeMs":  int((time.time() - t0) * 1000),
    }
