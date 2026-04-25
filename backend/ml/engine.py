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


# ── Linear regression ─────────────────────────────────────────────────────

def train_epoch_reg(X: list, y: list, w: list, b: float, lr: float) -> tuple:
    """One gradient descent epoch for regression (MSE loss)."""
    m, nf = len(X), len(w)
    dw = [0.0] * nf
    db = 0.0
    loss = 0.0
    for i in range(m):
        p = dot(X[i], w) + b
        err = p - y[i]
        loss += err ** 2
        for j in range(nf):
            dw[j] += err * X[i][j]
        db += err
    w_new = [w[j] - (lr * dw[j]) / m for j in range(nf)]
    return w_new, b - (lr * db) / m, loss / (2 * m)


def predict_reg(X: list, w: list, b: float) -> list:
    return [dot(x, w) + b for x in X]


# ── Decision Tree (Simple CART) ───────────────────────────────────────────

def get_gini(y: list) -> float:
    if not y: return 0.0
    p1 = sum(y) / len(y)
    return 1.0 - p1**2 - (1.0 - p1)**2

def get_mse(y: list) -> float:
    if not y: return 0.0
    avg = sum(y) / len(y)
    return sum((v - avg)**2 for v in y) / len(y)

def split_node(X: list, y: list, task: str = "classification"):
    best_gain = -1
    best_split = None
    n_features = len(X[0])
    current_score = get_gini(y) if task == "classification" else get_mse(y)
    
    for f_idx in range(n_features):
        values = sorted(set(row[f_idx] for row in X))
        for i in range(len(values) - 1):
            threshold = (values[i] + values[i+1]) / 2
            left_idx = [j for j, row in enumerate(X) if row[f_idx] <= threshold]
            right_idx = [j for j, row in enumerate(X) if row[f_idx] > threshold]
            
            if not left_idx or not right_idx: continue
            
            left_y = [y[j] for j in left_idx]
            right_y = [y[j] for j in right_idx]
            
            if task == "classification":
                score = (len(left_y) * get_gini(left_y) + len(right_y) * get_gini(right_y)) / len(y)
            else:
                score = (len(left_y) * get_mse(left_y) + len(right_y) * get_mse(right_y)) / len(y)
                
            gain = current_score - score
            if gain > best_gain:
                best_gain = gain
                best_split = (f_idx, threshold, left_idx, right_idx)
                
    return best_split

def build_tree(X: list, y: list, depth: int, max_depth: int = 5, min_samples: int = 2, task: str = "classification"):
    if depth >= max_depth or len(y) <= min_samples or len(set(y)) == 1:
        return {"val": sum(y) / len(y) if task == "regression" else (1 if sum(y)/len(y) >= 0.5 else 0)}
    
    split = split_node(X, y, task)
    if not split:
        return {"val": sum(y) / len(y) if task == "regression" else (1 if sum(y)/len(y) >= 0.5 else 0)}
    
    f_idx, threshold, left_idx, right_idx = split
    return {
        "f_idx": f_idx,
        "threshold": threshold,
        "left": build_tree([X[j] for j in left_idx], [y[j] for j in left_idx], depth + 1, max_depth, min_samples, task),
        "right": build_tree([X[j] for j in right_idx], [y[j] for j in right_idx], depth + 1, max_depth, min_samples, task)
    }

def predict_tree_one(node: dict, x: list) -> float:
    if "val" in node: return node["val"]
    if x[node["f_idx"]] <= node["threshold"]:
        return predict_tree_one(node["left"], x)
    return predict_tree_one(node["right"], x)

def predict_tree(node: dict, X: list) -> list:
    return [predict_tree_one(node, x) for x in X]


def predict_one_from_params(params: dict, x_raw: list) -> float:
    """
    Perform a single prediction using saved model parameters.
    Handles normalisation, algorithm selection, and task type.
    """
    algo = params.get("algo", "logistic")
    task = params.get("task", "classification")
    means = params.get("means", [])
    stds = params.get("stds", [])
    
    # 1. Normalise
    x = [(x_raw[j] - means[j]) / stds[j] if j < len(means) else x_raw[j] for j in range(len(x_raw))]
    
    # 2. Predict
    if algo == "decision_tree":
        return predict_tree_one(params["tree"], x)
    
    w, b = params.get("w", []), params.get("b", 0.0)
    score = sum(xi * wi for xi, wi in zip(x, w)) + b
    
    if algo == "linear":
        return score
    
    # Logistic
    return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, score))))


# ── Metrics ───────────────────────────────────────────────────────────────

def metrics(yp: list, yt: list, task: str = "classification") -> dict:
    if task == "regression":
        mse = sum((p - t)**2 for p, t in zip(yp, yt)) / len(yt)
        mae = sum(abs(p - t) for p, t in zip(yp, yt)) / len(yt)
        return {
            "mse": round(mse, 4),
            "rmse": round(math.sqrt(mse), 4),
            "mae": round(mae, 4),
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "confMatrix": {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        }
        
    tp = fp = fn = tn = 0
    for p, t in zip(yp, yt):
        p_bin = 1 if p >= 0.5 else 0
        t_bin = 1 if t >= 0.5 else 0
        if   t_bin == 1 and p_bin == 1: tp += 1
        elif t_bin == 0 and p_bin == 1: fp += 1
        elif t_bin == 1 and p_bin == 0: fn += 1
        else:                         tn += 1
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


# ── Logistic regression ───────────────────────────────────────────────────

def train_epoch_clf(X: list, y: list, w: list, b: float, lr: float) -> tuple:
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


def predict_clf(X: list, w: list, b: float) -> list:
    return [sigmoid(dot(x, w) + b) for x in X]


# Aliases for backward compatibility (DP/SecAgg)
train_epoch = train_epoch_clf
predict = predict_clf


# ── Training entry points ─────────────────────────────────────────────────

def central_train(
    rows: list, headers: list, target_idx: int,
    ftypes: dict, epochs: int = 100, lr: float = 0.1,
    algo: str = "logistic",  # logistic, linear, decision_tree
) -> dict:
    """
    Centralised training. Supports Logistic/Linear Regression and Decision Trees.
    """
    t0 = time.time()
    X, y, fcols, tcol, ulabels, lmap = prepare_data(rows, headers, target_idx, ftypes)
    Xtr, ytr, Xte, yte              = split_data(X, y)
    XtrN, means, stds               = normalise(Xtr)
    XteN                            = apply_norm(Xte, means, stds)

    task = "regression" if algo == "linear" else "classification"
    if algo == "decision_tree" and len(ulabels) > 5: task = "regression"

    params = {"algo": algo, "task": task, "means": means, "stds": stds, "lmap": lmap}
    hist = []

    if algo == "decision_tree":
        tree = build_tree(XtrN, ytr, 0, max_depth=5, task=task)
        yp_tr = predict_tree(tree, XtrN)
        yp_te = predict_tree(tree, XteN)
        params["tree"] = tree
    else:
        w = [0.0] * len(fcols)
        b = 0.0
        for _ in range(epochs):
            if algo == "linear":
                w, b, loss = train_epoch_reg(XtrN, ytr, w, b, lr)
            else:
                w, b, loss = train_epoch_clf(XtrN, ytr, w, b, lr)
            hist.append(round(loss, 6))
        
        if algo == "linear":
            yp_tr = predict_reg(XtrN, w, b)
            yp_te = predict_reg(XteN, w, b)
        else:
            yp_tr = predict_clf(XtrN, w, b)
            yp_te = predict_clf(XteN, w, b)
        
        params["w"], params["b"] = w, b

    return {
        "model":           f"Central ({algo})",
        "featureCols":     fcols,
        "targetCol":       tcol,
        "uniqueLabels":    ulabels,
        "trainSamples":    len(Xtr),
        "testSamples":     len(Xte),
        "epochs":          epochs if algo != "decision_tree" else 1,
        "lr":              lr if algo != "decision_tree" else 0,
        "lossHistory":     hist,
        "trainMetrics":    metrics(yp_tr, ytr, task),
        "testMetrics":     metrics(yp_te, yte, task),
        "finalLoss":       hist[-1] if hist else 0,
        "trainingTimeMs":  int((time.time() - t0) * 1000),
        "model_params":    params,
    }


def federated_train(
    rows: list, headers: list, target_idx: int,
    ftypes: dict, rounds: int = 25, local_epochs: int = 5,
    lr: float = 0.1, num_clients: int = 5,
    algo: str = "logistic",
) -> dict:
    """
    Federated Averaging. Decision Tree is not supported in FL yet (falls back to Central).
    """
    if algo == "decision_tree":
        return central_train(rows, headers, target_idx, ftypes, algo=algo)

    t0 = time.time()
    X, y, fcols, tcol, ulabels, lmap = prepare_data(rows, headers, target_idx, ftypes)
    Xtr, ytr, Xte, yte              = split_data(X, y)
    XtrN, means, stds               = normalise(Xtr)
    XteN                            = apply_norm(Xte, means, stds)

    m   = len(XtrN)
    nf  = len(fcols)
    cs  = m // num_clients
    clients = [
        {"X": XtrN[k*cs:(m if k==num_clients-1 else (k+1)*cs)],
         "y": ytr[k*cs:(m if k==num_clients-1 else (k+1)*cs)]}
        for k in range(num_clients)
    ]

    gw, gb = [0.0] * nf, 0.0
    hist = []
    task = "regression" if algo == "linear" else "classification"

    for _ in range(rounds):
        round_results = []
        for client in clients:
            w, b = list(gw), gb
            ll = 0.0
            for _ in range(local_epochs):
                if algo == "linear":
                    w, b, ll = train_epoch_reg(client["X"], client["y"], w, b, lr)
                else:
                    w, b, ll = train_epoch_clf(client["X"], client["y"], w, b, lr)
            round_results.append({"w": w, "b": b, "loss": ll, "n": len(client["X"])})

        total_n = sum(r["n"] for r in round_results)
        gw = [sum(r["w"][j] * r["n"] / total_n for r in round_results) for j in range(nf)]
        gb = sum(r["b"] * r["n"] / total_n for r in round_results)
        hist.append(round(sum(r["loss"] * r["n"] for r in round_results) / total_n, 6))

    if algo == "linear":
        yp_tr, yp_te = predict_reg(XtrN, gw, gb), predict_reg(XteN, gw, gb)
    else:
        yp_tr, yp_te = predict_clf(XtrN, gw, gb), predict_clf(XteN, gw, gb)

    return {
        "model":           f"Federated ({algo})",
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
        "trainMetrics":    metrics(yp_tr, ytr, task),
        "testMetrics":     metrics(yp_te, yte, task),
        "finalLoss":       hist[-1] if hist else 0,
        "trainingTimeMs":  int((time.time() - t0) * 1000),
        "model_params":    {"algo": algo, "task": task, "w": gw, "b": gb, 
                            "means": means, "stds": stds, "lmap": lmap},
    }
