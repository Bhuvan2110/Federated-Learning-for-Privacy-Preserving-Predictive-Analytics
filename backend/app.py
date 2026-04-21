"""
Training Models — Flask Backend  v5.0
(Celery Async Task Queue + PostgreSQL + E2E Encrypted Upload)
══════════════════════════════════════════════════════════════════════════════
What's new in v5.0 (Day 2 — Phase 1 Week 3-4)
───────────────────────────────────────────────
  ✅ Celery + Redis task queue — training runs as background jobs
  ✅ ML engine extracted to ml/engine.py (importable by workers too)
  ✅ Training routes now non-blocking:
       POST /api/train/central   → returns {jobId, experimentId} immediately
       POST /api/train/federated → returns {jobId, experimentId} immediately
  ✅ New polling endpoint:
       GET  /api/jobs/<task_id>  → {status, result?, experiment?}
  ✅ Sync fallback: if Redis unavailable, trains synchronously (dev mode)
  ✅ All Day 1 features retained (Postgres persistence, E2E encryption)

Training flow (async)
─────────────────────
  POST /api/train/* ──► creates Experiment row (status=pending)
                    ──► .apply_async() submits task to Redis
                    ──► returns {jobId, experimentId} to client   ← non-blocking

  Celery worker ──► picks up task from Redis queue
               ──► marks experiment running → trains → marks completed/failed
               ──► stores result in Redis result backend

  GET /api/jobs/<jobId> ──► reads Redis result backend
                        ──► returns {status, result, experiment}  ← client polls this
"""

import io, csv as csvlib, os, base64, logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from cryptography.hazmat.primitives.asymmetric import rsa, padding as asym_padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from celery.result import AsyncResult
from celery_app import celery

from database import (
    get_db, init_db, health_check,
    ModelType,
    FileRepo, ExperimentRepo, UserRepo,
)
from ml.engine import central_train, federated_train
from ml.dp_engine import dp_federated_train
from ml.secagg_engine import secagg_federated_train
from tracking import mlflow_health, MLFLOW_TRACKING_URI, MLFLOW_ENABLED
from monitoring import (
    init_flask_metrics, generate_metrics_output,
    metrics_health, record_upload, record_decrypt_failure,
    PROMETHEUS_ENABLED,
)
from auth import (
    require_auth, require_role, require_experiment_access,
    get_current_user, Action, AUTH_ENABLED, AUTH_OPTIONAL,
    create_token_pair, revoke_token, TokenError,
    JWT_ACCESS_TOKEN_TTL,
)
from audit import AuditLogger, AuditAction, AuditOutcome, log_from_request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
init_flask_metrics(app)   # Day 7: Prometheus HTTP instrumentation

# ── Detect async mode ─────────────────────────────────────────────────────
# If Redis is unavailable (local dev without Docker), fall back to
# synchronous in-process training so the app still works.
_ASYNC_ENABLED = os.getenv("CELERY_ASYNC_ENABLED", "true").lower() == "true"

# ── RSA KEY PAIR ──────────────────────────────────────────────────────────
print("\n🔐  Generating RSA-2048 key pair …", end=" ", flush=True)
_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_PUBLIC_KEY  = _PRIVATE_KEY.public_key()
_PUBLIC_KEY_PEM = _PUBLIC_KEY.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo,
).decode()
print("done ✓")


def _rsa_decrypt(ciphertext_b64: str) -> bytes:
    return _PRIVATE_KEY.decrypt(
        base64.b64decode(ciphertext_b64),
        asym_padding.OAEP(
            mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )


def _aes_gcm_decrypt(key: bytes, iv_b64: str, ciphertext_b64: str) -> bytes:
    return AESGCM(key).decrypt(
        base64.b64decode(iv_b64),
        base64.b64decode(ciphertext_b64),
        None,
    )


def _parse_csv_bytes(raw: bytes, filename: str) -> dict:
    content = raw.decode("utf-8", errors="replace")
    rows    = [dict(r) for r in csvlib.DictReader(io.StringIO(content))]
    if not rows:
        raise ValueError("CSV is empty.")
    headers = list(rows[0].keys())
    stats   = []
    for col in headers:
        vals = [r[col] for r in rows if r.get(col, "") != ""]
        nums = []
        for v in vals:
            try: nums.append(float(v))
            except: pass
        if len(nums) > len(vals) * 0.6 and nums:
            stats.append({"col": col, "type": "numeric",
                          "min": round(min(nums), 4), "max": round(max(nums), 4),
                          "avg": round(sum(nums) / len(nums), 4), "count": len(nums)})
        else:
            uv = list(set(vals))
            stats.append({"col": col,
                          "type": "binary" if len(uv) <= 2 else "text",
                          "uniqueCount": len(uv), "uniqueValues": uv[:8],
                          "count": len(vals)})
    return {"filename": filename, "totalRows": len(rows), "totalCols": len(headers),
            "headers": headers, "rows": rows[:1000], "stats": stats}


def _persist_upload(result: dict, filename: str,
                    encrypted: bool, enc_method: str | None = None) -> str | None:
    """Save upload metadata to Postgres; return file_id or None on failure."""
    try:
        with get_db() as db:
            f = FileRepo.create(
                db, filename=filename,
                total_rows=result["totalRows"], total_cols=result["totalCols"],
                headers=result["headers"], column_stats=result["stats"],
                encrypted_upload=encrypted, encryption_method=enc_method,
            )
            return f.id
    except Exception as exc:
        logger.warning("DB persist upload failed (continuing): %s", exc)
        return None


# ── Job status helper ─────────────────────────────────────────────────────

def _job_response(task_id: str, exp_id: str | None = None) -> dict:
    """
    Query Celery result backend and return a unified job status dict.

    States returned to client:
      PENDING  — task submitted, not yet picked up by a worker
      STARTED  — worker has picked up the task (track_started=True)
      SUCCESS  — training finished; result contains metrics
      FAILURE  — training failed; error contains the exception message
      REVOKED  — task was cancelled
    """
    ar     = AsyncResult(task_id, app=celery)
    state  = ar.state   # PENDING / STARTED / SUCCESS / FAILURE / REVOKED

    payload: dict = {
        "jobId":        task_id,
        "experimentId": exp_id,
        "status":       state,
    }

    if state == "SUCCESS":
        payload["result"] = ar.result
    elif state == "FAILURE":
        payload["error"] = str(ar.result)   # ar.result holds the exception on failure
    elif state == "STARTED":
        payload["info"] = ar.info           # e.g. {"pid": 1234}

    # Attach live experiment row from Postgres for richer status
    if exp_id:
        try:
            with get_db() as db:
                exp = ExperimentRepo.get_by_id(db, exp_id)
                if exp:
                    payload["experiment"] = exp.to_dict(include_result=(state == "SUCCESS"))
        except Exception as exc:
            logger.warning("Could not fetch experiment %s: %s", exp_id, exc)

    return payload


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


# ── Health ────────────────────────────────────────────────────────────────
@app.route("/api/health")
def health():
    db_status = health_check()

    # Quick Celery/Redis ping
    celery_status = "unknown"
    try:
        celery.control.ping(timeout=1.0)
        celery_status = "ok"
    except Exception:
        celery_status = "unavailable"

    return jsonify({
        "status":        "ok" if db_status["db"] == "ok" else "degraded",
        "version":       "10.0.0",
        "encryption":    "RSA-2048-OAEP + AES-256-GCM",
        "database":      db_status,
        "celery":        celery_status,
        "async_enabled": _ASYNC_ENABLED,
        "mlflow":        mlflow_health(),
        "prometheus":    metrics_health(),
        "auth":          {"enabled": AUTH_ENABLED, "optional": AUTH_OPTIONAL},
    })


# ── Public Key ────────────────────────────────────────────────────────────
@app.route("/api/pubkey")
def pubkey():
    return jsonify({
        "publicKey": _PUBLIC_KEY_PEM,
        "algorithm": "RSA-OAEP-SHA256",
        "keySize":   2048,
    })


# ── Upload — E2E Encrypted ────────────────────────────────────────────────
@app.route("/api/upload/encrypted", methods=["POST"])
def upload_encrypted():
    try:
        body = request.get_json(force=True) or {}
        for field in ("encryptedKey", "iv", "encryptedData", "filename"):
            if field not in body:
                return jsonify({"error": f"Missing field: {field}"}), 400
        fname = body["filename"]
        if not fname.lower().endswith(".csv"):
            return jsonify({"error": "Only .csv files are allowed."}), 400
        try:
            aes_key = _rsa_decrypt(body["encryptedKey"])
        except Exception:
            record_decrypt_failure()
            return jsonify({"error": "AES key decryption failed."}), 400
        if len(aes_key) != 32:
            return jsonify({"error": f"Expected 32-byte key, got {len(aes_key)}."}), 400
        try:
            csv_bytes = _aes_gcm_decrypt(aes_key, body["iv"], body["encryptedData"])
        except Exception:
            record_decrypt_failure()
            return jsonify({"error": "CSV decryption failed — possible tampering."}), 400

        result           = _parse_csv_bytes(csv_bytes, fname)
        result["fileId"] = _persist_upload(result, fname, True, "RSA-2048-OAEP + AES-256-GCM")
        result["encrypted"]        = True
        result["encryptionMethod"] = "RSA-2048-OAEP + AES-256-GCM"
        record_upload(encrypted=True)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Upload — Plain (legacy) ───────────────────────────────────────────────
@app.route("/api/upload", methods=["POST"])
def upload():
    try:
        if "csvFile" not in request.files:
            return jsonify({"error": "No file uploaded."}), 400
        f = request.files["csvFile"]
        if not f.filename.lower().endswith(".csv"):
            return jsonify({"error": "Only .csv files are allowed."}), 400
        result           = _parse_csv_bytes(f.read(), f.filename)
        result["fileId"] = _persist_upload(result, f.filename, False)
        result["encrypted"] = False
        record_upload(encrypted=False)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Train — Central ───────────────────────────────────────────────────────
@app.route("/api/train/central", methods=["POST"])
def api_train_central():
    """
    Submit a central training job.

    Async mode (Redis available):
        Returns immediately with {jobId, experimentId, status: "PENDING"}.
        Client polls GET /api/jobs/<jobId> for progress and result.

    Sync fallback (no Redis / CELERY_ASYNC_ENABLED=false):
        Blocks until training completes, returns result directly.
        Behaviour identical to v4.0 for local dev without Docker.
    """
    try:
        d             = request.get_json()
        target_idx    = int(d.get("targetColIndex", len(d["headers"]) - 1))
        feature_types = d.get("featureTypes", {})
        epochs        = int(d.get("epochs", 100))
        lr            = float(d.get("lr", 0.1))
        file_id       = d.get("fileId")

        # Create experiment record
        exp_id = None
        try:
            with get_db() as db:
                exp = ExperimentRepo.create(
                    db,
                    model_type=ModelType.CENTRAL,
                    hyperparameters={"epochs": epochs, "lr": lr},
                    target_col_index=target_idx,
                    feature_types=feature_types,
                    name=f"Central — {d['headers'][target_idx]}",
                    file_id=file_id,
                )
                exp_id = exp.id
        except Exception as db_exc:
            logger.warning("DB create experiment failed: %s", db_exc)

        # ── Async path ────────────────────────────────────────────────────
        if _ASYNC_ENABLED:
            from tasks.training_tasks import run_central_training
            task = run_central_training.apply_async(
                kwargs=dict(
                    experiment_id=exp_id,
                    rows=d["rows"],
                    headers=d["headers"],
                    target_col_index=target_idx,
                    feature_types=feature_types,
                    epochs=epochs,
                    lr=lr,
                ),
            )
            logger.info("Submitted central training task %s  exp=%s", task.id, exp_id)
            return jsonify({
                "jobId":        task.id,
                "experimentId": exp_id,
                "status":       "PENDING",
                "message":      "Training job submitted. Poll GET /api/jobs/<jobId> for status.",
            }), 202

        # ── Sync fallback ─────────────────────────────────────────────────
        logger.info("Async disabled — running central training synchronously")
        try:
            with get_db() as db:
                ExperimentRepo.mark_running(db, exp_id)
        except Exception: pass

        result = central_train(d["rows"], d["headers"], target_idx, feature_types, epochs, lr)

        try:
            with get_db() as db:
                ExperimentRepo.mark_completed(db, exp_id, result)
        except Exception: pass

        result["experimentId"] = exp_id
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Train — Federated ─────────────────────────────────────────────────────
@app.route("/api/train/federated", methods=["POST"])
def api_train_federated():
    """
    Submit a federated training job. Same async/sync behaviour as central.
    """
    try:
        d             = request.get_json()
        target_idx    = int(d.get("targetColIndex", len(d["headers"]) - 1))
        feature_types = d.get("featureTypes", {})
        rounds        = int(d.get("rounds", 25))
        local_epochs  = int(d.get("localEpochs", 5))
        lr            = float(d.get("lr", 0.1))
        num_clients   = int(d.get("numClients", 5))
        file_id       = d.get("fileId")

        exp_id = None
        try:
            with get_db() as db:
                exp = ExperimentRepo.create(
                    db,
                    model_type=ModelType.FEDERATED,
                    hyperparameters={
                        "rounds": rounds, "local_epochs": local_epochs,
                        "lr": lr, "num_clients": num_clients,
                    },
                    target_col_index=target_idx,
                    feature_types=feature_types,
                    name=f"Federated — {d['headers'][target_idx]}",
                    file_id=file_id,
                )
                exp_id = exp.id
        except Exception as db_exc:
            logger.warning("DB create experiment failed: %s", db_exc)

        if _ASYNC_ENABLED:
            from tasks.training_tasks import run_federated_training
            task = run_federated_training.apply_async(
                kwargs=dict(
                    experiment_id=exp_id,
                    rows=d["rows"],
                    headers=d["headers"],
                    target_col_index=target_idx,
                    feature_types=feature_types,
                    rounds=rounds,
                    local_epochs=local_epochs,
                    lr=lr,
                    num_clients=num_clients,
                ),
            )
            logger.info("Submitted federated training task %s  exp=%s", task.id, exp_id)
            return jsonify({
                "jobId":        task.id,
                "experimentId": exp_id,
                "status":       "PENDING",
                "message":      "Training job submitted. Poll GET /api/jobs/<jobId> for status.",
            }), 202

        # Sync fallback
        logger.info("Async disabled — running federated training synchronously")
        try:
            with get_db() as db:
                ExperimentRepo.mark_running(db, exp_id)
        except Exception: pass

        result = federated_train(
            d["rows"], d["headers"], target_idx, feature_types,
            rounds, local_epochs, lr, num_clients,
        )
        try:
            with get_db() as db:
                ExperimentRepo.mark_completed(db, exp_id, result)
        except Exception: pass

        result["experimentId"] = exp_id
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Job polling ───────────────────────────────────────────────────────────
@app.route("/api/jobs/<task_id>", methods=["GET"])
def get_job(task_id: str):
    """
    Poll a training job's status.

    Response shape:
    {
      "jobId":        "<celery-task-uuid>",
      "experimentId": "<postgres-uuid>",
      "status":       "PENDING" | "STARTED" | "SUCCESS" | "FAILURE" | "REVOKED",
      "result":       { ... },   // present when SUCCESS
      "error":        "...",     // present when FAILURE
      "experiment":   { ... }    // live Postgres row, always present if exp exists
    }
    """
    exp_id = request.args.get("experimentId")
    return jsonify(_job_response(task_id, exp_id))


@app.route("/api/jobs/<task_id>/cancel", methods=["POST"])
def cancel_job(task_id: str):
    """Revoke (cancel) a pending or running task."""
    try:
        celery.control.revoke(task_id, terminate=True, signal="SIGTERM")
        exp_id = request.get_json(silent=True, force=True).get("experimentId")
        if exp_id:
            try:
                with get_db() as db:
                    ExperimentRepo.mark_failed(db, exp_id, "Cancelled by user")
            except Exception: pass
        return jsonify({"jobId": task_id, "status": "REVOKED"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Train — DP Federated (Day 3) ──────────────────────────────────────────
@app.route("/api/train/dp-federated", methods=["POST"])
def api_train_dp_federated():
    """
    Submit a differentially private federated training job.

    Additional body fields vs /api/train/federated:
      targetEpsilon   float  — total ε privacy budget (default 2.0)
      delta           float  — failure probability δ (default 1e-5)
      clipThreshold   float  — L2 gradient clipping bound C (default 1.0)
      noiseMultiplier float  — z = σ/C; omit to auto-calibrate from ε,δ,T

    Async mode: returns {jobId, experimentId, status:"PENDING"} → 202
    Sync fallback: blocks and returns full result dict

    The result and /api/experiments/<id> response include a `privacy`
    sub-dict with the full Moments Accountant budget accounting.
    """
    exp_id = None
    try:
        d             = request.get_json()
        target_idx    = int(d.get("targetColIndex", len(d["headers"]) - 1))
        feature_types = d.get("featureTypes", {})
        rounds        = int(d.get("rounds", 25))
        local_epochs  = int(d.get("localEpochs", 5))
        lr            = float(d.get("lr", 0.1))
        num_clients   = int(d.get("numClients", 5))
        file_id       = d.get("fileId")

        # DP-specific parameters
        target_epsilon   = float(d.get("targetEpsilon", 2.0))
        delta            = float(d.get("delta", 1e-5))
        clip_threshold   = float(d.get("clipThreshold", 1.0))
        noise_multiplier = d.get("noiseMultiplier")   # None → auto-calibrate
        if noise_multiplier is not None:
            noise_multiplier = float(noise_multiplier)

        # Validation
        if target_epsilon <= 0:
            return jsonify({"error": "targetEpsilon must be > 0"}), 400
        if not (0 < delta < 1):
            return jsonify({"error": "delta must be in (0, 1)"}), 400
        if clip_threshold <= 0:
            return jsonify({"error": "clipThreshold must be > 0"}), 400

        hp = {
            "rounds": rounds, "local_epochs": local_epochs,
            "lr": lr, "num_clients": num_clients,
            "target_epsilon": target_epsilon, "delta": delta,
            "clip_threshold": clip_threshold,
            "noise_multiplier": noise_multiplier,
        }

        try:
            with get_db() as db:
                exp = ExperimentRepo.create(
                    db,
                    model_type=ModelType.DP_FEDERATED,
                    hyperparameters=hp,
                    target_col_index=target_idx,
                    feature_types=feature_types,
                    name=f"DP-Federated — {d['headers'][target_idx]} (ε={target_epsilon})",
                    file_id=file_id,
                    dp_enabled=True,
                    dp_target_epsilon=target_epsilon,
                    dp_delta=delta,
                    dp_clip_threshold=clip_threshold,
                    dp_noise_multiplier=noise_multiplier,
                )
                exp_id = exp.id
        except Exception as db_exc:
            logger.warning("DB create DP experiment failed: %s", db_exc)

        # ── Async path ────────────────────────────────────────────────────
        if _ASYNC_ENABLED:
            from tasks.training_tasks import run_dp_federated_training
            task = run_dp_federated_training.apply_async(
                kwargs=dict(
                    experiment_id=exp_id,
                    rows=d["rows"],
                    headers=d["headers"],
                    target_col_index=target_idx,
                    feature_types=feature_types,
                    rounds=rounds,
                    local_epochs=local_epochs,
                    lr=lr,
                    num_clients=num_clients,
                    target_epsilon=target_epsilon,
                    delta=delta,
                    clip_threshold=clip_threshold,
                    noise_multiplier=noise_multiplier,
                ),
            )
            logger.info(
                "Submitted DP-federated task %s  exp=%s  ε_target=%.2f",
                task.id, exp_id, target_epsilon,
            )
            return jsonify({
                "jobId":          task.id,
                "experimentId":   exp_id,
                "status":         "PENDING",
                "dpEnabled":      True,
                "targetEpsilon":  target_epsilon,
                "delta":          delta,
                "message": (
                    "DP training job submitted. "
                    "Poll GET /api/jobs/<jobId> for status. "
                    "GET /api/privacy/budget/<experimentId> for live ε accounting."
                ),
            }), 202

        # ── Sync fallback ─────────────────────────────────────────────────
        logger.info("Async disabled — running DP-federated training synchronously")
        try:
            with get_db() as db:
                ExperimentRepo.mark_running(db, exp_id)
        except Exception:
            pass

        result = dp_federated_train(
            d["rows"], d["headers"], target_idx, feature_types,
            rounds=rounds, local_epochs=local_epochs, lr=lr,
            num_clients=num_clients,
            target_epsilon=target_epsilon, delta=delta,
            clip_threshold=clip_threshold, noise_multiplier=noise_multiplier,
        )

        try:
            with get_db() as db:
                ExperimentRepo.mark_completed(db, exp_id, result)
        except Exception:
            pass

        result["experimentId"] = exp_id
        return jsonify(result)

    except Exception as e:
        if exp_id:
            try:
                with get_db() as db:
                    ExperimentRepo.mark_failed(db, exp_id, str(e))
            except Exception:
                pass
        return jsonify({"error": str(e)}), 500


# ── Privacy Budget endpoint (Day 3) ───────────────────────────────────────
@app.route("/api/privacy/budget/<exp_id>", methods=["GET"])
def get_privacy_budget(exp_id: str):
    """
    Return the live privacy budget accounting for a DP experiment.

    Response:
    {
      "experimentId":      "<uuid>",
      "dpEnabled":         true,
      "targetEpsilon":     2.0,
      "currentEpsilon":    1.234,   ← actual spend so far
      "remainingEpsilon":  0.766,
      "budgetFractionUsed": 0.617,
      "delta":             1e-5,
      "noiseMultiplier":   1.1,
      "roundsConsumed":    18,
      "epsilonHistory":    [0.08, 0.16, ...],  ← per-round epsilon
      "status":            "completed" | "running" | ...
    }
    """
    try:
        with get_db() as db:
            exp = ExperimentRepo.get_by_id(db, exp_id)
            if not exp:
                return jsonify({"error": "Experiment not found."}), 404

            if not exp.dp_enabled:
                return jsonify({
                    "error": "This experiment was not run with differential privacy.",
                    "experimentId": exp_id,
                    "dpEnabled": False,
                }), 400

            payload = {
                "experimentId":       exp_id,
                "dpEnabled":          True,
                "status":             exp.status.value,
                "targetEpsilon":      exp.dp_target_epsilon,
                "delta":              exp.dp_delta,
                "clipThreshold":      exp.dp_clip_threshold,
                "noiseMultiplier":    exp.dp_noise_multiplier,
            }

            # If completed, pull the full accounting from the result row
            if exp.result and exp.result.dp_budget_summary:
                bs = exp.result.dp_budget_summary
                payload.update({
                    "currentEpsilon":      bs.get("current_epsilon"),
                    "remainingEpsilon":    bs.get("remaining_epsilon"),
                    "budgetFractionUsed":  bs.get("budget_fraction_used"),
                    "roundsConsumed":      bs.get("rounds_consumed"),
                    "epsilonHistory":      bs.get("epsilon_history", []),
                    "noiseSigma":          exp.result.dp_noise_sigma,
                })
            else:
                # Not yet completed — return what we know from the experiment row
                payload.update({
                    "currentEpsilon":     None,
                    "remainingEpsilon":   None,
                    "budgetFractionUsed": None,
                    "roundsConsumed":     None,
                    "epsilonHistory":     [],
                    "note": "Training in progress — budget will be available on completion.",
                })

            return jsonify(payload)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Train — SecAgg Federated (Day 4) ──────────────────────────────────────
@app.route("/api/train/secagg-federated", methods=["POST"])
def api_train_secagg_federated():
    """
    Submit a SecAgg-protected federated training job.

    The server learns only the *sum* of client updates each round —
    individual gradients are never visible (honest-but-curious model).

    Additional body fields vs /api/train/federated:
      dropoutRate      float  — fraction of clients to simulate dropping (default 0.0)
      secaggThreshold  int    — Shamir reconstruction threshold (default: ceil(n/2))

    Async mode: returns {jobId, experimentId, status:"PENDING"} → 202
    Sync fallback: blocks and returns full result dict

    The result includes a `secagg` sub-dict with per-round audit log
    and a boolean `all_rounds_verified` confirming mask cancellation.
    """
    exp_id = None
    try:
        d             = request.get_json()
        target_idx    = int(d.get("targetColIndex", len(d["headers"]) - 1))
        feature_types = d.get("featureTypes", {})
        rounds        = int(d.get("rounds", 25))
        local_epochs  = int(d.get("localEpochs", 5))
        lr            = float(d.get("lr", 0.1))
        num_clients   = int(d.get("numClients", 5))
        file_id       = d.get("fileId")

        # SecAgg-specific parameters
        dropout_rate     = float(d.get("dropoutRate", 0.0))
        secagg_threshold = int(d.get("secaggThreshold", 0))   # 0 = auto

        # Validation
        if not (0.0 <= dropout_rate < 1.0):
            return jsonify({"error": "dropoutRate must be in [0, 1)"}), 400
        if num_clients < 2:
            return jsonify({"error": "numClients must be ≥ 2 for SecAgg"}), 400
        if secagg_threshold > num_clients:
            return jsonify({"error": "secaggThreshold cannot exceed numClients"}), 400

        # Effective threshold (for recording)
        eff_threshold = secagg_threshold if secagg_threshold > 0 else max(2, (num_clients + 1) // 2)

        hp = {
            "rounds": rounds, "local_epochs": local_epochs,
            "lr": lr, "num_clients": num_clients,
            "dropout_rate": dropout_rate, "secagg_threshold": eff_threshold,
        }

        try:
            with get_db() as db:
                exp = ExperimentRepo.create(
                    db,
                    model_type=ModelType.SECAGG_FEDERATED,
                    hyperparameters=hp,
                    target_col_index=target_idx,
                    feature_types=feature_types,
                    name=f"SecAgg-Federated — {d['headers'][target_idx]}",
                    file_id=file_id,
                    secagg_enabled=True,
                    secagg_threshold=eff_threshold,
                    secagg_dropout=dropout_rate,
                )
                exp_id = exp.id
        except Exception as db_exc:
            logger.warning("DB create SecAgg experiment failed: %s", db_exc)

        # ── Async path ────────────────────────────────────────────────────
        if _ASYNC_ENABLED:
            from tasks.training_tasks import run_secagg_federated_training
            task = run_secagg_federated_training.apply_async(
                kwargs=dict(
                    experiment_id=exp_id,
                    rows=d["rows"],
                    headers=d["headers"],
                    target_col_index=target_idx,
                    feature_types=feature_types,
                    rounds=rounds,
                    local_epochs=local_epochs,
                    lr=lr,
                    num_clients=num_clients,
                    dropout_rate=dropout_rate,
                    secagg_threshold=eff_threshold,
                ),
            )
            logger.info(
                "Submitted SecAgg-federated task %s  exp=%s  dropout=%.0f%%",
                task.id, exp_id, dropout_rate * 100,
            )
            return jsonify({
                "jobId":          task.id,
                "experimentId":   exp_id,
                "status":         "PENDING",
                "secaggEnabled":  True,
                "threshold":      eff_threshold,
                "dropoutRate":    dropout_rate,
                "message": (
                    "SecAgg training job submitted. "
                    "Poll GET /api/jobs/<jobId> for status. "
                    "GET /api/secagg/audit/<experimentId> for round audit log."
                ),
            }), 202

        # ── Sync fallback ─────────────────────────────────────────────────
        logger.info("Async disabled — running SecAgg training synchronously")
        try:
            with get_db() as db:
                ExperimentRepo.mark_running(db, exp_id)
        except Exception:
            pass

        result = secagg_federated_train(
            d["rows"], d["headers"], target_idx, feature_types,
            rounds=rounds, local_epochs=local_epochs, lr=lr,
            num_clients=num_clients,
            dropout_rate=dropout_rate,
            secagg_threshold=secagg_threshold,
        )

        try:
            with get_db() as db:
                ExperimentRepo.mark_completed(db, exp_id, result)
        except Exception:
            pass

        result["experimentId"] = exp_id
        return jsonify(result)

    except Exception as e:
        if exp_id:
            try:
                with get_db() as db:
                    ExperimentRepo.mark_failed(db, exp_id, str(e))
            except Exception:
                pass
        return jsonify({"error": str(e)}), 500


# ── SecAgg Audit endpoint (Day 4) ─────────────────────────────────────────
@app.route("/api/secagg/audit/<exp_id>", methods=["GET"])
def get_secagg_audit(exp_id: str):
    """
    Return the SecAgg round-by-round audit log for an experiment.

    Response:
    {
      "experimentId":       "<uuid>",
      "secaggEnabled":      true,
      "protocol":           "SecAgg (Bonawitz et al. 2017)",
      "nClients":           5,
      "threshold":          3,
      "dropoutRate":        0.2,
      "roundsCompleted":    25,
      "roundsAborted":      0,
      "allRoundsVerified":  true,      ← mask cancellation verified every round
      "totalOverheadMs":    12.4,
      "perRoundLog": [
        {"round":1, "n_survived":4, "n_dropped":1, "verified":true, "overhead_ms":0.5},
        ...
      ]
    }
    """
    try:
        with get_db() as db:
            exp = ExperimentRepo.get_by_id(db, exp_id)
            if not exp:
                return jsonify({"error": "Experiment not found."}), 404

            if not exp.secagg_enabled:
                return jsonify({
                    "error": "This experiment was not run with Secure Aggregation.",
                    "experimentId": exp_id,
                    "secaggEnabled": False,
                }), 400

            payload = {
                "experimentId":  exp_id,
                "secaggEnabled": True,
                "status":        exp.status.value,
                "nClients":      exp.hyperparameters.get("num_clients"),
                "threshold":     exp.secagg_threshold,
                "dropoutRate":   exp.secagg_dropout,
            }

            if exp.result and exp.result.secagg_summary:
                s = exp.result.secagg_summary
                payload.update({
                    "protocol":         s.get("protocol"),
                    "roundsCompleted":  s.get("rounds_completed"),
                    "roundsAborted":    s.get("rounds_aborted"),
                    "allRoundsVerified": exp.result.secagg_all_verified,
                    "totalOverheadMs":  s.get("total_overhead_ms"),
                    "perRoundLog":      s.get("per_round_log", []),
                })
            else:
                payload.update({
                    "protocol":         "SecAgg (Bonawitz et al. 2017)",
                    "roundsCompleted":  None,
                    "allRoundsVerified": None,
                    "perRoundLog":      [],
                    "note": "Training in progress — audit log available on completion.",
                })

            return jsonify(payload)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Experiments API ───────────────────────────────────────────────────────
@app.route("/api/experiments", methods=["GET"])
def list_experiments():
    try:
        limit  = int(request.args.get("limit", 50))
        offset = int(request.args.get("offset", 0))
        with get_db() as db:
            exps = ExperimentRepo.list_all(db, limit=limit, offset=offset)
            return jsonify([e.to_dict() for e in exps])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/experiments/<exp_id>", methods=["GET"])
def get_experiment(exp_id: str):
    try:
        with get_db() as db:
            exp = ExperimentRepo.get_by_id(db, exp_id)
            if not exp:
                return jsonify({"error": "Experiment not found."}), 404
            return jsonify(exp.to_dict(include_result=True))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Files API ─────────────────────────────────────────────────────────────
@app.route("/api/files", methods=["GET"])
def list_files():
    try:
        limit  = int(request.args.get("limit", 50))
        offset = int(request.args.get("offset", 0))
        with get_db() as db:
            files = FileRepo.list_all(db, limit=limit, offset=offset)
            return jsonify([f.to_dict() for f in files])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/files/<file_id>", methods=["GET"])
def get_file(file_id: str):
    try:
        with get_db() as db:
            f = FileRepo.get_by_id(db, file_id)
            if not f:
                return jsonify({"error": "File not found."}), 404
            return jsonify(f.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Auth API (Day 8) ──────────────────────────────────────────────────────
@app.route("/api/auth/token", methods=["POST"])
def auth_token():
    """
    Issue JWT access + refresh tokens.

    In production this endpoint is replaced by your OAuth provider's
    /oauth/token endpoint (Auth0 / Keycloak).  This local endpoint is
    provided for development and for users of the 'local' provider.

    Body: { "user_id": "...", "email": "...", "role": "...", "attributes": {} }
    Returns: { access_token, refresh_token, token_type, expires_in }
    """
    try:
        body = request.get_json(force=True) or {}
        user_id    = body.get("user_id",    "dev-user")
        email      = body.get("email",      "dev@localhost")
        role       = body.get("role",       "trainer")
        attributes = body.get("attributes", {})

        pair = create_token_pair(user_id, email, role, attributes)
        AuditLogger.log(
            action=AuditAction.LOGIN, outcome=AuditOutcome.SUCCESS,
            user_id=user_id, email=email, role=role,
            ip_address=request.remote_addr or "",
            details={"provider": "local"},
        )
        return jsonify(pair.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/auth/refresh", methods=["POST"])
def auth_refresh():
    """
    Exchange a refresh token for a new access token.
    Body: { "refresh_token": "..." }
    """
    try:
        from auth.jwt_handler import decode_refresh_token
        body    = request.get_json(force=True) or {}
        rt      = body.get("refresh_token", "")
        if not rt:
            return jsonify({"error": "refresh_token required"}), 400

        claims  = decode_refresh_token(rt)
        # Load user from DB to get current role + attributes
        user_id = claims.user_id
        role    = "trainer"
        email   = ""
        attrs   = {}
        try:
            with get_db() as db:
                user = UserRepo.get_by_id(db, user_id)
                if user:
                    role  = user.role.value
                    email = user.email
                    attrs = user.abac_attributes or {}
        except Exception:
            pass

        pair = create_token_pair(user_id, email, role, attrs)
        AuditLogger.log(
            action=AuditAction.TOKEN_REFRESH, outcome=AuditOutcome.SUCCESS,
            user_id=user_id, email=email,
            ip_address=request.remote_addr or "",
        )
        return jsonify({
            "access_token": pair.access_token,
            "token_type":   "Bearer",
            "expires_in":   JWT_ACCESS_TOKEN_TTL,
        })
    except TokenError as e:
        AuditLogger.log(
            action=AuditAction.AUTH_FAILED, outcome=AuditOutcome.FAILURE,
            ip_address=request.remote_addr or "",
            details={"reason": str(e)},
        )
        return jsonify({"error": str(e), "code": e.code}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/auth/logout", methods=["POST"])
@require_auth
def auth_logout():
    """Revoke the current access token."""
    try:
        claims = get_current_user()
        if claims:
            revoke_token(claims.jti, JWT_ACCESS_TOKEN_TTL)
            AuditLogger.log(
                action=AuditAction.LOGOUT, outcome=AuditOutcome.SUCCESS,
                user_id=claims.user_id, email=claims.email, role=claims.role,
                ip_address=request.remote_addr or "",
            )
        return jsonify({"message": "Logged out successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Audit log API (Day 8) ─────────────────────────────────────────────────
@app.route("/api/audit", methods=["GET"])
@require_auth
@require_role("admin")
def list_audit_log():
    """
    Return recent audit log entries.  Admin only.
    Query params: limit (default 100), user_id (filter by user)
    """
    try:
        limit   = int(request.args.get("limit", 100))
        user_id = request.args.get("user_id")
        entries = AuditLogger.list_recent(limit=limit, user_id=user_id)
        log_from_request(AuditAction.AUDIT_LOG_READ, AuditOutcome.SUCCESS,
                         details={"limit": limit, "filter_user": user_id})
        return jsonify(entries)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/audit/verify", methods=["GET"])
@require_auth
@require_role("admin")
def verify_audit_chain():
    """Verify the hash chain integrity of the audit log.  Admin only."""
    try:
        limit = int(request.args.get("limit", 1000))
        ok, reason = AuditLogger.verify_chain(limit=limit)
        return jsonify({
            "intact":        ok,
            "entries_checked": limit,
            "reason":        reason,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Users API (Day 8) ─────────────────────────────────────────────────────
@app.route("/api/users", methods=["GET"])
@require_auth
@require_role("admin")
def list_users():
    """List all users.  Admin only."""
    try:
        with get_db() as db:
            users = UserRepo.list_all(db, limit=100)
            return jsonify([u.to_dict() for u in users])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/users", methods=["POST"])
@require_auth
@require_role("admin")
def create_user():
    """Create a user.  Admin only."""
    try:
        body = request.get_json(force=True) or {}
        with get_db() as db:
            from database.models import UserRole as UR
            user = UserRepo.create(
                db,
                username=body.get("username", ""),
                email=body.get("email", ""),
                role=UR(body.get("role", "trainer")),
            )
        log_from_request(AuditAction.USER_CREATED, AuditOutcome.SUCCESS,
                         resource_id=user.id, resource_type="user",
                         details={"email": body.get("email")})
        return jsonify(user.to_dict()), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Prometheus metrics endpoint (Day 7) ──────────────────────────────────
@app.route("/metrics")
def prometheus_metrics():
    """
    Prometheus scrape endpoint.  Add to prometheus.yml:

        scrape_configs:
          - job_name: 'training-models'
            static_configs:
              - targets: ['backend:8080']

    In multiprocess mode (Gunicorn workers), set PROMETHEUS_MULTIPROC_DIR
    to a shared directory so all worker processes contribute their metrics.
    """
    data, content_type = generate_metrics_output()
    from flask import Response
    return Response(data, mimetype=content_type)


# ── MLflow API (Day 6) ────────────────────────────────────────────────────
@app.route("/api/mlflow/status", methods=["GET"])
def mlflow_status():
    """Return MLflow tracking server status and configuration."""
    return jsonify({
        "enabled":       MLFLOW_ENABLED,
        "tracking_uri":  MLFLOW_TRACKING_URI,
        **mlflow_health(),
    })


@app.route("/api/mlflow/experiments", methods=["GET"])
def mlflow_experiments():
    """
    List MLflow experiments (one per model_type).
    Requires mlflow to be installed and reachable.
    """
    try:
        import mlflow
        client = mlflow.MlflowClient()
        exps   = client.search_experiments()
        return jsonify([
            {
                "experiment_id":   e.experiment_id,
                "name":            e.name,
                "artifact_location": e.artifact_location,
                "lifecycle_stage": e.lifecycle_stage,
            }
            for e in exps
        ])
    except ImportError:
        return jsonify({"error": "mlflow not installed"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/mlflow/runs/<exp_id>", methods=["GET"])
def mlflow_runs_for_experiment(exp_id: str):
    """
    List MLflow runs for a given Postgres experiment UUID.
    Searches MLflow runs by the experiment_id tag set during log_run().
    """
    try:
        import mlflow
        client = mlflow.MlflowClient()
        runs   = client.search_runs(
            experiment_ids=[],   # search all MLflow experiments
            filter_string=f'tags.experiment_id = "{exp_id}"',
            max_results=50,
        )
        return jsonify([
            {
                "run_id":       r.info.run_id,
                "status":       r.info.status,
                "start_time":   r.info.start_time,
                "metrics":      r.data.metrics,
                "params":       r.data.params,
                "tags":         r.data.tags,
            }
            for r in runs
        ])
    except ImportError:
        return jsonify({"error": "mlflow not installed"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/mlflow/registry", methods=["GET"])
def mlflow_registry():
    """List all registered models in the MLflow Model Registry."""
    try:
        import mlflow
        client = mlflow.MlflowClient()
        models = client.search_registered_models()
        return jsonify([
            {
                "name":            m.name,
                "latest_versions": [
                    {
                        "version":        v.version,
                        "stage":          v.current_stage,
                        "run_id":         v.run_id,
                        "creation_time":  v.creation_timestamp,
                    }
                    for v in m.latest_versions
                ],
            }
            for m in models
        ])
    except ImportError:
        return jsonify({"error": "mlflow not installed"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Startup ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═" * 76)
    print("  Training Models  v10.0  —  OAuth + Audit + K8s + Prometheus + MLflow")
    print("═" * 76)
    try:
        init_db()
        print("  🗄️   Database        →  schema ready")
    except Exception as exc:
        print(f"  ⚠️   Database        →  UNAVAILABLE ({exc})")
        print("       Set DATABASE_URL or USE_SQLITE_FALLBACK=true")
    print(f"  🌐  Web UI           →  http://localhost:8080")
    print(f"  🔑  Auth Token       →  POST /api/auth/token")
    print(f"  🔄  Refresh Token    →  POST /api/auth/refresh")
    print(f"  🚪  Logout           →  POST /api/auth/logout")
    print(f"  📋  Audit Log        →  GET  /api/audit  (admin only)")
    print(f"  🔍  Verify Chain     →  GET  /api/audit/verify  (admin only)")
    print(f"  👥  Users            →  GET  /api/users  (admin only)")
    print(f"  🔒  Enc Upload       →  POST /api/upload/encrypted")
    print(f"  🚀  Async Train      →  POST /api/train/central|federated       → 202")
    print(f"  🔐  DP Train         →  POST /api/train/dp-federated            → 202")
    print(f"  🛡️   SecAgg Train     →  POST /api/train/secagg-federated        → 202")
    print(f"  📊  Poll Job         →  GET  /api/jobs/<jobId>")
    print(f"  🧮  Privacy Budget   →  GET  /api/privacy/budget/<expId>")
    print(f"  📈  MLflow Status    →  GET  /api/mlflow/status")
    print(f"  🔥  Prometheus       →  GET  /metrics")
    print(f"  🧪  Experiments      →  GET  /api/experiments")
    print(f"  ⚡  Async mode       →  {'ENABLED' if _ASYNC_ENABLED else 'DISABLED'}")
    print(f"  🔐  Auth             →  {'ENABLED' if AUTH_ENABLED else 'DISABLED'}"
          + (" (OPTIONAL — dev mode)" if AUTH_OPTIONAL else ""))
    print("═" * 76 + "\n")
    app.run(host="0.0.0.0", port=8080, debug=False)
