# Training Models — End-to-End Encrypted CSV Upload  (v3.0)

> **FL vs Central Training** — Offline-first | Web SPA + Flutter Android
> **NEW in v3.0:** End-to-End Encryption for all CSV uploads

---

## 🔐 Encryption Protocol

```
CLIENT (browser / Flutter)                   SERVER (Flask)
──────────────────────────                   ──────────────────────────
                           GET /api/pubkey
              ─────────────────────────────────────────→
                     { publicKey: "-----BEGIN PUBLIC KEY-----…" }
              ←─────────────────────────────────────────

  1. Generate random AES-256 key + 12-byte IV
  2. Encrypt CSV bytes with AES-256-GCM
  3. Wrap AES key with RSA-OAEP-SHA256 (server public key)
                    POST /api/upload/encrypted
              ─────────────────────────────────────────→
              { encryptedKey, iv, encryptedData, filename }
                                              4. Unwrap AES key (RSA private key)
                                              5. Decrypt CSV (AES-256-GCM + tag check)
                                              6. Parse and return stats
              ←─────────────────────────────────────────
                     { filename, rows, stats, encrypted:true }
```

### Security Properties

| Property | Guarantee |
|----------|-----------|
| **Confidentiality** | CSV bytes encrypted with AES-256-GCM before leaving the device |
| **Key security** | AES session key wrapped with RSA-2048-OAEP; private key never leaves server memory |
| **Integrity** | GCM authentication tag (16 bytes) detects any in-transit tampering |
| **Forward secrecy** | Fresh random AES key + IV generated for every single upload |
| **No persistence** | RSA private key is in-memory only; regenerated on each server restart |
| **Backward compat** | Plain `POST /api/upload` still works for legacy clients |

---

## 🚀 Quick Start

### 1 — Install & start the backend

```bash
cd backend
pip install -r requirements.txt   # flask, flask-cors, cryptography
python app.py
```

Server console output:
```
🔐  Generating RSA-2048 key pair … done ✓
══════════════════════════════════════════════════════════════
  Training Models  v3.0  —  E2E Encrypted Upload
══════════════════════════════════════════════════════════════
  🌐  Web UI       →  http://localhost:8080
  🔑  Public Key   →  GET  /api/pubkey
  🔒  Enc Upload   →  POST /api/upload/encrypted
  📂  Plain Upload →  POST /api/upload  (legacy)
══════════════════════════════════════════════════════════════
```

### 2 — Open the Web UI

Visit **http://localhost:8080** — the encryption banner will show
🔐 **E2E Encryption Ready** once the RSA key is loaded.

### 3 — Run the Flutter Android app

```bash
cd flutter_app
flutter pub get
flutter run
```

The upload screen shows:
- 🔐 **End-to-End Encryption Active** — when server key is loaded
- CSV is encrypted on-device before transmission
- Encryption badge confirmed on the Data explorer screen

---

## 📁 Changed Files (v2 → v3)

| File | Change |
|------|--------|
| `backend/app.py` | + RSA key generation; + `/api/pubkey`; + `/api/upload/encrypted` |
| `backend/requirements.txt` | + `cryptography==42.0.8` |
| `backend/templates/index.html` | + `initEncryption()`, `encryptFile()` using Web Crypto API |
| `flutter_app/lib/services/encryption_service.dart` | **NEW** — pointycastle RSA-OAEP + AES-256-GCM |
| `flutter_app/lib/services/api.dart` | + `initEncryption()`; upload uses encrypted endpoint |
| `flutter_app/lib/screens/upload.dart` | + E2E status card, encryption badge |
| `flutter_app/lib/models/models.dart` | + `isEncrypted` field on `CsvData` |
| `flutter_app/pubspec.yaml` | + `pointycastle ^3.7.4`, `crypto ^3.0.3` |

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/pubkey` | Returns server RSA-2048 public key (PEM) |
| POST | `/api/upload/encrypted` | **Encrypted** CSV upload (JSON body) |
| POST | `/api/upload` | Plain CSV upload (multipart, legacy) |
| POST | `/api/train/central` | Run Central Training |
| POST | `/api/train/federated` | Run Federated Learning |
| GET | `/api/health` | Health check + version |

### Encrypted upload payload

```json
{
  "encryptedKey":  "<base64>  RSA-OAEP-SHA256 wrapped 32-byte AES key",
  "iv":            "<base64>  12-byte AES-GCM nonce",
  "encryptedData": "<base64>  AES-256-GCM ciphertext + 16-byte GCM tag",
  "filename":      "my_data.csv"
}
```

---

## Android URL Settings

| Environment | URL |
|-------------|-----|
| Android Emulator | `http://10.0.2.2:8080` |
| Physical Device | `http://[PC_LAN_IP]:8080` |
