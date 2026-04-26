/// api.dart  —  HTTP client  +  E2E encrypted CSV upload
/// ═══════════════════════════════════════════════════════════════
/// Upload flow:
///   1. initEncryption()  →  GET /api/pubkey  →  load RSA key
///   2. upload(file)      →  encrypt → POST /api/upload/encrypted
///   3. Fallback          →  plain POST /api/upload  if key missing
/// ═══════════════════════════════════════════════════════════════
library;

import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import '../models/models.dart';
import 'encryption_service.dart';
import 'package:flutter/foundation.dart';

class Api {
  static String _base = kIsWeb ? 'https://federated-learning-backend.onrender.com/api' : 'https://federated-learning-backend.onrender.com/api';
  static String get base => _base;
  static String? _token;

  static void setBase(String url) {
    final u = url.trimRight().replaceAll(RegExp(r'/$'), '');
    _base = u.endsWith('/api') ? u : '$u/api';
  }

  static void setToken(String? token) => _token = token;

  static Map<String, String> get _headers => {
    'Content-Type': 'application/json',
    if (_token != null) 'Authorization': 'Bearer $_token',
  };

  // ── Health check ─────────────────────────────────────────────
  static Future<bool> ping() async {
    try {
      debugPrint('Attempting to ping: $_base/health');
      final r = await http
          .get(Uri.parse('$_base/health'), headers: _headers)
          .timeout(const Duration(seconds: 30));
      debugPrint('Ping response: ${r.statusCode}');
      return r.statusCode == 200;
    } catch (e) {
      debugPrint('Ping failed: $e');
      return false;
    }
  }

  // ── Authentication ───────────────────────────────────────────
  static Future<AuthToken> login(String id, String email, String role) async {
    final r = await http.post(
      Uri.parse('$_base/auth/token'),
      headers: {'Content-Type': 'application/json'}, // plain login doesn't need auth header
      body: jsonEncode({'user_id': id, 'email': email, 'role': role}),
    ).timeout(const Duration(seconds: 30));
    
    final j = jsonDecode(r.body) as Map<String, dynamic>;
    if (r.statusCode != 200) throw Exception(j['error'] ?? 'Login failed');
    final auth = AuthToken.fromJson(j);
    setToken(auth.access);
    return auth;
  }

  // ── Initialise E2E encryption ────────────────────────────────
  /// Call once after ping() succeeds.
  /// Fetches RSA-2048 public key from /api/pubkey and loads it
  /// into EncryptionService so uploads can be encrypted.
  static Future<bool> initEncryption() async {
    try {
      final r = await http
          .get(Uri.parse('$_base/pubkey'), headers: _headers)
          .timeout(const Duration(seconds: 30));
      if (r.statusCode != 200) return false;

      final j = jsonDecode(r.body) as Map<String, dynamic>;
      final pem = j['publicKey'] as String? ?? '';
      if (pem.isEmpty) return false;

      EncryptionService.instance.loadPublicKeyFromPem(pem);
      return true;
    } catch (_) {
      return false;
    }
  }

  // ── CSV upload (E2E encrypted when key is available) ─────────
  static Future<CsvData> upload(Uint8List bytes, String filename) async {

    if (EncryptionService.instance.isReady) {
      // ── ENCRYPTED PATH ────────────────────────────────────────
      final enc = EncryptionService.instance.encrypt(Uint8List.fromList(bytes));

      final body = jsonEncode({
        'encryptedKey':  enc['encryptedKey'],
        'iv':            enc['iv'],
        'encryptedData': enc['encryptedData'],
        'filename':      filename,
      });

      final r = await http
          .post(
            Uri.parse('$_base/upload/encrypted'),
            headers: _headers,
            body: body,
          )
          .timeout(const Duration(seconds: 60));

      final j = jsonDecode(r.body) as Map<String, dynamic>;
      if (r.statusCode != 200) throw Exception(j['error'] ?? 'Encrypted upload failed');

      final csv = CsvData.fromJson(j);
      csv.isEncrypted = true;
      return csv;
    } else {
      // ── PLAIN PATH (JSON-based for reliability) ─────────────
      final r = await http.post(
        Uri.parse('$_base/upload'),
        headers: _headers,
        body: jsonEncode({
          'data': base64Encode(bytes),
          'filename': filename,
        }),
      ).timeout(const Duration(seconds: 60));
      
      final j = jsonDecode(r.body) as Map<String, dynamic>;
      if (r.statusCode != 200) throw Exception(j['error'] ?? 'Upload failed');

      final csv = CsvData.fromJson(j);
      csv.isEncrypted = false;
      return csv;
    }
  }

  // ── Train: Central ───────────────────────────────────────────
  static Future<TrainResult> central({
    required List<Map<String, dynamic>> rows,
    required List<String> headers,
    required int targetIdx,
    required Map<String, String> ftypes,
    int epochs = 100,
    double lr = 0.1,
    Algorithm algo = Algorithm.logistic,
  }) async {
    final r = await http
        .post(
          Uri.parse('$_base/train/central'),
          headers: _headers,
          body: jsonEncode({
            'rows': rows, 'headers': headers,
            'targetColIndex': targetIdx, 'featureTypes': ftypes,
            'epochs': epochs, 'lr': lr,
            'algo': algo.name,
          }),
        )
        .timeout(const Duration(seconds: 180));
    final j = jsonDecode(r.body) as Map<String, dynamic>;
    if (r.statusCode != 200) throw Exception(j['error'] ?? 'Training failed');
    return TrainResult.fromJson(j);
  }

  // ── Train: Federated ─────────────────────────────────────────
  static Future<TrainResult> federated({
    required List<Map<String, dynamic>> rows,
    required List<String> headers,
    required int targetIdx,
    required Map<String, String> ftypes,
    int rounds = 25,
    int localEpochs = 5,
    double lr = 0.1,
    int numClients = 5,
    Algorithm algo = Algorithm.logistic,
  }) async {
    final r = await http
        .post(
          Uri.parse('$_base/train/federated'),
          headers: _headers,
          body: jsonEncode({
            'rows': rows, 'headers': headers,
            'targetColIndex': targetIdx, 'featureTypes': ftypes,
            'rounds': rounds, 'localEpochs': localEpochs,
            'lr': lr, 'numClients': numClients,
            'algo': algo.name,
          }),
        )
        .timeout(const Duration(seconds: 180));
    final j = jsonDecode(r.body) as Map<String, dynamic>;
    if (r.statusCode != 200) throw Exception(j['error'] ?? 'Training failed');
    return TrainResult.fromJson(j);
  }

  // ── Poll Job Status ──────────────────────────────────────────
  static Future<Map<String, dynamic>> pollJob(String jobId) async {
    final r = await http
        .get(Uri.parse('$_base/jobs/$jobId'), headers: _headers)
        .timeout(const Duration(seconds: 10));
    final j = jsonDecode(r.body) as Map<String, dynamic>;
    if (r.statusCode != 200) throw Exception(j['error'] ?? 'Polling failed');
    return j;
  }

  // ── Experiments ──────────────────────────────────────────────
  static Future<List<Experiment>> getExperiments() async {
    final r = await http
        .get(Uri.parse('$_base/experiments?limit=50'), headers: _headers)
        .timeout(const Duration(seconds: 15));
    final j = jsonDecode(r.body) as List;
    return j.map((e) => Experiment.fromJson(e)).toList();
  }

  // ── Predict ──────────────────────────────────────────────────
  static Future<PredictionResult> predict(String expId, Map<String, dynamic> inputs) async {
    final r = await http
        .post(
          Uri.parse('$_base/predict'),
          headers: _headers,
          body: jsonEncode({'experimentId': expId, 'inputs': inputs}),
        )
        .timeout(const Duration(seconds: 15));
    final j = jsonDecode(r.body) as Map<String, dynamic>;
    if (r.statusCode != 200) throw Exception(j['error'] ?? 'Prediction failed');
    return PredictionResult.fromJson(j);
  }
}
