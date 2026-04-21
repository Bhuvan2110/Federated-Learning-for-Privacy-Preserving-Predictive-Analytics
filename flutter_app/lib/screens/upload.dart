import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import '../models/models.dart';
import '../services/api.dart';
import '../theme/theme.dart';

import 'package:flutter/foundation.dart';

class UploadScreen extends StatefulWidget {
  final void Function(CsvData) onDone;
  const UploadScreen({super.key, required this.onDone});
  @override State<UploadScreen> createState() => _State();
}

class _State extends State<UploadScreen> {
  bool _loading = false;
  bool _serverOk = false;
  bool _encReady = false;       // true once RSA public key loaded
  bool _encLoading = false;     // animating key fetch
  String? _err;
  final _urlCtrl = TextEditingController(
    text: kIsWeb ? 'https://federated-learning-backend.onrender.com' : 'http://10.0.2.2:8080'
  );

  @override void initState() { super.initState(); _connect(); }
  @override void dispose() { _urlCtrl.dispose(); super.dispose(); }

  /// Ping server, then fetch RSA public key for E2E encryption
  Future<void> _connect() async {
    setState(() { _encLoading = true; _encReady = false; _serverOk = false; });
    Api.setBase(_urlCtrl.text.trim());

    final ok = await Api.ping();
    if (!mounted) return;
    setState(() => _serverOk = ok);

    if (ok) {
      final encOk = await Api.initEncryption();
      if (!mounted) return;
      setState(() { _encReady = encOk; _encLoading = false; });
    } else {
      setState(() => _encLoading = false);
    }
  }

  Future<void> _pick() async {
    setState(() { _loading = true; _err = null; });
    try {
      final r = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['csv'],
        withData: true,
        withReadStream: false,
      );
      if (r == null || r.files.single.name.isEmpty) {
        setState(() => _loading = false);
        return;
      }
      
      final bytes = r.files.single.bytes ?? Uint8List(0);
      final name = r.files.single.name;
      
      final data = await Api.upload(bytes, name);
      widget.onDone(data);
    } catch (e) {
      setState(() => _err = e.toString().replaceFirst('Exception: ', ''));
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext ctx) => Scaffold(
    body: SafeArea(child: SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [

        const SizedBox(height: 16),

        // ── App header ──────────────────────────────────────────
        Center(child: Column(children: [
          Container(
            width: 80, height: 80,
            decoration: BoxDecoration(
              gradient: const LinearGradient(colors: [T.fl, T.acc]),
              borderRadius: BorderRadius.circular(20),
              boxShadow: [BoxShadow(color: T.fl.withOpacity(0.35), blurRadius: 28)],
            ),
            child: const Center(child: Text('🔒', style: TextStyle(fontSize: 34))),
          ),
          const SizedBox(height: 16),
          const Text('TRAINING MODELS',
            style: TextStyle(fontSize: 22, fontWeight: FontWeight.w900, letterSpacing: 3)),
          const SizedBox(height: 4),
          const Text('FL vs Central  ·  E2E Encrypted',
            style: TextStyle(fontSize: 12, color: T.fl, fontFamily: 'monospace', letterSpacing: 2)),
          const SizedBox(height: 28),
        ])),

        // ── Server config card ──────────────────────────────────
        Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: T.card,
            border: Border.all(color: _serverOk ? T.ok.withOpacity(0.5) : T.border),
            borderRadius: BorderRadius.circular(14),
          ),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Row(children: [
              Container(width: 8, height: 8,
                decoration: BoxDecoration(
                  color: _serverOk ? T.ok : T.err, shape: BoxShape.circle)),
              const SizedBox(width: 8),
              Text(_serverOk ? 'Backend Connected' : 'Backend Offline',
                style: TextStyle(fontSize: 12, fontWeight: FontWeight.w700,
                  color: _serverOk ? T.ok : T.err)),
            ]),
            const SizedBox(height: 10),
            Row(children: [
              Expanded(child: TextField(
                controller: _urlCtrl,
                style: const TextStyle(fontSize: 12, fontFamily: 'monospace', color: T.txt),
                decoration: const InputDecoration(
                  labelText: 'Server URL', hintText: 'http://10.0.2.2:8080'),
              )),
              const SizedBox(width: 10),
              ElevatedButton(
                onPressed: _connect,
                style: ElevatedButton.styleFrom(
                  backgroundColor: T.acc,
                  padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 13)),
                child: const Text('Ping', style: TextStyle(fontSize: 12, color: Colors.white)),
              ),
            ]),
          ]),
        ),

        const SizedBox(height: 12),

        // ── E2E Encryption status card ──────────────────────────
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
          decoration: BoxDecoration(
            color: T.card,
            border: Border.all(
              color: _encReady
                  ? const Color(0xFF00CC88).withOpacity(0.6)
                  : _serverOk
                      ? Colors.orange.withOpacity(0.5)
                      : T.border,
            ),
            borderRadius: BorderRadius.circular(14),
          ),
          child: Row(children: [
            // Icon
            Container(
              width: 42, height: 42,
              decoration: BoxDecoration(
                color: _encReady
                    ? const Color(0xFF00CC88).withOpacity(0.15)
                    : Colors.orange.withOpacity(0.10),
                shape: BoxShape.circle,
              ),
              child: Center(
                child: _encLoading
                    ? const SizedBox(width: 20, height: 20,
                        child: CircularProgressIndicator(
                          color: Color(0xFF00CC88), strokeWidth: 2))
                    : Text(_encReady ? '🔐' : '🔓',
                        style: const TextStyle(fontSize: 20)),
              ),
            ),
            const SizedBox(width: 14),
            Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text(
                _encLoading
                    ? 'Loading RSA public key…'
                    : _encReady
                        ? 'End-to-End Encryption Active'
                        : _serverOk
                            ? 'Encryption unavailable'
                            : 'Connect backend first',
                style: TextStyle(
                  fontSize: 13, fontWeight: FontWeight.w800,
                  color: _encReady
                      ? const Color(0xFF00CC88)
                      : _serverOk ? Colors.orange : T.muted),
              ),
              const SizedBox(height: 3),
              Text(
                _encReady
                    ? 'RSA-2048-OAEP + AES-256-GCM\nCSV encrypted on device before transmission'
                    : _serverOk
                        ? 'Will use plain upload (no encryption)'
                        : 'Emulator → 10.0.2.2:8080\nDevice → your PC\'s LAN IP:8080',
                style: const TextStyle(fontSize: 10, color: T.muted, height: 1.6,
                  fontFamily: 'monospace'),
              ),
            ])),
          ]),
        ),

        const SizedBox(height: 18),

        // ── Drop / pick zone ────────────────────────────────────
        GestureDetector(
          onTap: _serverOk && !_loading ? _pick : null,
          child: Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(vertical: 40, horizontal: 20),
            decoration: BoxDecoration(
              color: T.card,
              border: Border.all(
                color: _serverOk ? T.fl.withOpacity(0.5) : T.border, width: 2),
              borderRadius: BorderRadius.circular(20),
              boxShadow: _serverOk
                  ? [BoxShadow(color: T.fl.withOpacity(0.07), blurRadius: 24)]
                  : null,
            ),
            child: _loading
                ? Column(children: [
                    const CircularProgressIndicator(color: T.fl, strokeWidth: 2.5),
                    const SizedBox(height: 14),
                    Text(
                      _encReady ? '🔐 Encrypting & uploading…' : '📤 Uploading…',
                      style: const TextStyle(color: T.muted, fontSize: 13)),
                  ])
                : Column(children: [
                    Text(_serverOk ? (_encReady ? '🔒' : '📂') : '🔌',
                      style: const TextStyle(fontSize: 46)),
                    const SizedBox(height: 12),
                    Text(
                      _serverOk
                          ? 'Tap to choose CSV file'
                          : 'Connect backend first',
                      style: const TextStyle(fontSize: 17, fontWeight: FontWeight.w800)),
                    const SizedBox(height: 4),
                    Text(
                      _encReady
                          ? '🛡 Encrypted before sending · .csv · max 10 MB'
                          : '.csv only · max 10 MB',
                      style: const TextStyle(fontSize: 11, color: T.muted)),
                    if (_serverOk) ...[
                      const SizedBox(height: 18),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 11),
                        decoration: BoxDecoration(
                          gradient: const LinearGradient(colors: [T.fl, T.acc]),
                          borderRadius: BorderRadius.circular(10)),
                        child: Text(
                          _encReady ? '🔐 Choose & Encrypt File' : 'Choose File',
                          style: const TextStyle(
                            fontWeight: FontWeight.w900, color: Colors.black, fontSize: 13)),
                      ),
                    ],
                  ]),
          ),
        ),

        // ── Error display ───────────────────────────────────────
        if (_err != null) ...[
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: T.err.withOpacity(0.1),
              border: Border.all(color: T.err.withOpacity(0.4)),
              borderRadius: BorderRadius.circular(10)),
            child: Row(children: [
              const Icon(Icons.warning_amber, color: T.err, size: 16),
              const SizedBox(width: 8),
              Expanded(child: Text(_err!,
                style: const TextStyle(color: T.err, fontSize: 12))),
            ]),
          ),
        ],

        const SizedBox(height: 28),

        // ── Feature badges ──────────────────────────────────────
        Row(children: [
          _badge('🏛', 'Central',   T.ct),
          const SizedBox(width: 8),
          _badge('🌐', 'Federated', T.fl),
          const SizedBox(width: 8),
          _badge('🔐', 'E2E Enc',   const Color(0xFF00CC88)),
        ]),
      ]),
    )),
  );

  Widget _badge(String icon, String label, Color c) => Expanded(
    child: Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: c.withOpacity(0.08),
        border: Border.all(color: c.withOpacity(0.3)),
        borderRadius: BorderRadius.circular(12)),
      child: Column(children: [
        Text(icon, style: const TextStyle(fontSize: 20)),
        const SizedBox(height: 4),
        Text(label, style: TextStyle(
          fontSize: 10, fontWeight: FontWeight.w700, color: c)),
      ]),
    ),
  );
}
