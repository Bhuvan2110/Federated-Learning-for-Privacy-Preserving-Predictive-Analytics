/// encryption_service.dart
/// ═══════════════════════════════════════════════════════════════
/// End-to-End Encryption for CSV uploads
/// Protocol:
///   1. Fetch server's RSA-2048 public key from /api/pubkey
///   2. Generate random AES-256 key + 12-byte IV
///   3. Encrypt CSV bytes with AES-256-GCM
///   4. Wrap AES key with RSA-OAEP-SHA256
///   5. POST { encryptedKey, iv, encryptedData, filename } to
///      /api/upload/encrypted  (all values base64-encoded)
/// ═══════════════════════════════════════════════════════════════
library;

import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';

import 'package:pointycastle/export.dart';
import 'package:pointycastle/asn1.dart';

class EncryptionService {
  // ── Singleton ────────────────────────────────────────────────
  EncryptionService._();
  static final EncryptionService instance = EncryptionService._();

  RSAPublicKey? _serverPublicKey;
  bool get isReady => _serverPublicKey != null;

  // ── Secure random source ─────────────────────────────────────
  static final _rng = FortunaRandom()
    ..seed(KeyParameter(
      Uint8List.fromList(
        List.generate(32, (_) => Random.secure().nextInt(256)),
      ),
    ));

  // ─────────────────────────────────────────────────────────────
  /// Parse the PEM public key string received from /api/pubkey
  /// and import it as an RSAPublicKey.
  // ─────────────────────────────────────────────────────────────
  void loadPublicKeyFromPem(String pem) {
    // Strip PEM headers and decode base64
    final b64 = pem
        .replaceAll('-----BEGIN PUBLIC KEY-----', '')
        .replaceAll('-----END PUBLIC KEY-----', '')
        .replaceAll(RegExp(r'\s+'), '');

    final derBytes = base64.decode(b64);

    // Parse SubjectPublicKeyInfo DER → RSAPublicKey
    final asn1  = ASN1Parser(Uint8List.fromList(derBytes));
    final seq1  = asn1.nextObject() as ASN1Sequence;        // SPKI wrapper
    final seq2  = ASN1Parser(
      (seq1.elements![1] as ASN1BitString).valueBytes!,
    ).nextObject() as ASN1Sequence;                          // RSAPublicKey

    final modulus  = (seq2.elements![0] as ASN1Integer).integer!;
    final exponent = (seq2.elements![1] as ASN1Integer).integer!;

    _serverPublicKey = RSAPublicKey(modulus, exponent);
  }

  // ─────────────────────────────────────────────────────────────
  /// Encrypt [plaintext] bytes.
  /// Returns a map with three base64-encoded fields:
  ///   encryptedKey  — RSA-OAEP-SHA256 wrapped AES-256 key
  ///   iv            — 12-byte GCM nonce
  ///   encryptedData — AES-256-GCM ciphertext + 16-byte GCM tag
  // ─────────────────────────────────────────────────────────────
  Map<String, String> encrypt(Uint8List plaintext) {
    if (_serverPublicKey == null) {
      throw StateError('Server public key not loaded yet.');
    }

    // 1. Generate random 256-bit AES key
    final aesKey = Uint8List(32);
    for (var i = 0; i < 32; i++) {
      aesKey[i] = _rng.nextUint8();
    }

    // 2. Generate random 12-byte IV (GCM nonce)
    final iv = Uint8List(12);
    for (var i = 0; i < 12; i++) {
      iv[i] = _rng.nextUint8();
    }

    // 3. Encrypt CSV bytes with AES-256-GCM
    //    pointycastle appends the 16-byte tag to the ciphertext automatically
    final gcmCipher = GCMBlockCipher(AESEngine())
      ..init(
        true,   // encrypt
        AEADParameters(KeyParameter(aesKey), 128, iv, Uint8List(0)),
      );

    final ciphertext = Uint8List(gcmCipher.getOutputSize(plaintext.length));
    var offset = 0;
    offset += gcmCipher.processBytes(plaintext, 0, plaintext.length, ciphertext, offset);
    gcmCipher.doFinal(ciphertext, offset);

    // 4. Wrap AES key with RSA-OAEP-SHA256
    final rsaCipher = OAEPEncoding.withSHA256(RSAEngine())
      ..init(
        true,   // encrypt
        PublicKeyParameter<RSAPublicKey>(_serverPublicKey!),
      );

    final wrappedKey = rsaCipher.process(aesKey);

    return {
      'encryptedKey':  base64.encode(wrappedKey),
      'iv':            base64.encode(iv),
      'encryptedData': base64.encode(ciphertext),
    };
  }
}
