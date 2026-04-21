import 'package:flutter/foundation.dart';
import '../models/models.dart';
import 'api.dart';

class AuthService extends ChangeNotifier {
  static final AuthService instance = AuthService._();
  AuthService._();

  User? _user;
  bool _initialized = false;

  User? get user => _user;
  bool get isAuthenticated => _user != null;
  bool get isInitialized => _initialized;

  Future<void> login(String id, String email, String role) async {
    await Api.login(id, email, role);
    _user = User(id: id, email: email, role: role, attrs: {});
    _initialized = true;
    notifyListeners();
  }

  void logout() {
    _user = null;
    Api.setToken(null);
    notifyListeners();
  }
}
