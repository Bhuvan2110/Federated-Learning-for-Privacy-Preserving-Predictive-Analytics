import 'package:flutter/material.dart';
import '../theme/theme.dart';
import '../services/auth_service.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});
  @override State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _id    = TextEditingController(text: 'college-admin');
  final _email = TextEditingController(text: 'admin@college.edu');
  String _role = 'admin';
  bool _loading = false;
  String? _err;

  Future<void> _doLogin() async {
    setState(() { _loading = true; _err = null; });
    try {
      await AuthService.instance.login(_id.text, _email.text, _role);
    } catch (e) {
      setState(() => _err = e.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override Widget build(BuildContext ctx) => Scaffold(
    backgroundColor: T.bg,
    body: Center(child: SingleChildScrollView(
      padding: const EdgeInsets.all(32),
      child: Column(mainAxisSize: MainAxisSize.min, children: [
        // Logo
        Container(width: 80, height: 80,
          decoration: BoxDecoration(
            gradient: const LinearGradient(colors: [T.fl, T.acc]),
            borderRadius: BorderRadius.circular(20),
            boxShadow: [BoxShadow(color: T.fl.withOpacity(0.3), blurRadius: 30)]),
          child: const Center(child: Text('⬡', style: TextStyle(fontSize: 40)))),
        const SizedBox(height: 24),
        const Text('TRAINING MODELS', style: TextStyle(fontFamily: 'monospace',
          fontSize: 24, fontWeight: FontWeight.w900, letterSpacing: 2)),
        const Text('Enterprise Federated Learning', style: TextStyle(color: T.muted, fontSize: 12)),
        const SizedBox(height: 48),

        // Inputs
        _field('USER ID', _id, Icons.person_outline),
        const SizedBox(height: 16),
        _field('EMAIL', _email, Icons.mail_outline),
        const SizedBox(height: 16),
        
        // Role Dropdown
        _label('ROLE'),
        Container(padding: const EdgeInsets.symmetric(horizontal: 16),
          decoration: BoxDecoration(color: T.card, borderRadius: BorderRadius.circular(12),
            border: Border.all(color: T.border)),
          child: DropdownButtonHideUnderline(child: DropdownButton<String>(
            value: _role, dropdownColor: T.card, isExpanded: true,
            style: const TextStyle(color: T.txt, fontSize: 14),
            items: ['admin', 'trainer', 'viewer'].map((r) => DropdownMenuItem(
              value: r, child: Text(r.toUpperCase()))).toList(),
            onChanged: (v) => setState(() => _role = v!)))),
        
        const SizedBox(height: 32),
        if (_err != null) Padding(padding: const EdgeInsets.only(bottom: 16),
          child: Text(_err!, style: const TextStyle(color: T.err, fontSize: 12))),

        SizedBox(width: double.infinity, height: 54,
          child: ElevatedButton(
            onPressed: _loading ? null : _doLogin,
            style: ElevatedButton.styleFrom(
              backgroundColor: T.fl, foregroundColor: Colors.black,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12))),
            child: _loading ? const SizedBox(width: 20, height: 20,
              child: CircularProgressIndicator(strokeWidth: 2, color: Colors.black))
              : const Text('AUTHENTICATE', style: TextStyle(fontWeight: FontWeight.w900, letterSpacing: 1)))),
        
        const SizedBox(height: 16),
        TextButton(onPressed: () {}, child: const Text('FORGOT CREDENTIALS?',
          style: TextStyle(color: T.muted, fontSize: 10, letterSpacing: 1))),
      ])),
    ));

  Widget _field(String label, TextEditingController ctrl, IconData icon) => Column(
    crossAxisAlignment: CrossAxisAlignment.start, children: [
      _label(label),
      TextField(controller: ctrl, style: const TextStyle(fontSize: 14),
        decoration: InputDecoration(
          prefixIcon: Icon(icon, size: 18, color: T.muted),
          filled: true, fillColor: T.card,
          enabledBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(12),
            borderSide: const BorderSide(color: T.border)),
          focusedBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(12),
            borderSide: const BorderSide(color: T.fl)))),
    ]);

  Widget _label(String txt) => Padding(padding: const EdgeInsets.only(left: 4, bottom: 6),
    child: Text(txt, style: const TextStyle(color: T.muted, fontSize: 10,
      fontFamily: 'monospace', fontWeight: FontWeight.bold, letterSpacing: 1.5)));
}
