import 'package:flutter/material.dart';

class T {
  static const bg      = Color(0xFF05050F);
  static const surface = Color(0xFF0B0B1E);
  static const card    = Color(0xFF111128);
  static const border  = Color(0xFF1A1A3E);
  static const border2 = Color(0xFF252550);
  static const fl      = Color(0xFF00F0FF);
  static const ct      = Color(0xFFFF5F1F);
  static const acc     = Color(0xFF8B5CF6);
  static const txt     = Color(0xFFE8E8F4);
  static const muted   = Color(0xFF4A4A7A);
  static const ok      = Color(0xFF00D48A);
  static const err     = Color(0xFFFF4D6D);

  static ThemeData get dark => ThemeData(
    useMaterial3: true,
    brightness: Brightness.dark,
    scaffoldBackgroundColor: bg,
    colorScheme: const ColorScheme.dark(
      primary: fl, secondary: acc, surface: surface, error: err),
    appBarTheme: const AppBarTheme(
      backgroundColor: surface, foregroundColor: txt, elevation: 0,
      titleTextStyle: TextStyle(fontFamily: 'monospace', fontSize: 16,
          fontWeight: FontWeight.w800, color: txt, letterSpacing: 1)),
    cardTheme: CardThemeData(color: card, elevation: 0,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16),
          side: const BorderSide(color: border))),
    elevatedButtonTheme: ElevatedButtonThemeData(style: ElevatedButton.styleFrom(
      backgroundColor: fl, foregroundColor: Colors.black,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
      textStyle: const TextStyle(fontWeight: FontWeight.w800, fontSize: 14))),
    inputDecorationTheme: InputDecorationTheme(
      filled: true, fillColor: surface,
      border: OutlineInputBorder(borderRadius: BorderRadius.circular(10),
          borderSide: const BorderSide(color: border)),
      enabledBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(10),
          borderSide: const BorderSide(color: border)),
      focusedBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(10),
          borderSide: const BorderSide(color: fl, width: 2)),
      labelStyle: const TextStyle(color: muted),
      hintStyle: const TextStyle(color: muted), isDense: true),
    dividerTheme: const DividerThemeData(color: border),
  );
}
