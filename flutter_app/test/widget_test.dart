import 'package:flutter_test/flutter_test.dart';
import 'package:training_models/main.dart';

void main() {
  testWidgets('App smoke test', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const App());

    // Verify that the login screen is shown.
    expect(find.text('TRAINING MODELS'), findsOneWidget);
    expect(find.text('AUTHENTICATE'), findsOneWidget);
  });
}
