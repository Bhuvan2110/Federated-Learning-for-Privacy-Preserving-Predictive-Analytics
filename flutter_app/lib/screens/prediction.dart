import 'package:flutter/material.dart';
import '../models/models.dart';
import '../services/api.dart';
import '../theme/theme.dart';

class PredictionScreen extends StatefulWidget {
  const PredictionScreen({super.key});
  @override State<PredictionScreen> createState() => _State();
}

class _State extends State<PredictionScreen> {
  List<Experiment>? _experiments;
  Experiment? _selected;
  final Map<String, TextEditingController> _controllers = {};
  bool _loadingExps = true;
  bool _predicting = false;
  PredictionResult? _result;
  String? _error;

  @override
  void initState() {
    super.initState();
    _loadExperiments();
  }

  Future<void> _loadExperiments() async {
    try {
      final exps = await Api.getExperiments();
      setState(() {
        _experiments = exps.where((e) => e.status == 'completed').toList();
        _loadingExps = false;
      });
    } catch (e) {
      setState(() {
        _error = 'Failed to load experiments: $e';
        _loadingExps = false;
      });
    }
  }

  void _onSelect(Experiment? e) {
    setState(() {
      _selected = e;
      _result = null;
      _controllers.clear();
      if (e?.featureCols != null) {
        for (var col in e!.featureCols!) {
          _controllers[col] = TextEditingController();
        }
      }
    });
  }

  Future<void> _predict() async {
    if (_selected == null) return;
    setState(() {
      _predicting = true;
      _error = null;
    });

    try {
      final inputs = <String, dynamic>{};
      for (var col in _selected!.featureCols!) {
        final val = _controllers[col]!.text;
        if (val.isEmpty) throw 'Please fill all fields';
        inputs[col] = val;
      }

      final res = await Api.predict(_selected!.id, inputs);
      setState(() => _result = res);
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      setState(() => _predicting = false);
    }
  }

  @override
  Widget build(BuildContext ctx) => Scaffold(
    appBar: AppBar(title: const Text('Target Value Prediction')),
    body: _loadingExps 
      ? const Center(child: CircularProgressIndicator())
      : SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _sectionTitle('1. Select Trained Model'),
              const SizedBox(height: 12),
              _buildDropdown(),
              const SizedBox(height: 32),
              if (_selected != null) ...[
                _sectionTitle('2. Enter Input Values'),
                const SizedBox(height: 16),
                _buildInputFields(),
                const SizedBox(height: 32),
                _buildPredictButton(),
                const SizedBox(height: 32),
                if (_result != null) _buildResultCard(),
              ],
              if (_error != null) _buildError(),
            ],
          ),
        ),
  );

  Widget _sectionTitle(String t) => Text(t.toUpperCase(),
    style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w900, color: T.muted, letterSpacing: 1.2));

  Widget _buildDropdown() => Container(
    padding: const EdgeInsets.symmetric(horizontal: 16),
    decoration: BoxDecoration(
      color: T.card,
      borderRadius: BorderRadius.circular(12),
      border: Border.all(color: T.border2),
    ),
    child: DropdownButtonHideUnderline(
      child: DropdownButton<Experiment>(
        value: _selected,
        isExpanded: true,
        hint: const Text('Choose an experiment...', style: TextStyle(color: T.muted, fontSize: 14)),
        dropdownColor: T.surface,
        items: _experiments?.map((e) => DropdownMenuItem(
          value: e,
          child: Text('${e.name} (${e.type})', style: const TextStyle(fontSize: 14)),
        )).toList(),
        onChanged: _onSelect,
      ),
    ),
  );

  Widget _buildInputFields() => GridView.builder(
    shrinkWrap: true,
    physics: const NeverScrollableScrollPhysics(),
    gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
      crossAxisCount: 2,
      crossAxisSpacing: 16,
      mainAxisSpacing: 16,
      childAspectRatio: 2.2,
    ),
    itemCount: _selected!.featureCols!.length,
    itemBuilder: (c, i) {
      final col = _selected!.featureCols![i];
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(col, style: const TextStyle(fontSize: 10, color: T.muted), maxLines: 1, overflow: TextOverflow.ellipsis),
          const SizedBox(height: 6),
          Expanded(child: TextField(
            controller: _controllers[col],
            keyboardType: TextInputType.text,
            style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w600),
            decoration: InputDecoration(
              filled: true,
              fillColor: T.card,
              contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              border: OutlineInputBorder(borderRadius: BorderRadius.circular(8), borderSide: BorderSide.none),
              enabledBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(8), borderSide: BorderSide(color: T.border2)),
              focusedBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(8), borderSide: const BorderSide(color: T.acc)),
            ),
          )),
        ],
      );
    },
  );

  Widget _buildPredictButton() => SizedBox(
    width: double.infinity,
    child: ElevatedButton(
      onPressed: _predicting ? null : _predict,
      style: ElevatedButton.styleFrom(
        backgroundColor: T.acc,
        padding: const EdgeInsets.symmetric(vertical: 16),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      ),
      child: _predicting 
        ? const SizedBox(height: 20, width: 20, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.black))
        : const Text('🎯  Generate Prediction', style: TextStyle(color: Colors.black, fontWeight: FontWeight.w900)),
    ),
  );

  Widget _buildResultCard() {
    final isClassification = _result!.label != null;
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        gradient: LinearGradient(colors: [T.acc.withOpacity(0.1), T.fl.withOpacity(0.05)]),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: T.acc.withOpacity(0.3)),
      ),
      child: Column(
        children: [
          const Text('PREDICTED OUTPUT', style: TextStyle(fontSize: 10, fontWeight: FontWeight.w900, color: T.acc, letterSpacing: 2)),
          const SizedBox(height: 16),
          Text(
            isClassification ? _result!.label! : _result!.prediction.toStringAsFixed(2),
            style: const TextStyle(fontSize: 42, fontWeight: FontWeight.w900, color: Colors.white),
          ),
          if (isClassification) ...[
            const SizedBox(height: 8),
            Text('Confidence: ${(_result!.prediction > 0.5 ? _result!.prediction * 100 : (1 - _result!.prediction) * 100).toStringAsFixed(1)}%',
              style: const TextStyle(fontSize: 12, color: T.muted, fontFamily: 'monospace')),
          ],
        ],
      ),
    );
  }

  Widget _buildError() => Container(
    margin: const EdgeInsets.only(top: 20),
    padding: const EdgeInsets.all(12),
    decoration: BoxDecoration(color: T.err.withOpacity(0.1), borderRadius: BorderRadius.circular(8)),
    child: Row(children: [
      const Icon(Icons.error_outline, color: T.err, size: 16),
      const SizedBox(width: 12),
      Expanded(child: Text(_error!, style: const TextStyle(color: T.err, fontSize: 13))),
    ]),
  );
}
