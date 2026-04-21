import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import '../models/models.dart';
import '../theme/theme.dart';

// ── Metric Bar ─────────────────────────────────────────────────────────────────
class MetricBar extends StatelessWidget {
  final String label;
  final double value;
  final Color color;
  const MetricBar({super.key, required this.label, required this.value, required this.color});
  @override Widget build(BuildContext ctx) => Column(
    crossAxisAlignment: CrossAxisAlignment.start,
    children: [
      Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
        Text(label, style: const TextStyle(fontSize:11, color:T.muted)),
        Text('${(value*100).toStringAsFixed(1)}%', style: TextStyle(
          fontSize:11, fontFamily:'monospace', color:color, fontWeight:FontWeight.w700)),
      ]),
      const SizedBox(height:4),
      ClipRRect(borderRadius: BorderRadius.circular(3),
        child: LinearProgressIndicator(value:value, backgroundColor:T.border,
          valueColor: AlwaysStoppedAnimation(color), minHeight:6)),
    ]);
}

// ── Loss Chart ─────────────────────────────────────────────────────────────────
class LossChart extends StatelessWidget {
  final List<double> data;
  final Color color;
  final String title;
  const LossChart({super.key, required this.data, required this.color, this.title='LOSS CURVE'});
  @override Widget build(BuildContext ctx) {
    if(data.isEmpty) return const SizedBox.shrink();
    final spots = data.asMap().entries.map((e)=>FlSpot(e.key.toDouble(),e.value)).toList();
    final mn=data.reduce((a,b)=>a<b?a:b), mx=data.reduce((a,b)=>a>b?a:b);
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
        Text(title, style: const TextStyle(fontSize:9, color:T.muted, fontFamily:'monospace', letterSpacing:1.2)),
        Text('final: ${data.last.toStringAsFixed(4)}',
          style: TextStyle(fontSize:10, color:color, fontFamily:'monospace')),
      ]),
      const SizedBox(height:8),
      SizedBox(height:90, child: LineChart(LineChartData(
        gridData: FlGridData(show:true, drawVerticalLine:false,
          getDrawingHorizontalLine:(_)=>const FlLine(color:T.border, strokeWidth:1)),
        titlesData: FlTitlesData(
          leftTitles: AxisTitles(sideTitles: SideTitles(showTitles:true, reservedSize:36,
            getTitlesWidget:(v,_)=>Text(v.toStringAsFixed(2),
              style: const TextStyle(fontSize:8, color:T.muted)))),
          bottomTitles: AxisTitles(sideTitles: SideTitles(showTitles:true, reservedSize:16,
            getTitlesWidget:(v,_){
              if(v.toInt()%(data.length~/5).clamp(1,999)!=0) return const SizedBox.shrink();
              return Text('${v.toInt()}', style: const TextStyle(fontSize:8, color:T.muted));
            })),
          topTitles: const AxisTitles(sideTitles: SideTitles(showTitles:false)),
          rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles:false))),
        borderData: FlBorderData(show:true, border:Border.all(color:T.border)),
        minY: mn*0.98, maxY: mx*1.02,
        lineBarsData: [LineChartBarData(
          spots: spots, isCurved:true, color:color, barWidth:2,
          dotData: const FlDotData(show:false),
          belowBarData: BarAreaData(show:true, color:color.withOpacity(0.1)))],
      ))),
    ]);
  }
}

// ── Confusion Matrix ───────────────────────────────────────────────────────────
class ConfMatrixWidget extends StatelessWidget {
  final ConfMatrix m;
  final Color color;
  const ConfMatrixWidget({super.key, required this.m, required this.color});
  @override Widget build(BuildContext ctx) => Column(
    crossAxisAlignment: CrossAxisAlignment.start,
    children: [
      Text('CONFUSION MATRIX', style: const TextStyle(
        fontSize:9, color:T.muted, fontFamily:'monospace', letterSpacing:1.5)),
      const SizedBox(height:8),
      Row(children: [
        const SizedBox(width:60),
        Expanded(child:Center(child:Text('Pred +', style:const TextStyle(fontSize:9,color:T.muted)))),
        Expanded(child:Center(child:Text('Pred −', style:const TextStyle(fontSize:9,color:T.muted)))),
      ]),
      const SizedBox(height:4),
      _row('Act +', [_Cell('TP',m.tp,true), _Cell('FN',m.fn,false)]),
      const SizedBox(height:4),
      _row('Act −', [_Cell('FP',m.fp,false), _Cell('TN',m.tn,true)]),
    ]);

  Widget _row(String lbl, List<_Cell> cells) => Row(children:[
    SizedBox(width:60, child:Text(lbl, style:const TextStyle(fontSize:9,color:T.muted))),
    ...cells.map((c)=>Expanded(child:Container(
      margin: const EdgeInsets.symmetric(horizontal:2),
      padding: const EdgeInsets.symmetric(vertical:10),
      decoration: BoxDecoration(
        color: c.hi ? color.withOpacity(0.14) : T.border.withOpacity(0.3),
        border: Border.all(color: c.hi ? color : T.border),
        borderRadius: BorderRadius.circular(8)),
      child: Column(children:[
        Text(c.lbl, style:const TextStyle(fontSize:8,color:T.muted,fontFamily:'monospace')),
        const SizedBox(height:3),
        Text('${c.val}', style:TextStyle(fontSize:18,fontWeight:FontWeight.w800,
          color:c.hi?color:T.muted)),
      ])))),
  ]);
}
class _Cell { final String lbl; final int val; final bool hi;
  const _Cell(this.lbl,this.val,this.hi); }

// ── Stat Card ──────────────────────────────────────────────────────────────────
class StatCard extends StatelessWidget {
  final String label, value;
  final Color? color;
  const StatCard({super.key, required this.label, required this.value, this.color});
  @override Widget build(BuildContext ctx) => Container(
    padding: const EdgeInsets.all(14),
    decoration: BoxDecoration(color:T.card,
      border: Border.all(color:color?.withOpacity(0.4)??T.border),
      borderRadius: BorderRadius.circular(12)),
    child: Column(crossAxisAlignment:CrossAxisAlignment.start, children:[
      Text(label.toUpperCase(), style:const TextStyle(
        fontSize:8, color:T.muted, fontFamily:'monospace', letterSpacing:1.5)),
      const SizedBox(height:6),
      Text(value, style:TextStyle(fontSize:20, fontWeight:FontWeight.w800,
        color:color??T.txt)),
    ]));
}
