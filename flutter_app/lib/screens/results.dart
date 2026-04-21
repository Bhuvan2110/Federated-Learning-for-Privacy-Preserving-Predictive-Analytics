import 'package:flutter/material.dart';
import '../models/models.dart';
import '../theme/theme.dart';
import '../widgets/widgets.dart';

class ResultsScreen extends StatefulWidget {
  final TrainResult? central, fl;
  final VoidCallback onTrainMore;
  const ResultsScreen({super.key,required this.central,required this.fl,required this.onTrainMore});
  @override State<ResultsScreen> createState() => _State();
}

class _State extends State<ResultsScreen> with SingleTickerProviderStateMixin {
  late TabController _tab;
  String _pct(double v)=>'${(v*100).toStringAsFixed(1)}%';
  String _msf(int ms)=>ms<1000?'${ms}ms':'${(ms/1000).toStringAsFixed(2)}s';

  @override void initState(){
    super.initState();
    int n=0;
    if(widget.central!=null) n++;
    if(widget.fl!=null) n++;
    if(widget.central!=null&&widget.fl!=null) n++;
    _tab=TabController(length:n.clamp(1,3),vsync:this);
  }
  @override void dispose(){ _tab.dispose(); super.dispose(); }

  List<(String,Color,Widget)> get _tabs {
    final t=<(String,Color,Widget)>[];
    if(widget.central!=null) t.add(('🏛 Central',T.ct,_modelCard(widget.central!,'central')));
    if(widget.fl!=null)      t.add(('🌐 Federated',T.fl,_modelCard(widget.fl!,'fl')));
    if(widget.central!=null&&widget.fl!=null) t.add(('⚔️ Compare',T.acc,_compareView()));
    return t;
  }

  @override Widget build(BuildContext ctx){
    if(widget.central==null&&widget.fl==null) return Scaffold(
      appBar:AppBar(title:const Text('Results')),
      body:Center(child:Column(mainAxisAlignment:MainAxisAlignment.center,children:[
        const Text('🔬',style:TextStyle(fontSize:48)),
        const SizedBox(height:16),
        const Text('No results yet.',style:TextStyle(color:T.muted)),
        const SizedBox(height:16),
        ElevatedButton(onPressed:widget.onTrainMore,child:const Text('Go to Train')),
      ])));
    final tabs=_tabs;
    return Scaffold(
      appBar:AppBar(title:const Text('Results & Comparison'),
        actions:[TextButton(onPressed:widget.onTrainMore,
          child:const Text('+ Train More',style:TextStyle(color:T.fl,fontSize:12)))],
        bottom:TabBar(controller:_tab,
          tabs:tabs.map((t)=>Tab(text:t.$1)).toList(),
          labelColor:T.txt,unselectedLabelColor:T.muted,
          indicatorColor:T.fl,
          labelStyle:const TextStyle(fontSize:11,fontWeight:FontWeight.w700))),
      body:TabBarView(controller:_tab,
        children:tabs.map((t)=>SingleChildScrollView(
          padding:const EdgeInsets.all(16),child:t.$3)).toList()));
  }

  Widget _modelCard(TrainResult r, String which){
    final col=which=='central'?T.ct:T.fl;
    final m=r.testMetrics;
    return Container(decoration:BoxDecoration(color:T.card,
      border:Border.all(color:col.withOpacity(0.35),width:1.5),
      borderRadius:BorderRadius.circular(20),
      boxShadow:[BoxShadow(color:col.withOpacity(0.07),blurRadius:24)]),
      child:Padding(padding:const EdgeInsets.all(22),child:Column(
        crossAxisAlignment:CrossAxisAlignment.start,children:[
        // header
        Row(children:[
          Text(which=='central'?'🏛':'🌐',style:const TextStyle(fontSize:28)),
          const SizedBox(width:12),
          Expanded(child:Column(crossAxisAlignment:CrossAxisAlignment.start,children:[
            Text('${r.model} Model',style:TextStyle(fontSize:17,fontWeight:FontWeight.w800,color:col)),
            Text('${r.trainSamples} train · ${r.testSamples} test · ${_msf(r.trainingTimeMs)}',
              style:const TextStyle(fontSize:10,color:T.muted,fontFamily:'monospace')),
          ])),
          Container(padding:const EdgeInsets.symmetric(horizontal:10,vertical:4),
            decoration:BoxDecoration(color:col.withOpacity(0.15),
              border:Border.all(color:col),borderRadius:BorderRadius.circular(6)),
            child:Text('✓ done',style:TextStyle(fontSize:9,color:col,fontFamily:'monospace'))),
        ]),
        const SizedBox(height:18),
        // accuracy
        Center(child:Column(children:[
          Text(_pct(m.accuracy),style:TextStyle(
            fontSize:60,fontWeight:FontWeight.w900,color:col,height:1.0)),
          const SizedBox(height:4),
          const Text('Test Accuracy',style:TextStyle(fontSize:11,color:T.muted)),
          const SizedBox(height:18),
        ])),
        // bars
        ...['Accuracy','Precision','Recall','F1'].asMap().entries.map((e){
          final vals=[m.accuracy,m.precision,m.recall,m.f1];
          return Padding(padding:const EdgeInsets.only(bottom:10),
            child:MetricBar(label:e.value,value:vals[e.key],color:col));
        }),
        const SizedBox(height:16),
        // confusion matrix
        ConfMatrixWidget(m:m.conf,color:col),
        const SizedBox(height:18),
        // loss chart
        LossChart(data:r.lossHistory,color:col,
          title:which=='central'
            ?'TRAINING LOSS (${r.epochs} epochs)'
            :'FEDERATED LOSS (${r.rounds} rounds)'),
        const SizedBox(height:14),
        // config badge
        Container(width:double.infinity,padding:const EdgeInsets.all(12),
          decoration:BoxDecoration(color:T.surface,borderRadius:BorderRadius.circular(10)),
          child:Text(which=='central'
            ?'🏛 Full dataset · Logistic Regression · ${r.epochs} epochs · lr=${r.lr}'
            :'🌐 ${r.numClients} clients · ${r.localEpochs} local epochs/round · FedAvg · lr=${r.lr}',
            style:const TextStyle(fontSize:10,color:T.muted,fontFamily:'monospace',height:1.6))),
      ])));
  }

  Widget _compareView(){
    final c=widget.central!.testMetrics, f=widget.fl!.testMetrics;
    final flWins=f.accuracy>=c.accuracy;
    final delta=(f.accuracy-c.accuracy).abs();
    final rows=[
      ('Accuracy', c.accuracy, f.accuracy, true, _pct),
      ('Precision',c.precision,f.precision,true, _pct),
      ('Recall',   c.recall,   f.recall,   true, _pct),
      ('F1 Score', c.f1,       f.f1,       true, _pct),
      ('Train Time',widget.central!.trainingTimeMs.toDouble(),
        widget.fl!.trainingTimeMs.toDouble(),false,(v)=>_msf(v.toInt())),
      ('Final Loss',widget.central!.finalLoss,widget.fl!.finalLoss,
        false,(v)=>(v as double).toStringAsFixed(4)),
    ];
    return Container(decoration:BoxDecoration(color:T.card,
      border:Border.all(color:T.border2),borderRadius:BorderRadius.circular(20)),
      padding:const EdgeInsets.all(22),
      child:Column(crossAxisAlignment:CrossAxisAlignment.start,children:[
      // title
      Row(children:[
        const Expanded(child:Text('⚔️  Comparison',
          style:TextStyle(fontSize:20,fontWeight:FontWeight.w800))),
        Container(padding:const EdgeInsets.symmetric(horizontal:10,vertical:5),
          decoration:BoxDecoration(
            color:(flWins?T.fl:T.ct).withOpacity(0.14),
            border:Border.all(color:flWins?T.fl:T.ct),
            borderRadius:BorderRadius.circular(7)),
          child:Text('🏆 ${flWins?"FL":"Central"} wins',
            style:TextStyle(fontSize:11,fontWeight:FontWeight.w800,color:flWins?T.fl:T.ct))),
      ]),
      const SizedBox(height:16),
      // column headers
      Row(children:[
        const SizedBox(width:96),
        Expanded(child:Center(child:Text('🏛 CENTRAL',
          style:TextStyle(fontSize:9,fontWeight:FontWeight.w800,
            color:T.ct,fontFamily:'monospace')))),
        Expanded(child:Center(child:Text('🌐 FEDERATED',
          style:TextStyle(fontSize:9,fontWeight:FontWeight.w800,
            color:T.fl,fontFamily:'monospace')))),
      ]),
      const SizedBox(height:8),
      // rows
      ...rows.map((row){
        final lbl=row.$1;final cv=row.$2;final fv=row.$3;
        final hi=row.$4;
        final fmt=row.$5;
        final cw=hi?cv>=fv:cv<=fv; final fw=hi?fv>=cv:fv<=cv;
        return Padding(padding:const EdgeInsets.only(bottom:7),
          child:Row(children:[
            SizedBox(width:96,child:Text(lbl,style:const TextStyle(
              fontSize:10,color:T.muted,fontFamily:'monospace'))),
            Expanded(child:_cmpCell(fmt(cv),cw,T.ct)),
            const SizedBox(width:6),
            Expanded(child:_cmpCell(fmt(fv),fw,T.fl)),
          ]));
      }),
      // overlap
      const SizedBox(height:14),
      const Divider(color:T.border),
      const SizedBox(height:12),
      const Text('ACCURACY OVERLAP',style:TextStyle(fontSize:9,color:T.muted,
        fontFamily:'monospace',letterSpacing:1.5)),
      const SizedBox(height:8),
      Row(children:[
        Text(_pct(c.accuracy),style:const TextStyle(fontSize:10,color:T.ct,fontFamily:'monospace')),
        const SizedBox(width:8),
        Expanded(child:Stack(children:[
          Container(height:12,decoration:BoxDecoration(
            color:T.border,borderRadius:BorderRadius.circular(6))),
          FractionallySizedBox(widthFactor:c.accuracy,child:Container(height:12,
            decoration:BoxDecoration(color:T.ct.withOpacity(0.7),
              borderRadius:BorderRadius.circular(6)))),
          FractionallySizedBox(widthFactor:f.accuracy,child:Container(height:12,
            decoration:BoxDecoration(color:T.fl.withOpacity(0.5),
              borderRadius:BorderRadius.circular(6)))),
        ])),
        const SizedBox(width:8),
        Text(_pct(f.accuracy),style:const TextStyle(fontSize:10,color:T.fl,
          fontFamily:'monospace')),
      ]),
      const SizedBox(height:4),
      Row(mainAxisAlignment:MainAxisAlignment.spaceBetween,children:[
        const Text('Central',style:TextStyle(fontSize:9,color:T.ct)),
        Text('Δ ${_pct(delta)}',style:const TextStyle(fontSize:9,color:T.muted)),
        const Text('Federated',style:TextStyle(fontSize:9,color:T.fl)),
      ]),
      // summary
      const SizedBox(height:14),
      Container(padding:const EdgeInsets.all(14),
        decoration:BoxDecoration(color:T.surface,borderRadius:BorderRadius.circular(12)),
        child:RichText(text:TextSpan(style:const TextStyle(fontSize:12,
          color:T.muted,height:1.75),children:[
          const TextSpan(text:'Summary: ',
            style:TextStyle(color:T.txt,fontWeight:FontWeight.w700)),
          TextSpan(text:'Central → ',
            style:const TextStyle(color:T.ct,fontWeight:FontWeight.w700)),
          TextSpan(text:'${_pct(c.accuracy)} acc in ${_msf(widget.central!.trainingTimeMs)}.  '),
          TextSpan(text:'Federated → ',
            style:const TextStyle(color:T.fl,fontWeight:FontWeight.w700)),
          TextSpan(text:'${_pct(f.accuracy)} acc in ${_msf(widget.fl!.trainingTimeMs)}.  '),
          TextSpan(text:flWins
            ?'🌐 Federated matched competitive accuracy with decentralised data.'
            :'🏛 Central had a slight edge — typical on small/clean datasets.'),
        ]))),
    ]));
  }

  Widget _cmpCell(String v,bool win,Color c)=>Container(
    padding:const EdgeInsets.symmetric(vertical:8),
    decoration:BoxDecoration(
      color:win?c.withOpacity(0.12):T.surface,
      border:Border.all(color:win?c:T.border),
      borderRadius:BorderRadius.circular(8)),
    child:Row(mainAxisAlignment:MainAxisAlignment.center,children:[
      Text(v,style:TextStyle(fontSize:12,fontWeight:FontWeight.w700,
        color:win?c:T.txt,fontFamily:'monospace')),
      if(win)Text(' ◀',style:TextStyle(fontSize:9,color:c)),
    ]));
}
