import 'package:flutter/material.dart';
import '../models/models.dart';
import '../services/api.dart';
import '../theme/theme.dart';

class TrainScreen extends StatefulWidget {
  final CsvData csv;
  final TrainConfig cfg;
  final void Function(TrainResult? c, TrainResult? f) onDone;
  const TrainScreen({super.key,required this.csv,required this.cfg,required this.onDone});
  @override State<TrainScreen> createState() => _State();
}

class _State extends State<TrainScreen> {
  TrainResult? _c, _f;
  bool _cl=false, _fl=false;
  String? _ce, _fe;
  String _pct(double v)=>'${(v*100).toStringAsFixed(1)}%';
  String _msf(int ms)=>ms<1000?'${ms}ms':'${(ms/1000).toStringAsFixed(2)}s';

  Future<void> _trainC() async {
    setState(()=>_cl=true);
    try{
      final r=await Api.central(rows:widget.csv.rows,headers:widget.csv.headers,
        targetIdx:widget.cfg.targetIdx,ftypes:widget.cfg.ftypes,
        epochs:widget.cfg.epochs,lr:widget.cfg.lr,algo:widget.cfg.algo);
      setState(()=>_c=r);
    }catch(e){setState(()=>_ce=e.toString().replaceFirst('Exception: ',''));}
    finally{if(mounted)setState(()=>_cl=false);}
  }

  Future<void> _trainF() async {
    setState(()=>_fl=true);
    try{
      final r=await Api.federated(rows:widget.csv.rows,headers:widget.csv.headers,
        targetIdx:widget.cfg.targetIdx,ftypes:widget.cfg.ftypes,
        rounds:widget.cfg.rounds,localEpochs:widget.cfg.localEpochs,
        lr:widget.cfg.lr,numClients:widget.cfg.numClients,algo:widget.cfg.algo);
      setState(()=>_f=r);
    }catch(e){setState(()=>_fe=e.toString().replaceFirst('Exception: ',''));}
    finally{if(mounted)setState(()=>_fl=false);}
  }

  @override Widget build(BuildContext ctx)=>Scaffold(
    appBar:AppBar(title:const Text('Train Models')),
    body:SingleChildScrollView(padding:const EdgeInsets.all(20),child:Column(
      crossAxisAlignment:CrossAxisAlignment.start,children:[

      // Summary
      Container(padding:const EdgeInsets.all(14),decoration:BoxDecoration(
        color:T.card,border:Border.all(color:T.border2),borderRadius:BorderRadius.circular(12)),
        child:Wrap(spacing:20,runSpacing:8,children:[
          _kv('Target',widget.csv.headers[widget.cfg.targetIdx]),
          _kv('Samples','${widget.csv.totalRows}'),
          _kv('Features','${widget.csv.headers.length-1}'),
          _kv('Active','${widget.cfg.ftypes.values.where((v)=>v!="ignore").length} cols'),
        ])),
      const SizedBox(height:22),

      // Central card
      _card(icon:'🏛',title:'Central Training',color:T.ct,
        sub:'${widget.cfg.epochs} epochs · lr=${widget.cfg.lr} · 80/20 split',
        loading:_cl,result:_c,err:_ce,
        onTrain:_cl?null:_trainC),
      const SizedBox(height:16),

      // FL card
      _card(icon:'🌐',title:'Federated Learning',color:T.fl,
        sub:'${widget.cfg.rounds} rounds · ${widget.cfg.numClients} clients · FedAvg',
        loading:_fl,result:_f,err:_fe,
        onTrain:_fl?null:_trainF),
      const SizedBox(height:20),

      // Both button
      SizedBox(width:double.infinity,child:ElevatedButton(
        onPressed:(_cl||_fl)?null:(){_trainC();_trainF();},
        style:ElevatedButton.styleFrom(
          backgroundColor:(_cl||_fl)?T.border:T.acc,minimumSize:const Size.fromHeight(50)),
        child:const Text('⚡  Train Both Simultaneously',
          style:TextStyle(fontSize:14,fontWeight:FontWeight.w900,color:Colors.black)))),

      if(_c!=null||_f!=null)...[const SizedBox(height:14),
        SizedBox(width:double.infinity,child:ElevatedButton(
          onPressed:()=>widget.onDone(_c,_f),
          style:ElevatedButton.styleFrom(
            backgroundColor:T.ok,minimumSize:const Size.fromHeight(50)),
          child:const Text('📊  View Results & Comparison',
            style:TextStyle(fontSize:14,fontWeight:FontWeight.w900,color:Colors.black))))],
    ])));

  Widget _kv(String k,String v)=>Column(crossAxisAlignment:CrossAxisAlignment.start,children:[
    Text(k,style:const TextStyle(fontSize:8,color:T.muted,fontFamily:'monospace')),
    Text(v,style:const TextStyle(fontSize:13,fontWeight:FontWeight.w700,color:T.fl)),
  ]);

  Widget _card({required String icon,required String title,required Color color,
    required String sub,required bool loading,required TrainResult? result,
    required String? err,required VoidCallback? onTrain})=>Container(
    decoration:BoxDecoration(color:T.card,
      border:Border.all(color:result!=null?color.withOpacity(0.5):color.withOpacity(0.25),
        width:result!=null?2:1),
      borderRadius:BorderRadius.circular(18)),
    padding:const EdgeInsets.all(20),
    child:Column(crossAxisAlignment:CrossAxisAlignment.start,children:[
      Row(children:[
        Text(icon,style:const TextStyle(fontSize:26)),
        const SizedBox(width:12),
        Expanded(child:Column(crossAxisAlignment:CrossAxisAlignment.start,children:[
          Text(title,style:TextStyle(fontSize:16,fontWeight:FontWeight.w800,color:color)),
          Text(sub,style:const TextStyle(fontSize:10,color:T.muted,fontFamily:'monospace')),
        ])),
        if(result!=null)Container(padding:const EdgeInsets.symmetric(horizontal:9,vertical:4),
          decoration:BoxDecoration(color:color.withOpacity(0.15),
            border:Border.all(color:color),borderRadius:BorderRadius.circular(6)),
          child:Text('✓ done',style:TextStyle(fontSize:9,color:color,fontFamily:'monospace'))),
      ]),
      if(loading)...[const SizedBox(height:12),
        LinearProgressIndicator(backgroundColor:T.border,
          valueColor:AlwaysStoppedAnimation(color),minHeight:4)],
      if(result!=null)...[const SizedBox(height:12),
        Row(children:[
          _mini('Accuracy',_pct(result.testMetrics.accuracy),color),
          const SizedBox(width:8),
          _mini('F1',_pct(result.testMetrics.f1),color),
          const SizedBox(width:8),
          _mini('Time',_msf(result.trainingTimeMs),T.muted),
        ]),
        const SizedBox(height:10),
        // mini confusion
        Row(children:[
          const Text('CM:',style:TextStyle(fontSize:9,color:T.muted,fontFamily:'monospace')),
          const SizedBox(width:8),
          ...[['TP',result.testMetrics.conf.tp,true],
              ['FP',result.testMetrics.conf.fp,false],
              ['FN',result.testMetrics.conf.fn,false],
              ['TN',result.testMetrics.conf.tn,true]].map((x){
            return Container(margin:const EdgeInsets.only(right:5),
              padding:const EdgeInsets.symmetric(horizontal:7,vertical:3),
              decoration:BoxDecoration(
                color:(x[2] as bool)?color.withOpacity(0.14):T.surface,
                border:Border.all(color:(x[2] as bool)?color:T.border),
                borderRadius:BorderRadius.circular(5)),
              child:Text('${x[0]}:${x[1]}',style:TextStyle(fontSize:9,
                color:(x[2] as bool)?color:T.muted,
                fontFamily:'monospace',
                fontWeight:(x[2] as bool)?FontWeight.w700:FontWeight.normal)));
          }),
        ]),
      ],
      if(err!=null)...[const SizedBox(height:10),
        Container(padding:const EdgeInsets.all(10),
          decoration:BoxDecoration(color:T.err.withOpacity(0.1),
            border:Border.all(color:T.err.withOpacity(0.4)),
            borderRadius:BorderRadius.circular(8)),
          child:Row(children:[const Icon(Icons.error_outline,color:T.err,size:14),
            const SizedBox(width:6),
            Expanded(child:Text(err,style:const TextStyle(color:T.err,fontSize:11)))]))],
      const SizedBox(height:14),
      SizedBox(width:double.infinity,child:ElevatedButton(
        onPressed:loading?null:onTrain,
        style:ElevatedButton.styleFrom(
          backgroundColor:loading?T.border:color,minimumSize:const Size.fromHeight(44)),
        child:Text(loading?'⏳ Training…':result!=null?'↻ Retrain':'▶  Train $icon Model',
          style:TextStyle(fontWeight:FontWeight.w900,fontSize:13,
            color:loading?T.muted:Colors.black)))),
    ]));

  Widget _mini(String k,String v,Color c)=>Expanded(child:Container(
    padding:const EdgeInsets.symmetric(vertical:7),
    decoration:BoxDecoration(color:c.withOpacity(0.08),
      border:Border.all(color:c.withOpacity(0.3)),
      borderRadius:BorderRadius.circular(8)),
    child:Column(children:[
      Text(k,style:const TextStyle(fontSize:8,color:T.muted,fontFamily:'monospace')),
      const SizedBox(height:2),
      Text(v,style:TextStyle(fontSize:13,fontWeight:FontWeight.w800,
        color:c,fontFamily:'monospace')),
    ])));
}
