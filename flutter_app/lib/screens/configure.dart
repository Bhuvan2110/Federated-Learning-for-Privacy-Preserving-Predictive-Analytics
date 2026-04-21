import 'package:flutter/material.dart';
import '../models/models.dart';
import '../theme/theme.dart';

class ConfigScreen extends StatefulWidget {
  final CsvData csv;
  final void Function(TrainConfig) onDone;
  const ConfigScreen({super.key, required this.csv, required this.onDone});
  @override State<ConfigScreen> createState() => _State();
}

class _State extends State<ConfigScreen> {
  late int _idx;
  late Map<String,String> _ft;
  late TextEditingController _tgt,_ep,_lr,_rounds,_le,_nc;
  String? _err;

  static const _opts=[
    {'k':'numeric','icon':'🔢','desc':'Continuous float'},
    {'k':'binary', 'icon':'⚡','desc':'0/1, Yes/No'},
    {'k':'text',   'icon':'🔤','desc':'Categorical (hashed)'},
    {'k':'ignore', 'icon':'🚫','desc':'Exclude from model'},
  ];

  @override void initState(){
    super.initState();
    _idx=widget.csv.headers.length-1;
    _tgt=TextEditingController(text:'${_idx+1}');
    _ep=TextEditingController(text:'100'); _lr=TextEditingController(text:'0.1');
    _rounds=TextEditingController(text:'25'); _le=TextEditingController(text:'5');
    _nc=TextEditingController(text:'5');
    _ft={};
    for(final s in widget.csv.stats){
      _ft[s.col]=s.type=='numeric'?'numeric':s.uniqueCount!=null&&s.uniqueCount!<=2?'binary':'text';
    }
  }
  @override void dispose(){ for(final c in [_tgt,_ep,_lr,_rounds,_le,_nc])c.dispose(); super.dispose(); }

  void _onTgt(String v){
    final n=int.tryParse(v);
    if(n!=null&&n>=1&&n<=widget.csv.headers.length){setState((){_idx=n-1;_err=null;});}
    else setState(()=>_err='Must be 1–${widget.csv.headers.length}');
  }

  void _go(){
    final n=int.tryParse(_tgt.text);
    if(n==null||n<1||n>widget.csv.headers.length){setState(()=>_err='Invalid');return;}
    widget.onDone(TrainConfig(targetIdx:n-1,ftypes:Map.from(_ft),
      epochs:(int.tryParse(_ep.text)??100).clamp(1,10000),
      lr:(double.tryParse(_lr.text)??0.1).clamp(0.0001,10.0),
      rounds:(int.tryParse(_rounds.text)??25).clamp(1,1000),
      localEpochs:(int.tryParse(_le.text)??5).clamp(1,100),
      numClients:(int.tryParse(_nc.text)??5).clamp(2,20)));
  }

  @override Widget build(BuildContext ctx){
    final tn=widget.csv.headers[_idx];
    return Scaffold(
      appBar:AppBar(title:const Text('Configure Training')),
      bottomNavigationBar:SafeArea(child:Padding(padding:const EdgeInsets.all(16),
        child:ElevatedButton(onPressed:_err==null?_go:null,
          style:ElevatedButton.styleFrom(backgroundColor:T.fl,minimumSize:const Size.fromHeight(50)),
          child:const Text('▶  Proceed to Training',
            style:TextStyle(fontSize:15,fontWeight:FontWeight.w900,color:Colors.black))))),
      body:SingleChildScrollView(padding:const EdgeInsets.all(20),child:Column(
        crossAxisAlignment:CrossAxisAlignment.start,children:[

        // File info
        Container(padding:const EdgeInsets.all(14),decoration:BoxDecoration(
          color:T.card,border:Border.all(color:T.fl.withOpacity(0.3)),
          borderRadius:BorderRadius.circular(12)),
          child:Row(children:[const Text('📄',style:TextStyle(fontSize:22)),const SizedBox(width:12),
            Expanded(child:Column(crossAxisAlignment:CrossAxisAlignment.start,children:[
              Text(widget.csv.filename,style:const TextStyle(fontWeight:FontWeight.w700,fontSize:13),
                overflow:TextOverflow.ellipsis),
              Text('${widget.csv.totalRows} rows · ${widget.csv.totalCols} cols',
                style:const TextStyle(fontSize:10,color:T.muted,fontFamily:'monospace')),
            ]))])),
        const SizedBox(height:24),

        // Target
        const Text('TARGET COLUMN',style:TextStyle(fontSize:10,color:T.muted,
          fontFamily:'monospace',letterSpacing:1.5)),
        const SizedBox(height:4),
        const Text('Column to predict — enter any number from 1 to ∞',
          style:TextStyle(fontSize:12,color:T.muted)),
        const SizedBox(height:10),
        Row(crossAxisAlignment:CrossAxisAlignment.start,children:[
          SizedBox(width:88,child:TextField(controller:_tgt,
            keyboardType:TextInputType.number,onChanged:_onTgt,
            style:const TextStyle(fontSize:20,fontFamily:'monospace',
              color:T.fl,fontWeight:FontWeight.w900),
            decoration:const InputDecoration(labelText:'Col #',helperText:'1 – ∞'))),
          const SizedBox(width:12),
          Expanded(child:Container(padding:const EdgeInsets.all(12),
            decoration:BoxDecoration(color:T.fl.withOpacity(0.08),
              border:Border.all(color:T.fl.withOpacity(0.4)),
              borderRadius:BorderRadius.circular(10)),
            child:Column(crossAxisAlignment:CrossAxisAlignment.start,children:[
              Text('Column ${_idx+1}',style:const TextStyle(fontSize:9,color:T.muted,fontFamily:'monospace')),
              const SizedBox(height:4),
              Text(tn,style:const TextStyle(fontSize:14,fontWeight:FontWeight.w800,color:T.fl)),
            ]))),
        ]),
        if(_err!=null)Padding(padding:const EdgeInsets.only(top:4),
          child:Text(_err!,style:const TextStyle(color:T.err,fontSize:11))),
        const SizedBox(height:8),
        Wrap(spacing:6,runSpacing:6,children:widget.csv.headers.asMap().entries.map((e){
          final sel=e.key==_idx;
          return GestureDetector(onTap:()=>setState((){_idx=e.key;_tgt.text='${e.key+1}';_err=null;}),
            child:Container(padding:const EdgeInsets.symmetric(horizontal:9,vertical:4),
              decoration:BoxDecoration(
                color:sel?T.fl.withOpacity(0.12):T.card,
                border:Border.all(color:sel?T.fl:T.border),
                borderRadius:BorderRadius.circular(7)),
              child:Text('${e.key+1}. ${e.value}',style:TextStyle(fontSize:10,
                color:sel?T.fl:T.txt,fontWeight:sel?FontWeight.w700:FontWeight.normal))));
        }).toList()),

        const SizedBox(height:28),

        // Feature types
        const Text('FEATURE TYPES',style:TextStyle(fontSize:10,color:T.muted,
          fontFamily:'monospace',letterSpacing:1.5)),
        const SizedBox(height:4),
        const Text('How each column is encoded for the model.',
          style:TextStyle(fontSize:12,color:T.muted)),
        const SizedBox(height:12),
        ...widget.csv.headers.where((h)=>h!=tn).map((col){
          final s=widget.csv.stats.firstWhere((x)=>x.col==col,
            orElse:()=>ColStat(col:col,type:'numeric',count:0));
          final cur=_ft[col]??'numeric';
          return Container(margin:const EdgeInsets.only(bottom:8),padding:const EdgeInsets.all(12),
            decoration:BoxDecoration(color:T.card,border:Border.all(color:T.border),
              borderRadius:BorderRadius.circular(12)),
            child:Column(crossAxisAlignment:CrossAxisAlignment.start,children:[
              Row(children:[
                Expanded(child:Column(crossAxisAlignment:CrossAxisAlignment.start,children:[
                  Text(col,style:const TextStyle(fontWeight:FontWeight.w700,fontSize:13)),
                  Text(s.type=='numeric'
                    ?'min ${s.min?.toStringAsFixed(2)} · max ${s.max?.toStringAsFixed(2)}'
                    :'${s.uniqueCount} unique',
                    style:const TextStyle(fontSize:9,color:T.muted,fontFamily:'monospace')),
                ])),
                Container(padding:const EdgeInsets.symmetric(horizontal:6,vertical:2),
                  decoration:BoxDecoration(color:T.border.withOpacity(0.5),
                    borderRadius:BorderRadius.circular(4)),
                  child:Text('auto:${s.type}',style:const TextStyle(fontSize:8,color:T.muted,fontFamily:'monospace'))),
              ]),
              const SizedBox(height:8),
              Row(children:_opts.map((o){
                final sel=cur==o['k'];final isIgnore=o['k']=='ignore';
                return Expanded(child:Padding(padding:const EdgeInsets.symmetric(horizontal:2),
                  child:GestureDetector(onTap:()=>setState(()=>_ft[col]=o['k']!),
                    child:Container(padding:const EdgeInsets.symmetric(vertical:6),
                      decoration:BoxDecoration(
                        color:sel?(isIgnore?T.err:T.acc).withOpacity(0.18):T.surface,
                        border:Border.all(color:sel?(isIgnore?T.err:T.acc):T.border),
                        borderRadius:BorderRadius.circular(7)),
                      child:Column(children:[
                        Text(o['icon']!,style:const TextStyle(fontSize:13)),
                        Text(o['k']!,style:TextStyle(fontSize:7,
                          color:sel?T.txt:T.muted,fontFamily:'monospace')),
                      ])))));
              }).toList()),
              const SizedBox(height:4),
              Text(_opts.firstWhere((o)=>o['k']==cur)['desc']!,
                style:const TextStyle(fontSize:10,color:T.muted)),
            ]));
        }),

        const SizedBox(height:28),

        // Hyperparams
        const Text('HYPERPARAMETERS',style:TextStyle(fontSize:10,color:T.muted,
          fontFamily:'monospace',letterSpacing:1.5)),
        const SizedBox(height:12),
        _sect('🏛 Central Training',T.ct,[
          _pf(_ep,'Epochs','1–10000',TextInputType.number),
          _pf(_lr,'Learning Rate','0.001–1.0',
            const TextInputType.numberWithOptions(decimal:true))]),
        const SizedBox(height:12),
        _sect('🌐 Federated Learning',T.fl,[
          _pf(_rounds,'Rounds','1–1000',TextInputType.number),
          _pf(_le,'Local Epochs / Round','1–100',TextInputType.number),
          _pf(_nc,'Num Clients','2–20',TextInputType.number),
          _pf(_lr,'Learning Rate','0.001–1.0',
            const TextInputType.numberWithOptions(decimal:true))]),
        const SizedBox(height:40),
      ])));
  }

  Widget _sect(String t,Color c,List<Widget> fs)=>Container(padding:const EdgeInsets.all(14),
    decoration:BoxDecoration(color:T.card,
      border:Border.all(color:c.withOpacity(0.35)),
      borderRadius:BorderRadius.circular(14)),
    child:Column(crossAxisAlignment:CrossAxisAlignment.start,children:[
      Text(t,style:TextStyle(fontWeight:FontWeight.w800,color:c,fontSize:13)),
      const SizedBox(height:12),...fs]));

  Widget _pf(TextEditingController c,String l,String h,TextInputType kt)=>Padding(
    padding:const EdgeInsets.only(bottom:10),
    child:TextField(controller:c,keyboardType:kt,
      style:const TextStyle(fontFamily:'monospace',fontSize:13,color:T.txt),
      decoration:InputDecoration(labelText:l,hintText:h)));
}
