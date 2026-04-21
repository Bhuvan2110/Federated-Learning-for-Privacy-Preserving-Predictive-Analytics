import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'models/models.dart';
import 'theme/theme.dart';
import 'screens/results.dart';
import 'screens/upload.dart';
import 'screens/configure.dart';
import 'screens/train.dart';
import 'screens/login.dart';
import 'services/auth_service.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  SystemChrome.setSystemUIOverlayStyle(const SystemUiOverlayStyle(
    statusBarColor: Colors.transparent,
    statusBarIconBrightness: Brightness.light,
    systemNavigationBarColor: T.bg,
  ));
  runApp(const App());
}

class App extends StatelessWidget {
  const App({super.key});
  @override Widget build(BuildContext ctx) => MaterialApp(
    title: 'Training Models',
    debugShowCheckedModeBanner: false,
    theme: T.dark,
    home: ListenableBuilder(
      listenable: AuthService.instance,
      builder: (ctx, _) => AuthService.instance.isAuthenticated
        ? const Nav()
        : const LoginScreen(),
    ),
  );
}

enum _Step { upload, config, train, results }

class Nav extends StatefulWidget {
  const Nav({super.key});
  @override State<Nav> createState() => _NavState();
}

class _NavState extends State<Nav> {
  _Step _step = _Step.upload;
  CsvData? _csv;
  TrainConfig? _cfg;
  TrainResult? _central, _fl;

  void _reset() => setState(() {
    _step = _Step.upload; _csv = null; _cfg = null; _central = null; _fl = null;
  });

  @override Widget build(BuildContext ctx) => Scaffold(
    appBar: _step != _Step.upload ? AppBar(
      leading: IconButton(icon:const Icon(Icons.arrow_back_ios,size:17),
        onPressed:() => setState((){
          if(_step==_Step.config) _step=_Step.upload;
          else if(_step==_Step.train) _step=_Step.config;
          else if(_step==_Step.results) _step=_Step.train;
        })),
      title: _StepBar(step: _step.index),
      actions: [
        IconButton(icon:const Icon(Icons.logout,size:20),onPressed:() => AuthService.instance.logout()),
        IconButton(icon:const Icon(Icons.refresh,size:20),onPressed:_reset)
      ],
    ) : null,
    body: _body(),
  );

  Widget _body() {
    switch (_step) {
      case _Step.upload:
        return UploadScreen(onDone: (d) => setState((){_csv=d;_step=_Step.config;}));
      case _Step.config:
        return ConfigScreen(csv: _csv!,
          onDone: (c) => setState((){_cfg=c;_step=_Step.train;}));
      case _Step.train:
        return TrainScreen(csv: _csv!, cfg: _cfg!,
          onDone: (c,f) => setState((){
            _central=c??_central; _fl=f??_fl; _step=_Step.results;
          }));
      case _Step.results:
        return ResultsScreen(central:_central, fl:_fl,
          onTrainMore: () => setState(()=>_step=_Step.train));
    }
  }
}

class _StepBar extends StatelessWidget {
  final int step;
  const _StepBar({required this.step});
  static const _labels = ['Upload','Config','Train','Results'];
  static const _icons  = ['📂','⚙️','🚀','📊'];

  @override Widget build(BuildContext ctx) => Row(
    mainAxisAlignment: MainAxisAlignment.center,
    children: List.generate(4, (i) {
      final done = i < step; final active = i == step;
      return Row(mainAxisSize: MainAxisSize.min, children: [
        if(i>0) Container(width:16,height:1,
          color:done?T.ok:T.border),
        Column(mainAxisSize:MainAxisSize.min,children:[
          Container(width:28,height:28,
            decoration:BoxDecoration(
              color: done?T.ok:active?T.fl.withOpacity(0.15):T.border,
              shape:BoxShape.circle,
              border:active?Border.all(color:T.fl,width:2):null),
            child:Center(child:Text(done?'✓':_icons[i],
              style:TextStyle(fontSize:done?11:10)))),
          Text(_labels[i],style:TextStyle(fontSize:7,fontFamily:'monospace',
            color:active?T.fl:done?T.ok:T.muted)),
        ]),
      ]);
    }),
  );
}
