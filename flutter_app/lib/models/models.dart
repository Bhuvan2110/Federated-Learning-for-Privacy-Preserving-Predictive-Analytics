class ColStat {
  final String col, type;
  final double? min, max, avg;
  final int? uniqueCount;
  final List<String>? uniqueValues;
  final int count;
  const ColStat({required this.col, required this.type, required this.count,
    this.min, this.max, this.avg, this.uniqueCount, this.uniqueValues});
  factory ColStat.fromJson(Map<String,dynamic> j) => ColStat(
    col:j['col'], type:j['type'], count:(j['count']as num).toInt(),
    min:j['min']!=null?(j['min']as num).toDouble():null,
    max:j['max']!=null?(j['max']as num).toDouble():null,
    avg:j['avg']!=null?(j['avg']as num).toDouble():null,
    uniqueCount:j['uniqueCount']!=null?(j['uniqueCount']as num).toInt():null,
    uniqueValues:j['uniqueValues']!=null?List<String>.from(j['uniqueValues']):null);
}

class CsvData {
  final String filename;
  final int totalRows, totalCols;
  final List<String> headers;
  final List<Map<String,dynamic>> rows;
  final List<ColStat> stats;
  bool isEncrypted;
  CsvData({required this.filename, required this.totalRows,
    required this.totalCols, required this.headers, required this.rows,
    required this.stats, this.isEncrypted = false});
  factory CsvData.fromJson(Map<String,dynamic> j) => CsvData(
    filename: j['filename'], totalRows:(j['totalRows']as num).toInt(),
    totalCols:(j['totalCols']as num).toInt(),
    headers: List<String>.from(j['headers']),
    rows: List<Map<String,dynamic>>.from(j['rows'].map((r)=>Map<String,dynamic>.from(r))),
    stats: List<ColStat>.from(j['stats'].map((s)=>ColStat.fromJson(s))),
    isEncrypted: j['encrypted'] == true);
}

class TrainConfig {
  int targetIdx;
  Map<String,String> ftypes;
  int epochs; double lr;
  int rounds, localEpochs, numClients;
  TrainConfig({required this.targetIdx, required this.ftypes,
    this.epochs=100, this.lr=0.1, this.rounds=25, this.localEpochs=5, this.numClients=5});
}

class ConfMatrix {
  final int tp, fp, fn, tn;
  const ConfMatrix({required this.tp, required this.fp, required this.fn, required this.tn});
  factory ConfMatrix.fromJson(Map<String,dynamic> j) => ConfMatrix(
    tp:(j['tp']as num).toInt(), fp:(j['fp']as num).toInt(),
    fn:(j['fn']as num).toInt(), tn:(j['tn']as num).toInt());
}

class Metrics {
  final double accuracy, precision, recall, f1;
  final ConfMatrix conf;
  const Metrics({required this.accuracy, required this.precision,
    required this.recall, required this.f1, required this.conf});
  factory Metrics.fromJson(Map<String,dynamic> j) => Metrics(
    accuracy:(j['accuracy']as num).toDouble(),
    precision:(j['precision']as num).toDouble(),
    recall:(j['recall']as num).toDouble(),
    f1:(j['f1']as num).toDouble(),
    conf:ConfMatrix.fromJson(j['confMatrix']));
}

class TrainResult {
  final String model, targetCol;
  final List<String> featureCols, uniqueLabels;
  final int trainSamples, testSamples, trainingTimeMs;
  final List<double> lossHistory;
  final Metrics trainMetrics, testMetrics;
  final double finalLoss;
  final int? epochs, numClients, rounds, localEpochs;
  final double? lr;
  const TrainResult({required this.model, required this.featureCols, required this.targetCol,
    required this.uniqueLabels, required this.trainSamples, required this.testSamples,
    required this.lossHistory, required this.trainMetrics, required this.testMetrics,
    required this.finalLoss, required this.trainingTimeMs,
    this.epochs, this.lr, this.numClients, this.rounds, this.localEpochs});
  factory TrainResult.fromJson(Map<String,dynamic> j) => TrainResult(
    model:j['model'], featureCols:List<String>.from(j['featureCols']),
    targetCol:j['targetCol'], uniqueLabels:List<String>.from(j['uniqueLabels']),
    trainSamples:(j['trainSamples']as num).toInt(), testSamples:(j['testSamples']as num).toInt(),
    lossHistory:List<double>.from(j['lossHistory'].map((v)=>(v as num).toDouble())),
    trainMetrics:Metrics.fromJson(j['trainMetrics']),
    testMetrics:Metrics.fromJson(j['testMetrics']),
    finalLoss:(j['finalLoss']as num).toDouble(),
    trainingTimeMs:(j['trainingTimeMs']as num).toInt(),
    epochs:j['epochs']!=null?(j['epochs']as num).toInt():null,
    lr:j['lr']!=null?(j['lr']as num).toDouble():null,
    numClients:j['numClients']!=null?(j['numClients']as num).toInt():null,
    rounds:j['rounds']!=null?(j['rounds']as num).toInt():null,
    localEpochs:j['localEpochs']!=null?(j['localEpochs']as num).toInt():null);
}

class AuthToken {
  final String access, refresh;
  final int expires;
  const AuthToken({required this.access, required this.refresh, required this.expires});
  factory AuthToken.fromJson(Map<String,dynamic> j) => AuthToken(
    access: j['access_token'], refresh: j['refresh_token'],
    expires: (j['expires_in'] as num).toInt());
}

class User {
  final String id, email, role;
  final Map<String,dynamic> attrs;
  const User({required this.id, required this.email, required this.role, required this.attrs});
}
