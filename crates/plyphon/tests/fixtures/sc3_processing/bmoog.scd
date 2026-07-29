(
var args = thisProcess.argv;
var outputPath = args.at(0);
var pluginPath = args.at(1);
var scsynthPath = args.at(2);
var frames = 512;
var sampleRate = 48000;
var blockSize = 64;
var duration = frames / sampleRate;
var options;
var def;
var score;

if(outputPath.isNil or: { pluginPath.isNil } or: { scsynthPath.isNil }) {
    "usage: bmoog.scd OUTPUT_F32 TEMP_PLUGIN_PATH SCSYNTH".postln;
    2.exit;
};

def = SynthDef(\spec100_bmoog_oracle, {
    var source = Phasor.ar(0, 0.021875, -0.8, 0.8, -0.6);
    var cutoff = \cutoff.kr(440);
    var q = \q.kr(0.2);
    Out.ar(0, [
        BMoog.ar(source, cutoff, q, 0),
        BMoog.ar(source, cutoff, q, 1),
        BMoog.ar(source, cutoff, q, 2),
        BMoog.ar(source, cutoff, q, 3)
    ]);
});

score = Score([
    [0.0, [\d_recv, def.asBytes]],
    [0.0, [\s_new, \spec100_bmoog_oracle, 1000, 0, 1]],
    [(2 * blockSize) / sampleRate, [\n_set, 1000, \cutoff, 1200, \q, 0.65]],
    [(4 * blockSize) / sampleRate, [\n_set, 1000, \cutoff, 6000, \q, 0.9]],
    [(6 * blockSize) / sampleRate, [\n_set, 1000, \cutoff, 220, \q, 0.1]],
    [duration, [\n_free, 1000]]
]);

options = ServerOptions.new
    .numOutputBusChannels_(4)
    .numInputBusChannels_(0)
    .blockSize_(blockSize)
    .sampleRate_(sampleRate)
    .loadDefs_(false)
    .zeroConf_(false)
    .ugenPluginsPath_(pluginPath);

Score.program = scsynthPath;
score.recordNRT(
    outputFilePath: outputPath,
    sampleRate: sampleRate,
    headerFormat: "RAW",
    sampleFormat: "float",
    options: options,
    duration: duration,
    action: { 0.exit }
);
)

