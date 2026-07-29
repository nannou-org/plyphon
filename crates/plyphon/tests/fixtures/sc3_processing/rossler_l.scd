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
    "usage: rossler_l.scd OUTPUT_F32 TEMP_PLUGIN_PATH SCSYNTH".postln;
    2.exit;
};

def = SynthDef(\spec100_rossler_l_oracle, {
    var outputs = RosslerL.ar(
        \freq.kr(6000),
        \a.kr(0.2),
        \b.kr(0.2),
        \c.kr(5.7),
        \h.kr(0.05),
        \xi.kr(0.1),
        \yi.kr(0),
        \zi.kr(0)
    );
    Out.ar(0, outputs);
});

score = Score([
    [0.0, [\d_recv, def.asBytes]],
    [0.0, [\s_new, \spec100_rossler_l_oracle, 1000, 0, 1]],
    [(3 * blockSize) / sampleRate,
        [\n_set, 1000, \xi, 0.35, \yi, -0.2, \zi, 0.15]],
    [(5 * blockSize) / sampleRate,
        [\n_set, 1000, \freq, 12000, \a, 0.36, \b, 0.35, \c, 4.5, \h, 0.03]],
    [(7 * blockSize) / sampleRate,
        [\n_set, 1000, \freq, 48000]],
    [duration, [\n_free, 1000]]
]);

options = ServerOptions.new
    .numOutputBusChannels_(3)
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

