(
var args = thisProcess.argv;
var outputPath = args.at(0);
var pluginPath = args.at(1);
var scsynthPath = args.at(2);
var frames = 512;
var sampleRate = 48000;
var duration = frames / sampleRate;
var options;
var def;
var score;

if(outputPath.isNil or: { pluginPath.isNil } or: { scsynthPath.isNil }) {
    "usage: rossler_l_boundaries.scd OUTPUT_F32 TEMP_PLUGIN_PATH SCSYNTH".postln;
    2.exit;
};

def = SynthDef(\spec100_rossler_l_boundaries, {
    var outputs = [
        RosslerL.ar(0, 0.2, 0.2, 5.7, 0.05, 0.1, 0, 0),
        RosslerL.ar(-1, 0.2, 0.2, 5.7, 0.05, 0.1, 0, 0),
        RosslerL.ar(0.0005, 0.2, 0.2, 5.7, 0.05, 0.1, 0, 0),
        RosslerL.ar(48000, 0.2, 0.2, 5.7, 5, 0.1, 0, 0)
    ].flat;
    Out.ar(0, outputs);
});

score = Score([
    [0.0, [\d_recv, def.asBytes]],
    [0.0, [\s_new, \spec100_rossler_l_boundaries, 1000, 0, 1]],
    [duration, [\n_free, 1000]]
]);

options = ServerOptions.new
    .numOutputBusChannels_(12)
    .numInputBusChannels_(0)
    .blockSize_(64)
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
