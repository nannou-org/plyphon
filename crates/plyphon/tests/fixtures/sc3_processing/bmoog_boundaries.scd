(
var args = thisProcess.argv;
var outputPath = args.at(0);
var variant = args.at(1);
var pluginPath = args.at(2);
var scsynthPath = args.at(3);
var frames = 512;
var sampleRate = 48000;
var duration = frames / sampleRate;
var options;
var def;
var score;

if(
    outputPath.isNil
    or: { variant.isNil }
    or: { pluginPath.isNil }
    or: { scsynthPath.isNil }
) {
    "usage: bmoog_boundaries.scd OUTPUT_F32 CASE TEMP_PLUGIN_PATH SCSYNTH".postln;
    2.exit;
};

def = SynthDef(\spec100_bmoog_boundaries, {
    var source = Phasor.ar(0, 0.021875, -0.8, 0.8, -0.6);
    var cases = [
        [19, 0.5],
        [-100, 0.5],
        [24001, 0.5],
        [440, -0.5],
        [440, 1.5],
        [20, 0.5],
        [24000, 0.5],
        [440, 0],
        [440, 1]
    ];
    var parameters = cases.at(variant.asInteger);
    var outputs = (0..3).collect({ |mode|
        BMoog.ar(source, parameters.at(0), parameters.at(1), mode)
    });
    Out.ar(0, outputs);
});

score = Score([
    [0.0, [\d_recv, def.asBytes]],
    [0.0, [\s_new, \spec100_bmoog_boundaries, 1000, 0, 1]],
    [duration, [\n_free, 1000]]
]);

options = ServerOptions.new
    .numOutputBusChannels_(4)
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
