(
var args = thisProcess.argv;
var outputPath = args.at(0);
var pluginPath = args.at(1);
var scsynthPath = args.at(2);
var frames = 640;
var sampleRate = 48000;
var blockSize = 64;
var duration = frames / sampleRate;
var options;
var def;
var score;

if(outputPath.isNil or: { pluginPath.isNil } or: { scsynthPath.isNil }) {
    "usage: env_detect.scd OUTPUT_F32 TEMP_PLUGIN_PATH SCSYNTH".postln;
    2.exit;
};

def = SynthDef(\spec100_env_detect_oracle, {
    var source = SinOsc.ar(337, 0.3, 0.55)
        + LFSaw.ar(73, 0.1, 0.25)
        + Phasor.ar(0, 0.004375, -0.2, 0.2, -0.15);
    Out.ar(0, EnvDetect.ar(source, \attack.kr(100), \release.kr(0)));
});

score = Score([
    [0.0, [\d_recv, def.asBytes]],
    [0.0, [\s_new, \spec100_env_detect_oracle, 1000, 0, 1]],
    [(2 * blockSize) / sampleRate, [\n_set, 1000, \attack, 8, \release, 2]],
    [(4 * blockSize) / sampleRate, [\n_set, 1000, \attack, 0, \release, 12]],
    [(6 * blockSize) / sampleRate, [\n_set, 1000, \attack, 250, \release, 50]],
    [(8 * blockSize) / sampleRate, [\n_set, 1000, \attack, 32, \release, 4]],
    [duration, [\n_free, 1000]]
]);

options = ServerOptions.new
    .numOutputBusChannels_(1)
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
