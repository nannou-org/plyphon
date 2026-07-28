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
    "usage: decimator.scd OUTPUT_F32 TEMP_PLUGIN_PATH SCSYNTH".postln;
    2.exit;
};

def = SynthDef(\spec100_decimator_oracle, {
    var source = Phasor.ar(0, 0.03125, -1, 1, -0.75);
    var changing = Decimator.ar(source, \changingRate.kr(44100), \changingBits.kr(24));
    var outputs = [
        changing,
        Decimator.ar(source, 0, 8),
        Decimator.ar(source, 24000, 8.5),
        Decimator.ar(source, 48000, 1),
        Decimator.ar(source, 12000, 0.999),
        Decimator.ar(source, 12000, 1.001),
        Decimator.ar(source, 12000, 30.999),
        Decimator.ar(source, 12000, 31),
        Decimator.ar(source.neg, 16000, 4),
        Decimator.ar(K2A.ar(0), 48000, 12)
    ];
    Out.ar(0, outputs);
});

score = Score([
    [0.0, [\d_recv, def.asBytes]],
    [0.0, [\s_new, \spec100_decimator_oracle, 1000, 0, 1]],
    [(4 * blockSize) / sampleRate,
        [\n_set, 1000, \changingRate, 12000, \changingBits, 8.5]],
    [(6 * blockSize) / sampleRate,
        [\n_set, 1000, \changingRate, 48000, \changingBits, 31]],
    [duration, [\n_free, 1000]]
]);

options = ServerOptions.new
    .numOutputBusChannels_(10)
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
