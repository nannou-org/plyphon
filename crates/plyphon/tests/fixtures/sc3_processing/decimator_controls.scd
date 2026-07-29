(
var args = thisProcess.argv;
var outputPath = args.at(0);
var pluginPath = args.at(1);
var scsynthPath = args.at(2);
var frames = 512;
var sampleRate = 48000;
var blockSize = 64;
var duration = frames / sampleRate;
var nan = 0.0 / 0.0;
var positiveInfinity = 1.0 / 0.0;
var negativeInfinity = -1.0 / 0.0;
var options;
var def;
var score;

if(outputPath.isNil or: { pluginPath.isNil } or: { scsynthPath.isNil }) {
    "usage: decimator_controls.scd OUTPUT_F32 TEMP_PLUGIN_PATH SCSYNTH".postln;
    2.exit;
};

def = SynthDef(\spec100_decimator_controls, {
    var source = Phasor.ar(0, 0.03125, -1, 1, -0.75);
    Out.ar(0, [
        Decimator.ar(source, \rateNan.kr(12000), 8),
        Decimator.ar(source, \ratePositiveInfinity.kr(12000), 8),
        Decimator.ar(source, \rateNegativeInfinity.kr(12000), 8),
        Decimator.ar(source, 12000, \bitsNan.kr(8)),
        Decimator.ar(source, 12000, \bitsPositiveInfinity.kr(8)),
        Decimator.ar(source, 12000, \bitsNegativeInfinity.kr(8)),
        Decimator.ar(source, \rateZeroReference.kr(12000), 8),
        Decimator.ar(source, \rateSampleRateReference.kr(12000), 8),
        Decimator.ar(source, 12000, \bits31Reference.kr(8))
    ]);
});

score = Score([
    [0.0, [\d_recv, def.asBytes]],
    [0.0, [\s_new, \spec100_decimator_controls, 1000, 0, 1]],
    [(2 * blockSize) / sampleRate, [
        \n_set, 1000,
        \rateNan, nan,
        \ratePositiveInfinity, positiveInfinity,
        \rateNegativeInfinity, negativeInfinity,
        \bitsNan, nan,
        \bitsPositiveInfinity, positiveInfinity,
        \bitsNegativeInfinity, negativeInfinity,
        \rateZeroReference, 0,
        \rateSampleRateReference, 48000,
        \bits31Reference, 31
    ]],
    [(4 * blockSize) / sampleRate, [
        \n_set, 1000,
        \rateNan, 24000,
        \ratePositiveInfinity, 24000,
        \rateNegativeInfinity, 24000,
        \bitsNan, 4,
        \bitsPositiveInfinity, 4,
        \bitsNegativeInfinity, 4,
        \rateZeroReference, 24000,
        \rateSampleRateReference, 24000,
        \bits31Reference, 4
    ]],
    [duration, [\n_free, 1000]]
]);

options = ServerOptions.new
    .numOutputBusChannels_(9)
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
