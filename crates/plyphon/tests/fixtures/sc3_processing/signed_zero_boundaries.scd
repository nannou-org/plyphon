(
var args = thisProcess.argv;
var outputPath = args.at(0);
var sourcePath = args.at(1);
var pluginPath = args.at(2);
var scsynthPath = args.at(3);
var frames = 512;
var sampleRate = 48000;
var blockSize = 64;
var fftSize = 128;
var duration = frames / sampleRate;
var positiveZero = Float.from32Bits(0);
var negativeZero = Float.from32Bits(16r80000000);
var rosslerDefaults = [48000, 0.2, 0.2, 5.7, 0.05, 0.1, 0.2, 0.3];
var options;
var def;
var score;
var sourceData;

if(
    outputPath.isNil
    or: { sourcePath.isNil }
    or: { pluginPath.isNil }
    or: { scsynthPath.isNil }
) {
    "usage: signed_zero_boundaries.scd OUTPUT_F32 SOURCE_F32 TEMP_PLUGIN_PATH SCSYNTH".postln;
    2.exit;
};

sourceData = Signal.newClear(frames);
File.use(sourcePath, "rb", { |file| file.readLE(sourceData) });

def = SynthDef(\spec100_signed_zero_boundaries, {
    var positiveZeroControl = \positiveZero.ir(1);
    var negativeZeroControl = \negativeZero.ir(-1);
    var positiveZeroSignal = \positiveZeroAudio.ar(1);
    var negativeZeroSignal = \negativeZeroAudio.ar(-1);
    var ramp = Phasor.ar(0, 1 / sampleRate, -0.75, 0.75, -0.75);
    var witnesses = [
        1 / positiveZeroSignal,
        1 / negativeZeroSignal
    ];
    var decimatorPositive = Decimator.ar(positiveZeroSignal, sampleRate, 24);
    var decimatorNegative = Decimator.ar(negativeZeroSignal, sampleRate, 24);
    var decimator = [
        decimatorPositive,
        decimatorNegative,
        1 / decimatorPositive,
        1 / decimatorNegative,
        Decimator.ar(ramp, positiveZeroControl, 24),
        Decimator.ar(ramp, negativeZeroControl, 24),
        Decimator.ar(ramp, sampleRate, positiveZeroControl),
        Decimator.ar(ramp, sampleRate, negativeZeroControl)
    ];
    var bmoogPositive = BMoog.ar(positiveZeroSignal, 440, 0.5, 0);
    var bmoogNegative = BMoog.ar(negativeZeroSignal, 440, 0.5, 0);
    var bmoog = [
        bmoogPositive,
        bmoogNegative,
        1 / bmoogPositive,
        1 / bmoogNegative,
        BMoog.ar(ramp, positiveZeroControl, 0.5, 0),
        BMoog.ar(ramp, negativeZeroControl, 0.5, 0),
        BMoog.ar(ramp, 440, positiveZeroControl, 0),
        BMoog.ar(ramp, 440, negativeZeroControl, 0),
        BMoog.ar(ramp, 440, 0.5, positiveZeroControl),
        BMoog.ar(ramp, 440, 0.5, negativeZeroControl)
    ];
    var perlin = [
        Perlin3.ar(positiveZeroSignal, DC.ar(0.375), DC.ar(-0.625)),
        Perlin3.ar(negativeZeroSignal, DC.ar(0.375), DC.ar(-0.625)),
        Perlin3.ar(DC.ar(-0.25), positiveZeroSignal, DC.ar(-0.625)),
        Perlin3.ar(DC.ar(-0.25), negativeZeroSignal, DC.ar(-0.625)),
        Perlin3.ar(DC.ar(-0.25), DC.ar(0.375), positiveZeroSignal),
        Perlin3.ar(DC.ar(-0.25), DC.ar(0.375), negativeZeroSignal)
    ];
    var rossler = (0..7).collect({ |controlIndex|
        var positiveInputs = rosslerDefaults.copy;
        var negativeInputs = rosslerDefaults.copy;
        positiveInputs.put(controlIndex, positiveZeroControl);
        negativeInputs.put(controlIndex, negativeZeroControl);
        [
            RosslerL.ar(*positiveInputs),
            RosslerL.ar(*negativeInputs)
        ]
    }).flat;
    var sourcePhase = Phasor.ar(0, 1, 0, frames, 0);
    var source = BufRd.ar(1, \sourceBuf.ir(10), sourcePhase, 0, 1);
    var pv = [positiveZeroControl, negativeZeroControl].collect({ |freeze|
        var chain = FFT(LocalBuf(fftSize, 1), source, 1, 0, 1, fftSize);
        var frozen = PV_Freeze(chain, freeze);
        var ready = frozen > -1;
        [
            IFFT(frozen, 0, fftSize),
            K2A.ar(frozen),
            K2A.ar(ready),
            Trig1.ar(K2A.ar(ready), SampleDur.ir)
        ]
    }).flat;
    Out.ar(0, [witnesses, decimator, bmoog, perlin, rossler, pv].flat);
});

score = Score([
    [0.0, [\b_alloc, 10, frames, 1]],
    [0.0, [\b_setn, 10, 0, frames] ++ sourceData],
    [0.0, [\d_recv, def.asBytes]],
    [0.0, [
        \s_new, \spec100_signed_zero_boundaries, 1000, 0, 1,
        \positiveZero, positiveZero,
        \negativeZero, negativeZero,
        \positiveZeroAudio, positiveZero,
        \negativeZeroAudio, negativeZero
    ]],
    [duration, [\n_free, 1000]]
]);

options = ServerOptions.new
    .numOutputBusChannels_(82)
    .numInputBusChannels_(0)
    .blockSize_(blockSize)
    .sampleRate_(sampleRate)
    .numWireBufs_(2048)
    .memSize_(131072)
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
