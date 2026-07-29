(
var args = thisProcess.argv;
var outputPath = args.at(0);
var sourcePath = args.at(1);
var pluginPath = args.at(2);
var scsynthPath = args.at(3);
var frames = 1536;
var sampleRate = 48000;
var blockSize = 64;
var fftSize = 128;
var duration = (frames - blockSize) / sampleRate;
var nan = 0.0 / 0.0;
var positiveInfinity = 1.0 / 0.0;
var negativeInfinity = -1.0 / 0.0;
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
    "usage: pv_freeze_controls.scd OUTPUT_F32 SOURCE_F32 TEMP_PLUGIN_PATH SCSYNTH".postln;
    2.exit;
};

sourceData = Signal.newClear(frames);
File.use(sourcePath, "rb", { |file| file.readLE(sourceData) });

def = SynthDef(\spec100_pv_freeze_controls, {
    var phase = Phasor.ar(0, 1, 0, frames, 0);
    var source = BufRd.ar(1, \sourceBuf.ir(10), phase, 0, 1);
    var controls = [
        \freezeNan.kr(0),
        \freezePositiveInfinity.kr(0),
        \freezeNegativeInfinity.kr(0),
        \freezeZero.kr(0),
        \freezeOne.kr(0)
    ];
    var outputs = controls.collect({ |freeze|
        var chain = FFT(LocalBuf(fftSize, 1), source, 1, 0, 1, fftSize);
        var frozen = PV_Freeze(chain, freeze);
        var ready = frozen > -1;
        var resynthesized = IFFT(frozen, 0, fftSize);
        var interpolated = K2A.ar([frozen, ready, freeze]);
        var readyEdgePulse = Trig1.ar(K2A.ar(ready), SampleDur.ir);
        [resynthesized] ++ interpolated ++ [readyEdgePulse]
    }).flat;
    Out.ar(0, outputs);
});

score = Score([
    [0.0, [\b_alloc, 10, frames, 1]],
    [0.0, [\b_setn, 10, 0, frames] ++ sourceData],
    [0.0, [\d_recv, def.asBytes]],
    [0.0, [\s_new, \spec100_pv_freeze_controls, 1000, 0, 1]],
    [(6 * blockSize) / sampleRate, [
        \n_set, 1000,
        \freezeNan, nan,
        \freezePositiveInfinity, positiveInfinity,
        \freezeNegativeInfinity, negativeInfinity,
        \freezeZero, 0,
        \freezeOne, 1
    ]],
    [(12 * blockSize) / sampleRate, [
        \n_set, 1000,
        \freezeNan, 0,
        \freezePositiveInfinity, 0,
        \freezeNegativeInfinity, 0,
        \freezeZero, 0,
        \freezeOne, 0
    ]],
    [duration, [\n_free, 1000]]
]);

options = ServerOptions.new
    .numOutputBusChannels_(25)
    .numInputBusChannels_(0)
    .blockSize_(blockSize)
    .sampleRate_(sampleRate)
    .numWireBufs_(512)
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
