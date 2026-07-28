(
var args = thisProcess.argv;
var outputPath = args.at(0);
var sourceAPath = args.at(1);
var sourceBPath = args.at(2);
var pluginPath = args.at(3);
var scsynthPath = args.at(4);
var frames = 1536;
var sampleRate = 48000;
var blockSize = 64;
var fftSize = 128;
// NRT buffer setup adds one render block; subtract it to retain exactly `frames`.
var duration = (frames - blockSize) / sampleRate;
var options;
var def;
var score;
var sourceAData;
var sourceBData;

if(
    outputPath.isNil
    or: { sourceAPath.isNil }
    or: { sourceBPath.isNil }
    or: { pluginPath.isNil }
    or: { scsynthPath.isNil }
) {
    "usage: pv_morph.scd OUTPUT_F32 SOURCE_A_F32 SOURCE_B_F32 TEMP_PLUGIN_PATH SCSYNTH".postln;
    2.exit;
};

sourceAData = Signal.newClear(frames);
sourceBData = Signal.newClear(frames);
File.use(sourceAPath, "rb", { |file| file.readLE(sourceAData) });
File.use(sourceBPath, "rb", { |file| file.readLE(sourceBData) });

def = SynthDef(\spec100_pv_morph_oracle, {
    var phase = Phasor.ar(0, 1, 0, frames, 0);
    var sourceA = BufRd.ar(1, \sourceABuf.ir(10), phase, 0, 1);
    var sourceB = BufRd.ar(1, \sourceBBuf.ir(11), phase, 0, 1);
    var chainA = FFT(LocalBuf(fftSize, 1), sourceA, 1, 0, 1, fftSize);
    var chainB = FFT(LocalBuf(fftSize, 1), sourceB, 1, 0, 1, fftSize);
    var inputReady = (chainA > -1) * (chainB > -1);
    var readyCount = PulseCount.kr(inputReady);
    var morph = Select.kr(
        (readyCount - 1).clip(0, 7),
        [0, 0.25, 0.5, 0.75, 1, 0.6, 0.2, 0.9]
    );
    var morphed = PV_Morph(chainA, chainB, morph);
    var ready = morphed > -1;
    var first = Demand.kr(ready, 0, UnpackFFT(morphed, fftSize, 0, 14));
    var second = Demand.kr(ready, 0, UnpackFFT(morphed, fftSize, 15, 29));
    var third = Demand.kr(ready, 0, UnpackFFT(morphed, fftSize, 30, 44));
    var fourth = Demand.kr(ready, 0, UnpackFFT(morphed, fftSize, 45, 59));
    var fifth = Demand.kr(ready, 0, UnpackFFT(morphed, fftSize, 60, 64));
    var decoded = first ++ second ++ third ++ fourth ++ fifth;
    var resynthesized = IFFT(morphed, 0, fftSize);
    // These control-rate values are linearly interpolated onto the audio bus.
    // The separate pulse records the audio-rate positive edge of that lane.
    var interpolated = K2A.ar([morphed, ready, morph] ++ decoded);
    var readyEdgePulse = Trig1.ar(K2A.ar(ready), SampleDur.ir);
    Out.ar(
        0,
        [resynthesized]
        ++ interpolated.copyRange(0, 2)
        ++ [readyEdgePulse]
        ++ interpolated.copyRange(3, interpolated.size - 1)
    );
});

score = Score([
    [0.0, [\b_alloc, 10, frames, 1]],
    [0.0, [\b_setn, 10, 0, frames] ++ sourceAData],
    [0.0, [\b_alloc, 11, frames, 1]],
    [0.0, [\b_setn, 11, 0, frames] ++ sourceBData],
    [0.0, [\d_recv, def.asBytes]],
    [0.0, [\s_new, \spec100_pv_morph_oracle, 1000, 0, 1]],
    [duration, [\n_free, 1000]]
]);

options = ServerOptions.new
    .numOutputBusChannels_(135)
    .numInputBusChannels_(0)
    .blockSize_(blockSize)
    .sampleRate_(sampleRate)
    .numWireBufs_(768)
    .memSize_(262144)
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
