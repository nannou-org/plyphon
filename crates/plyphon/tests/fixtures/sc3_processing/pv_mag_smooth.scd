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
// NRT buffer setup adds one render block; subtract it to retain exactly `frames`.
var duration = (frames - blockSize) / sampleRate;
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
    "usage: pv_mag_smooth.scd OUTPUT_F32 SOURCE_F32 TEMP_PLUGIN_PATH SCSYNTH".postln;
    2.exit;
};

sourceData = Signal.newClear(frames);
File.use(sourcePath, "rb", { |file| file.readLE(sourceData) });

def = SynthDef(\spec100_pv_mag_smooth_oracle, {
    var phase = Phasor.ar(0, 1, 0, frames, 0);
    var source = BufRd.ar(1, \sourceBuf.ir(10), phase, 0, 1);
    var chain = FFT(LocalBuf(fftSize, 1), source, 1, 0, 1, fftSize);
    var inputReady = chain > -1;
    var readyCount = PulseCount.kr(inputReady);
    var factor = Select.kr(
        (readyCount - 1).clip(0, 7),
        [0.1, 0.1, 0.25, 0.75, 1, 0, 0.5, 0.2]
    );
    var smoothed = PV_MagSmooth(chain, factor);
    var ready = smoothed > -1;
    var first = Demand.kr(ready, 0, UnpackFFT(smoothed, fftSize, 0, 14));
    var second = Demand.kr(ready, 0, UnpackFFT(smoothed, fftSize, 15, 29));
    var third = Demand.kr(ready, 0, UnpackFFT(smoothed, fftSize, 30, 44));
    var fourth = Demand.kr(ready, 0, UnpackFFT(smoothed, fftSize, 45, 59));
    var fifth = Demand.kr(ready, 0, UnpackFFT(smoothed, fftSize, 60, 64));
    var decoded = first ++ second ++ third ++ fourth ++ fifth;
    var resynthesized = IFFT(smoothed, 0, fftSize);
    // These control-rate values are linearly interpolated onto the audio bus.
    // The separate pulse records the audio-rate positive edge of that lane.
    var interpolated = K2A.ar([smoothed, ready, factor] ++ decoded);
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
    [0.0, [\b_setn, 10, 0, frames] ++ sourceData],
    [0.0, [\d_recv, def.asBytes]],
    [0.0, [\s_new, \spec100_pv_mag_smooth_oracle, 1000, 0, 1]],
    [duration, [\n_free, 1000]]
]);

options = ServerOptions.new
    .numOutputBusChannels_(135)
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
