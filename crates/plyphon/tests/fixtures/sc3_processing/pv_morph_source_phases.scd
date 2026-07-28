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
    "usage: pv_morph_source_phases.scd OUTPUT_F32 SOURCE_A_F32 SOURCE_B_F32 TEMP_PLUGIN_PATH SCSYNTH".postln;
    2.exit;
};

sourceAData = Signal.newClear(frames);
sourceBData = Signal.newClear(frames);
File.use(sourceAPath, "rb", { |file| file.readLE(sourceAData) });
File.use(sourceBPath, "rb", { |file| file.readLE(sourceBData) });

def = SynthDef(\spec100_pv_morph_source_phases, {
    var phase = Phasor.ar(0, 1, 0, frames, 0);
    var sourceA = BufRd.ar(1, \sourceABuf.ir(10), phase, 0, 1);
    var sourceB = BufRd.ar(1, \sourceBBuf.ir(11), phase, 0, 1);
    var chainA = FFT(LocalBuf(fftSize, 1), sourceA, 1, 0, 1, fftSize);
    var chainB = FFT(LocalBuf(fftSize, 1), sourceB, 1, 0, 1, fftSize);
    var polarA = PV_MagSmooth(chainA, 0);
    var polarB = PV_MagSmooth(chainB, 0);
    var ready = (polarA > -1) * (polarB > -1);
    var readPhases = { |chain|
        Array.fill(63, { |index|
            var slot = DC.ar((index + 1) * 2 + 1);
            var reader = BufRd.ar(1, chain, slot, 0, 1);
            Latch.kr(A2K.kr(reader), ready);
        });
    };
    var phasesA = readPhases.(polarA);
    var phasesB = readPhases.(polarB);

    Out.ar(0, K2A.ar(phasesA ++ phasesB));
});

score = Score([
    [0.0, [\b_alloc, 10, frames, 1]],
    [0.0, [\b_setn, 10, 0, frames] ++ sourceAData],
    [0.0, [\b_alloc, 11, frames, 1]],
    [0.0, [\b_setn, 11, 0, frames] ++ sourceBData],
    [0.0, [\d_recv, def.asBytes]],
    [0.0, [\s_new, \spec100_pv_morph_source_phases, 1000, 0, 1]],
    [duration, [\n_free, 1000]]
]);

options = ServerOptions.new
    .numOutputBusChannels_(126)
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
