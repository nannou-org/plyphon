(
var args = thisProcess.argv;
var outputPath = args.at(0);
var sourcePath = args.at(1);
var pluginPath = args.at(2);
var scsynthPath = args.at(3);
var frames = 640;
var sampleRate = 48000;
var blockSize = 64;
var duration = frames / sampleRate;
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
    "usage: dfm1.scd OUTPUT_F32 SOURCE_F32 TEMP_PLUGIN_PATH SCSYNTH".postln;
    2.exit;
};

sourceData = Signal.newClear(frames);
File.use(sourcePath, "rb", { |file| file.readLE(sourceData) });

def = SynthDef(\spec100_dfm1_oracle, {
    var phase = Phasor.ar(0, 1, 0, frames, 0);
    var source = BufRd.ar(1, \sourceBuf.ir(10), phase, 0, 1);
    var freq = \freq.kr(1000);
    var res = \res.kr(0.05);
    var inputGain = \inputGain.kr(1.5);
    Out.ar(0, [
        DFM1.ar(source, freq, res, inputGain, 0, 0),
        DFM1.ar(source, freq, res, inputGain, 1, 0)
    ]);
});

score = Score([
    [0.0, [\b_alloc, 10, frames, 1]],
    [0.0, [\b_setn, 10, 0, frames] ++ sourceData],
    [0.0, [\d_recv, def.asBytes]],
    [0.0, [\s_new, \spec100_dfm1_oracle, 1000, 0, 1]],
    [(2 * blockSize) / sampleRate,
        [\n_set, 1000, \freq, 1800, \res, 0.1, \inputGain, 1.75]],
    [(4 * blockSize) / sampleRate,
        [\n_set, 1000, \freq, 3200, \res, 0.15, \inputGain, 1.25]],
    [(6 * blockSize) / sampleRate,
        [\n_set, 1000, \freq, 700, \res, 0.08, \inputGain, 2]],
    [(8 * blockSize) / sampleRate,
        [\n_set, 1000, \freq, 4400, \res, 0.2, \inputGain, 1]],
    [duration, [\n_free, 1000]]
]);

options = ServerOptions.new
    .numOutputBusChannels_(2)
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
