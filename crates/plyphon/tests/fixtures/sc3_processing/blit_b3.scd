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
    "usage: blit_b3.scd OUTPUT_F32 TEMP_PLUGIN_PATH SCSYNTH".postln;
    2.exit;
};

def = SynthDef(\spec100_blit_b3_oracle, {
    var freq = \freq.kr(440);
    var leak = \leak.kr(0.99);
    var leak2 = \leak2.kr(0.97);
    Out.ar(0, [
        BlitB3.ar(freq),
        BlitB3Saw.ar(freq, leak),
        BlitB3Square.ar(freq, leak),
        BlitB3Tri.ar(freq, leak, leak2)
    ]);
});

score = Score([
    [0.0, [\d_recv, def.asBytes]],
    [0.0, [\s_new, \spec100_blit_b3_oracle, 1000, 0, 1]],
    [(2 * blockSize) / sampleRate, [\n_set, 1000, \freq, 0]],
    [(4 * blockSize) / sampleRate, [\n_set, 1000, \freq, -220]],
    [(6 * blockSize) / sampleRate, [\n_set, 1000, \freq, 24000]],
    [(8 * blockSize) / sampleRate,
        [\n_set, 1000, \freq, 880, \leak, 0.93, \leak2, 0.89]],
    [duration, [\n_free, 1000]]
]);

options = ServerOptions.new
    .numOutputBusChannels_(4)
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
