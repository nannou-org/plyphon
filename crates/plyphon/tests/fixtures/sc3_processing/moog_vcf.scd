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
    "usage: moog_vcf.scd OUTPUT_F32 TEMP_PLUGIN_PATH SCSYNTH".postln;
    2.exit;
};

def = SynthDef(\spec100_moog_vcf_oracle, {
    var source = LFSaw.ar(127, 0.2, 0.32)
        + SinOsc.ar(383, 0.4, 0.18);
    var cutoffControl = \cutoff.kr(440);
    var resControl = \res.kr(0.2);
    var cutoffAudio = SinOsc.ar(19, 0.1).range(220, 6200);
    var resAudio = LFTri.ar(13, 0.3).range(0.1, 0.9);
    Out.ar(0, [
        MoogVCF.ar(source, cutoffControl, resControl),
        MoogVCF.ar(source, cutoffAudio, resControl),
        MoogVCF.ar(source, cutoffControl, resAudio),
        MoogVCF.ar(source, cutoffAudio, resAudio)
    ]);
});

score = Score([
    [0.0, [\d_recv, def.asBytes]],
    [0.0, [\s_new, \spec100_moog_vcf_oracle, 1000, 0, 1]],
    [(2 * blockSize) / sampleRate, [\n_set, 1000, \cutoff, 1400, \res, 0.5]],
    [(4 * blockSize) / sampleRate, [\n_set, 1000, \cutoff, 6800, \res, 0.85]],
    [(6 * blockSize) / sampleRate, [\n_set, 1000, \cutoff, 180, \res, 0.1]],
    [(8 * blockSize) / sampleRate, [\n_set, 1000, \cutoff, 3200, \res, 0.7]],
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
