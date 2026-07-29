(
var args = thisProcess.argv;
var outputPath = args.at(0);
var variant = args.at(1);
var pluginPath = args.at(2);
var scsynthPath = args.at(3);
var frames = 640;
var sampleRate = 48000;
var blockSize = 64;
var duration = frames / sampleRate;
var options;
var def;
var score;
var controlEvents;

if(outputPath.isNil or: { variant.isNil } or: { pluginPath.isNil } or: { scsynthPath.isNil }) {
    "usage: moog_ladder.scd OUTPUT_F32 ar|kr TEMP_PLUGIN_PATH SCSYNTH".postln;
    2.exit;
};

if((variant != "ar") and: { variant != "kr" }) {
    "moog_ladder variant must be ar or kr".postln;
    2.exit;
};

def = SynthDef(\spec100_moog_ladder_oracle, {
    var cutoff = \cutoff.kr(440);
    var res = \res.kr(0.2);
    var outputs;
    if(variant == "ar") {
        var source = LFSaw.ar(131, 0.2, 0.3)
            + SinOsc.ar(389, 0.4, 0.2);
        var cutoffAudio = SinOsc.ar(17, 0.1).range(250, 6500);
        var resAudio = LFTri.ar(11, 0.3).range(0.1, 0.9);
        outputs = [
            MoogLadder.ar(source, cutoffAudio, resAudio),
            MoogLadder.ar(source, cutoff, res)
        ];
    } {
        var sourceA = \sourceA.kr(0.45);
        var cutoffA = \cutoffA.kr(120);
        var resA = \resA.kr(0.1);
        var sourceB = \sourceB.kr(0.3);
        var cutoffB = \cutoffB.kr(180);
        var resB = \resB.kr(0.2);
        outputs = K2A.ar([
            MoogLadder.kr(sourceA, cutoffA, resA),
            MoogLadder.kr(sourceB, cutoffB, resB)
        ]);
    };
    Out.ar(0, outputs);
});

controlEvents = if(variant == "ar") {
    [
        [(2 * blockSize) / sampleRate, [\n_set, 1000, \cutoff, 1600, \res, 0.55]],
        [(4 * blockSize) / sampleRate, [\n_set, 1000, \cutoff, 7200, \res, 0.85]],
        [(6 * blockSize) / sampleRate, [\n_set, 1000, \cutoff, 220, \res, 0.1]],
        [(8 * blockSize) / sampleRate, [\n_set, 1000, \cutoff, 3500, \res, 0.7]]
    ]
} {
    [
        [(2 * blockSize) / sampleRate, [
            \n_set, 1000,
            \sourceA, 0.48, \cutoffA, 160, \resA, 0.15,
            \sourceB, 0.27, \cutoffB, 220, \resB, 0.24
        ]],
        [(4 * blockSize) / sampleRate, [
            \n_set, 1000,
            \sourceA, 0.44, \cutoffA, 240, \resA, 0.2,
            \sourceB, 0.33, \cutoffB, 300, \resB, 0.28
        ]],
        [(6 * blockSize) / sampleRate, [
            \n_set, 1000,
            \sourceA, 0.5, \cutoffA, 80, \resA, 0.12,
            \sourceB, 0.29, \cutoffB, 140, \resB, 0.14
        ]],
        [(8 * blockSize) / sampleRate, [
            \n_set, 1000,
            \sourceA, 0.46, \cutoffA, 280, \resA, 0.25,
            \sourceB, 0.35, \cutoffB, 340, \resB, 0.32
        ]]
    ]
};

score = Score([
    [0.0, [\d_recv, def.asBytes]],
    [0.0, [\s_new, \spec100_moog_ladder_oracle, 1000, 0, 1]]
] ++ controlEvents ++ [
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
