(
var args = thisProcess.argv;
var outputPath = args.at(0);
var pluginPath = args.at(1);
var scsynthPath = args.at(2);
var frames = 1024;
var sampleRate = 48000;
var blockSize = 64;
var duration = frames / sampleRate;
var seed = 1956;
var options;
var baselineDef;
var changeZeroDef;
var changeOneDef;
var scheduledDef;
var stochasticDef;
var score;

if(outputPath.isNil or: { pluginPath.isNil } or: { scsynthPath.isNil }) {
    "usage: dnoise_ring.scd OUTPUT_F32 TEMP_PLUGIN_PATH SCSYNTH".postln;
    2.exit;
};

baselineDef = SynthDef(\spec100_dnoise_baseline, {
    var trig = Impulse.ar(750);
    var baseline;
    RandID.ir(0);
    RandSeed.ar(trig, seed);
    baseline = Demand.ar(trig, 0, [
        Dwhite(0, 1, inf),
        Dwhite(0, 1, inf),
        Dwhite(0, 1, inf)
    ]);
    Out.ar(\out.ir(0), baseline);
});

changeZeroDef = SynthDef(\spec100_dnoise_change_zero, {
    var trig = Impulse.ar(750);
    var ring;
    var probe;
    RandID.ir(0);
    RandSeed.ar(trig, seed);
    ring = Demand.ar(trig, 0, DNoiseRing(0, 0.5, 1, 4, 11));
    probe = Demand.ar(trig, 0, Dwhite(0, 1, inf));
    Out.ar(\out.ir(3), [ring, probe]);
});

changeOneDef = SynthDef(\spec100_dnoise_change_one, {
    var trig = Impulse.ar(750);
    var ring;
    var probe;
    RandID.ir(0);
    RandSeed.ar(trig, seed);
    ring = Demand.ar(trig, 0, DNoiseRing(1, Dseq([0, 1], inf), 1, 4, 11));
    probe = Demand.ar(trig, 0, Dwhite(0, 1, inf));
    Out.ar(\out.ir(5), [ring, probe]);
});

scheduledDef = SynthDef(\spec100_dnoise_scheduled, {
    var trig = Impulse.ar(750);
    var ring;
    var probe;
    RandID.ir(0);
    RandSeed.ar(trig, seed);
    ring = Demand.ar(
        trig,
        0,
        DNoiseRing(
            Dseq([0, 1, 1, 0], inf),
            Dseq([1, 0, 1, 0], inf),
            Dseq([1, 2, 3, 0], inf),
            4,
            13
        )
    );
    probe = Demand.ar(trig, 0, Dwhite(0, 1, inf));
    Out.ar(\out.ir(7), [ring, probe]);
});

stochasticDef = SynthDef(\spec100_dnoise_stochastic, {
    var trig = Impulse.ar(750);
    var ring;
    RandID.ir(0);
    RandSeed.ir(1, seed);
    ring = Demand.ar(trig, 0, DNoiseRing(0.5, 0.5, 1, 4, 9));
    Out.ar(\out.ir(9), ring);
});

score = Score([
    [0.0, [\d_recv, baselineDef.asBytes]],
    [0.0, [\d_recv, changeZeroDef.asBytes]],
    [0.0, [\d_recv, changeOneDef.asBytes]],
    [0.0, [\d_recv, scheduledDef.asBytes]],
    [0.0, [\d_recv, stochasticDef.asBytes]],
    [0.0, [\s_new, \spec100_dnoise_baseline, 1000, 0, 1]],
    [0.0, [\s_new, \spec100_dnoise_change_zero, 1001, 0, 1]],
    [0.0, [\s_new, \spec100_dnoise_change_one, 1002, 0, 1]],
    [0.0, [\s_new, \spec100_dnoise_scheduled, 1003, 0, 1]],
    [0.0, [\s_new, \spec100_dnoise_stochastic, 1004, 0, 1]],
    [duration, [\c_set, 0, 0]]
]);

options = ServerOptions.new
    .numOutputBusChannels_(10)
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
