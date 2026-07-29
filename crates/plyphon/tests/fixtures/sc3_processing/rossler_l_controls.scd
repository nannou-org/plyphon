(
var args = thisProcess.argv;
var outputPath = args.at(0);
var pluginPath = args.at(1);
var scsynthPath = args.at(2);
var frames = 512;
var sampleRate = 48000;
var blockSize = 64;
var duration = frames / sampleRate;
var nan = 0.0 / 0.0;
var positiveInfinity = 1.0 / 0.0;
var negativeInfinity = -1.0 / 0.0;
var invalidValues = [nan, positiveInfinity, negativeInfinity];
var defaults = [6000, 0.2, 0.2, 5.7, 0.05, 0.1, 0, 0];
var controlNames = (0..23).collect({ |index| ("control" ++ index).asSymbol });
var invalidMessage = [\n_set, 1000];
var recoveryMessage = [\n_set, 1000];
var options;
var def;
var score;

controlNames.do({ |name, index|
    invalidMessage = invalidMessage ++ [name, invalidValues.at(index div: 8)];
    recoveryMessage = recoveryMessage ++ [name, defaults.at(index % 8)];
});
invalidMessage = invalidMessage ++ [
    \freqHighReference, 48000,
    \freqFloorReference, 0.001
];
recoveryMessage = recoveryMessage ++ [
    \freqHighReference, 6000,
    \freqFloorReference, 6000
];

if(outputPath.isNil or: { pluginPath.isNil } or: { scsynthPath.isNil }) {
    "usage: rossler_l_controls.scd OUTPUT_F32 TEMP_PLUGIN_PATH SCSYNTH".postln;
    2.exit;
};

def = SynthDef(\spec100_rossler_l_controls, {
    var outputs = controlNames.collect({ |name, index|
        var controlIndex = index % 8;
        var inputs = defaults.copy;
        inputs.put(controlIndex, NamedControl.kr(name, defaults.at(controlIndex)));
        RosslerL.ar(*inputs)
    }).flat;
    outputs = outputs ++ [
        RosslerL.ar(\freqHighReference.kr(6000), 0.2, 0.2, 5.7, 0.05, 0.1, 0, 0),
        RosslerL.ar(\freqFloorReference.kr(6000), 0.2, 0.2, 5.7, 0.05, 0.1, 0, 0)
    ].flat;
    Out.ar(0, outputs);
});

score = Score([
    [0.0, [\d_recv, def.asBytes]],
    [0.0, [\s_new, \spec100_rossler_l_controls, 1000, 0, 1]],
    [(2 * blockSize) / sampleRate, invalidMessage],
    [(4 * blockSize) / sampleRate, recoveryMessage],
    [duration, [\n_free, 1000]]
]);

options = ServerOptions.new
    .numOutputBusChannels_(78)
    .numInputBusChannels_(0)
    .numWireBufs_(2048)
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
