(
var args = thisProcess.argv;
var outputPath = args.at(0);
var variant = args.at(1);
var pluginPath = args.at(2);
var scsynthPath = args.at(3);
var frames = 512;
var sampleRate = 48000;
var blockSize = 64;
var duration = frames / sampleRate;
var options;
var def;
var score;

if(outputPath.isNil or: { variant.isNil } or: { pluginPath.isNil } or: { scsynthPath.isNil }) {
    "usage: perlin3.scd OUTPUT_F32 ar|kr TEMP_PLUGIN_PATH SCSYNTH".postln;
    2.exit;
};

if((variant != "ar") and: { variant != "kr" }) {
    "perlin3 variant must be ar or kr".postln;
    2.exit;
};

def = SynthDef(\spec100_perlin3_oracle, {
    var x;
    var y;
    var z;
    var outputs;
    if(variant == "ar") {
        x = Phasor.ar(0, 0.0078125, -1.25, 1.25, -1.25);
        y = Phasor.ar(0, 0.01171875, -0.75, 1.75, 0.125);
        z = Phasor.ar(0, 0.015625, -1.5, 1.5, 0.5);
        outputs = [
            Perlin3.ar(x, y, z),
            Perlin3.ar(x + 0.25, y - 0.5, z + 0.75),
            Perlin3.ar(x.neg, y * 0.5, z * 1.25)
        ];
    } {
        x = Phasor.kr(0, 0.125, -1.25, 1.25, -1.25);
        y = Phasor.kr(0, 0.1875, -0.75, 1.75, 0.125);
        z = Phasor.kr(0, 0.25, -1.5, 1.5, 0.5);
        outputs = K2A.ar([
            Perlin3.kr(x, y, z),
            Perlin3.kr(x + 0.25, y - 0.5, z + 0.75),
            Perlin3.kr(x.neg, y * 0.5, z * 1.25)
        ]);
    };
    Out.ar(0, outputs);
});

score = Score([
    [0.0, [\d_recv, def.asBytes]],
    [0.0, [\s_new, \spec100_perlin3_oracle, 1000, 0, 1]],
    [duration, [\n_free, 1000]]
]);

options = ServerOptions.new
    .numOutputBusChannels_(3)
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

