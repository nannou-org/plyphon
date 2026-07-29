(
var args = thisProcess.argv;
var outputPath = args.at(0);
var pluginPath = args.at(1);
var scsynthPath = args.at(2);
var frames = 128;
var sampleRate = 48000;
var duration = frames / sampleRate;
var options;
var def;
var score;

if(outputPath.isNil or: { pluginPath.isNil } or: { scsynthPath.isNil }) {
    "usage: perlin3_boundaries.scd OUTPUT_F32 TEMP_PLUGIN_PATH SCSYNTH".postln;
    2.exit;
};

def = SynthDef(\spec100_perlin3_boundaries, {
    var y = 0.375;
    var z = -0.625;
    var base = [-0.25, 0.375, -0.625];
    var perlin = { |x, yCoordinate, zCoordinate|
        Perlin3.ar(K2A.ar(x), K2A.ar(yCoordinate), K2A.ar(zCoordinate))
    };
    var outputs = [
        perlin.value(-256.0009765625, y, z),
        perlin.value(-256, y, z),
        perlin.value(-255.9990234375, y, z),
        perlin.value(-0.0009765625, y, z),
        perlin.value(0, y, z),
        perlin.value(0.0009765625, y, z),
        perlin.value(255.9990234375, y, z),
        perlin.value(256, y, z),
        perlin.value(256.0009765625, y, z),
        perlin.value(base.at(0), base.at(1), base.at(2)),
        perlin.value(base.at(0) + 256, base.at(1), base.at(2)),
        perlin.value(base.at(0), base.at(1) + 256, base.at(2)),
        perlin.value(base.at(0), base.at(1), base.at(2) + 256),
        perlin.value(base.at(0) + 256, base.at(1) + 256, base.at(2) + 256)
    ];
    Out.ar(0, outputs);
});

score = Score([
    [0.0, [\d_recv, def.asBytes]],
    [0.0, [\s_new, \spec100_perlin3_boundaries, 1000, 0, 1]],
    [duration, [\n_free, 1000]]
]);

options = ServerOptions.new
    .numOutputBusChannels_(14)
    .numInputBusChannels_(0)
    .blockSize_(64)
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
