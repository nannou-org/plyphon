#!/bin/sh
set -eu

fixture_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
sc3_source=${SC3_PLUGINS_SOURCE:-/tmp/spec100-review.1gRi2n/sc3-plugins}
sc_source=${SUPERCOLLIDER_SOURCE:-/tmp/spec100-review.1gRi2n/supercollider-3.14.1}
sclang=${SCLANG:-/Applications/SuperCollider.app/Contents/MacOS/sclang}
scsynth=${SCSYNTH:-/Applications/SuperCollider.app/Contents/Resources/scsynth}
capture_scope=${SPEC100_CAPTURE_SCOPE:-all}

case "$capture_scope" in
    all | pv) ;;
    *)
        echo "SPEC100_CAPTURE_SCOPE must be 'all' or 'pv'" >&2
        exit 2
        ;;
esac

expected_sc3=66047341f83e25cbaf3b106f35bd1174a3bbee7c
expected_sc=426edf6d8742e1cc3bd85b51ca0c4e595d37a903

test "$(git -C "$sc3_source" rev-parse HEAD)" = "$expected_sc3"
test "$(git -C "$sc_source" rev-parse HEAD)" = "$expected_sc"
"$sclang" -v | grep "426edf6" >/dev/null
"$scsynth" -v | grep "426edf6" >/dev/null

build_dir=$(mktemp -d "${TMPDIR:-/tmp}/spec100-sc3-build.XXXXXX")
core_build_dir=$(mktemp -d "${TMPDIR:-/tmp}/spec100-sc-build.XXXXXX")
runtime_dir=$(mktemp -d "${TMPDIR:-/tmp}/spec100-sc3-runtime.XXXXXX")
cleanup() {
    rm -rf "$build_dir" "$core_build_dir" "$runtime_dir"
}
trap cleanup EXIT HUP INT TERM

git -C "$sc3_source" submodule update --init external_libraries/nova-simd external_libraries/stk
git -C "$sc_source" submodule update --init external_libraries/boost external_libraries/nova-simd

cmake -S "$sc_source" -B "$core_build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DSC_QT=OFF \
    -DSC_IDE=OFF \
    -DSUPERNOVA=OFF \
    -DENABLE_TESTSUITE=OFF \
    -DSC_ABLETON_LINK=OFF \
    -DSC_HIDAPI=OFF \
    -DNO_X11=ON \
    -DNO_LIBSNDFILE=ON \
    -DNATIVE=OFF \
    -DUSE_CCACHE=OFF \
    -DSCLANG_SERVER=OFF
cmake --build "$core_build_dir" \
    --target \
        BinaryOpUGens \
        DelayUGens \
        DemandUGens \
        FFT_UGens \
        IOUGens \
        LFUGens \
        MulAddUGens \
        NoiseUGens \
        OscUGens \
        TriggerUGens \
        UnaryOpUGens \
        UnpackFFTUGens \
    --parallel 8

cmake -S "$sc3_source" -B "$build_dir" -G Ninja \
    -DSC_PATH="$sc_source" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSUPERNOVA=OFF \
    -DNOVA_SIMD=OFF \
    -DNOVA_DISK_IO=OFF \
    -DAY=OFF \
    -DLADSPA=OFF \
    -DHOA_UGENS=OFF \
    -DIN_PLACE_BUILD=ON \
    -DUSE_CCACHE=OFF
cmake --build "$build_dir" \
    --target \
        AntiAliasingOscillators \
        BhobFFT \
        BhobFilt \
        BlackrainUGens \
        DistortionUGens \
        JoshPVUGens \
        JoshUGens \
        MCLDChaosUGens \
        MCLDFFTUGens \
        NoiseRing \
        SLUGens \
        TJUGens \
    --parallel 8

mkdir -p "$runtime_dir/plugins" "$runtime_dir/classes/Spec100Extensions"
cp -R "$sc_source/SCClassLibrary/." "$runtime_dir/classes/"

cp "$sc3_source/source/DistortionUGens/sc/DistortionPlugins.sc" \
    "$runtime_dir/classes/Spec100Extensions/"
cp "$sc3_source/source/BlackrainUGens/sc/blackrain_ugens.sc" \
    "$runtime_dir/classes/Spec100Extensions/"
cp "$sc3_source/source/MCLDUGens/sc/MCLDChaosUGens.sc" \
    "$runtime_dir/classes/Spec100Extensions/"
cp "$sc3_source/source/JoshUGens/sc/classes/JoshPV.sc" \
    "$runtime_dir/classes/Spec100Extensions/"
cp "$sc3_source/source/SLUGens/sc/classes/SLUGens.sc" \
    "$runtime_dir/classes/Spec100Extensions/"
cp "$sc3_source/source/TJUGens/sc/TJUGens.sc" \
    "$runtime_dir/classes/Spec100Extensions/"
cp "$sc3_source/source/BhobUGens/sc/classes/bhobGens.sc" \
    "$runtime_dir/classes/Spec100Extensions/"
cp "$sc3_source/source/BhobUGens/sc/classes/bhobFFT.sc" \
    "$runtime_dir/classes/Spec100Extensions/"
cp "$sc3_source/source/JoshUGens/sc/classes/MoogVCF.sc" \
    "$runtime_dir/classes/Spec100Extensions/"
cp "$sc3_source/source/AntiAliasingOscillators/sc/Classes/AntiAliasingOscillators.sc" \
    "$runtime_dir/classes/Spec100Extensions/"
cp "$sc3_source/source/ChaosUGens/sc/NoiseRing.sc" \
    "$runtime_dir/classes/Spec100Extensions/"
cp "$sc3_source/source/MCLDUGens/sc/MCLDFFTUGens.sc" \
    "$runtime_dir/classes/Spec100Extensions/"

for plugin in \
    BinaryOpUGens.scx \
    DemandUGens.scx \
    DelayUGens.scx \
    FFT_UGens.scx \
    IOUGens.scx \
    LFUGens.scx \
    MulAddUGens.scx \
    NoiseUGens.scx \
    OscUGens.scx \
    TriggerUGens.scx \
    UnaryOpUGens.scx \
    UnpackFFTUGens.scx
do
    cp "$core_build_dir/server/plugins/$plugin" "$runtime_dir/plugins/"
done

for plugin in \
    AntiAliasingOscillators \
    BhobFFT \
    BhobFilt \
    BlackrainUGens \
    DistortionUGens \
    JoshPVUGens \
    JoshUGens \
    MCLDChaosUGens \
    MCLDFFTUGens \
    NoiseRing \
    SLUGens \
    TJUGens
do
    cp "$build_dir/source/$plugin.scx" "$runtime_dir/plugins/"
done

run_sclang() {
    "$sclang" -a --include-path "$runtime_dir/classes" "$@"
}

python3 "$fixture_dir/generate_pv_inputs.py"

if [ "$capture_scope" = all ]; then
    python3 "$fixture_dir/generate_dfm1_input.py"
    run_sclang "$fixture_dir/decimator.scd" \
        "$fixture_dir/decimator.f32" "$runtime_dir/plugins" "$scsynth"
    run_sclang "$fixture_dir/bmoog.scd" \
        "$fixture_dir/bmoog.f32" "$runtime_dir/plugins" "$scsynth"
    run_sclang "$fixture_dir/perlin3.scd" \
        "$fixture_dir/perlin3_ar.f32" ar "$runtime_dir/plugins" "$scsynth"
    run_sclang "$fixture_dir/perlin3.scd" \
        "$fixture_dir/perlin3_kr.f32" kr "$runtime_dir/plugins" "$scsynth"
    run_sclang "$fixture_dir/rossler_l.scd" \
        "$fixture_dir/rossler_l.f32" "$runtime_dir/plugins" "$scsynth"
    run_sclang "$fixture_dir/env_detect.scd" \
        "$fixture_dir/env_detect.f32" "$runtime_dir/plugins" "$scsynth"
    run_sclang "$fixture_dir/dfm1.scd" \
        "$fixture_dir/dfm1_stability_1.f32" "$fixture_dir/dfm1_source.f32" \
        "$runtime_dir/plugins" "$scsynth"
    sleep 1
    run_sclang "$fixture_dir/dfm1.scd" \
        "$fixture_dir/dfm1_stability_2.f32" "$fixture_dir/dfm1_source.f32" \
        "$runtime_dir/plugins" "$scsynth"
    sleep 1
    run_sclang "$fixture_dir/dfm1.scd" \
        "$fixture_dir/dfm1_stability_3.f32" "$fixture_dir/dfm1_source.f32" \
        "$runtime_dir/plugins" "$scsynth"
    sleep 1
    run_sclang "$fixture_dir/dfm1.scd" \
        "$fixture_dir/dfm1.f32" "$fixture_dir/dfm1_source.f32" \
        "$runtime_dir/plugins" "$scsynth"
    run_sclang "$fixture_dir/moog_ladder.scd" \
        "$fixture_dir/moog_ladder_ar.f32" ar "$runtime_dir/plugins" "$scsynth"
    run_sclang "$fixture_dir/moog_ladder.scd" \
        "$fixture_dir/moog_ladder_kr.f32" kr "$runtime_dir/plugins" "$scsynth"
    run_sclang "$fixture_dir/moog_vcf.scd" \
        "$fixture_dir/moog_vcf.f32" "$runtime_dir/plugins" "$scsynth"
    run_sclang "$fixture_dir/blit_b3.scd" \
        "$fixture_dir/blit_b3.f32" "$runtime_dir/plugins" "$scsynth"
    run_sclang "$fixture_dir/dnoise_ring.scd" \
        "$fixture_dir/dnoise_ring.f32" "$runtime_dir/plugins" "$scsynth"
fi
run_sclang "$fixture_dir/pv_freeze.scd" \
    "$fixture_dir/pv_freeze.f32" "$fixture_dir/pv_source_a.f32" \
    "$runtime_dir/plugins" "$scsynth"
run_sclang "$fixture_dir/pv_mag_smooth.scd" \
    "$fixture_dir/pv_mag_smooth.f32" "$fixture_dir/pv_source_a.f32" \
    "$runtime_dir/plugins" "$scsynth"
run_sclang "$fixture_dir/pv_morph.scd" \
    "$fixture_dir/pv_morph.f32" "$fixture_dir/pv_source_a.f32" \
    "$fixture_dir/pv_source_b.f32" "$runtime_dir/plugins" "$scsynth"
run_sclang "$fixture_dir/pv_morph_source_phases.scd" \
    "$fixture_dir/pv_morph_source_phases.f32" "$fixture_dir/pv_source_a.f32" \
    "$fixture_dir/pv_source_b.f32" "$runtime_dir/plugins" "$scsynth"

python3 "$fixture_dir/generate_manifest.py" \
    "$runtime_dir/plugins" "$build_dir" "$core_build_dir" "$runtime_dir" \
    "$sc3_source" "$sc_source" "$sclang" "$scsynth"
python3 "$fixture_dir/verify.py"
