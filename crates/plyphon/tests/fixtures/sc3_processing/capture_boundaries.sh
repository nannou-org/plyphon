#!/bin/sh
set -eu

fixture_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
sc3_source=${SC3_PLUGINS_SOURCE:-/tmp/spec100-review.1gRi2n/sc3-plugins}
sc_source=${SUPERCOLLIDER_SOURCE:-/tmp/spec100-review.1gRi2n/supercollider-3.14.1}
sclang=${SCLANG:-/Applications/SuperCollider.app/Contents/MacOS/sclang}
scsynth=${SCSYNTH:-/Applications/SuperCollider.app/Contents/Resources/scsynth}

expected_sc3=66047341f83e25cbaf3b106f35bd1174a3bbee7c
expected_sc=426edf6d8742e1cc3bd85b51ca0c4e595d37a903

test "$(git -C "$sc3_source" rev-parse HEAD)" = "$expected_sc3"
test "$(git -C "$sc_source" rev-parse HEAD)" = "$expected_sc"
"$sclang" -v | grep "426edf6" >/dev/null
"$scsynth" -v | grep "426edf6" >/dev/null

build_dir=$(mktemp -d "${TMPDIR:-/tmp}/spec100-boundary-sc3-build.XXXXXX")
core_build_dir=$(mktemp -d "${TMPDIR:-/tmp}/spec100-boundary-sc-build.XXXXXX")
runtime_dir=$(mktemp -d "${TMPDIR:-/tmp}/spec100-boundary-runtime.XXXXXX")
cleanup() {
    rm -rf "$build_dir" "$core_build_dir" "$runtime_dir"
}
trap cleanup EXIT HUP INT TERM

git -C "$sc3_source" submodule update --init external_libraries/nova-simd \
    external_libraries/stk >/dev/null 2>&1
git -C "$sc_source" submodule update --init external_libraries/boost \
    external_libraries/nova-simd >/dev/null 2>&1

if ! cmake -S "$sc_source" -B "$core_build_dir" -G Ninja \
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
    -DSCLANG_SERVER=OFF >/dev/null 2>&1
then
    echo "pinned SuperCollider configure failed; diagnostics suppressed" >&2
    exit 1
fi
if ! cmake --build "$core_build_dir" \
    --target \
        BinaryOpUGens \
        DelayUGens \
        DemandUGens \
        FFT_UGens \
        IOUGens \
        LFUGens \
        MulAddUGens \
        TriggerUGens \
        UnpackFFTUGens \
    --parallel 8 >/dev/null 2>&1
then
    echo "pinned SuperCollider plugin build failed; diagnostics suppressed" >&2
    exit 1
fi

if ! cmake -S "$sc3_source" -B "$build_dir" -G Ninja \
    -DSC_PATH="$sc_source" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSUPERNOVA=OFF \
    -DNOVA_SIMD=OFF \
    -DNOVA_DISK_IO=OFF \
    -DAY=OFF \
    -DLADSPA=OFF \
    -DHOA_UGENS=OFF \
    -DIN_PLACE_BUILD=ON \
    -DUSE_CCACHE=OFF >/dev/null 2>&1
then
    echo "pinned sc3-plugins configure failed; diagnostics suppressed" >&2
    exit 1
fi
if ! cmake --build "$build_dir" \
    --target \
        BlackrainUGens \
        DistortionUGens \
        JoshPVUGens \
        MCLDChaosUGens \
    --parallel 8 >/dev/null 2>&1
then
    echo "pinned sc3-plugins build failed; diagnostics suppressed" >&2
    exit 1
fi

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

for plugin in \
    BinaryOpUGens.scx \
    DelayUGens.scx \
    DemandUGens.scx \
    FFT_UGens.scx \
    IOUGens.scx \
    LFUGens.scx \
    MulAddUGens.scx \
    TriggerUGens.scx \
    UnpackFFTUGens.scx
do
    cp "$core_build_dir/server/plugins/$plugin" "$runtime_dir/plugins/"
done

for plugin in \
    BlackrainUGens \
    DistortionUGens \
    JoshPVUGens \
    MCLDChaosUGens
do
    cp "$build_dir/source/$plugin.scx" "$runtime_dir/plugins/"
done

run_sclang() {
    "$sclang" -a --include-path "$runtime_dir/classes" "$@"
}

run_sclang "$fixture_dir/decimator_boundaries.scd" \
    "$fixture_dir/decimator_boundaries.f32" "$runtime_dir/plugins" "$scsynth"
run_sclang "$fixture_dir/decimator_controls.scd" \
    "$fixture_dir/decimator_controls.f32" "$runtime_dir/plugins" "$scsynth"
for bmoog_case in 0 1 2 3 4 5 6 7 8
do
    run_sclang "$fixture_dir/bmoog_boundaries.scd" \
        "$fixture_dir/bmoog_boundaries_${bmoog_case}.f32" "$bmoog_case" \
        "$runtime_dir/plugins" "$scsynth"
done
run_sclang "$fixture_dir/bmoog_controls.scd" \
    "$fixture_dir/bmoog_controls.f32" "$runtime_dir/plugins" "$scsynth"
run_sclang "$fixture_dir/perlin3_boundaries.scd" \
    "$fixture_dir/perlin3_boundaries.f32" "$runtime_dir/plugins" "$scsynth"
run_sclang "$fixture_dir/rossler_l_boundaries.scd" \
    "$fixture_dir/rossler_l_boundaries.f32" "$runtime_dir/plugins" "$scsynth"
run_sclang "$fixture_dir/rossler_l_controls.scd" \
    "$fixture_dir/rossler_l_controls.f32" "$runtime_dir/plugins" "$scsynth"
run_sclang "$fixture_dir/pv_freeze_early.scd" \
    "$fixture_dir/pv_freeze_early.f32" "$fixture_dir/pv_source_a.f32" \
    "$runtime_dir/plugins" "$scsynth"
run_sclang "$fixture_dir/pv_freeze_controls.scd" \
    "$fixture_dir/pv_freeze_controls.f32" "$fixture_dir/pv_source_a.f32" \
    "$runtime_dir/plugins" "$scsynth"
run_sclang "$fixture_dir/signed_zero_boundaries.scd" \
    "$fixture_dir/signed_zero_boundaries.f32" "$fixture_dir/pv_source_a.f32" \
    "$runtime_dir/plugins" "$scsynth"
python3 "$fixture_dir/generate_boundary_manifest.py" \
    "$runtime_dir/plugins" "$build_dir" "$core_build_dir" "$runtime_dir" \
    "$sc3_source" "$sc_source" "$sclang" "$scsynth"
python3 "$fixture_dir/verify_boundaries.py"
