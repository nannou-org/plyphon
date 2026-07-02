#!/usr/bin/env bash
# Generate the AudioWorklet web pages (web/worklet/*.html) from the default pages (web/*.html).
#
# The worklet pages are pure derivations of the default ones - same per-example content - plus:
#   - the coi-serviceworker (cross-origin isolation for SharedArrayBuffer on header-less hosts),
#   - the `audioworklet` cargo feature on the rust asset,
#   - the WASM-threads/SIMD wasm-opt flags (--enable-simd/--enable-threads/--enable-mutable-globals),
#   - one extra `../` on asset/crate paths, since these live a directory deeper.
#
# Re-run this (from anywhere) after editing any web/*.html, and commit the result.
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"

# The example set is every web/*.html page (except the landing page) - the same single source of
# truth the site build loops derive from, so this list can never drift from what ships.
mkdir -p web/worklet

names=()
for page in web/*.html; do
  name="$(basename "$page" .html)"
  [ "$name" = index ] && continue
  names+=("$name")
  sed \
    -e 's|href="../crates/|href="../../crates/|g' \
    -e 's|rel="copy-file" href="audio.js"|rel="copy-file" href="../audio.js"|' \
    -e 's|rel="copy-file" href="style.css"|rel="copy-file" href="../style.css"|' \
    -e 's|data-initializer="initializer.js"|data-initializer="../initializer.js"|' \
    -e 's|data-wasm-opt-params="--enable-bulk-memory|data-wasm-opt-params="--enable-simd --enable-threads --enable-bulk-memory --enable-mutable-globals|' \
    -e 's|<script src="audio.js"></script>|<script src="coi-serviceworker.min.js"></script>\n    <script src="audio.js"></script>|' \
    -e 's|\( *\)<link data-trunk rel="copy-file" href="../audio.js" />|\1<link data-trunk rel="copy-file" href="../coi-serviceworker.min.js" />\n\1<link data-trunk rel="copy-file" href="../audio.js" />|' \
    -e 's|^\( *\)\(data-bin="[^"]*"\)$|\1\2\n\1data-cargo-features="audioworklet"|' \
    "web/$name.html" >"web/worklet/$name.html"
done

# The landing page links to the per-example dirs (relative) and needs no per-page changes; the
# AudioWorklet backend is an implementation detail, so it is used verbatim.
cp web/index.html web/worklet/index.html

echo "generated web/worklet/{$(IFS=,; echo "${names[*]}"),index}.html"
