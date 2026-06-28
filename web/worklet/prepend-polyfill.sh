#!/usr/bin/env bash
# Trunk post-build hook (see Trunk.worklet.toml): prepend the TextEncoder/TextDecoder polyfill to
# the wasm-bindgen JS glue so it can load on the AudioWorklet thread (whose scope lacks them). The
# glue is identified by its `initSync` export; a marker keeps the hook idempotent.
set -euo pipefail

dir="${TRUNK_STAGING_DIR:-${TRUNK_DIST_DIR:?TRUNK_STAGING_DIR/TRUNK_DIST_DIR not set}}"
poly="$(dirname "$0")/textcodec-polyfill.js"

for js in "$dir"/*.js; do
  [ -f "$js" ] || continue
  grep -q "initSync" "$js" || continue
  grep -q "PLYPHON_TEXTCODEC_POLYFILL" "$js" && continue
  cat "$poly" "$js" >"$js.tmp"
  mv "$js.tmp" "$js"
  echo "prepended TextEncoder/TextDecoder polyfill to $(basename "$js")"
done
