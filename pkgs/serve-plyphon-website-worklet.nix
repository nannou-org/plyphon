# Short-hand for serving the AudioWorklet site locally with the COOP/COEP headers SharedArrayBuffer
# needs. With real headers the bundled coi-serviceworker is a no-op; it only matters on hosts that
# can't send headers (e.g. GitHub Pages).
{
  writeShellScriptBin,
  plyphon-website-worklet,
  miniserve,
}:
writeShellScriptBin "serve-plyphon-website-worklet" ''
  ${miniserve}/bin/miniserve \
    --index index.html \
    --disable-indexing \
    --hide-version-footer \
    --hide-theme-selector \
    --header "Cross-Origin-Opener-Policy:same-origin" \
    --header "Cross-Origin-Embedder-Policy:require-corp" \
    --header "Cache-Control:no-store, no-cache, must-revalidate" \
    --header "Pragma:no-cache" \
    --header "Expires:0" \
    -i 0.0.0.0 \
    --port 8089 \
    ${plyphon-website-worklet}
''
