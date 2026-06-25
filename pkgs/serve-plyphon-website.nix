# Short-hand for serving `plyphon-website` locally with the COOP/COEP headers a future
# AudioWorklet/SharedArrayBuffer backend needs.
{
  writeShellScriptBin,
  plyphon-website,
  miniserve,
}:
writeShellScriptBin "serve-plyphon-website" ''
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
    --port 8088 \
    ${plyphon-website}
''
