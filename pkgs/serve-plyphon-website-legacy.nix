# Short-hand for serving the legacy/fallback `plyphon-website-legacy` (the ScriptProcessor build)
# locally. The COOP/COEP headers aren't needed for that backend, but are harmless.
{
  writeShellScriptBin,
  plyphon-website-legacy,
  miniserve,
}:
writeShellScriptBin "serve-plyphon-website-legacy" ''
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
    ${plyphon-website-legacy}
''
