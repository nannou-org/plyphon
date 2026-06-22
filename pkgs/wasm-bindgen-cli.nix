# wasm-bindgen-cli version must match the exact version of wasm-bindgen used within the crate
# dependencies. nixpkgs' version doesn't always match the version picked up in our Cargo.lock, so
# here we pin to a particular wasm-bindgen-cli (matching Cargo.lock) so trunk doesn't try to
# download it. After generating Cargo.lock, confirm with `grep -A1 'name = "wasm-bindgen"'
# Cargo.lock` and update `version` + both hashes (set to `lib.fakeHash` and read the `got:` value
# from the first `nix build .#plyphon-website` failure).
{
  buildWasmBindgenCli,
  fetchCrate,
  rustPlatform,
}:
buildWasmBindgenCli rec {
  src = fetchCrate {
    pname = "wasm-bindgen-cli";
    version = "0.2.125";
    hash = "sha256-zRawtjxMOdTMX+mZaiNR3YYfTiZJhf9qj7kXSSeMxrc=";
  };
  cargoDeps = rustPlatform.fetchCargoVendor {
    inherit src;
    inherit (src) pname version;
    hash = "sha256-aZCfgR23Qb0Pn4Mm4ToMtuuRQqSJjXCR9li/VvP5CTM=";
  };
}
