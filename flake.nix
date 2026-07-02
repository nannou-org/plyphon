{
  description = "plyphon: a pure-Rust rewrite of SuperCollider's scsynth audio engine core.";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    rust-overlay = {
      inputs.nixpkgs.follows = "nixpkgs";
      url = "github:oxalica/rust-overlay";
    };
    systems.url = "github:nix-systems/default";
  };

  outputs =
    inputs:
    let
      overlays = [
        inputs.rust-overlay.overlays.default
        inputs.self.overlays.default
      ];
      perSystemPkgs =
        f:
        inputs.nixpkgs.lib.genAttrs (import inputs.systems) (
          system: f (import inputs.nixpkgs { inherit overlays system; })
        );
    in
    {
      overlays.default = final: prev: {
        # A pinned stable Rust toolchain with the wasm target (used by the dev shell, including the
        # wasm32 `no_std` build check).
        rustToolchain = final.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" ];
          targets = [ "wasm32-unknown-unknown" ];
        };
        # A pinned *nightly* toolchain for the AudioWorklet web build, which needs WASM atomics +
        # `-Z build-std` (both nightly-only). `selectLatestNightlyWith` pins the choice to the
        # locked rust-overlay input; `rust-src` lets build-std recompile `std` with atomics.
        rustToolchainWasmNightly = final.rust-bin.selectLatestNightlyWith (
          toolchain:
          toolchain.default.override {
            extensions = [ "rust-src" ];
            targets = [ "wasm32-unknown-unknown" ];
          }
        );
        # Pin wasm-bindgen-cli to the exact version in Cargo.lock so trunk uses it instead of
        # trying to download a matching release.
        wasm-bindgen-cli = prev.callPackage ./pkgs/wasm-bindgen-cli.nix { };
        # The `plyphon` CLI binary (the default package).
        plyphon = final.callPackage ./pkgs/plyphon.nix { };
        # The web demo: every example built for cpal's AudioWorklet backend (nightly + WASM
        # threads), plus a helper to build and serve the whole site locally.
        plyphon-website = final.callPackage ./pkgs/plyphon-website.nix { };
        serve-plyphon-website = final.callPackage ./pkgs/serve-plyphon-website.nix { };
      };

      packages = perSystemPkgs (pkgs: {
        plyphon = pkgs.plyphon;
        plyphon-website = pkgs.plyphon-website;
        serve-plyphon-website = pkgs.serve-plyphon-website;
        default = pkgs.plyphon;
      });

      devShells = perSystemPkgs (pkgs: {
        plyphon-dev = pkgs.callPackage ./shell.nix { };
        # Nightly + WASM-threads shell for local AudioWorklet builds (`trunk serve web/<name>.html`).
        plyphon-web = pkgs.callPackage ./shell-web.nix { };
        default = inputs.self.devShells.${pkgs.stdenv.hostPlatform.system}.plyphon-dev;
      });

      formatter = perSystemPkgs (pkgs: pkgs.nixfmt-tree);
    };
}
