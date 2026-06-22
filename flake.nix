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
        # A pinned Rust toolchain with the wasm target for the web demo.
        rustToolchain = final.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" ];
          targets = [ "wasm32-unknown-unknown" ];
        };
        # A `buildRustPackage` platform using the pinned toolchain, used by the website build to
        # cross-compile to `wasm32-unknown-unknown` with our toolchain rather than nixpkgs' rustc.
        rustPlatformWasm = final.makeRustPlatform {
          cargo = final.rustToolchain;
          rustc = final.rustToolchain;
        };
        # Pin wasm-bindgen-cli to the exact version in Cargo.lock so trunk uses it instead of
        # trying to download a matching release.
        wasm-bindgen-cli = prev.callPackage ./pkgs/wasm-bindgen-cli.nix { };
        plyphon-website = final.callPackage ./pkgs/plyphon-website.nix { };
        serve-plyphon-website = final.callPackage ./pkgs/serve-plyphon-website.nix { };
      };

      packages = perSystemPkgs (pkgs: {
        plyphon-website = pkgs.plyphon-website;
        serve-plyphon-website = pkgs.serve-plyphon-website;
        default = pkgs.plyphon-website;
      });

      devShells = perSystemPkgs (pkgs: {
        plyphon-dev = pkgs.callPackage ./shell.nix { };
        default = inputs.self.devShells.${pkgs.stdenv.hostPlatform.system}.plyphon-dev;
      });

      formatter = perSystemPkgs (pkgs: pkgs.nixfmt-tree);
    };
}
