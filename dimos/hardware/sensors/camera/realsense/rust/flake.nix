{
  description = "RealSense camera native module for dimos";

  inputs = {
    nix-filter.url = "github:numtide/nix-filter";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    crate2nix.url = "github:nix-community/crate2nix";
    crate2nix.inputs.nixpkgs.follows = "nixpkgs";
  };

  # Linux only: librealsense pulls v4l-utils, which nixpkgs will not evaluate on
  # darwin. Declaring darwin anyway is what main did, and `nix develop` there fails.
  outputs = { self, nix-filter, nixpkgs, flake-utils, crate2nix }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" ] (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        name = "dimos-realsense";

        src = nix-filter.lib { root = ./.; exclude = [ "target" "build" "result" "__pycache__" ]; };

        generated = crate2nix.tools.${system}.generatedCargoNix { inherit name src; };

        # Both crates run pkg-config against librealsense2: realsense-sys to generate
        # its bindings, and this module's own build.rs to turn the result into an
        # rpath. crate2nix builds each crate in its own sandbox, so the inputs have to
        # be declared for each of them.
        needsLibrealsense = _: {
          buildInputs = [ pkgs.librealsense ];
          nativeBuildInputs = [ pkgs.pkg-config ];
        };
        sysOverrides = {
          realsense-sys = needsLibrealsense;
          dimos-realsense = needsLibrealsense;
        };

        ours = [ name "dimos-module" "dimos-module-macros" ];
        callWith = lint: import generated {
          inherit pkgs;
          buildRustCrateForPkgs = cratePkgs:
            let build = cratePkgs.buildRustCrate.override {
                  defaultCrateOverrides = cratePkgs.defaultCrateOverrides // sysOverrides;
                };
            in crate: build (crate // pkgs.lib.optionalAttrs
              (lint && builtins.elem crate.crateName ours)
              {
                useClippy = true;
                capLints = "forbid";
                # Lint unoptimised. These first-party crates are the only ones
                # rebuilt for the check, and at the package's opt-level 3 + LTO
                # that recompile is most of its cost. Dependencies are untouched,
                # so nothing stops being shared.
                release = false;
                extraRustcOpts = (crate.extraRustcOpts or [ ]) ++ [ "-D" "warnings" ];
              });
        };
        buildOf = called:
          if called ? rootCrate then called.rootCrate.build
          else called.workspaceMembers.${name}.build;
      in {
        packages.default = buildOf (callWith false);
        packages.${name} = self.packages.${system}.default;
        packages.clippy = (buildOf (callWith true)).override {
          runTests = true;
          testCrateFlags = [ "--list" ];
        };
        checks.clippy = self.packages.${system}.clippy;

        devShells.default = pkgs.mkShell {
          packages = [ pkgs.cargo pkgs.rustc pkgs.clippy pkgs.rustfmt
                       pkgs.librealsense pkgs.pkg-config ];
        };
      });
}
