{
  description = "RealSense camera native module for dimos";

  inputs = {
    nix-filter.url = "github:numtide/nix-filter";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    crate2nix.url = "github:nix-community/crate2nix";
    crate2nix.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nix-filter, nixpkgs, flake-utils, crate2nix }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" ] (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        name = "dimos-realsense";

        src = nix-filter.lib { root = ./.; exclude = [ "target" "build" "result" "__pycache__" ]; };

        generated = crate2nix.tools.${system}.generatedCargoNix { inherit name src; };

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
                release = false;
                extraRustcOpts =
                  (crate.extraRustcOpts or [ ]) ++ [ "-D" "warnings" "-C" "debuginfo=0" ];
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
