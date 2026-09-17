{
  description = "MLS planner native module for dimos";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    crate2nix.url = "github:nix-community/crate2nix";
    crate2nix.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, flake-utils, crate2nix }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ] (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        name = "dimos-mls-planner";

        # `nix build path:.#…` copies this directory wholesale, gitignored output
        # included, so build output has to be filtered out or it lands in the
        # derivation.
        src = pkgs.lib.cleanSourceWith {
          src = pkgs.lib.cleanSource ./.;
          filter = path: type:
            let base = baseNameOf (toString path); in
            !(type == "directory"
              && (base == "target" || base == "build" || base == "__pycache__"));
        };

        generated = crate2nix.tools.${system}.generatedCargoNix { inherit name src; };

        # Our own crates compile with clippy-driver; everything else keeps its
        # ordinary derivation, so no dependency is built twice. capLints must be
        # "forbid": it is a ceiling, and the default "allow" suppresses every lint.
        ours = [ name "dimos-module" "dimos-module-macros" ];
        callWith = lint: import generated {
          inherit pkgs;
          buildRustCrateForPkgs = cratePkgs: crate:
            cratePkgs.buildRustCrate (crate // pkgs.lib.optionalAttrs
              (lint && builtins.elem crate.crateName ours)
              {
                useClippy = true;
                capLints = "forbid";
                extraRustcOpts = (crate.extraRustcOpts or [ ]) ++ [ "-D" "warnings" ];
              });
        };

        buildOf = called:
          if called ? rootCrate then called.rootCrate.build
          else called.workspaceMembers.${name}.build;

        rustTools = [ pkgs.cargo pkgs.rustc pkgs.clippy pkgs.rustfmt ];
      in {
        packages.${name} = buildOf (callWith false);

        # `runTests` is how crate2nix compiles test targets; `--list` makes the
        # binary enumerate and exit, so they are linted without being run --
        # exactly what `cargo clippy --all-targets` did.
        packages.clippy = (buildOf (callWith true)).override {
          runTests = true;
          testCrateFlags = [ "--list" ];
        };
        checks.clippy = self.packages.${system}.clippy;

        devShells.default = pkgs.mkShell { packages = rustTools; };
      });
}
