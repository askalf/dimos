{
  description = "Memory recorder native module for dimos";

  inputs = {
    nix-filter.url = "github:numtide/nix-filter";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    crate2nix.url = "github:nix-community/crate2nix";
    crate2nix.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nix-filter, nixpkgs, flake-utils, crate2nix }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ] (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        name = "dimos-memory-recorder";

        src = nix-filter.lib { root = ./.; exclude = [ "target" "build" "result" "__pycache__" ]; };

        generated = crate2nix.tools.${system}.generatedCargoNix { inherit name src; };

        # sqlite and the turbojpeg encoder are C libraries; crate2nix builds each
        # crate on its own, so the -sys crates name what they link rather than the
        # whole package doing it once.
        sysOverrides = {
          libsqlite3-sys = _: {
            buildInputs = [ pkgs.sqlite ];
            nativeBuildInputs = [ pkgs.pkg-config ];
            LIBSQLITE3_SYS_USE_PKG_CONFIG = "1";
          };
          # turbojpeg-sys vendors libjpeg-turbo and drives cmake from its own build
          # script. dontUseCmakeConfigure stops nixpkgs' cmake setup hook from *also*
          # configuring the crate root, which has no CMakeLists.txt.
          turbojpeg-sys = _: {
            nativeBuildInputs = [ pkgs.cmake pkgs.nasm ];
            dontUseCmakeConfigure = true;
          };
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

        rustTools = [ pkgs.cargo pkgs.rustc pkgs.clippy pkgs.rustfmt ];
      in {
        packages.default = buildOf (callWith false);
        packages.${name} = self.packages.${system}.default;
        packages.clippy = (buildOf (callWith true)).override {
          runTests = true;
          testCrateFlags = [ "--list" ];
        };
        checks.clippy = self.packages.${system}.clippy;

        devShells.default = pkgs.mkShell {
          packages = rustTools
            ++ [ pkgs.cmake pkgs.nasm pkgs.pkg-config pkgs.sqlite pkgs.sqlite.dev ]
            ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isDarwin [ pkgs.libiconv ];
          LIBSQLITE3_SYS_USE_PKG_CONFIG = "1";
        };
      });
}
