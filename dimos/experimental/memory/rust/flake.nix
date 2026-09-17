{
  description = "Memory recorder native module for dimos";

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
        name = "dimos-memory-recorder";

        src = pkgs.lib.cleanSourceWith {
          src = pkgs.lib.cleanSource ./.;
          filter = path: type:
            let base = baseNameOf (toString path); in
            !(type == "directory"
              && (base == "target" || base == "build" || base == "__pycache__"));
        };

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
                extraRustcOpts = (crate.extraRustcOpts or [ ]) ++ [ "-D" "warnings" ];
              });
        };
        buildOf = called:
          if called ? rootCrate then called.rootCrate.build
          else called.workspaceMembers.${name}.build;

        rustTools = [ pkgs.cargo pkgs.rustc pkgs.clippy pkgs.rustfmt ];
      in {
        packages.${name} = buildOf (callWith false);
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
