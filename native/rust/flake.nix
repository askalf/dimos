{
  description = "The shared dimos rust crates, and the builder every rust native module uses";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    crate2nix.url = "github:nix-community/crate2nix";
    crate2nix.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, flake-utils, crate2nix }:
    let
      sharedCrates = {
        "native/rust/dimos-module" = ./dimos-module;
        "native/rust/dimos-module-macros" = ./dimos-module-macros;
      };
      sharedCrateNames = [ "dimos-module" "dimos-module-macros" ];
    in
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ] (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        rustTools = [ pkgs.cargo pkgs.rustc pkgs.clippy pkgs.rustfmt ];

        cleanModuleSource = src: pkgs.lib.cleanSourceWith {
          src = pkgs.lib.cleanSource src;
          filter = path: type:
            let base = baseNameOf (toString path); in
            !(type == "directory"
              && (base == "target" || base == "build" || base == "__pycache__"));
        };

        moduleSource = { name, path, src }:
          pkgs.runCommand "${name}-source" { } (
            ''
              mkdir -p $out/${builtins.dirOf path}
              cp -r ${cleanModuleSource src} $out/${path}
            ''
            + pkgs.lib.concatStrings (pkgs.lib.mapAttrsToList (dest: tree: ''
              mkdir -p $out/${builtins.dirOf dest}
              cp -r ${tree} $out/${dest}
            '') sharedCrates)
            + "chmod -R u+w $out\n"
          );

        moduleCrates =
          { name              # the cargo package name, and the flake output name
          , path              # the module's path within the repository
          , src               # the module's own directory
          , crateOverrides ? (_: { })   # pkgs -> attrset of crate2nix overrides
          , cargoToml ? "${path}/Cargo.toml"
          }:
          let
            generated = crate2nix.tools.${system}.generatedCargoNix {
              inherit name;
              src = moduleSource { inherit name path src; };
              inherit cargoToml;
            };
            callWith = lintThese: import generated {
              inherit pkgs;
              buildRustCrateForPkgs = cratePkgs:
                let
                  build = cratePkgs.buildRustCrate.override {
                    defaultCrateOverrides =
                      cratePkgs.defaultCrateOverrides // (crateOverrides cratePkgs);
                  };
                in
                crate: build (crate // pkgs.lib.optionalAttrs
                  (builtins.elem crate.crateName lintThese)
                  {
                    useClippy = true;
                    capLints = "forbid";
                    extraRustcOpts = (crate.extraRustcOpts or [ ]) ++ [ "-D" "warnings" ];
                  });
            };
            plain = callWith [ ];

            buildOf = called:
              if called ? rootCrate
              then called.rootCrate.build
              else called.workspaceMembers.${name}.build;

            firstParty = pkgs.lib.unique (
              (if plain ? rootCrate then [ name ] else builtins.attrNames plain.workspaceMembers)
              ++ sharedCrateNames
            );
          in {
            package = buildOf plain;
            clippy = (buildOf (callWith firstParty)).override {
              runTests = true;
              testCrateFlags = [ "--list" ];
            };
          };
      in {
        lib = {
          inherit moduleSource;

          inherit rustTools;

          devShellWith = select: pkgs.mkShell { packages = rustTools ++ select pkgs; };

          buildNativeModule = args: (moduleCrates args).package;

          clippyNativeModule = args: (moduleCrates args).clippy;
        };


        devShells.default = pkgs.mkShell { packages = rustTools; };
      });
}
