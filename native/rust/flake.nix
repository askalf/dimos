{
  description = "The shared dimos rust crates, and the builder every rust native module uses";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    crate2nix.url = "github:nix-community/crate2nix";
    crate2nix.inputs.nixpkgs.follows = "nixpkgs";
  };

  # Every rust native module takes this flake as its one and only in-repo input and
  # follows its nixpkgs, so the whole repo resolves to a single nixpkgs revision
  # instead of the seven it used to carry. Nothing here reaches back out at the
  # repository: the crates below are the only dimos code a module is allowed to see.
  outputs = { self, nixpkgs, flake-utils, crate2nix }:
    let
      # Tracked sources only. A module's `nix build` copies this whole tree into its
      # sandbox, so an editor swap file or a stray target/ here would land in every
      # module's derivation.
      sharedCrates = {
        "native/rust/dimos-module" = ./dimos-module;
        "native/rust/dimos-module-macros" = ./dimos-module-macros;
      };
    in
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ] (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        rustTools = [ pkgs.cargo pkgs.rustc pkgs.clippy pkgs.rustfmt ];

        # A module's build_command is `nix build path:.#…`, and a `path:` ref copies
        # everything in the directory -- gitignored and untracked alike. Build output
        # is therefore an input to the next build unless it is filtered here: without
        # this, one `cargo build` in a module directory makes its next nix build a
        # cache miss and drags 2+ GB of `target/` into the store. `__pycache__` is
        # the same hazard in miniature: a module with python next to its crate grows
        # one the moment anybody imports it. `cleanSource` already drops `.git`,
        # editor backups and `result` symlinks.
        cleanModuleSource = src: pkgs.lib.cleanSourceWith {
          src = pkgs.lib.cleanSource src;
          filter = path: type:
            let base = baseNameOf (toString path); in
            !(type == "directory"
              && (base == "target" || base == "build" || base == "__pycache__"));
        };

        # The tree a module's crate2nix build runs in: the module's own directory at
        # the same relative path it occupies in the repository, plus the shared
        # crates at theirs, so the relative `path = ` dependency on dimos-module
        # in the module's Cargo.toml resolves without rewriting it.
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
      in {
        # `buildNativeModule` is the whole convention: a module flake names itself,
        # says where it sits in the repo, hands over its own directory, and gets a
        # package back. crate2nix builds one derivation per crate, so the shared
        # crates and every third-party dependency are built once and then substituted
        # into every other module -- which a single buildRustPackage per module
        # cannot do, because it vendors and compiles the whole graph privately.
        lib = {
          inherit moduleSource;

          # Exported so a module devShell can extend the toolchain instead of
          # restating it and drifting from it.
          inherit rustTools;

          buildNativeModule =
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
              called = import generated {
                inherit pkgs;
                buildRustCrateForPkgs = cratePkgs: cratePkgs.buildRustCrate.override {
                  defaultCrateOverrides =
                    cratePkgs.defaultCrateOverrides // (crateOverrides cratePkgs);
                };
              };
            in
            # crate2nix exposes `rootCrate` for a plain package and `workspaceMembers`
            # for a workspace manifest. A module that carries a pyo3 `py/` member is the
            # second kind even though it is one package to us, so accept both rather
            # than making the caller know which shape its own Cargo.toml produces.
            if called ? rootCrate
            then called.rootCrate.build
            else called.workspaceMembers.${name}.build;
        };

        # No `packages`: this flake ships source and a builder, not a binary. The
        # crates here are libraries every module compiles into itself.

        # The shell the pre-commit clippy hook runs in for crates that need nothing
        # but a toolchain. A module with system libraries extends this in its own
        # flake rather than adding them here.
        devShells.default = pkgs.mkShell { packages = rustTools; };
      });
}
