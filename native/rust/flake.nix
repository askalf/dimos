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
      # The same two crates by cargo package name, which is how crate2nix keys its
      # overrides. They are ours, so they get linted with the module's own crates.
      sharedCrateNames = [ "dimos-module" "dimos-module-macros" ];
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

        # crate2nix generates one derivation per crate. Called twice: once plainly, to
        # learn which crate names are ours, and once with those crates overridden --
        # the override has to be in place before the graph is built, so the names
        # cannot come from the graph it builds.
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
            # `useClippy` has to be set in the crate's own arguments, not through
            # crateOverrides: buildRustCrate reads it off the pre-override attrs
            # (`crate_.useClippy or false`), while capLints and extraRustcOpts it
            # reads after. Setting all three through the overrides therefore builds
            # with plain rustc and passes -- which it silently did here once.
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
                    # `--cap-lints` is a ceiling, not a floor. buildRustCrate's
                    # default of "allow" suppresses every lint including clippy's,
                    # and "warn" -- which the nixpkgs docs suggest -- caps them at
                    # warn, so `-D warnings` can never become an error and the
                    # derivation cannot fail. Only "forbid", the top of the scale,
                    # leaves the levels alone. Verified against a planted
                    # `clippy::ptr_arg`: under "warn" it printed the warning and
                    # exited 0; under "forbid" it fails the build.
                    capLints = "forbid";
                    extraRustcOpts = (crate.extraRustcOpts or [ ]) ++ [ "-D" "warnings" ];
                  });
            };
            plain = callWith [ ];

            # crate2nix exposes `rootCrate` for a plain package and `workspaceMembers`
            # for a workspace manifest. A module that carries a pyo3 `py/` member is the
            # second kind even though it is one package to us, so accept both rather
            # than making the caller know which shape its own Cargo.toml produces.
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
            # `runTests` is how crate2nix compiles the test targets -- without it the
            # test code is never fed to clippy-driver and a lint that only fires in a
            # #[test] reaches main. But crate2nix also *executes* what it compiles,
            # and that is not what a lint check is for: livox's tests bind a
            # non-loopback address, which the nix sandbox refuses, so the check failed
            # on `AddrNotAvailable` rather than on any finding. `--list` makes libtest
            # enumerate its tests and exit, so the targets are compiled and linted and
            # nothing runs -- which is precisely what `cargo clippy --all-targets`,
            # the command this replaces, already did. Running the tests stays the
            # rust job's business.
            clippy = (buildOf (callWith firstParty)).override {
              runTests = true;
              testCrateFlags = [ "--list" ];
            };
          };
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

          # The shell for a module that needs system libraries -- realsense links
          # librealsense, and the clippy hook runs `cargo clippy` inside this shell, so
          # a build script that probes pkg-config finds nothing without it. Takes a
          # selector rather than a package list because a module flake has no nixpkgs of
          # its own; this one is the single revision every module follows.
          devShellWith = select: pkgs.mkShell { packages = rustTools ++ select pkgs; };

          buildNativeModule = args: (moduleCrates args).package;

          # The same crate graph as `buildNativeModule`, compiled with clippy-driver
          # instead of rustc -- but only for the crates we wrote. Third-party crates
          # keep their ordinary derivation, so they are the very paths the module
          # build already put on Cachix and none of them is compiled twice.
          #
          # This is what the pre-commit clippy hook cannot do: `cargo clippy` in a
          # devShell has no view of the nix store and rebuilds all ~340 dependency
          # crates per module, from scratch, on every CI run.
          #
          # `capLints = "warn"` is required: buildRustCrate's default of "allow"
          # suppresses every lint, clippy's included, and `useClippy` alone would
          # silently pass. Build scripts keep plain rustc -- clippy findings in a
          # generated build.rs are not actionable.
          #
          # Reaches the same targets as `cargo clippy --all-targets`, test code
          # included, and like that command it runs none of them.
          clippyNativeModule = args: (moduleCrates args).clippy;
        };

        # No `packages`: this flake ships source and a builder, not a binary. The
        # crates here are libraries every module compiles into itself.

        # The shell the pre-commit clippy hook runs in for crates that need nothing
        # but a toolchain. A module with system libraries extends this in its own
        # flake rather than adding them here.
        devShells.default = pkgs.mkShell { packages = rustTools; };
      });
}
