{
  description = "Native Memory2 SQLite and MCAP recorder for dimos";

  # The shared crates come in as a remote git input, not a relative path. A `path:..`
  # input is unresolvable from a `path:.` build (`..` escapes the store path), and a
  # relative `git+file:` is deprecated (nix#12281); the remote ref is also the only
  # form that survives this module moving into a repository of its own. It costs one
  # 25 MB store path, shared by every module locked to the same revision -- against
  # the ~16 GB a local ref copies, because a local ref reads the working tree and so
  # picks up every smudged git-lfs blob under data/.
  inputs = {
    # NOTE: pinned to the branch that introduces native/{rust,cpp}/flake.nix,
    # because the default branch does not have those files yet. Drop the `ref=`
    # once this is on main -- without it the input follows the default branch,
    # which is what we want from then on.
    dimos-native-rust.url = "github:dimensionalOS/dimos?ref=jeff/fix/native_build_cargo_path&dir=native/rust";
    flake-utils.follows = "dimos-native-rust/flake-utils";
    nixpkgs.follows = "dimos-native-rust/nixpkgs";
  };

  outputs = { self, nixpkgs, dimos-native-rust, flake-utils }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ] (system:
      let
        shared = dimos-native-rust.lib.${system};
        pkgsFor = nixpkgs.legacyPackages.${system};
      in {
        packages.dimos-memory-recorder = shared.buildNativeModule {
          name = "dimos-memory-recorder";
          path = "dimos/experimental/memory/rust";
          src = ./.;
          # sqlite and the turbojpeg encoder are C libraries; crate2nix builds each
          # crate on its own, so the -sys crates name what they link rather than the
          # whole package doing it once.
          crateOverrides = pkgs: {
            libsqlite3-sys = _: {
              buildInputs = [ pkgs.sqlite ];
              nativeBuildInputs = [ pkgs.pkg-config ];
              LIBSQLITE3_SYS_USE_PKG_CONFIG = "1";
            };
            # turbojpeg-sys vendors libjpeg-turbo and drives cmake from its own
            # build script. dontUseCmakeConfigure stops nixpkgs' cmake setup hook
            # from *also* trying to configure the crate root, which has no
            # CMakeLists.txt -- the crate's build.rs is the only thing that should
            # invoke cmake.
            turbojpeg-sys = _: {
              nativeBuildInputs = [ pkgs.cmake pkgs.nasm ];
              dontUseCmakeConfigure = true;
            };
          };
        };

        devShells.default = pkgsFor.mkShell {
          packages = shared.rustTools
            ++ [ pkgsFor.cmake pkgsFor.nasm pkgsFor.pkg-config pkgsFor.sqlite pkgsFor.sqlite.dev ]
            ++ pkgsFor.lib.optionals pkgsFor.stdenv.hostPlatform.isDarwin [ pkgsFor.libiconv ];
          LIBSQLITE3_SYS_USE_PKG_CONFIG = "1";
        };
      });
}
