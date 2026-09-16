{
  description = "Native Memory2 SQLite and MCAP recorder for dimos";

  inputs = {
    dimos-native-rust.url = "github:dimensionalOS/dimos?ref=jeff/fix/native_build_cargo_path&dir=native/rust";
    flake-utils.follows = "dimos-native-rust/flake-utils";
    nixpkgs.follows = "dimos-native-rust/nixpkgs";
  };

  outputs = { self, nixpkgs, dimos-native-rust, flake-utils }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ] (system:
      let
        shared = dimos-native-rust.lib.${system};
        pkgsFor = nixpkgs.legacyPackages.${system};
        module = {
          name = "dimos-memory-recorder";
          path = "dimos/experimental/memory/rust";
          src = ./.;
          crateOverrides = pkgs: {
            libsqlite3-sys = _: {
              buildInputs = [ pkgs.sqlite ];
              nativeBuildInputs = [ pkgs.pkg-config ];
              LIBSQLITE3_SYS_USE_PKG_CONFIG = "1";
            };
            turbojpeg-sys = _: {
              nativeBuildInputs = [ pkgs.cmake pkgs.nasm ];
              dontUseCmakeConfigure = true;
            };
          };
        };
      in {
        packages.dimos-memory-recorder = shared.buildNativeModule module;

        checks.clippy = shared.clippyNativeModule module;

        packages.clippy = shared.clippyNativeModule module;

        devShells.default = pkgsFor.mkShell {
          packages = shared.rustTools
            ++ [ pkgsFor.cmake pkgsFor.nasm pkgsFor.pkg-config pkgsFor.sqlite pkgsFor.sqlite.dev ]
            ++ pkgsFor.lib.optionals pkgsFor.stdenv.hostPlatform.isDarwin [ pkgsFor.libiconv ];
          LIBSQLITE3_SYS_USE_PKG_CONFIG = "1";
        };
      });
}
