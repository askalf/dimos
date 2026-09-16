{
  description = "RealSense D4xx camera native module for dimos";

  inputs = {
    dimos-native-rust.url = "github:dimensionalOS/dimos?ref=jeff/fix/native_build_cargo_path&dir=native/rust";
    flake-utils.follows = "dimos-native-rust/flake-utils";
  };

  outputs = { self, dimos-native-rust, flake-utils }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" ] (system:
      let
        shared = dimos-native-rust.lib.${system};
        module = {
          name = "dimos-realsense";
          path = "dimos/hardware/sensors/camera/realsense/rust";
          src = ./.;
          crateOverrides = pkgs:
            let
              needsLibrealsense = _: {
                buildInputs = [ pkgs.librealsense ];
                nativeBuildInputs = [ pkgs.pkg-config ];
              };
            in
            {
              realsense-sys = needsLibrealsense;
              dimos-realsense = needsLibrealsense;
            };
        };
      in {
        packages.dimos-realsense = shared.buildNativeModule module;

        checks.clippy = shared.clippyNativeModule module;

        packages.clippy = shared.clippyNativeModule module;

        devShells.default = shared.devShellWith (pkgs: [ pkgs.librealsense pkgs.pkg-config ]);
      });
}
