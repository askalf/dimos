{
  description = "Virtual Livox Mid-360 pcap replay native module for dimos";

  inputs = {
    dimos-native-rust.url = "github:dimensionalOS/dimos?ref=jeff/fix/native_build_cargo_path&dir=native/rust";
    flake-utils.follows = "dimos-native-rust/flake-utils";
  };

  outputs = { self, dimos-native-rust, flake-utils }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ] (system:
      let
        shared = dimos-native-rust.lib.${system};
        module = {
          name = "dimos-virtual-mid360";
          path = "dimos/hardware/sensors/lidar/virtual_mid360";
          src = ./.;
        };
      in {
        packages.dimos-virtual-mid360 = shared.buildNativeModule module;

        checks.clippy = shared.clippyNativeModule module;

        packages.clippy = shared.clippyNativeModule module;

        devShells.default = dimos-native-rust.devShells.${system}.default;
      });
}
