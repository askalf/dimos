{
  description = "Virtual Livox Mid-360 pcap replay native module for dimos";

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
  };

  outputs = { self, dimos-native-rust, flake-utils }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ] (system:
      let shared = dimos-native-rust.lib.${system}; in {
        packages.dimos-virtual-mid360 = shared.buildNativeModule {
          name = "dimos-virtual-mid360";
          path = "dimos/hardware/sensors/lidar/virtual_mid360";
          src = ./.;
        };

        devShells.default = dimos-native-rust.devShells.${system}.default;
      });
}
