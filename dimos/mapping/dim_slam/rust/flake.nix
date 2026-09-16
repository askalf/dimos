{
  description = "dimSLAM native module for DimOS: the dim_slam library behind an LCM wrapper";

  # The shared crates come in as a remote git input, not a relative path. A `path:..`
  # input is unresolvable from a `path:.` build (`..` escapes the store path), and a
  # relative `git+file:` is deprecated (nix#12281); the remote ref is also the only
  # form that survives this module moving into a repository of its own. It costs one
  # 25 MB store path, shared by every module locked to the same revision -- against
  # the ~16 GB the old whole-repo input copied (a `path:` ref four levels up),
  # because a local ref reads the working tree and so picks up every smudged
  # git-lfs blob.
  inputs = {
    # NOTE: pinned to the branch that introduces native/{rust,cpp}/flake.nix,
    # because the default branch does not have those files yet. Drop the `ref=`
    # once this is on main -- without it the input follows the default branch,
    # which is what we want from then on. Nobody has to remember: the test
    # test_shared_flake_inputs_are_unpinned_once_they_exist_upstream goes red
    # as soon as the shared flakes appear on the default branch.
    dimos-native-rust.url = "github:dimensionalOS/dimos?ref=jeff/fix/native_build_cargo_path&dir=native/rust";
    flake-utils.follows = "dimos-native-rust/flake-utils";
    nixpkgs.follows = "dimos-native-rust/nixpkgs";
    cu-vslam-rs.url = "github:jeff-hykin/cu_vslam_rs";
    cu-vslam-rs.inputs.nixpkgs.follows = "nixpkgs";
    cu-vslam-rs.inputs.flake-utils.follows = "flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, cu-vslam-rs, dimos-native-rust }:
    # Not eachDefaultSystem: nixpkgs 26.11 dropped x86_64-darwin, and merely naming
    # it is an eval error.
    flake-utils.lib.eachSystem [ "aarch64-darwin" "aarch64-linux" "x86_64-linux" ] (system:
      let
        shared = dimos-native-rust.lib.${system};

        sdkPackages = nixpkgs.lib.filterAttrs
          (name: _: nixpkgs.lib.hasPrefix "sdk-" name)
          cu-vslam-rs.packages.${system};
        variants = map (nixpkgs.lib.removePrefix "sdk-") (builtins.attrNames sdkPackages);

        packageFor = variant:
          let sdkPackage = sdkPackages."sdk-${variant}"; in
          shared.buildNativeModule {
            name = "dim-slam-module";
            path = "dimos/mapping/dim_slam/rust";
            src = ./.;
            crateOverrides = _: {
              # cu_vslam_rs's build.rs compiles its shim against this SDK.
              cu_vslam_rs = _: { CUVSLAM_SDK_DIR = sdkPackage; };
              # buildRustCrate names DEP_ vars after the crate, cargo after the
              # `links` key, so cu_vslam_rs's lib_dir never reaches our build.rs
              # and the binary comes out with no rpath for libcuvslam.
              dim-slam-module = _: { DEP_CUVSLAM_LIB_DIR = "${sdkPackage}/lib"; };
            };
          };
      in {
        packages = nixpkgs.lib.genAttrs variants packageFor;

        # script needs to detect cuda/non-cuda to pick the right things to load.
        # The toolchain has to come from here: this used to be entered from inside
        # the nix/rust toolchain shell, and that shell is gone, so a devShell with
        # only a shellHook would silently hand cargo/clippy back to whatever the
        # host has on PATH.
        devShells.default = nixpkgs.legacyPackages.${system}.mkShellNoCC {
          packages = shared.rustTools;
          shellHook = ''
            if [ -z "''${CUVSLAM_SDK_DIR:-}" ]; then
              case "$(uname -s)-$(uname -m)" in
                Darwin-arm64) cuvslam_variant=metal ;;
                Linux-aarch64)
                  case "$(tr -d '\0' < /proc/device-tree/compatible 2>/dev/null)" in
                    *tegra264*) cuvslam_variant=thor ;;
                    *tegra234*) cuvslam_variant=orin ;;
                    *) cuvslam_variant=aarch64 ;;
                  esac ;;
                *)
                  cuda_major=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]*\).*/\1/p')
                  cuvslam_variant="x86_64''${cuda_major:+-cuda$cuda_major}" ;;
              esac
              case "$cuvslam_variant" in
${nixpkgs.lib.concatMapStringsSep "\n" (variant:
  "                ${variant}) cuvslam_sdk_drv=${builtins.unsafeDiscardStringContext sdkPackages."sdk-${variant}".drvPath} ;;"
) variants}
                *) cuvslam_sdk_drv= ;;
              esac
              if [ -n "$cuvslam_sdk_drv" ] \
                && CUVSLAM_SDK_DIR=$(nix build --no-link --print-out-paths "$cuvslam_sdk_drv^out"); then
                export CUVSLAM_SDK_DIR
              else
                echo "no cuVSLAM SDK for variant '$cuvslam_variant'; building the stub" >&2
              fi
              unset cuvslam_variant cuvslam_sdk_drv cuda_major
            fi
          '';
        };
      });
}
