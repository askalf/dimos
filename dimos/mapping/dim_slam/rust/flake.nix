{
  description = "dimSLAM native module for DimOS: the dim_slam library behind an LCM wrapper";

  inputs = {
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

        moduleFor = variant:
          let sdkPackage = sdkPackages."sdk-${variant}"; in
          {
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
        packageFor = variant: shared.buildNativeModule (moduleFor variant);
        clippyFor = variant: shared.clippyNativeModule (moduleFor variant);
        lintedVariant = builtins.head (builtins.sort builtins.lessThan variants);
      in {
        packages = nixpkgs.lib.genAttrs variants packageFor // {
          clippy = clippyFor lintedVariant;
        };

        checks.clippy = clippyFor lintedVariant;

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
