{
  description = "FAST-LIO2 + Livox Mid-360 native module";

  inputs = {
    zenoh.url = "github:jeff-hykin/zenoh_flake";
    zenoh.inputs.nixpkgs.follows = "nixpkgs";
    zenoh.inputs.flake-utils.follows = "flake-utils";
    nixpkgs.follows = "dimos-native-cpp/nixpkgs";
    flake-utils.follows = "dimos-native-cpp/flake-utils";
    dimos-native-cpp.url = "github:dimensionalOS/dimos?ref=jeff/fix/native_build_cargo_path&dir=native/cpp";
    dimos-lcm = {
      url = "github:dimensionalOS/dimos-lcm/main";
      flake = false;
    };
    # Standalone Boost.PFR, consumed by the SDK via a FetchContent source override.
    pfr = {
      url = "github:apolukhin/pfr_non_boost/2.3.2";
      flake = false;
    };
    fast-lio = {
      # get_body_cloud()/get_body_cloud_down() with the IMU<-lidar extrinsic
      # applied (PR #1); retarget to jeff/feat/fastlio-body-cloud once merged.
      url = "github:dimensionalOS/dimos-module-fastlio2?ref=ivan/fix/body-cloud-imu-extrinsic";
      flake = false;
    };
    lcm-extended = {
      url = "github:jeff-hykin/lcm_extended";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
  };

  outputs = { self, nixpkgs, zenoh, flake-utils, dimos-native-cpp, dimos-lcm, pfr, fast-lio, lcm-extended, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # Overlay fixes for darwin-broken nixpkgs recipes in our transitive
        # dep chain (pcl → vtk → pdal → tiledb → libpqxx).  Each of these
        # should go upstream; kept here so we can build in the meantime.
        #
        # Gated on isDarwin so Linux keeps binary-cache hits for the stock
        # libpqxx / tiledb / pdal / vtk / pcl derivations.  Applying the
        # override on Linux would change their input hashes and force a
        # from-source rebuild of the whole chain for no benefit.
        darwinDepFixes = final: prev:
          if !prev.stdenv.isDarwin then { } else {
            # libpqxx: postgresqlTestHook is in nativeCheckInputs
            # unconditionally and that package is marked broken on darwin.
            # The list is eagerly evaluated, so simply referencing it aborts
            # eval.  Upstream fix is to wrap the list in
            # `lib.optionals (meta.availableOn ...)`.
            libpqxx = prev.libpqxx.overrideAttrs (_old: {
              nativeCheckInputs = [ ];
              doCheck = false;
            });
            # tiledb: darwin-only patch `generate_embedded_data_header.patch`
            # targets a file that doesn't exist in tiledb 2.30.0 (the
            # upstream code path was reworked and `file(ARCHIVE_CREATE ...)`
            # is no longer used anywhere in the source).  Filter out only
            # that patch — don't drop everything, in case nixpkgs adds an
            # unrelated security patch in a future bump.
            tiledb = prev.tiledb.overrideAttrs (old: {
              patches = builtins.filter
                (p: !(prev.lib.hasSuffix "generate_embedded_data_header.patch" (toString p)))
                (old.patches or [ ]);
            });
          };
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ darwinDepFixes ];
        };
        livox-sdk2 = pkgs.stdenv.mkDerivation rec {
          pname = "livox-sdk2";
          version = "1.2.5";

          src = pkgs.fetchFromGitHub {
            owner = "Livox-SDK";
            repo = "Livox-SDK2";
            rev = "v${version}";
            hash = "sha256-NGscO/vLiQ17yQJtdPyFzhhMGE89AJ9kTL5cSun/bpU=";
          };

          # macOS socket fixes (SO_RCVBUF too large, broadcast bind fails).
          patches = [ ./livox-sdk2-darwin.patch ];

          nativeBuildInputs = [ pkgs.cmake ];

          cmakeFlags = [
            "-DBUILD_SHARED_LIBS=ON"
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
          ];

          preConfigure = ''
            substituteInPlace CMakeLists.txt \
              --replace-fail "add_subdirectory(samples)" ""
            sed -i '1i #include <cstdint>' sdk_core/comm/define.h
            sed -i '1i #include <cstdint>' sdk_core/logger_handler/file_manager.h
            find . -name CMakeLists.txt -exec sed -i 's/-Werror//g' {} +
          '';
        };
        lcm = lcm-extended.packages.${system}.lcm;
        zenohc = zenoh.packages.${system}.zenoh-c;
        zenohcpp = zenoh.packages.${system}.zenoh-cpp;

        livox-common = ./livox_common;

        fastlio2_native = pkgs.stdenv.mkDerivation {
          pname = "fastlio2_native";
          version = "0.2.0";

          src = dimos-native-cpp.lib.${system}.cleanModuleSource ./.;

          nativeBuildInputs = [ pkgs.cmake pkgs.pkg-config ];
          buildInputs = [
            livox-sdk2
            lcm
            pkgs.glib
            pkgs.eigen
            pkgs.pcl
            pkgs.boost
            pkgs.llvmPackages.openmp
            pkgs.nlohmann_json
            zenohc
            zenohcpp
          ];

          cmakeFlags = [
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
            "-DFETCHCONTENT_SOURCE_DIR_DIMOS_LCM=${dimos-lcm}"
            "-DFETCHCONTENT_SOURCE_DIR_PFR=${pfr}"
            "-DFASTLIO_DIR=${fast-lio}"
            "-DLIVOX_COMMON_DIR=${livox-common}"
            "-DDIMOS_NATIVE_CPP_DIR=${dimos-native-cpp.packages.${system}.default}"
          ];
        };
      in {
        packages = {
          default = fastlio2_native;
          inherit fastlio2_native;
        };
      });
}
