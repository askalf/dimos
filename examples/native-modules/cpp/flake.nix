{
  description = "dimos C++ native module ping-pong example";

  inputs = {
    zenoh.url = "github:jeff-hykin/zenoh_flake";
    zenoh.inputs.nixpkgs.follows = "nixpkgs";
    zenoh.inputs.flake-utils.follows = "flake-utils";
    # One nixpkgs for the whole C++ side. native/cpp is the single place the
    # revision is chosen and every module follows it, rather than each module
    # resolving `nixos-unstable` on its own clock. Independent resolution is
    # exactly what left these modules on a February nixpkgs while the shared SDK
    # had moved to September: two stdenvs, and no binary cache shared between them.
    nixpkgs.follows = "dimos-native-cpp/nixpkgs";
    flake-utils.follows = "dimos-native-cpp/flake-utils";
    lcm-extended = {
      url = "github:jeff-hykin/lcm_extended";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
    # Generated LCM message headers, consumed via a FetchContent source override.
    dimos-lcm = {
      url = "github:dimensionalOS/dimos-lcm/main";
      flake = false;
    };
    # Standalone Boost.PFR, consumed by the SDK via a FetchContent source override.
    pfr = {
      url = "github:apolukhin/pfr_non_boost/2.3.2";
      flake = false;
    };
    # The shared C++ SDK, as a remote ref rather than the bare path literal that used
    # to climb three directories to reach native/cpp. Under the `path:.` build ref
    # this example now uses, such a literal escapes the flake's store path.
    # NOTE: pinned to the branch that introduces native/{rust,cpp}/flake.nix,
    # because the default branch does not have those files yet. Point it at
    # `ref=main` once those land -- they are their own pull request, ahead of this
    # one, so the window is short. Nobody has to remember: the test
    # test_shared_flake_inputs_are_pinned_to_main_once_they_exist_upstream goes red
    # as soon as the shared flakes appear on the default branch.
    dimos-native-cpp.url = "github:dimensionalOS/dimos?ref=jeff/fix/native_build_cargo_path&dir=native/cpp";
  };

  outputs = { self, nixpkgs, zenoh, flake-utils, lcm-extended, dimos-lcm, pfr, dimos-native-cpp, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        lcm = lcm-extended.packages.${system}.lcm;
        zenohc = zenoh.packages.${system}.zenoh-c;
        zenohcpp = zenoh.packages.${system}.zenoh-cpp;
      in {
        packages.dimos-native-module-examples-cpp = pkgs.stdenv.mkDerivation {
          pname = "dimos-native-ping-pong";
          version = "0.1.0";
          # Not `./.`: a bare directory hands nix the `result` symlink of the previous
          # build, which changes this derivation's hash and loses the Cachix hit.
          src = dimos-native-cpp.lib.${system}.cleanModuleSource ./.;

          nativeBuildInputs = [ pkgs.cmake pkgs.pkg-config ];
          buildInputs = [ lcm pkgs.glib pkgs.nlohmann_json zenohc zenohcpp ];

          cmakeFlags = [
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
            "-DFETCHCONTENT_SOURCE_DIR_DIMOS_LCM=${dimos-lcm}"
            "-DFETCHCONTENT_SOURCE_DIR_PFR=${pfr}"
            # The header-only SDK lives outside this dir. A git-tree flake can
            # reach it as a path literal within the repo tree.
            "-DDIMOS_NATIVE_CPP_DIR=${dimos-native-cpp.packages.${system}.default}"
          ];
        };
      });
}
