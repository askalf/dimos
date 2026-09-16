{
  description = "micromamba for the dimos Habitat native module";

  # Nothing here is built from the shared C++ SDK; the input is taken for the one
  # thing every flake in this repository has to agree on, the nixpkgs revision.
  # native/cpp is where it is chosen, so there is exactly one answer and no list of
  # exceptions to keep correct.
  # NOTE: pinned to the branch that introduces native/{rust,cpp}/flake.nix,
  # because the default branch does not have those files yet. Point it at
  # `ref=main` once those land -- they are their own pull request, ahead of this
  # one, so the window is short. Nobody has to remember: the test
  # test_shared_flake_inputs_are_pinned_to_main_once_they_exist_upstream goes red
  # as soon as the shared flakes appear on the default branch.
  inputs.dimos-native-cpp.url = "github:dimensionalOS/dimos?ref=jeff/fix/native_build_cargo_path&dir=native/cpp";
  inputs.nixpkgs.follows = "dimos-native-cpp/nixpkgs";

  outputs = { self, nixpkgs, ... }:
    let
      # linux-64 only: the aihabitat conda channel has no aarch64 habitat-sim.
      systems = [ "x86_64-linux" ];
      forAll = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in {
      # Provides the installer, not the simulator: habitat-sim is conda-only and
      # headless rendering needs the host's EGL driver, so this cannot be a derivation.
      devShells = forAll (pkgs: {
        default = pkgs.mkShellNoCC {
          packages = [ pkgs.micromamba pkgs.curl pkgs.cacert ];
        };
      });
    };
}
