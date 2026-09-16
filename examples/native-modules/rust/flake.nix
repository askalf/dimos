{
  description = "Example rust native modules for dimos";

  # The shared crates come in as a remote git input, not a relative path. A `path:..`
  # input is unresolvable from a `path:.` build (`..` escapes the store path), and a
  # relative `git+file:` is deprecated (nix#12281); the remote ref is also the only
  # form that survives this module moving into a repository of its own. It costs one
  # 25 MB store path, shared by every module locked to the same revision -- against
  # the ~16 GB a local ref copies, because a local ref reads the working tree and so
  # picks up every smudged git-lfs blob under data/.
  inputs = {
    # NOTE: pinned to the branch that introduces native/{rust,cpp}/flake.nix,
    # because the default branch does not have those files yet. Point it at
    # `ref=main` once those land -- they are their own pull request, ahead of this
    # one, so the window is short. Nobody has to remember: the test
    # test_shared_flake_inputs_are_pinned_to_main_once_they_exist_upstream goes red
    # as soon as the shared flakes appear on the default branch.
    dimos-native-rust.url = "github:dimensionalOS/dimos?ref=jeff/fix/native_build_cargo_path&dir=native/rust";
    flake-utils.follows = "dimos-native-rust/flake-utils";
  };

  outputs = { self, dimos-native-rust, flake-utils }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ] (system:
      let shared = dimos-native-rust.lib.${system}; in {
        packages.dimos-native-module-examples = shared.buildNativeModule {
          name = "dimos-native-module-examples";
          path = "examples/native-modules/rust";
          src = ./.;
        };

        devShells.default = dimos-native-rust.devShells.${system}.default;
      });
}
