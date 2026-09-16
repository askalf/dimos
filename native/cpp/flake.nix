{
  description = "The shared dimos C++ native module SDK, as a source tree for cmake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  # A header-only INTERFACE library that consumers `add_subdirectory` through
  # `-DDIMOS_NATIVE_CPP_DIR=...`, so the useful output is the source tree itself
  # rather than a built artifact.
  #
  # This exists as a flake so a module can take the SDK as an input instead of a
  # `${../../../../../../native/cpp}` path literal. Under the `path:.` build refs the
  # modules now use, such a literal escapes the module's store path and fails to
  # resolve -- and when it did resolve, it resolved by copying the entire repository,
  # git-lfs blobs included.
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ] (system:
      let pkgs = nixpkgs.legacyPackages.${system}; in {
        packages.default = pkgs.runCommand "dimos-native-cpp" { } ''
          cp -r ${pkgs.lib.cleanSource ./.} $out
          chmod -R u+w $out
        '';

        devShells.default = pkgs.mkShell {
          packages = [ pkgs.cmake pkgs.pkg-config pkgs.nlohmann_json ];
        };
      });
}
