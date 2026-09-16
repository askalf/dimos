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
  # bare path literal climbing six directories to reach it. Under the `path:.` refs the
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

        # A C++ module is built with `nix build path:.#…` from its own directory, and a
        # `path:` ref copies that directory whole -- so everything in it is part of the
        # derivation hash. `result` is the trap: `nix build` writes that symlink itself,
        # which means a module's second build hashes differently from its first and
        # misses the binary cache forever after. Every C++ module therefore filters its
        # source through this rather than passing `./.` raw. The rust side gets the same
        # treatment inside `native/rust`'s `buildNativeModule`.
        lib.cleanModuleSource = src: pkgs.lib.cleanSourceWith {
          src = pkgs.lib.cleanSource src;
          filter = path: type:
            let base = baseNameOf (toString path); in
            !(type == "directory"
              && (base == "build" || base == "target" || base == "__pycache__"));
        };

        devShells.default = pkgs.mkShell {
          packages = [ pkgs.cmake pkgs.pkg-config pkgs.nlohmann_json ];
        };
      });
}
