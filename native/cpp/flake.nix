{
  description = "The shared dimos C++ native module SDK, as a source tree for cmake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ] (system:
      let pkgs = nixpkgs.legacyPackages.${system}; in {
        packages.default = pkgs.runCommand "dimos-native-cpp" { } ''
          cp -r ${pkgs.lib.cleanSource ./.} $out
          chmod -R u+w $out
        '';

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
