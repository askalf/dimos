{
  description = "A devShell for the shared dimos crates; they are consumed as a git dependency, not from here";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  # Nothing takes this as an input and it builds no package. It exists so the two
  # crates beside it can be formatted and linted like any other: the hooks walk cargo
  # workspace roots, and these are workspace roots with no module flake of their own.
  # Modules get these crates from github.com/dimensionalOS/dimos as an ordinary git
  # dependency, so there is no shared builder here to go stale.
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ] (system:
      let pkgs = nixpkgs.legacyPackages.${system}; in {
        devShells.default = pkgs.mkShell {
          packages = [ pkgs.cargo pkgs.rustc pkgs.clippy pkgs.rustfmt ];
        };
      });
}
