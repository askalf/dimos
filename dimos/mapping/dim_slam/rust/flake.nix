{
  description = "dimSLAM native module for DimOS: the dim_slam library behind an LCM wrapper";

  inputs = {
    nix-filter.url = "github:numtide/nix-filter";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    crate2nix.url = "github:nix-community/crate2nix";
    crate2nix.inputs.nixpkgs.follows = "nixpkgs";
    cu-vslam-rs.url = "github:jeff-hykin/cu_vslam_rs";
    cu-vslam-rs.inputs.nixpkgs.follows = "nixpkgs";
    cu-vslam-rs.inputs.flake-utils.follows = "flake-utils";
  };

  # Not eachDefaultSystem: nixpkgs 26.11 dropped x86_64-darwin, and merely naming it
  # is an eval error.
  outputs = { self, nix-filter, nixpkgs, flake-utils, crate2nix, cu-vslam-rs }:
    flake-utils.lib.eachSystem [ "aarch64-darwin" "aarch64-linux" "x86_64-linux" ] (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        name = "dim-slam-module";

        src = nix-filter.lib { root = ./.; exclude = [ "target" "build" "result" "__pycache__" ]; };

        generated = crate2nix.tools.${system}.generatedCargoNix { inherit name src; };

        sdkPackages = nixpkgs.lib.filterAttrs
          (n: _: nixpkgs.lib.hasPrefix "sdk-" n)
          cu-vslam-rs.packages.${system};
        variants = map (nixpkgs.lib.removePrefix "sdk-") (builtins.attrNames sdkPackages);

        ours = [ name "dimos-module" "dimos-module-macros" ];
        callWith = variant: lint:
          let sdkPackage = sdkPackages."sdk-${variant}"; in
          import generated {
            inherit pkgs;
            buildRustCrateForPkgs = cratePkgs:
              let build = cratePkgs.buildRustCrate.override {
                    defaultCrateOverrides = cratePkgs.defaultCrateOverrides // {
                      # cu_vslam_rs's build.rs compiles its shim against this SDK.
                      cu_vslam_rs = _: { CUVSLAM_SDK_DIR = sdkPackage; };
                      # buildRustCrate names DEP_ vars after the crate, cargo after the
                      # `links` key, so cu_vslam_rs's lib_dir never reaches our build.rs
                      # and the binary comes out with no rpath for libcuvslam.
                      dim-slam-module = _: { DEP_CUVSLAM_LIB_DIR = "${sdkPackage}/lib"; };
                    };
                  };
              in crate: build (crate // pkgs.lib.optionalAttrs
                (lint && builtins.elem crate.crateName ours)
                {
                  useClippy = true;
                  capLints = "forbid";
                  extraRustcOpts = (crate.extraRustcOpts or [ ]) ++ [ "-D" "warnings" ];
                });
          };
        buildOf = called:
          if called ? rootCrate then called.rootCrate.build
          else called.workspaceMembers.${name}.build;

        # Lint once, against the first SDK variant: they differ only in which cu_vslam
        # SDK the build script links, and the rust being linted is the same in all.
        lintedVariant = builtins.head (builtins.sort builtins.lessThan variants);
      in {
        packages = nixpkgs.lib.genAttrs variants (v: buildOf (callWith v false)) // {
          clippy = (buildOf (callWith lintedVariant true)).override {
            runTests = true;
            testCrateFlags = [ "--list" ];
          };
        };
        checks.clippy = self.packages.${system}.clippy;

        devShells.default = pkgs.mkShellNoCC {
          packages = [ pkgs.cargo pkgs.rustc pkgs.clippy pkgs.rustfmt ];
        };
      });
}
