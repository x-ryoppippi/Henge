{ inputs, self, ... }:
{
  perSystem =
    { pkgs, inputs', ... }:
    let
      fx = inputs'.fenix.packages;
      scope = pkgs.lib.makeScope pkgs.newScope (final: {
        inherit (inputs) husk stikjit;
        xcode = final.callPackage ./xcode.nix { };
        # Husk's rppairing is built for aarch64-apple-ios, whose std nixpkgs' rustc lacks.
        rust = fx.combine [
          fx.stable.cargo
          fx.stable.rustc
          fx.targets.aarch64-apple-ios.stable.rust-std
        ];
        python = pkgs.python3.withPackages (ps: [ ps.packaging ]);
        iosLibs = final.callPackage ./ios-libs.nix { };
        rppairing = final.callPackage ./rppairing.nix { };
        stikjitFramework = final.callPackage ./stikjit.nix { };
        henge = final.callPackage ../package.nix {
          version = self.shortRev or self.dirtyShortRev;
        };
      });
    in
    {
      packages = {
        default = scope.henge;
        ipa = scope.henge;
        tipa = scope.henge.override { trollstore = true; };
        inherit (scope) iosLibs rppairing stikjitFramework;
      };

      devShells.default = pkgs.mkShellNoCC {
        packages = [
          pkgs.just
          scope.rust
          scope.python
          pkgs.xcodegen
          pkgs.meson
          pkgs.ninja
          pkgs.cmake
          pkgs.pkg-config
        ];
      };
    };
}
