{
  description = "Henge: build the iOS app from source";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    # Same revisions as ci/build-bitrise.sh and ci/build-stikjit.sh.
    husk = {
      url = "github:Leviidev/Husk/afe83d45b31194c4cc9f7b3db7ec2c757db8346e";
      flake = false;
    };
    stikjit = {
      url = "github:StikDebug/StikJIT/32287268fa5824f9edce4cb359f5833ce0cf7b00";
      flake = false;
    };
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      # The iOS SDK comes only with Xcode, so only an Apple silicon Mac can build Henge.
      systems = [ "aarch64-darwin" ];
      imports = [ ./nix/packages.nix ];
    };
}
