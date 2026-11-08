# Henge.ipa (or Henge.tipa for TrollStore): ios/scripts/fetch_deps.sh,
# tools/build_ios_restore.sh, make ios-lib and ios/scripts/package_ipa.sh, with
# the dependencies built in nix/ in place of a Husk checkout.
{
  lib,
  xcode,
  iosLibs,
  rppairing,
  stikjitFramework,
  xcodegen,
  cmake,
  gnumake,
  git,
  python,
  version,
  trollstore ? false,
}:
let
  file = if trollstore then "Henge.tipa" else "Henge.ipa";
in
xcode.mkDerivation {
  pname = "henge";
  inherit version;
  src = lib.fileset.toSource {
    root = ./.;
    fileset = lib.fileset.difference ./. (
      lib.fileset.unions [
        ./flake.nix
        ./flake.lock
        ./package.nix
        ./nix
        ./Screenshots
      ]
    );
  };
  nativeBuildInputs = [
    xcodegen
    cmake
    gnumake
    git
    python
  ];
  buildPhase = ''
    mkdir -p ios/deps/lib
    cp -R ${stikjitFramework}/StikJIT.xcframework ios/deps/
    cp ${rppairing}/lib/libhusk_rppairing.a ios/deps/lib/
    for l in libslirp libglib-2.0 libintl libpcre2-8; do cp ${iosLibs}/lib/$l.a ios/deps/lib/; done
    chmod -R u+w ios/deps
    export IOS_DEPS=${iosLibs}

    ./tools/build_ios_restore.sh
    make ios-lib
    mkdir -p "$HOME/Desktop"
    ./ios/scripts/package_ipa.sh ${lib.optionalString trollstore "--trollstore"}
    install -Dm644 "$HOME/Desktop/${file}" "$out/${file}"
  '';
}
