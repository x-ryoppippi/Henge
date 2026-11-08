# Husk's scripts/build_rppairing_ios.sh, with crates vendored from Cargo.lock.
{
  xcode,
  husk,
  rust,
  rustPlatform,
}:
xcode.mkDerivation {
  pname = "husk-rppairing";
  version = husk.shortRev;
  src = "${husk}/src/rppairing-ios";
  nativeBuildInputs = [ rust ];
  cargoVendor = rustPlatform.importCargoLock {
    lockFile = "${husk}/src/rppairing-ios/Cargo.lock";
  };
  buildPhase = ''
    export CARGO_HOME="$TMPDIR/cargo"
    mkdir -p "$CARGO_HOME"
    cat > "$CARGO_HOME/config.toml" <<EOF
    [source.crates-io]
    replace-with = "vendored"
    [source.vendored]
    directory = "$cargoVendor"
    EOF
    export IPHONEOS_DEPLOYMENT_TARGET=15.5
    cargo build --release --locked --offline --target aarch64-apple-ios
    install -Dm644 target/aarch64-apple-ios/release/libhusk_rppairing.a "$out/lib/libhusk_rppairing.a"
  '';
}
