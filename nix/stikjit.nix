# ci/build-stikjit.sh. StikJIT links the libidevice_ffi.a its repository ships.
{
  xcode,
  stikjit,
  applyPatches,
  xcodegen,
}:
xcode.mkDerivation {
  pname = "stikjit-xcframework";
  version = "1.9.0";
  src = applyPatches {
    name = "stikjit-src";
    src = stikjit;
    patches = [ ../ci/stikjit-high-address.patch ];
  };
  nativeBuildInputs = [ xcodegen ];
  buildPhase = ''
    xcodegen generate
    xcodebuild archive -project StikJIT.xcodeproj -scheme StikJIT -configuration Release \
      -sdk iphoneos -destination 'generic/platform=iOS' -archivePath build/StikJIT \
      -derivedDataPath "$TMPDIR/DerivedData" \
      BUILD_LIBRARY_FOR_DISTRIBUTION=YES CODE_SIGNING_ALLOWED=NO -quiet
    mkdir -p "$out"
    xcodebuild -create-xcframework \
      -framework build/StikJIT.xcarchive/Products/Library/Frameworks/StikJIT.framework \
      -output "$out/StikJIT.xcframework"
  '';
}
