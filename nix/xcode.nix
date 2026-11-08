# The iOS SDK only ships with Xcode, which Nix cannot provide, so every Henge
# derivation compiles with the system Xcode (27 or later). That needs
# `sandbox = false` or `sandbox = relaxed` in nix.conf.
{ stdenvNoCC }:
let
  setup = ''
    unset DEVELOPER_DIR SDKROOT MACOSX_DEPLOYMENT_TARGET CC CXX LD AR NM RANLIB STRIP
    export PATH="$PATH:/usr/bin:/bin:/usr/sbin:/sbin"
    export HOME="$TMPDIR/home"
    mkdir -p "$HOME"
    # xcodegen and xcodebuild read these, which the Nix builder leaves unset.
    USER="$(id -un)"
    export USER LOGNAME="$USER"
    # Henge's scripts run under /bin/zsh, which always reads /etc/zshenv; nix-darwin's
    # rebuilds PATH there and drops the build inputs unless told it already ran.
    export __ETC_ZSHENV_SOURCED=1 __NIX_DARWIN_SET_ENVIRONMENT_DONE=1
    # The builder is already inside a Seatbelt sandbox, and macOS refuses a nested one,
    # so swiftc cannot sandbox macro plugins (SwiftUI's @State) the way it does by default.
    echo 'OTHER_SWIFT_FLAGS = $(inherited) -disable-sandbox' > "$TMPDIR/nix.xcconfig"
    export XCODE_XCCONFIG_FILE="$TMPDIR/nix.xcconfig"
    xcode_major="$(xcodebuild -version | sed -n 's/^Xcode \([0-9]*\).*/\1/p')"
    if [ "''${xcode_major:-0}" -lt 27 ]; then
      echo "Henge needs Xcode 27 or later (xcode-select points at: $(xcodebuild -version | head -1))" >&2
      exit 1
    fi
  '';
in
{
  mkDerivation =
    args:
    stdenvNoCC.mkDerivation (
      {
        __noChroot = true;
        dontConfigure = true;
        dontInstall = true;
        dontFixup = true;
      }
      // args
      // {
        buildPhase = setup + args.buildPhase;
      }
    );
}
