# Husk's scripts/build_ios.sh stages libffi, glib and libslirp: all of Husk's iOS
# sysroot that Henge links. Sources are the ones Husk's scripts/sources.sh names.
{
  xcode,
  husk,
  fetchurl,
  fetchFromGitHub,
  meson,
  ninja,
  pkg-config,
  python,
}:
let
  libffiSrc = fetchurl {
    url = "https://github.com/libffi/libffi/releases/download/v3.5.0/libffi-3.5.0.tar.gz";
    hash = "sha256-jHJnhiil3YeC8IrUIdWkQeQsHFwbM+C8IRy/zx87OXg=";
  };
  glibSrc = fetchurl {
    url = "https://download.gnome.org/sources/glib/2.83/glib-2.83.0.tar.xz";
    hash = "sha256-oH2eGlekJ5xeznHCbcRO6hK9UY6p/2ldU+cimXAythQ=";
  };
  slirpSrc = fetchurl {
    url = "https://github.com/utmapp/libslirp/releases/download/v4.9.1-release-mirror/libslirp-v4.9.1.tar.gz";
    hash = "sha256-OXBUIUO3wR5qCaTStQ8woTNHPEHxXtC9zDt6HEUNmlw=";
  };
  # glib's meson subprojects (subprojects/*.wrap), provided so meson never downloads.
  pcre2Src = fetchurl {
    url = "https://github.com/PCRE2Project/pcre2/releases/download/pcre2-10.44/pcre2-10.44.tar.bz2";
    hash = "sha256-008C4RPPcZOh6/J3DTrFJwiNSF1OBH7RDl0hfG713pY=";
  };
  pcre2Patch = fetchurl {
    name = "pcre2_10.44-2_patch.zip";
    url = "https://wrapdb.mesonbuild.com/v2/pcre2_10.44-2/get_patch";
    hash = "sha256-QzbUIu6QQ4R+XhDbu9AZQNTJ5QJ/MczcM6eJihypQAk=";
  };
  proxyLibintl = fetchFromGitHub {
    owner = "frida";
    repo = "proxy-libintl";
    rev = "33934de09af6a6627eb44e310a8079df009abdbb";
    hash = "sha256-KpFvULzn0SWGexclDE+Hd8oLaFiZzsTwMKEWqn0nLa0=";
  };
in
xcode.mkDerivation {
  pname = "henge-ios-libs";
  version = "husk-${husk.shortRev}";
  srcs = [
    libffiSrc
    glibSrc
    slirpSrc
  ];
  sourceRoot = ".";
  nativeBuildInputs = [
    meson
    ninja
    pkg-config
    python
  ];
  postUnpack = ''
    cp -R ${proxyLibintl} glib-2.83.0/subprojects/proxy-libintl
    chmod -R u+w glib-2.83.0/subprojects/proxy-libintl
    mkdir -p glib-2.83.0/subprojects/packagecache
    cp ${pcre2Src} glib-2.83.0/subprojects/packagecache/pcre2-10.44.tar.bz2
    cp ${pcre2Patch} glib-2.83.0/subprojects/packagecache/pcre2_10.44-2_patch.zip
    (cd libslirp-v4.9.1 && patch -p1 < ${husk}/patches/libslirp-v4.9.1.patch)
  '';
  buildPhase = ''
    sdk="$(xcrun --sdk iphoneos --show-sdk-path)"
    target="-target arm64-apple-ios15.5"
    export CC="$(xcrun --sdk iphoneos --find clang) $target"
    export CXX="$(xcrun --sdk iphoneos --find clang++) $target"
    export AR="$(xcrun --sdk iphoneos --find ar)"
    export NM="$(xcrun --sdk iphoneos --find nm)"
    export RANLIB="$(xcrun --sdk iphoneos --find ranlib)"
    export STRIP="$(xcrun --sdk iphoneos --find strip)"
    export CFLAGS="-arch arm64 -isysroot $sdk -I$out/include $target -O2"
    export CXXFLAGS="$CFLAGS"
    export CPPFLAGS="-arch arm64 -isysroot $sdk -I$out/include $target"
    export LDFLAGS="-arch arm64 -isysroot $sdk -L$out/lib $target"
    # Only this sysroot, so no host library can leak into an iOS link.
    export PKG_CONFIG_LIBDIR="$out/lib/pkgconfig:$out/share/pkgconfig"
    unset PKG_CONFIG_PATH

    q() { local out="" w; for w in "$@"; do out="$out'$w',"; done; echo "[''${out%,}]"; }
    cross() { # file system
      cat > "$1" <<EOF
    [properties]
    needs_exe_wrapper = true
    [built-in options]
    c_args = $(q $CFLAGS)
    cpp_args = $(q $CXXFLAGS)
    objc_args = $(q $CFLAGS)
    c_link_args = $(q $LDFLAGS)
    cpp_link_args = $(q $LDFLAGS)
    objc_link_args = $(q $LDFLAGS)
    [binaries]
    c = $(q $CC)
    cpp = $(q $CXX)
    objc = $(q $(xcrun --sdk iphoneos --find clang))
    ar = '$AR'
    nm = '$NM'
    ranlib = '$RANLIB'
    strip = ['$STRIP', '-x']
    pkg-config = '$(command -v pkg-config)'
    python = '$(command -v python3)'
    [host_machine]
    system = '$2'
    $([ "$2" = darwin ] && printf "kernel = 'xnu'\nsubsystem = 'ios'")
    cpu_family = 'aarch64'
    cpu = 'aarch64'
    endian = 'little'
    EOF
    }
    cross cross-ios.meson ios
    cross cross-darwin.meson darwin

    mesonBuild() { # dir cross options...
      local dir=$1 file=$PWD/$2; shift 2
      (cd "$dir" \
        && meson setup _build --cross-file "$file" --prefix="$out" --buildtype=release \
             --default-library=static --wrap-mode=nodownload "$@" \
        && meson compile -C _build -j "$NIX_BUILD_CORES" \
        && meson install -C _build)
    }

    (cd libffi-3.5.0 \
      && ./configure --host=aarch64-apple-darwin --prefix="$out" --enable-static --disable-shared \
      && make -j"$NIX_BUILD_CORES" && make install)
    mesonBuild glib-2.83.0 cross-ios.meson \
      -Dtests=false -Dnls=disabled -Dintrospection=disabled -Dselinux=disabled -Dlibmount=disabled \
      -Ddtrace=disabled -Dman-pages=disabled -Dglib_debug=disabled -Dxattr=false
    mesonBuild libslirp-v4.9.1 cross-darwin.meson
  '';
}
