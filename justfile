# Build Henge from source with Nix, sign it with your own Apple team and install it
# on a connected iPhone or iPad. See docs/nix.md.
#
#   HENGE_TEAM=ABCDE12345 HENGE_BUNDLE_ID=com.example.henge just deploy

set shell := ["bash", "-euo", "pipefail", "-c"]

team := env("HENGE_TEAM", "")
bundle_id := env("HENGE_BUNDLE_ID", "")
device := env("HENGE_DEVICE", "")
# "paid" signs extended virtual addressing too; a free personal team uses "personal".
entitlements := env("HENGE_ENTITLEMENTS_KIND", "paid")

signing := "build/signing"

# Xcode, not the Apple SDKs a Nix shell may put first on PATH, has to answer xcrun.
xcode_env := "unset SDKROOT DEVELOPER_DIR NIX_CFLAGS_COMPILE NIX_LDFLAGS MACOSX_DEPLOYMENT_TARGET; export PATH=/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

[private]
default:
    @just --list

# Build Henge.ipa from source (result-ipa/Henge.ipa)
build:
    nix build -L -o result-ipa .#ipa

# List the iPhones and iPads this Mac can install to
devices:
    #!/usr/bin/env bash
    set -euo pipefail
    {{ xcode_env }}
    out="$(mktemp)"; trap 'rm -f "$out"' EXIT
    xcrun devicectl list devices --json-output "$out" > /dev/null
    jq -r '.result.devices[] | select(.hardwareProperties.reality == "physical")
      | [.hardwareProperties.udid, .hardwareProperties.deviceType, .deviceProperties.osVersionNumber,
         .connectionProperties.tunnelState, "developer mode " + .deviceProperties.developerModeStatus,
         .deviceProperties.name] | @tsv' "$out"

# Let Xcode create your team's certificate, App IDs and profiles, and register the device (once per team, device or year)
provision team=team bundle_id=bundle_id device=device:
    #!/usr/bin/env bash
    set -euo pipefail
    {{ xcode_env }}
    [ -n "{{ team }}" ] || { echo "set HENGE_TEAM (your Apple team ID)" >&2; exit 1; }
    [ -n "{{ bundle_id }}" ] || { echo "set HENGE_BUNDLE_ID (e.g. com.yourname.henge)" >&2; exit 1; }
    udid="$({{ just_executable() }} _device '{{ device }}')"
    root="$PWD"; mkdir -p "{{ signing }}"

    # The engine and its libraries come from the same Nix derivations as the IPA.
    libs="$(nix build --no-link --print-out-paths .#iosLibs)"
    pairing="$(nix build --no-link --print-out-paths .#rppairing)"
    stikjit="$(nix build --no-link --print-out-paths .#stikjitFramework)"
    export PATH="$PATH:$(nix build --no-link --print-out-paths --inputs-from . nixpkgs#cmake nixpkgs#xcodegen | sed 's|$|/bin|' | paste -sd: -)"
    rm -rf ios/deps && mkdir -p ios/deps/lib
    cp -R "$stikjit/StikJIT.xcframework" ios/deps/
    cp "$pairing/lib/libhusk_rppairing.a" ios/deps/lib/
    for l in libslirp libglib-2.0 libintl libpcre2-8; do cp "$libs/lib/$l.a" ios/deps/lib/; done
    chmod -R u+w ios/deps
    export IOS_DEPS="$libs"

    ent="$root/{{ signing }}/app.entitlements"
    case "{{ entitlements }}" in
      paid)
        cp ios/Henge/Henge.entitlements "$ent"
        # No team can sign it; iOS 26 and later get executable memory from the attached debugger instead.
        plutil -remove dynamic-codesigning "$ent" ;;
      personal) cp ios/Henge/Henge.personal.entitlements "$ent" ;;
      *) echo "HENGE_ENTITLEMENTS_KIND must be paid or personal" >&2; exit 1 ;;
    esac

    cd ios
    xcodegen generate > /dev/null
    # Building for the device is what registers it and puts it in the profiles.
    if ! xcodebuild -project Henge.xcodeproj -scheme Henge -configuration Release \
        -destination "id=$udid" -derivedDataPath build/ProvisionDD \
        -allowProvisioningUpdates -allowProvisioningDeviceRegistration \
        DEVELOPMENT_TEAM="{{ team }}" CODE_SIGN_STYLE=Automatic \
        HENGE_BUNDLE_IDENTIFIER="{{ bundle_id }}" HENGE_ENTITLEMENTS="$ent" \
        build > "$root/{{ signing }}/provision.log" 2>&1; then
      grep -E 'error:' "$root/{{ signing }}/provision.log" | sort -u >&2
      echo "provisioning failed; full log: {{ signing }}/provision.log" >&2
      exit 1
    fi

    app=build/ProvisionDD/Build/Products/Release-iphoneos/Henge.app
    for part in "$app" "$app/PlugIns/HengeJITHelper.appex"; do
      name="$(basename "$part" | cut -d. -f1)"
      cp "$part/embedded.mobileprovision" "$root/{{ signing }}/$name.mobileprovision"
      codesign -d --entitlements - --xml "$part" > "$root/{{ signing }}/$name.entitlements" 2> /dev/null
    done
    authority="$(codesign -dvv "$app" 2>&1 | sed -n 's/^Authority=//p' | head -1)"
    security find-identity -v -p codesigning | grep -F "\"$authority\"" | awk '{print $2}' | head -1 > "$root/{{ signing }}/identity"
    echo "{{ team }} {{ bundle_id }}" > "$root/{{ signing }}/for"
    echo "provisioned {{ bundle_id }} for $udid, signing as $authority"

# Re-sign the Nix-built Henge.ipa with the provisioned identity and profiles (build/signing/Henge.app)
sign bundle_id=bundle_id:
    #!/usr/bin/env bash
    set -euo pipefail
    {{ xcode_env }}
    s="{{ signing }}"
    [ -s "$s/identity" ] || { echo "run 'just provision' first" >&2; exit 1; }
    [ "$(cut -d' ' -f2 "$s/for")" = "{{ bundle_id }}" ] || { echo "provisioned for $(cat "$s/for"), not {{ bundle_id }}: run 'just provision'" >&2; exit 1; }
    [ -f result-ipa/Henge.ipa ] || {{ just_executable() }} build
    rm -rf "$s/Payload" && unzip -q result-ipa/Henge.ipa -d "$s"
    app="$s/Payload/Henge.app"; ext="$app/PlugIns/HengeJITHelper.appex"; id="$(cat "$s/identity")"
    chmod -R u+w "$app"
    plutil -replace CFBundleIdentifier -string "{{ bundle_id }}" "$app/Info.plist"
    plutil -replace CFBundleIdentifier -string "{{ bundle_id }}.JITHelper" "$ext/Info.plist"
    plutil -replace HengeBuildCommit -string "$(git rev-parse --short=12 HEAD)" "$app/Info.plist"
    cp "$s/Henge.mobileprovision" "$app/embedded.mobileprovision"
    cp "$s/HengeJITHelper.mobileprovision" "$ext/embedded.mobileprovision"
    # Inside out: the framework, then the extension, then the app.
    codesign -f -s "$id" --timestamp=none "$app/Frameworks/StikJIT.framework"
    codesign -f -s "$id" --timestamp=none --entitlements "$s/HengeJITHelper.entitlements" "$ext"
    codesign -f -s "$id" --timestamp=none --entitlements "$s/Henge.entitlements" "$app"
    codesign --verify --deep --strict "$app"
    echo "signed $app"

# Install the signed app on the device
install device=device:
    #!/usr/bin/env bash
    set -euo pipefail
    {{ xcode_env }}
    udid="$({{ just_executable() }} _device '{{ device }}')"
    xcrun devicectl device install app --device "$udid" "{{ signing }}/Payload/Henge.app"

# Build, provision when needed, sign and install, in one go
deploy team=team bundle_id=bundle_id device=device: build
    #!/usr/bin/env bash
    set -euo pipefail
    {{ xcode_env }}
    udid="$({{ just_executable() }} _device '{{ device }}')"
    s="{{ signing }}"
    fresh() {
      [ -s "$s/identity" ] && [ "$(cat "$s/for" 2> /dev/null)" = "{{ team }} {{ bundle_id }}" ] || return 1
      local p; for p in Henge HengeJITHelper; do
        security cms -D -i "$s/$p.mobileprovision" > "$s/$p.profile.plist" 2> /dev/null || return 1
        plutil -extract ProvisionedDevices json -o - "$s/$p.profile.plist" | grep -q "\"$udid\"" || return 1
        # Renew a month ahead of expiry.
        [ "$(date -j -f %Y-%m-%dT%H:%M:%SZ "$(plutil -extract ExpirationDate raw -o - "$s/$p.profile.plist")" +%s)" -gt "$(( $(date +%s) + 30 * 86400 ))" ] || return 1
      done
    }
    fresh || {{ just_executable() }} provision "{{ team }}" "{{ bundle_id }}" "$udid"
    {{ just_executable() }} sign "{{ bundle_id }}"
    {{ just_executable() }} install "$udid"

# The UDID to use: the one given, else the only connected iPhone or iPad
[private]
_device given="":
    #!/usr/bin/env bash
    set -euo pipefail
    {{ xcode_env }}
    if [ -n "{{ given }}" ]; then echo "{{ given }}"; exit 0; fi
    out="$(mktemp)"; trap 'rm -f "$out"' EXIT
    xcrun devicectl list devices --json-output "$out" > /dev/null
    found="$(jq -r '.result.devices[] | select(.hardwareProperties.reality == "physical"
      and .connectionProperties.tunnelState == "connected") | .hardwareProperties.udid' "$out")"
    case "$(printf '%s' "$found" | grep -c .)" in
      1) echo "$found" ;;
      0) echo "no iPhone or iPad is connected (see 'just devices')" >&2; exit 1 ;;
      *) echo "several devices are connected; set HENGE_DEVICE to one of them (see 'just devices')" >&2; exit 1 ;;
    esac
