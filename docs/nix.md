# Building and installing with Nix

`flake.nix` builds Henge from pinned sources, and the `justfile` signs that build with
your own Apple team and installs it on a connected iPhone or iPad.

## Requirements

- An Apple silicon Mac with Xcode 27 or later, selected with `xcode-select`. The iOS SDK
  only ships with Xcode, so the derivations use it from outside the Nix store and need
  `sandbox = false` or `sandbox = relaxed` in `nix.conf`.
- Nix with flakes. `nix develop` gives a shell with `just` and the build tools; or run
  `nix run nixpkgs#just -- <recipe>`.
- To install: your Apple ID added in Xcode › Settings › Apple Accounts, and the device
  paired with the Mac with Developer Mode on (Settings › Privacy & Security › Developer Mode).

## Build

```sh
nix build .#ipa    # result/Henge.ipa
nix build .#tipa   # for TrollStore
```

Husk and StikJIT come from the revisions `flake.lock` pins, and every tarball is checked
against its hash. StikJIT still links the `libidevice_ffi.a` its repository ships.

## Install on your device

```sh
HENGE_TEAM=ABCDE12345 HENGE_BUNDLE_ID=com.yourname.henge just deploy
```

| Variable | Meaning |
| --- | --- |
| `HENGE_TEAM` | Your team ID. It is in any of your team's profiles, or under Membership details on developer.apple.com. |
| `HENGE_BUNDLE_ID` | A bundle ID of your own; `com.henge.app` belongs to someone else. The JIT helper gets `<id>.JITHelper`. |
| `HENGE_DEVICE` | The device's UDID. Optional when only one device is connected (`just devices` lists them). |
| `HENGE_ENTITLEMENTS_KIND` | `paid` (default) or `personal` for a free personal team, which cannot sign extended virtual addressing and expires after 7 days. |

`deploy` builds the IPA, provisions when it has to, re-signs and installs:

- `just provision` builds the app once with Xcode's automatic signing. That creates the team's
  development certificate, the two App IDs with their capabilities, and profiles that include
  the device. It runs again only for a new team, bundle ID or device, or a month before the
  profiles expire. What it keeps is in `build/signing/`.
- `just sign` re-signs the Nix-built IPA with that identity and those profiles; the app Xcode
  built is not what gets installed.
- `just install` installs it with `devicectl`.

A paid team signs everything in `ios/Henge/Henge.entitlements` except `dynamic-codesigning`,
which no developer team can sign. On iOS 26 and later Henge gets executable memory from an
attached debugger instead, so that costs nothing there.

## Turning on JIT

On iOS 26 and later, Henge's built-in StikJIT helper is the debugger; no jailbreak is needed.
Install LocalDevVPN from the App Store, tap **Turn On JIT** in Henge and pick **Pair on this
iPad** (iOS 27) or import a pairing file made on a computer.
