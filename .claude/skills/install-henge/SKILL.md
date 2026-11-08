---
name: install-henge
description: Builds Henge from source with Nix, signs it with the user's own Apple team and installs it on their iPhone or iPad. Use when the user asks to install, deploy, re-sign or update Henge on a device.
---

# Install Henge on a device

Follow `docs/nix.md`; the `justfile` does the work. Run recipes with `just` from `nix develop`, or
`nix run nixpkgs#just -- <recipe>`.

1. Check the device: `just devices`. It must show `connected` and `developer mode enabled`.
   Otherwise ask the user to unlock it, pair it with the Mac, or turn on Developer Mode
   (Settings › Privacy & Security › Developer Mode, then restart).
2. Find the team. Ask the user for their team ID if it is not already known. If they
   do not know it, have them press **Download Manual Profiles** for that team in
   Xcode › Settings › Apple Accounts, then read `TeamIdentifier` from the profiles in
   `~/Library/Developer/Xcode/UserData/Provisioning Profiles`
   (`security cms -D -i <file> | plutil -extract TeamIdentifier.0 raw -o - -`).
   Use only a team the user names as theirs; never pick an employer's or someone
   else's team because its profiles or certificates happen to be on the Mac.
3. Pick a bundle ID the user owns, such as `com.<name>.henge`.
4. Run `HENGE_TEAM=<id> HENGE_BUNDLE_ID=<bundle> just deploy`. Add `HENGE_DEVICE=<udid>`
   when several devices are connected, and `HENGE_ENTITLEMENTS_KIND=personal` for a free
   personal team.
5. If provisioning fails, read `build/signing/provision.log`. An unregistered device or a
   capability the team cannot sign shows up there.

After installing, point the user to **Turn On JIT** in Henge: on iOS 26 and later it uses the
built-in StikJIT helper with LocalDevVPN and a pairing file (**Pair on this iPad** on iOS 27).
