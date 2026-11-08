import SwiftUI
import AVFAudio

@main
struct HengeApp: App {
    /// Answers for external display scenes before iOS 27 (LegacyExternalDisplay.swift).
    @UIApplicationDelegateAdaptor(HengeAppDelegate.self) private var appDelegate
    @Environment(\.scenePhase) private var scenePhase

    init() {
        // Order matters: HengeLog takes over stderr, so the engine's and the JIT
        // allocator's first lines land in Documents/henge.log too.
        HengeLog.start()
        HengeLog.log("boot", "address space: \(henge_largest_reservation_gb()) GB can be reserved in one piece")
        EngineTuning.apply()
        HengeLog.logFootprint("app-launch")
        JITBootstrap.noteLaunchState()
        HengeLog.log("app", "increased memory limit: \(Entitlements.increasedMemory ? "yes" : "no"); extended addressing: \(Entitlements.extendedAddressing ? "yes" : "no"); limit \(Entitlements.memoryLimitAtLaunch >> 20) MB")
        // Without the guard, a brk no debugger answers kills the app instead of failing.
        JITBootstrap.installTrapGuard()
        try? AVAudioSession.sharedInstance().setCategory(.playback, options: [.mixWithOthers])
        Machines.shared.collectImports()
        // The display question: does this iOS have Apple's paravirtualized graphics at all?
        for path in ["/System/Library/Frameworks/ParavirtualizedGraphics.framework/ParavirtualizedGraphics",
                     "/System/Library/PrivateFrameworks/ParavirtualizedGraphics.framework/ParavirtualizedGraphics"] {
            HengeLog.log("app", "\(path): \(dlopen(path, RTLD_LAZY) != nil ? "present" : "absent")")
        }
    }

    var body: some Scene {
        WindowGroup {
            RootView()
                .onOpenURL { url in HengeLog.log("app", "opened with \(url.absoluteString)") } // henge:// back from LocalDevVPN
        }
    }
}

struct RootView: View {
    enum Tab: Hashable { case macs, install, settings }

    @ObservedObject private var machines = Machines.shared
    @ObservedObject private var jit = JITCoordinator.shared
    @Environment(\.scenePhase) private var scenePhase
    @State private var running: VirtualMac?
    @State private var tab = Tab.macs

    var body: some View {
        ZStack {
            TabView(selection: $tab) {
                HomeView(running: $running, tab: $tab)
                    .tabItem { Label("Macs", systemImage: "macwindow") }
                    .tag(Tab.macs)
                InstallView()
                    .tabItem { Label("Install", systemImage: "arrow.down.circle") }
                    .tag(Tab.install)
                SettingsView()
                    .tabItem { Label("Settings", systemImage: "gearshape") }
                    .tag(Tab.settings)
            }
            .tint(Desk.accentLight)
            if let running {
                SessionView(mac: running) { self.running = nil }
                    .transition(.opacity)
            }
        }
        .preferredColorScheme(.dark)
        .sheet(isPresented: $jit.showSetup) { JITSetupFlow() }
        .onChange(of: scenePhase) { phase in
            guard phase == .active else { return }
            // Back from StikDebug or TrollStore: claim the JIT region while the debugger is there.
            JITBootstrap.prewarm()
            machines.collectImports()
            jit.enableAtLaunchIfAsked()
        }
        .onChange(of: jit.attachGeneration) { _ in JITBootstrap.prewarm() } // the built-in helper attached
        .task { // testing only: HENGE_AUTOPILOT
            await Autopilot.run { mac in
                // A session still on screen (its Mac stopped) is closed first: the same value again would start nothing.
                guard running != nil else { running = mac; return }
                running = nil
                DispatchQueue.main.asyncAfter(deadline: .now() + 1) { running = mac }
            }
        }
    }
}

extension View {
    /// Husk's screens are written against an iOS 15 shim; Henge needs iOS 17, so this is the real thing.
    func hengeDestination<D: Hashable, C: View>(for data: D.Type, @ViewBuilder destination: @escaping (D) -> C) -> some View {
        navigationDestination(for: data, destination: destination)
    }
}
