// SPDX-License-Identifier: GPL-2.0-or-later
import Foundation
import dnssd

/// Finds the port this device's RemotePairing service listens on, for when Henge could not.
///
/// Since iOS 26 the device picks that port at boot and advertises it as `_remotepairing._tcp`; the fixed
/// 49152 StikJIT falls back to is closed then, so every attempt times out at 10.7.0.1:49152. Henge's own lookup
/// (RemotePairingPortDiscovery) connects to the service to learn its port, and on some devices that never
/// becomes ready. This reads the port from the service's SRV record with DNSServiceResolve instead, which
/// needs no connection.
enum RSDPortResolver {
    struct Result {
        var port: UInt16?
        /// What happened, for the log: the port found, or why there is none.
        var detail: String
    }

    private static let serviceType = "_remotepairing._tcp"

    /// Blocks for at most `timeout`. Call it off the main thread.
    static func resolve(timeout: TimeInterval = 4) -> Result {
        let state = State()
        let queue = DispatchQueue(label: "com.henge.app.jit-helper.rsd-port")
        let context = Unmanaged.passRetained(state)
        defer { context.release() }

        var browseRef: DNSServiceRef?
        let browseError = DNSServiceBrowse(&browseRef, 0, 0, serviceType, "local.", browseCallback,
                                           context.toOpaque())
        guard browseError == kDNSServiceErr_NoError, let browseRef else {
            return Result(port: nil, detail: "browse could not start (\(describe(browseError)))")
        }
        state.queue = queue
        DNSServiceSetDispatchQueue(browseRef, queue)

        let found = state.done.wait(timeout: .now() + timeout) == .success
        // Tear down on the queue the callbacks run on, so none runs after its ref is gone.
        return queue.sync {
            DNSServiceRefDeallocate(browseRef)
            state.resolveRefs.forEach(DNSServiceRefDeallocate)
            state.resolveRefs.removeAll()
            state.finished = true
            if found, let port = state.port {
                return Result(port: port, detail: "resolved \(serviceType) to port \(port) on \(state.host ?? "?")")
            }
            var detail = "no port within \(Int(timeout))s: \(state.seen) service(s) seen"
            if let error = state.lastError { detail += ", last error \(describe(error))" }
            return Result(port: nil, detail: detail)
        }
    }

    /// Only touched on `queue` once the browse has started.
    private final class State {
        let done = DispatchSemaphore(value: 0)
        var queue: DispatchQueue?
        var resolveRefs: [DNSServiceRef] = []
        var seen = 0
        var port: UInt16?
        var host: String?
        var lastError: DNSServiceErrorType?
        var finished = false

        func finish(port: UInt16, host: String) {
            guard self.port == nil, !finished else { return }
            self.port = port
            self.host = host
            done.signal()
        }
    }

    private static let browseCallback: DNSServiceBrowseReply = { _, flags, interface, error, name, type, domain, context in
        guard let context else { return }
        let state = Unmanaged<State>.fromOpaque(context).takeUnretainedValue()
        guard !state.finished, state.port == nil else { return }
        guard error == kDNSServiceErr_NoError else {
            state.lastError = error
            return
        }
        guard (flags & kDNSServiceFlagsAdd) != 0, let name, let type, let domain else { return }
        state.seen += 1
        var resolveRef: DNSServiceRef?
        let resolveError = DNSServiceResolve(&resolveRef, 0, interface, name, type, domain, resolveCallback, context)
        guard resolveError == kDNSServiceErr_NoError, let resolveRef, let queue = state.queue else {
            state.lastError = resolveError
            return
        }
        state.resolveRefs.append(resolveRef)
        DNSServiceSetDispatchQueue(resolveRef, queue)
    }

    private static let resolveCallback: DNSServiceResolveReply = { _, _, _, error, _, host, port, _, _, context in
        guard let context else { return }
        let state = Unmanaged<State>.fromOpaque(context).takeUnretainedValue()
        guard error == kDNSServiceErr_NoError else {
            state.lastError = error
            return
        }
        // The port arrives in network byte order.
        let value = UInt16(bigEndian: port)
        guard value != 0 else { return }
        state.finish(port: value, host: host.map { String(cString: $0) } ?? "?")
    }

    private static func describe(_ error: DNSServiceErrorType) -> String {
        switch Int(error) {
        case Int(kDNSServiceErr_PolicyDenied): return "PolicyDenied: Local Network access is off for Henge"
        case Int(kDNSServiceErr_NoAuth): return "NoAuth"
        case Int(kDNSServiceErr_ServiceNotRunning): return "ServiceNotRunning"
        default: return "DNSServiceErrorType \(error)"
        }
    }
}
