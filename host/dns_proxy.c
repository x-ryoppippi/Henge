/* The guest's DNS through the host's own resolver (issue #34).

   libslirp answers the guest's queries to 10.0.2.3 by sending them on to the host's first IPv4 name server, which
   fails where there is none (IPv6-only networks) or where the right one is somewhere else (a VPN's split DNS, iOS's
   per-interface resolvers). Here each UDP query is answered from DNSServiceQueryRecord instead, the resolver every
   iOS and macOS app uses: it knows the system's whole DNS configuration and caches.

   One query, one short-lived thread. The answer goes back as the frame 10.0.2.3 would have sent. A query this does
   not understand, or one whose lookup cannot even start, goes to libslirp as before; so do .local names (asking for
   those on iOS brings up the local network privacy prompt). */
#include "dns_proxy.h"
#include "machine.h"

#include <dns_sd.h>
#include <poll.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>

#define MAX_QUERY 512
#define MAX_ANSWERS 64
#define MAX_RDATA 1024
#define MAX_INFLIGHT 64
#define TIMEOUT_MS 4000

typedef struct {
    uint16_t type, rclass, rdlen;
    uint32_t ttl;
    uint8_t rdata[MAX_RDATA];
} Answer;

typedef struct {
    HvDnsSend send;
    uint8_t frame[14 + 60 + 8 + MAX_QUERY]; /* the guest's query, for its addresses (and to hand back) */
    uint32_t len;
    uint32_t ip_off, udp_off, dns_off, qend; /* offsets into frame; qend: the end of the question */
    uint16_t qtype, qclass, udp_max;
    int edns;
    char name[1025];
    Answer ans[MAX_ANSWERS];
    int nans, nodata, failed, done;
} Query;

static atomic_int inflight, unanswered_in_a_row;
static atomic_ullong answered, handed_back, timed_out;
#define GIVE_UP 8 /* lookups in a row the resolver took but never answered: back to libslirp for the session */

static int64_t now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] << 8 | p[1]); }
static void wr16(uint8_t *p, uint32_t v) { p[0] = (uint8_t)(v >> 8), p[1] = (uint8_t)v; }

/* The question's name in the form DNSServiceQueryRecord takes ("a.b.c.", dots and backslashes in labels escaped,
   other unprintable bytes as \DDD). Returns the offset after the name's terminating zero, or 0. */
static uint32_t read_name(const uint8_t *m, uint32_t len, uint32_t off, char *out, size_t cap, int *local) {
    size_t n = 0;
    const uint8_t *last = NULL;
    uint8_t lastlen = 0;
    for (;;) {
        if (off >= len) return 0;
        uint8_t l = m[off++];
        if (!l) break;
        if (l & 0xc0) return 0; /* no compression in a question we take */
        if (off + l > len) return 0;
        last = m + off, lastlen = l;
        for (uint8_t i = 0; i < l; i++) {
            uint8_t c = m[off + i];
            if (n + 5 >= cap) return 0;
            if (c == '.' || c == '\\') out[n++] = '\\', out[n++] = (char)c;
            else if (c <= ' ' || c >= 0x7f) n += (size_t)snprintf(out + n, cap - n, "\\%03u", c);
            else out[n++] = (char)c;
        }
        off += l;
        if (n + 2 >= cap) return 0;
        out[n++] = '.';
    }
    if (!n) out[n++] = '.';
    out[n] = 0;
    *local = last && lastlen == 5 && !strncasecmp((const char *)last, "local", 5);
    return off;
}

static void DNSSD_API on_record(DNSServiceRef ref, DNSServiceFlags flags, uint32_t ifindex, DNSServiceErrorType err,
                                const char *fullname, uint16_t rrtype, uint16_t rrclass, uint16_t rdlen,
                                const void *rdata, uint32_t ttl, void *ctx) {
    (void)ref, (void)ifindex, (void)fullname;
    Query *q = ctx;
    int more = (flags & kDNSServiceFlagsMoreComing) != 0;
    if (err == kDNSServiceErr_NoSuchRecord || err == kDNSServiceErr_NoSuchName) {
        q->nodata = 1; /* no such name, or nothing of this type: either way an empty answer */
    } else if (err) {
        q->failed = 1;
        q->done = 1;
        return;
    } else if (flags & kDNSServiceFlagsAdd) {
        /* Records of the type asked for, under the name asked for: a CNAME chain on the way (an intermediate
           result) is left out, the guest only needs what is at its end. */
        int wanted = rrtype == q->qtype || (q->qtype == 255 && rrtype != 5);
        if (wanted && q->nans < MAX_ANSWERS && rdlen <= MAX_RDATA) {
            Answer *a = &q->ans[q->nans++];
            a->type = rrtype, a->rclass = rrclass, a->rdlen = rdlen, a->ttl = ttl;
            memcpy(a->rdata, rdata, rdlen);
        }
    }
    /* Done at the end of a batch that answered (or said there is nothing); a batch of CNAMEs alone is not the end. */
    if (!more && (q->nans || q->nodata)) q->done = 1;
}

static uint16_t ip_checksum(const uint8_t *p, uint32_t n) {
    uint32_t s = 0;
    for (uint32_t i = 0; i + 1 < n; i += 2) s += (uint32_t)(p[i] << 8 | p[i + 1]);
    if (n & 1) s += (uint32_t)p[n - 1] << 8;
    while (s >> 16) s = (s & 0xffff) + (s >> 16);
    return (uint16_t)~s;
}

/* The reply frame, from 10.0.2.3:53 to the guest's port, with rcode and the answers that fit. */
static void reply(Query *q, int rcode) {
    static const size_t cap = 14 + 20 + 8 + 4096;
    uint8_t *f = calloc(1, cap);
    if (!f) return;
    const uint8_t *in = q->frame;
    memcpy(f, in + 6, 6);     /* to the guest's MAC */
    memcpy(f + 6, in, 6);     /* from the one it sent to */
    f[12] = 0x08, f[13] = 0x00;
    uint8_t *ip = f + 14, *udp = ip + 20, *dns = udp + 8;
    uint32_t qlen = q->qend - q->dns_off;  /* header and question, as asked */
    memcpy(dns, in + q->dns_off, qlen);
    const uint8_t *qh = in + q->dns_off;
    dns[2] = (uint8_t)(0x80 | (qh[2] & 0x79)); /* QR, the opcode and RD as asked; not authoritative, not truncated */
    dns[3] = (uint8_t)(0x80 | (rcode & 15));   /* RA */
    wr16(dns + 4, 1);
    wr16(dns + 8, 0);
    uint32_t limit = q->udp_max, n = qlen, nans = 0;
    int truncated = 0;
    for (int i = 0; i < q->nans; i++) {
        const Answer *a = &q->ans[i];
        uint32_t need = 2 + 10 + a->rdlen;
        if (n + need + (q->edns ? 11 : 0) > limit) {
            truncated = 1;
            break;
        }
        uint8_t *r = dns + n;
        wr16(r, 0xc00c); /* the question's name */
        wr16(r + 2, a->type);
        wr16(r + 4, a->rclass);
        r[6] = (uint8_t)(a->ttl >> 24), r[7] = (uint8_t)(a->ttl >> 16), r[8] = (uint8_t)(a->ttl >> 8), r[9] = (uint8_t)a->ttl;
        wr16(r + 10, a->rdlen);
        memcpy(r + 12, a->rdata, a->rdlen);
        n += need;
        nans++;
    }
    if (truncated) dns[2] |= 0x02;
    wr16(dns + 6, nans);
    if (q->edns) { /* an OPT record back, as the guest sent one */
        uint8_t *r = dns + n;
        r[0] = 0;
        wr16(r + 1, 41);
        wr16(r + 3, 1232);
        memset(r + 5, 0, 6);
        n += 11;
        wr16(dns + 10, 1);
    } else {
        wr16(dns + 10, 0);
    }
    uint32_t udplen = 8 + n, iplen = 20 + udplen;
    const uint8_t *qip = in + q->ip_off, *qudp = in + q->udp_off;
    ip[0] = 0x45;
    wr16(ip + 2, iplen);
    ip[8] = 64;
    ip[9] = 17;
    memcpy(ip + 12, qip + 16, 4); /* from 10.0.2.3 */
    memcpy(ip + 16, qip + 12, 4); /* to the guest */
    wr16(ip + 10, ip_checksum(ip, 20));
    wr16(udp, 53);
    memcpy(udp + 2, qudp, 2);
    wr16(udp + 4, udplen);
    wr16(udp + 6, 0); /* no UDP checksum: allowed over IPv4 */
    q->send(f, 14 + iplen, 1);
    free(f);
}

static void *lookup(void *arg) {
    Query *q = arg;
    DNSServiceRef ref = NULL;
    DNSServiceErrorType err = DNSServiceQueryRecord(&ref, kDNSServiceFlagsReturnIntermediates, 0, q->name, q->qtype,
                                                    q->qclass, on_record, q);
    if (err != kDNSServiceErr_NoError) {
        q->send(q->frame, q->len, 0); /* libslirp, as before */
        atomic_fetch_add(&handed_back, 1);
    } else {
        int fd = DNSServiceRefSockFD(ref);
        int64_t end = now_ms() + TIMEOUT_MS;
        while (!q->done) {
            int64_t left = end - now_ms();
            if (left <= 0) break;
            struct pollfd p = {fd, POLLIN, 0};
            int r = poll(&p, 1, (int)left);
            if (r > 0 && DNSServiceProcessResult(ref) != kDNSServiceErr_NoError) {
                q->failed = 1;
                break;
            }
        }
        DNSServiceRefDeallocate(ref);
        if (!q->done || q->failed) {
            atomic_fetch_add(&timed_out, 1);
            if (atomic_fetch_add(&unanswered_in_a_row, 1) + 1 == GIVE_UP && hv_knob[HV_K_DNSPROXY]) {
                hv_knob[HV_K_DNSPROXY] = 0; /* not hv_knob_set: no translation depends on it */
                fprintf(stderr, "hengevm: dns: %d lookups in a row unanswered by the system resolver: libslirp answers from now on\n", GIVE_UP);
            }
        } else {
            atomic_store(&unanswered_in_a_row, 0);
        }
        reply(q, q->done && !q->failed ? 0 : 2); /* SERVFAIL when nothing came */
        atomic_fetch_add(&answered, 1);
    }
    free(q);
    atomic_fetch_sub(&inflight, 1);
    return NULL;
}

int hv_dns_proxy_on(void) {
    return hv_knob[HV_K_DNSPROXY] != 0; /* HV_DNS_PROXY, on unless 0 (machine.c) */
}

int hv_dns_proxy_take(const uint8_t *p, uint32_t len, HvDnsSend send) {
    if (!hv_dns_proxy_on() || len < 14 + 20 + 8 + 12 || p[12] != 0x08 || p[13] != 0x00) return 0;
    const uint8_t *ip = p + 14;
    uint32_t ihl = (uint32_t)(ip[0] & 15) * 4;
    if ((ip[0] >> 4) != 4 || ihl < 20 || ip[9] != 17) return 0;
    if (ip[16] != 10 || ip[17] != 0 || ip[18] != 2 || ip[19] != 3) return 0; /* to 10.0.2.3 */
    if ((rd16(ip + 6) & 0x3fff) != 0) return 0;                            /* fragments: libslirp's */
    uint32_t iplen = rd16(ip + 2);
    if (iplen > len - 14 || iplen < ihl + 8 + 12) return 0;
    const uint8_t *udp = ip + ihl;
    if (rd16(udp + 2) != 53) return 0;
    uint32_t udplen = rd16(udp + 4);
    if (udplen < 8 + 12 || udplen > iplen - ihl || udplen - 8 > MAX_QUERY) return 0;
    const uint8_t *dns = udp + 8;
    uint32_t dlen = udplen - 8;
    /* A standard query with one question and nothing else but maybe an OPT record. */
    if ((dns[2] & 0xf8) != 0 || rd16(dns + 4) != 1 || rd16(dns + 6) || rd16(dns + 8) || rd16(dns + 10) > 1) return 0;
    Query *q = calloc(1, sizeof *q);
    if (!q) return 0;
    int local = 0;
    uint32_t off = read_name(dns, dlen, 12, q->name, sizeof q->name, &local);
    if (!off || off + 4 > dlen || local) {
        free(q);
        return 0;
    }
    q->qtype = rd16(dns + off), q->qclass = rd16(dns + off + 2);
    uint32_t qend = off + 4;
    q->udp_max = 512;
    if (rd16(dns + 10) == 1) { /* EDNS: its OPT record says how big an answer may be */
        if (qend + 11 > dlen || dns[qend] != 0 || rd16(dns + qend + 1) != 41) {
            free(q);
            return 0;
        }
        q->edns = 1;
        uint32_t size = rd16(dns + qend + 3);
        q->udp_max = (uint16_t)(size < 512 ? 512 : size > 1472 ? 1472 : size); /* one unfragmented frame */
    }
    if (q->qclass != 1 || q->qtype == 0 || q->qtype == 252 || q->qtype == 251) { /* IN only; no zone transfers */
        free(q);
        return 0;
    }
    if (atomic_fetch_add(&inflight, 1) >= MAX_INFLIGHT) {
        atomic_fetch_sub(&inflight, 1);
        free(q);
        return 0;
    }
    memcpy(q->frame, p, 14 + iplen);
    q->len = 14 + iplen;
    q->ip_off = 14, q->udp_off = 14 + ihl, q->dns_off = 14 + ihl + 8, q->qend = q->dns_off + qend;
    q->send = send;
    pthread_t t;
    pthread_attr_t at;
    pthread_attr_init(&at);
    pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
    pthread_attr_setstacksize(&at, 256 * 1024);
    int ok = pthread_create(&t, &at, lookup, q) == 0;
    pthread_attr_destroy(&at);
    if (!ok) {
        atomic_fetch_sub(&inflight, 1);
        free(q);
        return 0;
    }
    return 1;
}

void hv_dns_proxy_stats(uint64_t *n_answered, uint64_t *n_handed_back, uint64_t *n_timed_out) {
    *n_answered = atomic_load(&answered);
    *n_handed_back = atomic_load(&handed_back);
    *n_timed_out = atomic_load(&timed_out);
}
