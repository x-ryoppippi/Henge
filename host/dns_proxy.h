#ifndef HENGEVM_DNS_PROXY_H
#define HENGEVM_DNS_PROXY_H

#include <stdint.h>

/* The guest's DNS queries to 10.0.2.3, answered through the host's resolver (dns_proxy.c). On unless
   HV_DNS_PROXY=0 (a knob, so it can be switched while the guest runs). send(frame, len, to_guest): the answer
   for the guest (to_guest 1), or the query handed back for libslirp to send on as before (0); called from the
   lookup's own thread. */
typedef void (*HvDnsSend)(const uint8_t *frame, uint32_t len, int to_guest);

int hv_dns_proxy_on(void);
/* A frame the guest sent: 1 if it was a DNS query this took (its answer comes through send), 0 to pass it on. */
int hv_dns_proxy_take(const uint8_t *frame, uint32_t len, HvDnsSend send);
void hv_dns_proxy_stats(uint64_t *answered, uint64_t *handed_back, uint64_t *timed_out);

#endif
