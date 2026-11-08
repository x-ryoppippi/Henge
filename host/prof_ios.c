/* A sampling profiler for the iOS build (host/main.c has the Mac one).
   Every guest CPU thread registers itself; a sampler thread reads each one's PC
   1000 times a second and sorts it into: asleep, in translated code, or in a host
   function (kept as an offset into the app image, symbolized afterwards with the
   unstripped binary: tools/prof_symbolize.sh). Every HV_PROF_IOS_SECS seconds
   (default 30) the totals go to stderr, which Henge mirrors to henge-native.log.
   Off unless HV_PROF_IOS is set (Documents/henge-env.txt can set it): reading a running
   thread's PC stops it for a moment, so sampling is not free. */
#include <dlfcn.h>
#include <mach/mach.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "../core/machine.h"
#include "vm.h"

extern void (*hv_prof_register)(void);

#define MAX_THREADS 16
#define SLOTS 4096
static thread_act_t threads[MAX_THREADS];
static _Atomic int nthreads;
static struct { uintptr_t off; uint32_t n; } slots[SLOTS];
static uint64_t total, idle, jit, other_image, unknown;
static uintptr_t image_base;

static void reg(void) {
    int i = atomic_fetch_add(&nthreads, 1);
    if (i < MAX_THREADS) threads[i] = mach_thread_self();
}

/* Functions in other images (the system's libraries), by the address dladdr names: where the rest goes */
#define OTHER_SLOTS 256
static struct { uintptr_t fn; const char *name; uint32_t n; } other[OTHER_SLOTS];

static void count_other(const Dl_info *info) {
    uintptr_t fn = (uintptr_t)info->dli_saddr;
    uint32_t h = (uint32_t)((fn >> 2) * 2654435761u) % OTHER_SLOTS;
    for (int k = 0; k < 16; k++, h = (h + 1) % OTHER_SLOTS) {
        if (other[h].fn == fn) { other[h].n++; return; }
        if (!other[h].n) { other[h].fn = fn, other[h].name = info->dli_sname, other[h].n = 1; return; }
    }
}

static void count_host(uintptr_t pc) {
    Dl_info info;
    if (!dladdr((void *)pc, &info)) { other_image++; return; }
    if ((uintptr_t)info.dli_fbase != image_base) { other_image++; count_other(&info); return; }
    uintptr_t off = pc - image_base;
    uint32_t h = (uint32_t)((off >> 2) * 2654435761u) % SLOTS;
    for (int k = 0; k < 64; k++, h = (h + 1) % SLOTS) {
        if (slots[h].off == off) { slots[h].n++; return; }
        if (!slots[h].n) { slots[h].off = off, slots[h].n = 1; return; }
    }
    unknown++;
}

static int by_count(const void *a, const void *b) {
    const __typeof__(slots[0]) *x = a, *y = b;
    return (int)y->n - (int)x->n;
}

static int by_count_other(const void *a, const void *b) {
    const __typeof__(other[0]) *x = a, *y = b;
    return (int)y->n - (int)x->n;
}

static void dump(void) {
    static __typeof__(slots[0]) sorted[SLOTS];
    memcpy(sorted, slots, sizeof slots);
    qsort(sorted, SLOTS, sizeof sorted[0], by_count);
    uint64_t busy = total - idle;
    fprintf(stderr, "hvprof: %llu samples, idle %.1f%%, of busy: jit %.1f%%, other images %.1f%%, base %#lx\n",
            (unsigned long long)total, total ? 100.0 * idle / total : 0, busy ? 100.0 * jit / busy : 0,
            busy ? 100.0 * other_image / busy : 0, (unsigned long)image_base);
    for (int i = 0; i < 40 && sorted[i].n; i++)
        fprintf(stderr, "hvprof: %5.2f%% +%#lx\n", busy ? 100.0 * sorted[i].n / busy : 0, (unsigned long)sorted[i].off);
    static __typeof__(other[0]) osorted[OTHER_SLOTS];
    memcpy(osorted, other, sizeof other);
    qsort(osorted, OTHER_SLOTS, sizeof osorted[0], by_count_other);
    for (int i = 0; i < 12 && osorted[i].n; i++)
        fprintf(stderr, "hvprof: %5.2f%% %s (other image)\n", busy ? 100.0 * osorted[i].n / busy : 0, osorted[i].name ? osorted[i].name : "?");
    memset(other, 0, sizeof other);
    memset(slots, 0, sizeof slots);
    total = idle = jit = other_image = unknown = 0;
}

static void *sampler(void *unused) {
    (void)unused;
    int secs = getenv("HV_PROF_IOS_SECS") ? atoi(getenv("HV_PROF_IOS_SECS")) : 30;
    uint64_t ticks = 0;
    for (;;) {
        usleep(1000);
        HvMachine *m = hv_vm_machine();
        uintptr_t lo = m ? (uintptr_t)m->jit.base : 0, hi = m ? lo + m->jit.capacity * 4 : 0;
        int n = atomic_load(&nthreads);
        for (int i = 0; i < n && i < MAX_THREADS; i++) {
            thread_basic_info_data_t ti;
            mach_msg_type_number_t ni = THREAD_BASIC_INFO_COUNT;
            if (thread_info(threads[i], THREAD_BASIC_INFO, (thread_info_t)&ti, &ni) != KERN_SUCCESS) continue;
            total++;
            if (ti.run_state != TH_STATE_RUNNING) { idle++; continue; }
            arm_thread_state64_t st;
            mach_msg_type_number_t sn = ARM_THREAD_STATE64_COUNT;
            if (thread_get_state(threads[i], ARM_THREAD_STATE64, (thread_state_t)&st, &sn) != KERN_SUCCESS) { unknown++; continue; }
            uintptr_t pc = (uintptr_t)arm_thread_state64_get_pc(st);
            if (pc >= lo && pc < hi) jit++;
            else count_host(pc);
        }
        if (++ticks % (uint64_t)(secs * 1000) == 0) dump();
    }
    return NULL;
}

void hv_prof_ios_start(void) {
    static int started;
    if (started++) return;
    if (!getenv("HV_PROF_IOS")) return;
    Dl_info info;
    if (dladdr((void *)hv_prof_ios_start, &info)) image_base = (uintptr_t)info.dli_fbase;
    hv_prof_register = reg;
    pthread_t t;
    pthread_create(&t, NULL, sampler, NULL);
}
