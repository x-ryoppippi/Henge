#include "share.h"
#include "net_slirp.h"
#include <time.h>
/* hengevm: boots the virtual Mac on the macOS host.
   hengevm --rom AVPBooter.vmapple2.bin --disk disk.img --aux auxiliary.bin --ecid N [--seed N] [--ram MB] [--log N] */
#include <CoreFoundation/CoreFoundation.h>
#include <mach/mach.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <dlfcn.h>
#include <CommonCrypto/CommonDigest.h>

#include "dispatch.h"
#include "exception.h"
#include "mmu.h"
#include "pvg.h"
#include "pvg_lite.h"
#include "vm.h"
#include "install.h"

/* The machine lives in vm.c; these name its parts here. */
#define m (*hv_vm_machine())
#define board (*hv_vm_board())
#define cpu (*hv_vm_cpu(0))
static int ncpus = 3; /* --cpus: 3 boots as fast as 4 on a 2+4-core Mac and leaves room for the window */
static volatile sig_atomic_t interrupted;

static void on_signal(int sig) {
    (void)sig;
    interrupted = 1;
    hv_vm_stop();
}

#define reason_name hv_vm_reason_name

typedef struct RunArgs {
    uint64_t brk, brk_hits, dump_pa, dump_len;
    const char *dump_path;
} RunArgs;
static RunArgs run_args;
static uint64_t translate_va;

/* Chained blocks run without returning to the dispatcher; this makes them leave
   every millisecond so timers, devices and interrupts are serviced. */
static mach_port_t cpu_port;
/* --profile: where the CPU thread is right now. */
static void sample_cpu(void) {
    arm_thread_state64_t st;
    mach_msg_type_number_t n = ARM_THREAD_STATE64_COUNT;
    if (thread_suspend(cpu_port) != KERN_SUCCESS) return;
    kern_return_t kr = thread_get_state(cpu_port, ARM_THREAD_STATE64, (thread_state_t)&st, &n);
    thread_resume(cpu_port);
    if (kr != KERN_SUCCESS) return;
    m.prof[m.prof_head & (HV_PROF_RING - 1)] = (uint64_t)arm_thread_state64_get_pc(st);
    __atomic_store_n(&m.prof_head, m.prof_head + 1, __ATOMIC_RELEASE);
}

static void *profile_thread(void *unused) {
    (void)unused;
    for (;;) {
        usleep(1000);
        if (cpu_port) sample_cpu();
    }
    return NULL;
}

/* Debugging: HV_PROF_ALL=1 samples every guest CPU's thread 1000 times a second; "prof NAME" in
   the input file writes what it saw since the last time to NAME: where the host was, and
   which guest code. Guest addresses are symbolized afterwards (scratch/prof.py). */
#define PROF_SLOTS (1u << 18)
typedef struct { uint64_t va; uint32_t n, el; } ProfBlock;
static ProfBlock prof_blocks[PROF_SLOTS];
static struct { void *start; const char *name; uint32_t n; } prof_fn[512];
static int prof_nfn;
static uint64_t prof_total, prof_idle, prof_el[2], prof_unknown, prof_spin;
static uint64_t prof_off[12]; /* translated-code samples by words from their block's start: 0..7, 8-15, 16-31, 32+, other */
/* Per host instruction too, to see where translated code stalls. */
#define PROF_PCS (1u << 20)
typedef struct { uint32_t *pc; uint32_t n; } ProfPc;
static ProfPc *prof_pcs;
static pthread_mutex_t prof_mu = PTHREAD_MUTEX_INITIALIZER;
static void prof_sample(thread_act_t th) {
    arm_thread_state64_t st;
    mach_msg_type_number_t n = ARM_THREAD_STATE64_COUNT;
    thread_basic_info_data_t ti;
    mach_msg_type_number_t ni = THREAD_BASIC_INFO_COUNT;
    if (thread_info(th, THREAD_BASIC_INFO, (thread_info_t)&ti, &ni) != KERN_SUCCESS) return;
    if (ti.run_state != TH_STATE_RUNNING) { /* asleep (WFI) or waiting for a lock */
        pthread_mutex_lock(&prof_mu);
        prof_total++, prof_idle++;
        pthread_mutex_unlock(&prof_mu);
        return;
    }
    if (thread_suspend(th) != KERN_SUCCESS) return;
    kern_return_t kr = thread_get_state(th, ARM_THREAD_STATE64, (thread_state_t)&st, &n);
    thread_resume(th);
    if (kr != KERN_SUCCESS) return;
    uint32_t *pc = (uint32_t *)(uintptr_t)arm_thread_state64_get_pc(st);
    HvMachine *hm = hv_vm_machine();
    pthread_mutex_lock(&prof_mu);
    prof_total++;
    if (pc >= hm->jit.base && pc < hm->jit.base + hm->jit.capacity && hm->nblocks) {
        int lo = 0, hi = hm->nblocks - 1;
        while (lo < hi) {
            int mid = (lo + hi + 1) / 2;
            if (hm->blocks[mid].code && hm->blocks[mid].code <= pc) lo = mid;
            else hi = mid - 1;
        }
        HvBlock *b = &hm->blocks[lo];
        int el = (b->flags & HV_BLOCK_EL1) ? 1 : 0;
        {
            ptrdiff_t off = b->code && pc >= b->code ? pc - b->code : -1;
            prof_off[off < 0 ? 11 : off < 8 ? off : off < 16 ? 8 : off < 32 ? 9 : 10]++;
        }
        prof_el[el]++;
        if (prof_pcs) {
            uint32_t h = (uint32_t)(((uint64_t)(uintptr_t)pc >> 2) * 0x9e3779b1u) & (PROF_PCS - 1);
            for (int k = 0; k < 16; k++, h = (h + 1) & (PROF_PCS - 1))
                if (!prof_pcs[h].n || prof_pcs[h].pc == pc) {
                    prof_pcs[h].pc = pc, prof_pcs[h].n++;
                    break;
                }
        }
        uint32_t h = (uint32_t)((b->va >> 2) * 0x9e3779b1u) & (PROF_SLOTS - 1);
        for (int p = 0; p < 32; p++, h = (h + 1) & (PROF_SLOTS - 1)) {
            if (!prof_blocks[h].n || prof_blocks[h].va == b->va) {
                prof_blocks[h].va = b->va, prof_blocks[h].el = el, prof_blocks[h].n++;
                break;
            }
        }
    } else {
        Dl_info di;
        void *start = dladdr(pc, &di) ? di.dli_saddr : NULL;
        if (!start) prof_unknown++;
        else {
            int k = 0;
            while (k < prof_nfn && prof_fn[k].start != start) k++;
            if (k == prof_nfn && prof_nfn < 512) prof_fn[prof_nfn].start = start, prof_fn[prof_nfn].name = di.dli_sname ? di.dli_sname : "?", prof_nfn++;
            if (k < prof_nfn) prof_fn[k].n++;
        }
    }
    pthread_mutex_unlock(&prof_mu);
}

static thread_act_t prof_threads[HV_MAX_CPUS];
static _Atomic int prof_nthreads;
void hv_prof_register_thread(void) {
    if (getenv("HV_PROF_ALL")) prof_threads[atomic_fetch_add(&prof_nthreads, 1)] = mach_thread_self();
}

static void *prof_thread(void *unused) {
    (void)unused;
    for (;;) {
        usleep(1000);
        for (int i = 0; i < atomic_load(&prof_nthreads); i++) prof_sample(prof_threads[i]);
    }
    return NULL;
}

static int prof_cmp_fn(const void *a, const void *b) {
    return (int)((const __typeof__(prof_fn[0]) *)b)->n - (int)((const __typeof__(prof_fn[0]) *)a)->n;
}
static int prof_cmp_pc(const void *a, const void *b) {
    const ProfPc *x = a, *y = b;
    return x->pc < y->pc ? -1 : x->pc > y->pc;
}
static int prof_cmp_block(const void *a, const void *b) {
    return (int)((const ProfBlock *)b)->n - (int)((const ProfBlock *)a)->n;
}

static void prof_dump(const char *path) {
    pthread_mutex_lock(&prof_mu);
    FILE *f = fopen(path, "w");
    if (f) {
        fprintf(f, "total %llu idle %llu el0 %llu el1 %llu unknown %llu\n", (unsigned long long)prof_total, (unsigned long long)prof_idle, (unsigned long long)prof_el[0], (unsigned long long)prof_el[1], (unsigned long long)prof_unknown);
        hv_kva_hist_dump(f);
        fprintf(f, "offsets");
        for (int k = 0; k < 12; k++) fprintf(f, " %llu", (unsigned long long)prof_off[k]);
        fprintf(f, "\n");
        { /* the kernel's slide, from where its SHA-256 routine is */
            static const uint8_t sig[] = {0x5f, 0x24, 0x03, 0xd5, 0x03, 0x00, 0x00, 0x90, 0x41, 0x00, 0x00, 0xb5, 0xc0, 0x03, 0x5f, 0xd6, 0x63, 0x00, 0x36, 0x91, 0xe4, 0x43, 0x04, 0xd1, 0xff, 0x43, 0x04, 0xd1, 0x80, 0x28, 0x9f, 0x4c};
            HvVmapple *bd = hv_vm_board();
            HvCpu *c0 = hv_vm_cpu(0);
            HvWalk wv;
            hv_mmu_walk(c0, c0->vbar_el1, HV_ACC_READ, 1, &wv);
            uint8_t *at = memmem(bd->ram, bd->ram_size, sig, sizeof sig);
            if (at && !wv.fsc) fprintf(f, "kslide %llx\n", (unsigned long long)(c0->vbar_el1 + (0x70000000ull + (uint64_t)(at - bd->ram) - wv.pa) - 0xfffffe002b76de80ull));
        }
        qsort(prof_fn, (size_t)prof_nfn, sizeof prof_fn[0], prof_cmp_fn);
        for (int i = 0; i < prof_nfn && prof_fn[i].n; i++) fprintf(f, "fn %u %s\n", prof_fn[i].n, prof_fn[i].name);
        static ProfBlock sorted[PROF_SLOTS];
        memcpy(sorted, prof_blocks, sizeof sorted);
        qsort(sorted, PROF_SLOTS, sizeof sorted[0], prof_cmp_block);
        for (uint32_t i = 0; i < 20000 && sorted[i].n; i++) fprintf(f, "b %u %llx %u\n", sorted[i].n, (unsigned long long)sorted[i].va, sorted[i].el);
        if (prof_pcs) { /* each hot run of host code as it stands, with its samples */
            qsort(prof_pcs, PROF_PCS, sizeof prof_pcs[0], prof_cmp_pc);
            HvMachine *pm = hv_vm_machine();
            for (uint32_t i = 0; i < PROF_PCS; i++) {
                if (!prof_pcs[i].n || (prof_pcs[i].n < 8 && !(getenv("HV_PROF_PCS") && atoi(getenv("HV_PROF_PCS")) > 1))) continue;
                fprintf(f, "p %u %llx %08x\n", prof_pcs[i].n, (unsigned long long)(uintptr_t)prof_pcs[i].pc, *prof_pcs[i].pc);
            }
            /* a window of code around the hottest instructions, kept inside the region */
            for (uint32_t i = 0; i < PROF_PCS; i++) {
                if (prof_pcs[i].n < 40) continue;
                uint32_t *c = prof_pcs[i].pc, *lo = c - 24, *hi = c + 8;
                if (lo < pm->jit.base) lo = pm->jit.base;
                if (hi > pm->jit.base + pm->jit.capacity) hi = pm->jit.base + pm->jit.capacity;
                fprintf(f, "w %llx", (unsigned long long)(uintptr_t)lo);
                for (uint32_t *k = lo; k < hi; k++) fprintf(f, " %08x", *k);
                fprintf(f, "\n");
            }
            memset(prof_pcs, 0, PROF_PCS * sizeof prof_pcs[0]);
        }
        fclose(f);
    }
    memset(prof_blocks, 0, sizeof prof_blocks);
    for (int i = 0; i < prof_nfn; i++) prof_fn[i].n = 0;
    prof_total = prof_idle = prof_el[0] = prof_el[1] = prof_unknown = prof_spin = 0;
    memset(prof_off, 0, sizeof prof_off);
    pthread_mutex_unlock(&prof_mu);
}

/* Walks the pointer to a spot on the screen, watching where the display says
   it is, in small steps as a hand makes: macOS accelerates big ones past the target. */
static void pointer_to(int x, int y) {
    const uint8_t *page = hv_pvg_lite_display_page(board.pvg);
    for (int i = 0; page && i < 1500; i++) {
        uint32_t at = *(const volatile uint32_t *)(page + 0xe00);
        int dx = x - (int)(at & 0xffff), dy = y - (int)(at >> 16);
        if (abs(dx) < 3 && abs(dy) < 3) break;
        int sx = dx > 16 ? 16 : dx < -16 ? -16 : dx, sy = dy > 16 ? 16 : dy < -16 ? -16 : dy;
        hv_virtio_input_send(board.mouse, HV_EV_REL, HV_REL_X, sx);
        hv_virtio_input_send(board.mouse, HV_EV_REL, HV_REL_Y, sy);
        hv_virtio_input_send(board.mouse, HV_EV_SYN, 0, 0);
        usleep(30000);
    }
}

static void click_here(void) {
    hv_virtio_input_send(board.mouse, HV_EV_KEY, HV_BTN_LEFT, 1);
    hv_virtio_input_send(board.mouse, HV_EV_SYN, 0, 0);
    usleep(120000);
    hv_virtio_input_send(board.mouse, HV_EV_KEY, HV_BTN_LEFT, 0);
    hv_virtio_input_send(board.mouse, HV_EV_SYN, 0, 0);
}

static void key_event(int code, int down) {
    hv_xhci_key_linux(board.usb, (uint16_t)code, down);
    hv_virtio_input_send(board.keyboard, HV_EV_KEY, (uint16_t)code, down);
    hv_virtio_input_send(board.keyboard, HV_EV_SYN, 0, 0);
}

static void key_press(int code, int shift) {
    if (shift) key_event(42, 1), usleep(30000);
    key_event(code, 1);
    usleep(60000);
    key_event(code, 0);
    if (shift) usleep(30000), key_event(42, 0);
    usleep(60000);
}

static void type_text(const char *t) {
    static const char row1[] = "qwertyuiop", row2[] = "asdfghjkl", row3[] = "zxcvbnm", nums[] = "1234567890", shifted[] = "!@#$%^&*()";
    static const struct { char c; uint8_t code, shift; } extra[] = {
        {' ', 57, 0}, {'.', 52, 0}, {'-', 12, 0}, {'/', 53, 0}, {',', 51, 0}, {';', 39, 0}, {'\'', 40, 0}, {'=', 13, 0}, {'[', 26, 0},
        {']', 27, 0}, {'\\', 43, 0}, {'`', 41, 0}, {'_', 12, 1}, {'+', 13, 1}, {':', 39, 1}, {'"', 40, 1}, {'|', 43, 1}, {'<', 51, 1},
        {'>', 52, 1}, {'?', 53, 1}, {'~', 41, 1}, {'{', 26, 1}, {'}', 27, 1}};
    for (; *t; t++) {
        int c = *t, up = c >= 'A' && c <= 'Z', code = 0;
        if (up) c += 32;
        const char *f;
        if ((f = strchr(row1, c))) code = 16 + (int)(f - row1);
        else if ((f = strchr(row2, c))) code = 30 + (int)(f - row2);
        else if ((f = strchr(row3, c))) code = 44 + (int)(f - row3);
        else if ((f = strchr(nums, c))) code = 2 + (int)(f - nums);
        else if ((f = strchr(shifted, c))) code = 2 + (int)(f - shifted), up = 1;
        else
            for (size_t k = 0; k < sizeof extra / sizeof extra[0]; k++)
                if (extra[k].c == c) code = extra[k].code, up = extra[k].shift;
        if (code) key_press(code, up);
    }
}

/* Debugging: HV_CLICKS="seconds:x,y;seconds:x,y" clicks on a timetable, to drive a guest's UI without a window. */
static void *click_thread(void *unused) {
    (void)unused;
    const char *p = getenv("HV_CLICKS");
    time_t start = time(NULL);
    while (p && *p) {
        int when, x, y, n = 0;
        if (sscanf(p, "%d:%d,%d%n", &when, &x, &y, &n) != 3) break;
        p += n;
        if (*p == ';') p++;
        while (time(NULL) - start < when) sleep(1);
        pointer_to(x, y);
        fprintf(stderr, "hengevm: click (%d,%d) at %ld s\n", x, y, (long)(time(NULL) - start));
        click_here();
    }
    return NULL;
}

uint64_t hook_pages;
/* The guest kernel's call chain, by frame pointers. */
static void guest_frames(HvCpu *c) {
    HvMachine *hm = hv_vm_machine();
    uint64_t fp = c->x[29];
    fprintf(stderr, "  frames: lr %llx", (unsigned long long)c->x[30]);
    for (int i = 0; i < 12 && fp >> 40 == 0xfffffe; i++) {
        uint64_t rec[2];
        HvWalk w;
        hv_mmu_walk(c, fp, HV_ACC_READ, 1, &w);
        if (w.fsc || hv_phys_read(hm, w.pa, rec, 16)) break;
        fprintf(stderr, " %llx", (unsigned long long)(rec[1] & 0xfffffffffffffull) | 0xfff0000000000000ull);
        fp = rec[0];
    }
    fprintf(stderr, "; sha at %llx; process table n/a\n", (unsigned long long)c->pc);
}
/* SHA-256 compression, to know what the guest's should come to. */
static void sha_blocks(uint32_t st[8], const uint8_t *d, size_t n) {
    static const uint32_t K[64] = {0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
#define ROR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
    for (; n; n--, d += 64) {
        uint32_t w[64], a[8];
        for (int i = 0; i < 16; i++) w[i] = (uint32_t)d[i * 4] << 24 | d[i * 4 + 1] << 16 | d[i * 4 + 2] << 8 | d[i * 4 + 3];
        for (int i = 16; i < 64; i++) w[i] = w[i - 16] + (ROR(w[i - 15], 7) ^ ROR(w[i - 15], 18) ^ (w[i - 15] >> 3)) + w[i - 7] + (ROR(w[i - 2], 17) ^ ROR(w[i - 2], 19) ^ (w[i - 2] >> 10));
        memcpy(a, st, sizeof a);
        for (int i = 0; i < 64; i++) {
            uint32_t t1 = a[7] + (ROR(a[4], 6) ^ ROR(a[4], 11) ^ ROR(a[4], 25)) + ((a[4] & a[5]) ^ (~a[4] & a[6])) + K[i] + w[i];
            uint32_t t2 = (ROR(a[0], 2) ^ ROR(a[0], 13) ^ ROR(a[0], 22)) + ((a[0] & a[1]) ^ (a[0] & a[2]) ^ (a[1] & a[2]));
            a[7] = a[6], a[6] = a[5], a[5] = a[4], a[4] = a[3] + t1, a[3] = a[2], a[2] = a[1], a[1] = a[0], a[0] = t1 + t2;
        }
        for (int i = 0; i < 8; i++) st[i] += a[i];
    }
#undef ROR
}
static void memcmp_hook(HvCpu *c) {
    if (c->x[2] != 32) return;
    uint8_t x[32], y[32];
    HvWalk w1, w2;
    HvMachine *hm = hv_vm_machine();
    hv_mmu_walk(c, c->x[0], HV_ACC_READ, 1, &w1);
    hv_mmu_walk(c, c->x[1], HV_ACC_READ, 1, &w2);
    if (w1.fsc || w2.fsc || hv_phys_read(hm, w1.pa, x, 32) || hv_phys_read(hm, w2.pa, y, 32)) return;
    static const uint8_t want[8] = {0x52, 0x24, 0x67, 0x35, 0x8c, 0x28, 0x38, 0x7a};
    if (memcmp(x, want, 8) && memcmp(y, want, 8)) return;
    fprintf(stderr, "hengevm: at %ld s ", (long)time(NULL));
    fprintf(stderr, "hengevm: memcmp 32 bytes, lr %llx:\n  a %llx:", (unsigned long long)c->x[30], (unsigned long long)c->x[0]);
    for (int i = 0; i < 32; i++) fprintf(stderr, "%02x", x[i]);
    fprintf(stderr, "\n  b %llx:", (unsigned long long)c->x[1]);
    for (int i = 0; i < 32; i++) fprintf(stderr, "%02x", y[i]);
    fprintf(stderr, "\n");
    guest_frames(c);
}
static __thread uint64_t watch_state;
static __thread uint32_t watch_want[8];
/* Debugging: "hooksha file": the guest kernel's SHA-256 routine, watched for hashing this 16 KB page. */
static uint8_t hook_page[0x4000];
static void sha_hook(HvCpu *c) {
    static uint64_t calls;
    extern uint64_t hook_pages;
    uint64_t pages = hook_pages;
    calls++;
    if (watch_state) { /* the call after the page's: its state now is what the page's hash came to */
        uint32_t st[8];
        HvWalk wn;
        hv_mmu_walk(c, watch_state, HV_ACC_READ, 1, &wn);
        if (!wn.fsc && !hv_phys_read(hv_vm_machine(), wn.pa, st, 32)) {
            int ok = !memcmp(st, watch_want, 32);
            fprintf(stderr, "hengevm: sha hook: next call (x0 %s, %llu blocks): state after the page %s: %08x %08x %08x, should be %08x %08x %08x\n", c->x[0] == watch_state ? "same" : "other", (unsigned long long)c->x[1], ok ? "RIGHT" : "WRONG", st[0], st[1], st[2], watch_want[0], watch_want[1], watch_want[2]);
        }
        watch_state = 0;
    }
    uint64_t blocks = c->x[1], va = c->x[2];
    if (blocks >= 64) pages = ++hook_pages;
    if (!blocks || blocks > 256) return;
    uint64_t bytes = blocks * 64, in = va & 0x3fff;
    if (in + bytes > 0x4000) return;
    static __thread uint8_t now[0x4000];
    HvMachine *hm = hv_vm_machine();
    HvWalk w;
    for (uint64_t k = 0; k < bytes;) {
        uint64_t chunk = 0x1000 - ((va + k) & 0xfff);
        if (chunk > bytes - k) chunk = bytes - k;
        hv_mmu_walk(c, va + k, HV_ACC_READ, 1, &w);
        if (w.fsc || hv_phys_read(hm, w.pa, now + k, (unsigned)chunk)) return;
        k += chunk;
    }
    if (bytes == 0x4000) { /* any 4 KB of it the same as the file's? */
        char eq[5] = "....";
        int any = 0;
        for (int k = 0; k < 4; k++) {
            static const uint8_t zero4k[0x1000];
            if (memcmp(hook_page + k * 0x1000, zero4k, 0x1000) && !memcmp(now + k * 0x1000, hook_page + k * 0x1000, 0x1000)) eq[k] = 'S', any = 1;
        }
        if (any) {
            int zeros[4] = {0};
            for (int k = 0; k < 0x4000; k++) zeros[k >> 12] += now[k] == 0;
            fprintf(stderr, "hengevm: sha hook: part of the page: %s, zero bytes per 4K %d %d %d %d, va %llx lr %llx\n", eq, zeros[0], zeros[1], zeros[2], zeros[3], (unsigned long long)va, (unsigned long long)c->x[30]);
            {
                uint32_t st0[8] = {0};
                HvWalk ws0;
                hv_mmu_walk(c, c->x[0], HV_ACC_READ, 1, &ws0);
                if (!ws0.fsc && !hv_phys_read(hm, ws0.pa, st0, 32)) {
                    fprintf(stderr, "  state before %08x %08x, cpu %d\n", st0[0], st0[1], c->index);
                    guest_frames(c);
                    memcpy(watch_want, st0, 32);
                    sha_blocks(watch_want, now, 256);
                    watch_state = c->x[0];
                    static int switched;
                    if (getenv("HV_HOOK_SWITCH") && !switched++) { /* the page is about to be compared: from now on watch memcmp instead */
                        HvMachine *mm = hv_vm_machine();
                        mm->stop_hook = memcmp_hook;
                        mm->stop_pc = c->pc - 0x2274914ull; /* bcmp */
                        mm->flush_pending = 1;
                    }
                }
            }
            for (int k = 0; k < 4; k++) if (eq[k] != 'S') {
                fprintf(stderr, "  chunk %d there:", k);
                for (int j = 0; j < 32; j++) fprintf(stderr, " %02x", now[k * 0x1000 + j]);
                fprintf(stderr, "\n  chunk %d want: ", k);
                for (int j = 0; j < 32; j++) fprintf(stderr, " %02x", hook_page[k * 0x1000 + j]);
                fprintf(stderr, "\n");
            }
            return;
        }
    }
    /* Mostly the same as the file at that offset, on what is not zero in the file? */
    int same = 0, nonzero = 0;
    for (uint64_t k = 0; k + 16 <= bytes; k += 64) {
        static const uint8_t zero[16];
        if (!memcmp(hook_page + in + k, zero, 16)) continue;
        nonzero++;
        same += !memcmp(now + k, hook_page + in + k, 16);
    }
    if (nonzero < 1 || same * 2 < nonzero) return;
    int diff = 0, first = -1, last = -1;
    for (uint64_t k = 0; k < bytes; k++)
        if (now[k] != hook_page[in + k]) {
            diff++;
            if (first < 0) first = (int)k;
            last = (int)k;
        }
    uint8_t h[32];
    CC_SHA256(now, (CC_LONG)bytes, h);
    uint32_t st[8] = {0};
    HvWalk ws;
    hv_mmu_walk(c, c->x[0], HV_ACC_READ, 1, &ws);
    if (!ws.fsc) hv_phys_read(hm, ws.pa, st, 32);
    fprintf(stderr, "hengevm: sha hook: cpu %d, %llu blocks at %llx (pa %llx, in page +%llx), %llu pages so far: %d differ (+%x..+%x); sha256 %02x%02x%02x%02x%02x%02x%02x%02x; state %08x %08x; lr %llx\n",
            c->index, (unsigned long long)blocks, (unsigned long long)va, (unsigned long long)w.pa, (unsigned long long)in, (unsigned long long)pages, diff, first, last, h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], st[0], st[1], (unsigned long long)c->x[30]);
    if (diff) {
        fprintf(stderr, "  there:");
        for (int k = first & ~31; k < (first & ~31) + 64 && k < (int)bytes; k++) fprintf(stderr, " %02x", now[k]);
        fprintf(stderr, "\n");
    }
}

/* Both at once: remember what each 16 KB hash read; when the compare for the page fails, keep it. */
static uint64_t both_sha_pc;
static __thread uint8_t both_data[0x4000];
static __thread uint64_t both_va, both_pa;
static __thread int both_have;
static void both_hook(HvCpu *c) {
    HvMachine *hm = hv_vm_machine();
    if (c->pc == both_sha_pc) {
        if (c->x[1] != 256) return;
        HvWalk w;
        both_have = 0;
        for (int k = 0; k < 4; k++) {
            hv_mmu_walk(c, c->x[2] + (uint64_t)k * 0x1000, HV_ACC_READ, 1, &w);
            if (w.fsc || hv_phys_read(hm, w.pa, both_data + k * 0x1000, 0x1000)) return;
            if (!k) both_pa = w.pa;
        }
        both_va = c->x[2];
        both_have = 1;
        return;
    }
    if (c->x[2] != 32) return;
    uint8_t x[32], y[32];
    HvWalk w1, w2;
    hv_mmu_walk(c, c->x[0], HV_ACC_READ, 1, &w1);
    hv_mmu_walk(c, c->x[1], HV_ACC_READ, 1, &w2);
    if (w1.fsc || w2.fsc || hv_phys_read(hm, w1.pa, x, 32) || hv_phys_read(hm, w2.pa, y, 32)) return;
    static const uint8_t want[8] = {0x52, 0x24, 0x67, 0x35, 0x8c, 0x28, 0x38, 0x7a};
    if (memcmp(x, want, 8)) return;
    int same = !memcmp(x, y, 32);
    fprintf(stderr, "hengevm: compare %s; hash read pa %llx va %llx (have %d); now it holds:\n", same ? "equal" : "DIFFERENT", (unsigned long long)both_pa, (unsigned long long)both_va, both_have);
    if (!same && both_have) {
        static uint8_t now[0x4000];
        int same_now = hv_phys_read(hm, both_pa, now, 0x4000) == 0 && !memcmp(now, both_data, 0x4000);
        int diff_good = 0, part[4] = {0};
        for (int k = 0; k < 0x4000; k++) if (both_data[k] != hook_page[k]) diff_good++, part[k >> 12]++;
        uint8_t h[32];
        CC_SHA256(both_data, 0x4000, h);
        fprintf(stderr, "  bytes hashed: %d differ from the file's page (per 4K %d %d %d %d); same as RAM now: %d; sha256 %02x%02x%02x%02x; compare b %02x%02x%02x%02x\n", diff_good, part[0], part[1], part[2], part[3], same_now, h[0], h[1], h[2], h[3], y[0], y[1], y[2], y[3]);
        const char *out = getenv("HV_HOOK_OUT");
        if (out) {
            FILE *f = fopen(out, "wb");
            if (f) fwrite(both_data, 1, 0x4000, f), fclose(f);
        }
    }
    guest_frames(c);
}

/* Debugging: HV_INPUT_FILE=path follows a file of lines, for driving a running guest:
   "click x y", "move x y", "key code" (Linux), "hidkey usage", "hiddown usage", "hidup usage", "shiftkey code", "type words", "wait seconds". */
static void *input_thread(void *unused) {
    (void)unused;
    FILE *f = fopen(getenv("HV_INPUT_FILE"), "a+");
    if (!f) return NULL;
    long at = 0;
    for (;;) {
        char line[512];
        fseek(f, at, SEEK_SET);
        if (!fgets(line, sizeof line, f) || !strchr(line, '\n')) {
            clearerr(f);
            usleep(200000);
            continue;
        }
        at = ftell(f);
        line[strcspn(line, "\n")] = 0;
        int x, y;
        if (sscanf(line, "click %d %d", &x, &y) == 2) pointer_to(x, y), click_here();
        else if (sscanf(line, "move %d %d", &x, &y) == 2) pointer_to(x, y);
        else if (sscanf(line, "scroll %d %d", &x, &y) == 2) {
            if (y) hv_vm_mouse_wheel(y);
            if (x) hv_vm_mouse_hwheel(x);
        }
        else if (!strncmp(line, "pointer", 7)) {
            int px = -1, py = -1;
            int rc = hv_vm_pointer(&px, &py);
            fprintf(stderr, "hengevm: pointer %d: %d, %d\n", rc, px, py);
        }
        else if (!strncmp(line, "snapshot ", 9)) { /* FILE, a 32-bit BMP of what hv_vm_snapshot gives the app */
            enum { W = 640, H = 360 };
            static uint8_t px[W * H * 4];
            int rc = hv_vm_snapshot(px, W, H);
            fprintf(stderr, "hengevm: snapshot %d\n", rc);
            FILE *bf = rc == 0 ? fopen(line + 9, "wb") : NULL;
            if (bf) {
                uint32_t size = W * H * 4;
                uint8_t h[54] = {'B', 'M'};
                uint32_t v;
                v = 54 + size; memcpy(h + 2, &v, 4);
                v = 54; memcpy(h + 10, &v, 4);
                v = 40; memcpy(h + 14, &v, 4);
                int32_t iw = W, ih = -H; memcpy(h + 18, &iw, 4); memcpy(h + 22, &ih, 4);
                h[26] = 1; h[28] = 32;
                memcpy(h + 34, &size, 4);
                fwrite(h, 1, 54, bf);
                fwrite(px, 1, size, bf);
                fclose(bf);
            }
        }
        else if (!strncmp(line, "btn ", 4)) { /* "btn CODE 1|0": any evdev button code on the virtio mouse (testing) */
            unsigned code = 0;
            int down = 0;
            if (sscanf(line + 4, "%i %d", &code, &down) == 2) {
                hv_virtio_input_send(board.mouse, HV_EV_KEY, (uint16_t)code, down);
                hv_virtio_input_send(board.mouse, HV_EV_SYN, 0, 0);
            }
        }
        else if (!strncmp(line, "press", 5) || !strncmp(line, "release", 7)) {
            hv_virtio_input_send(board.mouse, HV_EV_KEY, HV_BTN_LEFT, line[0] == 'p');
            hv_virtio_input_send(board.mouse, HV_EV_SYN, 0, 0);
        }
        else if (!strncmp(line, "clip ", 5)) hv_share_set_clipboard(line + 5);
        else if (!strncmp(line, "getclip", 7)) {
            char *text = hv_share_take_clipboard();
            fprintf(stderr, "hengevm: guest clipboard: %s (agent seen %ld s ago)\n", text ? text : "(nothing new)", hv_share_agent_seen());
            free(text);
        }
        else if (sscanf(line, "hiddown %i", &x) == 1) hv_vm_key_hid((uint8_t)x, 1);
        else if (sscanf(line, "hidup %i", &x) == 1) hv_vm_key_hid((uint8_t)x, 0);
        else if (sscanf(line, "hidkey %i", &x) == 1) {
            hv_vm_key_hid((uint8_t)x, 1);
            usleep(60000);
            hv_vm_key_hid((uint8_t)x, 0);
        }
        else if (sscanf(line, "key %d", &x) == 1) key_press(x, 0);
        else if (sscanf(line, "shiftkey %d", &x) == 1) key_press(x, 1);
        else if (sscanf(line, "cmdkey %d", &x) == 1) { /* with Command held: 125 is the left one */
            key_event(125, 1);
            usleep(60000);
            key_press(x, 0);
            usleep(60000);
            key_event(125, 0);
        }
        else if (sscanf(line, "wait %d", &x) == 1) sleep((unsigned)x);
        else if (!strncmp(line, "type ", 5)) type_text(line + 5);
        else if (!strncmp(line, "prof ", 5)) prof_dump(line + 5);
        else if (!strncmp(line, "balloon ", 8)) { /* MB the guest should hand back (HV_BALLOON set at boot) */
            HvVmapple *bd = hv_vm_board();
            if (!bd || !bd->balloon) fprintf(stderr, "hengevm: no balloon (set HV_BALLOON)\n");
            else {
                hv_virtio_balloon_set_target(bd->balloon, (uint64_t)atoll(line + 8) << 20);
                fprintf(stderr, "hengevm: balloon: the guest has handed over %llu MB\n", (unsigned long long)(hv_virtio_balloon_actual(bd->balloon) >> 20));
            }
        }
        else if (!strncmp(line, "knob ", 5)) {
            char name[32];
            if (sscanf(line, "knob %31s %d", name, &x) == 2 && !hv_knob_set(hv_vm_machine(), name, x))
                fprintf(stderr, "hengevm: no knob %s\n", name);
        }
        else if (!strncmp(line, "kdump ", 6)) { /* the guest kernel's image, from where its first byte is mapped now */
            static const uint8_t sig[] = {0x5f, 0x24, 0x03, 0xd5, 0x03, 0x00, 0x00, 0x90, 0x41, 0x00, 0x00, 0xb5, 0xc0, 0x03, 0x5f, 0xd6, 0x63, 0x00, 0x36, 0x91, 0xe4, 0x43, 0x04, 0xd1, 0xff, 0x43, 0x04, 0xd1, 0x80, 0x28, 0x9f, 0x4c};
            HvVmapple *bd = hv_vm_board();
            uint8_t *at = memmem(bd->ram, bd->ram_size, sig, sizeof sig);
            FILE *kf = fopen(line + 6, "wb");
            if (at && kf) fwrite(at - (0xfffffe002b76de80ull - 0xfffffe0028604000ull), 1, 0x4800000, kf);
            if (kf) fclose(kf);
            fprintf(stderr, "hengevm: kernel image written\n");
        }
        else if (!strncmp(line, "hooksha ", 8)) {
            if (hv_vm_machine()->stop_hook) continue;
            FILE *pf = fopen(line + 8, "rb");
            size_t got = pf ? fread(hook_page, 1, sizeof hook_page, pf) : 0;
            if (pf) fclose(pf);
            static const uint8_t sig[] = {0x5f, 0x24, 0x03, 0xd5, 0x03, 0x00, 0x00, 0x90, 0x41, 0x00, 0x00, 0xb5, 0xc0, 0x03, 0x5f, 0xd6, 0x63, 0x00, 0x36, 0x91, 0xe4, 0x43, 0x04, 0xd1, 0xff, 0x43, 0x04, 0xd1, 0x80, 0x28, 0x9f, 0x4c};
            HvVmapple *bd = hv_vm_board();
            HvCpu *c0 = hv_vm_cpu(0);
            HvWalk wv;
            hv_mmu_walk(c0, c0->vbar_el1, HV_ACC_READ, 1, &wv);
            uint8_t *at = memmem(bd->ram, bd->ram_size, sig, sizeof sig);
            uint64_t pa = at ? 0x70000000ull + (uint64_t)(at - bd->ram) : 0;
            uint64_t va = c0->vbar_el1 + (pa - wv.pa);
            HvWalk wc;
            hv_mmu_walk(c0, va, HV_ACC_EXEC, 1, &wc);
            fprintf(stderr, "hengevm: sha at pa %llx, va %llx (walks to pa %llx, fsc %u); page file %zu bytes\n", (unsigned long long)pa, (unsigned long long)va, (unsigned long long)wc.pa, (unsigned)wc.fsc, got);
            if (at && !wc.fsc && wc.pa == pa && got == sizeof hook_page && c0->vbar_el1 >> 40 == 0xfffffe) {
                HvMachine *hm = hv_vm_machine();
                if (getenv("HV_HOOK_BOTH")) {
                    both_sha_pc = va;
                    hm->stop_hook = both_hook;
                    hm->stop_pc = va;
                    hm->stop_pc2 = va - 0x2274914ull;
                } else {
                hm->stop_hook = getenv("HV_HOOK_BCMP") ? memcmp_hook : sha_hook;
                hm->stop_pc = getenv("HV_HOOK_BCMP") ? va - 0x2274914ull : va;
                }
                hm->flush_pending = 1;
            }
        }
        else if (!strncmp(line, "find ", 5)) { /* bytes in hex: where in guest RAM, and what is around them */
            uint8_t want[32];
            size_t wn = 0;
            for (const char *h = line + 5; h[0] && h[1] && wn < sizeof want; h += 2) {
                unsigned v;
                if (sscanf(h, "%2x", &v) != 1) break;
                want[wn++] = (uint8_t)v;
            }
            HvVmapple *bd = hv_vm_board();
            int n = 0;
            for (uint64_t i = 0; wn && i + wn + 64 < bd->ram_size && n < 8; i += 4) {
                if (memcmp(bd->ram + i, want, wn)) continue;
                n++;
                fprintf(stderr, "ram %llx:", (unsigned long long)(0x70000000ull + i));
                for (int k = 0; k < 96; k++) fprintf(stderr, "%s%02x", k % 32 == 0 ? " " : "", bd->ram[i + k]);
                fprintf(stderr, "\n");
            }
            fprintf(stderr, "hengevm: find: %d places\n", n);
        }
        else if (!strncmp(line, "grep ", 5)) { /* lines of text in guest RAM (the kernel's log) */
            HvVmapple *bd = hv_vm_board();
            const char *word = line + 5;
            size_t wn = strlen(word);
            uint64_t hits = 0;
            for (uint64_t i = 0; i + wn < bd->ram_size && hits < 40; i++) {
                if (bd->ram[i] != (uint8_t)word[0] || memcmp(bd->ram + i, word, wn)) continue;
                uint64_t from = i;
                while (from > 0 && i - from < 200 && bd->ram[from - 1] >= 32 && bd->ram[from - 1] < 127) from--;
                uint64_t to = i;
                while (to < bd->ram_size && to - i < 400 && bd->ram[to] >= 32 && bd->ram[to] < 127) to++;
                fprintf(stderr, "ram %llx: %.*s\n", (unsigned long long)(0x70000000ull + from), (int)(to - from), bd->ram + from);
                hits++;
                i = to;
            }
            fprintf(stderr, "hengevm: grep: %llu lines\n", (unsigned long long)hits);
        }
        else if (!strncmp(line, "page ", 5)) { /* a 16 KB file: is it in guest RAM, and how does the copy there differ? */
            static uint8_t want[0x4000];
            FILE *pf = fopen(line + 5, "rb");
            size_t got = pf ? fread(want, 1, sizeof want, pf) : 0;
            if (pf) fclose(pf);
            HvVmapple *bd = hv_vm_board();
            int found = 0;
            for (uint64_t at = 0; got == sizeof want && at + 0x4000 <= bd->ram_size; at += 0x4000) {
                int same = 0;
                for (int k = 0; k < 16; k++) same += !memcmp(bd->ram + at + k * 0x400, want + k * 0x400, 8);
                if (same < 2) continue;
                int diff = 0, first = -1, last = -1;
                for (int k = 0; k < 0x4000; k++)
                    if (bd->ram[at + k] != want[k]) {
                        diff++;
                        if (first < 0) first = k;
                        last = k;
                    }
                if (diff > 9700) continue; /* a page of zeros differs by about this */
                fprintf(stderr, "hengevm: page copy at pa %llx: %d bytes differ, from +%x to +%x\n", (unsigned long long)(0x70000000ull + at), diff, first, last);
                { int part[4] = {0};
                  for (int k = 0; k < 0x4000; k++) part[k >> 12] += bd->ram[at + k] != want[k];
                  fprintf(stderr, "  differ per 4K: %d %d %d %d\n", part[0], part[1], part[2], part[3]); }
                if (diff && first >= 0 && found < 3) {
                    fprintf(stderr, "  there:");
                    for (int k = first & ~31; k < (first & ~31) + 64 && k < 0x4000; k++) fprintf(stderr, " %02x", bd->ram[at + k]);
                    fprintf(stderr, "\n  want: ");
                    for (int k = first & ~31; k < (first & ~31) + 64 && k < 0x4000; k++) fprintf(stderr, " %02x", want[k]);
                    fprintf(stderr, "\n");
                }
                found++;
            }
            fprintf(stderr, "hengevm: %d copies of that page in RAM\n", found);
        }
        fprintf(stderr, "hengevm: input: %s\n", line);
    }
    return NULL;
}

/* Debugging: HV_WATCH=seconds says what each CPU is at, and how many frames the guest has shown. */
static void *watch_thread(void *unused) {
    (void)unused;
    int secs = atoi(getenv("HV_WATCH"));
    time_t start = time(NULL);
    {
        HvVmapple *bd = hv_vm_board();
        hv_smc_pages = bd->ram_size >> 14;
        hv_smc_last = calloc(hv_smc_pages, 8);
        hv_smc_counts = calloc(hv_smc_pages, 4);
    }
    for (;;) {
        sleep((unsigned)(secs > 0 ? secs : 5));
        const uint8_t *page = hv_pvg_lite_display_page(board.pvg);
        fprintf(stderr, "hengevm: watch %ld s: swaps %u on %u;", (long)(time(NULL) - start), page ? *(const volatile uint32_t *)(page + 0xf98) : 0, page ? *(const volatile uint32_t *)(page + 0xfb0) : 0);
        for (int i = 0; i < ncpus; i++) fprintf(stderr, " %llx@%u", (unsigned long long)hv_vm_cpu(i)->pc, hv_vm_cpu(i)->el);
        { HvMachine *wm = hv_vm_machine(); fprintf(stderr, " | jit %llu MB, flushes %llu, ic %llu, code writes %llu", (unsigned long long)wm->jit.used * 4 >> 20, (unsigned long long)wm->stats.block_flushes, (unsigned long long)wm->stats.flush_ic, (unsigned long long)wm->stats.flush_smc);
          fprintf(stderr, " (cpu %llu dma %llu), %llu translations, blocks now %u; top:", (unsigned long long)hv_smc_cpu, (unsigned long long)hv_smc_dma, (unsigned long long)wm->stats.translations, (unsigned)wm->nblocks);
          uint32_t seen = 0; uint64_t distinct = 0; (void)seen;
          uint64_t best[4] = {0}; uint32_t bestn[4] = {0};
          for (uint64_t pg = 0; pg < hv_smc_pages; pg++) {
              uint32_t n = hv_smc_counts[pg];
              if (n) distinct++;
              for (int k = 0; k < 4; k++) if (n > bestn[k]) { for (int j = 3; j > k; j--) bestn[j] = bestn[j - 1], best[j] = best[j - 1]; bestn[k] = n, best[k] = pg; break; }
          }
          for (int k = 0; k < 4; k++) fprintf(stderr, " %llx*%u(pc %llx)", (unsigned long long)(0x70000000ull + (best[k] << 14)), bestn[k], (unsigned long long)hv_smc_last[best[k]]);
          fprintf(stderr, " distinct %llu", (unsigned long long)distinct);
          fprintf(stderr, " sha pages %llu", (unsigned long long)hook_pages);
          { uint64_t bs[4]; hv_pvg_lite_barrier_stats(board.pvg, bs); fprintf(stderr, " barriers %llu drains %llu retries %llu gaveup %llu", (unsigned long long)bs[0], (unsigned long long)bs[1], (unsigned long long)bs[2], (unsigned long long)bs[3]); }
        }
        fprintf(stderr, "\n");
    }
    return NULL;
}

/* Debugging: HV_TYPE_AT=seconds types "abc" then, to check input end to end without a window. */
static void *type_thread(void *unused) {
    (void)unused;
    sleep((unsigned)atoi(getenv("HV_TYPE_AT")));
    for (int i = 0; i < 60; i++) { /* shake it: macOS enlarges the pointer, so new glyphs get sent */
        hv_virtio_input_send(board.mouse, HV_EV_REL, HV_REL_X, i & 1 ? 120 : -120);
        hv_virtio_input_send(board.mouse, HV_EV_SYN, 0, 0);
        usleep(15000);
    }
    for (int i = 0; i < 40; i++) { /* the pointer: up and to the left, so it shows */
        hv_virtio_input_send(board.mouse, HV_EV_REL, HV_REL_X, -10);
        hv_virtio_input_send(board.mouse, HV_EV_REL, HV_REL_Y, -6);
        hv_virtio_input_send(board.mouse, HV_EV_SYN, 0, 0);
        usleep(20000);
    }
    if (getenv("HV_TYPE_NO_KEYS")) { /* out into the open, for a screenshot */
        for (int i = 0; i < 60; i++) {
            hv_virtio_input_send(board.mouse, HV_EV_REL, HV_REL_X, 12);
            hv_virtio_input_send(board.mouse, HV_EV_REL, HV_REL_Y, 8);
            hv_virtio_input_send(board.mouse, HV_EV_SYN, 0, 0);
            usleep(20000);
        }
        return NULL;
    }
    static const uint16_t keys[3] = {30, 48, 46}; /* a, b, c */
    for (int i = 0; i < 3; i++) {
        hv_xhci_key_linux(board.usb, keys[i], 1);
        hv_virtio_input_send(board.keyboard, HV_EV_KEY, keys[i], 1);
        hv_virtio_input_send(board.keyboard, HV_EV_SYN, 0, 0);
        usleep(100000);
        hv_xhci_key_linux(board.usb, keys[i], 0);
        hv_virtio_input_send(board.keyboard, HV_EV_KEY, keys[i], 0);
        hv_virtio_input_send(board.keyboard, HV_EV_SYN, 0, 0);
        usleep(700000);
    }
    for (int i = 0; i < 10; i++) {
        sleep(2);
        fprintf(stderr, "hengevm: typed abc: usb delivered %llu reports, keyboard delivered %llu of 12 events, mouse %llu\n",
                (unsigned long long)board.usb->delivered, (unsigned long long)board.keyboard->delivered, board.mouse ? (unsigned long long)board.mouse->delivered : 0ull);
    }
    return NULL;
}

static void *cpu_thread(void *unused) {
    (void)unused;
    RunArgs *a = &run_args;
    cpu_port = mach_thread_self();
    hv_vm_cpu_qos();
    /* --break PC: stop the brk_hits-th time execution reaches PC. */
    m.stop_pc = a->brk;
    int r;
    for (;;) {
        r = hv_vm_run_cpu0();
        if (r != HV_EXIT_HALT || cpu.halted || cpu.pc != a->brk || --a->brk_hits == 0) break;
        m.stop_pc = ~0ull; /* step past it: run one block, then re-arm */
        uint64_t at = cpu.pc;
        while (cpu.pc == at && !cpu.halted) {
            m.stop_pc = ~0ull;
            hv_run_one(&cpu);
        }
        m.stop_pc = a->brk;
    }
    if (cpu.pc == a->brk) {
        fprintf(stderr, "hengevm: breakpoint %llx, x30 %llx:", (unsigned long long)a->brk, (unsigned long long)cpu.x[30]);
        hv_debug_args(&cpu);
        fprintf(stderr, "  stack:");
        for (int i = 0; i < 8; i++) {
            uint64_t v = 0, pa;
            if (!hv_mmu_access(&cpu, cpu.sp + 8 * i, HV_ACC_READ, cpu.el, &pa)) hv_phys_read(&m, pa, &v, 8);
            fprintf(stderr, " %llx", (unsigned long long)v);
        }
        fprintf(stderr, "\n");
        if (translate_va) {
            uint64_t pa = 0;
            uint32_t fsc = hv_mmu_access(&cpu, translate_va, HV_ACC_READ, cpu.el, &pa);
            fprintf(stderr, "  translate %llx -> pa %llx (fsc %x)\n", (unsigned long long)translate_va,
                    (unsigned long long)pa, fsc);
        }
    }
    fprintf(stderr, "\nhengevm: stopped (%s) after %llu blocks, %d translated\n", interrupted ? "interrupted" : reason_name(r),
            (unsigned long long)cpu.icount, m.nblocks);
    fprintf(stderr, "  pc %016llx el%u sp %016llx insn %08x\n", (unsigned long long)cpu.pc, cpu.el,
            (unsigned long long)cpu.sp, cpu.exit_insn);
    fprintf(stderr, "  esr %08llx far %016llx elr %016llx vbar %016llx sctlr %08llx\n", (unsigned long long)cpu.esr_el1,
            (unsigned long long)cpu.far_el1, (unsigned long long)cpu.elr_el1, (unsigned long long)cpu.vbar_el1,
            (unsigned long long)cpu.sctlr_el1);
    for (int i = 0; i < 31; i += 4)
        fprintf(stderr, "  x%-2d %016llx %016llx %016llx %016llx\n", i, (unsigned long long)cpu.x[i],
                (unsigned long long)cpu.x[i + 1], (unsigned long long)cpu.x[i + 2],
                (unsigned long long)(i + 3 < 31 ? cpu.x[i + 3] : 0));
    if (a->dump_path) { /* --dump PA LEN FILE: guest physical memory after the run, for reading code */
        FILE *f = fopen(a->dump_path, "wb");
        void *p = hv_phys_ptr(&m, a->dump_pa, a->dump_len);
        if (f && p) fwrite(p, 1, a->dump_len, f);
        if (f) fclose(f);
        fprintf(stderr, "hengevm: %s %llu bytes at %llx to %s\n", p ? "wrote" : "cannot dump", (unsigned long long)a->dump_len,
                (unsigned long long)a->dump_pa, a->dump_path);
    }
    if (getenv("HV_RAM_GREP")) { /* debugging: text lines in guest RAM with any of these words (the kernel's log buffer) */
        HvVmapple *bd = hv_vm_board();
        char words[512];
        snprintf(words, sizeof words, "%s", getenv("HV_RAM_GREP"));
        const char *w[16];
        int nw = 0;
        for (char *t = strtok(words, ","); t && nw < 16; t = strtok(NULL, ",")) w[nw++] = t;
        FILE *out = fopen(getenv("HV_RAM_GREP_OUT") ? getenv("HV_RAM_GREP_OUT") : "/dev/stderr", "w");
        uint64_t hits = 0;
        for (uint64_t i = 0; i < bd->ram_size && out && hits < 20000;) {
            /* a run of printable bytes ending in a newline or NUL */
            uint64_t j = i;
            while (j < bd->ram_size && j - i < 1024 && (bd->ram[j] >= 32 && bd->ram[j] < 127)) j++;
            if (j - i >= 12) {
                for (int k = 0; k < nw; k++)
                    if (memmem(bd->ram + i, j - i, w[k], strlen(w[k]))) {
                        fprintf(out, "%llx: %.*s\n", (unsigned long long)(0x70000000ull + i), (int)(j - i), bd->ram + i);
                        hits++;
                        break;
                    }
            }
            i = j + 1;
        }
        if (out) fclose(out);
        fprintf(stderr, "hengevm: RAM grep: %llu lines\n", (unsigned long long)hits);
    }
    if (getenv("HV_BOOT_DISPLAY_DUMP")) { /* debugging: the boot framebuffer, raw BGRA, for the iOS view */
        HvBootDisplay bd;
        if (hv_vm_boot_display(&bd) == 0) {
            FILE *f = fopen(getenv("HV_BOOT_DISPLAY_DUMP"), "wb");
            if (f) fwrite(bd.pixels, 1, (size_t)bd.stride * bd.height, f), fclose(f);
            fprintf(stderr, "hengevm: boot display %ux%u stride %u written\n", bd.width, bd.height, bd.stride);
        } else {
            fprintf(stderr, "hengevm: no boot display found\n");
        }
    }
    hv_vm_flush_disks();
    exit(0);
    return NULL;
}

static void usage(void) {
    fprintf(stderr, "usage: hengevm --rom FILE --disk FILE --aux FILE --ecid N [--seed N] [--ram MB] [--log N] [--trace 1] [--screenshot PNG] [--watch VA] [--stats SECONDS [--profile 1]] [--cpus N] [--window] [--no-pvg] [--no-chain] [--break PC [--break-hits N]] [--dump PA LEN FILE] [--aux-offset BYTES] [--install IPSW]\n");
    exit(2);
}

static void install_progress(int step, double p, void *user) {
    (void)user;
    static int last_step = -1, last_pct = -1;
    int pct = (int)(p * 100);
    if (step == last_step && pct / 5 == last_pct / 5) return;
    last_step = step, last_pct = pct;
    if (step == HV_INSTALL_DONE) fprintf(stderr, "hengevm: install: done\n");
    else if (step == HV_INSTALL_FAILED) fprintf(stderr, "hengevm: install: failed\n");
    if ((step == HV_INSTALL_DONE || step == HV_INSTALL_FAILED) && getenv("HV_INSTALL_STOP_AT_END")) {
        sleep(3); /* debugging: stop the machine once the restore has a result (the exit dumps then run) */
        hv_vm_stop();
    }
    else fprintf(stderr, "hengevm: install: step %d %d%%\n", step, pct);
}

int main(int argc, char **argv) {
    const char *rom = NULL, *disk_path = NULL, *aux_path = NULL;
    uint64_t seed = 0, ecid = 0, ram_mb = 4096, aux_offset = 16384;
    const char *install_ipsw = NULL;
    const char *share_folder = NULL;
    int log = 1, trace = 0, use_pvg = 1, no_chain = 0;
    const char *screenshot = NULL;
    uint64_t watch = 0, watch_pa = 0;
    int stats = 0, profile = 0, window = 0;
    uint64_t brk = ~0ull, brk_hits = 1, dump_pa = 0, dump_len = 0;
    const char *dump_path = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--window")) {
            window = 1;
            continue;
        }
        if (!strcmp(argv[i], "--phone")) {
            /* Test iOS's setup on the Mac: no graphics framework, split JIT mapping. */
            use_pvg = 0;
            setenv("HV_JIT_SPLIT", "1", 1);
            memmove(&argv[i], &argv[i + 1], (size_t)(argc - i - 1) * sizeof *argv);
            argc--, i--;
            continue;
        }
        if (!strcmp(argv[i], "--no-pvg")) {
            use_pvg = 0;
            continue;
        }
        if (!strcmp(argv[i], "--no-chain")) {
            no_chain = 1;
            continue;
        }
        if (i + 1 >= argc) usage();
        if (!strcmp(argv[i], "--rom")) rom = argv[++i];
        else if (!strcmp(argv[i], "--share")) { /* FOLDER: the guest reaches it at http://10.0.2.4/ */
            share_folder = argv[++i];
        }
        else if (!strcmp(argv[i], "--display")) { /* WxH, in --phone mode */
            unsigned w = 0, h = 0;
            if (sscanf(argv[++i], "%ux%u", &w, &h) == 2 && w && h) hv_pvg_lite_set_mode(w, h);
        }
        else if (!strcmp(argv[i], "--disk")) disk_path = argv[++i];
        else if (!strcmp(argv[i], "--aux")) aux_path = argv[++i];
        else if (!strcmp(argv[i], "--ecid")) ecid = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--ram")) ram_mb = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--seed")) seed = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--log")) log = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--trace")) trace = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--screenshot")) screenshot = argv[++i];
        else if (!strcmp(argv[i], "--watch")) watch = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--stats")) stats = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--profile")) profile = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cpus")) ncpus = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--watch-pa")) watch_pa = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--translate")) translate_va = strtoull(argv[++i], NULL, 0);

        else if (!strcmp(argv[i], "--break")) brk = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--break-hits")) brk_hits = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--dump") && i + 3 < argc) {
            dump_pa = strtoull(argv[++i], NULL, 0);
            dump_len = strtoull(argv[++i], NULL, 0);
            dump_path = argv[++i];
        }
        else if (!strcmp(argv[i], "--aux-offset")) aux_offset = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--install")) install_ipsw = argv[++i];
        else usage();
    }
    if (!rom || !disk_path || !aux_path) usage();
    setvbuf(stderr, NULL, _IONBF, 0);

    HvVmConfig vc = {.rom = rom, .disk = disk_path, .aux = aux_path, .aux_offset = aux_offset, .ecid = ecid, .seed = seed,
                     .ram_mb = ram_mb, .ncpus = ncpus, .log = log, .no_net = getenv("HV_NO_NET") != NULL,
                     .installer = install_ipsw != NULL};
    if (getenv("HV_JIT_MB")) vc.jit_bytes = (size_t)atoi(getenv("HV_JIT_MB")) << 20; /* debugging: a small region flushes often */
    hv_prof_register = hv_prof_register_thread;
    char err[256];
    if (share_folder) {
        char sock[256];
        snprintf(sock, sizeof sock, "%s.sock", share_folder);
        if (hv_share_start(share_folder, sock, getenv("HV_AGENT")) == 0) hv_net_set_share_socket(sock);
    }
    if (hv_vm_create(&vc, err, sizeof err) != 0) {
        fprintf(stderr, "hengevm: %s\n", err);
        return 1;
    }
    m.trace = trace;
    m.watch_va = watch;
    m.watch_pa = watch_pa;
    m.stats_interval = stats; /* set after the machine is made, which clears it */
    if (profile) m.prof = calloc(HV_PROF_RING, sizeof *m.prof);
    m.chain = !no_chain && brk == ~0ull; /* breakpoints need every block to come back */
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    if (install_ipsw) {
        char work[1024];
        snprintf(work, sizeof work, "%s.restore", disk_path);
        mkdir(work, 0755);
        if (hv_install_start(board.vusb, install_ipsw, work, ecid, install_progress, NULL) != 0) {
            fprintf(stderr, "hengevm: cannot start the installer\n");
            return 1;
        }
        /* usbmuxd installs its own handlers; take SIGINT and SIGTERM back so
           the machine still stops on them. */
        usleep(500000);
        signal(SIGINT, on_signal);
        signal(SIGTERM, on_signal);
    }
    int pvg_on = 0;
    if (use_pvg) {
        if (hv_pvg_attach(&board, screenshot) != 0) fprintf(stderr, "hengevm: ParavirtualizedGraphics device unavailable; graphics stays inert\n");
        else pvg_on = 1;
    }
    if (pvg_on) hv_vm_set_poll(hv_pvg_poll);
    run_args = (RunArgs){brk, brk_hits, dump_pa, dump_len, dump_path};
    /* The CPU runs on its own thread: the graphics framework needs the main queue. */
    hv_vm_start(0, NULL); /* the other CPUs and the kicker; CPU 0 is cpu_thread, for --break */
    pthread_t t;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 8u << 20);
    pthread_create(&t, &attr, cpu_thread, NULL);
    if (profile) pthread_create(&t, NULL, profile_thread, NULL);
    if (getenv("HV_CLICKS")) {
        pthread_t cl;
        pthread_create(&cl, NULL, click_thread, NULL);
    }
    if (getenv("HV_PROF_PCS")) prof_pcs = calloc(PROF_PCS, sizeof *prof_pcs);
    if (getenv("HV_PROF_ALL")) {
        pthread_t pr;
        pthread_create(&pr, NULL, prof_thread, NULL);
    }
    if (getenv("HV_INPUT_FILE")) {
        pthread_t in;
        pthread_create(&in, NULL, input_thread, NULL);
    }
    if (getenv("HV_WATCH")) {
        pthread_t wa;
        pthread_create(&wa, NULL, watch_thread, NULL);
    }
    if (getenv("HV_TYPE_AT")) {
        pthread_t ty;
        pthread_create(&ty, NULL, type_thread, NULL);
    }
    if (window && use_pvg) hv_pvg_window_run();
    CFRunLoopRun();
    return 0;
}
