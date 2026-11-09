#include "machine.h"

#include <mach/mach_time.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mmu.h"
#include "sysreg.h"

extern void hv_exit(void);
extern void hv_helper_entry(void);
extern void hv_helper_thunk(void);
extern void hv_chain_thunk(void);
extern void hv_ijc_thunk(void);
extern void hv_xjump_thunk(void);

static int init_rest(HvMachine *m);

int hv_knob[HV_K_COUNT];
static const char *const hv_knob_names[HV_K_COUNT] = {"HV_NO_RAS", "HV_SUPERBLOCKS", "HV_GROUPS", "HV_WFI_SPIN", "HV_FLUSH", "HV_K1", "HV_K2", "HV_K3", "HV_TLBT", "HV_JIT_CAP", "HV_RECHECK", "HV_SYNC_WAIT", "HV_SWITCH_FLUSH", "HV_RANGE_DROP", "HV_MOVIMM", "HV_PREDSTAT", "HV_WALK_LOG", "HV_HOIST_CHECK", "HV_IJC_PAGES", "HV_DNS_PROXY"};

static void knobs_init(void) {
    for (int i = 0; i < HV_K_COUNT; i++)
        hv_knob[i] = getenv(hv_knob_names[i]) ? atoi(getenv(hv_knob_names[i])) : 0;
    if (getenv("HV_NO_RAS") && !*getenv("HV_NO_RAS")) hv_knob[HV_K_NO_RAS] = 1;
    /* On unless asked: Golden Gate on the phone (2026-11-08) ran 200 process spawns ~3% faster with it, and none
       of the checked skips (1 in 32 are checked, see sysreg.c) found an out-of-date entry. */
    if (!getenv("HV_SWITCH_FLUSH")) hv_knob[HV_K_SWITCHFLUSH] = 2;
    if (!getenv("HV_IJC_PAGES")) hv_knob[HV_K_IJCPAGES] = 1;
    if (!getenv("HV_DNS_PROXY")) hv_knob[HV_K_DNSPROXY] = 1;
}

int hv_knob_set(HvMachine *m, const char *name, int value) {
    for (int i = 0; i < HV_K_COUNT; i++)
        if (!strcmp(name, hv_knob_names[i])) {
            hv_knob[i] = value;
            __atomic_store_n(&m->flush_pending, 1, __ATOMIC_SEQ_CST);
            hv_machine_kick_all(m);
            return 1;
        }
    return 0;
}

int hv_machine_init(HvMachine *m, size_t jit_bytes) {
    memset(m, 0, sizeof *m);
    if (hv_jit_region_open(&m->jit, jit_bytes) != 0) return -1;
    return init_rest(m);
}

int hv_machine_init_jit(HvMachine *m, void *rx, void *rw, size_t jit_bytes) {
    knobs_init();
    memset(m, 0, sizeof *m);
    if (hv_jit_region_attach(&m->jit, rx, rw, jit_bytes) != 0) return -1;
    return init_rest(m);
}

static int init_rest(HvMachine *m) {
    /* The TLB layout (cpu.h): HV_TLBT from the environment here too, so the Mac tool and the tests can pick it. */
    hv_knob[HV_K_TLBT] = getenv("HV_TLBT") ? atoi(getenv("HV_TLBT")) : 1; /* table-major unless asked: 9-20% faster in 5 of 6 paired rounds of the mix */
    hv_tlb_tmajor = hv_knob[HV_K_TLBT] != 0;
    /* HV_MOVIMM=1: constants built in registers instead of loaded from the pool (translate.c put_lit). Off: no
       measured gain (Mac mix 24.1 s either way, 2026-11-08); the tests pass with it on (HV_MOVIMM=1 make test). */
    hv_knob[HV_K_MOVIMM] = getenv("HV_MOVIMM") ? atoi(getenv("HV_MOVIMM")) : 0;
    /* On unless asked: the Mac bench (2026-11-08) posted 130-400 raced remote drops/s with it against 710-1370
       without (most walks a drop met were of other pages), and the phone's login storm waits up to 24% of a CPU. */
    hv_knob[HV_K_WALKLOG] = getenv("HV_WALK_LOG") ? atoi(getenv("HV_WALK_LOG")) : 1;
    if (getenv("HV_HOIST_CHECK")) hv_knob[HV_K_HOIST] = atoi(getenv("HV_HOIST_CHECK"));
    /* On unless asked: a code change used to drop every CPU's whole branch cache (~100/s with JavaScriptCore's JIT
       on the Mac bench, 300-390K refills/s after); now only the page's targets (Mac 2026-11-09: resets 195 -> 2/s,
       a JIT check of 200 recompiled functions correct; bench times the same within noise). */
    hv_knob[HV_K_IJCPAGES] = getenv("HV_IJC_PAGES") ? atoi(getenv("HV_IJC_PAGES")) : 1;
    hv_knob[HV_K_DNSPROXY] = getenv("HV_DNS_PROXY") ? atoi(getenv("HV_DNS_PROXY")) : 1;
    /* So the tests can run the translator these ways too (the iOS and --phone paths read them in knobs_init). */
    if (getenv("HV_SUPERBLOCKS")) hv_knob[HV_K_SUPERBLOCKS] = atoi(getenv("HV_SUPERBLOCKS"));
    /* The exact check for an access in a page's last 15 bytes, in the access's out-of-line path (code size only):
       on the phone, Golden Gate's boot and login send 80-240K such accesses a second to the helper without it. It
       costs ~13% of the code (Mac 2026-11-09: 296-300 against 259-261 bytes a block), and the phone's code cache
       fills every 40-60 s: worth a phone A/B (HV_RECHECK 0 vs 1 over a login), not a Mac one (~7K such accesses/s). */
    hv_knob[HV_K_RECHECK] = getenv("HV_RECHECK") ? atoi(getenv("HV_RECHECK")) : 1;
    /* As many blocks as the region holds (a block is at least ~160 bytes of
       code), so the region filling, not this table, is what flushes; two
       chainable exits each at most. Untouched entries cost nothing. */
    size_t bytes = m->jit.capacity * sizeof(uint32_t);
    m->max_blocks = (uint32_t)(bytes / 160 < 4096 ? 4096 : bytes / 160);
    m->max_slots = m->max_blocks * 2;
    m->blocks = calloc(m->max_blocks, sizeof(HvBlock));
    if (!m->blocks) return -1;
    m->stop_pc = ~0ull;
    m->stop_pc2 = ~0ull;
    m->chain = 1;
    m->slots = calloc(m->max_slots, sizeof *m->slots);
    m->slot_site = calloc(m->max_slots, sizeof *m->slot_site);
    m->slot_stub = calloc(m->max_slots, sizeof *m->slot_stub);
    m->slot_cond = calloc(m->max_slots, sizeof *m->slot_cond);
    m->slot_xa = calloc(m->max_slots, sizeof *m->slot_xa);
    m->slot_xpa = calloc(m->max_slots, sizeof *m->slot_xpa);
    m->xchained = calloc(m->max_slots, sizeof *m->xchained);
    if (!m->slots || !m->slot_site || !m->slot_stub || !m->slot_cond || !m->slot_xa || !m->slot_xpa || !m->xchained) return -1;
    pthread_mutex_init(&m->big_mu, NULL);
    pthread_cond_init(&m->big_cv, NULL);
    pthread_mutexattr_t ra;
    pthread_mutexattr_init(&ra);
    pthread_mutexattr_settype(&ra, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&m->dev_mu, &ra);
    pthread_mutex_init(&m->tr_mu, NULL);
    pthread_mutex_init(&m->stw_mu, NULL);
    pthread_cond_init(&m->stw_cv, NULL);
    for (int i = 0; i < HV_MAX_CPUS; i++) { /* untouched pages cost nothing, so every CPU gets its tables now */
        HvPerCpu *p = &m->percpu[i];
        p->ijc = calloc(HV_IJC_TABLES * HV_IJC_TABLE_BYTES / 8, 8);
        p->ucode = calloc(1u << HV_UCODE_BITS, 8);
        p->cid_chunks = calloc(65536, sizeof *p->cid_chunks);
        if (!p->ijc || !p->ucode || !p->cid_chunks) return -1;
        for (int k = 0; k < HV_STASHES; k++) {
            p->stash[k].slot = calloc(HV_STASH_CAP, sizeof *p->stash[k].slot);
            p->stash[k].ent = calloc(HV_STASH_CAP, sizeof *p->stash[k].ent);
            if (!p->stash[k].slot || !p->stash[k].ent) return -1;
        }
        pthread_mutex_init(&p->wake_mu, NULL);
        pthread_cond_init(&p->wake_cv, NULL);
        pthread_mutex_init(&p->req_mu, NULL);
    }
    return 0;
}

__thread HvCpu *hv_self;

void hv_dev_lock(HvMachine *m) {
    if (pthread_mutex_trylock(&m->dev_mu) == 0) return;
    uint64_t t0 = m->stats_interval ? mach_absolute_time() : 0;
    pthread_mutex_lock(&m->dev_mu);
    if (m->stats_interval) {
        m->stats.dev_waits++;
        m->stats.dev_wait_ticks += mach_absolute_time() - t0;
    }
}

void hv_dev_unlock(HvMachine *m) {
    pthread_mutex_unlock(&m->dev_mu);
}

static uint64_t post_locked_done(HvCpu *target, HvPerCpu *p) {
    uint64_t seq = __atomic_add_fetch(&p->req_seq, 1, __ATOMIC_SEQ_CST);
    pthread_mutex_unlock(&p->req_mu);
    __atomic_store_n(&target->exit_req, 1, __ATOMIC_SEQ_CST);
    __asm__ volatile("sev" ::: "memory"); /* a WFE waiting in place (wfe() in dispatch.c) sees it now */
    if (target->m->stats_interval) target->m->stats.kick_src[0]++;
    return seq;
}

uint64_t hv_cpu_post_tlbi(HvCpu *target, uint32_t op1, uint32_t crn, uint32_t crm, uint32_t op2, uint64_t xt) {
    HvPerCpu *p = target->pc_state;
    pthread_mutex_lock(&p->req_mu);
    if (p->nreq < sizeof p->req / sizeof p->req[0]) {
        p->req[p->nreq].op1 = op1, p->req[p->nreq].crn = crn, p->req[p->nreq].crm = crm, p->req[p->nreq].op2 = op2;
        p->req[p->nreq].xt = xt;
        p->nreq++;
    } else {
        p->req_flags |= HV_REQ_TLB_ALL; /* too many queued: drop everything instead */
    }
    return post_locked_done(target, p);
}

/* Code changed in a page whose blocks were cached as branch targets at VA page v (| 2), or v = 1: only the return
   predictions. Too many queued: the whole branch cache instead. */
uint64_t hv_cpu_post_ijc_page(HvCpu *target, uint64_t v) {
    HvPerCpu *p = target->pc_state;
    pthread_mutex_lock(&p->req_mu);
    if (p->nijc_page < sizeof p->ijc_page / sizeof p->ijc_page[0]) p->ijc_page[p->nijc_page++] = v;
    else p->req_flags |= HV_REQ_IJC_RESET;
    return post_locked_done(target, p);
}

uint64_t hv_cpu_post(HvCpu *target, uint32_t flags) {
    HvPerCpu *p = target->pc_state;
    pthread_mutex_lock(&p->req_mu);
    p->req_flags |= flags;
    return post_locked_done(target, p);
}

/* HV_SYNC_WAIT: a CPU waiting for another's acknowledgement sleeps (after a short spin) instead of
   spinning and yielding, so a host core is free for the CPU it waits on: the phone has two fast cores
   for three guest CPUs, the display and the app. The kernel's compare-and-wait on a 32-bit word. */
extern int __ulock_wait(uint32_t operation, void *addr, uint64_t value, uint32_t timeout_us);
extern int __ulock_wake(uint32_t operation, void *addr, uint64_t wake_value);
#define HV_UL_COMPARE_AND_WAIT 1u
#define HV_ULF_WAKE_ALL 0x100u
#define HV_ULF_NO_ERRNO 0x1000000u

/* Wake whoever sleeps on p's ack_seq (it moved, or p stopped using its TLB). */
static void ack_wake(HvPerCpu *p) {
    if (__atomic_load_n(&p->ack_waiters, __ATOMIC_SEQ_CST))
        __ulock_wake(HV_UL_COMPARE_AND_WAIT | HV_ULF_WAKE_ALL | HV_ULF_NO_ERRNO, &p->ack_seq, 0);
}

void hv_cpu_ack_wake(HvCpu *cpu) {
    if (cpu->pc_state) ack_wake(cpu->pc_state);
}

void hv_cpu_process_requests(HvCpu *cpu) {
    HvPerCpu *p = cpu->pc_state;
    uint64_t seq = __atomic_load_n(&p->req_seq, __ATOMIC_ACQUIRE);
    if (seq == __atomic_load_n(&p->ack_seq, __ATOMIC_RELAXED)) return;
    static __thread struct { uint32_t op1, crn, crm, op2; uint64_t xt; } q[64];
    static __thread uint64_t pages[16];
    pthread_mutex_lock(&p->req_mu);
    seq = p->req_seq;
    uint32_t flags = p->req_flags, n = p->nreq, npages = p->nijc_page;
    memcpy(q, p->req, n * sizeof q[0]);
    memcpy(pages, p->ijc_page, npages * sizeof pages[0]);
    p->req_flags = 0;
    p->nreq = 0;
    p->nijc_page = 0;
    pthread_mutex_unlock(&p->req_mu);
    if (flags & HV_REQ_TLB_ALL) {
        cpu->m->stats.flush_why[7]++;
        hv_tlb_flush(cpu);
    } else {
        for (uint32_t i = 0; i < n; i++) hv_tlbi_apply(cpu, q[i].op1, q[i].crn, q[i].crm, q[i].op2, q[i].xt);
    }
    if (flags & HV_REQ_IJC_RESET) {
        hv_ijc_reset(cpu, 0);
        hv_ijc_reset(cpu, 1);
    } else {
        for (uint32_t i = 0; i < npages; i++) hv_ijc_drop_code_page(cpu, pages[i]);
    }
    __atomic_store_n(&p->ack_seq, seq, __ATOMIC_RELEASE);
    ack_wake(p);
}

void hv_cpu_sync(HvCpu *self, HvCpu *target, uint64_t seq) {
    HvPerCpu *t = target->pc_state, *s = self ? self->pc_state : NULL;
    HvMachine *mm = target->m;
    uint64_t t0 = mm->stats_interval ? mach_absolute_time() : 0;
    if (s) {
        __atomic_store_n(&s->tlb_live, 0, __ATOMIC_SEQ_CST); /* others need not wait for this one meanwhile */
        ack_wake(s);
    }
    int sleep = hv_knob[HV_K_SYNCWAIT];
    for (uint32_t spins = 0;; spins++) {
        uint64_t ack = __atomic_load_n(&t->ack_seq, __ATOMIC_ACQUIRE);
        if (ack >= seq) break;
        if (!__atomic_load_n(&t->tlb_live, __ATOMIC_SEQ_CST)) break; /* parked: it does the work before it runs again */
        if (self) hv_cpu_process_requests(self);
        if (sleep && spins >= 64) {
            __atomic_add_fetch(&t->ack_waiters, 1, __ATOMIC_SEQ_CST);
            /* Asleep until ack_seq's low word moves or 100 us pass (a missed wake costs no more than that). */
            if (__atomic_load_n(&t->ack_seq, __ATOMIC_SEQ_CST) == ack && __atomic_load_n(&t->tlb_live, __ATOMIC_SEQ_CST))
                __ulock_wait(HV_UL_COMPARE_AND_WAIT | HV_ULF_NO_ERRNO, &t->ack_seq, (uint32_t)ack, 100);
            __atomic_sub_fetch(&t->ack_waiters, 1, __ATOMIC_SEQ_CST);
        } else if (spins > 200) {
            sched_yield();
        }
    }
    if (s) {
        __atomic_store_n(&s->tlb_live, 1, __ATOMIC_SEQ_CST);
        hv_cpu_process_requests(self); /* anything that came in while we were not live */
    }
    if (mm->stats_interval) {
        mm->stats.syncs++;
        mm->stats.sync_ticks += mach_absolute_time() - t0;
    }
}

void hv_big_lock(HvMachine *m) {
    if (!m->big_lock_on) return;
    pthread_mutex_lock(&m->big_mu);
    uint64_t me = m->big_next++;
    while (m->big_serving != me) pthread_cond_wait(&m->big_cv, &m->big_mu);
    pthread_mutex_unlock(&m->big_mu);
}

void hv_big_unlock(HvMachine *m) {
    if (!m->big_lock_on) return;
    pthread_mutex_lock(&m->big_mu);
    m->big_serving++;
    pthread_cond_broadcast(&m->big_cv);
    pthread_mutex_unlock(&m->big_mu);
}

void hv_big_yield(HvMachine *m) {
    if (!m->big_lock_on) return;
    pthread_mutex_lock(&m->big_mu);
    int waiting = m->big_next - m->big_serving > 1;
    pthread_mutex_unlock(&m->big_mu);
    if (!waiting) return;
    hv_big_unlock(m);
    hv_big_lock(m);
}

/* Make cpu leave its block at its next checked exit (and end an in-place WFE wait). */
void hv_cpu_kick(HvCpu *cpu) {
    __atomic_store_n(&cpu->exit_req, 1, __ATOMIC_SEQ_CST);
    __asm__ volatile("sev" ::: "memory");
}

/* Return tables: nothing to return to. A tag is never 0: salts are not, and no VA has their bits. */
void hv_ret_clear(HvCpu *cpu) {
    memset(cpu->ret_tab, 0, sizeof cpu->ret_tab);
    cpu->ret_salt[0] = cpu->ret_salt[1] = 1ull << 48;
}

void hv_cpu_wake(HvCpu *cpu) {
    __atomic_store_n(&cpu->exit_req, 1, __ATOMIC_SEQ_CST);
    __asm__ volatile("sev" ::: "memory");
    if (cpu->m->stats_interval) cpu->m->stats.kick_src[1]++;
    HvPerCpu *p = cpu->pc_state;
    pthread_mutex_lock(&p->wake_mu);
    p->wake_pending = 1;
    pthread_cond_signal(&p->wake_cv);
    pthread_mutex_unlock(&p->wake_mu);
}

void hv_machine_kick_all(HvMachine *m) {
    if (m->stats_interval) m->stats.kick_src[2]++;
    for (int i = 0; i < m->ncpus; i++)
        if (m->cpus[i]) __atomic_store_n(&m->cpus[i]->exit_req, 1, __ATOMIC_SEQ_CST);
    __asm__ volatile("sev" ::: "memory");
}

/* Device work is for CPU 0, which polls the devices. */
void hv_machine_wake(HvMachine *m) {
    if (m->stats_interval) m->stats.kick_src[3]++;
    if (m->cpu) hv_cpu_wake(m->cpu);
}

void hv_machine_free(HvMachine *m) {
    hv_jit_region_close(&m->jit);
    for (int i = 0; i < m->nram; i++) {
        free(m->ram[i].code_pages);
        free(m->ram[i].page_gen);
    }
    free(m->blocks);
    for (int i = 0; i < HV_MAX_CPUS; i++) {
        free(m->percpu[i].ijc);
        free(m->percpu[i].ucode);
    }
    free(m->slots);
    free(m->slot_site);
    free(m->slot_stub);
    free(m->slot_cond);
    free(m->slot_xa);
    free(m->slot_xpa);
    free(m->xchained);
}

int hv_machine_add_ram(HvMachine *m, uint64_t pa, uint64_t size, void *host, int readonly) {
    if (m->nram == HV_MAX_RAM) return -1;
    HvRam *r = &m->ram[m->nram++];
    r->pa = pa;
    r->size = size;
    r->host = host;
    r->readonly = readonly;
    size_t pages = (size + (1u << HV_TLB_PAGE_BITS) - 1) >> HV_TLB_PAGE_BITS;
    r->pages_log2 = 0;
    while ((1ull << r->pages_log2) < pages) r->pages_log2++;
    /* A power of two, so translated code bounds-checks a page index with one shift (IC IVAU). */
    r->code_pages = calloc((size_t)1 << r->pages_log2, 1);
    if (!m->main_ram || size > m->main_ram->size) m->main_ram = r;
    r->page_gen = calloc(pages, sizeof *r->page_gen);
    r->npages = pages;
    r->whint = calloc(pages * HV_MAX_CPUS, sizeof *r->whint);
    r->ijc_va = calloc(pages, sizeof *r->ijc_va);
    return r->code_pages && r->page_gen && r->whint && r->ijc_va ? 0 : -1;
}

int hv_machine_add_mmio(HvMachine *m, uint64_t pa, uint64_t size, void *opaque,
                        uint64_t (*read)(void *, uint64_t, unsigned),
                        void (*write)(void *, uint64_t, uint64_t, unsigned), const char *name) {
    if (m->nmmio == HV_MAX_MMIO) return -1;
    m->mmio[m->nmmio++] = (HvMmio){pa, size, opaque, read, write, name, 0};
    return 0;
}

HvRam *hv_ram_lookup(HvMachine *m, uint64_t pa) {
    for (int i = 0; i < m->nram; i++) {
        HvRam *r = &m->ram[i];
        if (pa - r->pa < r->size) return r;
    }
    return NULL;
}

uint32_t *hv_smc_counts;
uint64_t *hv_smc_last;
uint64_t hv_smc_pages, hv_smc_cpu, hv_smc_dma;

void hv_invalidate_code(HvMachine *m, uint64_t pa) {
    HvRam *r = hv_ram_lookup(m, pa);
    if (!r) return;
    uint64_t pg = (pa - r->pa) >> HV_TLB_PAGE_BITS;
    if (!r->code_pages[pg]) return;
    /* Blocks remember the generation they were made under; a stale one is
       retranslated the next time it is looked up. The block running now may
       finish with the old code, as it would with a real instruction pipeline. */
    r->code_pages[pg] = 0;
    __atomic_add_fetch(&r->page_gen[pg], 1, __ATOMIC_SEQ_CST);
    m->stats.flush_smc++;
    if (hv_smc_counts && pg < hv_smc_pages) {
        hv_smc_counts[pg]++;
        uint64_t caller = hv_self ? hv_self->pc : 0;
        hv_smc_last[pg] = caller;
        if (hv_self) hv_smc_cpu++;
        else hv_smc_dma++;
    }
    /* Exits from other pages chained into the stale blocks go back to their stub
       (whose next chain finds the new translation). Under the translation lock,
       as chaining them is: one that saw the old generation is done by now. */
    pthread_mutex_lock(&m->tr_mu);
    uint64_t tpa = (r->pa + (pg << HV_TLB_PAGE_BITS)) | 1;
    for (uint32_t k = 0; k < m->nxchained;) {
        uint32_t idx = m->xchained[k];
        if (m->slot_xpa[idx] != tpa) {
            k++;
            continue;
        }
        m->slot_xpa[idx] = 0;
        __atomic_store_n(&m->slots[idx], 0, __ATOMIC_RELEASE);
        uint32_t *site = m->slot_site[idx];
        if (site) {
            uint32_t insn = 0x14000000u | (uint32_t)((m->slot_stub[idx] - site) & 0x3ffffff); /* back to its stub */
            hv_jit_write(&m->jit, site, &insn, 4);
            hv_jit_finish(&m->jit, site, 1);
        }
        m->xchained[k] = m->xchained[--m->nxchained];
    }
    pthread_mutex_unlock(&m->tr_mu);
    /* The indirect-branch cache may lead into the stale blocks: drop those entries (or, without HV_IJC_PAGES, the
       whole cache), here at once and on the other CPUs before they next leave a block (their cache lookups and
       chained exits check exit_req first). Which entries: the page's blocks were cached under one VA page (ijc_va,
       read after the generation went up; a fill notes its page before checking the generation, see
       hv_ijc_fill_checked), so only that page's; under none, only the return predictions go; under several, all. */
    uint64_t v = __atomic_exchange_n(&r->ijc_va[pg], 0, __ATOMIC_SEQ_CST);
    if (!hv_knob[HV_K_IJCPAGES]) v = ~0ull;
    if (m->stats_interval) m->stats.code_ijc[!v ? 0 : v == ~0ull ? 2 : 1]++;
    if (!v) v = 1;
    for (int i = 0; i < m->ncpus; i++) {
        HvCpu *c = m->cpus[i];
        if (!c) continue;
        /* Only a CPU may reset its own cache; from any other thread (a device,
           or no CPU running yet) it is a request, or a running CPU could read a
           half-written entry and jump into the wrong code. */
        if (c == hv_self) {
            if (v == ~0ull) {
                hv_ijc_reset(c, 0);
                hv_ijc_reset(c, 1);
            } else {
                hv_ijc_drop_code_page(c, v);
            }
        } else if (v == ~0ull) {
            hv_cpu_post(c, HV_REQ_IJC_RESET);
        } else {
            hv_cpu_post_ijc_page(c, v);
        }
    }
    m->stats.reset_why[4]++;
    /* Make a running chain leave (chains stay patched: see chain() in dispatch.c). */
    hv_machine_kick_all(m);
}

void hv_dma_written(HvMachine *m, uint64_t pa, uint64_t len) {
    HvRam *r = hv_ram_lookup(m, pa);
    if (!r || !len || pa + len - r->pa > r->size) return;
    uint64_t first = (pa - r->pa) >> HV_TLB_PAGE_BITS, last = (pa + len - 1 - r->pa) >> HV_TLB_PAGE_BITS;
    for (uint64_t pg = first; pg <= last; pg++)
        if (__atomic_load_n(&r->code_pages[pg], __ATOMIC_ACQUIRE)) hv_invalidate_code(m, r->pa + (pg << HV_TLB_PAGE_BITS));
}

HvMmio *hv_mmio_lookup(HvMachine *m, uint64_t pa) {
    for (int i = 0; i < m->nmmio; i++) {
        HvMmio *d = &m->mmio[i];
        if (pa - d->pa < d->size) return d;
    }
    return NULL;
}

void *hv_phys_ptr(HvMachine *m, uint64_t pa, uint64_t len) {
    HvRam *r = hv_ram_lookup(m, pa);
    if (!r || pa + len - r->pa > r->size) return NULL;
    return r->host + (pa - r->pa);
}

/* MMIO is accessed in naturally sized pieces of at most 8 bytes. */
int hv_phys_read(HvMachine *m, uint64_t pa, void *buf, unsigned len) {
    void *p = hv_phys_ptr(m, pa, len);
    if (p) {
        memcpy(buf, p, len);
        return 0;
    }
    HvMmio *d = hv_mmio_lookup(m, pa);
    if (!d || len > 8) return -1;
    if (!d->self_locked) hv_dev_lock(m);
    uint64_t v = d->read ? d->read(d->opaque, pa - d->pa, len) : 0;
    if (!d->self_locked) hv_dev_unlock(m);
    memcpy(buf, &v, len);
    if (m->log >= 3)
        fprintf(stderr, "mmio read  %s+%llx size %u = %llx pc %llx\n", d->name, (unsigned long long)(pa - d->pa), len,
                (unsigned long long)v, (unsigned long long)m->access_pc);
    return 0;
}

int hv_phys_write(HvMachine *m, uint64_t pa, const void *buf, unsigned len) {
    HvRam *r = hv_ram_lookup(m, pa);
    if (r && pa + len - r->pa <= r->size) {
        if (r->readonly) return 0; /* ROM: writes are ignored */
        uint64_t off = pa - r->pa;
        uint64_t first = off >> HV_TLB_PAGE_BITS, last = (off + len - 1) >> HV_TLB_PAGE_BITS;
        for (uint64_t pg = first; pg <= last; pg++)
            if (r->code_pages[pg]) hv_invalidate_code(m, r->pa + (pg << HV_TLB_PAGE_BITS)); /* a write to code */
        if (m->watch_pa && m->watch_pa - pa < len) {
            uint64_t v = 0;
            memcpy(&v, (const uint8_t *)buf + (m->watch_pa - pa), len - (m->watch_pa - pa) < 8 ? len - (m->watch_pa - pa) : 8);
            fprintf(stderr, "hengevm: watch: write to pa %llx = %llx (%u bytes at pa %llx) by pc %llx\n",
                    (unsigned long long)m->watch_pa, (unsigned long long)v, len, (unsigned long long)pa,
                    (unsigned long long)m->access_pc);
        }
        memcpy(r->host + off, buf, len);
        return 0;
    }
    HvMmio *d = hv_mmio_lookup(m, pa);
    if (!d || len > 8) return -1;
    uint64_t v = 0;
    memcpy(&v, buf, len);
    if (m->log >= 3)
        fprintf(stderr, "mmio write %s+%llx size %u = %llx pc %llx\n", d->name, (unsigned long long)(pa - d->pa), len,
                (unsigned long long)v, (unsigned long long)m->access_pc);
    if (!d->self_locked) hv_dev_lock(m);
    if (d->write) d->write(d->opaque, pa - d->pa, v, len);
    if (!d->self_locked) hv_dev_unlock(m);
    return 0;
}

void hv_cpu_reset(HvCpu *cpu, HvMachine *m) {
    hv_cpu_reset_index(cpu, m, 0);
}

void hv_cpu_reset_index(HvCpu *cpu, HvMachine *m, uint32_t index) {
    /* Everything but m and pc_state: other CPUs' TLBIs read those of a CPU
       being started (PSCI CPU_ON) at any time. */
    _Static_assert(offsetof(HvCpu, m) < offsetof(HvCpu, pc_state), "HvCpu layout");
    size_t a = offsetof(HvCpu, m), b = offsetof(HvCpu, pc_state);
    memset(cpu, 0, a);
    memset((char *)cpu + a + sizeof cpu->m, 0, b - a - sizeof cpu->m);
    memset((char *)cpu + b + sizeof cpu->pc_state, 0, sizeof *cpu - b - sizeof cpu->pc_state);
    cpu->m = m;
    cpu->index = index;
    cpu->pc_state = &m->percpu[index];
    m->cpus[index] = cpu;
    if (m->ncpus <= (int)index) m->ncpus = (int)index + 1;
    if (index == 0) m->cpu = cpu;
    HvPerCpu *p = cpu->pc_state;
    /* Ids start over, so entries tagged with the old ones must go. */
    for (int t = 0; t < HV_IJC_TABLES; t++)
        if (p->ijc_tables_used >> t & 1) memset(p->ijc + (size_t)t * (HV_IJC_TABLE_BYTES / 8), 0, HV_IJC_TABLE_BYTES);
    p->ijc_tables_used = 0;
    memset(p->ijc_keyed, 0, sizeof p->ijc_keyed);
    p->ijc_keyed_count = 0;
    memset(p->ijc_pages, 0, sizeof p->ijc_pages);
    memset(p->ijc_hint, 0, sizeof p->ijc_hint);
    memset(p->jc, 0, sizeof p->jc);
    cpu->ijc = (uint64_t)(uintptr_t)p->ijc;
    cpu->lines_ptr = (uint64_t)(uintptr_t)&p->irq_line;
    for (int t = 0; t < HV_IJC_TABLES; t++) cpu->ijc_tab[t] = (uint64_t)(uintptr_t)(p->ijc + (size_t)t * (HV_IJC_TABLE_BYTES / 8));
    hv_ret_clear(cpu);
    cpu->next_cid = 1;
    cpu->kernel_gen = 1;
    cpu->ijc_salt[1] = 1ull << 48; /* tags of an all-zero table never match */
    hv_ijc_switch(cpu);
    cpu->exit_routine = (uint64_t)hv_exit;
    cpu->helper_routine = (uint64_t)hv_helper_entry;
    cpu->helper_thunk = (uint64_t)hv_helper_thunk;
    cpu->chain_thunk = (uint64_t)hv_chain_thunk;
    cpu->ijc_thunk = (uint64_t)hv_ijc_thunk;
    cpu->xjump_thunk = (uint64_t)hv_xjump_thunk;
    cpu->el = 1;
    cpu->spsel = 1;
    cpu->daif = HV_PSTATE_DAIF;
    cpu->excl_addr = ~0ull;
    /* Architectural reset value has the MMU and caches off; RES1 bits set. */
    cpu->sctlr_el1 = 0x30d00800;
    cpu->nkernel_fills = HV_KERNEL_FILLS + 1; /* nothing is logged yet: make every entry invalid */
    hv_tlb_flush(cpu);
}
