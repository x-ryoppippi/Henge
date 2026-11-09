#ifndef HENGEVM_MACHINE_H
#define HENGEVM_MACHINE_H

#include <pthread.h>
#include <stdint.h>

#include "cpu.h"
#include "jit_region.h"

/* Guest physical memory: RAM regions backed by host memory, and MMIO regions
   handled by device callbacks. The fast path only ever sees RAM. */
typedef struct HvRam {
    uint64_t pa, size;
    uint8_t *host;
    int readonly;
    uint8_t *code_pages; /* one byte per 16 KB page: a translated block came from here (2^pages_log2 of them) */
    uint32_t pages_log2;
    uint32_t *page_gen;  /* per page: bumped to invalidate the blocks translated from it */
    /* Per page and CPU: up to three TLB entries (slot << 2 | table, plus 1;
       16 bits each) that mapped the page writable, or HV_WHINT_MANY when there
       were more. When the page becomes code only those are dropped, not a whole
       TLB scanned. */
    uint64_t *whint;     /* CPU-major: whint[cpu * npages + page], so CPUs filling the same pages write their own lines */
    /* Per page: the VA page (| 2) its blocks have been cached as branch targets under since its code last changed;
       0 none, ~0 more than one. So a code change drops just those cached targets (hv_invalidate_code). */
    uint64_t *ijc_va;
    size_t npages;
} HvRam;

typedef struct HvMmio {
    uint64_t pa, size;
    void *opaque;
    uint64_t (*read)(void *opaque, uint64_t off, unsigned size);
    void (*write)(void *opaque, uint64_t off, uint64_t val, unsigned size);
    const char *name;
    int self_locked;     /* the device takes its own lock (the GIC): no device lock around it */
} HvMmio;

typedef struct HvBlock {
    uint32_t jit_gen; /* full flushes when it was made */
    uint64_t va, pa;
    uint32_t flags;      /* HV_BLOCK_* state the translation depends on */
    uint32_t ninsns;
    uint32_t gen;        /* page_gen of its page when translated */
    struct HvRam *ram;   /* where its page lives, for a quick staleness check */
    uint64_t pg;
    uint32_t *code;
    struct HvBlock *next;
    uint32_t maker;      /* the CPU that translated it (index + 1; 0: not a CPU, as in tests) */
} HvBlock;

#define HV_BLOCK_EL1   1u        /* translated for EL1 (selects tlb[1]) */
#define HV_BLOCK_FPOFF 2u        /* CPACR_EL1.FPEN traps FP/SIMD at this EL */
#define HV_BLOCK_SPX   4u        /* EL1 with SPSel = 1 (SP_EL1 in use, SP_EL0 banked) */
#define HV_BLOCK_PAN   8u        /* EL1 with PSTATE.PAN set (selects TLB table 2) */

#define HV_MAX_RAM 8
#define HV_MAX_MMIO 48
#define HV_BLOCK_HASH (1 << 21) /* one bucket per block: chains are walked on every front-cache miss */
#define HV_JC_BITS 16
#define HV_MAX_BLOCKS (3 << 20) /* a macOS boot runs well over a million distinct blocks; a full flush retranslates them all */
#define HV_PROF_RING (1 << 16)
#define HV_MAX_SLOTS (6 << 20)  /* chainable exits: about two per block */

/* Counters for --stats: where the time goes. */
typedef struct HvStats {
    uint64_t blocks, translations, block_flushes;
    uint64_t flush_smc, flush_ic, flush_full;  /* why translations were dropped */
    uint64_t chained, ijc_fills, ijc_resets;
    uint64_t xchained;
    uint64_t kick_src[5];     /* exit_req set by: a request (TLBI), hv_cpu_wake, kick_all, hv_machine_wake, the CPU itself */        /* of them, exits to another page (TLB-checked) */
    uint64_t syncs, sync_ticks;        /* TLBI acks waited for, and host ticks spent waiting */
    uint64_t dev_waits, dev_wait_ticks; /* device lock taken while another CPU held it */
    uint64_t tr_waits, tr_wait_ticks;   /* translation lock likewise */
    uint64_t words_inline, words_ool, words_pool; /* translated code size by part */
    uint64_t refills;
    uint64_t ijc_light;       /* branch-cache misses resolved without the dispatcher (hv_ijc_resolve) */         /* TLB misses filled by the light path (hv_tlb_refill) */
    uint64_t stash_saves, stash_hits, stash_entries; /* user TLB entries kept across process switches */
    uint64_t code_protects;   /* pages that became code (every CPU's TLB scanned for writable entries) */
    uint64_t tlb_miss_kind[4]; /* TLB misses: EL0 empty, EL0 other page there, EL1 empty, EL1 other page there */
    uint64_t stash_stale;     /* process switches that found the stash dropped by a TLBI meanwhile */
    uint64_t slow_why[6];
    uint64_t small_pages;     /* of the TLB misses above: the guest maps the page with a granule under 16 KB (4 KB), which the TLB cannot hold */
    struct { uint64_t mb, n; } small_at[8]; /* where (VA >> 20, bit 0 = EL1) the first 8 such regions were */     /* slow accesses by cause: MMIO, code-page write, page end, TLB miss, other, TLB entry there but mismatched */
    uint64_t slow_kind[12];   /* hv_mem_emulate calls by HV_MEM_* kind */
    uint64_t light_kind[4];
    uint64_t light_fail[4];
    uint64_t light_walks;
    uint64_t fast_resumes;
    uint64_t fr_fail[5];    /* SVCs and ERETs that went to their block without the dispatcher */
    uint64_t protect_kind[3]; /* a page becoming code, per CPU: not mapped writable, one entry, scan */
    uint64_t light_region[2]; /* user light fills: in the shared region (0x18..0x28 << 28), elsewhere */
    uint64_t chained_early;   /* exits chained as their block was placed */
    uint64_t wfe_waits;       /* WFEs that waited on the monitored word in place */
    uint64_t kind_insns[16], kind_words[16]; /* --stats: inline words by instruction kind (HvInsnKind), as translated */
    uint64_t posted_kind[4];
    uint64_t race_kind[3];    /* remote drops that raced: walk under way, cache fill under way, either during the drop */
    uint64_t race_other_page; /* HV_WALK_LOG: drops not counted as raced, the walks meanwhile being of other pages */  /* TLBIs other CPUs had to do themselves: a page or small range that raced, a big range, the rest (ASID, all) */
    uint64_t range_spared;    /* big user-range TLBIs that needed no cached targets dropped */
    uint64_t code_ijc[3];     /* code changes: no cached target into the page, its VA page's targets dropped, all dropped (HV_IJC_PAGES) */
    uint64_t ijc_reset_kind[4]; /* whole branch-cache drops: user half, kernel half, one process's by TLBI ASID, one process's by a range of 64 MB or more */
    uint64_t kernel_ranges;   /* big kernel-range TLBIs, dropped by range rather than with everything */
    uint64_t tlb_range_flushes; /* and that dropped only the TLB entries in the range */     /* of hv_ijc_resolve's lookups, ones that walked for the target page */   /* hv_ijc_resolve could not: chaining off/odd target, filters full, no TLB entry, block not made yet */   /* the same for hv_ijc_resolve's fills */
    uint64_t fill_kind[4];    /* ijc fills into an empty slot, over another live salt, over the same salt, over a dead salt */
    uint64_t flush_why[8];
    uint64_t flush_after_switch[4];
    uint64_t flush_checked, flush_check_entries, flush_check_bad, flush_check_af; /* HV_SWITCH_FLUSH=2: flushes checked, entries walked, out of date (the flush was done), only the access flag */
    uint64_t flush_skipped;   /* TLBI all here right after a process switch, left out (HV_SWITCH_FLUSH=1) */
    uint64_t flush_skipped_late; /* of them, 20-200 us after the switch at a PC seen within 20 us of one (HV_SWITCH_FLUSH=3) */ /* TLBI all here, by time since this CPU's last TTBR0 change: <2 us, <20 us, <200 us, later */
    struct { uint64_t pc, n; } flush_pc[8]; /* where TLBI all here comes from (first 8 PCs seen) */    /* whole-TLB flushes: TLBI all (local), TLBI all (shared), SCTLR, TTBR1, TCR, MAIR, range of everything, requests overflowed */
    uint64_t reset_why[8];    /* cache drops: full flush, TLBI by ASID, TLBI user VA, TLBI kernel VA, code change, user VA in every ASID (whole user half), keyed filter full, big user range */
    uint64_t why[6];          /* dispatcher entries by cause: chain fill, kick, exception/sysreg, interrupt, cached target, cache miss */
    uint64_t miss_kind[7];    /* cache misses by the branch that missed: RET, BR/BLR, B/BL, conditional, MSR, other, unknown */
    uint64_t recent[16];      /* debugging: the last blocks the dispatcher entered (pc | el) */
    struct { uint64_t from, to, spsr; uint32_t kind, el; } elchg[16]; /* debugging: last exception entries (kind = type) and ERETs (kind 9) */
    uint32_t nelchg;
    uint64_t nrecent;
    uint64_t translate_ticks, translated_insns, translated_words; /* time spent translating, guest and host code size */
    uint64_t retrans_va, retrans_vapa; /* translations of a VA (and VA+PA) translated before: diagnostics */
    uint64_t helper[12];      /* by HV_OP_* */
    uint64_t tlb_flushes, fetch_walks, exceptions, interrupts;
    uint64_t wfi, wfi_ticks;  /* WFI exits, host ticks spent asleep */
} HvStats;

#define HV_MAX_CPUS 8
#define HV_WHINT_MANY (1ull << 63) /* whint: more than three; else up to three TLB codes + 1, 16 bits each */

/* What each CPU has to itself and keeps across hv_cpu_reset (tests copy HvCpu
   structs around, so it lives here, by CPU index): its indirect-branch cache and
   the filters for dropping entries from it, the front cache of blocks, its
   interrupt lines and its wakeup. */
/* A process's user TLB entries, put aside when it is switched out (TTBR0 now
   names another address space) and put back when it comes back, as a TLB
   tagged by ASID keeps them. Any TLBI that could touch the address space
   (here or from another CPU) bumps gen; a stash whose gen moved is dropped. */
#define HV_STASHES 8
#define HV_STASH_CAP 1024
typedef struct HvTlbStash {
    uint64_t ttbr0;      /* the address space (0: empty); read by other CPUs */
    uint32_t gen;        /* bumped by TLBIs (atomic) */
    uint32_t saved_gen;  /* gen when the entries were saved */
    uint32_t n;
    uint64_t used;       /* for replacing the least recently saved */
    uint16_t *slot;      /* slot << 2 | table, as user_fills */
    HvTlbEntry *ent;
} HvTlbStash;

typedef struct HvPerCpu {
    /* Interrupt lines into this CPU, driven by the GIC. First and together:
       translated code reads both with one load (HvCpu.lines_ptr). */
    int irq_line, fiq_line;
    uint64_t *ijc;           /* indirect-branch cache tables (cpu->ijc points here) */
    uint64_t ijc_pages[2][1024]; /* hashed bitmap of the user pages any process cached targets in (for all-ASID TLBIs) */
    /* Hashed bitmap of (page, salt) pairs with cached targets: a TLBI for one
       address space drops its entries only if that address space cached
       something in the page. A new salt makes old bits meaningless, so this is
       only cleared (with the whole cache) when it gets an eighth full. */
    uint64_t ijc_keyed[1 << 14];
    uint32_t ijc_keyed_count;
    uint32_t ijc_tables_used; /* bit per flags value whose table was ever filled */
    /* Exactly which (user id, user page) pairs have cached targets: a hash set,
       so unmapping a big range of one address space drops only its code pages'
       entries instead of the whole address space's. Emptied with the user half. */
    uint64_t *ucode;
    uint32_t ucode_count;
    /* Per user id: which 32 MB chunks of its VA (hashed into 256) have cached
       targets, so a TLBI for a big range elsewhere (data, nearly always) drops nothing. */
    uint64_t (*cid_chunks)[4];
    uint64_t last_fill[2];
    /* Per half, hashed by (page ^ salt): which 256-byte pieces of the page have
       cached targets, so dropping a page clears 64 entries per piece instead of
       all 4096. A slot two keys wanted is spoiled (key ~0: scan it all); only
       this CPU empties one again, after clearing its page. */
    struct { uint64_t key, bits; } ijc_hint[2][4096];   /* page ^ salt of the last fill in each half: its filter bits are set already */
    /* Direct-mapped cache in front of the hash, with the VA beside the block so
       a slot holding another block is passed over without reading it. */
    struct { uint64_t va; struct HvBlock *b; } jc[1 << HV_JC_BITS];
    /* WFI sleeps on this; device threads, the GIC and other CPUs signal it. */
    pthread_mutex_t wake_mu;
    pthread_cond_t wake_cv;
    int wake_pending;
    /* Work other CPUs ask of this one (only the owner may touch its TLB and
       branch cache while it runs). Posting bumps req_seq; the owner does the
       work at its next dispatcher entry and publishes ack_seq. */
    pthread_mutex_t req_mu;
    uint32_t req_flags;          /* HV_REQ_* */
    uint32_t nreq;
    struct { uint32_t op1, crn, crm, op2; uint64_t xt; } req[64]; /* TLBIs to apply */
    uint32_t nijc_page;
    uint64_t ijc_page[16];       /* code changed: cached targets to forget (hv_ijc_drop_code_page), queued as req */
    uint64_t req_seq, ack_seq;
    /* active: inside the dispatcher loop (0 while parked: WFI, powered off, at a
       flush barrier). tlb_live: active and not itself waiting for acks, i.e. it
       may be using TLB entries right now. */
    int active, tlb_live;
    uint32_t ack_waiters;        /* CPUs asleep in hv_cpu_sync waiting for ack_seq to move (HV_SYNC_WAIT) */
    uint32_t walk_seq;           /* odd while it walks page tables and fills its TLB (see hv_tlbi_drop_remote_va) */
    uint64_t walk_page[8];       /* the page walk k (walk_seq 2k -> 2k+1) fills, at [k % 8]; HV_WALK_ANY: a process switch */
    uint32_t ijc_seq;            /* odd while it checks a block's mapping in its TLB and caches it (hv_ijc_fill_checked) */
    HvTlbStash stash[HV_STASHES];
    uint64_t stash_clock;
} HvPerCpu;

#define HV_REQ_TLB_ALL 1u   /* drop the whole TLB and branch cache */
#define HV_REQ_IJC_RESET 2u /* drop the branch cache (code changed) */

typedef struct HvMachine {
    HvRam ram[HV_MAX_RAM];
    int nram;
    HvRam *main_ram;     /* the largest region: guest RAM */
    HvMmio mmio[HV_MAX_MMIO];
    int nmmio;

    HvJitRegion jit;
    HvBlock *hash[HV_BLOCK_HASH];
    HvBlock *blocks;
    int nblocks;
    uint32_t max_blocks, max_slots; /* sized from the JIT region (machine.c) */
    int flush_pending;   /* set by helpers; the dispatcher flushes once the block has left */

    void *gic;
    /* Devices: called between blocks every so often and while the CPU waits.
       Returns the host time (mach_absolute_time units) of the next timer event, or 0. */
    uint64_t (*tick)(struct HvMachine *m, HvCpu *cpu);
    void *tick_opaque;
    /* GIC CPU interface system registers (ICC_*_EL1), reg = op0:op1:CRn:CRm:op2. */
    uint64_t (*icc_read)(struct HvMachine *m, HvCpu *cpu, uint32_t reg);
    void (*icc_write)(struct HvMachine *m, HvCpu *cpu, uint32_t reg, uint64_t v);

    uint64_t stop_pc;    /* tests: leave the dispatcher when pc reaches this */
    void (*stop_hook)(HvCpu *cpu); /* debugging: call this there instead of leaving */
    uint64_t stop_pc2;   /* debugging: a second such place */
    uint32_t jit_gen;    /* full flushes so far, as blocks see them (HV_JIT_POISON checks) */
    int gen_checks;
    int trace;           /* print every block entered */
    int log;             /* 1: unknown sysregs, undefined instructions, unmapped accesses; 2: also every exception */
    int slow_mem;        /* tests: send every load and store through hv_mem_emulate */
    uint64_t switch_flush_pc[4]; /* HV_SWITCH_FLUSH=3: PCs of the TLBI VMALLE1s seen within 20 us of a TTBR0 change (xnu's pmap switch) */
    uint64_t stat_slow_mem; /* hv_mem_emulate calls */
    uint64_t access_pc;  /* guest PC of the load/store being emulated, for logs */
    uint64_t watch_va;   /* debugging: log every write that touches this guest VA (0: off) */
    uint64_t watch_pa;   /* debugging: the same by physical address */
    HvStats stats;
    int chain;               /* block chaining and the indirect-branch cache are enabled */
    uint64_t *slots;         /* chainable exits' targets (0: not chained) */
    uint32_t **slot_site;    /* the B in each exit's stub that chaining rewrites to jump to the target */
    uint32_t **slot_stub;    /* what that B goes to while it isn't chained */
    uint32_t **slot_cond;    /* a conditional branch in a block's hot part that chaining may send straight to the target too */
    /* Exits to another page (see put_branch_exit): the host addend the target
       page's TLB entry must hold for the exit to go straight to the chained
       block (0: a same-page exit), and once chained, the target's physical page
       (| 1) so that invalidating its code can unchain it. xchained lists them. */
    uint64_t *slot_xa, *slot_xpa;
    uint32_t *xchained, nxchained;
    uint32_t nslots;
    HvCpu *cpu;              /* CPU 0: the one that polls devices */
    HvCpu *cpus[HV_MAX_CPUS]; /* every CPU, so device and invalidation code can ask them to leave */
    int ncpus;
    HvPerCpu percpu[HV_MAX_CPUS];
    /* With more than one CPU, only the holder of this runs guest code (a fair
       ticket lock, handed over at every kick and while a CPU sleeps). */
    int big_lock_on;
    /* Devices, the GIC and the ICC registers: one CPU at a time (recursive). */
    pthread_mutex_t dev_mu;
    /* Translating and inserting blocks, the JIT region and the slots. */
    pthread_mutex_t tr_mu;
    /* Stopping every CPU to flush all translations. */
    pthread_mutex_t stw_mu;
    pthread_cond_t stw_cv;
    int stw_active;
    pthread_mutex_t big_mu;
    pthread_cond_t big_cv;
    uint64_t big_next, big_serving;
    int stats_interval;  /* seconds between --stats lines, 0 = off */
    /* --profile: host PCs of the CPU thread sampled every millisecond, drained by the stats line. */
    uint64_t *prof;
    uint32_t prof_head, prof_tail;
} HvMachine;

static inline uint32_t hv_block_hash(uint64_t va, uint64_t pa, uint32_t flags) {
    uint64_t h = (va >> 2) ^ (pa >> 2) * 0x9e3779b97f4a7c15ull ^ flags;
    return (uint32_t)(h ^ (h >> 29)) & (HV_BLOCK_HASH - 1);
}

/* Lock-free: blocks are only added (release store) until a flush, which stops every CPU. */
static inline HvBlock *hv_block_find(HvMachine *m, uint32_t h, uint64_t va, uint64_t pa, uint32_t flags, uint32_t gen) {
    for (HvBlock *b = __atomic_load_n(&m->hash[h], __ATOMIC_ACQUIRE); b; b = __atomic_load_n(&b->next, __ATOMIC_ACQUIRE))
        if (b->va == va && b->pa == pa && b->flags == flags && b->gen == gen) return b;
    return NULL;
}

int hv_machine_init(HvMachine *m, size_t jit_bytes);
/* The same with JIT memory the host provides (iOS: the region a debugger granted and its writable alias). */
int hv_machine_init_jit(HvMachine *m, void *rx, void *rw, size_t jit_bytes);
void hv_machine_free(HvMachine *m);
int hv_machine_add_ram(HvMachine *m, uint64_t pa, uint64_t size, void *host, int readonly);
int hv_machine_add_mmio(HvMachine *m, uint64_t pa, uint64_t size, void *opaque,
                        uint64_t (*read)(void *, uint64_t, unsigned),
                        void (*write)(void *, uint64_t, uint64_t, unsigned), const char *name);

HvRam *hv_ram_lookup(HvMachine *m, uint64_t pa);
/* Drop the translations made from the RAM page holding pa (it was written, or the guest invalidated it). */
void hv_invalidate_code(HvMachine *m, uint64_t pa);
void hv_ret_clear(HvCpu *cpu);
int hv_ic_ivau_drops(void);
/* Settings that say how to translate or wait, to compare ways of doing it in one running guest
   ("knob NAME VALUE" in HV_INPUT_FILE; a new way starts from an empty code cache). Each starts
   from the environment variable named in hv_knob_names. */
enum { HV_K_NO_RAS, HV_K_SUPERBLOCKS, HV_K_GROUPS, HV_K_WFI_SPIN, HV_K_FLUSH, HV_K_TEST1, HV_K_TEST2, HV_K_TEST3, HV_K_TLBT, HV_K_JITCAP, HV_K_RECHECK, HV_K_SYNCWAIT, HV_K_SWITCHFLUSH, HV_K_RANGEDROP, HV_K_MOVIMM, HV_K_PREDSTAT, HV_K_WALKLOG, HV_K_HOIST, HV_K_IJCPAGES, HV_K_DNSPROXY, HV_K_COUNT };
extern int hv_knob[HV_K_COUNT];
int hv_knob_set(HvMachine *m, const char *name, int value);
/* Debugging (HV_WATCH): which pages keep losing their translations. */
extern uint32_t *hv_smc_counts;
extern uint64_t *hv_smc_last;
extern uint64_t hv_smc_pages, hv_smc_cpu, hv_smc_dma;
/* A device wrote guest memory directly: translations made from it are out of date. Any thread. */
void hv_dma_written(HvMachine *m, uint64_t pa, uint64_t len);
HvMmio *hv_mmio_lookup(HvMachine *m, uint64_t pa);
/* Host pointer for `len` bytes of guest RAM at pa, or NULL. */
void *hv_phys_ptr(HvMachine *m, uint64_t pa, uint64_t len);

/* Physical accesses from C (page-table walks, devices doing DMA, slow paths). */
int hv_phys_read(HvMachine *m, uint64_t pa, void *buf, unsigned len);
int hv_phys_write(HvMachine *m, uint64_t pa, const void *buf, unsigned len);

void hv_cpu_reset(HvCpu *cpu, HvMachine *m); /* CPU 0 */
void hv_cpu_reset_index(HvCpu *cpu, HvMachine *m, uint32_t index);
/* The CPU this thread runs (NULL outside the dispatcher: tests, devices). */
extern __thread HvCpu *hv_self;
void hv_dev_lock(HvMachine *m);
void hv_dev_unlock(HvMachine *m);
/* Ask another CPU to apply a TLBI (or HV_REQ_* work); returns the sequence to wait for. */
uint64_t hv_cpu_post_tlbi(HvCpu *target, uint32_t op1, uint32_t crn, uint32_t crm, uint32_t op2, uint64_t xt);
uint64_t hv_cpu_post(HvCpu *target, uint32_t flags);
uint64_t hv_cpu_post_ijc_page(HvCpu *target, uint64_t v);
/* On cpu's own thread: code changed in a page cached as branch targets at VA page v (| 2), or v = 1 (none):
   forget the return predictions and those targets (mmu.c). */
void hv_ijc_drop_code_page(HvCpu *cpu, uint64_t v);
/* Wait until target has done everything up to seq (or is parked, and will before it runs). */
void hv_cpu_sync(HvCpu *self, HvCpu *target, uint64_t seq);
/* The owner: do what other CPUs asked. */
void hv_cpu_process_requests(HvCpu *cpu);
void hv_cpu_ack_wake(HvCpu *cpu);
void hv_big_lock(HvMachine *m);
void hv_big_unlock(HvMachine *m);
/* Let a waiting CPU run (the lock is held again on return). */
void hv_big_yield(HvMachine *m);
/* Make every CPU leave its block at the next exit. */
void hv_machine_kick_all(HvMachine *m);
/* Wake one CPU out of WFI (and make it leave its block). */
void hv_cpu_wake(HvCpu *cpu);
/* Make it leave its block at the next checked exit, without waking it from WFI. */
void hv_cpu_kick(HvCpu *cpu);

/* From any thread: device work is waiting for the CPU thread's tick. Makes a
   running block leave and ends a WFI sleep. */
void hv_machine_wake(HvMachine *m);

#endif
