#include "translate.h"

#include <string.h>
#include <stdlib.h>

#include "classify.h"
#include "memop.h"
#include "mmu.h"
#include "operands.h"

/* A block has two parts. The hot part is the code that runs every time: the translated
   instructions. The cold part is what hardly ever does: exits not chained yet, memory slow
   paths, branch-cache misses, and the literal pool. In a region of 1 MB (the reach of the
   short branches between them) hot parts grow from the bottom and cold parts from the top,
   so code that runs sits together. What runs is a small part of what a translation holds,
   and macOS runs so much code, so thinly, that how much fits in the CPU's caches and branch
   tables is a large part of how fast it goes. HV_K2=1: one part, in the order written.

   Each part is built in a static buffer and copied into the JIT region once. Branches between
   them are put right when it is known where they went (rel, hv_translate_place). */

#define MAX_WORDS 16384
#define MAX_LITS 2048
#define MAX_OOL 1024
#define WORDS_PER_INSN_MAX 96 /* stop the block before the buffer could overflow */
#define MAX_RELS 8192
#define JIT_CHUNK_WORDS (256u * 1024) /* 1 MB: CBZ, B.cond and LDR (literal) reach both ways from anywhere in it */

enum { FIX_LDR, FIX_ADR };

typedef struct {
    uint32_t at, kind, idx;
    uint8_t cold; /* the instruction is in the cold part */
} LitFix;

enum { OOL_EXIT, OOL_SLOW, OOL_SYS, OOL_MRS, OOL_GROUP };

typedef struct {
    uint32_t at;        /* the branch that jumps here */
    uint32_t join;      /* slow paths come back here */
    uint64_t pc;        /* exit target, or the guest PC of the slow-path instruction */
    uint32_t insn;
    uint8_t kind;
    uint8_t nrestore;
    uint8_t restore[4]; /* scratch registers to reload from cpu->spill first */
    /* Loads and stores: redo the TLB check exactly (the fast one rejects an
       access in a page's last HV_TLB_ADJ bytes) and go back if it passes. */
    uint8_t recheck, addr, t, u, bytes;
    uint32_t tag_off;
    uint64_t site;      /* guest PC of the instruction that branches out of line */
    uint32_t retry;     /* loads and stores: where their translation starts, to run again after a refill */
    uint32_t rflags;    /* hv_tlb_refill's flags for it (0: no refill, straight to the helper) */
    uint8_t plain;      /* exits: an ordinary branch (see put_branch_exit); 2: one whose exit_req check is done already */
} Ool;

/* Loads and stores on one base register, close together in a block: the first
   checks the TLB once for the whole range they touch and leaves the page's
   host addend in cpu->gaddend[slot]; the rest use it with no check of their
   own. If that check fails, the block runs a plain translation of the
   instructions from the first through `end` instead (copy_at) and comes back
   after them. See plan_groups. */
#define MAX_GROUPS 8
#define GROUP_SLOTS 4
#define GROUP_SPAN 4096   /* bytes one group may cover */
#define GROUP_INSNS 48    /* and instructions */
typedef struct {
    int32_t lo, hi;       /* bytes touched, relative to the base at the first access */
    uint8_t store, base, slot, live;
    uint16_t first, last, end;
    uint32_t copy_at;
} Group;

typedef struct {
    uint32_t hot[MAX_WORDS], cold[MAX_WORDS];
    uint32_t *w;        /* the part being written */
    uint32_t n;         /* its words so far */
    uint32_t other_n;   /* the other part's */
    int in_cold;        /* writing what belongs in the cold part */
    int flat;           /* HV_K2: all in one, in the order written */
    struct { uint32_t at, target; uint8_t from_cold; } rel[MAX_RELS]; /* a branch in one part to a word in the other */
    uint32_t nrel;
    uint32_t nhot, ncold; /* the finished block's parts in words, ncold with the pool */
    uint64_t pool[MAX_LITS];
    uint32_t npool;
    LitFix fix[MAX_LITS];
    uint32_t nfix;
    Ool ool[MAX_OOL];
    uint32_t nool;
    struct { uint32_t slot, at, flags, stub, cond_at; uint64_t target; uint8_t cold, stub_cold, has_cond; } site[MAX_OOL]; /* chainable exits: their slot, the B chaining patches, its stub, where to; the hot part's conditional branch to it if chaining may patch that too */
    int site_direct;    /* the exit just written needs nothing done before its B: no check, no restores */
    uint32_t nsite;
    uint32_t pool_at;    /* where the literal pool is, in the cold part */
    uint32_t flags;
    int slow_mem;
    uint64_t page;       /* guest VA of the 16 KB page the block is in */
    int no_ijc;          /* indirect branches always go to the dispatcher (breakpoints, tests' stop PC) */
    HvMachine *m;
    uint64_t cur_pc;     /* guest PC of the instruction being translated (or whose out-of-line exit is) */
    /* Grouped accesses (Group). */
    Group grp[MAX_GROUPS];
    uint32_t ngrp;
    int8_t gid[HV_MAX_BLOCK_INSNS];     /* the group an instruction is in, or -1 */
    uint8_t grole[HV_MAX_BLOCK_INSNS];  /* 1: its first access, 2: a later one */
    int32_t goff[HV_MAX_BLOCK_INSNS];   /* the access's address minus the base at the group's first */
    uint32_t after[HV_MAX_BLOCK_INSNS]; /* where each instruction's translation ends */
    uint32_t cur_i;
    int nogroup;                        /* translating plain (copies, or the retry when they do not fit) */
    int u_live, u_prev;                 /* slot + 1 when x15 holds that group's addend after the last instruction */
    int overflow;
} Emit;

static __thread Emit E; /* each CPU translates on its own thread, outside the translation lock */

/* ---- encodings ---- */

#define STR_X(t, n, off) (0xf9000000u | ((uint32_t)(off) / 8) << 10 | (n) << 5 | (t))
#define LDR_X(t, n, off) (0xf9400000u | ((uint32_t)(off) / 8) << 10 | (n) << 5 | (t))
#define STP_X(t1, t2, n, off) (0xa9000000u | ((((uint32_t)(off) / 8) & 0x7f) << 15) | (t2) << 10 | (n) << 5 | (t1))
#define LDP_X(t1, t2, n, off) (0xa9400000u | ((((uint32_t)(off) / 8) & 0x7f) << 15) | (t2) << 10 | (n) << 5 | (t1))
#define BR_X(n) (0xd61f0000u | (n) << 5)
#define BLR_X(n) (0xd63f0000u | (n) << 5)
#define SBFM_X(d, n, immr, imms) (0x93400000u | (immr) << 16 | (imms) << 10 | (n) << 5 | (d))
#define UBFM_X(d, n, immr, imms) (0xd3400000u | (immr) << 16 | (imms) << 10 | (n) << 5 | (d))
#define BFM_X(d, n, immr, imms) (0xb3400000u | (immr) << 16 | (imms) << 10 | (n) << 5 | (d))
#define B_OP 0x14000000u
#define MOV_X(d, m) (0xaa0003e0u | (m) << 16 | (d))
#define ADD_SP_TO(d) (0x910003e0u | (d)) /* add xd, sp, #0: what mov from SP is (an ORR would read XZR) */
#define ADD_LSL(d, n, m, sh) (0x8b000000u | (m) << 16 | (sh) << 10 | (n) << 5 | (d))
#define SUB_X(d, n, m) (0xcb000000u | (m) << 16 | (n) << 5 | (d))
#define ADD_EXT(d, n, m, opt, sh) (0x8b200000u | (m) << 16 | (opt) << 13 | (sh) << 10 | (n) << 5 | (d))

static void put(Emit *e, uint32_t w);

/* T = where the TLB entry for the page of the address in register va is, for table `table` (0 EL0, 1 EL1, 2 EL1 with PAN),
   in the layout in force (hv_tlb_tmajor, cpu.h). Returns the offset from T of the entry's first field. */
static uint32_t put_tlb_slot(Emit *e, int T, int va, int table) {
    put(e, UBFM_X(T, va, HV_TLB_PAGE_BITS, HV_TLB_PAGE_BITS + HV_TLB_BITS - 1));
    if (hv_tlb_tmajor) {
        put(e, ADD_LSL(T, 28, T, 5));
        if (table) put(e, 0x91400000u | (uint32_t)table * (HV_TLB_SIZE * 32 / 4096) << 10 | (uint32_t)T << 5 | (uint32_t)T); /* add T, T, #table's offset */
        return HV_CPU_TLB;
    }
    put(e, ADD_LSL(T, 28, T, 7));
    return HV_CPU_TLB + (uint32_t)table * HV_CPU_TLB_EL;
}
#define CBNZ_X(t) (0xb5000000u | (t))
#define CBZ_X(t) (0xb4000000u | (t))

static void patch_branch(uint32_t *w, int64_t off);

static inline int64_t sext(uint64_t v, int bits) {
    return (int64_t)(v << (64 - bits)) >> (64 - bits);
}

static void put(Emit *e, uint32_t w) {
    if (e->n < MAX_WORDS) e->w[e->n] = w;
    e->n++;
}

/* Write the cold part for a while. */
static void cold_on(Emit *e) {
    e->in_cold = 1;
    if (e->flat) return;
    uint32_t n = e->n;
    e->n = e->other_n, e->other_n = n;
    e->w = e->cold;
}

static void cold_off(Emit *e) {
    e->in_cold = 0;
    if (e->flat) return;
    uint32_t n = e->n;
    e->n = e->other_n, e->other_n = n;
    e->w = e->hot;
}

/* The branch at `at` in one part goes to `target` in that part or the other. */
static void plink(Emit *e, uint32_t at, int from_cold, uint32_t target, int to_cold) {
    if (e->flat || from_cold == to_cold) {
        uint32_t *w = e->flat || !from_cold ? e->hot : e->cold;
        patch_branch(&w[at], (int64_t)target - at);
        return;
    }
    if (e->nrel < MAX_RELS) {
        e->rel[e->nrel].at = at, e->rel[e->nrel].target = target, e->rel[e->nrel].from_cold = (uint8_t)from_cold;
        e->nrel++;
    } else {
        e->overflow = 1;
    }
}

/* To the next word to be written. */
static void link_here(Emit *e, uint32_t at, int from_cold) {
    plink(e, at, from_cold, e->n, e->in_cold);
}

/* xR = value with MOVZ or MOVN and then MOVKs, if that takes at most max instructions. */
static int put_mov_imm(Emit *e, int reg, uint64_t value, int max) {
    int zeros = 0, ones = 0;
    for (int i = 0; i < 4; i++) {
        uint16_t h = (uint16_t)(value >> 16 * i);
        zeros += h == 0, ones += h == 0xffff;
    }
    int inverted = ones > zeros; /* kernel addresses: start from all ones */
    int n = 4 - (inverted ? ones : zeros);
    if ((n ? n : 1) > max) return 0;
    uint16_t skip = inverted ? 0xffff : 0;
    int first = 1;
    for (uint32_t i = 0; i < 4; i++) {
        uint16_t h = (uint16_t)(value >> 16 * i);
        if (h == skip) continue;
        if (first) put(e, (inverted ? 0x92800000u : 0xd2800000u) | i << 21 | (uint32_t)(inverted ? (uint16_t)~h : h) << 5 | (uint32_t)reg);
        else put(e, 0xf2800000u | i << 21 | (uint32_t)h << 5 | (uint32_t)reg); /* movk */
        first = 0;
    }
    if (first) put(e, (inverted ? 0x92800000u : 0xd2800000u) | (uint32_t)reg); /* 0 or all ones */
    return 1;
}

/* ldr xR, =value. The pool sits after the cold code, so a hot block's literal is a load from a line the data
   cache rarely holds (5% of translated-code samples on the Mac): values that take at most three MOVZ/MOVN/MOVK
   (nearly all addresses, user and kernel) can be built in the register instead, in as many words: HV_MOVIMM=1
   (off: no measured gain on the Mac mix, 2026-11-08). */
static void put_lit(Emit *e, int reg, uint64_t value) {
    if (hv_knob[HV_K_MOVIMM] && put_mov_imm(e, reg, value, 3)) return;
    e->pool[e->npool] = value;
    e->fix[e->nfix++] = (LitFix){e->n, FIX_LDR, e->npool++, (uint8_t)(e->in_cold && !e->flat)};
    put(e, 0x58000000u | reg);
}

/* adr xR, <16-byte HvHelperInfo in the pool> */

static void put_ool(Emit *e, Ool o) {
    o.at = e->n - 1; /* the branch just emitted */
    o.site = e->cur_pc;
    e->ool[e->nool++] = o;
}

/* Guest x0/x1 are saved, x0 = target = cpu->pc. Jumps to the target block
   through the indirect-branch cache for this block's flags if it is there and
   nobody asked the CPU to leave, else leaves for the dispatcher (which fills
   the cache). A hit reloads guest x0 and x1, which the hash used as scratch,
   and enters the block directly: no block has a prologue. Tags are VA ^ salt;
   the salt changes when mappings or code do. */
/* Jump to the guest VA in host x15 through the indirect-branch cache for this
   block's flags. Guest x0/x1 stay in their registers on the hit path, and
   nothing is stored: the target block just runs. A miss (or exit_req) saves
   x0/x1 and pc and leaves for the dispatcher, which fills the cache. */
static void put_ijc_jump(Emit *e, int check) {
    if (e->no_ijc) {
        put(e, STP_X(0, 1, 28, 0));
        put(e, MOV_X(0, 15));
        put(e, STR_X(0, 28, HV_CPU_PC));
        put(e, LDR_X(1, 28, HV_CPU_EXIT));
        put(e, BR_X(1));
        return;
    }
    int p0 = e->in_cold;
    uint32_t to_exit = 0;
    if (check) {
        put(e, 0xb9400000u | (HV_CPU_EXIT_REQ / 4) << 10 | 28 << 5 | 6); /* ldr w6, [x28, #exit_req] */
        to_exit = e->n;
        put(e, 0x35000000u | 6);                            /* cbnz w6, exit (x15 still the VA) */
    }
    /* The tag, VA ^ salt, also picks the slot (hv_ijc_index), so each address
       space's copy of shared library code gets its own slots. */
    uint32_t salt = HV_CPU_IJC_SALT + 8 * (e->flags & HV_BLOCK_EL1);
    put(e, LDR_X(6, 28, salt));
    put(e, 0xca000000u | 6 << 16 | 15 << 5 | 15);           /* eor x15, x15, x6: the tag */
    put(e, 0xca400000u | 15 << 16 | 16 << 10 | 15 << 5 | 7); /* eor x7, x15, x15, lsr #16 */
    put(e, 0xca400000u | 15 << 16 | 46 << 10 | 7 << 5 | 7);  /* eor x7, x7, x15, lsr #46 */
    put(e, UBFM_X(7, 7, 2, 2 + HV_IJC_BITS - 2));           /* ubfx x7, x7, #2, #bits - 1: the set */
    put(e, LDR_X(6, 28, HV_CPU_IJC_TAB + 8 * (e->flags & (HV_IJC_TABLES - 1))));
    put(e, ADD_LSL(6, 6, 7, 5));
    put(e, LDR_X(7, 6, 0));                                 /* way 0's tag */
    put(e, 0xca000000u | 15 << 16 | 7 << 5 | 7);            /* eor x7, x7, x15 */
    uint32_t to_way1 = e->n;
    put(e, CBNZ_X(7));
    put(e, LDR_X(6, 6, 8));
    put(e, BR_X(6));                                        /* hit: x0, x1 are the guest's */
    if (!p0) cold_on(e);
    link_here(e, to_way1, p0);
    put(e, LDR_X(7, 6, 16));                                /* way 1's */
    put(e, 0xca000000u | 15 << 16 | 7 << 5 | 7);
    uint32_t to_miss = e->n;
    put(e, CBNZ_X(7));
    put(e, LDR_X(6, 6, 24));
    put(e, BR_X(6));
    patch_branch(&e->w[to_miss], (int64_t)e->n - to_miss);  /* miss: x15 is the tag, so the VA is x15 ^ salt */
    put(e, LDR_X(7, 28, salt));
    put(e, 0xca000000u | 15 << 16 | 7 << 5 | 7);            /* eor x7, x7, x15: the VA */
    put(e, STR_X(30, 28, HV_CPU_X(30)));
    put(e, LDR_X(15, 28, HV_CPU_IJC_THUNK));
    put(e, BLR_X(15));                                      /* the block exists? then the thunk goes there */
    put(e, e->flags);
    put(e, STP_X(0, 1, 28, 0));
    put(e, MOV_X(0, 7));
    if (check) {
        uint32_t to_common = e->n;
        put(e, B_OP);
        link_here(e, to_exit, p0);                          /* exit_req: x15 is the VA */
        put(e, STP_X(0, 1, 28, 0));
        put(e, MOV_X(0, 15));
        patch_branch(&e->w[to_common], (int64_t)e->n - to_common);
    }
    put(e, STR_X(0, 28, HV_CPU_PC));
    put_lit(e, 1, e->cur_pc);                               /* say where, for --stats */
    put(e, STR_X(1, 28, HV_CPU_MISS_SITE));
    put(e, LDR_X(1, 28, HV_CPU_EXIT));
    put(e, BR_X(1));
    if (!p0) cold_off(e);
}

/* A direct branch to another page through hv_xjump_thunk (the cache lookup in
   one shared copy): a BR, so the CPU's own stack of return addresses gets nothing it
   would use on the guest's next return. check: test exit_req first. */
static void put_xjump(Emit *e, uint64_t target, int check) {
    if (e->no_ijc) {
        put_lit(e, 15, target);
        put_ijc_jump(e, 1);
        return;
    }
    if (!(e->n & 1)) put(e, 0xd503201f); /* nop: the data after the br starts 8-byte aligned */
    put(e, 0x10000007u | 3u << 5);       /* adr x7, data */
    put(e, LDR_X(15, 28, HV_CPU_XJUMP_THUNK));
    put(e, BR_X(15));
    put(e, (e->flags & 0x7fffffffu) | (check ? 1u << 31 : 0));
    put(e, 0);
    put(e, (uint32_t)target);
    put(e, (uint32_t)(target >> 32));
}

/* Leave the block with pc = target. */
static void put_exit(Emit *e, uint64_t target) {
    put(e, STP_X(0, 1, 28, 0));
    put_lit(e, 0, target);
    put(e, STR_X(0, 28, HV_CPU_PC));
    put(e, LDR_X(1, 28, HV_CPU_EXIT));
    put(e, BR_X(1));
}

/* A direct branch to target. Within the block's own 16 KB page the exit is
   chainable: it jumps to the block whose entry the dispatcher stored in this
   exit's slot (plain data, so chaining never rewrites code), as long as nobody
   has asked the CPU to leave. The hit path touches only the free scratch
   registers, so guest x0/x1 stay live and the target needs no prologue. An
   empty slot leaves with its address in last_stub so the dispatcher can fill
   it. A same-page target always has the same physical mapping as this block,
   so the chain stays correct across page-table changes. Other pages go through
   the indirect-branch cache. */
/* plain: an ordinary branch, not an exit that synchronises (ISB, a system
   register write, an interrupt let through). One going forward (to a higher
   address than the branch) skips the exit_req check: blocks only run forward
   that way, so any loop of chained blocks still has a checked edge, and a CPU
   asked to leave reaches one soon. */
static int check_exits_always(void) {
    static int v = -1; /* HV_EXIT_CHECKS: check on every exit (debugging) */
    if (v < 0) v = getenv("HV_EXIT_CHECKS") != NULL;
    return v;
}

/* A direct branch to another page can chain too, if a check that the target
   page is still mapped where it was comes first: its TLB entry (the slot is
   known now, as the VA is) must allow fetching and hold the host addend the
   page had when this block was translated. The block the dispatcher chains
   it to must be in that physical page (chain() in dispatch.c). The addend is
   looked up now, in the translating CPU's page tables; 0 if it cannot be. */
static uint64_t cross_addend(Emit *e, uint64_t target) {
    static int off = -1; /* HV_NO_XCHAIN: exits to other pages always look the target up */
    if (off < 0) off = getenv("HV_NO_XCHAIN") != NULL;
    HvCpu *cpu = hv_self;
    if (off || !cpu || e->no_ijc || !e->m->chain) return 0;
    /* This CPU's TLB first: an entry there is the mapping now. */
    int table = !(e->flags & HV_BLOCK_EL1) ? 0 : (e->flags & HV_BLOCK_PAN) ? 2 : 1;
    const HvTlbEntry *te = hv_tlb_ent(cpu, (target >> HV_TLB_PAGE_BITS) & (HV_TLB_SIZE - 1), table);
    uint64_t tpage = target & ~((1ull << HV_TLB_PAGE_BITS) - 1);
    if (te->exec_tag == hv_exec_key(tpage, te->addend)) return (uint64_t)te->addend;
    HvWalk w;
    hv_mmu_walk(cpu, target, HV_ACC_EXEC, (e->flags & HV_BLOCK_EL1) ? 1 : 0, &w);
    if (w.fsc || w.block_size < (1ull << HV_TLB_PAGE_BITS)) return 0;
    uint64_t page_pa = w.pa & ~((1ull << HV_TLB_PAGE_BITS) - 1);
    HvRam *ram = hv_ram_lookup(e->m, page_pa);
    if (!ram || page_pa + (1ull << HV_TLB_PAGE_BITS) - ram->pa > ram->size) return 0;
    return (uint64_t)(uintptr_t)(ram->host + (page_pa - ram->pa)) - (target & ~((1ull << HV_TLB_PAGE_BITS) - 1));
}

static int use_xjump(void) {
    static int v = -1; /* HV_INLINE_XJUMP: each direct branch to another page has its own cache lookup */
    if (v < 0) v = getenv("HV_INLINE_XJUMP") == NULL;
    return v;
}

static void put_branch_exit(Emit *e, uint64_t target, int plain) {
    int check = !(plain && (target > e->cur_pc || plain == 2) && !check_exits_always());
    int p0 = e->in_cold;
    e->site_direct = 0;
    uint32_t slot_index = ~0u;
    int same_page = (target & ~((1ull << HV_TLB_PAGE_BITS) - 1)) == e->page;
    uint64_t xa = same_page ? 0 : cross_addend(e, target);
    if (same_page || xa) {
        slot_index = __atomic_fetch_add(&e->m->nslots, 1, __ATOMIC_RELAXED); /* translations run in parallel */
        if (slot_index >= e->m->max_slots) slot_index = ~0u; /* out of slots until the next flush: no chaining */
    }
    if (slot_index == ~0u) {
        /* Another page may be remapped under us, so look it up by VA instead of chaining. */
        if (!same_page && use_xjump()) put_xjump(e, target, check);
        else {
            put_lit(e, 15, target);
            put_ijc_jump(e, check);
        }
        return;
    }
    uint64_t *slot = &e->m->slots[slot_index];
    *slot = 0;
    e->m->slot_site[slot_index] = NULL; /* until placed: what it held is a dropped block's */
    e->m->slot_cond[slot_index] = NULL;
    e->m->slot_xa[slot_index] = xa;
    e->m->slot_xpa[slot_index] = 0;
    uint32_t to_exit = ~0u;
    if (check) {
        put(e, 0xb9400000u | (HV_CPU_EXIT_REQ / 4) << 10 | 28 << 5 | 15); /* ldr w15, [x28, #exit_req] */
        to_exit = e->n;
        put(e, 0x35000000u | 15);                                         /* cbnz w15, slow (patched below) */
    }
    uint32_t to_miss = ~0u;
    if (xa) {
        /* The target page's fetch tag: its VA and addend, against what they were. */
        int table = !(e->flags & HV_BLOCK_EL1) ? 0 : (e->flags & HV_BLOCK_PAN) ? 2 : 1;
        uint32_t off = HV_CPU_TLB + hv_tlb_code((uint32_t)(target >> HV_TLB_PAGE_BITS) & (HV_TLB_SIZE - 1), table) * (uint32_t)sizeof(HvTlbEntry) + 16;
        put(e, 0x91400000u | (off >> 12) << 10 | 28 << 5 | 7);  /* add x7, x28, #off & ~0xfff */
        put(e, LDR_X(7, 7, off & 0xfff));
        put_lit(e, 15, hv_exec_key(target & ~((1ull << HV_TLB_PAGE_BITS) - 1), (int64_t)xa));
        put(e, 0xca000000u | 15 << 16 | 7 << 5 | 7);            /* eor x7, x7, x15 */
        to_miss = e->n;
        put(e, CBNZ_X(7));                                      /* not mapped so now: the branch cache */
    }
    /* Chaining rewrites this B to go straight to the target block (one aligned
       B may be rewritten while other CPUs run it); until then, or when the
       target is out of B's range, it goes to the stub, into hv_chain_thunk. */
    uint32_t site_at = e->n;
    put(e, B_OP | 1);
    if (!p0) cold_on(e);
    link_here(e, site_at, p0);
    if (e->nsite < MAX_OOL) {
        e->site[e->nsite].slot = slot_index;
        e->site[e->nsite].flags = e->flags;
        e->site[e->nsite].target = target;
        e->site[e->nsite].at = site_at;
        e->site[e->nsite].cold = (uint8_t)(p0 && !e->flat);
        e->site[e->nsite].stub = e->n;
        e->site[e->nsite].has_cond = 0;
        e->site[e->nsite++].stub_cold = (uint8_t)(e->in_cold && !e->flat);
        e->site_direct = !check && !xa;
    }
    if (check) link_here(e, to_exit, p0);
    put(e, STR_X(30, 28, HV_CPU_X(30)));
    put(e, LDR_X(15, 28, HV_CPU_CHAIN_THUNK));
    if ((e->n & 1) == 0) put(e, 0xd503201f);                         /* nop: the data starts 8-byte aligned */
    put(e, BLR_X(15));
    put(e, (uint32_t)(uintptr_t)slot);
    put(e, (uint32_t)((uintptr_t)slot >> 32));
    put(e, (uint32_t)target);
    put(e, (uint32_t)(target >> 32));
    if (xa) {
        link_here(e, to_miss, p0);
        if (use_xjump()) put_xjump(e, target, 0); /* exit_req was checked above if it had to be */
        else {
            put_lit(e, 15, target);
            put_ijc_jump(e, 0);
        }
    }
    if (!p0) cold_off(e);
}

/* Call hv_helper(cpu, info) and continue unless it leaves. */
/* Call hv_helper: guest x30 saved, a blr to hv_helper_thunk with x6 = its HvHelperInfo,
   and x0 reloaded when it comes back. Host x6 and x15 are clobbered. */
static void put_helper(Emit *e, uint64_t pc, uint32_t insn, uint32_t op);

/* A memory slow path whose thunk may refill the TLB and resume at `retry`:
   the op word carries the refill flags (bits 18..8) and how far back retry is
   from where the helper would resume (bits 31..19, in words). */
static void put_helper_refill(Emit *e, uint64_t pc, uint32_t insn, uint32_t rflags, uint32_t retry) {
    put_helper(e, pc, insn, HV_OP_MEM);
    uint32_t resume = e->n - 1; /* the ldr x0, [x28] the helper comes back to */
    uint32_t back = resume - retry;
    if (back < (1u << 13) && rflags < (1u << 11)) e->pool[e->npool - 1] = (uint64_t)insn | (uint64_t)(HV_OP_MEM | rflags << 8 | back << 19) << 32;
}

/* The return address is where it goes back to, so the thunk's RET goes where the CPU expects
   and leaves its stack of return addresses as it was. The HvHelperInfo is in the pool. */
static void put_helper(Emit *e, uint64_t pc, uint32_t insn, uint32_t op) {
    uint32_t info = e->npool;
    e->pool[info] = pc;
    e->pool[info + 1] = (uint64_t)insn | (uint64_t)op << 32;
    e->npool += 2;
    put(e, STR_X(30, 28, HV_CPU_X(30)));
    put(e, LDR_X(15, 28, HV_CPU_HELPER_THUNK));
    e->fix[e->nfix++] = (LitFix){e->n, FIX_ADR, info, (uint8_t)(e->in_cold && !e->flat)};
    put(e, 0x10000006u); /* adr x6, info */
    put(e, BLR_X(15));
    put(e, LDR_X(0, 28, 0));
}

/* d = n + imm for |imm| < 2^24; n and d are general registers, never SP. */
static void put_add_imm(Emit *e, int d, int n, int64_t imm) {
    if (imm == 0) {
        if (d != n) put(e, MOV_X(d, n));
        return;
    }
    uint32_t op = imm < 0 ? 0xd1000000u : 0x91000000u;
    uint64_t a = imm < 0 ? (uint64_t)-imm : (uint64_t)imm;
    if (a < 4096) {
        put(e, op | (uint32_t)a << 10 | n << 5 | d);
        return;
    }
    put(e, op | 1u << 22 | (uint32_t)(a >> 12) << 10 | n << 5 | d);
    if (a & 0xfff) put(e, op | (uint32_t)(a & 0xfff) << 10 | d << 5 | d);
}

/* ---- scratch registers ---- */

/* Host registers that can be borrowed: any guest-allocated one, saved in cpu->spill. */
/* Host x6, x7 and x15 hold no guest register (guest x6, x7, x15 live in the
   HvCpu, like x18 and x28: they are the least used in compiled code), so they
   are free scratch: picked first and never spilled. */
static const uint8_t candidates[] = {6, 7, 15, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 8,
                                     16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30};

static int is_free(int reg) {
    return reg == 6 || reg == 7 || reg == 15;
}

static int pick(uint32_t *used) {
    for (unsigned i = 0; i < sizeof candidates; i++) {
        int r = candidates[i];
        if (!(*used & (1u << r))) {
            *used |= 1u << r;
            return r;
        }
    }
    return -1;
}

/* Free scratch registers need no saving; the rest go to cpu->spill in order. */
static int need_spill(const uint8_t *regs, int n, uint8_t *out) {
    int k = 0;
    for (int i = 0; i < n; i++)
        if (!is_free(regs[i])) out[k++] = regs[i];
    return k;
}

static void spill(Emit *e, const uint8_t *regs, int n) {
    uint8_t r[4];
    n = need_spill(regs, n, r);
    int i = 0;
    for (; i + 1 < n; i += 2) put(e, STP_X(r[i], r[i + 1], 28, HV_CPU_SPILL(i)));
    if (i < n) put(e, STR_X(r[i], 28, HV_CPU_SPILL(i)));
}

static void unspill(Emit *e, const uint8_t *regs, int n) {
    uint8_t r[4];
    n = need_spill(regs, n, r);
    int i = 0;
    for (; i + 1 < n; i += 2) put(e, LDP_X(r[i], r[i + 1], 28, HV_CPU_SPILL(i)));
    if (i < n) put(e, LDR_X(r[i], 28, HV_CPU_SPILL(i)));
}

/* Where a register the translator does not keep in a host register lives. */
static int is_stolen(int reg) {
    return reg == 18 || reg == 28 || is_free(reg);
}

static uint32_t slot_of(int reg) {
    return reg == 31 ? HV_CPU_SP : HV_CPU_X(reg);
}

/* ---- instruction groups ---- */

static void translate_copy(Emit *e, uint32_t insn) {
    uint32_t fields = hv_gpr_fields(insn);
    if (!fields) {
        put(e, insn);
        return;
    }
    int stolen[4] = {-1, -1, -1, -1}; /* per field: the guest register to rename, or -1 */
    uint32_t used = 0;
    int any = 0;
    for (int f = 0; f < 4; f++) {
        if (!(fields & (1u << f))) continue;
        int r = hv_field_reg(insn, f);
        used |= 1u << r;
        if (is_stolen(r)) { /* guest SP is the host SP while blocks run: copied as it is */
            stolen[f] = r;
            any = 1;
        }
    }
    if (!any) {
        put(e, insn);
        return;
    }
    /* One scratch per distinct stolen register; x6, x7 and x15 use their own host register. */
    int guest_of[4] = {-1, -1, -1, -1}, scratch_of[4];
    uint8_t regs[4];
    int n = 0;
    for (int f = 0; f < 4; f++) {
        if (stolen[f] < 0) continue;
        int k = 0;
        while (k < n && guest_of[k] != stolen[f]) k++;
        if (k == n) {
            guest_of[n] = stolen[f];
            scratch_of[n] = is_free(stolen[f]) ? stolen[f] : pick(&used);
            regs[n] = scratch_of[n];
            n++;
        }
    }
    spill(e, regs, n);
    /* All copied forms only write Rd but for MOVK and BFM (BFI, BFXIL), so what is in Rd alone
       needs no loading first. */
    int rd_read = (insn & 0x7f800000) == 0x72800000 || (insn & 0x7f800000) == 0x33000000 || hv_knob[HV_K_TEST3];
    for (int k = 0; k < n; k++) {
        int read = 0;
        for (int f = 0; f < 4; f++)
            if (stolen[f] == guest_of[k] && (f || rd_read)) read = 1;
        if (read) put(e, LDR_X(scratch_of[k], 28, slot_of(guest_of[k])));
    }
    uint32_t out = insn;
    for (int f = 0; f < 4; f++) {
        if (stolen[f] < 0) continue;
        int k = 0;
        while (guest_of[k] != stolen[f]) k++;
        out = hv_field_set(out, f, scratch_of[k]);
    }
    put(e, out);
    if (stolen[0] >= 0) { /* Rd was renamed: every copied form with an Rd field writes it */
        int k = 0;
        while (guest_of[k] != stolen[0]) k++;
        put(e, STR_X(scratch_of[k], 28, slot_of(stolen[0])));
    }
    unspill(e, regs, n);
}

/* Write a translation-time constant to guest register rd (31 is XZR here). */
static void put_const_to(Emit *e, int rd, uint64_t value) {
    if (rd == 31) return;
    if (!is_stolen(rd)) {
        put_lit(e, rd, value);
        return;
    }
    uint8_t s = 15; /* free scratch: the spill below is empty */
    spill(e, &s, 1);
    put_lit(e, s, value);
    put(e, STR_X(s, 28, slot_of(rd)));
    unspill(e, &s, 1);
}

static void translate_pcrel(Emit *e, uint32_t insn, uint64_t pc) {
    int64_t imm = sext(((insn >> 5) & 0x7ffff) << 2 | ((insn >> 29) & 3), 21);
    uint64_t value = (insn >> 31) ? (pc & ~0xfffull) + ((uint64_t)imm << 12) : pc + imm;
    put_const_to(e, insn & 0x1f, value);
}

/* Bits 54..48 take bit 55's value. Guest pointers are never signed, so for a
   canonical pointer this changes nothing; it only clears a stray PAC field. */
static void put_strip(Emit *e, int r) {
    if (r == 31) return;
    uint32_t used = 1u << r;
    if (!is_stolen(r)) {
        uint8_t t = pick(&used);
        spill(e, &t, 1);
        put(e, SBFM_X(t, r, 55, 55));
        put(e, BFM_X(r, t, 16, 6));
        unspill(e, &t, 1);
        return;
    }
    uint8_t regs[2];
    regs[0] = pick(&used);
    regs[1] = pick(&used);
    spill(e, regs, 2);
    put(e, LDR_X(regs[0], 28, slot_of(r)));
    put(e, SBFM_X(regs[1], regs[0], 55, 55));
    put(e, BFM_X(regs[0], regs[1], 16, 6));
    put(e, STR_X(regs[0], 28, slot_of(r)));
    unspill(e, regs, 2);
}

static void translate_pac(Emit *e, uint32_t insn) {
    if ((insn & 0xfffff01f) == 0xd503201f) { /* hint space */
        uint32_t imm = ((insn >> 8) & 0xf) << 3 | ((insn >> 5) & 7);
        if (imm == 7 || imm >= 28) put_strip(e, 30);       /* XPACLRI, AUTIASP and friends */
        else if (imm == 12 || imm == 14) put_strip(e, 17); /* AUTIA1716, AUTIB1716 */
        return;                                            /* PAC*: no-op */
    }
    int rd = insn & 0x1f;
    if ((insn & 0xffe0fc00) == 0x9ac03000) { /* PACGA: a fixed code keeps sign/verify pairs consistent */
        if (rd == 31) return;
        if (is_stolen(rd)) put(e, STR_X(31, 28, slot_of(rd)));
        else put(e, 0xd2800000u | rd); /* movz xd, #0 */
        return;
    }
    uint32_t opcode = (insn >> 10) & 0x3f;
    if (opcode >= 16 || (opcode & 4)) put_strip(e, rd); /* XPAC*, AUT* */
}

/* HV_SUPERBLOCKS=1: a conditional branch does not end its block; the block goes
   on with the instruction after it (the taken side leaves out of line). */
static int superblocks(void) {
    return hv_knob[HV_K_SUPERBLOCKS];
}

/* The block may go on past a conditional branch at pc: the next instruction is
   in the same page and the block is not long already. */
static int go_on(Emit *e, uint64_t pc) {
    return ((pc + 4) & ~((1ull << HV_TLB_PAGE_BITS) - 1)) == e->page && e->cur_i + 1 < 48;
}

/* Calls and returns the way the host's CPU can predict them. It keeps a stack of return
   addresses of its own, filled by its BL and used by its RET, and those are right nearly
   always, even in code run for the first time. A return reached through a register-indirect
   branch gets none of that, and a mispredicted branch costs some twenty instructions' time.
   So a guest call is a host BL, whose return address (H: the code for what follows the call)
   goes in a table by the guest's return address; a guest return that finds its own address
   there is a host RET to H. Otherwise the indirect-branch cache, as ever.

   A table and not a stack, so calls and returns don't wait for each other: a stack's top has
   to go through memory from each to the next. One table per block-flags value, so what H
   assumes about the CPU's state still holds; tags are VA ^ ret_salt, which changes whenever
   what is in the table may have gone out of date (see ret_new in mmu.c). Host x6, x7 and x15
   are free at a call and a return, and x30 is about to be set.

       bl   L
    H: (exit to the block after the call)
    L: add  x7, x28, #entry, lsl #12
       str  x30, [x7, #entry & 0xfff + 8]
       ldr  x30, =return address
       ldr  x6, [x28, #salt]
       eor  x6, x6, x30
       str  x6, [x7, #entry & 0xfff]                                    */
static int use_ras(const Emit *e) {
    return !hv_knob[HV_K_NO_RAS] && !e->no_ijc; /* HV_NO_RAS: every return through the indirect-branch cache */
}

static void put_call(Emit *e, uint64_t ret) {
    uint32_t off = HV_CPU_RET_TAB + (e->flags & (HV_IJC_TABLES - 1)) * HV_RET_TAB_BYTES + (((uint32_t)ret >> 2) & ((1u << HV_RET_BITS) - 1)) * 16;
    uint32_t to_l = e->n;
    put(e, 0x94000000u); /* bl L */
    put_branch_exit(e, ret, 1);
    patch_branch(&e->w[to_l], (int64_t)e->n - to_l);
    put(e, 0x91400000u | (off >> 12) << 10 | 28 << 5 | 7);                /* add x7, x28, #off, lsl #12 */
    put(e, STR_X(30, 7, (off & 0xfff) + 8));
    put_lit(e, 30, ret);
    put(e, LDR_X(6, 28, HV_CPU_RET_SALT + 8 * (e->flags & HV_BLOCK_EL1)));
    put(e, 0xca000000u | 30 << 16 | 6 << 5 | 6);                          /* eor x6, x6, x30 */
    put(e, STR_X(6, 7, off & 0xfff));
}

/* The other half: a return. A miss goes through the indirect-branch cache. */
static void put_return(Emit *e) {
    uint32_t tab = HV_CPU_RET_TAB + (e->flags & (HV_IJC_TABLES - 1)) * HV_RET_TAB_BYTES;
    put(e, UBFM_X(7, 30, 2, 2 + HV_RET_BITS - 1));                        /* ubfx x7, x30, #2, #bits */
    put(e, ADD_LSL(7, 28, 7, 4));
    put(e, 0x91400000u | (tab >> 12) << 10 | 7 << 5 | 7);                 /* add x7, x7, #tab, lsl #12 */
    put(e, 0xa9400000u | 6 << 10 | 7 << 5 | 15);                          /* ldp x15, x6, [x7] */
    put(e, LDR_X(7, 28, HV_CPU_RET_SALT + 8 * (e->flags & HV_BLOCK_EL1)));
    put(e, 0xca000000u | 30 << 16 | 7 << 5 | 7);                          /* eor x7, x7, x30 */
    put(e, 0xca000000u | 7 << 16 | 15 << 5 | 15);                         /* eor x15, x15, x7 */
    uint32_t to_miss = e->n;
    put(e, CBNZ_X(15));
    put(e, 0xd65f0000u | 6 << 5);                                         /* ret x6 */
    int p0 = e->in_cold;
    if (!p0) cold_on(e);
    link_here(e, to_miss, p0);
    put(e, MOV_X(15, 30));
    put_ijc_jump(e, 1);
    if (!p0) cold_off(e);
}

/* HV_HOIST_CHECK: a conditional branch back (a loop's) tests exit_req before it, in the hot part, where the test is
   one load and a branch not taken; its exit then needs none, so chaining takes the guest's branch straight to the
   target block, one taken branch an iteration (as forward ones are) where the exit's own test made it two. Asked to
   leave: the branch's condition again, and out to the dispatcher at whichever PC it picks. bare: the branch
   instruction with no offset, x15 free. */
static int hoist_check(Emit *e, uint64_t target, uint64_t pc) {
    return hv_knob[HV_K_HOIST] && target <= pc && !check_exits_always() && !e->no_ijc;
}

static void put_exit_to(Emit *e, uint64_t va) { /* to the dispatcher at va, guest x0/x1 kept */
    put(e, STP_X(0, 1, 28, 0));
    put_lit(e, 0, va);
    put(e, STR_X(0, 28, HV_CPU_PC));
    put(e, LDR_X(1, 28, HV_CPU_EXIT));
    put(e, BR_X(1));
}

static void put_hoisted_check(Emit *e, uint32_t bare, uint64_t target, uint64_t pc) {
    put(e, 0xb9400000u | (HV_CPU_EXIT_REQ / 4) << 10 | 28 << 5 | 15); /* ldr w15, [x28, #exit_req] */
    uint32_t to_req = e->n;
    put(e, 0x35000000u | 15);                                         /* cbnz w15, asked (below) */
    int p0 = e->in_cold;
    if (!p0) cold_on(e);
    link_here(e, to_req, p0);
    uint32_t to_taken = e->n;
    put(e, bare);                                                     /* the guest's condition: taken? */
    put_exit_to(e, pc + 4);
    patch_branch(&e->w[to_taken], (int64_t)e->n - to_taken);
    put_exit_to(e, target);
    if (!p0) cold_off(e);
}

/* Ends the block (returns 1), or 0 for a conditional branch the block goes on past. */
static int translate_branch(Emit *e, uint32_t insn, uint64_t pc) {
    if ((insn & 0x7c000000) == 0x14000000) { /* B, BL */
        uint64_t target = pc + (sext(insn & 0x3ffffff, 26) << 2);
        if (insn >> 31) {
            if (use_ras(e)) put_call(e, pc + 4);
            else put_const_to(e, 30, pc + 4);
        }
        put_branch_exit(e, target, 1);
        return 1;
    }
    if ((insn & 0xff000000) == 0x54000000) { /* B.cond, BC.cond */
        uint64_t target = pc + (sext((insn >> 5) & 0x7ffff, 19) << 2);
        uint32_t cond = insn & 0xf;
        if (cond >= 14) {
            put_branch_exit(e, target, 1);
            return 1;
        }
        int hoist = hoist_check(e, target, pc);
        if (hoist) put_hoisted_check(e, 0x54000000u | cond, target, pc);
        put(e, 0x54000000u | cond);
        put_ool(e, (Ool){.kind = OOL_EXIT, .pc = target, .plain = (uint8_t)(hoist ? 2 : 1)});
        if (superblocks() && go_on(e, pc)) return 0;
        put_branch_exit(e, pc + 4, 1);
        return 1;
    }
    /* CBZ/CBNZ (imm19) and TBZ/TBNZ (imm14) */
    int tb = (insn & 0x7e000000) == 0x36000000;
    uint64_t target = pc + (tb ? sext((insn >> 5) & 0x3fff, 14) << 2 : sext((insn >> 5) & 0x7ffff, 19) << 2);
    uint32_t bare = tb ? insn & ~(0x3fffu << 5) : insn & ~(0x7ffffu << 5);
    int rt = insn & 0x1f;
    if (!is_stolen(rt)) {
        int hoist = hoist_check(e, target, pc);
        if (hoist) put_hoisted_check(e, bare, target, pc);
        put(e, bare);
        put_ool(e, (Ool){.kind = OOL_EXIT, .pc = target, .plain = (uint8_t)(hoist ? 2 : 1)});
        if (superblocks() && go_on(e, pc)) return 0;
        put_branch_exit(e, pc + 4, 1);
        return 1;
    }
    uint8_t s = 15; /* free scratch: the spill below is empty */
    spill(e, &s, 1);
    put(e, LDR_X(s, 28, slot_of(rt)));
    put(e, (bare & ~0x1fu) | s);
    put_ool(e, (Ool){.kind = OOL_EXIT, .pc = target, .nrestore = 1, .restore = {s}, .plain = 1});
    unspill(e, &s, 1);
    if (superblocks() && go_on(e, pc)) return 0;
    put_branch_exit(e, pc + 4, 1);
    return 1;
}

/* BR, BLR, RET and their PAC forms. Ends the block. The target is looked up
   inline in the indirect-branch cache for this block's flags: a hit jumps
   straight to the target block's first entry (which reloads guest x1 and x0);
   a miss leaves for the dispatcher, which fills the cache. Tags are VA ^ salt,
   and the salt changes whenever mappings or code change. */
static void translate_branch_reg(Emit *e, uint32_t insn, uint64_t pc) {
    uint32_t opc = (insn >> 21) & 3;
    int rn = (insn >> 5) & 0x1f;
    if (opc == 2 && ((insn >> 11) & 0x1f) == 1) rn = 30; /* RETAA, RETAB: the Rn field is all ones, the target is x30 */
    if (opc == 2 && rn == 30 && use_ras(e)) { /* RET, the usual way out of a function */
        put_return(e);
        return;
    }
    if (is_stolen(rn)) put(e, LDR_X(15, 28, slot_of(rn)));
    else put(e, MOV_X(15, rn));                             /* 31 is XZR here */
    if (opc == 1) {                                         /* BLR: after reading Rn */
        if (use_ras(e)) put_call(e, pc + 4);
        else put_lit(e, 30, pc + 4);
    }
    put_ijc_jump(e, 1);
}

/* Loads and stores. A, T and U are borrowed host registers: A holds the guest
   address and then the host address, T and U index the TLB and then stand in
   for x18/x28 operands. A miss goes out of line to hv_mem_emulate, which does
   the whole instruction (writeback included) and comes back to the join. */
static void translate_mem(Emit *e, uint32_t insn, uint64_t pc);
static void put_load_field(Emit *e, int rt, uint32_t off);
static void put_store_field(Emit *e, int rt, uint32_t off);

/* LDXR/LDAXR and STXR/STLXR (not the pairs), the way QEMU does them: LDXR is
   an ordinary load that records the address, size and value; STXR, if the
   address still matches, is a compare-and-swap of that value (CASAL, through
   the ordinary fast path), and succeeds if memory still held it. Another CPU's
   store in between makes it fail, and so does an exception (which clears the
   address), as on hardware; only an ABA change goes unnoticed. */
static void translate_excl(Emit *e, uint32_t insn, uint64_t pc) {
    uint32_t size = insn >> 30;
    int rn = (insn >> 5) & 0x1f, rt = insn & 0x1f, rs = (insn >> 16) & 0x1f;
    int load = (insn >> 22) & 1;
    if (!load && (rs == 31 || rs == rt || rs == rn)) { /* the status register is the CAS's compare value: keep these in C */
        put_helper(e, pc, insn, HV_OP_MEM);
        return;
    }
    int a = rn;
    if (rn == 31) {
        put(e, ADD_SP_TO(15));
        a = 15;
    } else if (is_stolen(rn)) {
        put(e, LDR_X(15, 28, slot_of(rn)));
        a = 15;
    }
    if (load) {
        put(e, STR_X(a, 28, HV_CPU_EXCL_ADDR));
        put(e, 0x52800006u | (1u << size) << 5);                /* movz w6, #bytes */
        put(e, 0xb9000000u | (HV_CPU_EXCL_SIZE / 4) << 10 | 28 << 5 | 6); /* str w6, [x28, #excl_size] */
        if ((insn >> 15) & 1) /* LDAXR: keep the acquire (other CPUs' stores must be seen in order) */
            translate_mem(e, size << 30 | 0x08dffc00u | (uint32_t)rn << 5 | (uint32_t)rt, pc); /* ldar{b,h} rt, [rn] */
        else
            translate_mem(e, size << 30 | 0x39400000u | (uint32_t)rn << 5 | (uint32_t)rt, pc); /* ldr{b,h} rt, [rn] */
        put_store_field(e, rt, HV_CPU_EXCL_VAL);
        return;
    }
    put(e, LDR_X(6, 28, HV_CPU_EXCL_ADDR));
    put(e, 0xca000000u | (uint32_t)a << 16 | 6 << 5 | 6); /* eor x6, x6, xa */
    uint32_t to_fail = e->n;
    put(e, CBNZ_X(6));
    put(e, 0x92800006u);                                   /* movn x6, #0: the monitor clears either way */
    put(e, STR_X(6, 28, HV_CPU_EXCL_ADDR));
    put_load_field(e, rs, HV_CPU_EXCL_VAL);                /* rs = the value LDXR saw */
    translate_mem(e, size << 30 | 0x08e0fc00u | (uint32_t)rs << 16 | (uint32_t)rn << 5 | (uint32_t)rt, pc); /* casal */
    /* rs now holds what memory held: success if that is what LDXR saw. */
    if (is_stolen(rs)) put(e, LDR_X(7, 28, slot_of(rs)));
    else put(e, MOV_X(7, rs));
    put(e, LDR_X(6, 28, HV_CPU_EXCL_VAL));
    put(e, 0xca000000u | 7 << 16 | 6 << 5 | 6);            /* eor x6, x6, x7 */
    uint32_t to_lost = e->n;
    put(e, CBNZ_X(6));
    put_const_to(e, rs, 0);
    uint32_t to_done = e->n;
    put(e, B_OP);
    patch_branch(&e->w[to_fail], (int64_t)e->n - to_fail);
    put(e, 0x92800006u);
    put(e, STR_X(6, 28, HV_CPU_EXCL_ADDR));
    patch_branch(&e->w[to_lost], (int64_t)e->n - to_lost);
    put_const_to(e, rs, 1);
    patch_branch(&e->w[to_done], (int64_t)e->n - to_done);
}

static void translate_mem(Emit *e, uint32_t insn, uint64_t pc) {
    HvMemOp op;
    if (hv_mem_decode(insn, &op) != 0) {
        put_helper(e, pc, insn, HV_OP_UNDEF);
        return;
    }
    static int c_excl = -1; /* HV_C_EXCL: keep every exclusive on the C path (debugging) */
    if (c_excl < 0) c_excl = getenv("HV_C_EXCL") != NULL;
    if (op.kind == HV_MEM_EXCL && op.bytes == 1u << op.size && !e->slow_mem && !c_excl) {
        translate_excl(e, insn, pc);
        return;
    }
    /* CASP on the host address, as the guest wrote it, when its two pairs are guest registers that live
       in their own host registers (pairs must be even and consecutive: no stand-ins). macOS 27 runs tens of
       thousands a second, and each took the helper. */
    int casp = op.kind == HV_MEM_CASP;
    if (casp && (is_stolen(op.rs) || is_stolen(op.rs + 1) || is_stolen(op.rt) || is_stolen((op.rt + 1) & 31))) casp = -1;
    if (e->slow_mem || op.kind == HV_MEM_EXCL || casp < 0) {
        put_helper(e, pc, insn, HV_OP_MEM);
        return;
    }
    if (casp) op.align = 1; /* a misaligned one goes to the helper, which raises the alignment fault */
    int literal = op.mode == HV_ADDR_LITERAL;
    int vec_kind = op.kind == HV_MEM_VEC_MULTI || op.kind == HV_MEM_VEC_SINGLE;
    int pair = op.kind == HV_MEM_PAIR;
    int rmw = op.kind == HV_MEM_CAS || op.kind == HV_MEM_ATOMIC || casp;
    int rm_used = op.mode == HV_ADDR_REG || (vec_kind && op.mode == HV_ADDR_POST && op.rm != 31);

    uint32_t used = 0;
    if (!literal) used |= 1u << op.rn;
    if (!op.vec) used |= 1u << op.rt | (pair ? 1u << op.rt2 : 0);
    if (rm_used) used |= 1u << op.rm;
    if (rmw) used |= 1u << op.rs;
    if (casp) used |= 1u << (op.rs + 1) | 1u << op.rt | 1u << ((op.rt + 1) & 31);
    /* Guest x6, x7 and x15 live in the HvCpu: their host registers are as free as ever. */
    if (!hv_knob[HV_K_TEST3]) used &= ~(1u << 6 | 1u << 7 | 1u << 15);
    uint8_t s[3];
    for (int i = 0; i < 3; i++) s[i] = pick(&used);
    int A = s[0], T = s[1], U = s[2];
    uint32_t retry = e->n; /* a refilled TLB miss runs the access again from here */
    spill(e, s, 3);

    /* Grouped (see Group): plan_groups only puts plain single and pair accesses
       with an immediate offset on SP or an ordinary register here. */
    int role = 0;
    Group *grp = NULL;
    if (!e->nogroup && e->cur_i < HV_MAX_BLOCK_INSNS && e->gid[e->cur_i] >= 0 && e->grp[e->gid[e->cur_i]].live) {
        role = e->grole[e->cur_i];
        grp = &e->grp[e->gid[e->cur_i]];
    }
    int g_el = op.unpriv || !(e->flags & HV_BLOCK_EL1) ? 0 : (e->flags & HV_BLOCK_PAN) ? 2 : 1;
    if (role == 1) {
        /* The whole range: A = base + lo, then the usual check for hi - lo bytes. */
        int64_t lo = grp->lo, len = grp->hi - grp->lo;
        int ch = A; /* the register the check reads: base + lo */
        if (op.rn == 31) {
            if (lo) put_add_imm(e, A, 31, lo); /* ADD/SUB (immediate) read SP as register 31 */
            else put(e, ADD_SP_TO(A));
        } else if (lo) {
            put_add_imm(e, A, op.rn, lo);
        } else {
            ch = op.rn;
        }
        uint32_t gtlb = put_tlb_slot(e, T, ch, g_el);
        put(e, LDR_X(U, T, gtlb + (grp->store ? 8 : 0)));
        put(e, SUB_X(U, ch, U));
        if (len > HV_TLB_ADJ + 1) put_add_imm(e, U, U, len - 1 - HV_TLB_ADJ);
        put(e, UBFM_X(U, U, HV_TLB_PAGE_BITS, 63));
        put(e, CBNZ_X(U));
        put_ool(e, (Ool){.kind = OOL_GROUP, .pc = (uint64_t)(grp - e->grp), .nrestore = 3, .restore = {A, T, U}});
        put(e, LDR_X(U, T, gtlb + 24));
        put(e, STR_X(U, 28, HV_CPU_GADDEND + 8 * grp->slot));
        put(e, ADD_LSL(A, ch, U, 0)); /* host address of base + lo */
    } else if (role == 2) {
        if (!(e->u_prev == grp->slot + 1 && U == 15)) put(e, LDR_X(U, 28, HV_CPU_GADDEND + 8 * grp->slot));
        put(e, op.rn == 31 ? ADD_EXT(A, 31, U, 3, 0) : ADD_LSL(A, op.rn, U, 0)); /* host address of the base */
    }
    /* Grouped: A + gimm is the host address (the access encodes gimm when it can). */
    int64_t gimm = role ? (op.mode == HV_ADDR_POST ? 0 : op.imm) - (role == 1 ? grp->lo : 0) : 0;

    /* Effective address into A, or used where it is: a guest base register with nothing added. */
    int sp_base = !literal && op.rn == 31; /* the guest SP, which is the host SP while blocks run */
    int base_stolen = !literal && (sp_base || is_stolen(op.rn));
    int direct = !literal && !base_stolen &&
                 (op.mode == HV_ADDR_POST || ((op.mode == HV_ADDR_OFFSET || op.mode == HV_ADDR_PRE) && op.imm == 0));
    int addr = direct ? op.rn : A;
    int base = op.rn;
    if (role) {
        /* the address is in A already */
    } else if (literal) {
        put_lit(e, A, pc + op.imm);
    } else {
        if (base_stolen) {
            put(e, sp_base ? ADD_SP_TO(A) : LDR_X(A, 28, slot_of(op.rn)));
            base = A;
        }
        switch (op.mode) {
        case HV_ADDR_OFFSET:
        case HV_ADDR_PRE:
            if (!direct) put_add_imm(e, A, base, op.imm);
            break;
        case HV_ADDR_REG: {
            int rm = op.rm;
            if (is_stolen(rm)) {
                put(e, LDR_X(T, 28, slot_of(rm)));
                rm = T;
            }
            put(e, ADD_EXT(A, base, rm, op.ext, op.shift));
            break;
        }
        default:
            if (!direct && base != A) put(e, MOV_X(A, base));
            break;
        }
    }

    /* TLB: index from the first byte; the tag check (see HvTlbEntry) also
       makes an access that runs into the next page miss. Uses no flags. */
    int el_idx = g_el;
    uint32_t slow = ~0u;
    if (!role) {
    uint32_t tlb = put_tlb_slot(e, T, addr, el_idx);
    put(e, LDR_X(U, T, tlb + (op.store ? 8 : 0)));
    put(e, SUB_X(U, addr, U));
    if (op.align && op.bytes > 1) {
        /* and U, U, #~(0x3fff & ~(bytes-1)) after taking the adjustment back off:
           zero only if same page and aligned */
        uint32_t k = __builtin_ctz(op.bytes);
        put(e, 0xd1000000u | HV_TLB_ADJ << 10 | U << 5 | U); /* sub U, U, #ADJ */
        put(e, 0x92400000u | 50u << 16 | (49u + k) << 10 | U << 5 | U);
    } else {
        if (op.bytes > HV_TLB_ADJ + 1) put_add_imm(e, U, U, op.bytes - 1 - HV_TLB_ADJ);
        put(e, UBFM_X(U, U, HV_TLB_PAGE_BITS, 63));
    }
    put(e, CBNZ_X(U));
    /* Writeback forms change their base, and the retry recomputes it: those
       (and RMW, exclusive and vector forms) go straight to the helper. */
    int can_refill = !vec_kind && !rmw && op.mode != HV_ADDR_PRE && op.mode != HV_ADDR_POST && op.bytes <= 64 && !e->slow_mem;
    /* hv_tlb_refill's flags: bit 9 set (so never 0), bit 0 write, bits 2..1 the table, 8..3 bytes - 1 */
    uint32_t rflags = can_refill ? 1u << 9 | (uint32_t)(op.store ? 1 : 0) | (uint32_t)el_idx << 1 | (op.bytes - 1) << 3 : 0;
    put_ool(e, (Ool){.kind = OOL_SLOW, .pc = pc, .insn = insn, .nrestore = 3, .restore = {A, T, U},
                     /* The exact recheck for a page's last 15 bytes costs 6 words in every access's slow path (on by default,
                        see machine.c); with HV_RECHECK=0 such an access takes the refill, which sends it to the helper. */
                     .recheck = hv_knob[HV_K_RECHECK] && !(op.align && op.bytes > 1) && op.bytes <= HV_TLB_ADJ + 1, .addr = (uint8_t)addr,
                     .t = (uint8_t)T, .u = (uint8_t)U, .bytes = (uint8_t)op.bytes,
                     .tag_off = tlb + (op.store ? 8 : 0), .retry = retry, .rflags = rflags});
    slow = e->nool - 1;
    put(e, LDR_X(U, T, tlb + 24));
    }
    /* Single loads and stores take [addr, addend] directly; the rest want the host address in A. */
    int reg_offset = op.kind == HV_MEM_SINGLE || op.kind == HV_MEM_LITERAL || op.kind == HV_MEM_LDRA;
    if (!reg_offset && !role) put(e, ADD_LSL(A, addr, U, 0));
    if (role) {
        int64_t scale = 1ll << op.size;
        int fits = pair ? gimm % scale == 0 && gimm / scale >= -64 && gimm / scale < 64
                        : (gimm >= 0 && gimm % scale == 0 && gimm / scale < 4096) || (gimm >= -256 && gimm < 256);
        if (!fits) {
            put_add_imm(e, A, A, gimm);
            gimm = 0;
        }
    }

    /* Stand-ins for x18/x28 operands: Rt (or Rs for RMW) in T, the other in U. */
    int rt = op.rt, rt2 = op.rt2, rs = op.rs;
    int rt_stolen = !op.vec && is_stolen(op.rt);
    int rt2_stolen = pair && !op.vec && is_stolen(op.rt2);
    int rs_stolen = rmw && is_stolen(op.rs);
    if (rmw) {
        if (rs_stolen) rs = T;
        if (rt_stolen) rt = (op.rt == op.rs) ? T : U;
    } else {
        if (rt_stolen) rt = T;
        if (rt2_stolen) rt2 = (op.rt2 == op.rt) ? T : U;
    }
    if (rs_stolen) put(e, LDR_X(rs, 28, slot_of(op.rs)));
    /* Only RMW forms have an Rs; elsewhere bits 20..16 are immediate bits. */
    int plain_load = op.load && !rmw && !hv_knob[HV_K_TEST3]; /* it only overwrites them */
    if (rt_stolen && !plain_load && !(rmw && rs_stolen && rt == rs)) put(e, LDR_X(rt, 28, slot_of(op.rt)));
    if (rt2_stolen && rt2 != rt && !plain_load) put(e, LDR_X(rt2, 28, slot_of(op.rt2)));

    uint32_t out;
    switch (op.kind) {
    case HV_MEM_PAIR: {
        uint32_t opc = op.vec ? op.size - 2u : (op.sign ? 1u : op.size == 3 ? 2u : 0u);
        out = opc << 30 | 0x29000000u | (uint32_t)op.vec << 26 | (uint32_t)op.load << 22 | rt2 << 10 | A << 5 | rt;
        if (role) out |= ((uint32_t)(gimm >> op.size) & 0x7f) << 15;
        break;
    }
    case HV_MEM_SINGLE:
    case HV_MEM_LITERAL:
    case HV_MEM_LDRA: {
        uint32_t size, opc;
        if (op.vec) {
            size = op.size == 4 ? 0 : op.size;
            opc = (op.size == 4 ? 2u : 0u) | op.load;
        } else {
            size = op.size;
            opc = op.store ? 0 : op.sign == 64 ? 2 : op.sign == 32 ? 3 : 1;
        }
        if (role && gimm >= 0 && gimm % (1ll << op.size) == 0 && (gimm >> op.size) < 4096) /* [A, #gimm] */
            out = size << 30 | 0x39000000u | (uint32_t)op.vec << 26 | opc << 22 | (uint32_t)(gimm >> op.size) << 10 | (uint32_t)A << 5 | rt;
        else if (role) /* LDUR/STUR [A, #gimm] */
            out = size << 30 | 0x38000000u | (uint32_t)op.vec << 26 | opc << 22 | ((uint32_t)gimm & 0x1ff) << 12 | (uint32_t)A << 5 | rt;
        else /* LDR/STR (register), LSL #0: [addr, U] */
            out = size << 30 | 0x38206800u | (uint32_t)op.vec << 26 | opc << 22 | (uint32_t)U << 16 | (uint32_t)addr << 5 | rt;
        break;
    }
    case HV_MEM_VEC_MULTI:
    case HV_MEM_VEC_SINGLE:
        out = (insn & ~(1u << 23) & ~(0x1fu << 16) & ~(0x1fu << 5)) | A << 5;
        break;
    default: /* ORDERED, CAS, ATOMIC: same instruction on the host address */
        out = (insn & ~(0x1fu << 5) & ~0x1fu) | A << 5 | rt;
        if ((insn & 0x3f200c00) == 0x19000000) out &= ~(0x1ffu << 12); /* LDAPUR/STLUR: drop imm9 */
        if (rmw) out = (out & ~(0x1fu << 16)) | rs << 16;
        break;
    }
    put(e, out);

    if (rmw) {
        if (rs_stolen && op.kind == HV_MEM_CAS) put(e, STR_X(rs, 28, slot_of(op.rs)));
        if (rt_stolen && op.kind == HV_MEM_ATOMIC) put(e, STR_X(rt, 28, slot_of(op.rt)));
    } else if (op.load) {
        if (rt_stolen) put(e, STR_X(rt, 28, slot_of(op.rt)));
        if (rt2_stolen) put(e, STR_X(rt2, 28, slot_of(op.rt2)));
    }

    /* Base writeback. */
    if (op.mode == HV_ADDR_PRE || op.mode == HV_ADDR_POST) {
        int b = op.rn;
        if (base_stolen && !sp_base) {
            put(e, LDR_X(T, 28, slot_of(op.rn)));
            b = T;
        }
        if (vec_kind && op.rm != 31) {
            int rm = op.rm;
            if (is_stolen(rm)) {
                put(e, LDR_X(U, 28, slot_of(rm)));
                rm = U;
            }
            put(e, sp_base ? ADD_EXT(31, 31, rm, 3, 0) : ADD_LSL(b, b, rm, 0)); /* add sp, sp, rm, uxtx */
        } else {
            put_add_imm(e, b, b, op.imm); /* the immediate forms take SP as it is */
        }
        if (base_stolen && !sp_base) put(e, STR_X(T, 28, slot_of(op.rn)));
    }
    unspill(e, s, 3);
    if (slow != ~0u) e->ool[slow].join = e->n;
    /* Stand-ins for x18/x28 and writeback may have used U. */
    if (role && U == 15 && !rt2_stolen && !(rt_stolen && rt == U) && op.mode == HV_ADDR_OFFSET) e->u_live = grp->slot + 1;
}

/* DC ZVA: zero the 64-byte block holding Xt, on the host's own copy when the
   TLB maps it writable (xnu's bzero does this a million times a second).
   The host's DC ZVA block is 64 bytes too (checked at translation time);
   a miss, a code page or a device takes the SYS helper. */
static int host_zva_is_64(void) {
    static int known = -1;
    if (known < 0) {
        uint64_t dczid;
        __asm__ volatile("mrs %0, dczid_el0" : "=r"(dczid));
        known = (dczid & 0x1f) == 4;
    }
    return known;
}

/* IC IVAU at EL1 (xnu: 100K+ a second), without the helper when the page holds
   no translated code: look the address up like a load, find its host page, and
   check the main RAM's code-page flags. A TLB miss, another RAM region or a
   code page take the helper, which invalidates. */
static void translate_ic_ivau(Emit *e, uint32_t insn, uint64_t pc) {
    HvRam *ram = e->m->main_ram;
    int rt = insn & 0x1f;
    uint32_t used = 1u << rt;
    uint8_t s[3];
    for (int i = 0; i < 3; i++) s[i] = pick(&used);
    int A = s[0], T = s[1], U = s[2];
    spill(e, s, 3);
    if (is_stolen(rt)) put(e, LDR_X(A, 28, slot_of(rt)));
    else put(e, MOV_X(A, rt)); /* rt 31 is XZR here */
    int el_idx = (e->flags & HV_BLOCK_PAN) ? 2 : 1;
    uint32_t tlb = put_tlb_slot(e, T, A, el_idx);
    put(e, LDR_X(U, T, tlb + 0)); /* read tag */
    put(e, SUB_X(U, A, U));
    put(e, UBFM_X(U, U, HV_TLB_PAGE_BITS, 63));
    put(e, CBNZ_X(U));
    put_ool(e, (Ool){.kind = OOL_SYS, .pc = pc, .insn = insn, .nrestore = 3, .restore = {A, T, U}});
    uint32_t o1 = e->nool - 1;
    put(e, LDR_X(U, T, tlb + 24));
    put(e, ADD_LSL(A, A, U, 0));                                   /* host address */
    put_lit(e, U, (uint64_t)(uintptr_t)ram->host);
    put(e, SUB_X(A, A, U));
    put(e, UBFM_X(A, A, HV_TLB_PAGE_BITS, 63));                    /* page index in the main RAM */
    put(e, UBFM_X(U, A, ram->pages_log2, 63));
    put(e, CBNZ_X(U));                                             /* not in it */
    put_ool(e, (Ool){.kind = OOL_SYS, .pc = pc, .insn = insn, .nrestore = 3, .restore = {A, T, U}});
    uint32_t o2 = e->nool - 1;
    put_lit(e, U, (uint64_t)(uintptr_t)ram->code_pages);
    put(e, 0x38606800u | (uint32_t)A << 16 | (uint32_t)U << 5 | (uint32_t)U); /* ldrb wU, [xU, xA] */
    put(e, CBNZ_X(U) & ~(1u << 31));                               /* cbnz wU: code there */
    put_ool(e, (Ool){.kind = OOL_SYS, .pc = pc, .insn = insn, .nrestore = 3, .restore = {A, T, U}});
    uint32_t o3 = e->nool - 1;
    unspill(e, s, 3);
    e->ool[o1].join = e->ool[o2].join = e->ool[o3].join = e->n;
}

static void translate_dc_zva(Emit *e, uint32_t insn, uint64_t pc) {
    int rt = insn & 0x1f;
    uint32_t used = 1u << rt;
    uint8_t s[3];
    for (int i = 0; i < 3; i++) s[i] = pick(&used);
    int A = s[0], T = s[1], U = s[2];
    spill(e, s, 3);
    if (is_stolen(rt)) put(e, LDR_X(A, 28, slot_of(rt)));
    else put(e, MOV_X(A, rt)); /* rt 31 is XZR here */
    put(e, 0x92400000u | 58u << 16 | 57u << 10 | A << 5 | A); /* and A, A, #~63 */
    int el_idx = !(e->flags & HV_BLOCK_EL1) ? 0 : (e->flags & HV_BLOCK_PAN) ? 2 : 1;
    uint32_t tlb = put_tlb_slot(e, T, A, el_idx);
    put(e, LDR_X(U, T, tlb + 8)); /* write tag */
    put(e, SUB_X(U, A, U));
    put(e, UBFM_X(U, U, HV_TLB_PAGE_BITS, 63));
    put(e, CBNZ_X(U));
    put_ool(e, (Ool){.kind = OOL_SYS, .pc = pc, .insn = insn, .nrestore = 3, .restore = {A, T, U}});
    uint32_t slow = e->nool - 1;
    put(e, LDR_X(U, T, tlb + 24));
    put(e, ADD_LSL(A, A, U, 0));
    put(e, 0xd50b7420u | A); /* dc zva, A */
    unspill(e, s, 3);
    e->ool[slow].join = e->n;
}

/* System registers that are plain storage, or host registers that already
   hold the guest's value: MRS/MSR to these need no helper and do not end the
   block. Anything with side effects (DAIF, SPSel writes, the MMU, timers, GIC)
   still goes through the helper. Returns 1 if handled. */
#define SR(op0, op1, crn, crm, op2) ((op0) << 14 | (op1) << 11 | (crn) << 7 | (crm) << 3 | (op2))

static void put_load_field(Emit *e, int rt, uint32_t off) {
    if (rt == 31) return;
    if (!is_stolen(rt)) {
        put(e, LDR_X(rt, 28, off));
        return;
    }
    uint8_t s = 15; /* free scratch: the spill below is empty */
    spill(e, &s, 1);
    put(e, LDR_X(s, 28, off));
    put(e, STR_X(s, 28, slot_of(rt)));
    unspill(e, &s, 1);
}

static void put_store_field(Emit *e, int rt, uint32_t off) {
    if (!is_stolen(rt)) {
        put(e, STR_X(rt, 28, off)); /* 31 stores XZR */
        return;
    }
    uint8_t s = 15; /* free scratch: the spill below is empty */
    spill(e, &s, 1);
    put(e, LDR_X(s, 28, slot_of(rt)));
    put(e, STR_X(s, 28, off));
    unspill(e, &s, 1);
}

/* MRS/MSR of NZCV, FPCR, FPSR: the host registers are the guest's. */
static void put_native_sysreg(Emit *e, uint32_t insn, int read) {
    int rt = insn & 0x1f;
    if (!is_stolen(rt)) {
        put(e, insn);
        return;
    }
    uint8_t s = 15; /* free scratch: the spill below is empty */
    spill(e, &s, 1);
    if (read) {
        put(e, (insn & ~0x1fu) | s);
        put(e, STR_X(s, 28, slot_of(rt)));
    } else {
        put(e, LDR_X(s, 28, slot_of(rt)));
        put(e, (insn & ~0x1fu) | s);
    }
    unspill(e, &s, 1);
}

/* MRS/MSR of a PSTATE bit kept in cpu->pstate_extra at the same position as in the register. */
static void put_pstate_bit(Emit *e, int rt, int bit, int read) {
    uint32_t used = 1u << rt;
    uint8_t s[3];
    for (int i = 0; i < 3; i++) s[i] = pick(&used);
    int t = s[0], u = s[1], v = s[2];
    spill(e, s, 3);
    put(e, LDR_X(t, 28, HV_CPU_PSTATE_EXTRA));
    if (read) {
        put(e, UBFM_X(t, t, bit, bit));                       /* ubfx t, t, #bit, #1 */
        put(e, UBFM_X(t, t, (64 - bit) & 63, 63 - bit));      /* lsl t, t, #bit */
        if (rt != 31) put(e, is_stolen(rt) ? STR_X(t, 28, slot_of(rt)) : MOV_X(rt, t));
    } else {
        int r = rt;
        if (rt == 31) {
            put(e, 0xd2800000u | v);                          /* movz v, #0 */
            r = v;
        } else if (is_stolen(rt)) {
            put(e, LDR_X(v, 28, slot_of(rt)));
            r = v;
        }
        put(e, 0xca000000u | (uint32_t)r << 16 | (uint32_t)t << 5 | (uint32_t)u); /* eor u, t, r */
        put(e, UBFM_X(u, u, bit, bit));                       /* the bit that differs */
        put(e, 0xca000000u | (uint32_t)u << 16 | (uint32_t)bit << 10 | (uint32_t)t << 5 | (uint32_t)t); /* eor t, t, u, lsl #bit */
        put(e, STR_X(t, 28, HV_CPU_PSTATE_EXTRA));
    }
    unspill(e, s, 3);
}

/* The commonest PSTATE writes, at EL1, without the helper. DAIF: a masked
   interrupt let through is taken before the next instruction (leave if either
   line is up; the dispatcher sorts out masking). SPSel: the guest SP is the
   host SP, so the switch swaps it with the banked one, then continues through
   the new flags' branch cache. PAN: the bit, then likewise. Returns 0 if not
   handled, 1 if the block carries on, 2 if it ended. */
static int translate_msr_inline(Emit *e, uint32_t insn, uint64_t pc) {
    if (!(e->flags & HV_BLOCK_EL1) || e->no_ijc) return 0;
    int imm_form = ((insn >> 19) & 3) == 0;
    uint32_t field = ((insn >> 16) & 7) << 3 | ((insn >> 5) & 7), imm = (insn >> 8) & 0xf;
    uint32_t reg = (insn >> 5) & 0xffff;
    int daif_reg = !imm_form && reg == (3u << 14 | 3u << 11 | 4u << 7 | 2u << 3 | 1u); /* MSR DAIF, Xt */
    if ((imm_form && (field == 036 || field == 037)) || daif_reg) {
        if (daif_reg) {
            int rt = insn & 0x1f;
            if (rt == 31) put(e, 0xd2800006u);                                        /* movz x6, #0 */
            else if (is_stolen(rt)) put(e, LDR_X(6, 28, slot_of(rt)));
            else put(e, MOV_X(6, rt));
            put(e, 0x92400000u | 58u << 16 | 3u << 10 | 6 << 5 | 6);                 /* and x6, x6, #0x3c0 */
        } else {
            put(e, LDR_X(6, 28, HV_CPU_DAIF));
            put(e, 0xd2800007u | (imm << 6) << 5);                                   /* movz x7, #imm << 6 */
            put(e, (field == 036 ? 0xaa000000u : 0x8a200000u) | 7 << 16 | 6 << 5 | 6); /* orr / bic x6, x6, x7 */
        }
        put(e, STR_X(6, 28, HV_CPU_DAIF));
        if (field != 036 || daif_reg) { /* it may have unmasked something */
            put(e, LDR_X(7, 28, HV_CPU_LINES_PTR));
            put(e, LDR_X(7, 7, 0));                                                  /* irq and fiq lines */
            put(e, CBNZ_X(7));
            put_ool(e, (Ool){.kind = OOL_EXIT, .pc = pc + 4});
        }
        return 1;
    }
    if (!imm_form && reg == (3u << 14 | 0u << 11 | 4u << 7 | 2u << 3 | 0u)) { /* MSR SPSel, Xt: bit 0 picks, at run time */
        int rt = insn & 0x1f;
        uint32_t flags = e->flags, to_set = 0;
        if (rt != 31) {
            if (is_stolen(rt)) put(e, LDR_X(7, 28, slot_of(rt)));
            else put(e, MOV_X(7, rt));
            to_set = e->n;
            put(e, 0x37000007u);                                                     /* tbnz x7, #0, set (patched) */
        }
        for (uint32_t on = 0; on < (rt == 31 ? 1u : 2u); on++) {
            if (on) patch_branch(&e->w[to_set], (int64_t)e->n - to_set);
            if (!!(flags & HV_BLOCK_SPX) != on) {
                put(e, ADD_SP_TO(6));
                put(e, STR_X(6, 28, on ? HV_CPU_SP_EL0 : HV_CPU_SP_EL1));
                put(e, LDR_X(6, 28, on ? HV_CPU_SP_EL1 : HV_CPU_SP_EL0));
                put(e, 0x910000dfu);                                                 /* mov sp, x6 */
                put(e, on ? 0x52800026u : 0x52800006u);
                put(e, 0xb9000000u | (HV_CPU_SPSEL / 4) << 10 | 28 << 5 | 6);
            }
            put_lit(e, 15, pc + 4);
            e->flags = on ? flags | HV_BLOCK_SPX : flags & ~HV_BLOCK_SPX;
            put_ijc_jump(e, 1);
            e->flags = flags;
        }
        return 2;
    }
    if (imm_form && (field == 005 || field == 004)) { /* SPSel, PAN */
        uint32_t flags = e->flags, bit = field == 005 ? HV_BLOCK_SPX : HV_BLOCK_PAN, on = imm & 1;
        if (!!(flags & bit) == on) return 1; /* already so: nothing changes */
        if (field == 005) {
            put(e, ADD_SP_TO(6));                                                    /* mov x6, sp */
            put(e, STR_X(6, 28, on ? HV_CPU_SP_EL0 : HV_CPU_SP_EL1));                /* the one in use goes to its bank */
            put(e, LDR_X(6, 28, on ? HV_CPU_SP_EL1 : HV_CPU_SP_EL0));
            put(e, 0x910000dfu);                                                     /* mov sp, x6 */
            put(e, on ? 0x52800026u : 0x52800006u);                                  /* movz w6, #on */
            put(e, 0xb9000000u | (HV_CPU_SPSEL / 4) << 10 | 28 << 5 | 6);            /* str w6, [x28, #spsel] */
        } else {
            put(e, LDR_X(6, 28, HV_CPU_PSTATE_EXTRA));
            put(e, 0xd2a00807u);                                                     /* movz x7, #0x40, lsl #16: PAN */
            put(e, (on ? 0xaa000000u : 0x8a200000u) | 7 << 16 | 6 << 5 | 6);
            put(e, STR_X(6, 28, HV_CPU_PSTATE_EXTRA));
        }
        put_lit(e, 15, pc + 4);
        e->flags = on ? flags | bit : flags & ~bit;
        put_ijc_jump(e, 1);
        e->flags = flags;
        return 2;
    }
    return 0;
}

/* The guest's counter is the host's mach_absolute_time (CNTVOFF is 0): the
   counter register the commpage names plus the commpage's offset, read the way
   libsystem does. EL0 reads check CNTKCTL_EL1's enables and trap through the
   helper when they are off. */
#define COMMPAGE_TIMEBASE_OFFSET 0xfffffc088ull
#define COMMPAGE_TIMEBASE_TYPE   0xfffffc090ull
static int translate_counter(Emit *e, uint32_t insn, uint64_t pc, int physical) {
    static int type = -1;
    if (type < 0) type = *(volatile uint8_t *)(uintptr_t)COMMPAGE_TIMEBASE_TYPE;
    if (type == 0) return 0; /* the kernel's own path: leave it to the helper */
    int rt = insn & 0x1f;
    if (rt == 31) return 1;
    if (!(e->flags & HV_BLOCK_EL1)) {
        put(e, 0xb9400000u | (HV_CPU_EL0_CNT / 4) << 10 | 28 << 5 | 6); /* ldr w6, [x28, #el0_cnt] */
        put(e, 0x36000000u | (physical ? 0u : 1u) << 19 | 6);         /* tbz w6, #enable, helper (patched) */
        put_ool(e, (Ool){.kind = OOL_MRS, .pc = pc, .insn = insn});
    }
    uint32_t slow = e->nool - 1;
    int el0 = !(e->flags & HV_BLOCK_EL1);
    int r = is_stolen(rt) ? 15 : rt;
    if (type == 1) put(e, 0xd5033fdfu);                                  /* isb; then CNTVCT_EL0 */
    put(e, (type == 1 ? 0xd53be040u : type == 2 ? 0xd53be0c0u : 0xd53cfac0u) | (uint32_t)r);
    put_lit(e, 6, COMMPAGE_TIMEBASE_OFFSET);
    put(e, LDR_X(6, 6, 0));
    put(e, ADD_LSL(r, r, 6, 0));
    if (r != rt) put(e, STR_X(r, 28, slot_of(rt)));
    if (el0) e->ool[slow].join = e->n;
    return 1;
}

static int translate_sysreg_inline(Emit *e, uint32_t insn, uint64_t pc) {
    if (((insn >> 19) & 3) == 0) return 0; /* MSR (immediate) */
    int read = (insn >> 21) & 1, rt = insn & 0x1f;
    uint32_t reg = (insn >> 5) & 0xffff;
    int el1 = (e->flags & HV_BLOCK_EL1) != 0, spx = (e->flags & HV_BLOCK_SPX) != 0;
    int fp_ok = !(e->flags & HV_BLOCK_FPOFF);
    switch (reg) {
    case SR(3, 3, 14, 0, 2): /* CNTVCT_EL0 */
    case SR(3, 3, 14, 0, 6): /* CNTVCTSS_EL0 */
    case SR(3, 3, 14, 0, 1): /* CNTPCT_EL0 */
    case SR(3, 3, 14, 0, 5): /* CNTPCTSS_EL0 */
        if (!read) return 0;
        return translate_counter(e, insn, pc, reg == SR(3, 3, 14, 0, 1) || reg == SR(3, 3, 14, 0, 5));
    case SR(3, 3, 4, 2, 0): /* NZCV */
        put_native_sysreg(e, insn, read);
        return 1;
    case SR(3, 3, 4, 4, 0): /* FPCR */
    case SR(3, 3, 4, 4, 1): /* FPSR */
        if (!fp_ok) return 0; /* traps: leave it to the helper */
        put_native_sysreg(e, insn, read);
        return 1;
    case SR(3, 3, 13, 0, 2): /* TPIDR_EL0 */
        if (read) put_load_field(e, rt, HV_CPU_TPIDR_EL0);
        else put_store_field(e, rt, HV_CPU_TPIDR_EL0);
        return 1;
    case SR(3, 3, 13, 0, 3): /* TPIDRRO_EL0: EL0 may only read it */
        if (read) put_load_field(e, rt, HV_CPU_TPIDRRO_EL0);
        else if (el1) put_store_field(e, rt, HV_CPU_TPIDRRO_EL0);
        else return 0;
        return 1;
    case SR(3, 0, 13, 0, 4): /* TPIDR_EL1 */
    case SR(3, 0, 4, 0, 1):  /* ELR_EL1 */
    case SR(3, 0, 4, 0, 0): { /* SPSR_EL1 */
        if (!el1) return 0;
        uint32_t off = reg == SR(3, 0, 13, 0, 4) ? HV_CPU_TPIDR_EL1 : reg == SR(3, 0, 4, 0, 1) ? HV_CPU_ELR_EL1 : HV_CPU_SPSR_EL1;
        if (read) put_load_field(e, rt, off);
        else put_store_field(e, rt, off);
        return 1;
    }
    case SR(3, 0, 1, 0, 0): /* SCTLR_EL1, ESR_EL1, FAR_EL1: reads (writes have side effects or are rare) */
    case SR(3, 0, 5, 2, 0):
    case SR(3, 0, 6, 0, 0): {
        if (!read || !el1) return 0;
        uint32_t off = reg == SR(3, 0, 1, 0, 0) ? HV_CPU_SCTLR_EL1 : reg == SR(3, 0, 5, 2, 0) ? HV_CPU_ESR_EL1 : HV_CPU_FAR_EL1;
        put_load_field(e, rt, off);
        return 1;
    }
    case SR(3, 0, 4, 1, 0): /* SP_EL0: only while it is banked (SPSel = 1) */
        if (!el1 || !spx) return 0;
        if (read) put_load_field(e, rt, HV_CPU_SP_EL0);
        else put_store_field(e, rt, HV_CPU_SP_EL0);
        return 1;
    case SR(3, 3, 4, 2, 1): /* DAIF: reads only; writes can unmask interrupts */
        if (!read || !el1) return 0;
        put_load_field(e, rt, HV_CPU_DAIF);
        return 1;
    case SR(3, 0, 4, 2, 0): /* SPSel */
        if (!read || !el1) return 0;
        put_const_to(e, rt, (uint64_t)spx);
        return 1;
    case SR(3, 0, 4, 2, 2): /* CurrentEL */
        if (!read || !el1) return 0;
        put_const_to(e, rt, 4);
        return 1;
    case SR(3, 3, 4, 2, 5): /* DIT: one bit of PSTATE, nothing depends on it here */
    case SR(3, 3, 4, 2, 6): /* SSBS */
        put_pstate_bit(e, rt, reg == SR(3, 3, 4, 2, 5) ? 24 : 12, read);
        return 1;
    default:
        return 0;
    }
}

static int is_fp_insn(uint32_t insn) {
    uint32_t op0 = (insn >> 25) & 0xf;
    if ((op0 & 0x7) == 0x7) return 1;                        /* SIMD and FP data processing */
    if ((op0 & 0x5) == 0x4 && ((insn >> 26) & 1)) return 1;  /* SIMD and FP loads and stores */
    return 0;
}

/* Returns 1 if the instruction ends the block. */
static int translate_one(Emit *e, uint32_t insn, uint64_t pc) {
    HvInsnKind kind = hv_classify(insn);
    if ((e->flags & HV_BLOCK_FPOFF) && kind != HV_INSN_UNDEFINED && is_fp_insn(insn)) {
        put_helper(e, pc, insn, HV_OP_FPTRAP);
        return 1;
    }
    switch (kind) {
    case HV_INSN_COPY:
        translate_copy(e, insn);
        return 0;
    case HV_INSN_PCREL:
        translate_pcrel(e, insn, pc);
        return 0;
    case HV_INSN_BRANCH:
        return translate_branch(e, insn, pc);
    case HV_INSN_BRANCH_REG:
        translate_branch_reg(e, insn, pc);
        return 1;
    case HV_INSN_PAC:
        translate_pac(e, insn);
        return 0;
    case HV_INSN_PREFETCH:
        return 0;
    case HV_INSN_LOAD_LITERAL:
    case HV_INSN_MEMORY:
        translate_mem(e, insn, pc);
        return 0;
    case HV_INSN_SYSREG:
        if (translate_sysreg_inline(e, insn, pc)) return 0;
        if (((insn >> 21) & 1) && ((insn >> 19) & 3)) { /* MRS */
            put_helper(e, pc, insn, HV_OP_MRS);
            return 0;
        }
        {
            int r = translate_msr_inline(e, insn, pc);
            if (r) return r - 1; /* 1: carries on, 2: the block ended */
        }
        put_helper(e, pc, insn, HV_OP_MSR); /* state blocks depend on may change */
        if (((insn >> 19) & 3) == 0 && (e->flags & HV_BLOCK_EL1)) {
            /* MSR SPSel/PAN, #imm at EL1: the flags of the next block are known, so
               go to it through that flags' indirect-branch cache instead of the dispatcher. */
            uint32_t op1 = (insn >> 16) & 7, op2 = (insn >> 5) & 7, imm = (insn >> 8) & 1;
            uint32_t bit = op1 == 0 && op2 == 5 ? HV_BLOCK_SPX : op1 == 0 && op2 == 4 ? HV_BLOCK_PAN : 0;
            if (bit) {
                uint32_t flags = e->flags;
                put_lit(e, 15, pc + 4);
                e->flags = imm ? flags | bit : flags & ~bit;
                put_ijc_jump(e, 1);
                e->flags = flags;
                return 1;
            }
        }
        {
            /* Writes blocks are translated against (the flags: FP enable, PAN,
               SPSel) or that change translation leave for the dispatcher. The
               rest carry on: DAIF writes inside the block (the helper asks to
               leave if an interrupt is now due), others through a chainable exit,
               which still notices exit_req. */
            uint32_t reg = (insn >> 5) & 0xffff;
#define SR(op0, op1, crn, crm, op2) ((op0) << 14 | (op1) << 11 | (crn) << 7 | (crm) << 3 | (op2))
            int imm_form = ((insn >> 19) & 3) == 0;
            uint32_t field = ((insn >> 16) & 7) << 3 | ((insn >> 5) & 7);
            if ((imm_form && (field == 036 || field == 037)) || (!imm_form && reg == SR(3, 3, 4, 2, 1))) {
                /* An interrupt this let through is taken before the next instruction. */
                put(e, 0xb9400000u | (HV_CPU_EXIT_REQ / 4) << 10 | 28 << 5 | 6); /* ldr w6, [x28, #exit_req] */
                put(e, 0x35000000u | 6);                                         /* cbnz w6, out of line */
                put_ool(e, (Ool){.kind = OOL_EXIT, .pc = pc + 4});
                return 0;
            }
            if (!imm_form && reg == SR(3, 0, 4, 2, 0) && (e->flags & HV_BLOCK_EL1) && !e->no_ijc) {
                /* MSR SPSel, Xt at EL1: bit 0 of Xt picks the next block's flags, so pick the
                   cache at run time (the immediate form knows it now). */
                int rt = insn & 0x1f, r = rt;
                if (rt != 31 && is_stolen(rt)) {
                    put(e, LDR_X(15, 28, slot_of(rt)));
                    r = 15;
                }
                uint32_t flags = e->flags, to_set = 0;
                if (rt != 31) {
                    to_set = e->n;
                    put(e, 0x37000000u | (uint32_t)r); /* tbnz xr, #0, set (patched below) */
                }
                for (int set = 0; set < (rt == 31 ? 1 : 2); set++) {
                    if (set) patch_branch(&e->w[to_set], (int64_t)e->n - to_set);
                    put_lit(e, 15, pc + 4);
                    e->flags = set ? flags | HV_BLOCK_SPX : flags & ~HV_BLOCK_SPX;
                    put_ijc_jump(e, 1);
                    e->flags = flags;
                }
                return 1;
            }
            /* TTBR0 (a process switch) and SCTLR carry on too: the indirect-branch
               cache reads the new salt at run time, and a change that matters to
               chained exits sets exit_req. */
            if (!imm_form && reg != SR(3, 0, 1, 0, 2) && reg != SR(3, 0, 4, 2, 3) && reg != SR(3, 0, 4, 2, 0) &&
                reg != SR(3, 0, 2, 0, 1) && reg != SR(3, 0, 2, 0, 2) && !e->no_ijc) {
                put_branch_exit(e, pc + 4, 0);
                return 1;
            }
#undef SR
        }
        put_exit(e, pc + 4);
        return 1;
    case HV_INSN_SYS: {
        uint32_t crn = (insn >> 12) & 0xf, crm = (insn >> 8) & 0xf, op1 = (insn >> 16) & 7, op2 = (insn >> 5) & 7;
        /* Data-cache maintenance by VA or set/way: memory here is coherent, so
           there is nothing to do. DC ZVA writes memory and AT translates, so
           those still go to the helper. */
        int dc_zva = op1 == 3 && crm == 4 && op2 == 1;
        int at = crm == 8 || crm == 9;
        if (crn == 7 && crm != 1 && crm != 5 && !dc_zva && !at) return 0;
        if (dc_zva && !e->slow_mem && host_zva_is_64()) {
            translate_dc_zva(e, insn, pc);
            return 0;
        }
        if (crn == 7 && op1 == 3 && crm == 5 && op2 == 1 && !hv_ic_ivau_drops()) return 0; /* IC IVAU: nothing to do, see there */
        if (crn == 7 && op1 == 3 && crm == 5 && op2 == 1 && (e->flags & HV_BLOCK_EL1) && !e->slow_mem && e->m->main_ram) {
            translate_ic_ivau(e, insn, pc);
            return 0;
        }
        put_helper(e, pc, insn, HV_OP_SYS);
        /* DC and AT continue, and so does IC IVAU: invalidating code sets exit_req,
           so the block leaves at its next exit if it mattered. So do TLBIs: the
           helper leaves itself when one may cover the page this block runs from
           (see hv_helper). IC IALLU leaves. */
        if (crn == 7 && ((crm != 1 && crm != 5) || (op1 == 3 && crm == 5 && op2 == 1))) return 0;
        static int tlbi_leaves = -1; /* HV_TLBI_LEAVES: every TLBI ends its block, as before */
        if (tlbi_leaves < 0) tlbi_leaves = getenv("HV_TLBI_LEAVES") != NULL;
        if (crn == 8 && !e->no_ijc && !tlbi_leaves) return 0;
        put_exit(e, pc + 4);
        return 1;
    }
    case HV_INSN_BARRIER: {
        uint32_t op2 = (insn >> 5) & 7;
        if (op2 == 2) {
            put_helper(e, pc, insn, HV_OP_CLREX);
            return 0;
        }
        if (op2 == 6) { /* ISB: later instructions must see earlier system register writes */
            /* Writes that change how blocks are translated already left. What is
               left to synchronise is code invalidation, which sets exit_req: the
               chainable exit checks it. */
            if (e->no_ijc) put_exit(e, pc + 4);
            else put_branch_exit(e, pc + 4, 0);
            return 1;
        }
        put(e, insn); /* DSB, DMB, SB */
        return 0;
    }
    case HV_INSN_WAIT:
        if (((insn >> 5) & 7) == 3 || ((insn >> 12) & 0xf) == 1) put_helper(e, pc, insn, HV_OP_WFI); /* WFI, WFIT, WFET */
        else { /* WFE: waiting for another CPU; the helper waits for the monitored word, then the block goes on */
            put_helper(e, pc, insn, HV_OP_WFE);
            if (e->no_ijc) put_exit(e, pc + 4);
            else put_branch_exit(e, pc + 4, 0); /* checks exit_req: the wait may have ended for a request */
            return 1;
        }
        put_exit(e, pc + 4);
        return 1;
    case HV_INSN_EXCEPTION:
        put_helper(e, pc, insn, HV_OP_EXCEPTION);
        if ((insn & 0xffe0001f) == 0xd4000002 && (e->flags & HV_BLOCK_EL1)) return 0; /* HVC returns like a call */
        put_exit(e, pc + 4);
        return 1;
    case HV_INSN_ERET:
        put_helper(e, pc, insn, HV_OP_ERET);
        put_exit(e, pc + 4);
        return 1;
    case HV_INSN_UNDEFINED:
        put_helper(e, pc, insn, HV_OP_UNDEF);
        put_exit(e, pc + 4);
        return 1;
    }
    return 1;
}

static void patch_branch(uint32_t *w, int64_t off) {
    uint32_t op = *w;
    if ((op & 0x7c000000) == B_OP) *w = (op & 0xfc000000) | (off & 0x3ffffff);
    else if ((op & 0x7e000000) == 0x36000000) *w = (op & ~(0x3fffu << 5)) | (uint32_t)(off & 0x3fff) << 5;
    else *w = (op & ~(0x7ffffu << 5)) | (uint32_t)(off & 0x7ffff) << 5; /* B.cond, CBZ, LDR literal */
}

/* The hot-part branch at `at` to the next cold word. TBZ and TBNZ reach only 32 KB, so they
   go through one word at the end of the hot part. */
static uint32_t link_site(Emit *e, uint32_t at) {
    if (!e->flat && (e->hot[at] & 0x7e000000) == 0x36000000) {
        uint32_t v = e->other_n++;
        if (v < MAX_WORDS) e->hot[v] = B_OP;
        patch_branch(&e->hot[at], (int64_t)v - at);
        at = v;
    }
    link_here(e, at, 0);
    return at;
}

/* Written in the cold part; everything it branches back to is in the hot one. */
static void emit_ool(Emit *e) {
    for (uint32_t i = 0; i < e->nool; i++) {
        Ool *o = &e->ool[i];
        int refill = o->kind == OOL_SLOW && o->rflags;
        uint32_t retry = o->retry;
        if (refill && !e->flat) { /* the refill's way back counts words, so a word near it to go on from */
            retry = e->n;
            put(e, B_OP);
            plink(e, retry, 1, o->retry, 0);
        }
        uint32_t site_at = link_site(e, o->at);
        if (o->recheck) { /* the fast check refuses a page's last 15 bytes: check exactly, and go back if it fits */
            put(e, LDR_X(o->u, o->t, o->tag_off));
            put(e, SUB_X(o->u, o->addr, o->u));
            put_add_imm(e, o->u, o->u, (int64_t)o->bytes - 1 - HV_TLB_ADJ);
            put(e, UBFM_X(o->u, o->u, HV_TLB_PAGE_BITS, 63));
            put(e, CBNZ_X(o->u) | 2u << 5); /* cbnz u, +2: a real miss */
            put(e, B_OP);
            plink(e, e->n - 1, 1, o->at + 1, 0);
        }
        if (o->kind == OOL_GROUP) {
            unspill(e, o->restore, o->nrestore);
            put(e, B_OP);
            plink(e, e->n - 1, 1, e->grp[o->pc].copy_at, 0);
            continue;
        }
        if (refill) {
            /* A TLB miss: hv_helper_thunk tries the light refill first (x7 = the
               address; x7 is never a guest register) and, if it filled the TLB,
               comes back to run the access again; else it is the usual helper. */
            put(e, MOV_X(7, o->addr));
            unspill(e, o->restore, o->nrestore);
            put_helper_refill(e, o->pc, o->insn, o->rflags, retry);
            put(e, B_OP);
            plink(e, e->n - 1, 1, o->join, 0);
            continue;
        }
        unspill(e, o->restore, o->nrestore);
        if (o->kind == OOL_EXIT) {
            e->cur_pc = o->site;
            put_branch_exit(e, o->pc, o->plain); /* branches, and exits for interrupts let through */
            /* With nothing to do on the way, chaining can take the hot part's own branch there:
               a taken branch that goes where it means to, like the guest's, in place of two. */
            if (e->site_direct && !o->nrestore && (e->flat || (e->hot[site_at] & 0x7e000000) != 0x36000000)) {
                e->site[e->nsite - 1].has_cond = 1;
                e->site[e->nsite - 1].cond_at = site_at;
            }
        } else {
            put_helper(e, o->pc, o->insn, o->kind == OOL_SYS ? HV_OP_SYS : o->kind == OOL_MRS ? HV_OP_MRS : HV_OP_MEM);
            put(e, B_OP);
            plink(e, e->n - 1, 1, o->join, 0);
        }
    }
}

/* Find the groups (see Group) in the instructions a block may hold. Only
   instructions whose effect on general registers is plain are walked through:
   ALU and SIMD (writing at most Rd), PC-relative, PAC, prefetches and other
   loads and stores; anything else ends every group. */
static int group_op(const HvMemOp *op) {
    return (op->kind == HV_MEM_SINGLE || op->kind == HV_MEM_PAIR) && !op->align && !op->unpriv &&
           (op->mode == HV_ADDR_OFFSET || op->mode == HV_ADDR_PRE || op->mode == HV_ADDR_POST) &&
           (op->rn == 31 || !is_stolen(op->rn)) && op->bytes <= 64;
}

static void plan_groups(Emit *e, const uint32_t *code, uint32_t max, uint64_t va) {
    e->ngrp = 0;
    memset(e->gid, -1, sizeof e->gid);
    if (e->nogroup || e->slow_mem) return;
    int cur[32];
    int64_t delta[32];
    for (int r = 0; r < 32; r++) cur[r] = -1, delta[r] = 0;
#define KILL(r) (cur[(r) & 31] = -1)
    for (uint32_t i = 0; i < max && i < HV_MAX_BLOCK_INSNS; i++) {
        uint32_t insn = code[i];
        if (i > 0 && (va + 4ull * i == e->m->stop_pc || va + 4ull * i == e->m->stop_pc2)) break;
        HvInsnKind kind = hv_classify(insn);
        if ((e->flags & HV_BLOCK_FPOFF) && kind != HV_INSN_UNDEFINED && is_fp_insn(insn)) break;
        if (kind == HV_INSN_COPY && (insn & 0xfffff01fu) == 0xd503201fu) continue; /* hints (NOP, BTI): no register */
        if (kind == HV_INSN_COPY || kind == HV_INSN_PCREL || kind == HV_INSN_LOAD_LITERAL) {
            KILL(insn);
            continue;
        }
        if (kind == HV_INSN_PAC) {
            KILL(insn), KILL(30), KILL(17), KILL(16);
            continue;
        }
        if (kind == HV_INSN_PREFETCH) continue;
        if (kind != HV_INSN_MEMORY) break;
        HvMemOp op;
        if (hv_mem_decode(insn, &op) != 0) break;
        if (group_op(&op)) {
            int b = op.rn;
            int64_t off = delta[b] + (op.mode == HV_ADDR_POST ? 0 : op.imm);
            Group *g = cur[b] >= 0 ? &e->grp[cur[b]] : NULL;
            if (g) {
                int64_t lo = off < g->lo ? off : g->lo, hi = off + op.bytes > g->hi ? off + op.bytes : g->hi;
                if (hi - lo > GROUP_SPAN || i - g->first > GROUP_INSNS) g = NULL;
                else {
                    g->lo = (int32_t)lo, g->hi = (int32_t)hi, g->last = (uint16_t)i, g->store |= op.store;
                    e->gid[i] = (int8_t)cur[b], e->grole[i] = 2, e->goff[i] = (int32_t)off;
                }
            }
            if (!g && e->ngrp < MAX_GROUPS) {
                cur[b] = (int)e->ngrp;
                delta[b] = 0;
                off = op.mode == HV_ADDR_POST ? 0 : op.imm;
                e->grp[e->ngrp++] = (Group){.lo = (int32_t)off, .hi = (int32_t)(off + op.bytes), .store = op.store,
                                            .base = (uint8_t)b, .first = (uint16_t)i, .last = (uint16_t)i};
                e->gid[i] = (int8_t)cur[b], e->grole[i] = 1, e->goff[i] = (int32_t)off;
            } else if (!g) {
                cur[b] = -1;
            }
            if (op.mode == HV_ADDR_PRE || op.mode == HV_ADDR_POST) delta[b] += op.imm;
            if (op.load && !op.vec) {
                KILL(op.rt);
                if (op.kind == HV_MEM_PAIR) KILL(op.rt2);
            }
            continue;
        }
        /* Another load or store: what it may write. */
        if (op.load && !op.vec) KILL(op.rt), KILL(op.rt2);
        KILL(op.rs);
        if (op.mode == HV_ADDR_PRE || op.mode == HV_ADDR_POST || op.mode == HV_ADDR_REG) KILL(op.rn);
        if (op.kind == HV_MEM_VEC_MULTI || op.kind == HV_MEM_VEC_SINGLE) KILL(op.rn);
    }
#undef KILL
    /* A group needs a second access; then a slot of its own while it lives. */
    uint16_t slot_free[GROUP_SLOTS] = {0};
    for (uint32_t k = 0; k < e->ngrp; k++) {
        Group *g = &e->grp[k];
        g->live = 0;
        if (g->last == g->first) continue;
        for (int s = 0; s < GROUP_SLOTS; s++)
            if (slot_free[s] <= g->first) {
                g->slot = (uint8_t)s, g->live = 1;
                slot_free[s] = (uint16_t)(g->last + 1);
                break;
            }
    }
    /* A failed group runs its instructions plain, through its last access and
       through every later-starting group that runs on past it (whose first
       access the plain copy then makes too). */
    for (uint32_t k = 0; k < e->ngrp; k++) {
        Group *g = &e->grp[k];
        if (!g->live) continue;
        uint16_t end = g->last;
        for (int again = 1; again;) {
            again = 0;
            for (uint32_t j = 0; j < e->ngrp; j++) {
                Group *h = &e->grp[j];
                if (h->live && h->first > g->first && h->first <= end && h->last > end) end = h->last, again = 1;
            }
        }
        g->end = end;
    }
}

/* After the block: each live group's plain copy. Returns -1 if they do not fit. */
static int emit_group_copies(Emit *e, const uint32_t *code, uint64_t va, uint32_t ntrans) {
    e->nogroup = 1;
    for (uint32_t k = 0; k < e->ngrp; k++) {
        Group *g = &e->grp[k];
        if (!g->live || g->first >= ntrans) continue;
        uint32_t end = g->end < ntrans ? g->end : ntrans - 1;
        g->copy_at = e->n;
        for (uint32_t i = g->first; i <= end; i++) {
            if (e->n + WORDS_PER_INSN_MAX * 2 > MAX_WORDS || e->npool + 8 > MAX_LITS || e->nool + 4 > MAX_OOL) return -1;
            e->cur_pc = va + 4ull * i;
            e->cur_i = i;
            e->u_prev = e->u_live = 0;
            if (translate_one(e, code[i], va + 4ull * i)) return -1; /* cannot happen: groups hold no block ends */
        }
        put(e, B_OP);
        patch_branch(&e->w[e->n - 1], (int64_t)e->after[end] - (e->n - 1));
    }
    return 0;
}

static int translate_emit(HvMachine *m, HvBlock *b, const uint32_t *code, uint32_t avail, int nogroup);

int hv_translate_emit(HvMachine *m, HvBlock *b, const uint32_t *code, uint32_t avail) {
    /* HV_GROUPS=1: group accesses (off by default: measured no faster, and the
       plain copies make the code about 15% bigger, so the JIT region fills sooner). */
    if (hv_knob[HV_K_GROUPS] && translate_emit(m, b, code, avail, 0) == 0) return 0;
    return translate_emit(m, b, code, avail, 1);
}

static int translate_emit(HvMachine *m, HvBlock *b, const uint32_t *code, uint32_t avail, int nogroup) {
    Emit *e = &E;
    e->n = e->other_n = e->npool = e->nfix = e->nool = e->nsite = e->nrel = 0;
    e->w = e->hot;
    e->in_cold = 0;
    e->flat = hv_knob[HV_K_TEST2];
    e->overflow = 0;
    e->flags = b->flags;
    e->slow_mem = m->slow_mem || hv_knob[HV_K_PREDSTAT]; /* HV_PREDSTAT (measuring): every access through hv_mem_emulate */
    e->page = b->va & ~((1ull << HV_TLB_PAGE_BITS) - 1);
    e->no_ijc = !m->chain;
    e->m = m;
    e->nogroup = nogroup;
    e->u_live = e->u_prev = 0;

    /* No prologue: hv_enter, chained exits and the indirect-branch cache all
       enter with guest x0 and x1 live. */
    uint32_t max = avail < HV_MAX_BLOCK_INSNS ? avail : HV_MAX_BLOCK_INSNS;
    plan_groups(e, code, max, b->va);
    uint32_t i = 0;
    int ended = 0;
    while (i < max && !ended) {
        uint64_t pc = b->va + 4ull * i;
        if (i > 0 && (pc == m->stop_pc || pc == m->stop_pc2)) break; /* so the dispatcher sees stop_pc at a block start */
        e->cur_pc = pc;
        e->cur_i = i;
        e->u_prev = e->u_live, e->u_live = 0;
        uint32_t n0 = e->n;
        ended = translate_one(e, code[i], pc);
        if (m->stats_interval) {
            HvInsnKind k = hv_classify(code[i]);
            m->stats.kind_insns[k & 15]++;
            m->stats.kind_words[k & 15] += e->n - n0;
        }
        e->after[i] = e->n;
        i++;
        if (e->n + WORDS_PER_INSN_MAX * 2 > MAX_WORDS || e->other_n + WORDS_PER_INSN_MAX * 2 > MAX_WORDS || e->npool + 8 > MAX_LITS || e->nool + 4 > MAX_OOL || e->nrel + 256 > MAX_RELS) break;
    }
    if (!ended) put_branch_exit(e, b->va + 4ull * i, 1);
    b->ninsns = i;
    uint32_t inline_words = e->n;
    if (!e->nogroup && e->ngrp && emit_group_copies(e, code, b->va, i) != 0) return -1;
    cold_on(e);
    emit_ool(e);
    if (m->stats_interval) {
        m->stats.words_inline += inline_words;
        m->stats.words_ool += e->n - (e->flat ? inline_words : 0);
        m->stats.words_pool += e->npool * 2;
    }

    /* Pool after the cold code, 8-byte aligned; both parts start 8-byte aligned. */
    if (e->n & 1) put(e, 0xd503201f);
    uint32_t pool_at = e->n;
    uint32_t ncold = e->n + e->npool * 2;
    cold_off(e);
    if (e->n & 1) put(e, 0xd503201f);
    if (e->flat) e->ncold = 0, e->nhot = ncold;
    else e->ncold = ncold, e->nhot = e->n;
    if (e->nhot > MAX_WORDS || e->ncold > MAX_WORDS || e->overflow) return -1;

    for (uint32_t k = 0; k < e->nfix; k++) {
        LitFix *f = &e->fix[k];
        plink(e, f->at, f->cold, pool_at + f->idx * 2, !e->flat);
    }

    e->pool_at = pool_at;
    return 0;
}

uint32_t hv_translate_last_words(void) { return E.flat ? E.pool_at : E.nhot; } /* hot words (no pool) of this thread's last block */

/* A B, B.cond, CBZ or CBNZ now going off words away, if it reaches. 1 if it does. */
int hv_branch_retarget(uint32_t *w, int64_t off) {
    uint32_t op = *w;
    if ((op & 0x7c000000) == B_OP) {
        if (off < -(1 << 25) || off >= (1 << 25)) return 0;
        *w = (op & 0xfc000000) | (off & 0x3ffffff);
        return 1;
    }
    if (off < -(1 << 18) || off >= (1 << 18)) return 0;
    *w = (op & ~(0x7ffffu << 5)) | (uint32_t)(off & 0x7ffff) << 5;
    return 1;
}

/* Each CPU places code in its own 1 MB at a time. */
static __thread struct {
    HvMachine *m;
    uint64_t gen;
    uint32_t *hot, *cold;
} chunk;

static int chunk_take(HvMachine *m, uint32_t hot, uint32_t cold, uint32_t **hot_at, uint32_t **cold_at) {
    uint64_t gen = __atomic_load_n(&m->jit_gen, __ATOMIC_SEQ_CST);
    if (chunk.m != m || chunk.gen != gen || !chunk.hot || (size_t)(chunk.cold - chunk.hot) < (size_t)hot + cold) {
        size_t want = JIT_CHUNK_WORDS;
        if (want > m->jit.capacity) want = m->jit.capacity;
        if (want < (size_t)hot + cold) want = (size_t)hot + cold;
        uint32_t *p = hv_jit_reserve(&m->jit, want);
        if (!p) {
            chunk.hot = NULL;
            return -1;
        }
        chunk.m = m, chunk.gen = gen, chunk.hot = p, chunk.cold = p + ((want + 1) & ~(size_t)1);
    }
    *hot_at = chunk.hot;
    chunk.hot += hot;
    chunk.cold -= cold;
    *cold_at = chunk.cold;
    return 0;
}

int hv_translate_place(HvMachine *m, HvBlock *b) {
    Emit *e = &E;
    uint32_t pad = hv_knob[HV_K_TEST1] & ~1u; /* HV_K1: words of nothing after each block's hot part, to see what density is worth */
    uint32_t *dst, *cold;
    if (chunk_take(m, e->nhot + pad, e->ncold, &dst, &cold) != 0) return -1;
    for (uint32_t k = 0; k < e->nrel; k++) {
        uint32_t at = e->rel[k].at, target = e->rel[k].target;
        if (e->rel[k].from_cold) patch_branch(&e->cold[at], (dst + target) - (cold + at));
        else patch_branch(&e->hot[at], (cold + target) - (dst + at));
    }
    /* Exits in the page to blocks that exist already (or back to this one's
       start) are chained now, in the copy about to be written: no trip through
       the dispatcher to do it. As chain() does for the rest. */
    for (uint32_t k = 0; k < e->nsite && !m->trace; k++) { /* tracing logs every block entry */
        uint32_t slot = e->site[k].slot;
        uint64_t t = e->site[k].target;
        if (m->slot_xa[slot] || e->site[k].flags != b->flags || t == m->stop_pc || t == m->stop_pc2) continue; /* the dispatcher stops there */
        const uint32_t *code = NULL;
        if (t == b->va) code = dst;
        else {
            uint64_t pa = (b->pa & ~((1ull << HV_TLB_PAGE_BITS) - 1)) | (t & ((1ull << HV_TLB_PAGE_BITS) - 1));
            HvBlock *tb = hv_block_find(m, hv_block_hash(t, pa, b->flags), t, pa, b->flags, b->gen);
            /* Only to this CPU's own blocks: one another CPU just wrote must be
               entered through an isb first (the dispatcher, a light lookup). */
            if (tb && tb->maker == b->maker) code = tb->code;
        }
        if (!code) continue;
        m->slots[slot] = (uint64_t)(uintptr_t)code;
        uint32_t at = e->site[k].at;
        int64_t off = code - ((e->site[k].cold ? cold : dst) + at);
        if (off >= -(1 << 25) && off < (1 << 25)) (e->site[k].cold ? e->cold : e->hot)[at] = 0x14000000u | (uint32_t)(off & 0x3ffffff);
        if (e->site[k].has_cond) hv_branch_retarget(&e->hot[e->site[k].cond_at], code - (dst + e->site[k].cond_at));
        if (m->stats_interval) m->stats.chained_early++;
    }
    if (e->flat) {
        hv_jit_write(&m->jit, dst, e->hot, e->pool_at * 4);
        hv_jit_write(&m->jit, dst + e->pool_at, e->pool, e->npool * 8);
    } else {
        hv_jit_write(&m->jit, dst, e->hot, e->nhot * 4);
        hv_jit_write(&m->jit, cold, e->cold, e->pool_at * 4);
        hv_jit_write(&m->jit, cold + e->pool_at, e->pool, e->npool * 8);
        hv_jit_finish(&m->jit, cold, e->ncold);
    }
    hv_jit_finish(&m->jit, dst, e->nhot);
    b->code = dst;
    for (uint32_t k = 0; k < e->nsite; k++) {
        m->slot_site[e->site[k].slot] = (e->site[k].cold ? cold : dst) + e->site[k].at;
        m->slot_stub[e->site[k].slot] = (e->site[k].stub_cold ? cold : dst) + e->site[k].stub;
        m->slot_cond[e->site[k].slot] = e->site[k].has_cond ? dst + e->site[k].cond_at : NULL;
    }
    return 0;
}

int hv_translate_block(HvMachine *m, HvBlock *b, const uint32_t *code, uint32_t avail) {
    if (hv_translate_emit(m, b, code, avail) != 0) return -1;
    return hv_translate_place(m, b);
}
