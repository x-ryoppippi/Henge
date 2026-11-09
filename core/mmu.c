#include "mmu.h"

#include <stdlib.h>
#include <string.h>

#include "exception.h"
#include "machine.h"

/* VMSAv8-64 stage 1 translation for the EL1&0 regime, 4 KB, 16 KB and 64 KB
   granules, 48-bit VA and PA. There is no stage 2: HengeVM is the hypervisor. */

#define PAGE_MASK ((1ull << HV_TLB_PAGE_BITS) - 1)
#define OA_MASK 0x0000ffffffffffffull

static int granule_bits(uint32_t tg, int ttbr1) {
    if (ttbr1) return tg == 1 ? 14 : tg == 3 ? 16 : 12; /* TG1: 01 16K, 10 4K, 11 64K */
    return tg == 2 ? 14 : tg == 1 ? 16 : 12;            /* TG0: 00 4K, 01 64K, 10 16K */
}

/* This file is built with -mgeneral-regs-only, and the walk and the TLB fill
   call nothing in libc: hv_tlb_refill runs them from translated code without
   saving the guest's SIMD registers, which libc's memory routines would use. */
static HvRam *ram_of(HvMachine *m, uint64_t pa) {
    for (int i = 0; i < m->nram; i++)
        if (pa - m->ram[i].pa < m->ram[i].size) return &m->ram[i];
    return NULL;
}

/* A descriptor: RAM, 8 bytes, aligned. Device memory as a page table is not supported. */
static int read_desc(HvMachine *m, uint64_t pa, uint64_t *out) {
    HvRam *r = ram_of(m, pa);
    if (!r || pa + 8 - r->pa > r->size) return -1;
    *out = __atomic_load_n((const uint64_t *)(const void *)(r->host + (pa - r->pa)), __ATOMIC_RELAXED);
    return 0;
}

void hv_mmu_walk(HvCpu *cpu, uint64_t va, HvAccess acc, int el, HvWalk *out) {
    *out = (HvWalk){0};
    uint64_t sctlr = cpu->sctlr_el1;
    if (!(sctlr & 1)) {
        /* MMU off: flat, everything allowed. Data is Device-nGnRnE, fetches Normal. */
        out->pa = va;
        out->r = out->w = out->x = 1;
        out->block_size = 1ull << 30;
        out->attr = acc == HV_ACC_EXEC ? 0xff : 0x00;
        return;
    }

    uint64_t tcr = cpu->tcr_el1;
    int hi = (va >> 55) & 1;
    int tbi = hi ? (tcr >> 38) & 1 : (tcr >> 37) & 1;
    int tbid = hi ? (tcr >> 52) & 1 : (tcr >> 51) & 1;
    if (acc == HV_ACC_EXEC && tbid) tbi = 0;
    uint32_t tsz = hi ? (tcr >> 16) & 63 : tcr & 63;
    int epd = hi ? (tcr >> 23) & 1 : (tcr >> 7) & 1;
    int hpd = hi ? (tcr >> 42) & 1 : (tcr >> 41) & 1;
    int g = granule_bits(hi ? (tcr >> 30) & 3 : (tcr >> 14) & 3, hi);
    if (tsz < 16) tsz = 16;
    if (tsz > 39) tsz = 39;
    int inputsize = 64 - (int)tsz;

    /* Bits above the input size must all match bit 55 (or bit 63 without TBI). */
    int top = tbi ? 55 : 63;
    uint64_t mask = (top - inputsize + 1 >= 64 ? ~0ull : ((1ull << (top - inputsize + 1)) - 1)) << inputsize;
    if ((va & mask) != (hi ? mask : 0) || epd) {
        out->fsc = HV_FSC_TRANSLATION(0);
        return;
    }

    int stride = g - 3;
    int levels = (inputsize - g + stride - 1) / stride;
    int level = 4 - levels;
    uint64_t base = (hi ? cpu->ttbr1_el1 : cpu->ttbr0_el1) & OA_MASK & ~1ull;
    int ap_table = 0, pxn_table = 0, uxn_table = 0;

    for (;; level++) {
        int shift = g + stride * (3 - level);
        int idx_bits = level == 4 - levels ? inputsize - shift : stride;
        uint64_t idx = (va >> shift) & ((1ull << idx_bits) - 1);
        uint64_t desc;
        if (read_desc(cpu->m, base + idx * 8, &desc) != 0) {
            out->fsc = 0x14 + level; /* synchronous external abort on a table walk */
            out->s1ptw = 1;
            return;
        }
        if (!(desc & 1)) {
            out->fsc = HV_FSC_TRANSLATION(level);
            return;
        }
        if (level < 3 && (desc & 2)) { /* table */
            base = desc & OA_MASK & ~((1ull << g) - 1);
            if (!hpd) {
                pxn_table |= (desc >> 59) & 1;
                uxn_table |= (desc >> 60) & 1;
                ap_table |= (desc >> 61) & 3;
            }
            continue;
        }
        if (level == 3 && !(desc & 2)) {
            out->fsc = HV_FSC_TRANSLATION(level);
            return;
        }
        if (level < 3) { /* block: 4K at levels 1-2, 16K and 64K at level 2 */
            if (level == 0 || (g != 12 && level == 1)) {
                out->fsc = HV_FSC_TRANSLATION(level);
                return;
            }
        }
        uint64_t size = 1ull << shift;
        out->block_size = size;
        out->pa = (desc & OA_MASK & ~(size - 1)) | (va & (size - 1));
        out->attr = (cpu->mair_el1 >> (((desc >> 2) & 7) * 8)) & 0xff;
        if (!(desc & (1u << 10))) {
            out->fsc = HV_FSC_ACCESS(level);
            return;
        }
        int ap = (desc >> 6) & 3;
        int uxn = ((desc >> 54) & 1) | uxn_table;
        int pxn = ((desc >> 53) & 1) | pxn_table;
        int el0_access = (ap & 1) && !(ap_table & 1);
        int read_only = (ap & 2) || (ap_table & 2);
        if (el == 0) {
            out->r = el0_access;
            out->w = el0_access && !read_only;
            out->x = !uxn;
        } else {
            int pan = (cpu->pstate_extra & HV_PSTATE_PAN) && el0_access;
            out->r = !pan;
            out->w = !pan && !read_only;
            /* EL1 never executes what EL0 can write. */
            out->x = !pxn && !(el0_access && !read_only);
        }
        if ((sctlr >> 19) & 1 && !read_only && (el == 0 ? el0_access : 1)) out->x = 0; /* WXN */
        int ok = acc == HV_ACC_READ ? out->r : acc == HV_ACC_WRITE ? out->w : out->x;
        if (!ok) out->fsc = HV_FSC_PERMISSION(level);
        return;
    }
}

int hv_tlb_tmajor;

void hv_tlb_flush(HvCpu *cpu) {
    if (cpu->m) cpu->m->stats.tlb_flushes++;
    if (cpu->m) cpu->m->stats.reset_why[0]++;
    hv_ijc_reset(cpu, 0); /* mappings changed: branch targets cached by VA are no longer trusted */
    hv_ijc_reset(cpu, 1);
    /* Only the entries filled since the last flush can be valid (both logs, unless one ran over). */
    uint32_t nu = cpu->nuser_fills, nk = cpu->nkernel_fills;
    if (nu <= HV_USER_FILLS && nk <= HV_KERNEL_FILLS) {
        for (uint32_t k = 0; k < nu; k++) memset(&cpu->tlb[cpu->user_fills[k]], 0x80, sizeof cpu->tlb[0]);
        for (uint32_t k = 0; k < nk; k++) memset(&cpu->tlb[cpu->kernel_fills[k]], 0x80, sizeof cpu->tlb[0]);
    } else {
        memset(cpu->tlb, 0x80, sizeof cpu->tlb);
    }
    cpu->nuser_fills = 0;
    cpu->nkernel_fills = 0;
    hv_tlb_stash_drop(cpu, -1);
}

/* Clear the tables of one half (user tables are the even flags values, kernel ones odd). */
static uint32_t hint_slot(uint64_t key) {
    return (uint32_t)((key * 0x9e3779b97f4a7c15ull) >> 52);
}

static void ijc_clear(HvCpu *cpu, int kernel) {
    if (!cpu->pc_state) return;
    memset(cpu->pc_state->ijc_hint[kernel], 0, sizeof cpu->pc_state->ijc_hint[kernel]);
    for (int t = kernel; t < HV_IJC_TABLES; t += 2)
        memset(cpu->pc_state->ijc + (size_t)t * (HV_IJC_TABLE_BYTES / 8), 0, HV_IJC_TABLE_BYTES);
}

/* ASIDs live in TTBR0 unless TCR.A1 puts them in TTBR1; then every process looks alike to us. */
static int current_asid(HvCpu *cpu) {
    return (cpu->tcr_el1 >> 22) & 1 ? -1 : (int)(cpu->ttbr0_el1 >> 48);
}

/* ---- the (user id, page) set ---- */

static uint64_t ucode_key(uint32_t cid, uint64_t va) {
    return ((uint64_t)cid << 40 | ((va >> HV_TLB_PAGE_BITS) & 0xffffffffffull)) + 1; /* never 0, the empty slot */
}

static uint32_t ucode_slot(uint64_t key) {
    return (uint32_t)((key * 0x9e3779b97f4a7c15ull) >> (64 - HV_UCODE_BITS));
}

static void ucode_clear(HvPerCpu *m) {
    if (m) m->last_fill[0] = 0;
    if (m && m->ucode) memset(m->ucode, 0, (sizeof *m->ucode) << HV_UCODE_BITS);
    if (m) m->ucode_count = 0;
}

static int ucode_has(HvPerCpu *m, uint32_t cid, uint64_t va) {
    uint64_t key = ucode_key(cid, va);
    for (uint32_t i = ucode_slot(key);; i = (i + 1) & ((1u << HV_UCODE_BITS) - 1)) {
        if (m->ucode[i] == key) return 1;
        if (!m->ucode[i]) return 0;
    }
}

void hv_ucode_add(HvCpu *cpu, uint64_t va) {
    HvPerCpu *m = cpu->pc_state;
    uint32_t cid = (uint32_t)(cpu->ijc_salt[0] >> 48);
    if (!m || !cid) return;
    uint64_t key = ucode_key(cid, va);
    uint32_t i = ucode_slot(key);
    for (; m->ucode[i]; i = (i + 1) & ((1u << HV_UCODE_BITS) - 1))
        if (m->ucode[i] == key) return;
    if (m->ucode_count >= (1u << HV_UCODE_BITS) / 2) { /* half full: start the user half over */
        ucode_clear(m);
        hv_ijc_reset(cpu, 0);
        hv_ijc_switch(cpu);
        cid = (uint32_t)(cpu->ijc_salt[0] >> 48);
        key = ucode_key(cid, va);
        i = ucode_slot(key);
    }
    m->ucode[i] = key;
    m->ucode_count++;
}

/* A user id no tag in the tables uses yet; when they run out, empty the user tables and start over. */
static uint16_t fresh_cid(HvCpu *cpu) {
    if (cpu->next_cid == 0 || cpu->next_cid > 0xffff) {
        if (cpu->pc_state && cpu->pc_state->cid_chunks) memset(cpu->pc_state->cid_chunks, 0, 65536 * sizeof *cpu->pc_state->cid_chunks);
        if (cpu->pc_state) cpu->pc_state->last_fill[0] = cpu->pc_state->last_fill[1] = 0;
        ijc_clear(cpu, 0);
        ucode_clear(cpu->pc_state);
        memset(cpu->asid_cid, 0, sizeof cpu->asid_cid);
        cpu->nasid_log = 0;
        cpu->next_cid = 1;
    }
    if (cpu->pc_state && cpu->pc_state->cid_chunks) memset(cpu->pc_state->cid_chunks[cpu->next_cid], 0, sizeof cpu->pc_state->cid_chunks[0]);
    return (uint16_t)cpu->next_cid++;
}

/* Empty a half of the return tables by giving it a new salt. From the CPU's own thread;
   the salts run out here, and it is the only one that may clear the tables. */
static void ret_new(HvCpu *cpu, int kernel) {
    if (cpu->ret_salt[kernel] >> 48 >= 0xf000) {
        hv_ret_clear(cpu); /* both halves: nothing is left under any salt */
        return;
    }
    __atomic_add_fetch(&cpu->ret_salt[kernel], 1ull << 48, __ATOMIC_RELAXED);
}

/* The same from another CPU's thread. 0 if it is too near running out to do it here. */
static int ret_new_remote(HvCpu *cpu, int kernel) {
    if (__atomic_load_n(&cpu->ret_salt[kernel], __ATOMIC_RELAXED) >> 48 >= 0xff00) return 0;
    __atomic_add_fetch(&cpu->ret_salt[kernel], 1ull << 48, __ATOMIC_RELAXED);
    return 1;
}

void hv_ijc_switch(HvCpu *cpu) {
    int asid = current_asid(cpu);
    ret_new(cpu, 0); /* a switch forgets the old process's returns: its frames miss once at worst */
    if (asid < 0) { /* no ASIDs to tell processes apart: a switch forgets the old one */
        cpu->ijc_salt[0] = (uint64_t)fresh_cid(cpu) << 48;
        return;
    }
    uint64_t base = cpu->ttbr0_el1 & OA_MASK & ~1ull;
    if (!cpu->asid_cid[asid] || cpu->asid_base[asid] != base) {
        cpu->asid_cid[asid] = fresh_cid(cpu);
        cpu->asid_base[asid] = base;
        cpu->cid_asid[cpu->asid_cid[asid]] = (uint16_t)asid;
        if (cpu->nasid_log < HV_ASID_LOG) cpu->asid_log[cpu->nasid_log] = (uint16_t)asid;
        if (cpu->nasid_log <= HV_ASID_LOG) cpu->nasid_log++;
    }
    cpu->ijc_salt[0] = (uint64_t)cpu->asid_cid[asid] << 48;
}

void hv_ijc_drop_asid(HvCpu *cpu, uint16_t asid) {
    if (cpu->m) cpu->m->stats.ijc_resets++;
    cpu->asid_cid[asid] = 0;
    int cur = current_asid(cpu);
    if (cur == asid || cur < 0) hv_ijc_switch(cpu); /* without ASIDs, any drop may be about the current process */
}

void hv_ijc_reset(HvCpu *cpu, int kernel) {
    if (cpu->m && cpu->m->stats_interval) cpu->m->stats.ijc_reset_kind[kernel ? 1 : 0]++;
    ret_new(cpu, kernel);
    if (cpu->pc_state) cpu->pc_state->last_fill[0] = cpu->pc_state->last_fill[1] = 0;
    /* Every entry of the half is dead now (new ids or kernel generation): so are their hints. */
    if (cpu->pc_state) memset(cpu->pc_state->ijc_hint[kernel], 0, sizeof cpu->pc_state->ijc_hint[kernel]);
    if (cpu->m) {
        cpu->m->stats.ijc_resets++;
        memset(cpu->pc_state->ijc_pages[kernel], 0, sizeof cpu->pc_state->ijc_pages[kernel]);
    }
    if (kernel) {
        /* Kernel VAs have 0xffff in bits 63..48, so the salt must not make that 0 (the empty tag). */
        if (++cpu->kernel_gen >= 0xffff) {
            ijc_clear(cpu, 1);
            cpu->kernel_gen = 1;
        }
        cpu->ijc_salt[1] = (uint64_t)cpu->kernel_gen << 48;
        return;
    }
    /* Every process gets a new user id: clear the ids given since the last time (all, if the log ran over).
       The (id, page) set keeps the dead ids' pages (no live id matches them) until it is half full. */
    if (cpu->nasid_log <= HV_ASID_LOG)
        for (uint32_t k = 0; k < cpu->nasid_log; k++) cpu->asid_cid[cpu->asid_log[k]] = 0;
    else
        memset(cpu->asid_cid, 0, sizeof cpu->asid_cid);
    cpu->nasid_log = 0;
    if (cpu->pc_state) cpu->pc_state->last_fill[0] = 0;
    hv_ijc_switch(cpu);
}

/* TTBR0 changed or an ASID was dropped: only user-half (TTBR0) translations go;
   the kernel's TTBR1 entries stay. */
void hv_tlb_flush_user(HvCpu *cpu) {
    if (cpu->m) cpu->m->stats.tlb_flushes++;
    uint32_t n = cpu->nuser_fills;
    cpu->nuser_fills = 0;
    if (n <= HV_USER_FILLS) {
        for (uint32_t k = 0; k < n; k++) {
            HvTlbEntry *e = &cpu->tlb[0] + cpu->user_fills[k];
            uint64_t tag = e->read_tag != HV_TLB_INVALID ? e->read_tag + HV_TLB_ADJ : e->exec_tag != HV_TLB_INVALID ? hv_exec_key(e->exec_tag, e->addend) : e->write_tag + HV_TLB_ADJ;
            if (e->read_tag != HV_TLB_INVALID || e->exec_tag != HV_TLB_INVALID || e->write_tag != HV_TLB_INVALID) if (!((tag >> 55) & 1)) memset(e, 0x80, sizeof *e); /* the slot may hold a kernel page by now */
        }
        return;
    }
    for (int t = 0; t < 3; t++)
        for (uint32_t i = 0; i < HV_TLB_SIZE; i++) {
            HvTlbEntry *e = hv_tlb_ent(cpu, i, t);
            uint64_t tag = e->read_tag != HV_TLB_INVALID ? e->read_tag + HV_TLB_ADJ : e->exec_tag != HV_TLB_INVALID ? hv_exec_key(e->exec_tag, e->addend) : e->write_tag + HV_TLB_ADJ;
            if (e->read_tag != HV_TLB_INVALID || e->exec_tag != HV_TLB_INVALID || e->write_tag != HV_TLB_INVALID) if (!((tag >> 55) & 1)) memset(e, 0x80, sizeof *e);
        }
}

/* A TLBI for a big user range: only the user entries filled since the last
   flush that fall in it (all of them if there were too many to list). */
void hv_tlb_flush_user_range(HvCpu *cpu, uint64_t va, uint64_t len) {
    uint32_t n = cpu->nuser_fills;
    if (n > HV_USER_FILLS) {
        hv_tlb_flush_user(cpu);
        return;
    }
    if (cpu->m) if (cpu->m->stats_interval) cpu->m->stats.tlb_range_flushes++;
    uint64_t lo = va & 0x00ffffffffffffffull;
    for (uint32_t k = 0; k < n; k++) {
        HvTlbEntry *e = &cpu->tlb[0] + cpu->user_fills[k];
        if (e->read_tag == HV_TLB_INVALID && e->exec_tag == HV_TLB_INVALID && e->write_tag == HV_TLB_INVALID) continue;
        uint64_t tag = e->read_tag != HV_TLB_INVALID ? e->read_tag + HV_TLB_ADJ : e->exec_tag != HV_TLB_INVALID ? hv_exec_key(e->exec_tag, e->addend) : e->write_tag + HV_TLB_ADJ;
        if ((tag >> 55) & 1) continue; /* the slot holds a kernel page by now */
        /* Any overlap (a 4 KB-granule range may start inside the page); tagged addresses: by the low 56 bits. */
        uint64_t pg = tag & 0x00ffffffffffffffull;
        if (pg < lo + len && pg + PAGE_MASK + 1 > lo) memset(e, 0x80, sizeof *e);
    }
}

/* A big range of kernel VAs. A full flush here also dropped every process's entries and both halves of the branch
   caches, and macOS 27 sends ranges like this hundreds of times a second (a phone running it did 120-260 full flushes a
   second, Sequoia none). Up to 64 MB page by page, as small ranges are; past that, a scan for kernel entries in the range
   and the kernel half of the branch caches. */
void hv_tlb_flush_kernel_range(HvCpu *cpu, uint64_t va, uint64_t len) {
    if (cpu->m && cpu->m->stats_interval) cpu->m->stats.kernel_ranges++;
    uint64_t first = va & ~(uint64_t)PAGE_MASK;
    if (len <= (4096ull << HV_TLB_PAGE_BITS)) {
        for (uint64_t p = first; p < va + len; p += 1ull << HV_TLB_PAGE_BITS) hv_tlb_flush_va_asid(cpu, p, -1);
        return;
    }
    uint64_t lo = first & 0x00ffffffffffffffull, hi = (va + len) & 0x00ffffffffffffffull; /* tagged addresses: by the low 56 bits */
    for (int t = 0; t < 3; t++)
        for (uint32_t i = 0; i < HV_TLB_SIZE; i++) {
            HvTlbEntry *e = hv_tlb_ent(cpu, i, t);
            if (e->read_tag == HV_TLB_INVALID && e->exec_tag == HV_TLB_INVALID && e->write_tag == HV_TLB_INVALID) continue;
            uint64_t tag = e->read_tag != HV_TLB_INVALID ? e->read_tag + HV_TLB_ADJ : e->exec_tag != HV_TLB_INVALID ? hv_exec_key(e->exec_tag, e->addend) : e->write_tag + HV_TLB_ADJ;
            if (!((tag >> 55) & 1)) continue; /* a process's page */
            uint64_t pg = tag & 0x00ffffffffffffffull;
            if (pg < hi && pg + PAGE_MASK + 1 > lo) memset(e, 0x80, sizeof *e);
        }
    hv_ijc_reset(cpu, 1);
}

/* A page just got translated code: writes to it must take the slow path from now
   on, so drop just the write permission of the entries that map it. */
static int maps_writable(const HvTlbEntry *e, const uint8_t *host_page) {
    return e->write_tag != HV_TLB_INVALID && (const uint8_t *)(uintptr_t)(e->write_tag + HV_TLB_ADJ + e->addend) == host_page;
}

/* RAM page pg just became code: drop write permission from cpu's entries for it. */
void hv_tlb_protect_code_page(HvCpu *cpu, HvRam *ram, uint64_t pg) {
    uint64_t *hint = &ram->whint[(size_t)cpu->index * ram->npages + pg];
    uint64_t h = __atomic_exchange_n(hint, 0, __ATOMIC_SEQ_CST);
    const uint8_t *host_page = ram->host + (pg << HV_TLB_PAGE_BITS);
    if (cpu->m) if (cpu->m->stats_interval) cpu->m->stats.protect_kind[h & HV_WHINT_MANY ? 2 : h ? 1 : 0]++;
    if (h & HV_WHINT_MANY) {
        hv_tlb_protect_code(cpu, host_page);
        return;
    }
    for (int k = 0; k < 3; k++) {
        uint64_t c = h >> (16 * k) & 0xffff;
        if (!c) continue;
        HvTlbEntry *e = &cpu->tlb[0] + (c - 1);
        if (maps_writable(e, host_page)) e->write_tag = HV_TLB_INVALID;
    }
}

void hv_tlb_protect_code(HvCpu *cpu, const uint8_t *host_page) {
    for (int t = 0; t < 3; t++)
        for (uint32_t i = 0; i < HV_TLB_SIZE; i++) {
            HvTlbEntry *e = hv_tlb_ent(cpu, i, t);
            if (e->write_tag != HV_TLB_INVALID && (const uint8_t *)(uintptr_t)(e->write_tag + HV_TLB_ADJ + e->addend) == host_page)
                e->write_tag = HV_TLB_INVALID;
        }
}

/* Forget one salt's cached targets in one 16 KB page. A page's 4096 possible
   targets have 4096 known slots in each table, so only those are looked at. */
/* local: cpu itself (it may empty the page's hint slot afterwards). */
static void ijc_clear_page(HvCpu *cpu, uint64_t page, int kernel, uint64_t salt, int local) {
    HvPerCpu *m = cpu->pc_state;
    uint64_t key = (page ^ salt) | 1;
    __typeof__(m->ijc_hint[0][0]) *h = &m->ijc_hint[kernel][hint_slot(key)];
    uint64_t hk = __atomic_load_n(&h->key, __ATOMIC_RELAXED);
    /* Another key's slot means this one has filled nothing since its own drop or the last reset. */
    uint64_t bits = hk == key ? __atomic_load_n(&h->bits, __ATOMIC_RELAXED) : hk == ~0ull ? ~0ull : 0;
    if (!bits) return;
    for (int t = kernel; t < HV_IJC_TABLES; t += 2) {
        if (!(m->ijc_tables_used >> t & 1)) continue;
        uint64_t *tab = m->ijc + (size_t)t * (HV_IJC_TABLE_BYTES / 8);
        for (uint32_t piece = 0; piece < 64; piece++) {
            if (!(bits >> piece & 1)) continue;
            for (uint32_t o = piece * 64; o < piece * 64 + 64; o++) {
                uint64_t tag = (page | (uint64_t)o << 2) ^ salt;
                uint64_t *ent = tab + (size_t)hv_ijc_index(tag) * 2;
                if (ent[0] == tag) ent[0] = 0;
                if (ent[2] == tag) ent[2] = 0;
            }
        }
    }
    if (local && hk == key) h->key = 0, h->bits = 0;
}

/* The page may hold cached branch targets even if its TLB entry is gone: forget
   them, if the filters say it has any. */
/* remote: another CPU doing it (hv_tlbi_drop_remote_va): returns 0 instead of
   starting the user half over, which only cpu itself may do. */
static int ijc_drop_va(HvCpu *cpu, uint64_t va, int asid, int remote) {
    int kernel = (va >> 55) & 1;
    HvMachine *m = cpu->m;
    HvPerCpu *p = cpu->pc_state;
    {   /* A return may lead into the page. For the kernel only if it is code the CPU has run;
           xnu remaps data all the time and its code hardly ever. For a process only if it is the
           one running: a switch forgets the returns of the others anyway. */
        uint64_t page = va & ~((1ull << HV_TLB_PAGE_BITS) - 1);
        uint32_t sl = (uint32_t)(va >> HV_TLB_PAGE_BITS) & (HV_TLB_SIZE - 1);
        const HvTlbEntry *t1 = hv_tlb_ent(cpu, sl, 1), *t2 = hv_tlb_ent(cpu, sl, 2);
        int some = kernel ? t1->exec_tag == hv_exec_key(page, t1->addend) || t2->exec_tag == hv_exec_key(page, t2->addend) : asid < 0 || asid == current_asid(cpu) || current_asid(cpu) < 0;
        if (some) {
            if (!remote) ret_new(cpu, kernel);
            else if (!ret_new_remote(cpu, kernel)) return 0;
        }
    }
    if (m && p) {
        int drop = 0;
        if (kernel) {
            uint32_t k = hv_ijc_key(va, cpu->ijc_salt[1]);
            drop = p->ijc_keyed[k >> 6] >> (k & 63) & 1;
        } else if (asid < 0) { /* every address space: any process's targets in the page count */
            uint32_t h = (uint32_t)(va >> HV_TLB_PAGE_BITS) & 0xffff;
            drop = p->ijc_pages[0][h >> 6] >> (h & 63) & 1;
        } else {
            uint64_t salt = current_asid(cpu) < 0 ? cpu->ijc_salt[0] : (uint64_t)cpu->asid_cid[asid] << 48;
            drop = (salt >> 48) && ucode_has(p, (uint32_t)(salt >> 48), va);
        }
        if (drop) {
            if (remote && !kernel && asid < 0) return 0;
            m->stats.reset_why[kernel ? 3 : 2]++;
            uint64_t page = va & ~((1ull << HV_TLB_PAGE_BITS) - 1);
            if (kernel) ijc_clear_page(cpu, page, 1, cpu->ijc_salt[1], !remote);
            else if (asid < 0) m->stats.reset_why[5]++, hv_ijc_reset(cpu, 0); /* every address space: their slots differ, so start the user half over */
            else ijc_clear_page(cpu, page, 0, current_asid(cpu) < 0 ? cpu->ijc_salt[0] : (uint64_t)cpu->asid_cid[asid] << 48, !remote);
        }
    }
    return 1;
}

void hv_ijc_drop_va(HvCpu *cpu, uint64_t va, int asid) {
    ijc_drop_va(cpu, va, asid, 0);
}

static void stash_drop_va(HvPerCpu *p, uint64_t va, int asid);

void hv_tlb_flush_va_asid(HvCpu *cpu, uint64_t va, int asid) {
    /* Tagged addresses may sit in any slot's tag byte, so check by the low 56 bits. */
    uint32_t idx = (va >> HV_TLB_PAGE_BITS) & (HV_TLB_SIZE - 1);
    hv_ijc_drop_va(cpu, va, asid);
    if (!((va >> 55) & 1)) hv_tlb_stash_drop(cpu, asid);
    for (int el = 0; el < 3; el++) {
        HvTlbEntry *e = hv_tlb_ent(cpu, idx, el);
        memset(e, 0x80, sizeof *e);
    }
}

/* One page (a TLBI by VA): as hv_tlb_flush_va_asid, but in the stashes only that page's entries go,
   not every entry of the address space (macOS 27 sends tens of thousands a second, and each made a
   process refill its whole TLB the next time it ran). */
void hv_tlb_flush_va_page(HvCpu *cpu, uint64_t va, int asid) {
    uint32_t idx = (va >> HV_TLB_PAGE_BITS) & (HV_TLB_SIZE - 1);
    hv_ijc_drop_va(cpu, va, asid);
    if (!((va >> 55) & 1) && cpu->pc_state) stash_drop_va(cpu->pc_state, va, asid);
    for (int el = 0; el < 3; el++) {
        HvTlbEntry *e = hv_tlb_ent(cpu, idx, el);
        memset(e, 0x80, sizeof *e);
    }
}

/* A TLBI by range too big to do page by page in the TLB (which then drops its
   user half): forget cached branch targets only in the range's code pages. */
void hv_ijc_drop_range(HvCpu *cpu, uint64_t va, uint64_t len, int asid) {
    HvMachine *m = cpu->m;
    if (!m) return;
    if (asid < 0) { /* every address space */
        m->stats.reset_why[5]++;
        hv_ijc_reset(cpu, 0);
        return;
    }
    if (asid == current_asid(cpu) || current_asid(cpu) < 0) ret_new(cpu, 0);
    uint64_t salt = current_asid(cpu) < 0 ? cpu->ijc_salt[0] : (uint64_t)cpu->asid_cid[asid] << 48;
    uint32_t cid = (uint32_t)(salt >> 48);
    if (!cid) return; /* nothing cached for it */
    uint64_t first = va >> HV_TLB_PAGE_BITS, last = (va + len - 1) >> HV_TLB_PAGE_BITS;
    {   /* nothing cached in the chunks the range covers: nothing to drop */
        uint64_t c0 = va >> 25, c1 = (va + len - 1) >> 25;
        int any = c1 - c0 >= 255;
        for (uint64_t c = c0; !any && c <= c1; c++)
            any = cpu->pc_state->cid_chunks[cid][(c & 255) >> 6] >> (c & 63) & 1;
        if (!any) {
            if (m->stats_interval) m->stats.range_spared++;
            return;
        }
    }
    /* Page by page, but only in the range's 32 MB chunks that hold cached targets (a big unmapping, a heap or a
       file, mostly holds none); past 4096 such pages, scanning costs more than the cold start: drop the address space. */
    uint64_t c0 = va >> 25, c1 = (va + len - 1) >> 25, n = 0;
    int wide = c1 - c0 >= 255; /* the chunk bits wrap at 8 GB */
    for (uint64_t c = c0; !wide && c <= c1 && n < 4096; c++)
        if (cpu->pc_state->cid_chunks[cid][(c & 255) >> 6] >> (c & 63) & 1) {
            uint64_t a = c << 11 > first ? c << 11 : first, b = ((c + 1) << 11) - 1 < last ? ((c + 1) << 11) - 1 : last;
            n += b - a + 1;
        }
    if (wide || n >= 4096) {
        if (m->stats_interval) m->stats.ijc_reset_kind[3]++;
        hv_ijc_drop_asid(cpu, (uint16_t)asid);
        return;
    }
    for (uint64_t c = c0; c <= c1; c++) {
        if (!(cpu->pc_state->cid_chunks[cid][(c & 255) >> 6] >> (c & 63) & 1)) continue;
        uint64_t a = c << 11 > first ? c << 11 : first, b = ((c + 1) << 11) - 1 < last ? ((c + 1) << 11) - 1 : last;
        for (uint64_t p = a; p <= b; p++)
            if (ucode_has(cpu->pc_state, cid, p << HV_TLB_PAGE_BITS)) {
                m->stats.reset_why[2]++;
                ijc_clear_page(cpu, p << HV_TLB_PAGE_BITS, 0, salt, 1);
            }
    }
}

void hv_tlb_flush_va(HvCpu *cpu, uint64_t va) {
    hv_tlb_flush_va_asid(cpu, va, -1);
}

/* A writable entry just went into the TLB at code (slot << 2 | table) for RAM
   page pg. Another CPU may have just translated code from the page, setting
   code_pages and then dropping write tags in every TLB: if that came before
   this entry was there, it must not stay writable. */
static void stash_drop(HvPerCpu *p, int asid);

/* Say where, so that the page becoming code drops just this entry (or, if
   this CPU now maps it writable twice, makes the protector scan). The check
   that must follow, after a full fence, is write_guard's last two lines. */
static void write_hint(HvCpu *cpu, HvRam *ram, uint64_t pg, uint32_t code) {
    uint64_t *hint = &ram->whint[(size_t)cpu->index * ram->npages + pg];
    uint64_t mine = code + 1, old = __atomic_load_n(hint, __ATOMIC_RELAXED), now = old;
    const uint8_t *host_page = ram->host + (pg << HV_TLB_PAGE_BITS);
    if (!(old & HV_WHINT_MANY)) {
        /* Up to three entries: this one, in a free place or one whose entry maps
           something else by now; beyond that the protector scans. */
        int have = 0, place = -1;
        for (int k = 0; k < 3; k++) {
            uint64_t c = old >> (16 * k) & 0xffff;
            if (c == mine) have = 1;
            else if (place < 0 && (!c || !maps_writable(&cpu->tlb[0] + (c - 1), host_page))) place = k;
        }
        if (!have) now = place >= 0 ? (old & ~(0xffffull << (16 * place))) | mine << (16 * place) : HV_WHINT_MANY;
        if (now != old) __atomic_store_n(hint, now, __ATOMIC_RELAXED);
    }
}

static void write_guard(HvCpu *cpu, HvRam *ram, uint64_t pg, uint32_t code, HvTlbEntry *e) {
    write_hint(cpu, ram, pg, code);
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    if (__atomic_load_n(&ram->code_pages[pg], __ATOMIC_RELAXED)) e->write_tag = HV_TLB_INVALID;
}

/* HV_KVA_HIST (measuring, for a kernel fastmem window): which kernel VAs get TLB fills, per GB from the bottom of xnu's
   kernel VA space, and how many distinct 16 KB pages. hv_kva_hist_dump writes it out. */
#define KVA_GBS 2048 /* buckets: 2048 of 1 GB from 0xfffffe0000000000 by default; HV_KVA_BASE (hex) and HV_KVA_SHIFT move and widen them */
static uint64_t KVA_BASE = 0xfffffe0000000000ull;
static int kva_shift = 30;
static uint64_t kva_fills[KVA_GBS], kva_pages_n[KVA_GBS], kva_below, kva_above;
static uint8_t *kva_seen; /* one bit per page */
static int kva_on = -1;
static void kva_note(uint64_t va) {
    if (kva_on < 0) {
        kva_on = getenv("HV_KVA_HIST") != NULL;
        if (getenv("HV_KVA_BASE")) KVA_BASE = strtoull(getenv("HV_KVA_BASE"), NULL, 16);
        if (getenv("HV_KVA_SHIFT")) kva_shift = atoi(getenv("HV_KVA_SHIFT"));
    }
    if (!kva_on) return;
    if (!kva_seen) kva_seen = calloc((size_t)KVA_GBS << (kva_shift - HV_TLB_PAGE_BITS - 3), 1);
    if (va < KVA_BASE) { __atomic_add_fetch(&kva_below, 1, __ATOMIC_RELAXED); return; }
    uint64_t off = va - KVA_BASE, gb = off >> kva_shift;
    if (gb >= KVA_GBS) { __atomic_add_fetch(&kva_above, 1, __ATOMIC_RELAXED); return; }
    __atomic_add_fetch(&kva_fills[gb], 1, __ATOMIC_RELAXED);
    uint64_t pg = off >> HV_TLB_PAGE_BITS;
    uint8_t bit = (uint8_t)(1u << (pg & 7));
    if (!(__atomic_fetch_or(&kva_seen[pg >> 3], bit, __ATOMIC_RELAXED) & bit)) __atomic_add_fetch(&kva_pages_n[gb], 1, __ATOMIC_RELAXED);
}

void hv_kva_hist_dump(FILE *f) {
    if (kva_on <= 0) return;
    fprintf(f, "kva below %llu above %llu (outside %d buckets of 2^%d bytes from %llx)\n", (unsigned long long)kva_below, (unsigned long long)kva_above, KVA_GBS, kva_shift, (unsigned long long)KVA_BASE);
    for (int g = 0; g < KVA_GBS; g++)
        if (kva_fills[g]) fprintf(f, "kva gb %d fills %llu pages %llu\n", g, (unsigned long long)kva_fills[g], (unsigned long long)kva_pages_n[g]);
}

static void tlb_fill(HvCpu *cpu, uint64_t va, int el, const HvWalk *w) {
    if ((va >> 55) & 1) kva_note(va & ~(uint64_t)PAGE_MASK);
    if (w->block_size < (1ull << HV_TLB_PAGE_BITS)) return; /* smaller guest pages: slow path only */
    HvRam *ram = ram_of(cpu->m, w->pa);
    if (!ram) return;
    uint64_t page_pa = w->pa & ~PAGE_MASK;
    if (page_pa < ram->pa || page_pa + PAGE_MASK + 1 - ram->pa > ram->size) return;
    uint64_t pg = (page_pa - ram->pa) >> HV_TLB_PAGE_BITS;
    uint64_t page_va = va & ~PAGE_MASK;
    int table = hv_tlb_table(cpu, el);
    uint32_t slot = (uint32_t)(va >> HV_TLB_PAGE_BITS) & (HV_TLB_SIZE - 1);
    HvTlbEntry *e = hv_tlb_ent(cpu, slot, table);
    if (!((page_va >> 55) & 1)) {
        if (cpu->nuser_fills < HV_USER_FILLS) cpu->user_fills[cpu->nuser_fills] = (uint16_t)hv_tlb_code(slot, table);
        if (cpu->nuser_fills <= HV_USER_FILLS) cpu->nuser_fills++;
    } else {
        if (cpu->nkernel_fills < HV_KERNEL_FILLS) cpu->kernel_fills[cpu->nkernel_fills] = (uint16_t)hv_tlb_code(slot, table);
        if (cpu->nkernel_fills <= HV_KERNEL_FILLS) cpu->nkernel_fills++;
    }
    e->addend = (int64_t)((uint64_t)(uintptr_t)(ram->host + (page_pa - ram->pa)) - page_va);
    e->read_tag = w->r ? page_va - HV_TLB_ADJ : HV_TLB_INVALID;
    /* Writes to ROM or to pages we translated from take the slow path. */
    e->write_tag = w->w && !ram->readonly && !ram->code_pages[pg] ? page_va - HV_TLB_ADJ : HV_TLB_INVALID;
    /* A watched page takes the slow path for every write. */
    if (cpu->m->watch_va && (cpu->m->watch_va & ~PAGE_MASK & 0x00ffffffffffffffull) == (page_va & 0x00ffffffffffffffull))
        e->write_tag = HV_TLB_INVALID;
    if (cpu->m->watch_pa && (cpu->m->watch_pa & ~PAGE_MASK) == page_pa) e->write_tag = HV_TLB_INVALID;
    e->exec_tag = w->x ? hv_exec_key(page_va, e->addend) : HV_TLB_INVALID;
    if (e->write_tag != HV_TLB_INVALID) write_guard(cpu, ram, pg, hv_tlb_code(slot, table), e);
}

/* Walks that may fill the TLB are bracketed by the CPU's walk_seq (odd while
   one runs), so another CPU's TLBI can drop entries here itself and know no
   fill from an old page table raced it (hv_tlbi_drop_remote_va).
   Each walk logs the page it may fill first (walk_page), so a TLBI of another page need not count it as a race.
   HV_WALK_ANY: a process switch, which saves and restores many entries and the stashes. */
#define HV_WALK_ANY (~0ull)
#define WALK_LOG 8
#define WALK_PAGE_BITS 0x00ffffffffffc000ull /* by the low 56 bits, as drops compare tagged addresses */

static void walk_begin(HvCpu *cpu, uint64_t va) {
    HvPerCpu *p = cpu->pc_state;
    if (!p) return;
    uint32_t k = p->walk_seq >> 1; /* only this CPU moves it */
    /* The entry is walk k - WALK_LOG's too: a reader that sees it rewritten must also see walk_seq at 2k or past it.
       A release store (walks_elsewhere's acquire fence pairs with it; walk_end's release does not order a later store). */
    __atomic_store_n(&p->walk_page[k & (WALK_LOG - 1)], va == HV_WALK_ANY ? HV_WALK_ANY : va & WALK_PAGE_BITS, __ATOMIC_RELEASE);
    __atomic_add_fetch(&p->walk_seq, 1, __ATOMIC_SEQ_CST); /* after the log entry, as its release */
}

static void walk_end(HvCpu *cpu) {
    if (cpu->pc_state) __atomic_add_fetch(&cpu->pc_state->walk_seq, 1, __ATOMIC_RELEASE);
}

static HvRam *ram_of_host(HvMachine *m, const uint8_t *host);

uint32_t hv_mmu_access(HvCpu *cpu, uint64_t va, HvAccess acc, int el, uint64_t *pa) {
    /* The TLB entry first, as translated code trusts it: slow paths that are slow for another reason (a page's
       last bytes, the access kind) need no walk (macOS 27: 100-250K a second). */
    if (acc != HV_ACC_EXEC && cpu->m) {
        const HvTlbEntry *e = hv_tlb_ent(cpu, (uint32_t)(va >> HV_TLB_PAGE_BITS) & (HV_TLB_SIZE - 1), hv_tlb_table(cpu, el));
        uint64_t tag = acc == HV_ACC_WRITE ? e->write_tag : e->read_tag;
        if (tag != HV_TLB_INVALID && tag + HV_TLB_ADJ == (va & ~(uint64_t)PAGE_MASK)) {
            const uint8_t *host = (const uint8_t *)(uintptr_t)(va + (uint64_t)e->addend);
            HvRam *ram = ram_of_host(cpu->m, host);
            if (ram) {
                *pa = ram->pa + (uint64_t)(host - ram->host);
                return 0;
            }
        }
    }
    HvWalk w;
    walk_begin(cpu, va);
    hv_mmu_walk(cpu, va, acc, el, &w);
    if (!w.fsc) tlb_fill(cpu, va, el, &w);
    walk_end(cpu);
    if (w.fsc) return w.fsc;
    *pa = w.pa;
    return 0;
}

/* HV_WALK_LOG: whether target's walks that ran (or were running) while walk_seq went from s1 to s2 all filled pages
   outside the npages from va, so none can have refilled one of them from the old page tables. By its log of the last
   WALK_LOG walks (a process switch counts as every page). Without the knob, only "no walk at all" passes. */
static int walks_elsewhere(HvCpu *target, uint32_t s1, uint32_t s2, uint64_t va, uint64_t npages) {
    if (s1 == s2 && !(s1 & 1)) return 1;
    if (!hv_knob[HV_K_WALKLOG]) return 0;
    HvPerCpu *p = target->pc_state;
    uint32_t k1 = s1 >> 1, k2 = (s2 + 1) >> 1; /* walks k1 .. k2 - 1 */
    if (k2 - k1 > WALK_LOG) return 0;
    uint64_t lo = va & WALK_PAGE_BITS, span = npages << HV_TLB_PAGE_BITS;
    for (uint32_t k = k1; k != k2; k++) {
        uint64_t pg = __atomic_load_n(&p->walk_page[k & (WALK_LOG - 1)], __ATOMIC_RELAXED);
        if (pg == HV_WALK_ANY || ((pg - lo) & 0x00ffffffffffffffull) < span) return 0;
    }
    /* and no later walk wrote over an entry while it was read: walk k1 + WALK_LOG writes k1's entry once walk_seq is
       2 (k1 + WALK_LOG), before it goes odd */
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    uint32_t s3 = __atomic_load_n(&p->walk_seq, __ATOMIC_SEQ_CST);
    if ((s3 >> 1) - k1 >= WALK_LOG) return 0;
    if (target->m->stats_interval) target->m->stats.race_other_page++;
    return 1;
}

int hv_tlbi_drop_remote_va(HvCpu *target, uint64_t va, int asid) {
    HvPerCpu *p = target->pc_state;
    uint32_t s1 = __atomic_load_n(&p->walk_seq, __ATOMIC_SEQ_CST);
    /* mid-walk: it may be about to fill from the old table, unless the walk is of another page (and not a switch's) */
    if ((s1 & 1) && !walks_elsewhere(target, s1, s1, va, 1)) { if (target->m->stats_interval) target->m->stats.race_kind[0]++; return 0; }
    uint32_t c1 = __atomic_load_n(&p->ijc_seq, __ATOMIC_SEQ_CST);
    if (c1 & 1) { if (target->m->stats_interval) target->m->stats.race_kind[1]++; return 0; } /* caching a block it may have looked up through the old mapping */
    if (!((va >> 55) & 1)) stash_drop_va(p, va, asid);
    uint32_t idx = (va >> HV_TLB_PAGE_BITS) & (HV_TLB_SIZE - 1);
    int fetched = 0; /* the target may be running code from the page: it must leave and look again */
    for (int t = 0; t < 3; t++) { /* the slot whatever its tag byte, as hv_tlb_flush_va_asid does */
        HvTlbEntry *e = hv_tlb_ent(target, idx, t);
        uint64_t x = __atomic_load_n(&e->exec_tag, __ATOMIC_RELAXED);
        if (x != HV_TLB_INVALID && ((hv_exec_key(x, e->addend) ^ va) & 0x00ffffffffffc000ull) == 0) fetched = 1;
        __atomic_store_n(&e->read_tag, HV_TLB_INVALID, __ATOMIC_RELAXED);
        __atomic_store_n(&e->write_tag, HV_TLB_INVALID, __ATOMIC_RELAXED);
        __atomic_store_n(&e->exec_tag, HV_TLB_INVALID, __ATOMIC_RELAXED);
    }
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    /* Its cached branch targets in the page too: a cache fill checks the TLB
       entry first (hv_ijc_fill_checked), so one after this sees it gone, and one
       during it changed ijc_seq. */
    if (!ijc_drop_va(target, va, asid, 1)) return -1; /* only target can do it: no use trying again */
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    int ok = walks_elsewhere(target, s1, __atomic_load_n(&p->walk_seq, __ATOMIC_SEQ_CST), va, 1) && /* no fill of it in between */
             __atomic_load_n(&p->ijc_seq, __ATOMIC_SEQ_CST) == c1;
    if (!ok) {
        if (target->m->stats_interval) target->m->stats.race_kind[2]++;
    } else if (fetched) {
        hv_cpu_kick(target);
    }
    return ok;
}

/* hv_tlbi_drop_remote_va for npages pages from va at once (big range TLBIs, HV_RANGE_DROP): one pass over the
   stashes instead of one per page, then each page's TLB slots and cached branch targets, all inside one
   walk_seq/ijc_seq window. Same answers: 1 done, 0 raced (try again), -1 only the target can do it. */
static void stash_drop_range(HvPerCpu *p, uint64_t va, uint64_t npages, int asid);

int hv_tlbi_drop_remote_range(HvCpu *target, uint64_t va, uint64_t npages, int asid) {
    HvPerCpu *p = target->pc_state;
    uint32_t s1 = __atomic_load_n(&p->walk_seq, __ATOMIC_SEQ_CST);
    if ((s1 & 1) && !walks_elsewhere(target, s1, s1, va, npages)) { if (target->m->stats_interval) target->m->stats.race_kind[0]++; return 0; }
    uint32_t c1 = __atomic_load_n(&p->ijc_seq, __ATOMIC_SEQ_CST);
    if (c1 & 1) { if (target->m->stats_interval) target->m->stats.race_kind[1]++; return 0; }
    if (!((va >> 55) & 1)) stash_drop_range(p, va, npages, asid);
    int fetched = 0;
    for (uint64_t i = 0; i < npages && i < HV_TLB_SIZE; i++) { /* past HV_TLB_SIZE pages the slots come round again */
        uint32_t idx = (uint32_t)((va >> HV_TLB_PAGE_BITS) + i) & (HV_TLB_SIZE - 1);
        for (int t = 0; t < 3; t++) {
            HvTlbEntry *e = hv_tlb_ent(target, idx, t);
            uint64_t x = __atomic_load_n(&e->exec_tag, __ATOMIC_RELAXED);
            /* the slot whatever its tag, as for one page; fetched if it holds code from the range */
            if (x != HV_TLB_INVALID) {
                uint64_t off = ((hv_exec_key(x, e->addend) & 0x00ffffffffffc000ull) - (va & 0x00ffffffffffc000ull)) & 0x00ffffffffffffffull;
                if (off < (npages << HV_TLB_PAGE_BITS)) fetched = 1;
            }
            __atomic_store_n(&e->read_tag, HV_TLB_INVALID, __ATOMIC_RELAXED);
            __atomic_store_n(&e->write_tag, HV_TLB_INVALID, __ATOMIC_RELAXED);
            __atomic_store_n(&e->exec_tag, HV_TLB_INVALID, __ATOMIC_RELAXED);
        }
    }
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    for (uint64_t i = 0; i < npages; i++)
        if (!ijc_drop_va(target, va + (i << HV_TLB_PAGE_BITS), asid, 1)) return -1;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    int ok = walks_elsewhere(target, s1, __atomic_load_n(&p->walk_seq, __ATOMIC_SEQ_CST), va, npages) &&
             __atomic_load_n(&p->ijc_seq, __ATOMIC_SEQ_CST) == c1;
    if (!ok) {
        if (target->m->stats_interval) target->m->stats.race_kind[2]++;
    } else if (fetched) {
        hv_cpu_kick(target);
    }
    return ok;
}

const uint32_t *hv_mmu_fetch(HvCpu *cpu, uint64_t va, uint64_t *pa, uint32_t *avail) {
    if (va & 3) {
        hv_raise_sync(cpu, HV_EC_PC_ALIGN, 0, va, va);
        return NULL;
    }
    HvTlbEntry *t = hv_tlb_ent(cpu, (va >> HV_TLB_PAGE_BITS) & (HV_TLB_SIZE - 1), hv_tlb_table(cpu, cpu->el));
    if (t->exec_tag == hv_exec_key(va & ~PAGE_MASK, t->addend)) {
        uint8_t *host = (uint8_t *)(uintptr_t)(va + t->addend);
        for (int i = 0; i < cpu->m->nram; i++) {
            HvRam *r = &cpu->m->ram[i];
            if ((uint64_t)(host - r->host) < r->size) {
                *pa = r->pa + (uint64_t)(host - r->host);
                *avail = (uint32_t)(((1ull << HV_TLB_PAGE_BITS) - (va & PAGE_MASK)) / 4);
                return (const uint32_t *)host;
            }
        }
    }
    HvWalk w;
    cpu->m->stats.fetch_walks++;
    walk_begin(cpu, va);
    hv_mmu_walk(cpu, va, HV_ACC_EXEC, cpu->el, &w);
    if (!w.fsc) tlb_fill(cpu, va, cpu->el, &w);
    walk_end(cpu);
    if (w.fsc) {
        hv_raise_abort(cpu, 1, va, w.fsc, 0, w.s1ptw, va);
        return NULL;
    }
    uint64_t chunk = w.block_size < (1ull << HV_TLB_PAGE_BITS) ? w.block_size : 1ull << HV_TLB_PAGE_BITS;
    uint64_t left = chunk - (va & (chunk - 1));
    const uint32_t *p = hv_phys_ptr(cpu->m, w.pa, 4);
    if (!p) {
        hv_raise_abort(cpu, 1, va, HV_FSC_EXTERNAL, 0, 0, va);
        return NULL;
    }
    HvRam *ram = hv_ram_lookup(cpu->m, w.pa);
    uint64_t ram_left = ram->size - (w.pa - ram->pa);
    if (left > ram_left) left = ram_left;
    *pa = w.pa;
    *avail = (uint32_t)(left / 4);
    return p;
}

/* A TLB miss in translated code, before the full helper: walk and fill, and
   say whether the fast path will now succeed (0) or the helper must do the
   access (a fault, device memory, a write to a code page, an unusual spot in
   the page). Called by hv_helper_thunk with the guest's SIMD registers live,
   so nothing here may use them (see read_desc). flags: bit 0 write, bits 2..1
   the TLB table (0 EL0 or unprivileged, 1 EL1, 2 EL1 with PAN), bits 8..3 bytes - 1. */
int hv_tlb_refill(HvCpu *cpu, uint64_t va, uint32_t flags) {
    int write = flags & 1, table = (flags >> 1) & 3;
    uint32_t bytes = ((flags >> 3) & 63) + 1;
    int el = table ? 1 : 0;
    uint64_t off = va & PAGE_MASK;
    if (off + bytes > PAGE_MASK + 1) return 1; /* crosses into the next page: the helper */
    /* In the page, but where the fast check cannot pass (it allows HV_TLB_ADJ bytes of every access, so a small one in a
       page's last HV_TLB_ADJ bytes misses): a refill would come straight back here. */
    if (bytes <= HV_TLB_ADJ + 1 && off + HV_TLB_ADJ > PAGE_MASK) return 1;
    HvWalk w;
    walk_begin(cpu, va);
    hv_mmu_walk(cpu, va, write ? HV_ACC_WRITE : HV_ACC_READ, el, &w);
    if (!w.fsc) tlb_fill(cpu, va, el, &w);
    walk_end(cpu);
    if (w.fsc) return 1;
    HvTlbEntry *e = hv_tlb_ent(cpu, (va >> HV_TLB_PAGE_BITS) & (HV_TLB_SIZE - 1), table);
    uint64_t tag = write ? e->write_tag : e->read_tag;
    if (cpu->m->stats_interval) cpu->m->stats.refills++;
    return tag != HV_TLB_INVALID && tag + HV_TLB_ADJ == (va & ~(uint64_t)PAGE_MASK) ? 0 : 1;
}

/* Put block b in this CPU's indirect-branch cache under its VA (b->va must be in
   the half its flags say: kernel blocks, kernel VAs). */
void hv_ijc_fill(HvCpu *cpu, HvBlock *b) {
    HvMachine *m = cpu->m;
    HvPerCpu *pcs = cpu->pc_state;
    int kernel = b->flags & HV_BLOCK_EL1;
    uint64_t *t = pcs->ijc + (size_t)(b->flags & (HV_IJC_TABLES - 1)) * (HV_IJC_TABLE_BYTES / 8);
    uint64_t tag = b->va ^ cpu->ijc_salt[kernel];
    uint64_t *ent = t + (size_t)hv_ijc_index(tag) * 2;
    if (ent[0] != tag && ent[0]) ent[2] = ent[0], ent[3] = ent[1]; /* way 0's goes to way 1 (over this one's, if there) */
    ent[0] = tag;
    ent[1] = (uint64_t)(uintptr_t)b->code;
    pcs->ijc_tables_used |= 1u << (b->flags & (HV_IJC_TABLES - 1));
    {   /* the page's hint: which piece has a target now */
        uint64_t key = ((b->va & ~(uint64_t)PAGE_MASK) ^ cpu->ijc_salt[kernel]) | 1, bit = 1ull << ((b->va >> 8) & 63);
        __typeof__(pcs->ijc_hint[0][0]) *h = &pcs->ijc_hint[kernel][hint_slot(key)];
        if (h->key == key) h->bits |= bit;
        else if (!h->key) h->bits = bit, __atomic_store_n(&h->key, key, __ATOMIC_RELEASE);
        else if (h->key != ~0ull) h->bits = ~0ull, __atomic_store_n(&h->key, ~0ull, __ATOMIC_RELEASE);
    }
    /* The filters below are per page: the next fill is mostly in the same one. */
    uint64_t pkey = (b->va & ~(uint64_t)PAGE_MASK) ^ cpu->ijc_salt[kernel] ^ 1;
    if (pcs->last_fill[kernel] == pkey) return;
    if (!kernel) {
        hv_ucode_add(cpu, b->va);
        uint32_t h = (uint32_t)(b->va >> HV_TLB_PAGE_BITS) & 0xffff;
        pcs->ijc_pages[0][h >> 6] |= 1ull << (h & 63);
        uint32_t ch = (uint32_t)(b->va >> 25) & 255, cid = (uint32_t)(cpu->ijc_salt[0] >> 48); /* the id after any reset in ucode_add */
        pcs->cid_chunks[cid][ch >> 6] |= 1ull << (ch & 63);
    }
    uint32_t k = hv_ijc_key(b->va, cpu->ijc_salt[kernel]);
    if (!(pcs->ijc_keyed[k >> 6] >> (k & 63) & 1)) {
        pcs->ijc_keyed[k >> 6] |= 1ull << (k & 63);
        if (++pcs->ijc_keyed_count > (1u << 17)) { /* an eighth full: start the cache and the filter over */
            m->stats.reset_why[6]++;
            hv_ijc_reset(cpu, 0);
            hv_ijc_reset(cpu, 1);
            memset(pcs->ijc_keyed, 0, sizeof pcs->ijc_keyed);
            pcs->ijc_keyed_count = 0;
        }
    }
    pcs->last_fill[kernel] = (b->va & ~(uint64_t)PAGE_MASK) ^ cpu->ijc_salt[kernel] ^ 1; /* the salt after any reset above */
}

/* An indirect-branch cache miss in translated code, before the dispatcher: if
   the target's block already exists (found through this CPU's TLB and the
   block hash), cache it and return its code for the thunk to jump to; else 0
   and the dispatcher does it. Called by hv_ijc_thunk with the guest's SIMD
   registers live (see read_desc), so nothing that could reach libc runs: a
   fill that would reset the cache's filters goes to the dispatcher instead.
   flags: the block flags the branch was translated under (the target's too). */
/* Does this CPU's TLB (the table for flags) map b's page where b came from? */
static int tlb_maps_block(HvCpu *cpu, const HvBlock *b) {
    int table = !(b->flags & HV_BLOCK_EL1) ? 0 : (b->flags & HV_BLOCK_PAN) ? 2 : 1;
    const HvTlbEntry *e = hv_tlb_ent(cpu, (b->va >> HV_TLB_PAGE_BITS) & (HV_TLB_SIZE - 1), table);
    uint64_t page_va = b->va & ~(uint64_t)PAGE_MASK, page_pa = b->pa & ~(uint64_t)PAGE_MASK;
    return __atomic_load_n(&e->exec_tag, __ATOMIC_RELAXED) == hv_exec_key(page_va, e->addend) &&
           (uint64_t)e->addend == (uint64_t)(uintptr_t)(b->ram->host + (page_pa - b->ram->pa)) - page_va;
}

/* The dispatcher caching a block it found for va: only while the TLB still
   says va is there, inside ijc_seq, so another CPU's TLBI can drop this CPU's
   cached targets itself (hv_tlbi_drop_remote_va). */
/* Note on b's physical page the VA page it is cached under (the page's code changing then drops just those targets,
   hv_invalidate_code), then check b is still current. Both read-modify-writes and the load are sequentially
   consistent, against the generation bump and the note's exchange there: either the change sees this note, or this
   sees the new generation and caches nothing. */
static int ijc_note_page(const HvBlock *b) {
    HvRam *r = b->ram;
    if (!r || !r->ijc_va) return 1;
    uint64_t pg = (b->pa - r->pa) >> HV_TLB_PAGE_BITS, v = (b->va & ~(uint64_t)PAGE_MASK) | 2;
    uint64_t *slot = &r->ijc_va[pg];
    uint64_t old = __atomic_load_n(slot, __ATOMIC_RELAXED);
    /* Noted already (v, or ~0): what this read saw is what the next change's exchange sees first, so that change
       drops this target too (its request reaches this CPU after the fill). Else noted here, in the order. */
    while (old != v && old != ~0ull)
        if (__atomic_compare_exchange_n(slot, &old, old ? ~0ull : v, 0, __ATOMIC_SEQ_CST, __ATOMIC_RELAXED)) break;
    return __atomic_load_n(&r->page_gen[pg], __ATOMIC_SEQ_CST) == b->gen;
}

void hv_ijc_drop_code_page(HvCpu *cpu, uint64_t v) {
    ret_new(cpu, 0);
    ret_new(cpu, 1);
    if (v == 1 || !cpu->pc_state) return;
    uint64_t page = v & ~(uint64_t)PAGE_MASK;
    int kernel = (page >> 55) & 1;
    if (kernel) {
        ijc_clear_page(cpu, page, 1, cpu->ijc_salt[1], 1);
        return;
    }
    if (current_asid(cpu) < 0) {
        ijc_clear_page(cpu, page, 0, cpu->ijc_salt[0], 1);
        return;
    }
    if (cpu->nasid_log > HV_ASID_LOG) { /* the ids given out are not all known: start the user half over */
        hv_ijc_reset(cpu, 0);
        return;
    }
    /* Every live process id that cached a target in the page (the (id, page) set says which). */
    for (uint32_t k = 0; k < cpu->nasid_log; k++) {
        uint32_t cid = cpu->asid_cid[cpu->asid_log[k]];
        if (cid && ucode_has(cpu->pc_state, cid, page)) ijc_clear_page(cpu, page, 0, (uint64_t)cid << 48, 1);
    }
}

int hv_ijc_fill_checked(HvCpu *cpu, HvBlock *b) {
    HvPerCpu *p = cpu->pc_state;
    if (!p) {
        hv_ijc_fill(cpu, b);
        return 1;
    }
    if (!ijc_note_page(b)) return 0;
    __atomic_add_fetch(&p->ijc_seq, 1, __ATOMIC_SEQ_CST);
    int ok = tlb_maps_block(cpu, b);
    if (ok) hv_ijc_fill(cpu, b);
    __atomic_add_fetch(&p->ijc_seq, 1, __ATOMIC_RELEASE);
    return ok;
}

static void *ijc_resolve(HvCpu *cpu, uint64_t va, uint32_t flags);

void *hv_ijc_resolve(HvCpu *cpu, uint64_t va, uint32_t flags) {
    return ijc_resolve(cpu, va, flags);
}

static void *ijc_resolve(HvCpu *cpu, uint64_t va, uint32_t flags) {
    HvMachine *m = cpu->m;
    HvPerCpu *pcs = cpu->pc_state;
    int kernel = flags & HV_BLOCK_EL1;
#define FAIL(k) do { if (m->stats_interval) m->stats.light_fail[k]++; return 0; } while (0)
    if (!m->chain || va == m->stop_pc || va == m->stop_pc2 || (int)((va >> 55) & 1) != kernel || (va & 3)) FAIL(0);
    if (pcs->ijc_keyed_count >= (1u << 17) || pcs->ucode_count + 1 >= (1u << HV_UCODE_BITS) / 2) FAIL(1);
    int table = !kernel ? 0 : (flags & HV_BLOCK_PAN) ? 2 : 1;
    HvTlbEntry *e = hv_tlb_ent(cpu, (va >> HV_TLB_PAGE_BITS) & (HV_TLB_SIZE - 1), table);
    if (e->exec_tag != hv_exec_key(va & ~(uint64_t)PAGE_MASK, e->addend)) {
        /* Not in the TLB (a process switch, or another page took the slot):
           walk for it here, as hv_tlb_refill does for data. A fault is for the
           dispatcher to take. */
        HvWalk w;
        walk_begin(cpu, va);
        hv_mmu_walk(cpu, va, HV_ACC_EXEC, kernel ? 1 : 0, &w);
        if (!w.fsc) tlb_fill(cpu, va, kernel ? 1 : 0, &w);
        walk_end(cpu);
        if (w.fsc || e->exec_tag != hv_exec_key(va & ~(uint64_t)PAGE_MASK, e->addend)) FAIL(2);
        if (m->stats_interval) m->stats.light_walks++;
    }
    const uint8_t *host = (const uint8_t *)(uintptr_t)(va + (uint64_t)e->addend);
    HvRam *ram = NULL;
    for (int i = 0; i < m->nram; i++)
        if ((uint64_t)(host - m->ram[i].host) < m->ram[i].size) ram = &m->ram[i];
    if (!ram) return 0;
    uint64_t pa = ram->pa + (uint64_t)(host - ram->host);
    uint32_t gen = __atomic_load_n(&ram->page_gen[(pa - ram->pa) >> HV_TLB_PAGE_BITS], __ATOMIC_ACQUIRE);
    /* This CPU's small cache in front of the block hash first (as the dispatcher's lookup). */
    uint32_t j = (uint32_t)((va >> 2) ^ (va >> 18) ^ ((uint64_t)flags << 13)) & ((1u << HV_JC_BITS) - 1);
    HvBlock *b = pcs->jc[j].va == va ? pcs->jc[j].b : NULL;
    if (!(b && b->pa == pa && b->flags == flags && b->gen == gen)) {
        b = hv_block_find(m, hv_block_hash(va, pa, flags), va, pa, flags, gen);
        if (!b) FAIL(3);
#undef FAIL
        pcs->jc[j].va = va, pcs->jc[j].b = b;
    }
    if (m->stats_interval) { /* what the fill replaces, as the dispatcher counts it */
        uint64_t *t = pcs->ijc + (size_t)(b->flags & (HV_IJC_TABLES - 1)) * (HV_IJC_TABLE_BYTES / 8);
        uint64_t tag = b->va ^ cpu->ijc_salt[kernel];
        uint64_t old = t[(size_t)hv_ijc_index(tag) * 2];
        int k = !old ? 0 : (old >> 48) != (tag >> 48) ? 1 : 2;
        if (k == 1) {
            uint32_t c = (uint32_t)(old >> 48);
            if (!(kernel ? (c ^ 0xffff) == cpu->kernel_gen : cpu->asid_cid[cpu->cid_asid[c]] == c)) k = 3;
        }
        m->stats.light_kind[k]++;
        m->stats.ijc_light++;
        if (!kernel) m->stats.light_region[(va - 0x180000000ull) < 0x100000000ull ? 0 : 1]++;
    }
    /* Cached only while the TLB still maps it, inside ijc_seq (see hv_ijc_fill_checked). */
    if (!hv_ijc_fill_checked(cpu, b)) return 0;
    return b->code;
}

/* ---- user TLB entries kept across process switches (HvTlbStash) ---- */

/* Forget the stashed entries of address space asid (-1: all). Other CPUs call
   this on the target's stashes too, between reading and rechecking its
   walk_seq: saves and restores run inside walk_begin/walk_end, so a TLBI that
   races one of them falls back to waiting for the target. */
static void stash_drop(HvPerCpu *p, int asid) {
    for (int i = 0; i < HV_STASHES; i++) {
        uint64_t t = __atomic_load_n(&p->stash[i].ttbr0, __ATOMIC_SEQ_CST);
        if (t && (asid < 0 || (int)(t >> 48) == asid)) __atomic_add_fetch(&p->stash[i].gen, 1, __ATOMIC_SEQ_CST);
    }
}

/* Forget the stashed entries for va's page in address space asid (-1: all). Other CPUs do this to
   the target's stashes too, under the same walk_seq check as stash_drop; they only ever write tags
   (never an addend), so a restore racing it reads each entry whole or with a tag gone. */
static void stash_drop_va(HvPerCpu *p, uint64_t va, int asid) {
    uint64_t pg = va & 0x00ffffffffffffffull & ~(uint64_t)PAGE_MASK; /* tagged addresses: by the low 56 bits */
    for (int i = 0; i < HV_STASHES; i++) {
        HvTlbStash *s = &p->stash[i];
        uint64_t t = __atomic_load_n(&s->ttbr0, __ATOMIC_SEQ_CST);
        if (!t || (asid >= 0 && (int)(t >> 48) != asid)) continue;
        uint32_t n = __atomic_load_n(&s->n, __ATOMIC_RELAXED);
        if (n > HV_STASH_CAP) n = HV_STASH_CAP;
        for (uint32_t k = 0; k < n; k++) {
            HvTlbEntry *e = &s->ent[k];
            uint64_t rt = __atomic_load_n(&e->read_tag, __ATOMIC_RELAXED), wt = __atomic_load_n(&e->write_tag, __ATOMIC_RELAXED),
                     xt = __atomic_load_n(&e->exec_tag, __ATOMIC_RELAXED);
            if (rt == HV_TLB_INVALID && wt == HV_TLB_INVALID && xt == HV_TLB_INVALID) continue;
            uint64_t tag = rt != HV_TLB_INVALID ? rt + HV_TLB_ADJ : xt != HV_TLB_INVALID ? hv_exec_key(xt, e->addend) : wt + HV_TLB_ADJ;
            if ((tag & 0x00ffffffffffffffull & ~(uint64_t)PAGE_MASK) != pg) continue;
            __atomic_store_n(&e->read_tag, HV_TLB_INVALID, __ATOMIC_RELAXED);
            __atomic_store_n(&e->write_tag, HV_TLB_INVALID, __ATOMIC_RELAXED);
            __atomic_store_n(&e->exec_tag, HV_TLB_INVALID, __ATOMIC_RELAXED);
        }
    }
}

/* stash_drop_va for npages pages from va, in one pass. */
static void stash_drop_range(HvPerCpu *p, uint64_t va, uint64_t npages, int asid) {
    uint64_t lo = va & 0x00ffffffffffffffull & ~(uint64_t)PAGE_MASK, span = npages << HV_TLB_PAGE_BITS;
    for (int i = 0; i < HV_STASHES; i++) {
        HvTlbStash *s = &p->stash[i];
        uint64_t t = __atomic_load_n(&s->ttbr0, __ATOMIC_SEQ_CST);
        if (!t || (asid >= 0 && (int)(t >> 48) != asid)) continue;
        uint32_t n = __atomic_load_n(&s->n, __ATOMIC_RELAXED);
        if (n > HV_STASH_CAP) n = HV_STASH_CAP;
        for (uint32_t k = 0; k < n; k++) {
            HvTlbEntry *e = &s->ent[k];
            uint64_t rt = __atomic_load_n(&e->read_tag, __ATOMIC_RELAXED), wt = __atomic_load_n(&e->write_tag, __ATOMIC_RELAXED),
                     xt = __atomic_load_n(&e->exec_tag, __ATOMIC_RELAXED);
            if (rt == HV_TLB_INVALID && wt == HV_TLB_INVALID && xt == HV_TLB_INVALID) continue;
            uint64_t tag = rt != HV_TLB_INVALID ? rt + HV_TLB_ADJ : xt != HV_TLB_INVALID ? hv_exec_key(xt, e->addend) : wt + HV_TLB_ADJ;
            if ((((tag & 0x00ffffffffffffffull & ~(uint64_t)PAGE_MASK) - lo) & 0x00ffffffffffffffull) >= span) continue;
            __atomic_store_n(&e->read_tag, HV_TLB_INVALID, __ATOMIC_RELAXED);
            __atomic_store_n(&e->write_tag, HV_TLB_INVALID, __ATOMIC_RELAXED);
            __atomic_store_n(&e->exec_tag, HV_TLB_INVALID, __ATOMIC_RELAXED);
        }
    }
}

void hv_tlb_stash_drop(HvCpu *cpu, int asid) {
    if (cpu->pc_state) stash_drop(cpu->pc_state, asid);
}

static int user_entry(const HvTlbEntry *e) {
    if (e->read_tag == HV_TLB_INVALID && e->exec_tag == HV_TLB_INVALID && e->write_tag == HV_TLB_INVALID) return 0;
    uint64_t tag = e->read_tag != HV_TLB_INVALID ? e->read_tag + HV_TLB_ADJ : e->exec_tag != HV_TLB_INVALID ? hv_exec_key(e->exec_tag, e->addend) : e->write_tag + HV_TLB_ADJ;
    return !((tag >> 55) & 1);
}

static HvRam *ram_of_host(HvMachine *m, const uint8_t *host) {
    for (int i = 0; i < m->nram; i++)
        if ((uint64_t)(host - m->ram[i].host) < m->ram[i].size) return &m->ram[i];
    return NULL;
}

/* HV_SWITCH_FLUSH=2 (checking): every entry this CPU holds, in its TLB and in the stashes it would bring back, walked again
   in its own address space. Returns how many no longer match the page tables (a walk fault, or another page), so that
   a whole-TLB flush that is left out would have dropped something the guest expects gone. A cleared access flag alone
   is counted apart (stats) and not as out of date: xnu ages pages with its own TLBIs. */
int hv_tlb_check_all(HvCpu *cpu) {
    HvMachine *m = cpu->m;
    HvPerCpu *p = cpu->pc_state;
    uint64_t ttbr0 = cpu->ttbr0_el1;
    int bad = 0;
    static int told;
    for (int pass = -1; pass < (p ? HV_STASHES : 0); pass++) {
        /* The stash passes walk with cpu->ttbr0_el1 set to the stash's: inside a walk of any page, so another CPU's
           TLBI meanwhile (which reads it for the running ASID, ijc_drop_va) counts as raced and is left to this CPU. */
        if (pass == 0) walk_begin(cpu, HV_WALK_ANY);
        uint32_t n = 3 * HV_TLB_SIZE;
        const HvTlbStash *s = NULL;
        if (pass >= 0) {
            s = &p->stash[pass];
            uint64_t t = __atomic_load_n(&s->ttbr0, __ATOMIC_ACQUIRE);
            if (!t || __atomic_load_n(&s->gen, __ATOMIC_ACQUIRE) != s->saved_gen) continue; /* would not be brought back */
            n = s->n > HV_STASH_CAP ? HV_STASH_CAP : s->n;
            cpu->ttbr0_el1 = t;
        }
        for (uint32_t k = 0; k < n; k++) {
            const HvTlbEntry *e;
            int table;
            if (!s) table = (int)(k / HV_TLB_SIZE), e = hv_tlb_ent(cpu, k % HV_TLB_SIZE, table);
            else table = hv_tlb_tmajor ? s->slot[k] >> HV_TLB_BITS : s->slot[k] & 3, e = &s->ent[k];
            uint64_t rt = e->read_tag, wt = e->write_tag, xt = e->exec_tag;
            if (rt == HV_TLB_INVALID && wt == HV_TLB_INVALID && xt == HV_TLB_INVALID) continue;
            uint64_t va = rt != HV_TLB_INVALID ? rt + HV_TLB_ADJ : wt != HV_TLB_INVALID ? wt + HV_TLB_ADJ : hv_exec_key(xt, e->addend);
            if (s && (va >> 55) & 1) continue; /* stashes hold the user half only */
            HvWalk w;
            hv_mmu_walk(cpu, va, rt != HV_TLB_INVALID || wt != HV_TLB_INVALID ? HV_ACC_READ : HV_ACC_EXEC, table ? 1 : 0, &w);
            if (m->stats_interval) m->stats.flush_check_entries++;
            const char *why = NULL;
            if (w.fsc >= HV_FSC_ACCESS(0) && w.fsc <= HV_FSC_ACCESS(3)) {
                if (m->stats_interval) m->stats.flush_check_af++;
                continue;
            }
            if (w.fsc >= HV_FSC_PERMISSION(0) && w.fsc <= HV_FSC_PERMISSION(3)) continue; /* PAN or EL: not about where the page is */
            if (w.fsc) why = "no mapping";
            else {
                uint64_t page_pa = w.pa & ~(uint64_t)PAGE_MASK;
                const uint8_t *host = (const uint8_t *)(uintptr_t)(va + (uint64_t)e->addend);
                HvRam *ram = ram_of_host(m, host);
                if (!ram || ram->pa + (uint64_t)(host - ram->host) != page_pa) why = "another page";
            }
            if (why) {
                bad++;
                if (told++ < 16)
                    fprintf(stderr, "hengevm: switch-flush check: %s entry (table %d, va %llx, ttbr0 %llx) is out of date: %s (fsc %x)\n",
                            s ? "stashed" : "live", table, (unsigned long long)va, (unsigned long long)cpu->ttbr0_el1, why, w.fsc);
            }
        }
        cpu->ttbr0_el1 = ttbr0;
    }
    cpu->ttbr0_el1 = ttbr0;
    if (p) walk_end(cpu);
    if (m->stats_interval) {
        m->stats.flush_checked++;
        if (bad) m->stats.flush_check_bad++;
    }
    return bad;
}

/* TTBR0 went from old to now (a process switch): keep old's user entries
   aside, drop them from the TLB, and bring back now's if they were kept and no
   TLBI touched them since. */
void hv_tlb_switch_user(HvCpu *cpu, uint64_t old, uint64_t now) {
    HvPerCpu *p = cpu->pc_state;
    HvMachine *m = cpu->m;
    if (old == now) return; /* the same address space: a TLB tagged by ASID would keep everything too */
    static int no_stash = -1; /* HV_NO_STASH: a switch just drops the user entries */
    if (no_stash < 0) no_stash = getenv("HV_NO_STASH") != NULL;
    if (!p || current_asid(cpu) < 0 || m->watch_va || m->watch_pa || no_stash) {
        hv_tlb_flush_user(cpu);
        return;
    }
    walk_begin(cpu, HV_WALK_ANY);
    m->stats.tlb_flushes++;
    uint32_t n = cpu->nuser_fills;
    cpu->nuser_fills = 0;
    HvTlbStash *s = NULL;
    if (old && n <= HV_STASH_CAP) {
        for (int i = 0; i < HV_STASHES && !s; i++)
            if (p->stash[i].ttbr0 == old) s = &p->stash[i];
        for (int i = 0; i < HV_STASHES && !s; i++)
            if (!p->stash[i].ttbr0) s = &p->stash[i];
        if (!s) {
            s = &p->stash[0];
            for (int i = 1; i < HV_STASHES; i++)
                if (p->stash[i].used < s->used) s = &p->stash[i];
        }
        __atomic_store_n(&s->ttbr0, old, __ATOMIC_SEQ_CST);
        s->saved_gen = __atomic_load_n(&s->gen, __ATOMIC_SEQ_CST);
        s->used = ++p->stash_clock;
        if (m->stats_interval) m->stats.stash_saves++;
    }
    uint32_t k = 0;
    if (n <= HV_USER_FILLS) {
        for (uint32_t i = 0; i < n; i++) {
            uint16_t code = cpu->user_fills[i];
            HvTlbEntry *e = &cpu->tlb[0] + code;
            if (!user_entry(e)) continue; /* the slot may hold a kernel page by now */
            if (s) {
                s->slot[k] = code;
                s->ent[k++] = *e;
            }
            memset(e, 0x80, sizeof *e);
        }
    } else {
        for (uint32_t i = 0; i < HV_TLB_SIZE; i++)
            for (int t = 0; t < 3; t++)
                if (user_entry(hv_tlb_ent(cpu, i, t))) memset(hv_tlb_ent(cpu, i, t), 0x80, sizeof(HvTlbEntry));
    }
    if (s) s->n = k;

    for (int i = 0; i < HV_STASHES; i++) {
        HvTlbStash *r = &p->stash[i];
        if (r->ttbr0 != now) continue;
        int fresh = __atomic_load_n(&r->gen, __ATOMIC_SEQ_CST) == r->saved_gen;
        __atomic_store_n(&r->ttbr0, 0, __ATOMIC_SEQ_CST); /* used up either way: the live TLB has them now */
        if (!fresh) {
            if (m->stats_interval) m->stats.stash_stale++;
            break;
        }
        /* Writable entries: all their hints, one fence, then each page's code check (write_guard, batched: one fence per
           switch instead of one per entry; every entry is in place before the fence and checked after it). */
        uint16_t pend[HV_STASH_CAP];
        uint32_t npend = 0;
        for (uint32_t j = 0; j < r->n; j++) {
            const HvTlbEntry *from = &r->ent[j];
            if (from->read_tag == HV_TLB_INVALID && from->write_tag == HV_TLB_INVALID && from->exec_tag == HV_TLB_INVALID)
                continue; /* its page had a TLBI while it was put aside (stash_drop_va) */
            HvTlbEntry *e = &cpu->tlb[0] + r->slot[j];
            *e = *from;
            cpu->user_fills[cpu->nuser_fills++] = r->slot[j];
            if (e->write_tag != HV_TLB_INVALID) {
                const uint8_t *host = (const uint8_t *)(uintptr_t)(from->write_tag + HV_TLB_ADJ + from->addend);
                HvRam *ram = ram_of_host(m, host);
                if (!ram) e->write_tag = HV_TLB_INVALID;
                else write_hint(cpu, ram, (uint64_t)(host - ram->host) >> HV_TLB_PAGE_BITS, r->slot[j]), pend[npend++] = (uint16_t)j;
            }
        }
        if (npend) __atomic_thread_fence(__ATOMIC_SEQ_CST);
        for (uint32_t q = 0; q < npend; q++) {
            const HvTlbEntry *from = &r->ent[pend[q]];
            const uint8_t *host = (const uint8_t *)(uintptr_t)(from->write_tag + HV_TLB_ADJ + from->addend);
            HvRam *ram = ram_of_host(m, host);
            if (ram && __atomic_load_n(&ram->code_pages[(uint64_t)(host - ram->host) >> HV_TLB_PAGE_BITS], __ATOMIC_RELAXED))
                (&cpu->tlb[0] + r->slot[pend[q]])->write_tag = HV_TLB_INVALID;
        }
        if (m->stats_interval) m->stats.stash_hits++;
        if (m->stats_interval) m->stats.stash_entries += r->n;
        break;
    }
    walk_end(cpu);
}
