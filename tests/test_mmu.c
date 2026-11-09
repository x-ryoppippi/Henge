// Stage 1 walker: granules, block sizes, TTBR0/TTBR1, TBI, permissions and
// fault codes, checked directly against hv_mmu_walk.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "machine.h"
#include "mmu.h"
#include "sysreg.h"

#define PT_PA 0x80000000ull
#define PT_SIZE (1u << 20)

static uint8_t *pt;
static uint64_t pt_next;
static int failures, checks;

static uint64_t *host(uint64_t pa) {
    return (uint64_t *)(pt + (pa - PT_PA));
}

static uint64_t alloc_table(void) {
    uint64_t pa = PT_PA + pt_next;
    pt_next += 65536; /* big enough and aligned for any granule */
    memset(host(pa), 0, 65536);
    return pa;
}

/* Install a leaf for va at `leaf_level`, creating tables from the start level. */
static uint64_t *map(uint64_t root, int g, int inputsize, uint64_t va, int leaf_level, uint64_t desc,
                     uint64_t table_bits) {
    int stride = g - 3;
    int levels = (inputsize - g + stride - 1) / stride;
    uint64_t table = root;
    for (int level = 4 - levels;; level++) {
        int shift = g + stride * (3 - level);
        int idx_bits = level == 4 - levels ? inputsize - shift : stride;
        uint64_t *e = &host(table)[(va >> shift) & ((1ull << idx_bits) - 1)];
        if (level == leaf_level) {
            *e = desc;
            return e;
        }
        if (!(*e & 1)) *e = alloc_table() | 3 | table_bits;
        table = *e & 0x0000fffffffff000ull & ~((1ull << g) - 1);
    }
}

#define AF (1ull << 10)
#define PAGE 3ull
#define BLOCK 1ull
#define AP(x) ((uint64_t)(x) << 6)
#define PXN (1ull << 53)
#define UXN (1ull << 54)

static void expect(const char *what, HvCpu *cpu, uint64_t va, HvAccess acc, int el, uint64_t want_pa, uint32_t want_fsc) {
    HvWalk w;
    hv_mmu_walk(cpu, va, acc, el, &w);
    checks++;
    if (w.fsc != want_fsc || (!want_fsc && w.pa != want_pa)) {
        printf("  FAIL %s: va %llx -> pa %llx fsc %02x, expected pa %llx fsc %02x\n", what, (unsigned long long)va,
               (unsigned long long)w.pa, w.fsc, (unsigned long long)want_pa, want_fsc);
        failures++;
    }
}

int main(void) {
    static HvMachine m;
    static HvCpu cpu;
    hv_machine_init(&m, 1 << 20);
    pt = aligned_alloc(65536, PT_SIZE);
    hv_machine_add_ram(&m, PT_PA, PT_SIZE, pt, 0);
    hv_cpu_reset(&cpu, &m);

    expect("mmu off", &cpu, 0x12345678, HV_ACC_WRITE, 1, 0x12345678, 0);

    /* 16 KB granule, 48-bit TTBR0 (four levels), 39-bit TTBR1 (three levels). */
    uint64_t r0 = alloc_table(), r1 = alloc_table();
    cpu.ttbr0_el1 = r0 | 0x1234ull << 48; /* an ASID must not leak into the table address */
    cpu.ttbr1_el1 = r1;
    cpu.mair_el1 = 0x00ff;
    cpu.tcr_el1 = 16 | 2u << 14 | 25u << 16 | 1u << 30 | 5ull << 32 | 1ull << 37; /* T0SZ 16, TG0 16K, T1SZ 25, TG1 16K, TBI0 */
    cpu.sctlr_el1 |= 1;

    map(r0, 14, 48, 0x100004000, 3, 0x40008000 | PAGE | AF | AP(0), 0);
    expect("16K page", &cpu, 0x100004abc, HV_ACC_READ, 1, 0x40008abc, 0);
    expect("16K page, next page unmapped", &cpu, 0x100008000, HV_ACC_READ, 1, 0, HV_FSC_TRANSLATION(3));
    expect("16K, no level 1 table", &cpu, 0x7f0000000000, HV_ACC_READ, 1, 0, HV_FSC_TRANSLATION(1));
    expect("16K, VA above T0SZ", &cpu, 0x0001000000000000, HV_ACC_READ, 1, 0, HV_FSC_TRANSLATION(0));
    expect("TBI0 ignores the top byte", &cpu, 0x5a00000100004010, HV_ACC_READ, 1, 0x40008010, 0);
    cpu.tcr_el1 &= ~(1ull << 37);
    expect("without TBI0 a tag faults", &cpu, 0x5a00000100004010, HV_ACC_READ, 1, 0, HV_FSC_TRANSLATION(0));
    cpu.tcr_el1 |= 1ull << 37;

    map(r0, 14, 48, 0x200000000, 2, 0x82000000 | BLOCK | AF, 0);
    expect("16K: 32 MB block", &cpu, 0x201234560, HV_ACC_WRITE, 1, 0x83234560, 0);
    map(r0, 14, 48, 0x300000000, 3, 0x40010000 | PAGE, 0);
    expect("access flag clear", &cpu, 0x300000000, HV_ACC_READ, 1, 0, HV_FSC_ACCESS(3));

    /* Permissions. */
    map(r0, 14, 48, 0x100010000, 3, 0x40010000 | PAGE | AF | AP(2), 0); /* EL1 RO */
    expect("EL1 RO: read", &cpu, 0x100010000, HV_ACC_READ, 1, 0x40010000, 0);
    expect("EL1 RO: write", &cpu, 0x100010000, HV_ACC_WRITE, 1, 0, HV_FSC_PERMISSION(3));
    expect("EL1 only: EL0 read", &cpu, 0x100004000, HV_ACC_READ, 0, 0, HV_FSC_PERMISSION(3));
    map(r0, 14, 48, 0x100014000, 3, 0x40014000 | PAGE | AF | AP(1), 0); /* EL0 and EL1 RW */
    expect("EL0 RW: EL0 write", &cpu, 0x100014000, HV_ACC_WRITE, 0, 0x40014000, 0);
    expect("EL0 RW: EL1 cannot execute", &cpu, 0x100014000, HV_ACC_EXEC, 1, 0, HV_FSC_PERMISSION(3));
    expect("EL0 RW: EL0 executes", &cpu, 0x100014000, HV_ACC_EXEC, 0, 0x40014000, 0);
    cpu.pstate_extra |= HV_PSTATE_PAN;
    expect("PAN: EL1 read of an EL0 page", &cpu, 0x100014000, HV_ACC_READ, 1, 0, HV_FSC_PERMISSION(3));
    expect("PAN: EL1 page unaffected", &cpu, 0x100004000, HV_ACC_READ, 1, 0x40008000, 0);
    expect("PAN: LDTR uses EL0 rights", &cpu, 0x100014000, HV_ACC_READ, 0, 0x40014000, 0);
    cpu.pstate_extra &= ~(uint64_t)HV_PSTATE_PAN;
    map(r0, 14, 48, 0x100018000, 3, 0x40018000 | PAGE | AF | AP(3) | UXN, 0); /* RO both, EL0 XN */
    expect("UXN: EL0 exec", &cpu, 0x100018000, HV_ACC_EXEC, 0, 0, HV_FSC_PERMISSION(3));
    expect("UXN: EL1 exec", &cpu, 0x100018000, HV_ACC_EXEC, 1, 0x40018000, 0);
    map(r0, 14, 48, 0x10001c000, 3, 0x4001c000 | PAGE | AF | PXN, 0);
    expect("PXN: EL1 exec", &cpu, 0x10001c000, HV_ACC_EXEC, 1, 0, HV_FSC_PERMISSION(3));
    cpu.sctlr_el1 |= 1u << 19;
    expect("WXN: writable is not executable", &cpu, 0x100004000, HV_ACC_EXEC, 1, 0, HV_FSC_PERMISSION(3));
    cpu.sctlr_el1 &= ~(1ull << 19);
    expect("no WXN: executable", &cpu, 0x100004000, HV_ACC_EXEC, 1, 0x40008000, 0);

    /* APTable on a new level 1 table: bit 61 = no EL0, bit 62 = no writes. */
    map(r0, 14, 48, 0x500000000000, 3, 0x40020000 | PAGE | AF | AP(1), 1ull << 61);
    expect("APTable no-EL0", &cpu, 0x500000000000, HV_ACC_READ, 0, 0, HV_FSC_PERMISSION(3));
    map(r0, 14, 48, 0x600000000000, 3, 0x40024000 | PAGE | AF | AP(0), 1ull << 62);
    expect("APTable read-only", &cpu, 0x600000000000, HV_ACC_WRITE, 1, 0, HV_FSC_PERMISSION(3));
    expect("APTable read-only: read", &cpu, 0x600000000000, HV_ACC_READ, 1, 0x40024000, 0);

    /* TTBR1, 39-bit: starts at level 1. */
    map(r1, 14, 39, 0xffffffc012340000, 3, 0x40028000 | PAGE | AF, 0);
    expect("TTBR1 page", &cpu, 0xffffffc012343210, HV_ACC_READ, 1, 0x4002b210, 0);
    expect("TTBR1: hole between ranges", &cpu, 0xffff000012340000, HV_ACC_READ, 1, 0, HV_FSC_TRANSLATION(0));
    cpu.tcr_el1 |= 1u << 23;
    expect("EPD1 disables TTBR1 walks", &cpu, 0xffffffc012343210, HV_ACC_READ, 1, 0, HV_FSC_TRANSLATION(0));
    cpu.tcr_el1 &= ~(1ull << 23);

    /* The TLB only takes 16 KB or larger mappings. */
    uint64_t pa;
    hv_tlb_flush(&cpu);
    hv_mmu_access(&cpu, 0x100004000, HV_ACC_READ, 1, &pa);
    checks++;
    /* 0x40008000 is not RAM here, so nothing may be filled */
    if ((*hv_tlb_ent(&cpu, (0x100004000 >> 14) & (HV_TLB_SIZE - 1), 1)).read_tag != HV_TLB_INVALID) {
        printf("  FAIL TLB filled for non-RAM\n");
        failures++;
    }

    /* 4 KB granule, 39-bit TTBR0 (levels 1-3): pages, 2 MB and 1 GB blocks. */
    pt_next = 2 * 65536;
    uint64_t r4 = alloc_table();
    cpu.ttbr0_el1 = r4;
    cpu.tcr_el1 = 25 | 0u << 14 | 1u << 23 | 5ull << 32;
    map(r4, 12, 39, 0x12345000, 3, 0x40001000 | PAGE | AF, 0);
    expect("4K page", &cpu, 0x12345678, HV_ACC_READ, 1, 0x40001678, 0);
    map(r4, 12, 39, 0x40200000, 2, 0x80400000 | BLOCK | AF, 0);
    expect("4K: 2 MB block", &cpu, 0x40212345, HV_ACC_READ, 1, 0x80412345, 0);
    map(r4, 12, 39, 0x4000000000 - 0x40000000, 1, 0xc0000000 | BLOCK | AF, 0);
    expect("4K: 1 GB block", &cpu, 0x3fc0123456, HV_ACC_READ, 1, 0xc0123456, 0);
    hv_tlb_flush(&cpu);
    hv_mmu_access(&cpu, 0x12345678, HV_ACC_READ, 1, &pa);
    checks++;
    if ((*hv_tlb_ent(&cpu, (0x12345678 >> 14) & (HV_TLB_SIZE - 1), 1)).read_tag != HV_TLB_INVALID) {
        printf("  FAIL a 4K mapping went into the 16K TLB\n");
        failures++;
    }

    /* 64 KB granule, 42-bit TTBR0 (levels 2-3): pages and 512 MB blocks. */
    uint64_t r6 = alloc_table();
    cpu.ttbr0_el1 = r6;
    cpu.tcr_el1 = 22 | 1u << 14 | 1u << 23 | 5ull << 32;
    map(r6, 16, 42, 0x123450000, 3, 0x40050000 | PAGE | AF, 0);
    expect("64K page", &cpu, 0x12345abcd, HV_ACC_READ, 1, 0x4005abcd, 0);
    map(r6, 16, 42, 0x200000000, 2, 0x60000000 | BLOCK | AF, 0);
    expect("64K: 512 MB block", &cpu, 0x21234abcd, HV_ACC_READ, 1, 0x7234abcd, 0);
    map(r6, 16, 42, 0x300000000, 3, 0x40060000 | BLOCK | AF, 0);
    expect("block descriptor at level 3 is invalid", &cpu, 0x300000000, HV_ACC_READ, 1, 0, HV_FSC_TRANSLATION(3));

    /* TLBI RVAE1IS: 6 pages of 16 KB from the base drop, the ones after stay; also for a kernel range. */
    {
        static const uint64_t bases[2] = {0x100000000ull, 0xfffffe0012340000ull};
        for (int b = 0; b < 2; b++) {
            hv_tlb_flush(&cpu);
            int table = (bases[b] >> 55) & 1 ? 1 : 0;
            for (int k = 0; k < 8; k++) {
                uint64_t va = bases[b] + ((uint64_t)k << 14);
                HvTlbEntry *t = &(*hv_tlb_ent(&cpu, (va >> 14) & (HV_TLB_SIZE - 1), table));
                t->read_tag = va - HV_TLB_ADJ;
                t->addend = 0;
            }
            cpu.el = 1;
            cpu.x[1] = 5ull << 48 | 2ull << 46 | 0ull << 44 | 2ull << 39 | ((bases[b] >> 14) & 0x1fffffffffull);
            hv_sysreg_sys(&cpu, 0xd5088221u, 0); /* tlbi rvae1is, x1 */
            for (int k = 0; k < 8; k++) {
                uint64_t va = bases[b] + ((uint64_t)k << 14);
                int dropped = (*hv_tlb_ent(&cpu, (va >> 14) & (HV_TLB_SIZE - 1), table)).read_tag == HV_TLB_INVALID;
                checks++;
                if (dropped != (k < 6)) {
                    printf("  FAIL range TLBI, base %llx page %d: %s\n", (unsigned long long)bases[b], k,
                           dropped ? "dropped" : "kept");
                    failures++;
                }
            }
        }
    }

    /* A big kernel-range TLBI drops the kernel pages in the range and keeps the rest, the processes' entries included
       (it used to flush everything). 512 pages goes page by page, 6144 pages by a scan. */
    {
        static const struct { uint32_t scale, num, pages; } ranges[2] = {{1, 7, 512}, {2, 2, 6144}};
        const uint64_t base = 0xfffffe0040000000ull;
        for (int r = 0; r < 2; r++) {
            hv_tlb_flush(&cpu);
            const int32_t ks[6] = {0, 1, (int32_t)ranges[r].pages / 2, (int32_t)ranges[r].pages - 1, (int32_t)ranges[r].pages, (int32_t)ranges[r].pages + 300};
            for (int k = 0; k < 6; k++) {
                uint64_t va = base + ((uint64_t)ks[k] << 14);
                HvTlbEntry *t = hv_tlb_ent(&cpu, (va >> 14) & (HV_TLB_SIZE - 1), 1);
                t->read_tag = va - HV_TLB_ADJ;
                t->addend = 0;
            }
            uint64_t uva = 0x100000000ull + ((uint64_t)7000 << 14); /* a process's page, in a slot outside the range's */
            HvTlbEntry *u = hv_tlb_ent(&cpu, (uva >> 14) & (HV_TLB_SIZE - 1), 0);
            u->read_tag = uva - HV_TLB_ADJ;
            u->addend = 0;
            cpu.el = 1;
            cpu.x[1] = 0ull << 48 | 2ull << 46 | (uint64_t)ranges[r].scale << 44 | (uint64_t)ranges[r].num << 39 | ((base >> 14) & 0x1fffffffffull);
            hv_sysreg_sys(&cpu, 0xd5088221u, 0); /* tlbi rvae1is, x1 */
            for (int k = 0; k < 6; k++) {
                uint64_t va = base + ((uint64_t)ks[k] << 14);
                int dropped = hv_tlb_ent(&cpu, (va >> 14) & (HV_TLB_SIZE - 1), 1)->read_tag == HV_TLB_INVALID;
                checks++;
                if (dropped != (ks[k] < (int32_t)ranges[r].pages)) {
                    printf("  FAIL big kernel range %u pages, page %d: %s\n", ranges[r].pages, ks[k], dropped ? "dropped" : "kept");
                    failures++;
                }
            }
            checks++;
            if (hv_tlb_ent(&cpu, (uva >> 14) & (HV_TLB_SIZE - 1), 0)->read_tag == HV_TLB_INVALID) {
                printf("  FAIL big kernel range %u pages dropped a process's entry\n", ranges[r].pages);
                failures++;
            }
        }
    }

    /* A TLBI by VA and ASID forgets the indirect-branch cache entries of that page
       for that address space, and nothing else. Entries are filled the way the dispatcher does it. */
    {
        hv_cpu_reset(&cpu, &m);
        cpu.tcr_el1 = 0;
        cpu.ttbr0_el1 = 7ull << 48;
        hv_ijc_switch(&cpu);
        uint64_t salt = cpu.ijc_salt[0];
        static const uint64_t vas[3] = {0x100004010ull, 0x100007ffcull, 0x100008010ull}; /* page P twice, then page Q */
        uint64_t *ents[3];
        static HvBlock fake[3];
        for (int i = 0; i < 3; i++) {
            fake[i] = (HvBlock){.va = vas[i], .flags = 0, .code = (uint32_t *)(uintptr_t)0x1234};
            hv_ijc_fill(&cpu, &fake[i]);
            ents[i] = m.percpu[0].ijc + (size_t)hv_ijc_index(vas[i] ^ salt) * 2;
        }
        hv_tlb_flush_va_asid(&cpu, 0x100004000ull, 7);
        int want[3] = {0, 0, 1};
        for (int i = 0; i < 3; i++) {
            checks++;
            if ((ents[i][0] != 0) != want[i]) {
                printf("  FAIL branch cache after TLBI of page P: entry %d %s\n", i, ents[i][0] ? "kept" : "dropped");
                failures++;
            }
        }
        /* Another address space's TLBI of page Q leaves this one's entry alone. */
        hv_tlb_flush_va_asid(&cpu, 0x100008000ull, 9);
        checks++;
        if (!ents[2][0]) {
            printf("  FAIL another ASID's TLBI dropped our entry\n");
            failures++;
        }
        hv_tlb_flush_va_asid(&cpu, 0x100008000ull, -1); /* all address spaces */
        checks++;
        uint64_t now = vas[2] ^ cpu.ijc_salt[0]; /* what a lookup would compare against now */
        if (m.percpu[0].ijc[(size_t)hv_ijc_index(now) * 2] == now) {
            printf("  FAIL an all-ASID TLBI kept the entry\n");
            failures++;
        }
    }

    /* A big range TLBI (128 MB) of one address space: only the cached targets in it go, found through the 32 MB chunks
       that hold any; the address space keeps its id (and so its other targets). */
    {
        hv_cpu_reset(&cpu, &m);
        cpu.tcr_el1 = 0;
        cpu.ttbr0_el1 = 7ull << 48;
        hv_ijc_switch(&cpu);
        uint64_t salt = cpu.ijc_salt[0];
        uint16_t cid = cpu.asid_cid[7];
        static const uint64_t vas[2] = {0x203004010ull, 0x210000020ull}; /* in the range (its 3rd chunk), past it */
        uint64_t *ents[2];
        static HvBlock fake[2];
        for (int i = 0; i < 2; i++) {
            fake[i] = (HvBlock){.va = vas[i], .flags = 0, .code = (uint32_t *)(uintptr_t)0x1234};
            hv_ijc_fill(&cpu, &fake[i]);
            ents[i] = m.percpu[0].ijc + (size_t)hv_ijc_index(vas[i] ^ salt) * 2;
        }
        hv_ijc_drop_range(&cpu, 0x200000000ull, 128ull << 20, 7);
        checks += 3;
        if (ents[0][0] == (vas[0] ^ salt)) printf("  FAIL big range kept a target in it\n"), failures++;
        if (ents[1][0] != (vas[1] ^ salt)) printf("  FAIL big range dropped a target past it\n"), failures++;
        if (cpu.asid_cid[7] != cid) printf("  FAIL big range dropped the whole address space\n"), failures++;
    }

    /* A code change drops the cached branch targets into its page (HV_IJC_PAGES): noted under one VA page when they
       were cached, just that page's; noted under none, none (only the return predictions); under several, all. */
    {
        hv_cpu_reset(&cpu, &m);
        cpu.tcr_el1 = 0;
        cpu.ttbr0_el1 = 7ull << 48;
        hv_ijc_switch(&cpu);
        HvCpu *self = hv_self;
        hv_self = &cpu;
        int knob = hv_knob[HV_K_IJCPAGES];
        hv_knob[HV_K_IJCPAGES] = 1;
        HvRam *r = &m.ram[0];
        const uint64_t P = 0x100004000ull, Q = 0x100008000ull;
        static HvBlock fb[2];
        struct { uint64_t note; int p_kept, q_kept; const char *what; } cases[] = {
            {P | 2, 0, 1, "noted at P"}, {0, 1, 1, "noted nowhere"}, {~0ull, 0, 0, "noted at several"}};
        for (int c = 0; c < 3; c++) {
            uint64_t salt = cpu.ijc_salt[0];
            uint64_t *ents[2];
            for (int i = 0; i < 2; i++) {
                fb[i] = (HvBlock){.va = (i ? Q + 0x20 : P + 0x10), .flags = 0, .code = (uint32_t *)(uintptr_t)0x1234};
                hv_ijc_fill(&cpu, &fb[i]);
                ents[i] = m.percpu[0].ijc + (size_t)hv_ijc_index(fb[i].va ^ salt) * 2;
            }
            r->code_pages[10] = 1;
            r->ijc_va[10] = cases[c].note;
            hv_invalidate_code(&m, r->pa + (10ull << 14));
            for (int i = 0; i < 2; i++) {
                int kept = ents[i][0] == (fb[i].va ^ salt) && cpu.ijc_salt[0] == salt;
                checks++;
                if (kept != (i ? cases[c].q_kept : cases[c].p_kept)) {
                    printf("  FAIL code change %s: the target in %s %s\n", cases[c].what, i ? "Q" : "P", kept ? "kept" : "dropped");
                    failures++;
                }
            }
            checks++;
            if (r->ijc_va[10] != 0) printf("  FAIL code change left the page's note\n"), failures++;
        }
        hv_knob[HV_K_IJCPAGES] = knob;
        hv_self = self;
    }

    /* Another CPU's big range TLBI (64 user pages of ASID 5: over the 32 done from here by default). With HV_RANGE_DROP it is done from here in one pass:
       the second CPU's TLB slots and stashed entries in the range go, the rest stay, nothing is queued for it.
       Without, it is queued for that CPU to do (as before) and only happens when it does. */
    {
        static HvCpu cpu2;
        hv_cpu_reset_index(&cpu2, &m, 1);
        cpu2.sctlr_el1 = cpu.sctlr_el1, cpu2.tcr_el1 = cpu.tcr_el1, cpu2.mair_el1 = cpu.mair_el1;
        cpu2.ttbr0_el1 = 5ull << 48 | (cpu.ttbr0_el1 & 0x0000fffffffffffeull); /* running ASID 5: its live entries are that one's */
        HvPerCpu *p2 = cpu2.pc_state;
        const uint64_t base = 0x100000000ull;
        const int32_t ks[6] = {0, 1, 32, 63, 64, 100};
        for (int knob = 1; knob >= 0; knob--) {
            hv_knob[HV_K_RANGEDROP] = knob ? 1024 : 0;
            hv_tlb_flush(&cpu2);
            for (int k = 0; k < 6; k++) {
                uint64_t va = base + ((uint64_t)ks[k] << 14);
                HvTlbEntry *t = hv_tlb_ent(&cpu2, (va >> 14) & (HV_TLB_SIZE - 1), 0);
                t->read_tag = va - HV_TLB_ADJ;
                t->addend = 0;
            }
            /* a stashed (switched-out) address space of ASID 5: one page inside the range, one past it */
            HvTlbStash *st = &p2->stash[0];
            st->ttbr0 = 5ull << 48 | 0x4000;
            st->n = 2;
            st->ent[0] = (HvTlbEntry){.read_tag = base + (40ull << 14) - HV_TLB_ADJ, .write_tag = HV_TLB_INVALID, .exec_tag = HV_TLB_INVALID};
            st->ent[1] = (HvTlbEntry){.read_tag = base + (200ull << 14) - HV_TLB_ADJ, .write_tag = HV_TLB_INVALID, .exec_tag = HV_TLB_INVALID};
            p2->nreq = 0, p2->req_flags = 0;
            cpu.el = 1;
            cpu.x[1] = 5ull << 48 | 2ull << 46 | 0ull << 44 | 31ull << 39 | ((base >> 14) & 0x1fffffffffull); /* (31+1) << 1 = 64 pages */
            hv_sysreg_sys(&cpu, 0xd5088221u, 0); /* tlbi rvae1is, x1 */
            int queued = p2->nreq != 0 || p2->req_flags != 0;
            checks++;
            if (queued == knob) {
                printf("  FAIL remote range, HV_RANGE_DROP %d: %s\n", knob ? 1024 : 0, queued ? "queued for the other CPU" : "not queued");
                failures++;
            }
            if (!knob) hv_cpu_process_requests(&cpu2); /* what it does when it next runs */
            for (int k = 0; k < 6; k++) {
                uint64_t va = base + ((uint64_t)ks[k] << 14);
                int dropped = hv_tlb_ent(&cpu2, (va >> 14) & (HV_TLB_SIZE - 1), 0)->read_tag == HV_TLB_INVALID;
                checks++;
                if (dropped != (ks[k] < 64)) {
                    printf("  FAIL remote range (knob %d), page %d: %s\n", knob, ks[k], dropped ? "dropped" : "kept");
                    failures++;
                }
            }
            if (knob) { /* the stash is the dropping CPU's job only on this path; the target's own TLBI handles it otherwise */
                checks += 2;
                if (st->ent[0].read_tag != HV_TLB_INVALID) printf("  FAIL remote range kept a stashed page in the range\n"), failures++;
                if (st->ent[1].read_tag == HV_TLB_INVALID) printf("  FAIL remote range dropped a stashed page past the range\n"), failures++;
            }
            st->ttbr0 = 0, st->n = 0;
        }
        hv_knob[HV_K_RANGEDROP] = 0;
        /* A target in the middle of a page walk (of a page in the range): raced, so it is queued instead. */
        hv_tlb_flush(&cpu2);
        p2->walk_page[(p2->walk_seq >> 1) & 7] = base + (3ull << 14);
        __atomic_store_n(&p2->walk_seq, p2->walk_seq | 1, __ATOMIC_SEQ_CST);
        checks++;
        if (hv_tlbi_drop_remote_range(&cpu2, base, 64, 5) != 0) printf("  FAIL remote range ignored a page walk in progress\n"), failures++;
        __atomic_store_n(&p2->walk_seq, p2->walk_seq + 1, __ATOMIC_SEQ_CST);
    }

    /* HV_WALK_LOG: a drop races only a walk of one of its pages, or a process switch's (any page); a walk of another
       page is let through. Walks under way, set up as walk_begin does (the page logged, then walk_seq odd). */
    {
        static HvCpu cpu3;
        hv_cpu_reset_index(&cpu3, &m, 1);
        HvPerCpu *p3 = cpu3.pc_state;
        const uint64_t P = 0x100004000ull, Q = 0x100008000ull;
        struct { const char *what; uint64_t walked; int knob, range, want; } cases[] = {
            {"a walk of another page", Q, 1, 0, 1},       {"a walk of the page", P + 0x10, 1, 0, 0},
            {"a process switch", ~0ull, 1, 0, 0},         {"a walk of another page, knob off", Q, 0, 0, 0},
            {"a range with the walked page in it", Q, 1, 1, 0}, {"a range without it", Q + (8ull << 14), 1, 1, 1},
            {"a tagged walk of another page", 0x5a00000000000000ull | Q, 1, 0, 1},
            {"a tagged walk of the page", 0x5a00000000000000ull | P, 1, 0, 0},
        };
        for (size_t c = 0; c < sizeof cases / sizeof cases[0]; c++) {
            hv_knob[HV_K_WALKLOG] = cases[c].knob;
            uint32_t seq = __atomic_load_n(&p3->walk_seq, __ATOMIC_SEQ_CST) & ~1u;
            uint64_t pg = cases[c].walked == ~0ull ? ~0ull : cases[c].walked & 0x00ffffffffffc000ull;
            p3->walk_page[(seq >> 1) & 7] = pg;
            __atomic_store_n(&p3->walk_seq, seq + 1, __ATOMIC_SEQ_CST);
            int r = cases[c].range ? hv_tlbi_drop_remote_range(&cpu3, P, 2, 5) : hv_tlbi_drop_remote_va(&cpu3, P, 5);
            __atomic_store_n(&p3->walk_seq, seq + 2, __ATOMIC_SEQ_CST);
            checks++;
            if ((r == 1) != cases[c].want) {
                printf("  FAIL remote drop during %s: %s\n", cases[c].what, r == 1 ? "done" : "raced");
                failures++;
            }
        }
        hv_knob[HV_K_WALKLOG] = 1;
    }

    printf("%d checks, %d failures\n", checks, failures);
    return failures != 0;
}
