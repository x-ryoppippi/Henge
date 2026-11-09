#include "memop.h"

#include <stdio.h>
#include <string.h>

#include "exception.h"
#include "machine.h"
#include "mmu.h"

/* Decoding follows the "Loads and Stores" encoding index of the Arm ARM. */

static inline int64_t sext(uint64_t v, int bits) {
    return (int64_t)(v << (64 - bits)) >> (64 - bits);
}

static int decode_vector(uint32_t insn, HvMemOp *op) {
    uint32_t q = (insn >> 30) & 1, l = (insn >> 22) & 1, size = (insn >> 10) & 3;
    int single = (insn >> 24) & 1, post = (insn >> 23) & 1;
    op->kind = single ? HV_MEM_VEC_SINGLE : HV_MEM_VEC_MULTI;
    op->vec = 1;
    op->vq = q;
    op->load = l;
    op->store = !l;
    if (!post && op->rm != 0) return -1;
    if (single) {
        uint32_t opcode = (insn >> 13) & 7, s = (insn >> 12) & 1, r = (insn >> 21) & 1;
        uint32_t scale = opcode >> 1;
        op->selem = (((opcode & 1) << 1) | r) + 1;
        switch (scale) {
        case 3:
            if (!l || s) return -1;
            op->index = 0xff;
            op->size = size;
            break;
        case 0:
            op->index = q << 3 | s << 2 | size;
            op->size = 0;
            break;
        case 1:
            if (size & 1) return -1;
            op->index = q << 2 | s << 1 | size >> 1;
            op->size = 1;
            break;
        default:
            if (size & 2) return -1;
            if (!(size & 1)) {
                op->index = q << 1 | s;
                op->size = 2;
            } else {
                if (s) return -1;
                op->index = q;
                op->size = 3;
            }
            break;
        }
        op->bytes = op->selem << op->size;
    } else {
        if ((insn >> 21) & 1) return -1;
        static const uint8_t rpt[16] = {1, 0, 4, 0, 1, 0, 3, 1, 1, 0, 2, 0, 0, 0, 0, 0};
        static const uint8_t selem[16] = {4, 0, 1, 0, 3, 0, 1, 1, 2, 0, 1, 0, 0, 0, 0, 0};
        uint32_t opcode = (insn >> 12) & 0xf;
        if (!rpt[opcode]) return -1;
        if (size == 3 && !q && selem[opcode] > 1) return -1;
        op->rpt = rpt[opcode];
        op->selem = selem[opcode];
        op->size = size;
        op->bytes = (uint32_t)op->rpt * op->selem * (q ? 16 : 8);
    }
    if (post) {
        op->mode = HV_ADDR_POST;
        op->imm = op->bytes; /* used when Rm is 31; otherwise Xm is the amount */
    } else {
        op->mode = HV_ADDR_BASE;
        op->rm = 31;
    }
    return 0;
}

static int decode_exclusive(uint32_t insn, HvMemOp *op) {
    uint32_t size = insn >> 30, o2 = (insn >> 23) & 1, l = (insn >> 22) & 1, o1 = (insn >> 21) & 1;
    op->mode = HV_ADDR_BASE;
    op->align = 1;
    op->size = size;
    op->rm = 31;
    if (!o2 && !o1) {
        op->kind = HV_MEM_EXCL;
        op->load = l;
        op->store = !l;
        op->bytes = 1u << size;
        return 0;
    }
    if (!o2 && o1) {
        op->size = 2 + (size & 1);
        op->bytes = 2u << op->size;
        if (size >> 1) {
            op->kind = HV_MEM_EXCL;
            op->load = l;
            op->store = !l;
            op->rt2 = (insn >> 10) & 0x1f;
            return 0;
        }
        if ((op->rs & 1) || (op->rt & 1)) return -1;
        op->kind = HV_MEM_CASP;
        op->load = op->store = 1;
        return 0;
    }
    if (o2 && !o1) {
        op->kind = HV_MEM_ORDERED;
        op->load = l;
        op->store = !l;
        op->bytes = 1u << size;
        return 0;
    }
    op->kind = HV_MEM_CAS;
    op->load = op->store = 1;
    op->bytes = 1u << size;
    return 0;
}

int hv_mem_decode(uint32_t insn, HvMemOp *op) {
    memset(op, 0, sizeof *op);
    op->rt = insn & 0x1f;
    op->rn = (insn >> 5) & 0x1f;
    op->rt2 = (insn >> 10) & 0x1f;
    op->rm = op->rs = (insn >> 16) & 0x1f;
    uint32_t size = insn >> 30, v = (insn >> 26) & 1;

    if ((insn & 0x3b000000) == 0x18000000) { /* load register (literal) */
        op->kind = HV_MEM_LITERAL;
        op->mode = HV_ADDR_LITERAL;
        op->load = 1;
        op->vec = v;
        op->rm = 31;
        op->imm = sext((insn >> 5) & 0x7ffff, 19) * 4;
        if (v) {
            if (size == 3) return -1;
            op->size = 2 + size;
        } else {
            if (size == 3) return -1; /* PRFM is dropped before here */
            op->size = size == 1 ? 3 : 2;
            if (size == 2) op->sign = 64;
        }
        op->bytes = 1u << op->size;
        return 0;
    }
    if ((insn & 0xbe000000) == 0x0c000000) { /* SIMD structures */
        if (!v) return -1;
        return decode_vector(insn, op);
    }
    if ((insn & 0x3f000000) == 0x08000000) { /* exclusive, ordered, compare and swap */
        if (v) return -1;
        return decode_exclusive(insn, op);
    }
    if ((insn & 0x3f200c00) == 0x19000000) { /* LDAPUR/STLUR (RCpc, unscaled) */
        uint32_t opc = (insn >> 22) & 3;
        op->kind = HV_MEM_ORDERED;
        op->mode = HV_ADDR_OFFSET;
        op->imm = sext((insn >> 12) & 0x1ff, 9);
        op->size = size;
        op->rm = 31;
        op->bytes = 1u << size;
        op->store = opc == 0;
        op->load = opc != 0;
        if (opc == 2) {
            if (size == 3) return -1;
            op->sign = 64;
        } else if (opc == 3) {
            if (size >= 2) return -1;
            op->sign = 32;
        }
        return 0;
    }
    if ((insn & 0x38000000) == 0x28000000) { /* register pairs */
        uint32_t opc = insn >> 30, l = (insn >> 22) & 1, mode = (insn >> 23) & 3;
        op->kind = HV_MEM_PAIR;
        op->vec = v;
        op->load = l;
        op->store = !l;
        op->rm = 31;
        if (opc == 3) return -1;
        if (v) {
            op->size = 2 + opc;
        } else if (opc == 1) {
            if (!l || mode == 0) return -1; /* STGP (MTE) is not exposed */
            op->size = 2;
            op->sign = 64;
        } else {
            op->size = opc ? 3 : 2;
        }
        op->imm = sext((insn >> 15) & 0x7f, 7) << op->size;
        op->mode = mode == 1 ? HV_ADDR_POST : mode == 3 ? HV_ADDR_PRE : HV_ADDR_OFFSET;
        op->bytes = 2u << op->size;
        return 0;
    }
    if ((insn & 0x38000000) != 0x38000000) return -1;

    /* Single register: unsigned offset, imm9 forms, register offset, atomics, LDRA*. */
    uint32_t opc = (insn >> 22) & 3;
    op->kind = HV_MEM_SINGLE;
    op->vec = v;
    op->rm = 31;
    int atomic = !((insn >> 24) & 1) && ((insn >> 21) & 1) && ((insn >> 10) & 3) == 0;
    int ldra = !((insn >> 24) & 1) && ((insn >> 21) & 1) && ((insn >> 10) & 1);
    if (atomic) {
        if (v) return -1;
        uint32_t o3 = (insn >> 15) & 1, aop = (insn >> 12) & 7, a = (insn >> 23) & 1, r = (insn >> 22) & 1;
        op->mode = HV_ADDR_BASE;
        op->size = size;
        op->bytes = 1u << size;
        op->align = 1;
        if (o3 && aop == 4 && a && !r && op->rs == 31) { /* LDAPR */
            op->kind = HV_MEM_ORDERED;
            op->load = 1;
            return 0;
        }
        if (o3 && aop != 0) return -1; /* LD64B and friends */
        op->kind = HV_MEM_ATOMIC;
        op->atomic_op = o3 << 3 | aop;
        op->load = op->store = 1;
        return 0;
    }
    if (ldra) {
        if (v || size != 3) return -1;
        op->kind = HV_MEM_LDRA;
        op->load = 1;
        op->size = 3;
        op->bytes = 8;
        op->imm = sext(((insn >> 22) & 1) << 9 | ((insn >> 12) & 0x1ff), 10) * 8;
        op->mode = (insn >> 11) & 1 ? HV_ADDR_PRE : HV_ADDR_OFFSET;
        return 0;
    }
    if (v) {
        if (opc & 2) {
            if (size != 0) return -1;
            op->size = 4;
        } else {
            op->size = size;
        }
        op->load = opc & 1;
        op->store = !(opc & 1);
    } else {
        op->size = size;
        if (opc == 0) {
            op->store = 1;
        } else if (opc == 1) {
            op->load = 1;
        } else if (opc == 2) {
            if (size == 3) return -1; /* PRFM forms are dropped before here */
            op->load = 1;
            op->sign = 64;
        } else {
            if (size >= 2) return -1;
            op->load = 1;
            op->sign = 32;
        }
    }
    op->bytes = 1u << op->size;
    if ((insn >> 24) & 1) {
        op->mode = HV_ADDR_OFFSET;
        op->imm = (int64_t)((insn >> 10) & 0xfff) << op->size;
        return 0;
    }
    if ((insn >> 21) & 1) { /* register offset */
        uint32_t option = (insn >> 13) & 7;
        if (((insn >> 10) & 3) != 2 || !(option & 2)) return -1;
        op->mode = HV_ADDR_REG;
        op->rm = (insn >> 16) & 0x1f;
        op->ext = option;
        op->shift = (insn >> 12) & 1 ? op->size : 0;
        return 0;
    }
    op->imm = sext((insn >> 12) & 0x1ff, 9);
    switch ((insn >> 10) & 3) {
    case 0: op->mode = HV_ADDR_OFFSET; break;
    case 1: op->mode = HV_ADDR_POST; break;
    case 2:
        if (v) return -1;
        op->mode = HV_ADDR_OFFSET;
        op->unpriv = 1;
        break;
    default: op->mode = HV_ADDR_PRE; break;
    }
    return 0;
}

/* ---- C-side access ---- */

int hv_vmem_xfer(HvCpu *cpu, uint64_t va, void *buf, uint32_t len, int write, int el, uint64_t pc) {
    uint8_t *b = buf;
    uint64_t w = cpu->m->watch_va;
    if (write && w && w - va < len) {
        uint64_t v = 0;
        memcpy(&v, b + (w - va), len - (w - va) < 8 ? len - (w - va) : 8);
        fprintf(stderr, "hengevm: watch: write to %llx = %llx (%u-byte store at %llx) by pc %llx x30 %llx\n",
                (unsigned long long)w, (unsigned long long)v, len, (unsigned long long)va, (unsigned long long)pc,
                (unsigned long long)cpu->x[30]);
    }
    while (len) {
        uint32_t chunk = 4096 - (uint32_t)(va & 4095);
        if (chunk > len) chunk = len;
        uint64_t pa;
        int had_entry = 0;
        if (cpu->m->stats_interval) { /* --stats: was the TLB entry there before this access? */
            HvTlbEntry *t = hv_tlb_ent(cpu, (va >> HV_TLB_PAGE_BITS) & (HV_TLB_SIZE - 1), hv_tlb_table(cpu, el));
            uint64_t tag = write ? t->write_tag : t->read_tag;
            had_entry = tag != HV_TLB_INVALID && ((va - tag) >> HV_TLB_PAGE_BITS) == 0 ? 1 :
                        tag != HV_TLB_INVALID && ((va - tag - HV_TLB_ADJ) >> HV_TLB_PAGE_BITS) == 0 ? 2 : 0;
            if (!had_entry) {
                int other = t->read_tag != HV_TLB_INVALID || t->write_tag != HV_TLB_INVALID || t->exec_tag != HV_TLB_INVALID;
                cpu->m->stats.tlb_miss_kind[(el ? 2 : 0) + other]++;
            }
        }
        uint32_t fsc = hv_mmu_access(cpu, va, write ? HV_ACC_WRITE : HV_ACC_READ, el, &pa);
        if (fsc) {
            hv_raise_abort(cpu, 0, va, fsc, write, fsc >= 0x14 && fsc <= 0x17, pc);
            return 1;
        }
        int rc = 0;
        if (cpu->m->stats_interval) {
            HvRam *ram = hv_ram_lookup(cpu->m, pa);
            uint64_t inpage = va & ((1ull << HV_TLB_PAGE_BITS) - 1);
            int why = !ram ? 0 : write && ram->code_pages[(pa - ram->pa) >> HV_TLB_PAGE_BITS] ? 1
                    : inpage + len > (1ull << HV_TLB_PAGE_BITS) - HV_TLB_ADJ ? 2 : had_entry == 0 ? 3 : had_entry == 2 ? 5 : 4;
            cpu->m->stats.slow_why[why]++;
            if (why == 3) { /* a miss the light refill could not fill: a small guest page? */
                HvWalk w;
                hv_mmu_walk(cpu, va, write ? HV_ACC_WRITE : HV_ACC_READ, el, &w);
                if (!w.fsc && w.block_size < (1ull << HV_TLB_PAGE_BITS)) {
                    cpu->m->stats.small_pages++;
                    uint64_t key = (va >> 20) << 1 | (el ? 1 : 0);
                    for (int i = 0; i < 8; i++) {
                        if (cpu->m->stats.small_at[i].mb == key) { cpu->m->stats.small_at[i].n++; break; }
                        if (!cpu->m->stats.small_at[i].mb) { cpu->m->stats.small_at[i].mb = key; cpu->m->stats.small_at[i].n = 1; break; }
                    }
                }
            }
        }
        if (hv_phys_ptr(cpu->m, pa, chunk)) {
            rc = write ? hv_phys_write(cpu->m, pa, b, chunk) : hv_phys_read(cpu->m, pa, b, chunk);
        } else {
            for (uint32_t off = 0; off < chunk && !rc; off += 8) { /* MMIO: at most 8 bytes at a time */
                uint32_t piece = chunk - off < 8 ? chunk - off : 8;
                rc = write ? hv_phys_write(cpu->m, pa + off, b + off, piece) : hv_phys_read(cpu->m, pa + off, b + off, piece);
            }
        }
        if (rc) {
            if (cpu->m->log)
                fprintf(stderr, "unmapped %s: pa %llx (va %llx) size %u pc %llx\n", write ? "write" : "read",
                        (unsigned long long)pa, (unsigned long long)va, chunk, (unsigned long long)pc);
            hv_raise_abort(cpu, 0, va, HV_FSC_EXTERNAL, write, 0, pc);
            return 1;
        }
        va += chunk;
        b += chunk;
        len -= chunk;
    }
    return 0;
}

static uint64_t gpr(HvCpu *cpu, int r) {
    return r == 31 ? 0 : cpu->x[r];
}

static void set_gpr(HvCpu *cpu, int r, uint64_t v) {
    if (r != 31) cpu->x[r] = v;
}

static uint64_t base_reg(HvCpu *cpu, int r) {
    return r == 31 ? cpu->sp : cpu->x[r];
}

static void set_base(HvCpu *cpu, int r, uint64_t v) {
    if (r == 31) cpu->sp = v;
    else cpu->x[r] = v;
}

static uint8_t *vreg(HvCpu *cpu, int r) {
    return (uint8_t *)&cpu->v[r & 31];
}

static uint64_t extend(uint64_t v, const HvMemOp *op) {
    int bits = 8 << op->size;
    if (op->sign == 64) return (uint64_t)sext(v, bits);
    if (op->sign == 32) return (uint32_t)sext(v, bits);
    return bits == 64 ? v : v & ((1ull << bits) - 1);
}

static uint64_t le(const uint8_t *p, int bytes) {
    uint64_t v = 0;
    memcpy(&v, p, bytes);
    return v;
}

static int alignment_fault(HvCpu *cpu, uint64_t va, int write, uint64_t pc) {
    hv_raise_abort(cpu, 0, va, HV_FSC_ALIGNMENT, write, 0, pc);
    return 1;
}

/* For atomics: a host pointer when va is RAM the guest may write, else NULL with
   *mmio_pa set. Returns -1 after an abort. */
static int rmw_target(HvCpu *cpu, uint64_t va, uint32_t bytes, int el, uint64_t pc, void **host, uint64_t *mmio_pa) {
    uint64_t pa;
    uint32_t fsc = hv_mmu_access(cpu, va, HV_ACC_WRITE, el, &pa);
    if (fsc) {
        hv_raise_abort(cpu, 0, va, fsc, 1, fsc >= 0x14 && fsc <= 0x17, pc);
        return -1;
    }
    HvRam *ram = hv_ram_lookup(cpu->m, pa);
    if (ram && ram->readonly) { /* ROM: nothing to write; treat like a device */
        *host = NULL;
        *mmio_pa = pa;
        return 0;
    }
    *host = hv_phys_ptr(cpu->m, pa, bytes);
    if (*host) {
        /* Same rule as other writes: drop translations made from this page. */
        uint64_t pg = (pa - ram->pa) >> HV_TLB_PAGE_BITS;
        if (ram->code_pages[pg]) hv_invalidate_code(cpu->m, pa);
    }
    *mmio_pa = pa;
    return 0;
}

static int cas_host(void *p, int size, uint64_t *expected, uint64_t desired) {
    switch (size) {
    case 0: { uint8_t e = (uint8_t)*expected; int ok = __atomic_compare_exchange_n((uint8_t *)p, &e, (uint8_t)desired, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST); *expected = e; return ok; }
    case 1: { uint16_t e = (uint16_t)*expected; int ok = __atomic_compare_exchange_n((uint16_t *)p, &e, (uint16_t)desired, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST); *expected = e; return ok; }
    case 2: { uint32_t e = (uint32_t)*expected; int ok = __atomic_compare_exchange_n((uint32_t *)p, &e, (uint32_t)desired, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST); *expected = e; return ok; }
    default: return __atomic_compare_exchange_n((uint64_t *)p, expected, desired, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    }
}

static uint64_t load_host(void *p, int size) {
    switch (size) {
    case 0: return __atomic_load_n((uint8_t *)p, __ATOMIC_SEQ_CST);
    case 1: return __atomic_load_n((uint16_t *)p, __ATOMIC_SEQ_CST);
    case 2: return __atomic_load_n((uint32_t *)p, __ATOMIC_SEQ_CST);
    default: return __atomic_load_n((uint64_t *)p, __ATOMIC_SEQ_CST);
    }
}

static uint64_t atomic_result(int aop, int size, uint64_t old, uint64_t s) {
    int bits = 8 << size;
    int64_t so = sext(old, bits), ss = sext(s, bits);
    uint64_t mask = bits == 64 ? ~0ull : (1ull << bits) - 1;
    old &= mask;
    s &= mask;
    switch (aop) {
    case 0: return (old + s) & mask;             /* LDADD */
    case 1: return old & ~s;                     /* LDCLR */
    case 2: return old ^ s;                      /* LDEOR */
    case 3: return old | s;                      /* LDSET */
    case 4: return (so > ss ? old : s);          /* LDSMAX */
    case 5: return (so < ss ? old : s);          /* LDSMIN */
    case 6: return old > s ? old : s;            /* LDUMAX */
    case 7: return old < s ? old : s;            /* LDUMIN */
    default: return s;                           /* SWP */
    }
}

static int do_atomic(HvCpu *cpu, const HvMemOp *op, uint64_t va, int el, uint64_t pc) {
    int size = op->size;
    uint32_t bytes = 1u << size;
    if (va & (bytes - 1)) return alignment_fault(cpu, va, 1, pc);
    void *host;
    uint64_t pa;
    if (rmw_target(cpu, va, bytes, el, pc, &host, &pa)) return 1;
    uint64_t mask = size == 3 ? ~0ull : (1ull << (8 << size)) - 1;
    uint64_t old, s = gpr(cpu, op->rs);
    if (op->kind == HV_MEM_CAS) {
        uint64_t desired = gpr(cpu, op->rt) & mask;
        if (host) {
            old = s & mask;
            cas_host(host, size, &old, desired);
        } else {
            old = 0;
            if (hv_phys_read(cpu->m, pa, &old, bytes)) goto external;
            if (old == (s & mask) && hv_phys_write(cpu->m, pa, &desired, bytes)) goto external;
        }
        set_gpr(cpu, op->rs, old);
        return 0;
    }
    if (host) {
        old = load_host(host, size);
        while (!cas_host(host, size, &old, atomic_result(op->atomic_op, size, old, s))) {
        }
    } else {
        old = 0;
        if (hv_phys_read(cpu->m, pa, &old, bytes)) goto external;
        uint64_t n = atomic_result(op->atomic_op, size, old, s);
        if (hv_phys_write(cpu->m, pa, &n, bytes)) goto external;
    }
    set_gpr(cpu, op->rt, old & mask);
    return 0;
external:
    hv_raise_abort(cpu, 0, va, HV_FSC_EXTERNAL, 1, 0, pc);
    return 1;
}

static int do_casp(HvCpu *cpu, const HvMemOp *op, uint64_t va, int el, uint64_t pc) {
    uint32_t esize = 1u << op->size, bytes = esize * 2;
    if (va & (bytes - 1)) return alignment_fault(cpu, va, 1, pc);
    void *host;
    uint64_t pa;
    if (rmw_target(cpu, va, bytes, el, pc, &host, &pa)) return 1;
    uint64_t mask = esize == 8 ? ~0ull : 0xffffffffull;
    uint64_t c0 = gpr(cpu, op->rs) & mask, c1 = gpr(cpu, op->rs + 1) & mask;
    uint64_t n0 = gpr(cpu, op->rt) & mask, n1 = gpr(cpu, op->rt + 1) & mask;
    uint64_t o0, o1;
    if (host && esize == 4) {
        uint64_t old = c0 | c1 << 32;
        cas_host(host, 3, &old, n0 | n1 << 32);
        o0 = old & mask;
        o1 = old >> 32;
    } else if (host) {
        __uint128_t old = (__uint128_t)c1 << 64 | c0;
        __atomic_compare_exchange_n((__uint128_t *)host, &old, (__uint128_t)n1 << 64 | n0, 0, __ATOMIC_SEQ_CST,
                                    __ATOMIC_SEQ_CST);
        o0 = (uint64_t)old;
        o1 = (uint64_t)(old >> 64);
    } else {
        uint8_t buf[16] = {0};
        if (hv_phys_read(cpu->m, pa, buf, esize) || hv_phys_read(cpu->m, pa + esize, buf + esize, esize)) goto external;
        o0 = le(buf, esize);
        o1 = le(buf + esize, esize);
        if (o0 == c0 && o1 == c1) {
            if (hv_phys_write(cpu->m, pa, &n0, esize) || hv_phys_write(cpu->m, pa + esize, &n1, esize)) goto external;
        }
    }
    set_gpr(cpu, op->rs, o0);
    set_gpr(cpu, op->rs + 1, o1);
    return 0;
external:
    hv_raise_abort(cpu, 0, va, HV_FSC_EXTERNAL, 1, 0, pc);
    return 1;
}

static int do_exclusive(HvCpu *cpu, const HvMemOp *op, uint64_t va, int el, uint64_t pc) {
    uint32_t bytes = op->bytes;
    int pair = bytes == (2u << op->size); /* LDXP/STXP: decode sets bytes to two registers' worth */
    if (va & (bytes - 1)) return alignment_fault(cpu, va, op->store, pc);
    uint32_t esize = pair ? 1u << op->size : bytes;
    if (op->load) {
        uint8_t buf[16];
        if (hv_vmem_xfer(cpu, va, buf, bytes, 0, el, pc)) return 1;
        cpu->excl_addr = va;
        cpu->excl_size = bytes;
        cpu->excl_val[0] = le(buf, esize);
        cpu->excl_val[1] = pair ? le(buf + esize, esize) : 0;
        set_gpr(cpu, op->rt, cpu->excl_val[0]);
        if (pair) set_gpr(cpu, op->rt2, cpu->excl_val[1]);
        return 0;
    }
    void *host;
    uint64_t pa;
    if (rmw_target(cpu, va, bytes, el, pc, &host, &pa)) return 1;
    uint64_t mask = esize == 8 ? ~0ull : (1ull << (esize * 8)) - 1;
    uint64_t n0 = gpr(cpu, op->rt) & mask, n1 = pair ? gpr(cpu, op->rt2) & mask : 0;
    int ok = 0;
    if (cpu->excl_addr == va && cpu->excl_size == bytes) {
        if (!host) {
            ok = !hv_phys_write(cpu->m, pa, &n0, esize) && (!pair || !hv_phys_write(cpu->m, pa + esize, &n1, esize));
        } else if (!pair) {
            uint64_t expect = cpu->excl_val[0];
            ok = cas_host(host, op->size, &expect, n0);
        } else if (esize == 4) {
            uint64_t expect = cpu->excl_val[0] | cpu->excl_val[1] << 32;
            ok = cas_host(host, 3, &expect, n0 | n1 << 32);
        } else {
            __uint128_t expect = (__uint128_t)cpu->excl_val[1] << 64 | cpu->excl_val[0];
            ok = __atomic_compare_exchange_n((__uint128_t *)host, &expect, (__uint128_t)n1 << 64 | n0, 0,
                                             __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
        }
    }
    cpu->excl_addr = ~0ull;
    set_gpr(cpu, op->rs, ok ? 0 : 1);
    return 0;
}

static int do_vector(HvCpu *cpu, const HvMemOp *op, uint64_t va, int el, uint64_t pc) {
    uint8_t buf[64];
    uint32_t ebytes = 1u << op->size;
    if (op->store) {
        uint32_t off = 0;
        if (op->kind == HV_MEM_VEC_MULTI) {
            uint32_t elements = (op->vq ? 16 : 8) / ebytes;
            for (int r = 0; r < op->rpt; r++)
                for (uint32_t e = 0; e < elements; e++)
                    for (int s = 0; s < op->selem; s++, off += ebytes)
                        memcpy(buf + off, vreg(cpu, op->rt + r + s) + e * ebytes, ebytes);
        } else {
            for (int s = 0; s < op->selem; s++, off += ebytes)
                memcpy(buf + off, vreg(cpu, op->rt + s) + op->index * ebytes, ebytes);
        }
        return hv_vmem_xfer(cpu, va, buf, op->bytes, 1, el, pc);
    }
    if (hv_vmem_xfer(cpu, va, buf, op->bytes, 0, el, pc)) return 1;
    uint32_t off = 0;
    if (op->kind == HV_MEM_VEC_MULTI) {
        uint32_t elements = (op->vq ? 16 : 8) / ebytes;
        for (int r = 0; r < op->rpt; r++)
            for (uint32_t e = 0; e < elements; e++)
                for (int s = 0; s < op->selem; s++, off += ebytes)
                    memcpy(vreg(cpu, op->rt + r + s) + e * ebytes, buf + off, ebytes);
        if (!op->vq) /* 64-bit destinations clear the top half */
            for (int r = 0; r < op->rpt; r++)
                for (int s = 0; s < op->selem; s++) memset(vreg(cpu, op->rt + r + s) + 8, 0, 8);
    } else if (op->index == 0xff) {
        for (int s = 0; s < op->selem; s++, off += ebytes) {
            uint8_t *d = vreg(cpu, op->rt + s);
            memset(d, 0, 16);
            for (uint32_t i = 0; i < (op->vq ? 16u : 8u); i += ebytes) memcpy(d + i, buf + off, ebytes);
        }
    } else {
        for (int s = 0; s < op->selem; s++, off += ebytes)
            memcpy(vreg(cpu, op->rt + s) + op->index * ebytes, buf + off, ebytes);
    }
    return 0;
}

/* HV_PREDSTAT (measuring, with every access sent here): how often an access is on the same page as the last one
   through the same base register, or the last one by the same instruction, or the access just before. The page is
   keyed with the address space for user addresses (a process switch is a miss, as it would be for a prediction). */
static void pred_stat(HvCpu *cpu, const HvMemOp *op, uint64_t va, uint64_t pc, int el) {
    static __thread uint64_t last_reg[2][32], last_site[1 << 16], last_any;
    static __thread uint64_t n, reg_hit, site_hit, either, any_hit, sp_n, sp_hit, k_n, k_reg, k_site;
    int kernel = va >> 63;
    uint64_t key = va >> 14;
    if (!kernel) key ^= (cpu->ttbr0_el1 & ~1ull) << 20; /* user pages of different address spaces differ */
    int r = op->mode == HV_ADDR_LITERAL ? 31 : op->rn; /* 31 here is SP (or the literal's PC) */
    uint32_t s = (uint32_t)((pc >> 2) * 2654435761u) >> 16;
    int rh = last_reg[el ? 1 : 0][r] == key, sh = last_site[s] == key;
    n++, reg_hit += rh, site_hit += sh, either += rh || sh, any_hit += last_any == key;
    if (r == 31 && op->mode != HV_ADDR_LITERAL) sp_n++, sp_hit += rh;
    if (kernel) k_n++, k_reg += rh, k_site += sh;
    last_reg[el ? 1 : 0][r] = key, last_site[s] = key, last_any = key;
    if ((n & ((1u << 24) - 1)) == 0) {
        fprintf(stderr, "predstat cpu %u: %lluM accesses: same page as the base register's last %.1f%%, as the instruction's last %.1f%%, "
                        "either %.1f%%, as the access before %.1f%%; SP-based %.1f%% of accesses (register hit %.1f%%); kernel %.1f%% "
                        "(register %.1f%%, instruction %.1f%%)\n",
                cpu->index, (unsigned long long)(n >> 20), 100.0 * reg_hit / n, 100.0 * site_hit / n, 100.0 * either / n,
                100.0 * any_hit / n, 100.0 * sp_n / n, sp_n ? 100.0 * sp_hit / sp_n : 0, 100.0 * k_n / n,
                k_n ? 100.0 * k_reg / k_n : 0, k_n ? 100.0 * k_site / k_n : 0);
        n = reg_hit = site_hit = either = any_hit = sp_n = sp_hit = k_n = k_reg = k_site = 0;
    }
}

int hv_mem_emulate(HvCpu *cpu, uint32_t insn, uint64_t pc) {
    cpu->m->access_pc = pc;
    HvMemOp op;
    if (hv_mem_decode(insn, &op) != 0) {
        hv_raise_undef(cpu, pc);
        return 1;
    }
    int el = op.unpriv ? 0 : (int)cpu->el;
    uint64_t base = op.mode == HV_ADDR_LITERAL ? pc : base_reg(cpu, op.rn);
    uint64_t va = base;
    switch (op.mode) {
    case HV_ADDR_OFFSET:
    case HV_ADDR_PRE:
    case HV_ADDR_LITERAL:
        va = base + op.imm;
        break;
    case HV_ADDR_REG: {
        uint64_t m = gpr(cpu, op.rm);
        switch (op.ext) {
        case 2: m = (uint32_t)m; break;
        case 6: m = (uint64_t)(int64_t)(int32_t)m; break;
        default: break;
        }
        va = base + (m << op.shift);
        break;
    }
    default:
        break;
    }
    if (hv_knob[HV_K_PREDSTAT]) pred_stat(cpu, &op, va, pc, el);

    int rc = 0;
    switch (op.kind) {
    case HV_MEM_EXCL:
        rc = do_exclusive(cpu, &op, va, el, pc);
        break;
    case HV_MEM_CAS:
    case HV_MEM_ATOMIC:
        rc = do_atomic(cpu, &op, va, el, pc);
        break;
    case HV_MEM_CASP:
        rc = do_casp(cpu, &op, va, el, pc);
        break;
    case HV_MEM_VEC_MULTI:
    case HV_MEM_VEC_SINGLE:
        rc = do_vector(cpu, &op, va, el, pc);
        break;
    default: {
        if (op.align && (va & (op.bytes - 1))) return alignment_fault(cpu, va, op.store, pc);
        int pair = op.kind == HV_MEM_PAIR;
        uint32_t esize = 1u << op.size;
        uint8_t buf[32] = {0};
        if (op.store) {
            if (op.vec) {
                memcpy(buf, vreg(cpu, op.rt), esize);
                if (pair) memcpy(buf + esize, vreg(cpu, op.rt2), esize);
            } else {
                uint64_t a = gpr(cpu, op.rt), b = pair ? gpr(cpu, op.rt2) : 0;
                memcpy(buf, &a, esize);
                if (pair) memcpy(buf + esize, &b, esize);
            }
            rc = hv_vmem_xfer(cpu, va, buf, op.bytes, 1, el, pc);
            break;
        }
        rc = hv_vmem_xfer(cpu, va, buf, op.bytes, 0, el, pc);
        if (rc) break;
        if (op.vec) {
            memset(vreg(cpu, op.rt), 0, 16);
            memcpy(vreg(cpu, op.rt), buf, esize);
            if (pair) {
                memset(vreg(cpu, op.rt2), 0, 16);
                memcpy(vreg(cpu, op.rt2), buf + esize, esize);
            }
        } else {
            uint64_t a = extend(le(buf, esize > 8 ? 8 : esize), &op);
            uint64_t b = pair ? extend(le(buf + esize, esize), &op) : 0;
            /* Writeback first, so a loaded base register keeps the loaded value. */
            if (op.mode == HV_ADDR_PRE) set_base(cpu, op.rn, va);
            else if (op.mode == HV_ADDR_POST) set_base(cpu, op.rn, base + op.imm);
            set_gpr(cpu, op.rt, a);
            if (pair) set_gpr(cpu, op.rt2, b);
            return 0;
        }
        break;
    }
    }
    if (rc) return 1;
    if (op.mode == HV_ADDR_PRE) {
        set_base(cpu, op.rn, va);
    } else if (op.mode == HV_ADDR_POST) {
        int vec_reg = (op.kind == HV_MEM_VEC_MULTI || op.kind == HV_MEM_VEC_SINGLE) && op.rm != 31;
        set_base(cpu, op.rn, base + (vec_reg ? gpr(cpu, op.rm) : (uint64_t)op.imm));
    }
    return 0;
}
