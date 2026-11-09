#include "sysreg.h"

#include <mach/mach_time.h>
#include <stdio.h>
#include <string.h>

#include "exception.h"
#include "machine.h"
#include "memop.h"
#include "mmu.h"
#include <stdlib.h>

/* System registers are named by op0:op1:CRn:CRm:op2, which is bits 20..5 of MRS/MSR. */
#define R(op0, op1, crn, crm, op2) ((op0) << 14 | (op1) << 11 | (crn) << 7 | (crm) << 3 | (op2))

/* ID registers: only features the translator handles. Host features are a
   superset (Apple M1 and later), so copied instructions always exist. */
#define ID_AA64PFR0  0x1100000001110011ull /* EL0/EL1 AArch64, FP+AdvSIMD with FP16, GICv3 sysregs, CSV2, CSV3 */
#define ID_AA64ISAR0 0x0011100110212120ull /* AES+PMULL, SHA1, SHA2/512, CRC32, LSE, RDM, SHA3, DotProd, FHM, FlagM */
#define ID_AA64ISAR1 0x0000001110211011ull /* DPB, PAuth (no-op), JSCVT, FCMA, LRCPC2, PACGA, FRINTTS, SB */
#define ID_AA64MMFR0 0x0000000000100025ull /* 48-bit PA, 16-bit ASID, 4K/16K/64K granules */
#define ID_AA64MMFR1 0x0000000000111000ull /* HPDS, LOR, PAN */
#define ID_AA64DFR0  0x0000000000000006ull /* Armv8 debug, no PMU */
#define MIDR         0x611f0220ull         /* Apple; a configuration constant for now */
#define CTR_EL0      0x9444c004ull         /* 64-byte lines, PIPT I-cache, IDC */
#define CLIDR        0x0a200023ull         /* L1 split, L2 unified */
#define CNTFRQ       24000000ull

static uint64_t counter(void) {
    return mach_absolute_time(); /* 24 MHz on Apple silicon */
}

uint64_t hv_counter(void) {
    return counter();
}

static int undefined(HvCpu *cpu, uint64_t pc) {
    hv_raise_undef(cpu, pc);
    return 1;
}

/* Timer control: ISTATUS (bit 2) reflects the compare right now. */
static uint64_t timer_ctl(uint64_t ctl, uint64_t cval, uint64_t now) {
    ctl &= 3;
    if ((ctl & 1) && now >= cval) ctl |= 4;
    return ctl;
}

/* EL0 is allowed only a few registers, some gated by SCTLR_EL1 / CNTKCTL_EL1. */
static int el0_allowed(HvCpu *cpu, uint32_t reg, int write) {
    uint64_t k = cpu->cntkctl_el1;
    switch (reg) {
    case R(3, 3, 4, 2, 0): case R(3, 3, 4, 4, 0): case R(3, 3, 4, 4, 1): /* NZCV, FPCR, FPSR */
    case R(3, 3, 13, 0, 2):                                               /* TPIDR_EL0 */
    case R(3, 3, 4, 2, 5): case R(3, 3, 4, 2, 6):                         /* DIT, SSBS */
        return 1;
    case R(3, 3, 4, 2, 1): return (cpu->sctlr_el1 >> 9) & 1;            /* DAIF: UMA */
    case R(3, 3, 13, 0, 3): return !write;                               /* TPIDRRO_EL0 */
    case R(3, 3, 0, 0, 1): return !write && ((cpu->sctlr_el1 >> 15) & 1); /* CTR_EL0: UCT */
    case R(3, 3, 0, 0, 7): return !write;                                /* DCZID_EL0 */
    case R(3, 3, 14, 0, 0): return !write && (k & 3);                    /* CNTFRQ */
    case R(3, 3, 14, 0, 1): case R(3, 3, 14, 0, 5): return !write && (k & 1);       /* CNTPCT */
    case R(3, 3, 14, 0, 2): case R(3, 3, 14, 0, 6): return !write && ((k >> 1) & 1); /* CNTVCT */
    case R(3, 3, 14, 3, 0): case R(3, 3, 14, 3, 1): case R(3, 3, 14, 3, 2): return (k >> 8) & 1; /* CNTV_* */
    case R(3, 3, 14, 2, 0): case R(3, 3, 14, 2, 1): case R(3, 3, 14, 2, 2): return (k >> 9) & 1; /* CNTP_* */
    default: return 0;
    }
}

/* Returns 0 and sets *v, or -1 if the register does not exist. */
static int read_reg(HvCpu *cpu, uint32_t reg, uint64_t *v) {
    uint64_t now = counter();
    HvMachine *m = cpu->m;
    switch (reg) {
    case R(3, 3, 4, 2, 0): *v = cpu->nzcv & 0xf0000000; return 0;
    case R(3, 3, 4, 2, 1): *v = cpu->daif; return 0;
    case R(3, 0, 4, 2, 2): *v = (uint64_t)cpu->el << 2; return 0;      /* CurrentEL */
    case R(3, 0, 4, 2, 0): *v = cpu->spsel; return 0;
    case R(3, 0, 4, 2, 3): *v = cpu->pstate_extra & (1u << 22); return 0; /* PAN */
    case R(3, 0, 4, 2, 4): *v = cpu->pstate_extra & (1u << 23); return 0; /* UAO */
    case R(3, 3, 4, 2, 5): *v = cpu->pstate_extra & (1u << 24); return 0; /* DIT */
    case R(3, 3, 4, 2, 6): *v = cpu->pstate_extra & (1u << 12); return 0; /* SSBS */
    /* FPCR/FPSR live in the host registers while the guest runs (hv_fp_enter). */
    case R(3, 3, 4, 4, 0): __asm__ volatile("mrs %0, fpcr" : "=r"(*v)); return 0;
    case R(3, 3, 4, 4, 1): __asm__ volatile("mrs %0, fpsr" : "=r"(*v)); return 0;
    case R(3, 0, 4, 1, 0): *v = cpu->spsel ? cpu->sp_el0 : cpu->sp; return 0; /* SP_EL0 */
    case R(3, 0, 4, 0, 0): *v = cpu->spsr_el1; return 0;
    case R(3, 0, 4, 0, 1): *v = cpu->elr_el1; return 0;
    case R(3, 0, 1, 0, 0): *v = cpu->sctlr_el1; return 0;
    case R(3, 0, 1, 0, 1): *v = cpu->actlr_el1; return 0;
    case R(3, 0, 1, 0, 2): *v = cpu->cpacr_el1; return 0;
    case R(3, 0, 2, 0, 0): *v = cpu->ttbr0_el1; return 0;
    case R(3, 0, 2, 0, 1): *v = cpu->ttbr1_el1; return 0;
    case R(3, 0, 2, 0, 2): *v = cpu->tcr_el1; return 0;
    case R(3, 0, 5, 1, 0): *v = cpu->afsr0_el1; return 0;
    case R(3, 0, 5, 1, 1): *v = cpu->afsr1_el1; return 0;
    case R(3, 0, 5, 2, 0): *v = cpu->esr_el1; return 0;
    case R(3, 0, 6, 0, 0): *v = cpu->far_el1; return 0;
    case R(3, 0, 7, 4, 0): *v = cpu->par_el1; return 0;
    case R(3, 0, 10, 2, 0): *v = cpu->mair_el1; return 0;
    case R(3, 0, 10, 3, 0): *v = cpu->amair_el1; return 0;
    case R(3, 0, 12, 0, 0): *v = cpu->vbar_el1; return 0;
    case R(3, 0, 12, 1, 0): /* ISR_EL1: interrupts pending at the CPU, masked or not */
        *v = (cpu->pc_state && cpu->pc_state->irq_line ? 1u << 7 : 0) | (cpu->pc_state && cpu->pc_state->fiq_line ? 1u << 6 : 0);
        return 0;
    case R(3, 0, 13, 0, 1): *v = cpu->contextidr_el1; return 0;
    case R(3, 0, 13, 0, 4): *v = cpu->tpidr_el1; return 0;
    case R(3, 3, 13, 0, 2): *v = cpu->tpidr_el0; return 0;
    case R(3, 3, 13, 0, 3): *v = cpu->tpidrro_el0; return 0;
    case R(3, 0, 14, 1, 0): *v = cpu->cntkctl_el1; return 0;
    case R(2, 0, 0, 2, 2): *v = cpu->mdscr_el1; return 0;
    case R(2, 0, 1, 1, 4): *v = 0x8; return 0;                           /* OSLSR_EL1: OS lock implemented, unlocked */
    case R(3, 2, 0, 0, 0): *v = cpu->csselr_el1; return 0;

    /* Identification */
    case R(3, 0, 0, 0, 0): *v = MIDR; return 0;
    case R(3, 0, 0, 0, 5): *v = cpu->mpidr; return 0;
    case R(3, 0, 0, 0, 6): *v = 0; return 0;                             /* REVIDR */
    case R(3, 0, 0, 4, 0): *v = ID_AA64PFR0; return 0;
    case R(3, 0, 0, 5, 0): *v = ID_AA64DFR0; return 0;
    case R(3, 0, 0, 6, 0): *v = ID_AA64ISAR0; return 0;
    case R(3, 0, 0, 6, 1): *v = ID_AA64ISAR1; return 0;
    case R(3, 0, 0, 7, 0): *v = ID_AA64MMFR0; return 0;
    case R(3, 0, 0, 7, 1): *v = ID_AA64MMFR1; return 0;
    case R(3, 3, 0, 0, 1): *v = CTR_EL0; return 0;
    case R(3, 3, 0, 0, 7): *v = 4; return 0;                             /* DCZID: 64-byte DC ZVA */
    case R(3, 1, 0, 0, 1): *v = CLIDR; return 0;
    case R(3, 1, 0, 0, 7): *v = 0; return 0;                             /* AIDR */
    case R(3, 1, 0, 0, 0):                                               /* CCSIDR for CSSELR's level */
        *v = (cpu->csselr_el1 >> 1) == 0 ? (127u << 13 | 7u << 3 | 2) : (2047u << 13 | 15u << 3 | 2);
        return 0;

    /* Generic timer */
    case R(3, 3, 14, 0, 0): *v = CNTFRQ; return 0;
    case R(3, 3, 14, 0, 1): case R(3, 3, 14, 0, 5): *v = now; return 0;
    case R(3, 3, 14, 0, 2): case R(3, 3, 14, 0, 6): *v = now - cpu->cntvoff; return 0;
    case R(3, 3, 14, 2, 0): *v = (uint32_t)(cpu->cntp_cval - now); return 0;
    case R(3, 3, 14, 2, 1): *v = timer_ctl(cpu->cntp_ctl, cpu->cntp_cval, now); return 0;
    case R(3, 3, 14, 2, 2): *v = cpu->cntp_cval; return 0;
    case R(3, 3, 14, 3, 0): *v = (uint32_t)(cpu->cntv_cval - (now - cpu->cntvoff)); return 0;
    case R(3, 3, 14, 3, 1): *v = timer_ctl(cpu->cntv_ctl, cpu->cntv_cval, now - cpu->cntvoff); return 0;
    case R(3, 3, 14, 3, 2): *v = cpu->cntv_cval; return 0;
    }
    /* GIC CPU interface (ICC_*_EL1): CRn 12, CRm 8..12, and ICC_PMR_EL1. */
    if (reg == R(3, 0, 4, 6, 0) || ((reg >> 14) == 3 && ((reg >> 11) & 7) == 0 && ((reg >> 7) & 15) == 12 && ((reg >> 3) & 15) >= 8)) {
        *v = m->icc_read ? m->icc_read(m, cpu, reg) : 0; /* the GIC locks itself */
        return 0;
    }
    /* The rest of the ID space reads as zero; so do debug and PMU registers. */
    if ((reg >> 14) == 3 && ((reg >> 11) & 7) == 0 && ((reg >> 7) & 15) == 0 && ((reg >> 3) & 15) >= 1) {
        *v = 0;
        return 0;
    }
    if ((reg >> 14) == 2 || (((reg >> 7) & 15) == 9 && (reg >> 14) == 3)) {
        *v = 0;
        return 0;
    }
    /* Apple implementation-defined registers (op0 3, CRn 15). The vmapple
       kernel reads error-status registers there (S3_6_C15_C0_0, S3_3_C15_C8_0..)
       when it reports a machine error; zero means no error recorded. */
    if ((reg >> 14) == 3 && ((reg >> 7) & 15) == 15) {
        if (m->log) fprintf(stderr, "hengevm: read of implementation-defined S3_%u_C15_C%u_%u = 0\n", (reg >> 11) & 7,
                            (reg >> 3) & 15, reg & 7);
        *v = 0;
        return 0;
    }
    return -1;
}

static int write_reg(HvCpu *cpu, uint32_t reg, uint64_t v) {
    HvMachine *m = cpu->m;
    switch (reg) {
    case R(3, 3, 4, 2, 0): cpu->nzcv = v & 0xf0000000; return 0;
    case R(3, 3, 4, 2, 1): cpu->daif = v & HV_PSTATE_DAIF; return 0;
    case R(3, 0, 4, 2, 0): hv_sp_switch(cpu, cpu->el, v & 1); return 0;
    case R(3, 0, 4, 2, 3): hv_pstate_set(cpu, (hv_pstate_get(cpu) & ~(1ull << 22)) | (v & (1u << 22))); return 0;
    case R(3, 0, 4, 2, 4): cpu->pstate_extra = (cpu->pstate_extra & ~(1ull << 23)) | (v & (1u << 23)); return 0;
    case R(3, 3, 4, 2, 5): cpu->pstate_extra = (cpu->pstate_extra & ~(1ull << 24)) | (v & (1u << 24)); return 0;
    case R(3, 3, 4, 2, 6): cpu->pstate_extra = (cpu->pstate_extra & ~(1ull << 12)) | (v & (1u << 12)); return 0;
    case R(3, 3, 4, 4, 0): v &= 0x07ff9f00; __asm__ volatile("msr fpcr, %0" ::"r"(v)); return 0;
    case R(3, 3, 4, 4, 1): v &= 0xf800009f; __asm__ volatile("msr fpsr, %0" ::"r"(v)); return 0;
    case R(3, 0, 4, 1, 0):
        if (!cpu->spsel) return -1; /* SP_EL0 is not accessible as a sysreg while it is the SP */
        cpu->sp_el0 = v;
        return 0;
    case R(3, 0, 4, 0, 0): cpu->spsr_el1 = v; return 0;
    case R(3, 0, 4, 0, 1): cpu->elr_el1 = v; return 0;
    case R(3, 0, 1, 0, 0): { /* SCTLR: xnu flips the PAC enables (EnIA/IB/DA/DB) on every process switch */
        uint64_t changed = cpu->sctlr_el1 ^ v;
        cpu->sctlr_el1 = v;
        if (changed & (1u | 1u << 19 | 1u << 24 | 1u << 25)) { /* M, WXN, E0E, EE: what translations depend on */
            cpu->m->stats.flush_why[2]++;
            hv_tlb_flush(cpu);
            __atomic_store_n(&cpu->exit_req, 1, __ATOMIC_RELAXED); /* the block's chained exit must not follow the old mapping */
        }
        return 0;
    }
    case R(3, 0, 1, 0, 1): cpu->actlr_el1 = v; return 0;
    case R(3, 0, 1, 0, 2): cpu->cpacr_el1 = v; return 0;
    case R(3, 0, 2, 0, 0): { /* a process switch */
        uint64_t old = cpu->ttbr0_el1;
        cpu->ttbr0_el1 = v;
        if (old != v) cpu->ttbr0_changed_at = mach_absolute_time();
        hv_tlb_switch_user(cpu, old, v);
        hv_ijc_switch(cpu);
        return 0;
    }
    case R(3, 0, 2, 0, 1): { /* TTBR1: rewritten unchanged on context switches; its ASID field only counts with TCR.A1 */
        uint64_t changed = (cpu->ttbr1_el1 ^ v) & ((cpu->tcr_el1 >> 22) & 1 ? ~0ull : 0x0000ffffffffffffull);
        cpu->ttbr1_el1 = v;
        if (changed) cpu->m->stats.flush_why[3]++, hv_tlb_flush(cpu);
        return 0;
    }
    case R(3, 0, 2, 0, 2):
        if (cpu->tcr_el1 != v) {
            cpu->tcr_el1 = v;
            cpu->m->stats.flush_why[4]++;
            hv_tlb_flush(cpu);
        }
        return 0;
    case R(3, 0, 5, 1, 0): cpu->afsr0_el1 = v; return 0;
    case R(3, 0, 5, 1, 1): cpu->afsr1_el1 = v; return 0;
    case R(3, 0, 5, 2, 0): cpu->esr_el1 = v; return 0;
    case R(3, 0, 6, 0, 0): cpu->far_el1 = v; return 0;
    case R(3, 0, 7, 4, 0): cpu->par_el1 = v; return 0;
    case R(3, 0, 10, 2, 0):
        if (cpu->mair_el1 != v) {
            cpu->mair_el1 = v;
            cpu->m->stats.flush_why[5]++;
            hv_tlb_flush(cpu);
        }
        return 0;
    case R(3, 0, 10, 3, 0): cpu->amair_el1 = v; return 0;
    case R(3, 0, 12, 0, 0): cpu->vbar_el1 = v & ~0x7ffull; return 0;
    case R(3, 0, 13, 0, 1): cpu->contextidr_el1 = v; return 0;
    case R(3, 0, 13, 0, 4): cpu->tpidr_el1 = v; return 0;
    case R(3, 3, 13, 0, 2): cpu->tpidr_el0 = v; return 0;
    case R(3, 3, 13, 0, 3): cpu->tpidrro_el0 = v; return 0;
    case R(3, 0, 14, 1, 0): cpu->cntkctl_el1 = v; cpu->el0_cnt = (uint32_t)(v & 3); return 0;
    case R(2, 0, 0, 2, 2): cpu->mdscr_el1 = v; return 0;
    case R(3, 2, 0, 0, 0): cpu->csselr_el1 = v & 0xf; return 0;
    case R(3, 3, 14, 2, 0): cpu->cntp_cval = counter() + (uint64_t)(int64_t)(int32_t)v; break;
    case R(3, 3, 14, 2, 1): cpu->cntp_ctl = v & 3; break;
    case R(3, 3, 14, 2, 2): cpu->cntp_cval = v; break;
    case R(3, 3, 14, 3, 0): cpu->cntv_cval = counter() - cpu->cntvoff + (uint64_t)(int64_t)(int32_t)v; break;
    case R(3, 3, 14, 3, 1): cpu->cntv_ctl = v & 3; break;
    case R(3, 3, 14, 3, 2): cpu->cntv_cval = v; break;
    default: goto other;
    }
    /* Timer outputs follow their registers at once, as in hardware: a masked
       timer must drop its line before the handler's EOI. */
    if (m->tick) m->tick(m, cpu);
    return 0;
other:
    /* PAuth key registers (CRn 2, CRm 1..3): kept, never used, since signing is a no-op. */
    if ((reg >> 14) == 3 && ((reg >> 11) & 7) == 0 && ((reg >> 7) & 15) == 2 && ((reg >> 3) & 15) >= 1 &&
        ((reg >> 3) & 15) <= 3)
        return 0;
    if (reg == R(3, 0, 4, 6, 0) ||
        ((reg >> 14) == 3 && ((reg >> 11) & 7) == 0 && ((reg >> 7) & 15) == 12 && ((reg >> 3) & 15) >= 8)) {
        if (m->icc_write) m->icc_write(m, cpu, reg, v);
        return 0;
    }
    /* Debug and PMU registers: ignored. */
    if ((reg >> 14) == 2 || (((reg >> 7) & 15) == 9 && (reg >> 14) == 3)) return 0;
    /* Apple implementation-defined registers (op0 3, CRn 15): ignored, see read_reg. */
    if ((reg >> 14) == 3 && ((reg >> 7) & 15) == 15) {
        if (m->log) fprintf(stderr, "hengevm: write to implementation-defined S3_%u_C15_C%u_%u ignored\n", (reg >> 11) & 7,
                            (reg >> 3) & 15, reg & 7);
        return 0;
    }
    return -1;
}

/* PAuth keys read back as zero. */
static int is_pauth_key(uint32_t reg) {
    return (reg >> 14) == 3 && ((reg >> 11) & 7) == 0 && ((reg >> 7) & 15) == 2 && ((reg >> 3) & 15) >= 1 &&
           ((reg >> 3) & 15) <= 3;
}

int hv_sysreg_mrs(HvCpu *cpu, uint32_t insn, uint64_t pc) {
    uint32_t reg = (insn >> 5) & 0xffff;
    int rt = insn & 0x1f;
    if (cpu->el == 0 && !el0_allowed(cpu, reg, 0)) return undefined(cpu, pc);
    uint64_t v;
    if (is_pauth_key(reg)) v = 0;
    else if (read_reg(cpu, reg, &v) != 0) return undefined(cpu, pc);
    if (cpu->m->log >= 3)
        fprintf(stderr, "mrs S%u_%u_C%u_C%u_%u = %llx pc %llx el%u\n", reg >> 14, (reg >> 11) & 7, (reg >> 7) & 15,
                (reg >> 3) & 15, reg & 7, (unsigned long long)v, (unsigned long long)pc, cpu->el);
    if (rt != 31) cpu->x[rt] = v;
    return 0;
}

/* Blocks carry on after most MSRs (see the translator), so an interrupt this
   write let through (unmasking, or a GIC register raising a line) must make
   the block leave at its next exit for the dispatcher to take it. */
static void irq_check(HvCpu *cpu) {
    HvMachine *m = cpu->m;
    HvPerCpu *p = cpu->pc_state;
    (void)m;
    if (p && ((p->irq_line && !(cpu->daif & HV_PSTATE_I)) || (p->fiq_line && !(cpu->daif & HV_PSTATE_F)))) {
        __atomic_store_n(&cpu->exit_req, 1, __ATOMIC_RELAXED);
        if (m->stats_interval) m->stats.kick_src[4]++;
    }
}

int hv_sysreg_msr(HvCpu *cpu, uint32_t insn, uint64_t pc) {
    if (((insn >> 19) & 3) == 0) { /* MSR (immediate) to a PSTATE field */
        uint32_t op1 = (insn >> 16) & 7, op2 = (insn >> 5) & 7, imm = (insn >> 8) & 0xf;
        uint32_t field = op1 << 3 | op2;
        if (cpu->el == 0 && field != 036 && field != 037 && field != 031 && field != 032) return undefined(cpu, pc);
        if (cpu->el == 0 && (field == 036 || field == 037) && !((cpu->sctlr_el1 >> 9) & 1)) return undefined(cpu, pc);
        switch (field) {
        case 005: hv_sp_switch(cpu, cpu->el, imm & 1); break;                       /* SPSel */
        case 036: cpu->daif |= (uint64_t)imm << 6; break;                           /* DAIFSet */
        case 037: cpu->daif &= ~((uint64_t)imm << 6); break;                        /* DAIFClr */
        case 004: write_reg(cpu, R(3, 0, 4, 2, 3), (uint64_t)(imm & 1) << 22); break; /* PAN */
        case 003: write_reg(cpu, R(3, 0, 4, 2, 4), (uint64_t)(imm & 1) << 23); break; /* UAO */
        case 032: write_reg(cpu, R(3, 3, 4, 2, 5), (uint64_t)(imm & 1) << 24); break; /* DIT */
        case 031: write_reg(cpu, R(3, 3, 4, 2, 6), (uint64_t)(imm & 1) << 12); break; /* SSBS */
        default: return undefined(cpu, pc);
        }
        irq_check(cpu);
        return 0;
    }
    uint32_t reg = (insn >> 5) & 0xffff;
    int rt = insn & 0x1f;
    if (cpu->m->log >= 3)
        fprintf(stderr, "msr S%u_%u_C%u_C%u_%u = %llx pc %llx el%u\n", reg >> 14, (reg >> 11) & 7, (reg >> 7) & 15,
                (reg >> 3) & 15, reg & 7, (unsigned long long)(rt == 31 ? 0 : cpu->x[rt]), (unsigned long long)pc,
                cpu->el);
    if (cpu->el == 0 && !el0_allowed(cpu, reg, 1)) return undefined(cpu, pc);
    if (write_reg(cpu, reg, rt == 31 ? 0 : cpu->x[rt]) != 0) return undefined(cpu, pc);
    irq_check(cpu);
    return 0;
}

/* One TLBI, applied to one CPU's TLB and indirect-branch cache. */
void hv_tlbi_apply(HvCpu *cpu, uint32_t op1, uint32_t crn, uint32_t crm, uint32_t op2, uint64_t xt) {
    uint64_t pc = cpu->pc;
    /* VAE1, VAAE1, VALE1, VAALE1 (and their IS/OS forms) name one page:
       Xt[43:0] is VA[55:12]. Everything else (all, by ASID, ranges) drops it all. */
    int by_va = crn == 8 && op1 == 0 && (crm == 1 || crm == 3 || crm == 7) && (op2 & 1);
    int by_asid = crn == 8 && op1 == 0 && (crm == 1 || crm == 3 || crm == 7) && op2 == 2; /* ASIDE1: non-global (user) entries */
    /* RVAE1, RVAAE1, RVALE1, RVAALE1 (IS/OS/local): xnu uses these for every
       multi-page unmap. Xt: ASID[63:48], TG[47:46], SCALE[45:44], NUM[43:39],
       BaseADDR[36:0] in granule units; (NUM+1) << (5*SCALE+1) granules. */
    int by_range = crn == 8 && op1 == 0 && (crm == 2 || crm == 5 || crm == 6) && (op2 & 1);
    if (by_range) {
        uint32_t tg = (xt >> 46) & 3;
        int shift = tg == 1 ? 12 : tg == 2 ? 14 : tg == 3 ? 16 : 0;
        uint64_t pages = (((xt >> 39) & 31) + 1) << (5 * ((xt >> 44) & 3) + 1);
        int all_asids = op2 == 3 || op2 == 7;
        /* BaseADDR is VA[36+shift:shift], sign-extended so TTBR1 ranges come out as kernel VAs. */
        uint64_t va = shift ? (uint64_t)((int64_t)((xt & 0x1fffffffffull) << (64 - 37)) >> (64 - 37 - shift)) : 0;
        uint64_t len = shift ? pages << shift : 0;
        if (!shift || len > (256ull << HV_TLB_PAGE_BITS)) {
            /* Reserved granule: everything. A big range: what of that half lies in it. */
            if (!shift) cpu->m->stats.flush_why[6]++, hv_tlb_flush(cpu);
            else if ((va >> 55) & 1) hv_tlb_flush_kernel_range(cpu, va, len);
            else {
                hv_tlb_flush_user_range(cpu, va, len);
                hv_tlb_stash_drop(cpu, all_asids ? -1 : (int)(xt >> 48));
                cpu->m->stats.reset_why[7]++;
                hv_ijc_drop_range(cpu, va, len, all_asids ? -1 : (int)(xt >> 48));
            }
        } else {
            uint64_t first = va & ~((1ull << HV_TLB_PAGE_BITS) - 1);
            for (uint64_t p = first; p < va + len; p += 1ull << HV_TLB_PAGE_BITS)
                hv_tlb_flush_va_asid(cpu, p, all_asids ? -1 : (int)(xt >> 48));
        }
        return;
    }
    if (by_va) {
        uint64_t va = (uint64_t)((int64_t)((xt & 0xfffffffffffull) << 20) >> 8);
        int all_asids = op2 == 3 || op2 == 7; /* VAAE1, VAALE1 */
        hv_tlb_flush_va_page(cpu, va, all_asids ? -1 : (int)(xt >> 48));
    } else if (by_asid) {
        hv_tlb_flush_user(cpu);
        hv_tlb_stash_drop(cpu, (int)(xt >> 48));
        cpu->m->stats.reset_why[1]++;
        if (cpu->m->stats_interval) cpu->m->stats.ijc_reset_kind[2]++;
        hv_ijc_drop_asid(cpu, (uint16_t)(xt >> 48));
    } else {
        if (cpu->m->log >= 2)
            fprintf(stderr, "tlbi-all S1_%u_C%u_C%u_%u pc %llx\n", op1, crn, crm, op2, (unsigned long long)pc);
        cpu->m->stats.flush_why[crm == 7 ? 0 : 1]++;
        uint64_t since_switch = crm == 7 ? mach_absolute_time() - cpu->ttbr0_changed_at : ~0ull; /* 24 MHz ticks */
        /* HV_SWITCH_FLUSH=1 (testing): macOS 27's xnu follows a process switch between two shared regions with a
           local TLBI VMALLE1, to drop the old region's global entries. Here every process's user entries are put
           aside under its own TTBR0 and its cached branch targets under its own ASID, so nothing of the old one
           can be seen; such a flush, within 20 us of a TTBR0 change, is left out. */
        /* HV_SWITCH_FLUSH=2: one in 32 is checked first (hv_tlb_check_all, too slow for all of macOS 27's ~800/s) and done
           if any entry kept no longer matches the page tables. */
        /* HV_SWITCH_FLUSH=3: under emulation the same flush often comes later than 20 us (the phone: 30-100/s at
           20-200 us, each emptying every TLB and branch cache of the CPU): up to 200 us it is left out too, if it is at
           a PC already seen within 20 us of a TTBR0 change, i.e. the same instruction in xnu's switch path. */
        static _Thread_local uint32_t nth;
        int late = 0;
        if (crm == 7 && hv_knob[HV_K_SWITCHFLUSH] >= 3 && since_switch >= 480 && since_switch < 4800) {
            for (int i = 0; i < 4; i++)
                if (__atomic_load_n(&cpu->m->switch_flush_pc[i], __ATOMIC_RELAXED) == pc) late = 1;
        }
        if (crm == 7 && hv_knob[HV_K_SWITCHFLUSH] >= 3 && since_switch < 480) {
            int known = 0;
            for (int i = 0; i < 4 && !known; i++) {
                uint64_t have = __atomic_load_n(&cpu->m->switch_flush_pc[i], __ATOMIC_RELAXED), zero = 0;
                if (have == pc) known = 1;
                else if (!have && __atomic_compare_exchange_n(&cpu->m->switch_flush_pc[i], &zero, pc, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) known = 1;
            }
        }
        if (crm == 7 && hv_knob[HV_K_SWITCHFLUSH] && (since_switch < 480 || late) &&
            (hv_knob[HV_K_SWITCHFLUSH] < 2 || (nth++ & 31) || !hv_tlb_check_all(cpu))) {
            if (cpu->m->stats_interval) cpu->m->stats.flush_skipped++, cpu->m->stats.flush_skipped_late += (uint64_t)late;
            return;
        }
        if (crm == 7 && cpu->m->stats_interval) {
            uint64_t dt = since_switch;
            cpu->m->stats.flush_after_switch[dt < 48 ? 0 : dt < 480 ? 1 : dt < 4800 ? 2 : 3]++;
            for (int i = 0; i < 8 && dt < 480; i++) { /* where the ones right after a switch come from */
                uint64_t *slot_pc = &cpu->m->stats.flush_pc[i].pc;
                if (*slot_pc == pc) { __atomic_add_fetch(&cpu->m->stats.flush_pc[i].n, 1, __ATOMIC_RELAXED); break; }
                uint64_t zero = 0;
                if (!*slot_pc && __atomic_compare_exchange_n(slot_pc, &zero, pc, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
                    __atomic_add_fetch(&cpu->m->stats.flush_pc[i].n, 1, __ATOMIC_RELAXED);
                    break;
                }
            }
        }
        hv_tlb_flush(cpu);
    }
}

/* Translations are never out of date: CPU writes to code pages and device DMA drop them as
   they happen. So IC IVAU has nothing to do, though xnu issues it for pages nobody wrote, and
   dropping them there meant translating them all again and every CPU's branch cache starting
   over. HV_IC_IVAU: as before. */
int hv_ic_ivau_drops(void) {
    static int v = -1;
    if (v < 0) v = getenv("HV_IC_IVAU") != NULL;
    return v;
}

int hv_sysreg_sys(HvCpu *cpu, uint32_t insn, uint64_t pc) {
    uint32_t op1 = (insn >> 16) & 7, crn = (insn >> 12) & 0xf, crm = (insn >> 8) & 0xf, op2 = (insn >> 5) & 7;
    int rt = insn & 0x1f;
    uint64_t xt = rt == 31 ? 0 : cpu->x[rt];
    if (crn == 7 && op1 == 3 && crm == 4 && op2 == 1) { /* DC ZVA */
        if (cpu->el == 0 && !((cpu->sctlr_el1 >> 14) & 1)) return undefined(cpu, pc);
        static const uint8_t zero[64];
        return hv_vmem_xfer(cpu, xt & ~63ull, (void *)zero, 64, 1, cpu->el, pc);
    }
    if (crn == 7 && op1 == 3 && crm == 5 && op2 == 1) { /* IC IVAU: just the page holding the address */
        if (!hv_ic_ivau_drops()) return 0;
        HvWalk w;
        hv_mmu_walk(cpu, xt, HV_ACC_READ, cpu->el, &w);
        if (!w.fsc) hv_invalidate_code(cpu->m, w.pa);
        return 0;
    }
    if (crn == 7 && (crm == 1 || crm == 5)) { /* IC IALLUIS, IALLU */
        /* Translations are never out of date: CPU writes to code pages and device
           DMA drop them as they happen. Dropping them all here cost a full round of
           translating everything again, many times a boot. HV_IC_FLUSH: as before. */
        static int flush = -1;
        if (flush < 0) flush = getenv("HV_IC_FLUSH") != NULL;
        cpu->m->stats.flush_ic++;
        if (flush) cpu->m->flush_pending = 1;
        return 0;
    }
    if (crn == 7 && crm == 8 && op1 == 0 && op2 <= 3) { /* AT S1E1R, S1E1W, S1E0R, S1E0W */
        if (cpu->el == 0) return undefined(cpu, pc);
        HvWalk w;
        hv_mmu_walk(cpu, xt, (op2 & 1) ? HV_ACC_WRITE : HV_ACC_READ, op2 >= 2 ? 0 : 1, &w);
        if (w.fsc) cpu->par_el1 = 1 | (uint64_t)w.fsc << 1 | (uint64_t)w.s1ptw << 9 | 1ull << 11;
        else cpu->par_el1 = (w.pa & 0x0000fffffffff000ull) | (uint64_t)w.attr << 56 | 3u << 7;
        return 0;
    }
    if (crn == 8 || crn == 9) { /* TLBI */
        if (cpu->el == 0) return undefined(cpu, pc);
        /* The shareable forms (IS, OS: CRm 1, 2, 3, 5) reach every CPU: each does
           it to its own TLB, and this one waits until they have (a TLBI has
           finished for everyone once the DSB after it completes). */
        HvMachine *m = cpu->m;
        hv_tlbi_apply(cpu, op1, crn, crm, op2, xt);
        int broadcast = (crm == 1 || crm == 2 || crm == 3 || crm == 5) && m && m->ncpus > 1;
        if (broadcast) {
            /* One page or a small range (the common cases): the others' TLB
               entries, stashed entries and cached branch targets for it go from
               here (hv_tlbi_drop_remote_va), with nobody waited for unless a page
               walk or cache fill of theirs raced; one that may be running code
               from the page is made to leave its block. */
            int by_va = crn == 8 && op1 == 0 && (crm == 1 || crm == 3) && (op2 & 1);
            uint64_t va = (uint64_t)((int64_t)((xt & 0xfffffffffffull) << 20) >> 8);
            int asid = op2 == 3 || op2 == 7 ? -1 : (int)(xt >> 48);
            /* A small range (RVAE1IS and kin): page by page, the same way. */
            uint64_t npages = 0;
            int by_range = crn == 8 && op1 == 0 && (crm == 2 || crm == 5) && (op2 & 1);
            if (by_range && ((xt >> 46) & 3) == 2) { /* 16 KB granule, as xnu uses */
                npages = (((xt >> 39) & 31) + 1) << (5 * ((xt >> 44) & 3) + 1);
                va = (uint64_t)((int64_t)((xt & 0x1fffffffffull) << (64 - 37)) >> (64 - 37 - 14));
                /* HV_RANGE_DROP=N: ranges up to N pages are done from here too, in one pass (hv_tlbi_drop_remote_range) */
                if (npages > (hv_knob[HV_K_RANGEDROP] > 32 ? (uint64_t)hv_knob[HV_K_RANGEDROP] : 32)) npages = 0;
            } else if (by_va) {
                npages = 1;
            }
            uint64_t seq[HV_MAX_CPUS] = {0};
            for (int i = 0; i < m->ncpus; i++) {
                HvCpu *c = m->cpus[i];
                if (!c || c == cpu) continue;
                /* Done from here entirely if it can be; else c does it, and we wait.
                   A drop that raced a page walk or cache fill of c's (they are
                   short) is tried again a few times first. */
                int done = npages > 0, why = !npages ? (by_range ? 1 : 2) : 0;
                if (done && npages > 1 && hv_knob[HV_K_RANGEDROP] > 0) {
                    int r = 0;
                    for (int t = 0; t < 4 && (r = hv_tlbi_drop_remote_range(c, va, npages, asid)) == 0; t++)
                        for (int s = 0; s < 64; s++) __asm__ volatile("yield");
                    done = r == 1;
                    if (r < 0) why = 3;
                }
                else for (uint64_t p = 0; done && p < npages; p++) {
                    uint64_t pva = va + (p << HV_TLB_PAGE_BITS);
                    int r = 0;
                    for (int t = 0; t < 4 && (r = hv_tlbi_drop_remote_va(c, pva, asid)) == 0; t++)
                        for (int s = 0; s < 64; s++) __asm__ volatile("yield");
                    done = r == 1;
                    if (r < 0) why = 3;
                }
                if (done) continue;
                if (m->stats_interval) m->stats.posted_kind[why]++;
                seq[i] = hv_cpu_post_tlbi(c, op1, crn, crm, op2, xt);
            }
            for (int i = 0; i < m->ncpus; i++)
                if (seq[i]) hv_cpu_sync(cpu, m->cpus[i], seq[i]);
        }
        return 0;
    }
    if (crn == 7) return 0; /* other cache maintenance: caches are coherent here */
    return undefined(cpu, pc);
}
