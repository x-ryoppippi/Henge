#include "bdif.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEVID_MASK 0xffff0000u
#define DEVID_ROOT 0x00000000u
#define DEVID_AUX 0x00010000u
#define DEVID_USB 0x00100000u
#define REG_QSEL 0x410
#define USB_DEVICE_TYPE 0x1a01

#define REG_STATUS 0x0
#define REG_CFG 0x4
#define REG_UNK1 0x8
#define REG_BUSY 0x10
#define REG_UNK2 0x400
#define REG_CMD 0x408
#define REG_NEXT_DEVICE 0x420
#define REG_UNK3 0x434

#define DATA_FLAGS_READ 0x00030001u  /* the data buffer is written by the device: a read from the disk */
#define DATA_FLAGS_WRITE 0x00010001u /* read by the device: a write to the disk (macOS 27's iBSS writes aux sector 32768) */

/* A request, in guest memory: three (addr, len, flags) commands. */
typedef struct {
    uint64_t addr;
    uint32_t len, flags;
} ReqCmd;

static uint64_t usb_read(HvBdif *b, uint64_t reg) {
    switch (reg) {
    case REG_STATUS: return 1;
    case REG_CFG: return USB_DEVICE_TYPE;
    case REG_UNK1: return 0x420;
    case REG_BUSY: return b->usb_qsel < 2 ? (uint64_t)b->usb_q[b->usb_qsel].done : 0;
    case REG_UNK2: return 1;
    case 0x404: return b->usb_irq;
    case REG_QSEL: return b->usb_qsel;
    case REG_UNK3: return 0;
    case REG_NEXT_DEVICE: return ~0ull;
    default: return 0;
    }
}

typedef struct {
    uint64_t addr;
    uint32_t len;
    uint16_t one, flags;
} UsbDesc;

/* Walk a submitted chain; copy to (write) or from its buffers. */
static uint32_t usb_xfer(HvBdif *b, uint64_t chain, int write, uint8_t *buf, uint32_t len, uint32_t *room) {
    UsbDesc dsc[5];
    uint32_t done = 0, total = 0;
    for (int i = 0; i < 5; i++) {
        if (hv_phys_read(b->m, chain + 16u * i, &dsc[i], 16) != 0) break;
        int dev_writes = (dsc[i].flags & 2) != 0;
        if (dev_writes == write) {
            total += dsc[i].len;
            uint32_t n = dsc[i].len < len - done ? dsc[i].len : len - done;
            if (n) {
                void *p = hv_phys_ptr(b->m, dsc[i].addr, n);
                if (!p) break;
                if (write) memcpy(p, buf + done, n), hv_dma_written(b->m, dsc[i].addr, n);
                else memcpy(buf + done, p, n);
                done += n;
            }
        }
        if (!(dsc[i].flags & 1)) break;
    }
    if (room) *room = total;
    return done;
}

/* Queue 0 has a buffer and the host has a request: fill it. */
static void usb_deliver(HvBdif *b) {
    HvBdifUsbQ *q = &b->usb_q[0];
    if (!b->usb || !b->usb->rom_on || !q->pending || q->done) return;
    HvVusbMsg m;
    if (!hv_vusb_take_request(b->usb, &m)) return;
    uint32_t room = 0;
    usb_xfer(b, q->pending, 1, NULL, 0, &room);
    uint32_t payload = m.data ? m.len : 0;
    uint8_t *tmp = malloc(6 + payload);
    if (tmp) {
        memcpy(tmp, &m.len, 4);
        tmp[4] = m.ep;
        tmp[5] = m.type;
        if (payload) memcpy(tmp + 6, m.data, payload);
        if (room < 6 + payload) fprintf(stderr, "bdif usb: request ep %#x len %u does not fit a %u-byte buffer\n", m.ep, payload, room);
        usb_xfer(b, q->pending, 1, tmp, 6 + payload, NULL);
        free(tmp);
    }
    free(m.data);
    q->done = 1;
    q->pending = 0;
}

void hv_bdif_poll(HvBdif *b) {
    if (b && b->usb) usb_deliver(b);
}

static void usb_write(HvBdif *b, uint64_t reg, uint64_t v) {
    switch (reg) {
    case REG_UNK2: /* (re)enable: a new stage of the boot chain connecting */
        b->usb->rom_on = v != 0;
        hv_vusb_set_online(b->usb, v != 0);
        memset(b->usb_q, 0, sizeof b->usb_q);
        b->usb_q[0].done = b->usb_q[1].done = 1; /* idle: nothing submitted */
        break;
    case 0x404: b->usb_irq = (uint32_t)v; break;
    case REG_QSEL: b->usb_qsel = (uint32_t)v; break;
    case REG_CMD: {
        if (b->usb_qsel >= 2) break;
        HvBdifUsbQ *q = &b->usb_q[b->usb_qsel];
        q->done = 0;
        if (b->usb_qsel == 0) {
            q->pending = v;
            usb_deliver(b);
            break;
        }
        /* A response: take it now. */
        if (!b->usb->rom_on) break;
        uint32_t room = 0;
        usb_xfer(b, v, 0, NULL, 0, &room);
        uint8_t *tmp = room ? malloc(room) : NULL;
        uint32_t n = tmp ? usb_xfer(b, v, 0, tmp, room, NULL) : 0;
        if (n >= 2) hv_vusb_put_response(b->usb, tmp[0], tmp[1], tmp + 2, n - 2);
        free(tmp);
        q->done = 1;
        break;
    }
    default: break;
    }
}

static uint64_t bdif_read(void *opaque, uint64_t off, unsigned size) {
    (void)size;
    HvBdif *b = opaque;
    uint32_t devid = (uint32_t)off & DEVID_MASK;
    if (devid == DEVID_USB && b->usb) return usb_read(b, off & ~(uint64_t)DEVID_MASK);
    switch (off & ~(uint64_t)DEVID_MASK) {
    case REG_STATUS: return 1;    /* active */
    case REG_CFG: return 2;       /* active */
    case REG_UNK1: return 0x420;
    case REG_BUSY: return 1;      /* ready */
    case REG_UNK2: return 1;
    case REG_UNK3: return 0;
    case REG_NEXT_DEVICE:
        /* For storage devices this register is the capacity in 512-byte
           sectors. iBoot validates GPT addresses against this value. */
        if (devid == DEVID_ROOT) return b->root ? b->root->size / 512 : 0;
        if (devid == DEVID_AUX) return b->aux ? b->aux->size / 512 : 0;
        return ~0ull;
    default: return ~0ull;
    }
}

static void command(HvBdif *b, HvDisk *disk, uint64_t req_pa) {
    uint8_t ret = 1; /* failed */
    ReqCmd req[3];   /* sector, data, retval */
    if (hv_phys_read(b->m, req_pa, req, sizeof req) != 0) return;
    if (disk && req[0].len == 16) {
        uint32_t sector[4];
        if (hv_phys_read(b->m, req[0].addr, sector, sizeof sector) == 0) {
            /* The transfer uses validated guest RAM directly. Newer iBoot
               reads boot assets larger than the old 128 MiB allocation cap. */
            void *buf = hv_phys_ptr(b->m, req[1].addr, req[1].len);
            uint64_t at = (uint64_t)sector[2] * 512;
            if (buf && req[1].flags == DATA_FLAGS_READ) {
                if (hv_disk_read(disk, at, buf, req[1].len) == 0) ret = 0, hv_dma_written(b->m, req[1].addr, req[1].len);
            } else if (buf && req[1].flags == DATA_FLAGS_WRITE) {
                if (hv_disk_write(disk, at, buf, req[1].len) == 0) ret = 0;
            }
        }
    }
    if (b->m->log >= 2 || ret) {
        uint32_t sector[4] = {0};
        hv_phys_read(b->m, req[0].addr, sector, sizeof sector);
        fprintf(stderr, "bdif %s: sector %u len %u flags %x -> %s\n", disk == b->aux ? "aux" : "root", sector[2],
                req[1].len, req[1].flags, ret ? "failed" : "ok");
    }
    hv_phys_write(b->m, req[2].addr, &ret, 1);
}

static void bdif_write(void *opaque, uint64_t off, uint64_t v, unsigned size) {
    (void)size;
    HvBdif *b = opaque;
    uint32_t devid = (uint32_t)off & DEVID_MASK;
    if (devid == DEVID_USB && b->usb) {
        usb_write(b, off & ~(uint64_t)DEVID_MASK, v);
        return;
    }
    if ((off & ~(uint64_t)DEVID_MASK) != REG_CMD) return;
    if (devid == DEVID_ROOT) command(b, b->root, v);
    else if (devid == DEVID_AUX) command(b, b->aux, v);
}

HvBdif *hv_bdif_create(HvMachine *m, uint64_t pa, HvDisk *root, HvDisk *aux) {
    HvBdif *b = calloc(1, sizeof *b);
    b->m = m;
    b->root = root;
    b->aux = aux;
    hv_machine_add_mmio(m, pa, 0x200000, b, bdif_read, bdif_write, "bdif");
    return b;
}
