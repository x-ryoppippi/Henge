#include "pvg_lite.h"
#include "share.h"
#include "net_slirp.h"
#include <sys/mman.h>
#include "henge-vm.h"

#include <os/proc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dispatch.h"
#include "henge-ios-jit.h"
#include "net_slirp.h"
#include "dns_proxy.h"
#include "vm.h"
#include "install.h"

#include <fcntl.h>
#include <unistd.h>

HENGE_EXPORT HengeDualMapping *henge_ios_jit_get_mapping(void);

static volatile int state;
static volatile int exit_reason;

static void on_exit(int reason) {
    exit_reason = reason;
    state = 2;
    fprintf(stderr, "henge: the machine stopped (%s)\n", hv_vm_reason_name(reason));
}

int henge_vm_boot(const char *rom, const char *disk, const char *aux, uint64_t ecid, uint64_t seed, uint64_t ram_mb,
                  int cpus, char *err, size_t errlen) {
    { extern void hv_prof_ios_start(void); hv_prof_ios_start(); }
    if (state) {
        snprintf(err, errlen, "A machine has already run in this session. Quit Henge and open it again to start another.");
        return -1;
    }
    HengeDualMapping *jit = henge_ios_jit_get_mapping();
    if (!jit || !jit->rw_addr) {
        snprintf(err, errlen, "Henge has no JIT region. Turn on JIT first.");
        return -1;
    }
    HvVmConfig c = {.rom = rom, .disk = disk, .aux = aux, .ecid = ecid, .seed = seed, .ram_mb = ram_mb, .ncpus = cpus, .log = 1,
                    .jit_rx = jit->rx_addr, .jit_rw = jit->rw_addr, .jit_bytes = jit->size};
    fprintf(stderr, "henge: booting with %d CPUs, %llu MB RAM, a %zu MB JIT region\n", cpus, (unsigned long long)ram_mb,
            jit->size >> 20);
    if (hv_vm_create(&c, err, errlen) != 0) return -1;
    if (getenv("HV_STATS")) hv_vm_machine()->stats_interval = atoi(getenv("HV_STATS"));
    state = 1;
    hv_vm_start(1, on_exit);
    return 0;
}

static volatile int install_step = -1;
static volatile double install_progress;

static void on_install_progress(int step, double p, void *user) {
    (void)user;
    install_step = step;
    install_progress = p;
    static int last = -1, lastpct = -1;
    int pct = (int)(p * 100);
    if (step != last || pct / 10 != lastpct / 10) {
        fprintf(stderr, "henge: install step %d %d%%\n", step, pct);
        last = step, lastpct = pct;
    }
}

int henge_vm_install(const char *rom, const char *disk, const char *aux, uint64_t ecid, const char *ipsw,
                     const char *workdir, uint64_t ram_mb, int cpus, char *err, size_t errlen) {
    if (state) {
        snprintf(err, errlen, "A machine has already run in this session. Quit Henge and open it again to start another.");
        return -1;
    }
    HengeDualMapping *jit = henge_ios_jit_get_mapping();
    if (!jit || !jit->rw_addr) {
        snprintf(err, errlen, "Henge has no JIT region. Turn on JIT first.");
        return -1;
    }
    HvVmConfig c = {.rom = rom, .disk = disk, .aux = aux, .ecid = ecid, .ram_mb = ram_mb, .ncpus = cpus, .log = 1,
                    .installer = 1, .no_sound = 1, .jit_rx = jit->rx_addr, .jit_rw = jit->rw_addr, .jit_bytes = jit->size};
    fprintf(stderr, "henge: installing %s with %d CPUs, %llu MB RAM\n", ipsw, cpus, (unsigned long long)ram_mb);
    if (hv_vm_create(&c, err, errlen) != 0) return -1;
    install_step = 0;
    install_progress = 0;
    if (hv_install_start(hv_vm_board()->vusb, ipsw, workdir, ecid, on_install_progress, NULL) != 0) {
        snprintf(err, errlen, "The installer could not start.");
        return -1;
    }
    state = 1;
    hv_vm_start(1, on_exit);
    return 0;
}

int henge_install_step(void) { return install_step; }
double henge_install_progress(void) { return install_progress; }

/* VZMacAuxiliaryStorage's new file: a 64-byte header, the rest zero (32 MB + 16 KB). */
int henge_make_aux(const char *path) {
    static const unsigned char header[24] = {0xa5, 0xd5, 0xb3, 0xfe, 0x97, 0xad, 0x1d, 0x27, 1, 0, 0, 0, 0, 0, 0, 0,
                                             1, 0, 0, 0, 0, 0, 0, 0};
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) return -1;
    int ok = write(fd, header, sizeof header) == (ssize_t)sizeof header && ftruncate(fd, 33570816) == 0;
    close(fd);
    return ok ? 0 : -1;
}

int henge_vm_state(void) { return state; }
const char *henge_vm_exit_reason(void) { return hv_vm_reason_name(exit_reason); }
void henge_vm_stop(void) { if (state == 1) hv_vm_stop(); }

/* Testing (Autopilot.swift): a translation knob, as "knob NAME VALUE" in the Mac tool's input file. */
int henge_vm_knob(const char *name, int value) { return state == 1 ? hv_knob_set(hv_vm_machine(), name, value) : 0; }

long henge_vm_balloon(long target_mb) {
    HvVmapple *b = state == 1 ? hv_vm_board() : NULL;
    if (!b || !b->balloon) return -1;
    if (target_mb >= 0) hv_virtio_balloon_set_target(b->balloon, (uint64_t)target_mb << 20);
    return (long)(hv_virtio_balloon_actual(b->balloon) >> 20);
}

uint64_t henge_vm_blocks(void) { return state ? hv_vm_machine()->stats.blocks : 0; }
uint64_t henge_vm_net_frames(int sent) { return sent ? hv_net_tx_frames : hv_net_rx_frames; }

uint64_t henge_vm_dns(int which) {
    uint64_t n[3];
    hv_dns_proxy_stats(&n[0], &n[1], &n[2]);
    return which >= 0 && which < 3 ? n[which] : 0;
}

int henge_vm_cpus_on(void) {
    int n = 0;
    for (int i = 0; i < hv_vm_ncpus(); i++)
        if (!hv_vm_cpu(i)->off) n++;
    return state ? n : 0;
}

int henge_vm_guest_el0(void) {
    for (int i = 0; state && i < hv_vm_ncpus(); i++)
        if (!hv_vm_cpu(i)->off && hv_vm_cpu(i)->el == 0) return 1;
    return 0;
}

void henge_vm_share(const char *folder, const char *socket_path, const char *agent) {
    if (hv_share_start(folder, socket_path, agent) == 0) hv_net_set_share_socket(socket_path);
}
void henge_share_set_clipboard(const char *text) { hv_share_set_clipboard(text); }
char *henge_share_take_clipboard(void) { return hv_share_take_clipboard(); }
long henge_share_agent_seen(void) { return hv_share_agent_seen(); }

void henge_vm_set_display(uint32_t width, uint32_t height) { hv_pvg_lite_set_mode(width, height); }
void henge_vm_set_display_mm(uint32_t width_mm, uint32_t height_mm) { hv_pvg_lite_set_size_mm(width_mm, height_mm); }

void henge_vm_key(uint16_t code, int down) { if (state == 1) hv_vm_key(code, down); }
void henge_vm_key_hid(uint8_t usage, int down) { if (state == 1) hv_vm_key_hid(usage, down); }
void henge_vm_mouse_move(int dx, int dy) { if (state == 1) hv_vm_mouse_move(dx, dy); }
void henge_vm_mouse_button(int button, int down) { if (state == 1) hv_vm_mouse_button(button, down); }
void henge_vm_mouse_scroll(int dx, int dy) {
    if (state != 1) return;
    if (dy) hv_vm_mouse_wheel(dy);
    if (dx) hv_vm_mouse_hwheel(dx);
}
int henge_vm_pointer(int *x, int *y) { return state == 1 ? hv_vm_pointer(x, y) : -1; }
int henge_vm_snapshot(uint8_t *bgra, int width, int height) { return state == 1 ? hv_vm_snapshot(bgra, width, height) : -1; }

void henge_vm_set_display_layer(void *layer) {
    hv_vm_set_display_layer(layer);
}

uint64_t henge_vm_gpu_frames(void) {
    return hv_vm_gpu_frames();
}

int henge_vm_boot_display(const uint8_t **pixels, int *width, int *height, int *stride) {
    HvBootDisplay d;
    if (!state || hv_vm_boot_display(&d) != 0) return -1;
    *pixels = d.pixels;
    *width = (int)d.width, *height = (int)d.height, *stride = (int)d.stride;
    return 0;
}

size_t henge_available_memory(void) { return os_proc_available_memory(); }

/* The most address space this app can reserve in one piece, in GB (measuring, for a fastmem window): PROT_NONE
   reservations cost no memory, and each is given back at once. */
uint64_t henge_largest_reservation_gb(void) {
    for (uint64_t gb = 4096; gb >= 4; gb /= 2) {
        void *p = mmap(NULL, gb << 30, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (p != MAP_FAILED) {
            munmap(p, gb << 30);
            return gb;
        }
    }
    return 0;
}

uint64_t henge_largest_mapping_mb(void) {
    for (uint64_t mb = 16896; mb >= 1024; mb -= 512) { /* 16 GB, the most a Mac can ask for, plus what the caller keeps back */
        void *p = mmap(NULL, mb << 20, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (p != MAP_FAILED) {
            munmap(p, mb << 20);
            return mb;
        }
    }
    return 0;
}
