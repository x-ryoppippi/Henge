/* Runs the guest's GPU commands on the host's Metal, on any Apple platform.
   The guest driver serializes Metal calls: an exec names command buffers in a
   task's memory, each a segment holding {tag, size} records, one per call, in
   the formats worked out from the guest driver and listed by the decoders
   below. Objects are named by a number (a ref); the task's object list says
   what each is, and they are made the first time they are used. */

#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>
#include <stdatomic.h>
#include <os/lock.h>
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <TargetConditionals.h>
#import <IOSurface/IOSurfaceRef.h>
#include <string.h>
#include <sys/utsname.h>

/* 1080p shrunk onto a phone would make a normal pointer a few dots across; an iPad's screen needs less. */
static int cursor_scale(void) {
#if TARGET_OS_IPHONE
    NSInteger saved = [[NSUserDefaults standardUserDefaults] integerForKey:@"henge.cursor.scale"];
    if (saved >= 1 && saved <= 3) return (int)saved;
    static int defaultScale;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        struct utsname u;
        uname(&u);
        defaultScale = strncmp(u.machine, "iPad", 4) ? 3 : 1;
    });
    return defaultScale;
#else
    return 1;
#endif
}

#include <mach/mach.h>
#include <stdatomic.h>
#include <mach/vm_map.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pvg_lite.h"
#include "metallib.h"

#if !TARGET_OS_IPHONE
#import <CoreGraphics/CoreGraphics.h>
#import <ImageIO/ImageIO.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#endif

/* Metal's own pipeline serialization, on _MTLDevice. The guest's Metal wrote the data. */
@protocol HvMetalSPI
- (id)newRenderPipelineDescriptorWithSerializedData:(dispatch_data_t)d deserializationContext:(id)c;
- (id)newTileRenderPipelineDescriptorWithSerializedData:(dispatch_data_t)d deserializationContext:(id)c;
- (id)newComputePipelineDescriptorWithSerializedData:(dispatch_data_t)d deserializationContext:(id)c;
@end

enum { OBJ_BUFFER = 1, OBJ_TEXTURE = 2, OBJ_FUNCTION = 6, OBJ_SERIALIZED = 7, OBJ_LINEAR_TEXTURE = 8, OBJ_MAPPED_TEXTURE = 0xb };

/* What is known about a made object, so a reused ref is noticed. */
@interface HvObject : NSObject {
@public
    id object;
    uint32_t entry[3];
    /* A buffer's guest pages, mapped contiguously. */
    uint8_t *mapped;
    BOOL mapping_in_buffer; /* the buffer over them frees them */
    BOOL plain_buffer; /* the object is that buffer */
    uint32_t generation; /* a new mapping since: what was made from it is out of date */
    HvObject *follows; /* made from this one's buffer */
    uint32_t follows_generation;
    uint64_t length;
    uint32_t vpage;
    uint32_t *pages;
    /* A texture's layout in guest memory. */
    uint32_t tex_mips, tex_slices;
    uint64_t tex_base, tex_stride;
    uint8_t tex_block_h;
    uint8_t *tex_layout; /* the mip table, 36 bytes each */
    BOOL guest_backed;
    /* A surface: */
    HvObject *surface;
    uint64_t row;
    uint32_t first;
    uint32_t surface_w, surface_h, fourcc; /* its own, whatever views the guest makes of it */
    id<MTLTexture> scanout; /* the same pixels, as the display reads them */
    /* Multi-plane surfaces cannot sit on a buffer: a real IOSurface, kept up to date from the guest's pages. */
    IOSurfaceRef iosurface;
    uint64_t alloc, content_hash;
    /* Source layout for a multi-plane video surface. */
    uint32_t plane_count;
    uint64_t plane_from[4], plane_row[4], plane_h[4];
    /* Each plane's sampled guest bytes when last copied: only the planes the guest changed are copied, so a CPU
       write to one plane (a decoder refilling luma) does not wipe what the GPU drew into another (chroma, alpha). */
    uint64_t plane_hash[4];
    /* A heap: private memory, a buffer over all of it. */
    id<MTLBuffer> heap_whole;
    /* A heap's or a private texture's copy in guest memory, and which side has the newest bytes. */
    id<MTLBuffer> guest_copy;
    BOOL host_valid, guest_stale;
    /* What this one is made from, kept for as long as it lives. */
    HvObject *parent;
    /* A multi-plane surface's key in the replay's planar table, while it counts as one of its users. */
    NSString *planar_key;
}
@end

@implementation HvObject
- (void)dealloc {
    if (mapped && !mapping_in_buffer) vm_deallocate(mach_task_self(), (vm_address_t)mapped, (length + 0x3fff) & ~0x3fffull);
    if (iosurface) CFRelease(iosurface);
    free(pages);
    free(tex_layout);
}
@end

@class HvContext;

@interface HvReplay : NSObject {
@public
    HvPvgLite *lite;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    dispatch_queue_t worker;
    NSMutableDictionary<NSNumber *, NSMutableDictionary<NSNumber *, HvObject *> *> *tasks;
    NSMutableDictionary<NSNumber *, id<MTLTexture>> *mapped_textures;
    NSMutableDictionary<NSNumber *, HvObject *> *surfaces;
    /* Multi-plane surfaces' host IOSurfaces by the guest memory they stand for: each process maps a surface
       under its own mapping id, and what one draws (macOS 27's WebKit draws pages into b3a8 tiles in its GPU
       process) has to be what another reads (WindowServer). */
    NSMutableDictionary<NSString *, id> *planar;
    NSMutableDictionary<NSString *, NSNumber *> *planar_uses; /* mappings using each: the entry goes with the last one */
    NSMutableSet<NSString *> *complaints;
    uint64_t execs, draws, failed, heap_bytes, synced, remaps;
    /* Work the GPU has not finished. Stamps are given back in the order they came: the
       GPU's completion handlers run in the order of commit. */
    _Atomic int inflight;
    id<MTLCommandBuffer> last_cb;
    double encode_time, gpu_wait_time;
    BOOL pipeline_set; /* in the open encoder */
    BOOL tess_factors; /* so is a tessellation factor buffer */
    uint32_t pre_raster_samples, pre_imageblock, pre_threadgroup, pre_tile_w, pre_tile_h; /* for the next pass */
    BOOL tracing; /* HV_PVG_TRACE_FILE exists: say what each exec draws */
    /* The newest frame, BGRA, for hv_pvg_metal_frame, and the same without the pointer on it. */
    uint8_t *frame, *base;
    uint32_t frame_w, frame_h;
    /* The pointer: macOS sends it apart from the screen, for the display hardware to overlay. */
    uint8_t *cursor;
    uint32_t cursor_w, cursor_h, cursor_stride;
    int cursor_hot_x, cursor_hot_y;
    BOOL cursor_shown;
    int drawn_x, drawn_y, drawn_w, drawn_h; /* where it is on the frame now */
    double last_capture;
    /* A frame that came too soon after the last: shown shortly, for it may be the last for a long while. */
    uint64_t late_mapping;
    BOOL late, late_queued;
    uint64_t frames, frames_shot;
    /* Showing frames with the GPU: a layer the app gave us. No copies, no conversion on a CPU. */
    CAMetalLayer *layer; /* screen_lock protects the target and its generation */
    uint64_t layer_generation;
    __weak CAMetalLayer *prepared_layer; /* presenter only: each target needs its own setup */
    id<MTLRenderPipelineState> screen_pipe, linear_pipe, cursor_pipe;
    id<MTLFunction> stand_in; /* computeStandIn */
    id<MTLTexture> cursor_tex;
    id<MTLSamplerState> screen_sampler, cursor_sampler;
    uint64_t on_screen; /* the mapping showing */
    _Atomic uint64_t gpu_frames;
    dispatch_queue_t presenter;
    os_unfair_lock screen_lock; /* what to show, and the pointer as the presenter reads it */
    id<MTLTexture> screen_tex;
    BOOL screen_half, screen_dirty, presenter_running;
    BOOL drawable_retry_pending;
    double drawable_retry_cooldown; /* fresh work coalesces one wake at its expiry */
    uint64_t screen_requests; /* fresh frames/cursor changes, not automatic retries */
    double drawable_retry_deadline; /* at most two seconds of retrying an unavailable target */
    uint32_t screen_w, screen_h;
    id<MTLBuffer> test_readback; /* HV_PVG_GPU_TEST: the drawable's pixels for screenshots */
    id<MTLBuffer> readback;
    const char *shot;
    unsigned shots;
    dispatch_source_t shot_timer;
}
- (void)complain:(NSString *)what;
- (HvObject *)object:(uint32_t)ref task:(uint32_t)task;
- (void)drawCursor;
- (BOOL)hasDisplayLayer;
- (void)kickPresenter:(BOOL)fresh;
- (void)present:(uint64_t)mapping;
- (void)screenChanged;
@end

/* What Metal asks while it reads a serialized pipeline. */
@interface HvContext : NSObject {
@public
    __unsafe_unretained HvReplay *replay;
    uint32_t task;
    NSString *failure;
    id standIn; /* what a failed function lookup returns: nil, except for compute pipelines (makeSerialized) */
}
@end

static double now_seconds(void) {
    return (double)clock_gettime_nsec_np(CLOCK_MONOTONIC) * 1e-9;
}

/* How the half-float display surface is shown. WindowServer writes DISPLAY-ENCODED values there (extended sRGB: the guest's gray 128 is
   0.502, measured with the colour probe), so mode 0 passes them through. Mode 1 treats them as LINEAR light and applies the sRGB encode,
   which is what this code did before (and was only right for a linear display profile such as ACES CG Linear: with every other profile
   it encodes twice and the screen looks pale). HV_PVG_HALF=linear selects mode 1 at start; with the colour probe on, a file named
   mode-linear or mode-encoded in its folder switches it while running. */
static _Atomic int hv_half_mode = -1;
static int hv_half_mode_get(void) {
    int m = atomic_load(&hv_half_mode);
    if (m < 0) {
        const char *e = getenv("HV_PVG_HALF");
        m = e && !strcmp(e, "linear") ? 1 : 0;
        atomic_store(&hv_half_mode, m);
    }
    return m;
}
static float hv_half_shown(float v, int mode) { /* what the screen shows for a half-float component */
    v = v < 0 ? 0 : v > 1 ? 1 : v;
    return mode == 1 ? (v <= 0.0031308f ? v * 12.92f : 1.055f * powf(v, 1.0f / 2.4f) - 0.055f) : v;
}

#include "pvg_colorprobe.h" /* HV_COLOR_PROBE: what the display is told and shown, for chasing colour problems */

/* A contiguous view of a task's guest pages; holes are left as zero pages. */
/* A range of task memory at one address, page by page. Metal pins a buffer's
   pages when the buffer is made, so a mapping never changes under a buffer: when
   the guest's pages change there is a new mapping and a new buffer. (A GPU still
   writing to pages the guest has freed and used again would corrupt its memory.) */
static BOOL map_pages(HvReplay *r, uint32_t task, HvObject *o) {
    uint64_t n = (o->length + 0x3fff) >> 14;
    vm_address_t base = 0;
    if (vm_allocate(mach_task_self(), &base, n << 14, VM_FLAGS_ANYWHERE) != KERN_SUCCESS) return NO;
    uint32_t *pages = calloc(n, 4);
    hv_pvg_lite_task_pages(r->lite, task, o->vpage, n, pages);
    for (uint64_t i = 0; i < n; i++) {
        uint8_t *src = pages[i] ? hv_pvg_lite_page(r->lite, pages[i]) : NULL;
        if (!src) {
            pages[i] = 0;
            continue;
        }
        vm_address_t to = base + (i << 14);
        vm_prot_t cur, max;
        kern_return_t kr = vm_remap(mach_task_self(), &to, 0x4000, 0, VM_FLAGS_OVERWRITE, mach_task_self(), (vm_address_t)src, FALSE, &cur, &max, VM_INHERIT_NONE);
        if (kr != KERN_SUCCESS) {
            fprintf(stderr, "hengevm: pvg-metal: vm_remap failed: %d\n", kr);
            vm_deallocate(mach_task_self(), base, n << 14);
            free(pages);
            return NO;
        }
    }
    free(o->pages);
    o->pages = pages;
    o->mapped = (uint8_t *)base;
    return YES;
}

/* A buffer over the mapping, that frees it when the last user is done. */
static id<MTLBuffer> buffer_over_mapping(HvReplay *r, HvObject *o) {
    uint8_t *base = o->mapped;
    NSUInteger bytes = (NSUInteger)((o->length + 0x3fff) & ~0x3fffull);
    id<MTLBuffer> b = [r->device newBufferWithBytesNoCopy:base length:bytes options:MTLResourceStorageModeShared
                                              deallocator:^(void *at, NSUInteger size) { vm_deallocate(mach_task_self(), (vm_address_t)at, size); }];
    if (b) o->mapping_in_buffer = YES;
    return b;
}

/* The guest's table for this range against what is mapped: a change means a new mapping. */
static void refresh_pages(HvReplay *r, uint32_t task, HvObject *o) {
    if (!o->mapped) return;
    uint64_t n = (o->length + 0x3fff) >> 14;
    uint32_t stack[256];
    uint32_t *now = n <= 256 ? stack : malloc(n * 4);
    hv_pvg_lite_task_pages(r->lite, task, o->vpage, n, now);
    BOOL same = memcmp(now, o->pages, n * 4) == 0;
    if (now != stack) free(now);
    if (same) return;
    uint8_t *old = o->mapped;
    BOOL old_in_buffer = o->mapping_in_buffer;
    if (!map_pages(r, task, o)) return;
    if (!old_in_buffer) vm_deallocate(mach_task_self(), (vm_address_t)old, n << 14);
    o->mapping_in_buffer = NO;
    o->generation++;
    o->guest_copy = nil;
    if (o->plain_buffer) o->object = buffer_over_mapping(r, o);
    r->remaps++;
}

static void set_framebuffer_only(MTLTextureDescriptor *t, BOOL on) {
    SEL sel = sel_registerName("setFramebufferOnly:");
    if ([t respondsToSelector:sel]) ((void (*)(id, SEL, BOOL))objc_msgSend)(t, sel, on);
}

/* What Apple's host takes from both serialized forms: flags (framebuffer only bit 4, GPU
   optimized bit 6) and resource options, of which iOS has no managed memory. */
static void descriptor_common(MTLTextureDescriptor *t, uint8_t flags, uint16_t options) {
    set_framebuffer_only(t, (flags >> 4) & 1);
    t.allowGPUOptimizedContents = (flags >> 6) & 1;
    MTLResourceOptions o = options & 0x3ff;
#if TARGET_OS_IPHONE
    if ((o & 0xf0) == 0x10) o = (o & ~(MTLResourceOptions)0xf0) | MTLResourceStorageModeShared;
#endif
    t.resourceOptions = o;
}

/* Metal aborts the whole app on a swizzle channel outside 0...5 (seen as 193 when a surface's description was
   read short, see makeMappedTexture), so an unknown one falls back to the identity, logged a few times. */
static MTLTextureSwizzleChannels safe_swizzle(const uint8_t *sw, const uint8_t *desc, size_t desc_len, const char *what) {
    if (!(sw[0] | sw[1] | sw[2] | sw[3])) /* all Zero: never meant, it would draw nothing; an unread field */
        return MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue, MTLTextureSwizzleAlpha);
    for (int i = 0; i < 4; i++) {
        if (sw[i] > MTLTextureSwizzleAlpha) {
            static int told;
            if (told++ < 3) {
                fprintf(stderr, "hengevm: pvg-metal: %s swizzle %u %u %u %u is not a Metal swizzle; using RGBA. desc:", what, sw[0], sw[1], sw[2], sw[3]);
                for (size_t j = 0; j < desc_len; j++) fprintf(stderr, " %02x", desc[j]);
                fputc('\n', stderr);
            }
            return MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue, MTLTextureSwizzleAlpha);
        }
    }
    return MTLTextureSwizzleChannelsMake(sw[0], sw[1], sw[2], sw[3]);
}

/* The two serialized forms of a texture description. */
static MTLTextureDescriptor *descriptor_v1(const uint8_t *d) {
    MTLTextureDescriptor *t = [MTLTextureDescriptor new];
    t.textureType = (MTLTextureType)(d[0] & 0xf);
    t.usage = d[1];
    t.pixelFormat = (MTLPixelFormat)(d[2] | d[3] << 8);
    t.width = *(const uint32_t *)(d + 4);
    t.height = *(const uint32_t *)(d + 8);
    t.depth = *(const uint32_t *)(d + 12);
    t.mipmapLevelCount = *(const uint16_t *)(d + 16);
    t.sampleCount = *(const uint16_t *)(d + 18);
    t.arrayLength = *(const uint16_t *)(d + 20);
    descriptor_common(t, d[0], *(const uint16_t *)(d + 22));
    return t;
}

static MTLTextureDescriptor *descriptor_v2(const uint8_t *d) {
    MTLTextureDescriptor *t = [MTLTextureDescriptor new];
    uint32_t usage, w, h, depth;
    uint16_t mips, samples, array, options;
    memcpy(&usage, d + 3, 4), memcpy(&w, d + 7, 4), memcpy(&h, d + 11, 4), memcpy(&depth, d + 15, 4);
    memcpy(&mips, d + 19, 2), memcpy(&samples, d + 21, 2), memcpy(&array, d + 23, 2), memcpy(&options, d + 25, 2);
    t.textureType = (MTLTextureType)(d[0] & 0xf);
    t.pixelFormat = (MTLPixelFormat)(d[1] | d[2] << 8);
    t.usage = usage;
    t.width = w, t.height = h, t.depth = depth;
    t.mipmapLevelCount = mips, t.sampleCount = samples, t.arrayLength = array;
    descriptor_common(t, d[0], options);
    t.swizzle = safe_swizzle(d + 0x23, d, 0x27, "texture");
    if (t.usage & MTLTextureUsageRenderTarget) /* Metal aborts on a swizzled render target */
        t.swizzle = MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue, MTLTextureSwizzleAlpha);
    return t;
}

@implementation HvContext
- (void)resetFailureState { failure = nil; }
- (BOOL)hadLookupFailure { return failure != nil; }
- (NSString *)firstFailureReason { return failure; }
- (id)functionWithFunctionRef:(unsigned long long)ref {
    HvObject *o = [replay object:(uint32_t)ref task:task];
    if (!o || !o->object) {
        if (!failure) failure = [NSString stringWithFormat:@"function %llu", ref];
        return standIn;
    }
    return o->object;
}
@end

@implementation HvReplay

- (void)complain:(NSString *)what {
    if ([complaints containsObject:what]) return;
    [complaints addObject:what];
    fprintf(stderr, "hengevm: pvg-metal: %s\n", what.UTF8String);
}

/* A kernel that does nothing. Metal's deserializer aborts the whole app when a compute pipeline's function is nil, which
   happens when this OS cannot load the library macOS sent ("This library format is not supported on this platform", seen on
   iPadOS 17 with macOS 15). The lookup has failed either way, so makeSerialized drops the pipeline: a missing effect, not a
   crash. */
- (id)computeStandIn {
    if (!stand_in) {
        NSError *err = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:@"#include <metal_stdlib>\nkernel void hv_stand_in() {}\n" options:nil error:&err];
        stand_in = [lib newFunctionWithName:@"hv_stand_in"];
        if (!stand_in) [self complain:[NSString stringWithFormat:@"stand-in kernel: %@", err.localizedDescription]];
    }
    return stand_in;
}

- (id)makeFunction:(uint64_t)at task:(uint32_t)task {
    if (getenv("HV_PVG_FAIL_LIBS")) { /* testing: every library refused, as an OS that cannot load them does */
        [self complain:@"library: refused (HV_PVG_FAIL_LIBS)"];
        return nil;
    }
    struct __attribute__((packed)) { uint64_t off; uint32_t len; uint32_t tag, size; } h;
    if (hv_pvg_lite_task_read(lite, task, at, &h, sizeof h)) return nil;
    NSMutableData *lib = [NSMutableData dataWithLength:h.len];
    if (hv_pvg_lite_task_read(lite, task, h.off, lib.mutableBytes, h.len)) return nil;
    /* iOS 17 refuses macOS's libraries as they are (a macOS 15 build, AIR 2.7): relabelled as iOS 17 ones
       (devices/metallib.c), header first, then the bitcode's target too, it may load them. The stage that
       worked is tried first from then on. HV_PVG_NO_RELABEL: never. */
    static int start_stage = -1; /* 0: as sent; 1, 2: relabelled */
    static int no_relabel = -1;
    if (no_relabel < 0) no_relabel = getenv("HV_PVG_NO_RELABEL") != NULL;
    if (start_stage < 0) { /* HV_PVG_RELABEL=N (testing): every library relabelled at stage N from the start */
        const char *f = getenv("HV_PVG_RELABEL");
        start_stage = f && atoi(f) >= 1 && atoi(f) <= 2 ? atoi(f) : 0;
    }
    NSError *err = nil, *first_err = nil;
    id<MTLLibrary> library = nil;
    for (int stage = no_relabel ? 0 : start_stage; stage <= (no_relabel ? 0 : 2) && !library; stage++) {
        NSMutableData *try = lib;
        if (stage) {
            try = [lib mutableCopy];
            if (hv_metallib_relabel(try.mutableBytes, try.length, stage) <= 0) continue;
        }
        dispatch_data_t data = dispatch_data_create(try.bytes, try.length, NULL, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        err = nil;
        library = [device newLibraryWithData:data error:&err];
        if (!library && !first_err) first_err = err;
        if (library && (stage != start_stage || (stage && first_err == nil && !getenv("HV_PVG_RELABEL_QUIET")))) {
            [self complain:[NSString stringWithFormat:@"library: loaded relabelled for iOS 17 (stage %d; as sent: %@)", stage,
                                                      first_err.localizedDescription ?: @"not tried"]];
            start_stage = stage;
        } else if (!library && stage) {
            [self complain:[NSString stringWithFormat:@"library: relabelled (stage %d): %@", stage, err.localizedDescription]];
        }
    }
    if (!library) {
        [self complain:[NSString stringWithFormat:@"library: %@", (first_err ?: err).localizedDescription]];
        return nil;
    }
    {   /* m4 fork: while tracing, keep each library macOS sends, to read its shaders on a Mac */
        const char *tf = getenv("HV_PVG_TRACE_FILE");
        if (tf && getenv("HV_SHADER_DUMP") && atoi(getenv("HV_SHADER_DUMP"))) {
            NSString *dir = [[[NSString stringWithUTF8String:tf] stringByDeletingLastPathComponent] stringByAppendingPathComponent:@"shaders"];
            [[NSFileManager defaultManager] createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil];
            NSString *name = [NSString stringWithFormat:@"%@-%llx.metallib", library.functionNames.firstObject ?: @"lib", (unsigned long long)at];
            [lib writeToFile:[dir stringByAppendingPathComponent:name] atomically:NO];
        }
    }
    return [library newFunctionWithName:library.functionNames.firstObject];
}

- (id)makeSerialized:(uint64_t)at length:(uint32_t)size task:(uint32_t)task {
    NSMutableData *all = [NSMutableData dataWithLength:size];
    if (hv_pvg_lite_task_read(lite, task, at, all.mutableBytes, size)) return nil;
    const uint32_t *w = all.bytes;
    uint32_t tag = w[0];
    if (tag == 0xe || tag == 0x33) { /* render or tile pipeline: {tag, size, ref, length, data} */
        id<HvMetalSPI> spi = (id)device;
        HvContext *ctx = [HvContext new];
        ctx->replay = self, ctx->task = task;
        dispatch_data_t data = dispatch_data_create((const uint8_t *)all.bytes + 16, w[3], NULL, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        id descriptor = tag == 0xe ? [spi newRenderPipelineDescriptorWithSerializedData:data deserializationContext:ctx]
                                   : [spi newTileRenderPipelineDescriptorWithSerializedData:data deserializationContext:ctx];
        if (!descriptor || ctx->failure) {
            [self complain:[NSString stringWithFormat:@"pipeline descriptor %u: %@", tag, ctx->failure]];
            return nil;
        }
        NSError *err = nil;
        id state = tag == 0xe ? [device newRenderPipelineStateWithDescriptor:descriptor error:&err]
                              : [device newRenderPipelineStateWithTileDescriptor:descriptor options:MTLPipelineOptionNone reflection:nil error:&err];
        if (!state) [self complain:[NSString stringWithFormat:@"pipeline %u: %@", tag, err.localizedDescription]];
        if (tag == 0xe && getenv("HV_SHADER_DUMP") && atoi(getenv("HV_SHADER_DUMP"))) { /* m4 fork: blend state as this iOS read it */
            MTLRenderPipelineDescriptor *rp = descriptor;
            for (int i = 0; i < 4; i++) {
                MTLRenderPipelineColorAttachmentDescriptor *c = rp.colorAttachments[i];
                if (c.pixelFormat == MTLPixelFormatInvalid) continue;
                fprintf(stderr, "hvpipe: %s c%d fmt %lu blend %d rgb %lu*src %lu %lu*dst alpha %lu*src %lu %lu*dst mask %lx\n",
                        rp.fragmentFunction.name.UTF8String ?: "?", i, (unsigned long)c.pixelFormat, (int)c.blendingEnabled,
                        (unsigned long)c.sourceRGBBlendFactor, (unsigned long)c.rgbBlendOperation, (unsigned long)c.destinationRGBBlendFactor,
                        (unsigned long)c.sourceAlphaBlendFactor, (unsigned long)c.alphaBlendOperation, (unsigned long)c.destinationAlphaBlendFactor,
                        (unsigned long)c.writeMask);
            }
        }
        if (state && tag == 0xe && getenv("HV_PVG_TRACE_FILE")) {
            MTLRenderPipelineDescriptor *rp = descriptor;
            objc_setAssociatedObject(state, "hvname", [NSString stringWithFormat:@"%@ / %@ -> fmt %lu blend %d", rp.vertexFunction.name, rp.fragmentFunction.name, (unsigned long)rp.colorAttachments[0].pixelFormat, (int)rp.colorAttachments[0].blendingEnabled], OBJC_ASSOCIATION_RETAIN);
        }
        return state;
    }
    if (tag == 0xb) { /* compute pipeline: {tag, size, ref, length, data} */
        id<HvMetalSPI> spi = (id)device;
        HvContext *ctx = [HvContext new];
        ctx->replay = self, ctx->task = task;
        dispatch_data_t data = dispatch_data_create((const uint8_t *)all.bytes + 16, w[3], NULL, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        ctx->standIn = [self computeStandIn];
        id descriptor = [spi newComputePipelineDescriptorWithSerializedData:data deserializationContext:ctx];
        if (!descriptor || ctx->failure) {
            [self complain:[NSString stringWithFormat:@"compute descriptor: %@", ctx->failure]];
            return nil;
        }
        NSError *err = nil;
        id state = [device newComputePipelineStateWithDescriptor:descriptor options:MTLPipelineOptionNone reflection:nil error:&err];
        if (!state) [self complain:[NSString stringWithFormat:@"compute pipeline: %@", err.localizedDescription]];
        return state;
    }
    if (tag == 3) { /* sampler: {tag, size, ref, bits, lod min, lod max, ...} */
        uint64_t bits;
        memcpy(&bits, w + 3, 8);
        MTLSamplerDescriptor *d = [MTLSamplerDescriptor new];
        d.minFilter = (MTLSamplerMinMagFilter)(bits & 3);
        d.magFilter = (MTLSamplerMinMagFilter)((bits >> 2) & 3);
        d.lodAverage = (bits >> 4) & 1;
        d.compareFunction = (MTLCompareFunction)((bits >> 5) & 7);
        d.sAddressMode = (MTLSamplerAddressMode)((bits >> 8) & 15);
        d.tAddressMode = (MTLSamplerAddressMode)((bits >> 12) & 15);
        d.rAddressMode = (MTLSamplerAddressMode)((bits >> 16) & 15);
        d.borderColor = (MTLSamplerBorderColor)((bits >> 20) & 15);
        d.mipFilter = (MTLSamplerMipFilter)((bits >> 24) & 3);
        NSUInteger aniso = (bits >> 26) & 31;
        d.maxAnisotropy = aniso ? aniso : 1;
        d.normalizedCoordinates = (bits >> 31) & 1;
        d.supportArgumentBuffers = (bits >> 32) & 1;
        float lod[2];
        memcpy(lod, w + 5, 8);
        d.lodMinClamp = lod[0], d.lodMaxClamp = lod[1];
        id s = [device newSamplerStateWithDescriptor:d];
        if (!s) [self complain:@"sampler failed"];
        return s;
    }
    if (tag == 0xd) return [device newFence];
    if (tag == 4) { /* depth and stencil: {tag, size, ref, flags, front {ops, read, write}, back {...}} */
        MTLDepthStencilDescriptor *d = [MTLDepthStencilDescriptor new];
        uint32_t f = w[3];
        d.depthCompareFunction = (MTLCompareFunction)(f & 7);
        d.depthWriteEnabled = (f >> 3) & 1;
        for (int side = 0; side < 2; side++) {
            if (!((f >> (4 + side)) & 1)) continue;
            const uint32_t *t = w + 4 + side * 3;
            MTLStencilDescriptor *sd = [MTLStencilDescriptor new];
            sd.stencilCompareFunction = (MTLCompareFunction)(t[0] & 7);
            sd.stencilFailureOperation = (MTLStencilOperation)((t[0] >> 3) & 7);
            sd.depthFailureOperation = (MTLStencilOperation)((t[0] >> 6) & 7);
            sd.depthStencilPassOperation = (MTLStencilOperation)((t[0] >> 9) & 7);
            sd.readMask = t[1], sd.writeMask = t[2];
            if (side == 0) d.frontFaceStencil = sd;
            else d.backFaceStencil = sd;
        }
        return [device newDepthStencilStateWithDescriptor:d];
    }
    [self complain:[NSString stringWithFormat:@"serialized object tag %u", tag]];
    return nil;
}

/* A heap: {guest page, length, flags}. As on Apple's host its memory is the GPU's
   own, filled from the guest's pages when the guest says those changed. */
- (void)makeHeap:(HvObject *)o at:(uint64_t)at task:(uint32_t)task {
    struct __attribute__((packed)) { uint32_t vpage; uint64_t length; uint8_t flags; } h;
    if (hv_pvg_lite_task_read(lite, task, at, &h, sizeof h)) return;
    MTLHeapDescriptor *d = [MTLHeapDescriptor new];
    d.size = (NSUInteger)h.length;
    d.type = MTLHeapTypePlacement;
    d.storageMode = MTLStorageModePrivate;
    d.hazardTrackingMode = h.flags & 2 ? MTLHazardTrackingModeTracked : MTLHazardTrackingModeDefault;
    id<MTLHeap> heap = [device newHeapWithDescriptor:d];
    if (!heap && d.hazardTrackingMode == MTLHazardTrackingModeTracked) { /* not every GPU tracks placement heaps */
        d.hazardTrackingMode = MTLHazardTrackingModeDefault;
        heap = [device newHeapWithDescriptor:d];
        [self complain:@"placement heaps here cannot be tracked"];
    }
    if (!heap) { [self complain:[NSString stringWithFormat:@"heap of %llu bytes failed", h.length]]; return; }
    o->object = heap;
    o->heap_whole = [heap newBufferWithLength:(NSUInteger)h.length options:MTLResourceStorageModePrivate offset:0];
    o->length = (h.length + 0x3fff) & ~0x3fffull;
    o->vpage = h.vpage;
    o->guest_backed = YES;
    heap_bytes += h.length;
    if (getenv("HV_PVG_HEAP_LOG")) fprintf(stderr, "hengevm: pvg-metal: heap %llu KB at page %u, flags %x, now %llu MB of heaps\n", h.length >> 10, h.vpage, h.flags, heap_bytes >> 20);
}

/* A buffer in a heap: {heap, offset, length}. */
- (void)makeHeapBuffer:(HvObject *)o at:(uint64_t)at task:(uint32_t)task {
    struct __attribute__((packed)) { uint32_t heap; uint64_t offset, length; } h;
    if (hv_pvg_lite_task_read(lite, task, at, &h, sizeof h)) return;
    HvObject *parent = [self object:h.heap task:task];
    id<MTLHeap> heap = parent ? parent->object : nil;
    o->parent = parent;
    o->object = [heap newBufferWithLength:(NSUInteger)h.length options:MTLResourceStorageModePrivate offset:(NSUInteger)h.offset];
    if (!o->object) {
        [self complain:[NSString stringWithFormat:@"heap buffer: %llu bytes at %llu of ref %u failed", h.length, h.offset, h.heap]];
        o->object = [device newBufferWithLength:(NSUInteger)h.length options:MTLResourceStorageModePrivate];
    }
}

- (void)makeBuffer:(HvObject *)o at:(uint64_t)at task:(uint32_t)task {
    struct __attribute__((packed)) { uint64_t length; uint32_t vpage; uint32_t flags; } h;
    if (hv_pvg_lite_task_read(lite, task, at, &h, sizeof h)) return;
    o->length = (h.length + 0x3fff) & ~0x3fffull;
    o->vpage = h.vpage;
    if (!map_pages(self, task, o)) return;
    o->plain_buffer = YES;
    o->object = buffer_over_mapping(self, o);
    if (!o->object) [self complain:@"buffer failed"];
}

- (void)makeTexture:(HvObject *)o at:(uint64_t)at size:(uint32_t)size task:(uint32_t)task {
    NSMutableData *all = [NSMutableData dataWithLength:size];
    if (hv_pvg_lite_task_read(lite, task, at, all.mutableBytes, size)) return;
    const uint8_t *d = all.bytes;
    /* {length, page}, a layout table, {tag, size, ref}, the description. */
    memcpy(&o->length, d, 8);
    memcpy(&o->vpage, d + 8, 4);
    const uint8_t *layout = d + 12;
    o->tex_mips = *(const uint16_t *)layout & 0x3fff;
    o->tex_slices = *(const uint16_t *)(layout + 2);
    memcpy(&o->tex_base, layout + 4, 8);
    memcpy(&o->tex_stride, layout + 12, 8);
    o->tex_block_h = layout[0x15];
    uint32_t table = o->tex_mips * 36;
    if (12 + 0x18 + table + 12 > size) { [self complain:@"texture entry too small"]; return; }
    o->tex_layout = malloc(table);
    memcpy(o->tex_layout, layout + 0x18, table);
    const uint8_t *rec = layout + 0x18 + table;
    uint32_t tag = *(const uint32_t *)rec;
    MTLTextureDescriptor *t = (tag == 0x34 || tag == 0x39) ? descriptor_v2(rec + 12) : descriptor_v1(rec + 12);
    t.storageMode = MTLStorageModePrivate;
    t.usage |= MTLTextureUsageShaderRead;
    o->object = [device newTextureWithDescriptor:t];
    o->guest_backed = YES;
    if (!o->object) [self complain:[NSString stringWithFormat:@"texture format %lu %lux%lu failed", (unsigned long)t.pixelFormat, (unsigned long)t.width, (unsigned long)t.height]];
}

/* A texture over a buffer: {tag 9, size, ref, buffer ref, offset, bytes per row, description}. */
- (void)makeLinearTexture:(HvObject *)o at:(uint64_t)at size:(uint32_t)size task:(uint32_t)task {
    struct __attribute__((packed)) { uint32_t tag, size, ref, buffer; uint64_t offset, row; uint8_t d[64]; } h;
    memset(&h, 0, sizeof h);
    if (hv_pvg_lite_task_read(lite, task, at, &h, MIN(size, (uint32_t)sizeof h))) return;
    if (h.tag == 0x34 || h.tag == 0x14 || h.tag == 1) { /* no guest memory at all: {tag, size, ref, description} */
        const uint8_t *desc = (const uint8_t *)&h.buffer;
        MTLTextureDescriptor *t = h.tag == 0x34 ? descriptor_v2(desc) : descriptor_v1(desc);
        t.storageMode = MTLStorageModePrivate;
        o->object = [device newTextureWithDescriptor:t];
        if (!o->object) [self complain:[NSString stringWithFormat:@"texture format %lu %lux%lu samples %lu failed", (unsigned long)t.pixelFormat, (unsigned long)t.width, (unsigned long)t.height, (unsigned long)t.sampleCount]];
        return;
    }
    if (h.tag == 0x15 || h.tag == 0x38) { /* in a heap: {tag, size, ref, heap, description, offset after 0x24 or 0x2c} */
        const uint8_t *desc = (const uint8_t *)&h.offset;
        uint64_t offset;
        memcpy(&offset, desc + (h.tag == 0x38 ? 0x2c : 0x24), 8);
        HvObject *parent = [self object:h.buffer task:task];
        id<MTLHeap> heap = parent ? parent->object : nil;
        MTLTextureDescriptor *t = h.tag == 0x38 ? descriptor_v2(desc) : descriptor_v1(desc);
        t.resourceOptions = heap ? heap.resourceOptions : MTLResourceStorageModePrivate;
        o->parent = parent;
        o->object = [heap newTextureWithDescriptor:t offset:(NSUInteger)offset];
        if (!o->object) {
            [self complain:[NSString stringWithFormat:@"heap texture format %lu %lux%lu at %llu of ref %u failed", (unsigned long)t.pixelFormat, (unsigned long)t.width, (unsigned long)t.height, offset, h.buffer]];
            t.storageMode = MTLStorageModePrivate;
            o->object = [device newTextureWithDescriptor:t];
        }
        return;
    }
    if (h.tag == 7 || h.tag == 8 || h.tag == 0x1b) { /* a view: {tag, size, ref, parent, format u16, type u16, levels, slices, swizzle} */
        struct __attribute__((packed)) { uint16_t format, type; uint64_t levels[2], slices[2]; uint8_t swizzle[4]; } v;
        memcpy(&v, &h.offset, sizeof v);
        HvObject *parent = [self object:h.buffer task:task];
        id<MTLTexture> from = parent ? parent->object : nil;
        if (!from) { [self complain:[NSString stringWithFormat:@"view of nothing (tag %u, ref %u)", h.tag, h.buffer]]; return; }
        o->parent = parent;
        o->surface = parent->surface;
        NSRange levels = NSMakeRange((NSUInteger)v.levels[0], (NSUInteger)v.levels[1]), slices = NSMakeRange((NSUInteger)v.slices[0], (NSUInteger)v.slices[1]);
        if (h.tag == 7) o->object = [from newTextureViewWithPixelFormat:(MTLPixelFormat)v.format];
        else if (h.tag == 8) o->object = [from newTextureViewWithPixelFormat:(MTLPixelFormat)v.format textureType:(MTLTextureType)v.type levels:levels slices:slices];
        else o->object = [from newTextureViewWithPixelFormat:(MTLPixelFormat)v.format textureType:(MTLTextureType)v.type levels:levels slices:slices
                                                     swizzle:safe_swizzle(v.swizzle, (const uint8_t *)&v, sizeof v, "view")];
        if (!o->object) [self complain:[NSString stringWithFormat:@"view tag %u of format %lu as %u failed", h.tag, (unsigned long)from.pixelFormat, v.format]];
        return;
    }
    if (h.tag != 9 && h.tag != 0x37) {
        uint32_t w[24] = {0};
        hv_pvg_lite_task_read(lite, task, at, w, sizeof w);
        NSMutableString *m = [NSMutableString stringWithFormat:@"serialized texture tag %u:", h.tag];
        for (int i = 0; i < 24; i++) [m appendFormat:@" %08x", w[i]];
        [self complain:m];
        return;
    }
    HvObject *b = [self object:h.buffer task:task];
    id<MTLBuffer> buffer = b ? b->object : nil;
    if (!buffer || ![buffer conformsToProtocol:@protocol(MTLBuffer)]) { [self complain:@"linear texture without a buffer"]; return; }
    MTLTextureDescriptor *t = h.tag == 0x37 ? descriptor_v2(h.d) : descriptor_v1(h.d);
    t.resourceOptions = buffer.resourceOptions;
    set_framebuffer_only(t, NO);
    t.usage |= MTLTextureUsageShaderRead;
    o->parent = b;
    o->follows = b, o->follows_generation = b->generation;
    o->object = [buffer newTextureWithDescriptor:t offset:h.offset bytesPerRow:h.row];
    if (!o->object) [self complain:[NSString stringWithFormat:@"linear texture format %lu row %llu failed", (unsigned long)t.pixelFormat, h.row]];
}

/* A multi-plane surface as a real IOSurface, laid out as the guest laid it out.
   Planes are in the descriptor from +0x80, 0x80 apart: offset +8, size +0x18,
   width +0x20, height +0x28, row +0x30, bytes per element +0x38. */
static IOSurfaceRef make_iosurface(HvObject *to, const uint8_t *desc, uint64_t alloc) {
    uint32_t fourcc, planes = desc[0x3e];
    uint64_t width, height;
    memcpy(&fourcc, desc + 4, 4), memcpy(&width, desc + 0x20, 8), memcpy(&height, desc + 0x28, 8);
    NSMutableArray *info = [NSMutableArray new], *plain = [NSMutableArray new];
    uint64_t at = 0;
    for (uint32_t i = 0; i < planes && i < 4; i++) {
        const uint8_t *pl = desc + 0x80 + i * 0x80;
        uint64_t off, size, w, h, row;
        uint16_t bpe;
        memcpy(&off, pl + 8, 8), memcpy(&size, pl + 0x18, 8), memcpy(&w, pl + 0x20, 8), memcpy(&h, pl + 0x28, 8), memcpy(&row, pl + 0x30, 8), memcpy(&bpe, pl + 0x38, 2);
        [info addObject:@{(__bridge id)kIOSurfacePlaneWidth: @(w), (__bridge id)kIOSurfacePlaneHeight: @(h), (__bridge id)kIOSurfacePlaneBytesPerRow: @(row),
                          (__bridge id)kIOSurfacePlaneOffset: @(off), (__bridge id)kIOSurfacePlaneSize: @(size), (__bridge id)kIOSurfacePlaneBytesPerElement: @(bpe),
                          (__bridge id)kIOSurfacePlaneElementWidth: @1, (__bridge id)kIOSurfacePlaneElementHeight: @1}];
        [plain addObject:@{(__bridge id)kIOSurfacePlaneWidth: @(w), (__bridge id)kIOSurfacePlaneHeight: @(h), (__bridge id)kIOSurfacePlaneBytesPerRow: @(row),
                           (__bridge id)kIOSurfacePlaneOffset: @(at), (__bridge id)kIOSurfacePlaneSize: @(row * h), (__bridge id)kIOSurfacePlaneBytesPerElement: @(bpe),
                           (__bridge id)kIOSurfacePlaneElementWidth: @1, (__bridge id)kIOSurfacePlaneElementHeight: @1}];
        to->plane_from[i] = off, to->plane_row[i] = row, to->plane_h[i] = h;
        at += row * h;
    }
    NSDictionary *props = @{(__bridge id)kIOSurfaceWidth: @(width), (__bridge id)kIOSurfaceHeight: @(height), (__bridge id)kIOSurfacePixelFormat: @(fourcc),
                            (__bridge id)kIOSurfaceAllocSize: @(alloc), (__bridge id)kIOSurfacePlaneInfo: info};
    IOSurfaceRef made = IOSurfaceCreate((__bridge CFDictionaryRef)props);
    if (!made) {
        /* Video formats can refuse the guest's padding: request packed planes instead. */
        NSDictionary *plain_props = @{(__bridge id)kIOSurfaceWidth: @(width), (__bridge id)kIOSurfaceHeight: @(height), (__bridge id)kIOSurfacePixelFormat: @(fourcc), (__bridge id)kIOSurfacePlaneInfo: plain};
        made = IOSurfaceCreate((__bridge CFDictionaryRef)plain_props);
    }
    if (made) {
        to->plane_count = planes;
        static int logged;
        if (planes >= 2 && logged++ < 4)
            for (uint32_t i = 0; i < planes && i < IOSurfaceGetPlaneCount(made) && i < 4; i++)
                fprintf(stderr, "hengevm: video plane %u guest off %llu row %llu h %llu; host row %zu h %zu\n",
                        i, (unsigned long long)to->plane_from[i], (unsigned long long)to->plane_row[i],
                        (unsigned long long)to->plane_h[i], IOSurfaceGetBytesPerRowOfPlane(made, i),
                        IOSurfaceGetHeightOfPlane(made, i));
    }
    return made;
}

/* A sample of every 64th byte, to tell cheaply whether pixels changed. */
static uint64_t hash_bytes(const uint8_t *from, uint64_t len) {
    uint64_t hash = 0xcbf29ce484222325ull;
    for (uint64_t i = 0; i + 8 <= len; i += 64) {
        uint64_t v;
        memcpy(&v, from + i, 8);
        hash = (hash ^ v) * 0x100000001b3ull;
    }
    return hash;
}

static uint64_t guest_hash(HvObject *s) { return hash_bytes(s->mapped + s->first, s->alloc); }

/* Plane i of a multi-plane surface in the guest's memory, or 0 bytes if its layout does not fit the allocation. */
static uint64_t plane_bytes(HvObject *s, uint32_t i) {
    uint64_t row = s->plane_row[i], off = s->plane_from[i];
    if (!row || off > s->alloc || s->plane_h[i] > (s->alloc - off) / row) return 0;
    return row * s->plane_h[i];
}

/* Every plane's hash as the guest has it now, for a host surface already holding those pixels. */
static void note_plane_hashes(HvObject *s) {
    for (uint32_t i = 0; i < s->plane_count && i < 4; i++)
        s->plane_hash[i] = hash_bytes(s->mapped + s->first + s->plane_from[i], plane_bytes(s, i));
}

/* Guest pixels into its IOSurface, when they changed. */
static void sync_iosurface(HvObject *s) {
    if (!s->iosurface) return;
    const uint8_t *from = s->mapped + s->first;
    BOOL by_plane = s->plane_count && IOSurfaceGetPlaneCount(s->iosurface) >= s->plane_count;
    uint64_t hashes[4] = {0};
    uint32_t changed = 0;
    if (by_plane) {
        for (uint32_t i = 0; i < s->plane_count && i < 4; i++) {
            hashes[i] = hash_bytes(from + s->plane_from[i], plane_bytes(s, i));
            if (hashes[i] != s->plane_hash[i]) changed |= 1u << i;
        }
        if (!changed) return;
    } else {
        uint64_t hash = guest_hash(s);
        if (hash == s->content_hash) return;
        s->content_hash = hash;
    }
    IOSurfaceLock(s->iosurface, 0, NULL);
    /* HV_PVG_HOST_DUMP=prefix (debugging): the host surface as it is just before the guest's pixels go over it,
       for the first few multi-plane surfaces, with its layout in a .txt beside it: did the GPU draw something here
       that this copy throws away? */
    const char *host_dump = getenv("HV_PVG_HOST_DUMP");
    static int host_dumped;
    if (host_dump && s->plane_count && host_dumped < 12) {
        char path[1100];
        snprintf(path, sizeof path, "%s-%d.bin", host_dump, host_dumped);
        FILE *f = fopen(path, "wb");
        if (f) fwrite(IOSurfaceGetBaseAddress(s->iosurface), 1, IOSurfaceGetAllocSize(s->iosurface), f), fclose(f);
        snprintf(path, sizeof path, "%s-%d.txt", host_dump, host_dumped++);
        f = fopen(path, "w");
        if (f) {
            fprintf(f, "fourcc %08x w %zu h %zu alloc %zu guest %llu\n", IOSurfaceGetPixelFormat(s->iosurface), IOSurfaceGetWidth(s->iosurface),
                    IOSurfaceGetHeight(s->iosurface), IOSurfaceGetAllocSize(s->iosurface), (unsigned long long)s->alloc);
            for (size_t i = 0; i < IOSurfaceGetPlaneCount(s->iosurface); i++)
                fprintf(f, "plane %zu off %ld row %zu h %zu w %zu bpe %zu\n", i,
                        (long)((uint8_t *)IOSurfaceGetBaseAddressOfPlane(s->iosurface, i) - (uint8_t *)IOSurfaceGetBaseAddress(s->iosurface)),
                        IOSurfaceGetBytesPerRowOfPlane(s->iosurface, i), IOSurfaceGetHeightOfPlane(s->iosurface, i),
                        IOSurfaceGetWidthOfPlane(s->iosurface, i), IOSurfaceGetBytesPerElementOfPlane(s->iosurface, i));
            fclose(f);
        }
    }
    if (by_plane) {
        /* IOSurface can choose different offsets and row strides from those
           requested. Copy each changed video plane through its actual host layout. */
        for (uint32_t i = 0; i < s->plane_count && i < 4; i++) {
            if (!(changed & 1u << i)) continue;
            s->plane_hash[i] = hashes[i];
            uint8_t *dst = IOSurfaceGetBaseAddressOfPlane(s->iosurface, i);
            size_t dst_row = IOSurfaceGetBytesPerRowOfPlane(s->iosurface, i);
            size_t rows = MIN((size_t)s->plane_h[i], IOSurfaceGetHeightOfPlane(s->iosurface, i));
            uint64_t src_row = s->plane_row[i], src_off = s->plane_from[i];
            if (!dst || !src_row || src_off > s->alloc || rows > (s->alloc - src_off) / src_row) continue;
            size_t bytes = MIN((size_t)src_row, dst_row);
            for (size_t y = 0; y < rows; y++)
                memcpy(dst + y * dst_row, from + src_off + y * src_row, bytes);
        }
    } else {
        uint8_t *base = IOSurfaceGetBaseAddress(s->iosurface);
        memcpy(base, from, MIN(s->alloc, IOSurfaceGetAllocSize(s->iosurface)));
    }
    IOSurfaceUnlock(s->iosurface, 0, NULL);
    const char *dump = getenv("HV_PVG_SURFACE_DUMP");
    static int dumped;
    if (dump && s->plane_count && dumped < 6) {
        char path[1100];
        snprintf(path, sizeof path, "%s-%d.bin", dump, dumped++);
        FILE *f = fopen(path, "wb");
        if (f) fwrite(from, 1, s->alloc, f), fclose(f);
    }
}

/* A surface (IOSurface in the guest): a texture over its guest pages, so the
   guest's CPU and the GPU see the same pixels, as they do through a real one.
   {mapping id, tag, size, ref, description}. Its descriptor: page list at +0
   (physical >> 12, bit 0 present, one per 16 KB), first-page offset +0x60,
   allocation +0x18, bytes per row +0x30. */
- (void)makeMappedTexture:(HvObject *)o at:(uint64_t)at task:(uint32_t)task {
    /* d holds a v2 description (tag 0x39, 0x38 bytes) whose swizzle sits at +0x23..0x26: with only 32 bytes
       read those came from the stack, and macOS 27.2's first surface made Metal abort Henge (swizzle 193). */
    struct __attribute__((packed)) { uint64_t mapping; uint32_t tag, size, ref; uint8_t d[64]; } h;
    memset(&h, 0, sizeof h);
    const uint32_t head = (uint32_t)offsetof(__typeof__(h), d);
    if (hv_pvg_lite_task_read(lite, task, at, &h, head + 32)) return;
    /* size is the description's own length (0x38 for a v2 one), after the header */
    if (h.size > 32 && hv_pvg_lite_task_read(lite, task, at, &h, head + MIN(h.size, (uint32_t)sizeof h.d))) {
        memset(h.d + 32, 0, sizeof h.d - 32); /* past the end of the task: keep what the first read had */
    }
    NSNumber *key = @(h.mapping);
    HvObject *surface = surfaces[key];
    MTLTextureDescriptor *t = h.tag == 0x39 ? descriptor_v2(h.d) : descriptor_v1(h.d);
    if (!surface) {
        const uint8_t *desc = hv_pvg_lite_surface(lite, (uint32_t)h.mapping);
        if (!desc) { [self complain:[NSString stringWithFormat:@"surface %llu has no descriptor", h.mapping]]; return; }
        uint32_t list, first;
        uint64_t alloc, row;
        memcpy(&list, desc, 4), memcpy(&alloc, desc + 0x18, 8), memcpy(&row, desc + 0x30, 8), memcpy(&first, desc + 0x60, 4);
        surface = [HvObject new];
        surface->length = (alloc + first + 0x3fff) & ~0x3fffull;
        surface->row = row;
        surface->first = first;
        uint64_t sw, sh;
        memcpy(&sw, desc + 0x20, 8), memcpy(&sh, desc + 0x28, 8), memcpy(&surface->fourcc, desc + 4, 4);
        surface->surface_w = (uint32_t)sw, surface->surface_h = (uint32_t)sh;
        uint64_t n = surface->length >> 14;
        vm_address_t base = 0;
        const uint32_t *pages = (list & 1) ? hv_pvg_lite_phys(lite, (uint64_t)(list & ~1u) << 12, n * 4) : NULL;
        if (!pages || vm_allocate(mach_task_self(), &base, surface->length, VM_FLAGS_ANYWHERE) != KERN_SUCCESS) {
            [self complain:[NSString stringWithFormat:@"surface %llu has no pages", h.mapping]];
            return;
        }
        surface->mapped = (uint8_t *)base;
        for (uint64_t i = 0; i < n; i++) {
            if (!(pages[i] & 1)) continue;
            uint8_t *src = hv_pvg_lite_phys(lite, (uint64_t)(pages[i] & ~1u) << 12, 0x4000);
            if (!src) continue;
            vm_address_t to = base + (i << 14);
            vm_prot_t cur, max;
            vm_remap(mach_task_self(), &to, 0x4000, 0, VM_FLAGS_OVERWRITE, mach_task_self(), (vm_address_t)src, FALSE, &cur, &max, VM_INHERIT_NONE);
        }
        surface->object = [device newBufferWithBytesNoCopy:surface->mapped length:surface->length options:MTLResourceStorageModeShared
                                               deallocator:^(void *at, NSUInteger size) { vm_deallocate(mach_task_self(), (vm_address_t)at, size); }];
        surface->mapping_in_buffer = surface->object != nil;
        surfaces[key] = surface;
        if (desc[0x3e] >= 2) {
            surface->alloc = alloc;
            IOSurfaceRef made = make_iosurface(surface, desc, alloc);
            /* The same guest memory, format and size under another mapping id (another process): the same host
               IOSurface, or what that process drew is not what this one sees. Its pixels are already there (or
               drawn over since), so they are not copied again now. */
            NSString *pk = [NSString stringWithFormat:@"%x/%08x/%llux%llu/%llu", pages[0] & ~1u, surface->fourcc,
                            (unsigned long long)sw, (unsigned long long)sh, (unsigned long long)alloc];
            id shared = planar[pk];
            if (shared && made) {
                CFRelease(made);
                surface->iosurface = (IOSurfaceRef)CFRetain((__bridge CFTypeRef)shared);
                surface->content_hash = guest_hash(surface);
                note_plane_hashes(surface);
                surface->planar_key = pk;
                planar_uses[pk] = @(planar_uses[pk].unsignedIntValue + 1);
                static int told;
                if (told++ < 8) fprintf(stderr, "hengevm: pvg-metal: surface %llu shares the host surface of the same guest memory (%s)\n", h.mapping, pk.UTF8String);
            } else {
                surface->iosurface = made;
                if (made) {
                    if (planar.count >= 512) [planar removeAllObjects], [planar_uses removeAllObjects]; /* old ones only lose their sharing */
                    planar[pk] = (__bridge id)made;
                    planar_uses[pk] = @1;
                    surface->planar_key = pk;
                }
                if (!surface->iosurface) {
                    NSMutableString *m = [NSMutableString stringWithFormat:@"surface %llu: no IOSurface for its planes (alloc %llu):", h.mapping, alloc];
                    for (int i = 0; i < 96; i++) [m appendFormat:@" %08x", ((const uint32_t *)desc)[i]];
                    [self complain:m];
                }
                sync_iosurface(surface);
            }
        }
        fprintf(stderr, "hengevm: pvg-metal: surface %llu: %lux%lu format %lu, row %llu, %llu KB, fourcc %c%c%c%c planes %u\n", h.mapping, (unsigned long)t.width, (unsigned long)t.height, (unsigned long)t.pixelFormat, row, alloc >> 10, desc[7], desc[6], desc[5], desc[4], desc[0x3e]);
    }
    t.storageMode = MTLStorageModeShared;
    set_framebuffer_only(t, NO);
    t.usage |= MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
    /* Metal aborts on a swizzled render target, and a surface is always drawn into here: views over it
       (tags 7/8/0x1b) can still swizzle. */
    MTLTextureSwizzleChannels sw = t.swizzle;
    if (sw.red != MTLTextureSwizzleRed || sw.green != MTLTextureSwizzleGreen || sw.blue != MTLTextureSwizzleBlue || sw.alpha != MTLTextureSwizzleAlpha) {
        static int told;
        if (told++ < 3) fprintf(stderr, "hengevm: pvg-metal: surface %llu asked for swizzle %u %u %u %u; drawing it unswizzled\n", h.mapping, sw.red, sw.green, sw.blue, sw.alpha);
        t.swizzle = MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue, MTLTextureSwizzleAlpha);
    }
    o->surface = surface;
    /* The plane the texture is over, after the description in the guest's record (PGSerializedTextureDescriptor
       then plane and rotation, 16 bits each: record +0x24 for v1, +0x2c for v2, the record starting at ref).
       macOS draws video and Safari's page tiles one plane at a time (R8 luma, RG8 chroma, R8 alpha): every view
       made over plane 0 left plane 1 empty, so chroma read back as zero (all green) and alpha as zero (blank pages). */
    uint16_t plane;
    memcpy(&plane, h.d + (h.tag == 0x39 ? 0x28 : 0x20), 2);
    static int plane0 = -1; /* HV_PVG_PLANE0 (testing): every view on plane 0, as before this was read */
    if (plane0 < 0) plane0 = getenv("HV_PVG_PLANE0") != NULL;
    if (plane0) plane = 0;
    if (surface->iosurface && plane >= IOSurfaceGetPlaneCount(surface->iosurface)) {
        [self complain:[NSString stringWithFormat:@"surface %llu: texture on plane %u of %zu; using plane 0", h.mapping, plane,
                        IOSurfaceGetPlaneCount(surface->iosurface)]];
        plane = 0;
    }
    if (plane) {
        static int told;
        if (told++ < 8) fprintf(stderr, "hengevm: pvg-metal: surface %llu: texture %lux%lu format %lu on plane %u\n", h.mapping,
                                (unsigned long)t.width, (unsigned long)t.height, (unsigned long)t.pixelFormat, plane);
    }
    if (surface->iosurface) {
        o->object = [device newTextureWithDescriptor:t iosurface:surface->iosurface plane:plane];
        if (!o->object) { /* video formats cannot be drawn into */
            t.usage = MTLTextureUsageShaderRead;
            o->object = [device newTextureWithDescriptor:t iosurface:surface->iosurface plane:plane];
        }
    }
    else o->object = [(id<MTLBuffer>)surface->object newTextureWithDescriptor:t offset:surface->first bytesPerRow:surface->row];
    if (!o->object) {
        [self complain:[NSString stringWithFormat:@"surface %llu: format %lu row %llu not possible", h.mapping, (unsigned long)t.pixelFormat, surface->row]];
        t.storageMode = MTLStorageModePrivate;
        o->object = [device newTextureWithDescriptor:t]; /* something to bind: it stays empty */
    }
    if (!plane) mapped_textures[key] = o->object;
}

- (HvObject *)object:(uint32_t)ref task:(uint32_t)task {
    if (!ref) return nil;
    uint64_t list = hv_pvg_lite_task_objects(lite, task);
    uint32_t e[3];
    if (!list || hv_pvg_lite_task_read(lite, task, list + (uint64_t)ref * 12, e, 12)) return nil;
    NSMutableDictionary *objs = tasks[@(task)];
    if (!objs) tasks[@(task)] = objs = [NSMutableDictionary new];
    HvObject *o = objs[@(ref)];
    if (o && !memcmp(o->entry, e, 12)) {
        if (!o->follows) return o;
        if (o->follows->mapped) refresh_pages(self, task, o->follows);
        if (o->follows->generation == o->follows_generation) return o;
    }
    o = [HvObject new];
    memcpy(o->entry, e, 12);
    objs[@(ref)] = o;
    uint32_t type = e[0] & 0xff, size = e[0] >> 8;
    uint64_t at = (uint64_t)e[2] << 32 | e[1];
    switch (type) {
    case OBJ_BUFFER: [self makeBuffer:o at:at task:task]; break;
    case OBJ_TEXTURE: [self makeTexture:o at:at size:size task:task]; break;
    case OBJ_FUNCTION: o->object = [self makeFunction:at task:task]; break;
    case OBJ_SERIALIZED: o->object = [self makeSerialized:at length:size task:task]; break;
    case OBJ_LINEAR_TEXTURE: case 9: [self makeLinearTexture:o at:at size:size task:task]; break;
    case 13: [self makeHeap:o at:at task:task]; break;
    case 14: [self makeHeapBuffer:o at:at task:task]; break;
    case 19: /* macOS 27: a buffer, with the same {length, page, flags} entry as type 1 (16 KB ones seen). The guest
                has object info (tags 450...) written into these, so leaving them out broke its argument buffers:
                black layers, and "info 450 ... no buffer" over and over */
        [self makeBuffer:o at:at task:task];
        break;
    case 0: break; /* an empty slot */
    case OBJ_MAPPED_TEXTURE: [self makeMappedTexture:o at:at task:task]; break;
    default: {
        uint32_t w[24] = {0};
        hv_pvg_lite_task_read(lite, task, at, w, sizeof w);
        NSMutableString *m = [NSMutableString stringWithFormat:@"object type %u size %u:", type, size];
        for (int i = 0; i < 24; i++) [m appendFormat:@" %08x", w[i]];
        [self complain:m];
        break;
    }
    }
    return o;
}

- (id)get:(uint32_t)ref task:(uint32_t)task {
    HvObject *o = [self object:ref task:task];
    return o ? o->object : nil;
}

/* The object's guest memory as a buffer, once it has some. */
static id<MTLBuffer> guest_copy(HvReplay *r, uint32_t task, HvObject *o) {
    if (!o->vpage || !o->length) return nil;
    if (!o->mapped && !map_pages(r, task, o)) return nil;
    refresh_pages(r, task, o);
    if (!o->guest_copy) o->guest_copy = buffer_over_mapping(r, o);
    return o->guest_copy;
}

static BOOL has_guest_pages(HvObject *o) {
    uint64_t n = (o->length + 0x3fff) >> 14;
    for (uint64_t i = 0; o->pages && i < n; i++)
        if (o->pages[i]) return YES;
    return NO;
}

/* Between guest memory and the GPU's own copy, in the layout the guest was told. */
- (void)page:(HvObject *)o toGuest:(BOOL)down task:(uint32_t)task blit:(id<MTLBlitCommandEncoder> *)blit commandBuffer:(id<MTLCommandBuffer>)cb {
    id<MTLBuffer> copy = guest_copy(self, task, o);
    if (!copy || !has_guest_pages(o)) return;
    if (o->heap_whole) {
        if (!*blit) *blit = [cb blitCommandEncoder];
        NSUInteger bytes = MIN(copy.length, o->heap_whole.length);
        if (down) [*blit copyFromBuffer:o->heap_whole sourceOffset:0 toBuffer:copy destinationOffset:0 size:bytes];
        else [*blit copyFromBuffer:copy sourceOffset:0 toBuffer:o->heap_whole destinationOffset:0 size:bytes];
        return;
    }
    id<MTLTexture> tex = o->object;
    if (!tex || !o->tex_layout || tex.sampleCount > 1 || tex.isFramebufferOnly) return;
    NSUInteger faces = tex.textureType == MTLTextureTypeCube || tex.textureType == MTLTextureTypeCubeArray ? 6 : 1;
    if (!*blit) *blit = [cb blitCommandEncoder];
    for (uint32_t s = 0; s < o->tex_slices * faces; s++)
        for (uint32_t m = 0; m < o->tex_mips; m++) {
            const uint8_t *mip = o->tex_layout + m * 36;
            uint64_t off, row;
            uint32_t w, h, d;
            memcpy(&off, mip, 8), memcpy(&row, mip + 0x10, 8), memcpy(&w, mip + 0x18, 4), memcpy(&h, mip + 0x1c, 4), memcpy(&d, mip + 0x20, 4);
            uint32_t bh = o->tex_block_h ? o->tex_block_h : 1;
            uint64_t from = o->tex_base + o->tex_stride * s + off;
            uint64_t image = row * ((h + bh - 1) / bh);
            if (from + image * d > o->length) continue;
            if (down)
                [*blit copyFromTexture:tex sourceSlice:s sourceLevel:m sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(w, h, d)
                              toBuffer:copy destinationOffset:from destinationBytesPerRow:row destinationBytesPerImage:image];
            else
                [*blit copyFromBuffer:copy sourceOffset:from sourceBytesPerRow:row sourceBytesPerImage:image
                           sourceSize:MTLSizeMake(w, h, d) toTexture:tex destinationSlice:s destinationLevel:m destinationOrigin:MTLOriginMake(0, 0, 0)];
        }
}

/* The guest asks to see what the GPU made: {task, count, refs}. */
- (void)synchronize:(const uint32_t *)a length:(uint32_t)n {
    if (n < 8) return;
    uint32_t task = a[0], count = a[1];
    if (n < 8 + count * 4ull) return;
    NSMutableDictionary *objs = tasks[@(task)];
    id<MTLCommandBuffer> cb = nil;
    id<MTLBlitCommandEncoder> blit = nil;
    for (uint32_t i = 0; i < count; i++) {
        HvObject *o = objs[@(a[2 + i])];
        if (!o || !o->guest_backed || !o->host_valid) continue;
        if (!cb) cb = [queue commandBuffer];
        [self page:o toGuest:YES task:task blit:&blit commandBuffer:cb];
        o->guest_stale = NO;
    }
    [blit endEncoding];
    if (cb) {
        [self commit:cb channel:0 stamp:0 complete:NO];
        synced++;
    }
}

/* {task, count, {ref, host invalid, host valid, guest invalid, guest valid}}. */
- (void)invalidate:(const uint32_t *)a length:(uint32_t)n {
    if (n < 8 || n < 8 + a[1] * 8ull) return;
    NSMutableDictionary *objs = tasks[@(a[0])];
    const uint8_t *e = (const uint8_t *)(a + 2);
    for (uint32_t i = 0; i < a[1]; i++, e += 8) {
        HvObject *o = objs[@(*(const uint32_t *)e)];
        if (!o) continue;
        if (e[6]) o->guest_stale = YES;
        if (e[7]) o->guest_stale = NO;
        if (e[4]) o->host_valid = NO;
        if (e[5]) o->host_valid = YES;
    }
}

/* A record's bytes, as a pointer into the buffer's private copy. */
typedef struct {
    const uint8_t *p;
    uint32_t size;
} Rec;

#define U32(off) (*(const uint32_t *)(r.p + (off)))
#define U16(off) (*(const uint16_t *)(r.p + (off)))
#define U64(off) ({ uint64_t _v; memcpy(&_v, r.p + (off), 8); _v; })

- (id<MTLRenderCommandEncoder>)beginPass:(Rec)r task:(uint32_t)task commandBuffer:(id<MTLCommandBuffer>)cb {
    MTLRenderPassDescriptor *d = [MTLRenderPassDescriptor renderPassDescriptor];
#define ATTACH(a, o)                                                 \
    do {                                                             \
        a.texture = [self get:U32((o)) task:task];                   \
        a.resolveTexture = [self get:U32((o) + 4) task:task];        \
        a.level = U16((o) + 8), a.slice = U16((o) + 10);             \
        a.depthPlane = U16((o) + 12), a.resolveLevel = U16((o) + 14); \
        a.resolveSlice = U16((o) + 16), a.resolveDepthPlane = U16((o) + 18); \
        a.loadAction = (MTLLoadAction)U16((o) + 20);                 \
        a.storeAction = (MTLStoreAction)U16((o) + 22);               \
        a.storeActionOptions = U16((o) + 24);                        \
    } while (0)
    ATTACH(d.depthAttachment, 0);
    double clear_depth;
    memcpy(&clear_depth, r.p + 0x1c, 8);
    d.depthAttachment.clearDepth = clear_depth;
    ATTACH(d.stencilAttachment, 0x28);
    d.stencilAttachment.clearStencil = U32(0x44);
    for (int i = 0; i < 8; i++) {
        uint32_t o = 0x4c + (uint32_t)i * 0x3c;
        if (o + 0x3c > r.size) break;
        MTLRenderPassColorAttachmentDescriptor *c = d.colorAttachments[i];
        ATTACH(c, o);
        double col[4];
        memcpy(col, r.p + o + 0x1c, sizeof col);
        c.clearColor = MTLClearColorMake(col[0], col[1], col[2], col[3]);
    }
    /* Memoryless and resolve store actions want the matching textures; a pass with none stays dontCare. */
    for (int i = 0; i < 8; i++) {
        MTLRenderPassColorAttachmentDescriptor *c = d.colorAttachments[i];
        if (!c.texture) c.storeAction = MTLStoreActionDontCare, c.loadAction = MTLLoadActionDontCare;
        else if (!c.resolveTexture && (c.storeAction == MTLStoreActionMultisampleResolve || c.storeAction == MTLStoreActionStoreAndMultisampleResolve))
            c.storeAction = MTLStoreActionStore;
    }
    if (!d.depthAttachment.texture) d.depthAttachment.loadAction = MTLLoadActionDontCare, d.depthAttachment.storeAction = MTLStoreActionDontCare;
    if (!d.stencilAttachment.texture) d.stencilAttachment.loadAction = MTLLoadActionDontCare, d.stencilAttachment.storeAction = MTLStoreActionDontCare;
    /* After the attachments: the visibility buffer and the render target's size. */
    if (r.size >= 0x248) {
        if (U32(0x22c)) d.visibilityResultBuffer = [self get:U32(0x22c) task:task];
        uint64_t layers = U64(0x230), rw = U64(0x238), rh = U64(0x240);
        if (layers) d.renderTargetArrayLength = (NSUInteger)layers;
        if (rw) d.renderTargetWidth = (NSUInteger)rw;
        if (rh) d.renderTargetHeight = (NSUInteger)rh;
    }
    /* What came before the pass itself. */
    if (pre_raster_samples) d.defaultRasterSampleCount = pre_raster_samples;
    if (pre_imageblock) d.imageblockSampleLength = pre_imageblock;
    if (pre_threadgroup) d.threadgroupMemoryLength = pre_threadgroup;
    if (pre_tile_w) d.tileWidth = pre_tile_w, d.tileHeight = pre_tile_h;
    pre_raster_samples = pre_imageblock = pre_threadgroup = pre_tile_w = pre_tile_h = 0;
    if (tracing) {
        id<MTLTexture> c0 = d.colorAttachments[0].texture;
        MTLClearColor cc = d.colorAttachments[0].clearColor;
        HvObject *co = [self object:U32(0x4c) task:task];
        fprintf(stderr, "trace: pass c0 ref %u %lux%lu fmt %lu load %lu store %lu clear %.2f %.2f %.2f %.2f surface %s heap %d\n", U32(0x4c), (unsigned long)c0.width, (unsigned long)c0.height, (unsigned long)c0.pixelFormat, (unsigned long)d.colorAttachments[0].loadAction, (unsigned long)d.colorAttachments[0].storeAction, cc.red, cc.green, cc.blue, cc.alpha, co && co->surface ? "yes" : "no", c0.heap != nil);
    }
    id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:d];
    if (!e) {
        NSMutableString *m = [NSMutableString stringWithString:@"no render encoder:"];
        for (int i = 0; i < 8; i++) {
            MTLRenderPassColorAttachmentDescriptor *c = d.colorAttachments[i];
            uint32_t ref = U32(0x4c + (uint32_t)i * 0x3c);
            if (ref) [m appendFormat:@" c%d ref %u %s %lux%lu fmt %lu load %lu store %lu usage %lu", i, ref, c.texture ? "ok" : "MISSING", (unsigned long)c.texture.width, (unsigned long)c.texture.height, (unsigned long)c.texture.pixelFormat, (unsigned long)c.loadAction, (unsigned long)c.storeAction, (unsigned long)c.texture.usage];
        }
        [m appendFormat:@" depth ref %u stencil ref %u", U32(0), U32(0x28)];
        [self complain:m];
    }
    return e;
}

/* One serialized call. Tags are the guest's. */
- (void)useHeaps:(const uint32_t *)refs count:(uint32_t)n task:(uint32_t)task encoder:(id)e stages:(int)stages {
    id<MTLHeap> __unsafe_unretained list[64];
    NSMutableArray *keep = [NSMutableArray new];
    uint32_t have = 0;
    for (uint32_t i = 0; i < n && have < 64; i++) {
        id heap = [self get:refs[i] task:task];
        if (!heap) continue;
        [keep addObject:heap];
        list[have++] = heap;
    }
    if (!have) return;
    if (stages >= 0) [(id<MTLRenderCommandEncoder>)e useHeaps:list count:have stages:(MTLRenderStages)stages];
    else if ([e conformsToProtocol:@protocol(MTLRenderCommandEncoder)]) [(id<MTLRenderCommandEncoder>)e useHeaps:list count:have];
    else [(id<MTLComputeCommandEncoder>)e useHeaps:list count:have];
}

- (void)render:(id<MTLRenderCommandEncoder>)e tag:(uint32_t)tag rec:(Rec)r task:(uint32_t)task {
    BOOL draw = tag <= 19 || tag == 28 || tag == 29 || tag == 155 || tag == 162;
    if (!pipeline_set && draw) return;
    if (tracing && draw) fprintf(stderr, "trace:  draw tag %u\n", tag);
    switch (tag) {
    case 24: { /* {fence, stages} */
        id f = [self get:U32(0) task:task];
        if (f) [e updateFence:f afterStages:(MTLRenderStages)(U32(4) & 0xffff)];
        break;
    }
    case 25: {
        id f = [self get:U32(0) task:task];
        if (f) [e waitForFence:f beforeStages:(MTLRenderStages)(U32(4) & 0xffff)];
        break;
    }
    case 27: /* {count, stages}, the heaps */
        if (r.size >= 8 + U32(0) * 4ull) [self useHeaps:(const uint32_t *)(r.p + 8) count:U32(0) task:task encoder:e stages:U32(4) & 0xffff];
        break;
    case 134: /* {count}, the heaps */
        if (r.size >= 4 + U32(0) * 4ull) [self useHeaps:(const uint32_t *)(r.p + 4) count:U32(0) task:task encoder:e stages:-1];
        break;
    case 116: { /* Metal crashes on a missing pipeline, and on drawing without one */
        id state = [self get:U32(0) task:task];
        if (state) [e setRenderPipelineState:state];
        pipeline_set = state != nil;
        if (tracing) fprintf(stderr, "trace:  pipeline %u %s\n", U32(0), [objc_getAssociatedObject(state, "hvname") UTF8String]);
        break;
    }
    case 125: case 110: case 157: { /* vertex, fragment, tile buffers: {index, count}, {ref, offset}... */
        uint32_t index = U32(0), count = U32(4);
        if (count > 31) break;
        __unsafe_unretained id<MTLBuffer> buffers[31];
        id strong[31];
        NSUInteger offsets[31];
        for (uint32_t i = 0; i < count; i++) {
            strong[i] = [self get:U32(8 + i * 12) task:task];
            buffers[i] = strong[i];
            offsets[i] = (NSUInteger)U64(12 + i * 12);
        }
        NSRange range = NSMakeRange(index, count);
        if (tracing) for (uint32_t i = 0; i < count; i++) { /* m4 fork: what the shaders are given */
            id<MTLBuffer> bb = strong[i];
            fprintf(stderr, "trace:  %s buf %u ref %u off %lu len %lu:", tag == 125 ? "vtx" : tag == 110 ? "frag" : "tile", index + i, U32(8 + i * 12),
                    (unsigned long)offsets[i], (unsigned long)bb.length);
            if (bb && bb.storageMode != MTLStorageModePrivate && offsets[i] < bb.length) {
                const float *f = (const float *)((const uint8_t *)bb.contents + offsets[i]);
                NSUInteger nf = MIN((bb.length - offsets[i]) / 4, (NSUInteger)96);
                for (NSUInteger k = 0; k < nf; k++) fprintf(stderr, " %g", f[k]);
            }
            fprintf(stderr, "\n");
        }
        if (tag == 125) [e setVertexBuffers:buffers offsets:offsets withRange:range];
        else if (tag == 110) [e setFragmentBuffers:buffers offsets:offsets withRange:range];
        else [e setTileBuffers:buffers offsets:offsets withRange:range];
        break;
    }
    case 126: [e setVertexBufferOffset:(NSUInteger)U64(4) atIndex:U32(0)]; break;
    case 111: [e setFragmentBufferOffset:(NSUInteger)U64(4) atIndex:U32(0)]; break;
    case 114: case 161: { /* fragment, tile textures */
        uint32_t index = U32(0), count = U32(4);
        if (count > 128) break;
        __unsafe_unretained id<MTLTexture> items[128];
        id strong[128];
        for (uint32_t i = 0; i < count; i++) {
            strong[i] = [self get:U32(8 + i * 4) task:task];
            items[i] = strong[i];
        }
        if (tracing) for (uint32_t i = 0; i < count; i++) {
            id<MTLTexture> x = strong[i];
            HvObject *xo = [self object:U32(8 + i * 4) task:task];
            fprintf(stderr, "trace:  tex %u ref %u %lux%lu fmt %lu type %u swz %d%d%d%d surface %s heap %d valid %d\n", index + i, U32(8 + i * 4), (unsigned long)x.width, (unsigned long)x.height, (unsigned long)x.pixelFormat, xo ? xo->entry[0] & 0xff : 0, x.swizzle.red, x.swizzle.green, x.swizzle.blue, x.swizzle.alpha, xo && xo->surface ? "yes" : "no", x.heap != nil, xo ? xo->host_valid : 0);
        }
        if (tag == 114) [e setFragmentTextures:items withRange:NSMakeRange(index, count)];
        else [e setTileTextures:items withRange:NSMakeRange(index, count)];
        break;
    }
    case 112: { /* fragment samplers */
        uint32_t index = U32(0), count = U32(4);
        if (count > 16) break;
        __unsafe_unretained id<MTLSamplerState> items[16];
        id strong[16];
        for (uint32_t i = 0; i < count; i++) {
            strong[i] = [self get:U32(8 + i * 4) task:task];
            items[i] = strong[i];
        }
        [e setFragmentSamplerStates:items withRange:NSMakeRange(index, count)];
        break;
    }
    case 117: {
        MTLScissorRect s;
        memcpy(&s, r.p, sizeof s);
        [e setScissorRect:s];
        if (tracing) fprintf(stderr, "trace:  scissor %lu %lu %lu %lu\n", (unsigned long)s.x, (unsigned long)s.y, (unsigned long)s.width, (unsigned long)s.height);
        break;
    }
    case 130: {
        MTLViewport v;
        memcpy(&v, r.p, sizeof v);
        [e setViewport:v];
        break;
    }
    case 122: { /* {ref, offset, stride} */
        id<MTLBuffer> b = [self get:U32(0) task:task];
        [e setTessellationFactorBuffer:b offset:(NSUInteger)U64(4) instanceStride:(NSUInteger)U64(12)];
        tess_factors = b != nil;
        break;
    }
    case 123: {
        float f;
        memcpy(&f, r.p, 4);
        [e setTessellationFactorScale:f];
        break;
    }
    case 104: {
        id state = [self get:U32(0) task:task];
        [e setDepthStencilState:state];
        break;
    }
    case 107: [e setCullMode:(MTLCullMode)U32(0)]; break;
    case 108: {
        float f[3];
        memcpy(f, r.p, sizeof f);
        [e setDepthBias:f[0] slopeScale:f[1] clamp:f[2]];
        break;
    }
    case 109: [e setDepthClipMode:(MTLDepthClipMode)U32(0)]; break;
    case 115: [e setFrontFacingWinding:(MTLWinding)U32(0)]; break;
    case 124: [e setTriangleFillMode:(MTLTriangleFillMode)U32(0)]; break;
    case 119: [e setStencilFrontReferenceValue:U32(0) backReferenceValue:U32(4)]; break;
    case 118: { /* a header, then the rectangles */
        uint32_t head = r.size % sizeof(MTLScissorRect), n = (r.size - head) / sizeof(MTLScissorRect);
        if (n) [e setScissorRects:(const MTLScissorRect *)(r.p + head) count:n];
        break;
    }
    case 131: {
        uint32_t head = r.size % sizeof(MTLViewport), n = (r.size - head) / sizeof(MTLViewport);
        if (n) [e setViewports:(const MTLViewport *)(r.p + head) count:n];
        break;
    }
    case 129: { /* vertex textures: {index, count}, refs */
        uint32_t index = U32(0), count = U32(4);
        if (count > 128) break;
        __unsafe_unretained id<MTLTexture> items[128];
        id strong[128];
        for (uint32_t i = 0; i < count; i++) {
            strong[i] = [self get:U32(8 + i * 4) task:task];
            items[i] = strong[i];
        }
        [e setVertexTextures:items withRange:NSMakeRange(index, count)];
        break;
    }
    case 127: { /* vertex samplers */
        uint32_t index = U32(0), count = U32(4);
        if (count > 16) break;
        __unsafe_unretained id<MTLSamplerState> items[16];
        id strong[16];
        for (uint32_t i = 0; i < count; i++) {
            strong[i] = [self get:U32(8 + i * 4) task:task];
            items[i] = strong[i];
        }
        [e setVertexSamplerStates:items withRange:NSMakeRange(index, count)];
        break;
    }
    case 113: case 128: { /* samplers with LOD clamps: {index, count}, refs, minimums, maximums */
        uint32_t index = U32(0), count = U32(4);
        if (count > 16 || r.size < 8 + count * 12ull) break;
        __unsafe_unretained id<MTLSamplerState> items[16];
        id strong[16];
        float lo[16], hi[16];
        for (uint32_t i = 0; i < count; i++) {
            strong[i] = [self get:U32(8 + i * 4) task:task];
            items[i] = strong[i];
            memcpy(&lo[i], r.p + 8 + count * 4 + i * 4, 4);
            memcpy(&hi[i], r.p + 8 + count * 8 + i * 4, 4);
        }
        if (tag == 113) [e setFragmentSamplerStates:items lodMinClamps:lo lodMaxClamps:hi withRange:NSMakeRange(index, count)];
        else [e setVertexSamplerStates:items lodMinClamps:lo lodMaxClamps:hi withRange:NSMakeRange(index, count)];
        break;
    }
    case 135: case 137: { /* {count, usage}, or {count, usage 16, stages 16}, then the resources */
        uint32_t count = U32(0);
        if (count > 256 || r.size < 8 + count * 4ull) break;
        __unsafe_unretained id<MTLResource> items[256];
        NSMutableArray *keep = [NSMutableArray new];
        uint32_t have = 0;
        for (uint32_t i = 0; i < count; i++) {
            id x = [self get:U32(8 + i * 4) task:task];
            if (!x || [x conformsToProtocol:@protocol(MTLHeap)]) continue;
            [keep addObject:x];
            items[have++] = x;
        }
        if (!have) break;
        if (tag == 135) [e useResources:items count:have usage:(MTLResourceUsage)U32(4)];
        else [e useResources:items count:have usage:(MTLResourceUsage)U16(4) stages:(MTLRenderStages)U16(6)];
        break;
    }
    case 165: { /* vertex buffers with stride: {index, count}, {ref, offset, stride}... */
        uint32_t index = U32(0), count = U32(4);
        if (count > 31 || r.size < 8 + count * 20ull) break;
        __unsafe_unretained id<MTLBuffer> buffers[31];
        id strong[31];
        NSUInteger offsets[31], strides[31];
        for (uint32_t i = 0; i < count; i++) {
            strong[i] = [self get:U32(8 + i * 20) task:task];
            buffers[i] = strong[i];
            offsets[i] = (NSUInteger)U64(12 + i * 20);
            strides[i] = (NSUInteger)U64(20 + i * 20);
        }
        [e setVertexBuffers:buffers offsets:offsets attributeStrides:strides withRange:NSMakeRange(index, count)];
        break;
    }
    case 166: [e setVertexBufferOffset:(NSUInteger)U64(4) attributeStride:(NSUInteger)U64(12) atIndex:U32(0)]; break;
    case 101: {
        float c[4];
        memcpy(c, r.p, sizeof c);
        [e setBlendColorRed:c[0] green:c[1] blue:c[2] alpha:c[3]];
        break;
    }
    case 13: { /* {start 16, count 16, index ref, offset 16, instances 16, base instance 16, control points 16} */
        if (!tess_factors) break; /* Metal crashes without a factor buffer */
        id<MTLBuffer> b = [self get:U32(4) task:task];
        [e drawPatches:U16(14) patchStart:U16(0) patchCount:U16(2) patchIndexBuffer:b patchIndexBufferOffset:U16(8) instanceCount:U16(10) baseInstance:U16(12)];
        draws++;
        break;
    }
    case 12: { /* {start 64, count 64, index ref, offset 64, instances 64, base instance 64, control points 16} */
        if (!tess_factors) break;
        id<MTLBuffer> b = [self get:U32(16) task:task];
        [e drawPatches:U16(44) patchStart:(NSUInteger)U64(0) patchCount:(NSUInteger)U64(8) patchIndexBuffer:b patchIndexBufferOffset:(NSUInteger)U64(20)
         instanceCount:(NSUInteger)U64(28) baseInstance:(NSUInteger)U64(36)];
        draws++;
        break;
    }
    case 14: { /* {start 64, count 64, patch index ref, offset 64, control point index ref, offset 64, instances 64, base instance 64, control points 16} */
        if (!tess_factors) break;
        id<MTLBuffer> patches = [self get:U32(16) task:task], points = [self get:U32(28) task:task];
        if (points) [e drawIndexedPatches:U16(56) patchStart:(NSUInteger)U64(0) patchCount:(NSUInteger)U64(8) patchIndexBuffer:patches patchIndexBufferOffset:(NSUInteger)U64(20)
                  controlPointIndexBuffer:points controlPointIndexBufferOffset:(NSUInteger)U64(32) instanceCount:(NSUInteger)U64(40) baseInstance:(NSUInteger)U64(48)];
        draws++;
        break;
    }
    case 15: { /* {start 16, count 16, patch index ref, offset 16, control point index ref, offset 16, instances 16, base instance 16, control points 16} */
        if (!tess_factors) break;
        id<MTLBuffer> patches = [self get:U32(4) task:task], points = [self get:U32(10) task:task];
        if (points) [e drawIndexedPatches:U16(20) patchStart:U16(0) patchCount:U16(2) patchIndexBuffer:patches patchIndexBufferOffset:U16(8)
                  controlPointIndexBuffer:points controlPointIndexBufferOffset:U16(14) instanceCount:U16(16) baseInstance:U16(18)];
        draws++;
        break;
    }
    case 18: { /* {patch index ref, indirect ref, patch index offset 64, indirect offset 64, control points 16} */
        if (!tess_factors) break;
        id<MTLBuffer> patches = [self get:U32(0) task:task], indirect = [self get:U32(4) task:task];
        if (indirect) [e drawPatches:U16(24) patchIndexBuffer:patches patchIndexBufferOffset:(NSUInteger)U64(8) indirectBuffer:indirect indirectBufferOffset:(NSUInteger)U64(16)];
        draws++;
        break;
    }
    case 19: { /* {patch index ref, control point index ref, indirect ref, then their three offsets, 64, and control points 16} */
        if (!tess_factors) break;
        id<MTLBuffer> patches = [self get:U32(0) task:task], points = [self get:U32(4) task:task], indirect = [self get:U32(8) task:task];
        if (points && indirect) [e drawIndexedPatches:U16(36) patchIndexBuffer:patches patchIndexBufferOffset:(NSUInteger)U64(12) controlPointIndexBuffer:points
                                controlPointIndexBufferOffset:(NSUInteger)U64(20) indirectBuffer:indirect indirectBufferOffset:(NSUInteger)U64(28)];
        draws++;
        break;
    }
    case 1: /* {primitive, start 16, count 16} */
        [e drawPrimitives:(MTLPrimitiveType)U32(0) vertexStart:U16(4) vertexCount:U16(6)];
        draws++;
        break;
    case 3: /* {start, count, instances, primitive} */
        [e drawPrimitives:(MTLPrimitiveType)U16(6) vertexStart:U16(0) vertexCount:U16(2) instanceCount:U16(4)];
        draws++;
        break;
    case 7: { /* {primitive, index type, ref, count, offset} */
        id<MTLBuffer> b = [self get:U32(4) task:task];
        if (b) [e drawIndexedPrimitives:(MTLPrimitiveType)U16(0) indexCount:U16(8) indexType:(MTLIndexType)U16(2) indexBuffer:b indexBufferOffset:U16(10)];
        draws++;
        break;
    }
    case 6: {
        id<MTLBuffer> b = [self get:U32(4) task:task];
        if (b) [e drawIndexedPrimitives:(MTLPrimitiveType)U16(0) indexCount:(NSUInteger)U64(8) indexType:(MTLIndexType)U16(2) indexBuffer:b indexBufferOffset:(NSUInteger)U64(16)];
        draws++;
        break;
    }
    case 9: { /* {primitive, index type, ref, count 16, offset 16, instances 16} */
        id<MTLBuffer> b = [self get:U32(4) task:task];
        if (b) [e drawIndexedPrimitives:(MTLPrimitiveType)U16(0) indexCount:U16(8) indexType:(MTLIndexType)U16(2) indexBuffer:b indexBufferOffset:U16(10) instanceCount:U16(12)];
        draws++;
        break;
    }
    case 8: { /* {primitive, index type, ref, count 64, offset 64, instances 64} */
        id<MTLBuffer> b = [self get:U32(4) task:task];
        if (b) [e drawIndexedPrimitives:(MTLPrimitiveType)U16(0) indexCount:(NSUInteger)U64(8) indexType:(MTLIndexType)U16(2) indexBuffer:b indexBufferOffset:(NSUInteger)U64(16) instanceCount:(NSUInteger)U64(24)];
        draws++;
        break;
    }
    case 0: /* {primitive 32, start 64, count 64} */
        [e drawPrimitives:(MTLPrimitiveType)U32(0) vertexStart:(NSUInteger)U64(4) vertexCount:(NSUInteger)U64(12)];
        draws++;
        break;
    case 2: /* {start 64, count 64, instances 64, primitive} */
        [e drawPrimitives:(MTLPrimitiveType)U16(24) vertexStart:(NSUInteger)U64(0) vertexCount:(NSUInteger)U64(8) instanceCount:(NSUInteger)U64(16)];
        draws++;
        break;
    case 4: /* {start 64, count 64, instances 64, base instance 64, primitive} */
        [e drawPrimitives:(MTLPrimitiveType)U16(32) vertexStart:(NSUInteger)U64(0) vertexCount:(NSUInteger)U64(8) instanceCount:(NSUInteger)U64(16) baseInstance:(NSUInteger)U64(24)];
        draws++;
        break;
    case 5: /* {start, count, instances, base instance, primitive}, all 16 */
        [e drawPrimitives:(MTLPrimitiveType)U16(8) vertexStart:U16(0) vertexCount:U16(2) instanceCount:U16(4) baseInstance:U16(6)];
        draws++;
        break;
    case 10: case 28: { /* {primitive, index type, ref, offset 64, count 64, instances 64, base vertex 64, base instance 64} */
        id<MTLBuffer> b = [self get:U32(4) task:task];
        if (b) [e drawIndexedPrimitives:(MTLPrimitiveType)U16(0) indexCount:(NSUInteger)U64(16) indexType:(MTLIndexType)U16(2) indexBuffer:b indexBufferOffset:(NSUInteger)U64(8)
                          instanceCount:(NSUInteger)U64(24) baseVertex:(NSInteger)U64(32) baseInstance:(NSUInteger)U64(40)];
        draws++;
        break;
    }
    case 11: case 29: { /* {primitive, index type, ref, offset 16, count 16, instances 16, base vertex 16, base instance 16}; 29's base vertex can be negative */
        id<MTLBuffer> b = [self get:U32(4) task:task];
        NSInteger base_vertex = tag == 29 ? (NSInteger)(int16_t)U16(14) : (NSInteger)U16(14);
        if (b) [e drawIndexedPrimitives:(MTLPrimitiveType)U16(0) indexCount:U16(10) indexType:(MTLIndexType)U16(2) indexBuffer:b indexBufferOffset:U16(8)
                          instanceCount:U16(12) baseVertex:base_vertex baseInstance:U16(16)];
        draws++;
        break;
    }
    case 16: { /* {ref, primitive 16, 2 unused, offset 64} */
        id<MTLBuffer> b = [self get:U32(0) task:task];
        if (b) [e drawPrimitives:(MTLPrimitiveType)U16(4) indirectBuffer:b indirectBufferOffset:(NSUInteger)U64(8)];
        draws++;
        break;
    }
    case 17: { /* {primitive, index type, index ref, indirect ref, index offset 64, indirect offset 64} */
        id<MTLBuffer> index = [self get:U32(4) task:task], indirect = [self get:U32(8) task:task];
        if (index && indirect) [e drawIndexedPrimitives:(MTLPrimitiveType)U16(0) indexType:(MTLIndexType)U16(2) indexBuffer:index indexBufferOffset:(NSUInteger)U64(12)
                                          indirectBuffer:indirect indirectBufferOffset:(NSUInteger)U64(20)];
        draws++;
        break;
    }
    case 154: { /* {count, then a viewport and a render target offset for each view} */
        uint32_t count = U32(0);
        MTLVertexAmplificationViewMapping views[8];
        if (count == 0 || count > 8 || r.size < 4 + count * 8ull || ![e.device supportsVertexAmplificationCount:count]) break;
        for (uint32_t i = 0; i < count; i++) {
            views[i].viewportArrayIndexOffset = U32(4 + i * 8);
            views[i].renderTargetArrayIndexOffset = U32(8 + i * 8);
        }
        [e setVertexAmplificationCount:count viewMappings:views];
        break;
    }
    case 155: {
        MTLSize s;
        memcpy(&s, r.p, sizeof s);
        [e dispatchThreadsPerTile:s];
        break;
    }
    case 162: { /* threads per tile, then a tile region */
        MTLSize s;
        MTLRegion region;
        memcpy(&s, r.p, sizeof s);
        memcpy(&region, r.p + sizeof s, sizeof region);
        SEL sel = sel_registerName("dispatchThreadsPerTile:inRegion:");
        if ([(id)e respondsToSelector:sel]) ((void (*)(id, SEL, MTLSize, MTLRegion))objc_msgSend)(e, sel, s, region);
        else [e dispatchThreadsPerTile:s];
        break;
    }
    case 164: { /* where to write the tile size: {buffer ref, offset} */
        HvObject *o = [self object:U32(0) task:task];
        uint64_t off = U64(4);
        if (o && o->mapped && off + 4 <= o->length) {
            uint16_t dims[2] = {(uint16_t)e.tileWidth, (uint16_t)e.tileHeight};
            memcpy(o->mapped + off, dims, 4);
        }
        break;
    }
    default:
        [self complain:[NSString stringWithFormat:@"render command %u (%u bytes) skipped", tag, r.size]];
        break;
    }
}

/* Compute calls. Tags start at 200. */
- (void)compute:(id<MTLComputeCommandEncoder>)e tag:(uint32_t)tag rec:(Rec)r task:(uint32_t)task {
    if (!pipeline_set && (tag == 200 || tag == 202)) return;
    switch (tag) {
    case 209: {
        MTLRegion region;
        if (r.size >= sizeof region) {
            memcpy(&region, r.p, sizeof region);
            [e setStageInRegion:region];
        }
        break;
    }
    case 214: /* barriers: encoders here are in order, and each exec is waited for */
    case 215:
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers | MTLBarrierScopeTextures];
        break;
    case 212: {
        id f = [self get:U32(0) task:task];
        if (f) [e updateFence:f];
        break;
    }
    case 213: {
        id f = [self get:U32(0) task:task];
        if (f) [e waitForFence:f];
        break;
    }
    case 208: {
        id state = [self get:U32(0) task:task];
        if (state) [e setComputePipelineState:state];
        pipeline_set = state != nil;
        break;
    }
    case 203: {
        uint32_t index = U32(0), count = U32(4);
        if (count > 31) break;
        __unsafe_unretained id<MTLBuffer> buffers[31];
        id strong[31];
        NSUInteger offsets[31];
        for (uint32_t i = 0; i < count; i++) {
            strong[i] = [self get:U32(8 + i * 12) task:task];
            buffers[i] = strong[i];
            offsets[i] = (NSUInteger)U64(12 + i * 12);
        }
        [e setBuffers:buffers offsets:offsets withRange:NSMakeRange(index, count)];
        break;
    }
    case 207: [e setBufferOffset:(NSUInteger)U64(4) atIndex:U32(0)]; break;
    case 206: {
        uint32_t index = U32(0), count = U32(4);
        if (count > 128) break;
        __unsafe_unretained id<MTLTexture> items[128];
        id strong[128];
        for (uint32_t i = 0; i < count; i++) {
            strong[i] = [self get:U32(8 + i * 4) task:task];
            items[i] = strong[i];
        }
        [e setTextures:items withRange:NSMakeRange(index, count)];
        break;
    }
    case 204: {
        uint32_t index = U32(0), count = U32(4);
        if (count > 16) break;
        __unsafe_unretained id<MTLSamplerState> items[16];
        id strong[16];
        for (uint32_t i = 0; i < count; i++) {
            strong[i] = [self get:U32(8 + i * 4) task:task];
            items[i] = strong[i];
        }
        [e setSamplerStates:items withRange:NSMakeRange(index, count)];
        break;
    }
    case 211: [e setThreadgroupMemoryLength:(NSUInteger)U64(0) atIndex:U32(8)]; break;
    case 200: case 202: {
        MTLSize a, b;
        memcpy(&a, r.p, sizeof a), memcpy(&b, r.p + sizeof a, sizeof b);
        if (tag == 200) [e dispatchThreadgroups:a threadsPerThreadgroup:b];
        else [e dispatchThreads:a threadsPerThreadgroup:b];
        break;
    }
    case 216: /* {width, height} of the imageblock, 32x32 from macOS 27's glass and blur kernels: without it their
                 imageblock memory is undefined, and toolbars came out black with coloured blocks */
        if (r.size >= 8 && U32(0) && U32(4) && U32(0) <= 256 && U32(4) <= 256) [e setImageblockWidth:U32(0) height:U32(4)];
        break;
    case 219: /* not a command the framework knows either, though the guest sends it */
        break;
    default:
        [self complain:[NSString stringWithFormat:@"compute command %u (%u bytes) skipped", tag, r.size]];
        break;
    }
}

/* Blit calls. Tags start at 300. */
- (void)blit:(id<MTLBlitCommandEncoder>)e tag:(uint32_t)tag rec:(Rec)r task:(uint32_t)task {
    switch (tag) {
    case 316: {
        id f = [self get:U32(0) task:task];
        if (f) [e updateFence:f];
        break;
    }
    case 317: {
        id f = [self get:U32(0) task:task];
        if (f) [e waitForFence:f];
        break;
    }
    case 300: { /* buffer to texture */
        id<MTLBuffer> b = [self get:U32(0) task:task];
        id<MTLTexture> t = [self get:U32(4) task:task];
        if (!b || !t) break;
        MTLSize size;
        MTLOrigin origin;
        memcpy(&size, r.p + 0x20, sizeof size), memcpy(&origin, r.p + 0x38, sizeof origin);
        [e copyFromBuffer:b sourceOffset:(NSUInteger)U64(8) sourceBytesPerRow:(NSUInteger)U64(0x10) sourceBytesPerImage:(NSUInteger)U64(0x18) sourceSize:size
                toTexture:t destinationSlice:U16(0x50) destinationLevel:U16(0x52) destinationOrigin:origin options:(MTLBlitOption)U32(0x54)];
        break;
    }
    case 301: { /* buffer to buffer */
        id<MTLBuffer> a = [self get:U32(0) task:task], b = [self get:U32(4) task:task];
        if (a && b) [e copyFromBuffer:a sourceOffset:(NSUInteger)U64(8) toBuffer:b destinationOffset:(NSUInteger)U64(0x10) size:(NSUInteger)U64(0x18)];
        break;
    }
    case 302: { /* texture to buffer */
        id<MTLTexture> t = [self get:U32(0) task:task];
        id<MTLBuffer> b = [self get:U32(4) task:task];
        if (!b || !t) break;
        MTLOrigin origin;
        MTLSize size;
        memcpy(&origin, r.p + 8, sizeof origin), memcpy(&size, r.p + 0x20, sizeof size);
        [e copyFromTexture:t sourceSlice:U16(0x50) sourceLevel:U16(0x52) sourceOrigin:origin sourceSize:size toBuffer:b
            destinationOffset:(NSUInteger)U64(0x38) destinationBytesPerRow:(NSUInteger)U64(0x40) destinationBytesPerImage:(NSUInteger)U64(0x48) options:(MTLBlitOption)U16(0x54)];
        break;
    }
    case 303: { /* texture to texture */
        id<MTLTexture> a = [self get:U32(0) task:task], b = [self get:U32(4) task:task];
        if (!a || !b) break;
        MTLOrigin from, to;
        MTLSize size;
        memcpy(&from, r.p + 8, sizeof from), memcpy(&size, r.p + 0x20, sizeof size), memcpy(&to, r.p + 0x38, sizeof to);
        [e copyFromTexture:a sourceSlice:U16(0x50) sourceLevel:U16(0x52) sourceOrigin:from sourceSize:size
                 toTexture:b destinationSlice:U16(0x54) destinationLevel:U16(0x56) destinationOrigin:to];
        break;
    }
    case 306: { /* fill: {ref, location, length, value} */
        id<MTLBuffer> b = [self get:U32(0) task:task];
        if (b) [e fillBuffer:b range:NSMakeRange((NSUInteger)U64(4), (NSUInteger)U64(12)) value:r.p[20]];
        break;
    }
    case 307: {
        id<MTLTexture> t = [self get:U32(0) task:task];
        if (t) [e generateMipmapsForTexture:t];
        break;
    }
    case 308: case 309: case 310: case 311: case 314: case 315: /* hints, and syncing memory Apple GPUs share */
        break;
    default:
        [self complain:[NSString stringWithFormat:@"blit command %u (%u bytes) skipped", tag, r.size]];
        break;
    }
}

/* What the guest asks about objects, answered in its own buffers: tags start at
   450, {ref, buffer ref, offset}. The guest's Metal puts these real host values
   into its argument buffers. */
- (void)info:(uint32_t)tag rec:(Rec)r task:(uint32_t)task {
    uint32_t ref = U32(0), bufref = U32(4);
    uint64_t off = U64(8);
    if (tag == 451 || tag == 469) { /* how big a texture would be in a heap: {description, buffer ref, offset} */
        uint32_t at = tag == 451 ? 0x20 : 0x28;
        if (r.size < at + 12) return;
        MTLTextureDescriptor *t = tag == 451 ? descriptor_v1(r.p) : descriptor_v2(r.p);
        HvObject *to = [self object:U32(at) task:task];
        refresh_pages(self, task, to);
        uint64_t where = U64(at + 4);
        if (!to || !to->mapped || where + 16 > to->length) { [self complain:@"heap size query: no buffer"]; return; }
        MTLSizeAndAlign sa = [device heapTextureSizeAndAlignWithDescriptor:t];
        uint64_t v[2] = {sa.size, sa.align};
        memcpy(to->mapped + where, v, 16);
        return;
    }
    if (tag == 460) { /* an object's identifier: {ref, kind, buffer ref, offset} */
        bufref = U32(8);
        off = U64(12);
    }
    HvObject *to = [self object:bufref task:task];
    if (to) refresh_pages(self, task, to);
    uint8_t *out = to && to->mapped && off + 28 <= to->length ? to->mapped + off : NULL;
    id obj = [self get:ref task:task];
    if (!out || !obj) {
        [self complain:[NSString stringWithFormat:@"info %u: ref %u buffer %u (%s, %llu bytes) offset %llu: %s", tag, ref, bufref,
                        !to ? "unknown" : to->mapped ? "mapped" : "not mapped", to ? to->length : 0, off, out ? "no object" : "no buffer"]];
        return;
    }
    SEL index_sel = sel_registerName("resourceIndex");
    uint64_t index = [obj respondsToSelector:index_sel] ? ((uint64_t (*)(id, SEL))objc_msgSend)(obj, index_sel) : 0;
    switch (tag) {
    case 460: {
        SEL sel = sel_registerName("uniqueIdentifier");
        uint64_t v = [obj respondsToSelector:sel] ? ((uint64_t (*)(id, SEL))objc_msgSend)(obj, sel) : (uint64_t)(uintptr_t)obj;
        memcpy(out, &v, 8);
        break;
    }
    case 457: { /* render pipeline: {max threads, imageblock length, tile match, indirect} */
        id<MTLRenderPipelineState> p = obj;
        uint32_t v[2] = {(uint32_t)p.maxTotalThreadsPerThreadgroup, (uint32_t)p.imageblockSampleLength};
        memcpy(out, v, 8);
        out[8] = p.threadgroupSizeMatchesTileSize, out[9] = p.supportIndirectCommandBuffers;
        break;
    }
    case 450: { /* compute pipeline */
        id<MTLComputePipelineState> p = obj;
        uint64_t v[3] = {p.maxTotalThreadsPerThreadgroup, p.threadExecutionWidth, p.staticThreadgroupMemoryLength};
        memcpy(out, v, 24);
        out[24] = p.supportIndirectCommandBuffers;
        break;
    }
    case 463: { /* heap: where the whole of it is */
        id<MTLHeap> h = obj;
        HvObject *ho = [self object:ref task:task];
        uint64_t a = ho && ho->heap_whole ? ho->heap_whole.gpuAddress : 0;
        (void)h;
        memcpy(out, &a, 8);
        break;
    }
    case 461: { /* buffer */
        uint64_t a = [(id<MTLBuffer>)obj gpuAddress];
        memcpy(out, &a, 8);
        break;
    }
    case 462: case 464: case 466: case 467: case 468: { /* texture, sampler, render, compute, depth-stencil */
        MTLResourceID id_ = ((MTLResourceID (*)(id, SEL))objc_msgSend)(obj, @selector(gpuResourceID));
        memcpy(out, &id_, 8);
        memcpy(out + 8, &index, 8);
        break;
    }
    default:
        [self complain:[NSString stringWithFormat:@"info command %u (%u bytes) skipped", tag, r.size]];
        break;
    }
}

/* One exec: {task, resources, buffers}, {ref, flags} per resource, {offset, length} per buffer. */
/* Commits without waiting: the next batch can be encoded while the GPU does this one. */
- (void)commit:(id<MTLCommandBuffer>)cb channel:(uint32_t)ch stamp:(uint32_t)stamp complete:(BOOL)complete {
    static int sync = -1; /* HV_PVG_SYNC: wait for the GPU after every batch */
    if (sync < 0) sync = getenv("HV_PVG_SYNC") != NULL;
    HvPvgLite *l = lite;
    atomic_fetch_add(&inflight, 1);
    [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
        if (done.error) {
            self->failed++;
            fprintf(stderr, "hengevm: pvg-metal: command buffer: %s\n", done.error.localizedDescription.UTF8String);
        }
        if (complete) hv_pvg_lite_complete(l, ch, stamp);
        atomic_fetch_sub(&self->inflight, 1);
    }];
    [cb commit];
    last_cb = cb;
    if (sync) [cb waitUntilCompleted];
}

/* A command that used the GPU not at all, or waited for by others: its stamp after everything before it. */
- (void)finishChannel:(uint32_t)ch stamp:(uint32_t)stamp {
    if (atomic_load(&inflight) == 0) {
        hv_pvg_lite_complete(lite, ch, stamp);
        return;
    }
    [self commit:[queue commandBuffer] channel:ch stamp:stamp complete:YES];
}

/* Returns YES if the GPU's completion gives the stamp back. */
- (BOOL)exec:(const uint32_t *)a length:(uint32_t)len channel:(uint32_t)ch stamp:(uint32_t)stamp {
    uint32_t task = a[0], nres = a[1], nbuf = a[2];
    if (12 + nres * 8ull + nbuf * 16ull > len) { /* never quietly: a lost job is a frame that never gets drawn */
        [self complain:[NSString stringWithFormat:@"exec of %u bytes dropped: task %u, %u resources, %u buffers", len, task, nres, nbuf]];
        return NO;
    }
    const uint32_t *res = a + 3;
    const uint8_t *bufs = (const uint8_t *)(res + nres * 2);
    static const char *trace_file;
    static BOOL trace_checked;
    if (!trace_checked) trace_checked = YES, trace_file = getenv("HV_PVG_TRACE_FILE");
    tracing = trace_file && access(trace_file, F_OK) == 0;
    if (tracing) fprintf(stderr, "trace: exec task %u, %u resources, %u buffers\n", task, nres, nbuf);
    double t_begin = now_seconds();
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = nil;
    BOOL sync_after = NO;
    for (uint32_t i = 0; i < nres; i++) {
        HvObject *o = [self object:res[i * 2] task:task];
        if (!o) continue;
        if (o->mapped) refresh_pages(self, task, o);
        if (o->surface) sync_iosurface(o->surface);
        if (getenv("HV_PVG_TALLY") && (o->entry[0] & 0xff) == OBJ_MAPPED_TEXTURE) {
            static uint32_t seen[256];
            uint32_t f = res[i * 2 + 1] & 0xff, key = res[i * 2] & 0xff;
            if (seen[key] != f + 1) {
                seen[key] = f + 1;
                fprintf(stderr, "pvgres task %u ref %u mapped flags %x\n", task, res[i * 2], res[i * 2 + 1]);
            }
        }
        if (res[i * 2 + 1] & 1) o->guest_stale = NO, o->host_valid = NO; /* the guest wrote it */
        if (o->guest_backed && !o->host_valid) {
            if (!o->guest_stale) [self page:o toGuest:NO task:task blit:&blit commandBuffer:cb];
            o->host_valid = YES;
        }
        if (res[i * 2 + 1] & 4) sync_after = YES;
    }
    [blit endEncoding];
    /* Bit 1: this exec writes it, so the guest's copy is out of date from here on. Never
       upload that over what the GPU drew: mostly zeros, which show as black boxes. */
    for (uint32_t i = 0; i < nres; i++)
        if (res[i * 2 + 1] & 2) { /* and what it is made from: a texture in a heap is the heap's bytes */
            for (HvObject *o = [self object:res[i * 2] task:task]; o; o = o->parent) o->guest_stale = YES;
        }
    id e = nil; /* the open encoder */
    uint32_t etype = 0;
    for (uint32_t b = 0; b < nbuf; b++) {
        uint64_t off, blen;
        memcpy(&off, bufs + b * 16, 8);
        memcpy(&blen, bufs + b * 16 + 8, 8);
        if (blen > (64u << 20) || blen < 8) continue;
        NSMutableData *copy = [NSMutableData dataWithLength:(NSUInteger)blen];
        if (hv_pvg_lite_task_read(lite, task, off, copy.mutableBytes, (uint32_t)blen)) { [self complain:@"command buffer unreadable"]; continue; }
        const uint8_t *p = copy.bytes, *end = p + blen;
        uint32_t seglen, info;
        memcpy(&seglen, p, 4), memcpy(&info, p + 4, 4);
        p += 8;
        if ((info & 0xff) == 5) { /* protection options, then the real header */
            p += 8;
            if (p + 8 > end) continue;
            memcpy(&seglen, p, 4), memcpy(&info, p + 4, 4);
            p += 8;
        }
        uint32_t type = info & 0xff, cont = (info >> 8) & 0xff, more = (info >> 16) & 0xff;
        if (p + seglen < end) end = p + seglen;
        if (!cont && e) { [e endEncoding]; e = nil; }
        if (type > 2) {
            const uint8_t *q = p;
            while (q + 8 <= end) {
                uint32_t tag = *(const uint32_t *)q, sz = *(const uint32_t *)(q + 4);
                if (sz < 8 || q + sz > end) break;
                Rec r = {q + 8, sz - 8};
                if (type == 4) [self info:tag rec:r task:task];
                else [self complain:[NSString stringWithFormat:@"encoder type %u command %u (%u bytes) skipped", type, tag, sz]];
                q += sz;
            }
            continue;
        }
        if (!e && type == 1) e = [cb computeCommandEncoder], pipeline_set = NO;
        if (!e && type == 2) e = [cb blitCommandEncoder];
        etype = type;
        @try {
            while (p + 8 <= end) {
                uint32_t tag, size;
                memcpy(&tag, p, 4), memcpy(&size, p + 4, 4);
                if (size < 8 || p + size > end) break;
                Rec r = {p + 8, size - 8};
                if (etype == 0 && tag >= 30 && tag <= 36) { /* these describe the pass that follows */
                    if (tag == 30 && size >= 12) pre_raster_samples = U32(0);
                    else if (tag == 34 && size >= 12) pre_imageblock = U32(0);
                    else if (tag == 35 && size >= 12) pre_threadgroup = U32(0);
                    else if (tag == 36 && size >= 12) pre_tile_w = U32(0) & 0xffff, pre_tile_h = U32(0) >> 16;
                    else if (tag == 33 || tag == 32) [self complain:[NSString stringWithFormat:@"pass description %u (%u bytes) not done", tag, size]];
                } else if (etype == 0 && tag == 26) {
                    if (e) [e endEncoding];
                    e = [self beginPass:r task:task commandBuffer:cb];
                    pipeline_set = NO, tess_factors = NO;
                } else if (e) {
                    if (etype == 0) [self render:e tag:tag rec:r task:task];
                    else if (etype == 1) [self compute:e tag:tag rec:r task:task];
                    else [self blit:e tag:tag rec:r task:task];
                }
                p += size;
            }
        } @catch (NSException *x) {
            [self complain:[NSString stringWithFormat:@"exception: %@", x.reason]];
            failed++;
        }
        if (!more && e) { [e endEncoding]; e = nil; }
    }
    if (e) [e endEncoding];
    if (sync_after) { /* the guest wants these back as soon as this is done */
        id<MTLBlitCommandEncoder> back = nil;
        for (uint32_t i = 0; i < nres; i++) {
            if (!(res[i * 2 + 1] & 4)) continue;
            HvObject *o = [self object:res[i * 2] task:task];
            if (!o || !o->guest_backed) continue;
            [self page:o toGuest:YES task:task blit:&back commandBuffer:cb];
            o->guest_stale = NO;
        }
        [back endEncoding];
    }
    double t1 = now_seconds();
    [self commit:cb channel:ch stamp:stamp complete:YES];
    double t2 = now_seconds();
    encode_time += t1 - t_begin, gpu_wait_time += t2 - t1;
    if (++execs % 2000 == 0) {
        static double t0;
        if (!t0) t0 = now_seconds();
        fprintf(stderr, "hengevm: pvg-metal: %llu execs, %llu draws, %llu failed, %llu synced, %llu remaps, %.0f s; encoding %.2f s, waiting for the GPU %.2f s\n", (unsigned long long)execs, (unsigned long long)draws, (unsigned long long)failed, (unsigned long long)synced, (unsigned long long)remaps, now_seconds() - t0, encode_time, gpu_wait_time);
        encode_time = gpu_wait_time = 0;
    }
    return YES;
}

/* Every possible half-float value to a display byte, as the screen would show it in the current mode (see hv_half_mode). */
static const uint8_t *half_to_byte(void) {
    static uint8_t *tables[2];
    int mode = hv_half_mode_get();
    if (tables[mode]) return tables[mode];
    uint8_t *t = malloc(65536);
    for (uint32_t i = 0; i < 65536; i++) {
        uint16_t bits = (uint16_t)i;
        __fp16 h;
        memcpy(&h, &bits, 2);
        float v = h;
        t[i] = !(v > 0) ? 0 : v >= 1 ? 255 : (uint8_t)(hv_half_shown(v, mode) * 255.0f + 0.5f);
    }
    tables[mode] = t;
    return t;
}

/* Op 4: a new pointer image in the kernel task's memory, premultiplied BGRA:
   {port, 0, offset, mapped length, stride, width, height, hot x, hot y, sum}. */
- (void)cursorGlyph:(const uint32_t *)a length:(uint32_t)n {
    if (n < 0x2c) return;
    uint64_t off, stride;
    memcpy(&off, a + 2, 8), memcpy(&stride, a + 6, 8);
    uint16_t w, h, hx, hy;
    memcpy(&w, (const uint8_t *)a + 0x20, 2), memcpy(&h, (const uint8_t *)a + 0x22, 2);
    memcpy(&hx, (const uint8_t *)a + 0x24, 2), memcpy(&hy, (const uint8_t *)a + 0x26, 2);
    if (!w || !h || w > 512 || h > 512 || stride < (uint64_t)w * 4 || stride > 8192) return;
    uint8_t *bits = malloc((size_t)stride * h);
    if (hv_pvg_lite_task_read(lite, 0, off, bits, (uint32_t)stride * h)) {
        free(bits);
        [self complain:@"pointer image unreadable"];
        return;
    }
    free(cursor);
    MTLTextureDescriptor *ct = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:w height:h mipmapped:NO];
    ct.storageMode = MTLStorageModeShared;
    id<MTLTexture> glyph = [device newTextureWithDescriptor:ct];
    [glyph replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0 withBytes:bits bytesPerRow:(NSUInteger)stride];
    os_unfair_lock_lock(&screen_lock);
    cursor = bits, cursor_w = w, cursor_h = h, cursor_stride = (uint32_t)stride, cursor_hot_x = hx, cursor_hot_y = hy;
    cursor_tex = glyph;
    os_unfair_lock_unlock(&screen_lock);
    if (getenv("HV_PVG_CURSOR_LOG")) fprintf(stderr, "pvgcursor glyph %ux%u hot %d,%d stride %u first alpha %u\n", w, h, hx, hy, (uint32_t)stride, bits[(size_t)(h / 2) * stride + (size_t)(w / 2) * 4 + 3]);
    [self drawCursor];
}

- (BOOL)hasDisplayLayer {
    os_unfair_lock_lock(&screen_lock);
    BOOL attached = layer != nil;
    os_unfair_lock_unlock(&screen_lock);
    return attached;
}

/* Puts back what the pointer covered, and draws it where the shared page says it is now. */
- (void)drawCursor {
    if ([self hasDisplayLayer]) {
        [self screenChanged];
        return;
    }
    if (!frame || !base) return;
    uint32_t w = frame_w, h = frame_h;
    for (int y = drawn_y; y < drawn_y + drawn_h; y++) memcpy(frame + (size_t)y * w * 4 + (size_t)drawn_x * 4, base + (size_t)y * w * 4 + (size_t)drawn_x * 4, (size_t)drawn_w * 4);
    drawn_w = drawn_h = 0;
    const uint8_t *page = hv_pvg_lite_display_page(lite);
    if (!page || !cursor || !(cursor_shown || page[0xe04])) return;
    uint32_t where = __atomic_load_n((const uint32_t *)(page + 0xe00), __ATOMIC_ACQUIRE);
    if (getenv("HV_PVG_CURSOR_LOG")) {
        static uint32_t last;
        if (where != last) fprintf(stderr, "pvgcursor at %u,%u shown %d flag %d\n", where & 0xffff, where >> 16, cursor_shown, page[0xe04]);
        last = where;
    }
    if (where == 0xffffffff) return;
    /* Bigger than the guest draws it. The hot spot stays where it points. */
    const int scale = cursor_scale();
    int x = (int)(where & 0xffff) - cursor_hot_x * scale, y = (int)(where >> 16) - cursor_hot_y * scale;
    int cw = (int)cursor_w * scale, ch = (int)cursor_h * scale;
    int x0 = x < 0 ? 0 : x, y0 = y < 0 ? 0 : y;
    int x1 = x + cw > (int)w ? (int)w : x + cw, y1 = y + ch > (int)h ? (int)h : y + ch;
    if (x1 <= x0 || y1 <= y0) return;
    for (int py = y0; py < y1; py++) {
        const uint8_t *row = cursor + (size_t)((py - y) / scale) * cursor_stride;
        uint8_t *dst = frame + (size_t)py * w * 4 + (size_t)x0 * 4;
        for (int px = x0; px < x1; px++, dst += 4) {
            const uint8_t *src = row + (size_t)((px - x) / scale) * 4;
            uint32_t keep = 255u - src[3];
            dst[0] = (uint8_t)(src[0] + (dst[0] * keep + 127) / 255);
            dst[1] = (uint8_t)(src[1] + (dst[1] * keep + 127) / 255);
            dst[2] = (uint8_t)(src[2] + (dst[2] * keep + 127) / 255);
        }
    }
    drawn_x = x0, drawn_y = y0, drawn_w = x1 - x0, drawn_h = y1 - y0;
}

static NSString *const screen_shaders = @"#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct V { float4 at [[position]]; float2 uv; };\n"
    "vertex V screen_v(uint i [[vertex_id]], constant float4 &rect [[buffer(0)]]) {\n"
    "  float2 c = float2(i & 1, (i >> 1) & 1);\n"
    "  V o; o.at = float4(rect.xy + c * rect.zw, 0, 1); o.uv = float2(c.x, 1 - c.y); return o; }\n"
    "fragment float4 screen_plain(V v [[stage_in]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]]) {\n"
    "  return float4(t.sample(s, v.uv).rgb, 1); }\n"
    "fragment float4 screen_linear(V v [[stage_in]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]], constant uint &mode [[buffer(0)]]) {\n"
    "  float3 c = saturate(t.sample(s, v.uv).rgb);\n"
    "  if (mode == 1) c = select(1.055 * powr(c, 1.0 / 2.4) - 0.055, c * 12.92, c <= 0.0031308);\n"
    "  return float4(c, 1); }\n"
    "fragment float4 screen_cursor(V v [[stage_in]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]]) {\n"
    "  return t.sample(s, v.uv); }\n";

- (BOOL)prepareScreenForLayer:(CAMetalLayer *)target {
    if (!screen_pipe || !linear_pipe || !cursor_pipe) {
        NSError *err = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:screen_shaders options:nil error:&err];
        if (!lib) { [self complain:[NSString stringWithFormat:@"screen shaders: %@", err.localizedDescription]]; return NO; }
        MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
        d.vertexFunction = [lib newFunctionWithName:@"screen_v"];
        d.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
        d.fragmentFunction = [lib newFunctionWithName:@"screen_plain"];
        screen_pipe = [device newRenderPipelineStateWithDescriptor:d error:&err];
        d.fragmentFunction = [lib newFunctionWithName:@"screen_linear"];
        linear_pipe = [device newRenderPipelineStateWithDescriptor:d error:&err];
        d.fragmentFunction = [lib newFunctionWithName:@"screen_cursor"];
        d.colorAttachments[0].blendingEnabled = YES;
        d.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
        d.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        d.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
        d.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        cursor_pipe = [device newRenderPipelineStateWithDescriptor:d error:&err];
        MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
        sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterLinear;
        screen_sampler = [device newSamplerStateWithDescriptor:sd];
        sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterNearest;
        cursor_sampler = [device newSamplerStateWithDescriptor:sd];
        if (!screen_pipe || !linear_pipe || !cursor_pipe || !screen_sampler || !cursor_sampler) { [self complain:[NSString stringWithFormat:@"screen pipelines: %@", err.localizedDescription]]; return NO; }
    }
    // Pipelines are shared, but a newly connected monitor has a fresh layer.
    if (prepared_layer != target) {
        [CATransaction begin];
        [CATransaction setDisableActions:YES];
        target.device = device;
        target.pixelFormat = MTLPixelFormatBGRA8Unorm;
        target.framebufferOnly = !(getenv("HV_PVG_GPU_TEST") || getenv("HV_COLOR_PROBE")); /* the probe copies the drawable out */
        target.allowsNextDrawableTimeout = YES;
        target.maximumDrawableCount = 3;
        [CATransaction commit];
        prepared_layer = target;
        fprintf(stderr, "hengevm: pvg-metal: half-float display surfaces are shown %s (HV_PVG_HALF=%s)\n",
                hv_half_mode_get() == 1 ? "as LINEAR light with an sRGB encode" : "as display-encoded values, passed through", hv_half_mode_get() == 1 ? "linear" : "encoded/default");
        if (cp.on) cp_log_layer(self, target);
    }
    return YES;
}

/* The display, drawn by the GPU. Waiting for a drawable can take until the screen's next
   refresh, so that happens on a thread of its own and never holds up the guest's GPU work:
   this only says what to show next. Whatever is newest when a drawable comes is what is shown. */
- (void)kickPresenter:(BOOL)fresh {
    /* screen_lock held */
    if (fresh) screen_requests++;
    if (presenter_running || !layer || !screen_tex || drawable_retry_pending) return;
    if (drawable_retry_cooldown) {
        double delay = drawable_retry_cooldown - now_seconds();
        if (delay > 0) {
            if (!fresh) return;
            // Coalesce fresh work arriving during cooldown; it may be the guest's
            // last update. Automatic retries alone never reopen an exhausted budget.
            drawable_retry_pending = YES;
            uint64_t generation = layer_generation;
            dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(delay * NSEC_PER_SEC)), presenter, ^{
                os_unfair_lock_lock(&self->screen_lock);
                if (self->layer_generation == generation) {
                    self->drawable_retry_pending = NO;
                    self->drawable_retry_cooldown = self->drawable_retry_deadline = 0;
                    [self kickPresenter:NO];
                }
                os_unfair_lock_unlock(&self->screen_lock);
            });
            return;
        }
        drawable_retry_cooldown = drawable_retry_deadline = 0;
    }
    presenter_running = YES;
    dispatch_async(presenter, ^{ [self presentLoop]; });
}

- (void)showOnLayer:(uint64_t)mapping {
    HvObject *surface = surfaces[@(mapping)];
    if (!surface) { [self complain:[NSString stringWithFormat:@"swap to unknown surface %llu", mapping]]; return; }
    on_screen = mapping;
    BOOL half = surface->fourcc == 'RGhA';
    if (!screen_tex) fprintf(stderr, "hengevm: display scanout format %s (%ux%u)\n",
                             half ? "RGBA16Float" : "BGRA8Unorm", surface->surface_w, surface->surface_h);
    if (!surface->scanout) {
        MTLTextureDescriptor *t = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:half ? MTLPixelFormatRGBA16Float : MTLPixelFormatBGRA8Unorm
                                                                                     width:surface->surface_w height:surface->surface_h mipmapped:NO];
        t.storageMode = MTLStorageModeShared;
        t.usage = MTLTextureUsageShaderRead;
        surface->scanout = [(id<MTLBuffer>)surface->object newTextureWithDescriptor:t offset:surface->first bytesPerRow:surface->row];
        if (!surface->scanout) { [self complain:[NSString stringWithFormat:@"no scanout for surface %llu, row %llu", mapping, surface->row]]; return; }
    }
    os_unfair_lock_lock(&screen_lock);
    screen_tex = surface->scanout, screen_half = half, screen_w = surface->surface_w, screen_h = surface->surface_h;
    screen_dirty = YES;
    [self kickPresenter:YES];
    os_unfair_lock_unlock(&screen_lock);
}

/* The pointer moved or changed: the screen again, with no need to go through the GPU queue's worker. */
- (void)screenChanged {
    os_unfair_lock_lock(&screen_lock);
    if (screen_tex) {
        screen_dirty = YES;
        [self kickPresenter:YES];
    }
    os_unfair_lock_unlock(&screen_lock);
}

- (void)presentLoop {
    for (;;) @autoreleasepool {
        os_unfair_lock_lock(&screen_lock);
        if (!screen_dirty || !layer) {
            presenter_running = NO;
            os_unfair_lock_unlock(&screen_lock);
            return;
        }
        screen_dirty = NO;
        CAMetalLayer *target = layer; // retained across nextDrawable and a concurrent disconnect
        uint64_t generation = layer_generation, requests = screen_requests;
        id<MTLTexture> tex = screen_tex, ctex = cursor_tex;
        BOOL half = screen_half, shown = cursor_shown;
        uint32_t W = screen_w, H = screen_h, cw0 = cursor_w, ch0 = cursor_h;
        int hx = cursor_hot_x, hy = cursor_hot_y;
        os_unfair_lock_unlock(&screen_lock);
        if (![self prepareScreenForLayer:target]) {
            os_unfair_lock_lock(&screen_lock);
            BOOL changed = layer_generation != generation;
            if (!changed) presenter_running = NO;
            os_unfair_lock_unlock(&screen_lock);
            if (changed) continue;
            return;
        }
        CGSize want = CGSizeMake(W, H);
        if (!CGSizeEqualToSize(target.drawableSize, want)) {
            [CATransaction begin];
            [CATransaction setDisableActions:YES];
            target.drawableSize = want;
            [CATransaction commit];
        }
        id<CAMetalDrawable> drawable = [target nextDrawable]; /* the wait is here */
        if (!drawable) {
            os_unfair_lock_lock(&screen_lock);
            screen_dirty = YES;
            BOOL changed = layer_generation != generation;
            if (!changed) {
                presenter_running = NO;
                double now = now_seconds();
                if (!drawable_retry_deadline) drawable_retry_deadline = now + 2;
                BOOL exhausted = now >= drawable_retry_deadline;
                if (exhausted) {
                    drawable_retry_cooldown = now + 1;
                    // A fresh frame/cursor event received while nextDrawable waited
                    // must not lose its wakeup at the transition into cooldown.
                    if (screen_requests != requests) [self kickPresenter:YES];
                } else {
                    // A monitor can connect before its first drawable is available.
                    // Retry even if the guest is idle, but never poll a hidden target forever.
                    drawable_retry_pending = YES;
                    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 100 * NSEC_PER_MSEC), presenter, ^{
                        os_unfair_lock_lock(&self->screen_lock);
                        if (self->layer_generation == generation) {
                            self->drawable_retry_pending = NO;
                            [self kickPresenter:NO];
                        }
                        os_unfair_lock_unlock(&self->screen_lock);
                    });
                }
            }
            os_unfair_lock_unlock(&screen_lock);
            if (changed) continue;
            return;
        }
        /* Something newer came during the wait? Then that is the frame to show. */
        os_unfair_lock_lock(&screen_lock);
        tex = screen_tex, half = screen_half, ctex = cursor_tex, shown = cursor_shown;
        cw0 = cursor_w, ch0 = cursor_h, hx = cursor_hot_x, hy = cursor_hot_y;
        BOOL same_target = layer_generation == generation && layer == target;
        BOOL same_size = screen_w == W && screen_h == H;
        if (same_target && same_size) {
            screen_dirty = NO;
            drawable_retry_deadline = 0;
        }
        os_unfair_lock_unlock(&screen_lock);
        if (!same_target || !same_size) { /* a target or mode change during nextDrawable */
            os_unfair_lock_lock(&screen_lock);
            screen_dirty = YES;
            os_unfair_lock_unlock(&screen_lock);
            continue;
        }
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = drawable.texture;
        pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:pass];
        [e setRenderPipelineState:half ? linear_pipe : screen_pipe];
        float rect[4] = {-1, -1, 2, 2};
        [e setVertexBytes:rect length:sizeof rect atIndex:0];
        [e setFragmentTexture:tex atIndex:0];
        [e setFragmentSamplerState:screen_sampler atIndex:0];
        if (half) {
            uint32_t half_mode = (uint32_t)hv_half_mode_get();
            [e setFragmentBytes:&half_mode length:sizeof half_mode atIndex:0];
        }
        [e drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
        const uint8_t *page = hv_pvg_lite_display_page(lite);
        uint32_t where = page ? __atomic_load_n((const uint32_t *)(page + 0xe00), __ATOMIC_ACQUIRE) : 0xffffffff;
        if (ctex && where != 0xffffffff && (shown || page[0xe04])) {
            const float scale = cursor_scale();
            float x = (float)(where & 0xffff) - hx * scale, y = (float)(where >> 16) - hy * scale;
            float cw = cw0 * scale, ch = ch0 * scale;
            float r[4] = {x / W * 2 - 1, 1 - (y + ch) / H * 2, cw / W * 2, ch / H * 2};
            [e setRenderPipelineState:cursor_pipe];
            [e setVertexBytes:r length:sizeof r atIndex:0];
            [e setFragmentTexture:ctex atIndex:0];
            NSNumber *sharpSetting = [[NSUserDefaults standardUserDefaults] objectForKey:@"henge.cursor.sharp"];
            [e setFragmentSamplerState:sharpSetting && !sharpSetting.boolValue ? screen_sampler : cursor_sampler atIndex:0];
            [e drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
        }
        [e endEncoding];
        if (cp.on) cp_capture_post(self, drawable.texture, W, H, cb);
#if !TARGET_OS_IPHONE
        if (getenv("HV_PVG_GPU_TEST")) { /* the same pixels for a screenshot: this is the path a phone runs */
            NSUInteger bytes = (NSUInteger)W * H * 4;
            if (!test_readback || test_readback.length < bytes) test_readback = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> rb = test_readback;
            id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
            [b copyFromTexture:drawable.texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(W, H, 1)
                      toBuffer:rb destinationOffset:0 destinationBytesPerRow:(NSUInteger)W * 4 destinationBytesPerImage:bytes];
            [b endEncoding];
            [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
                (void)done;
                dispatch_async(self->worker, ^{
                    if (!self->frame || self->frame_w != W || self->frame_h != H) {
                        self->frame = malloc((size_t)W * H * 4), self->frame_w = W, self->frame_h = H;
                    }
                    memcpy(self->frame, rb.contents, (size_t)W * H * 4);
                    self->frames++;
                });
            }];
        }
#endif
        // Serialize the final check and commit with set_layer: a switch that has
        // returned must not be followed by a commit to its old target.
        os_unfair_lock_lock(&screen_lock);
        if (layer_generation == generation && layer == target) {
            [cb presentDrawable:drawable];
            [cb commit];
            gpu_frames++;
        } else {
            screen_dirty = YES;
        }
        os_unfair_lock_unlock(&screen_lock);
    }
}

/* What the display should show: at most a few times a second by CPU, for tests and the boot's first frames. */
- (void)present:(uint64_t)mapping {
    double now = now_seconds();
    on_screen = mapping;
    if ([self hasDisplayLayer]) {
        late = NO;
        [self showOnLayer:mapping];
        return;
    }
    if (now - last_capture < 0.066) { /* 15 a second is plenty until this is GPU work */
        late_mapping = mapping, late = YES;
        if (!late_queued) {
            late_queued = YES;
            dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 70 * NSEC_PER_MSEC), worker, ^{
                self->late_queued = NO;
                if (self->late) [self present:self->late_mapping];
            });
        }
        return;
    }
    late = NO;
    HvObject *surface = surfaces[@(mapping)];
    if (!surface) { [self complain:[NSString stringWithFormat:@"swap to unknown surface %llu", mapping]]; return; }
    last_capture = now;
    [last_cb waitUntilCompleted]; /* the CPU copies the pixels here, so the GPU must be done with them */
    NSUInteger bpp = surface->fourcc == 'RGhA' ? 8 : 4, row = surface->row;
    uint32_t w = surface->surface_w, h = surface->surface_h;
    if (!frame || !base || frame_w != w || frame_h != h) {
        free(base);
        base = malloc((size_t)w * h * 4);
        uint8_t *old = frame;
        frame = calloc((size_t)w * h, 4); /* the old one may be on screen for a moment yet: a small leak beats a crash */
        (void)old;
        frame_w = w, frame_h = h;
        drawn_w = drawn_h = 0;
    }
    uint8_t *out = base;
    const uint8_t *src = surface->mapped + surface->first;
    const uint8_t *lut = half_to_byte();
    if (bpp == 8) {
        for (uint32_t y = 0; y < h; y++) {
            const uint16_t *in = (const uint16_t *)(src + y * row);
            uint8_t *o = out + (size_t)y * w * 4;
            for (uint32_t x = 0; x < w; x++, in += 4, o += 4) {
                o[0] = lut[in[2]], o[1] = lut[in[1]], o[2] = lut[in[0]], o[3] = 255;
            }
        }
    } else {
        for (uint32_t y = 0; y < h; y++) memcpy(out + (size_t)y * w * 4, src + y * row, (size_t)w * 4);
    }
    memcpy(frame, base, (size_t)w * h * 4);
    drawn_w = drawn_h = 0;
    [self drawCursor];
    frames++;
}

#if !TARGET_OS_IPHONE
/* Debugging: HV_PVG_SHOT=prefix keeps what is on the display, every HV_PVG_SHOT_SECS seconds it has changed. */
- (void)saveShot {
    if (!frame || frames == frames_shot) return;
    frames_shot = frames;
    char path[1100];
    snprintf(path, sizeof path, "%s-%04u.png", shot, shots++);
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(frame, frame_w, frame_h, 8, (size_t)frame_w * 4, cs, (CGBitmapInfo)kCGImageAlphaNoneSkipFirst | kCGBitmapByteOrder32Little);
    CGImageRef img = CGBitmapContextCreateImage(ctx);
    CFURLRef url = CFURLCreateFromFileSystemRepresentation(NULL, (const UInt8 *)path, (CFIndex)strlen(path), false);
    CGImageDestinationRef dest = CGImageDestinationCreateWithURL(url, (__bridge CFStringRef)UTTypePNG.identifier, 1, NULL);
    if (dest) {
        CGImageDestinationAddImage(dest, img, NULL);
        CGImageDestinationFinalize(dest);
        CFRelease(dest);
    }
    CFRelease(url), CGImageRelease(img), CGContextRelease(ctx), CGColorSpaceRelease(cs);
}
#endif

@end

HvPvgMetal *hv_pvg_metal_create(HvPvgLite *lite) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) return NULL;
        if (![dev respondsToSelector:@selector(newRenderPipelineDescriptorWithSerializedData:deserializationContext:)]) {
            fprintf(stderr, "hengevm: pvg-metal: this Metal cannot read serialized pipelines; no drawing\n");
            return NULL;
        }
        Protocol *proto = objc_getProtocol("MTLDeserializationContext");
        if (proto) class_addProtocol([HvContext class], proto);
        HvReplay *r = [HvReplay new];
        r->lite = lite;
        r->device = dev;
        r->queue = [dev newCommandQueueWithMaxCommandBufferCount:64];
        r->worker = dispatch_queue_create("hengevm.pvg", dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0));
        r->presenter = dispatch_queue_create("hengevm.present", dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0));
        r->screen_lock = OS_UNFAIR_LOCK_INIT;
        r->tasks = [NSMutableDictionary new];
        r->mapped_textures = [NSMutableDictionary new];
        r->surfaces = [NSMutableDictionary new];
        r->planar = [NSMutableDictionary new];
        r->planar_uses = [NSMutableDictionary new];
        r->complaints = [NSMutableSet new];
        r->shot = getenv("HV_PVG_SHOT");
        cp_start(r);
#if !TARGET_OS_IPHONE
        if (getenv("HV_PVG_GPU_TEST")) r->layer = [CAMetalLayer layer];
#endif
#if !TARGET_OS_IPHONE
        if (r->shot) {
            double every = getenv("HV_PVG_SHOT_SECS") ? atof(getenv("HV_PVG_SHOT_SECS")) : 5;
            dispatch_source_t timer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, r->worker);
            dispatch_source_set_timer(timer, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(every * NSEC_PER_SEC)), (uint64_t)(every * NSEC_PER_SEC), NSEC_PER_SEC / 10);
            HvReplay *__weak weak = r;
            dispatch_source_set_event_handler(timer, ^{ [weak saveShot]; });
            dispatch_resume(timer);
            r->shot_timer = timer;
        }
#endif
        fprintf(stderr, "hengevm: pvg-metal: drawing on %s\n", dev.name.UTF8String);
        return (HvPvgMetal *)CFBridgingRetain(r);
    }
}

void hv_pvg_metal_submit(HvPvgMetal *m, HvPvgLite *lite, uint32_t ch, uint32_t *cmd, uint32_t len) {
    HvReplay *r = (__bridge HvReplay *)m;
    dispatch_async(r->worker, ^{
        @autoreleasepool {
            uint32_t op = cmd[0] & 0xffff, skip = (cmd[0] >> 16) * 8; /* the barriers come first */
            const uint32_t *args = (const uint32_t *)((const uint8_t *)cmd + 12 + skip);
            uint32_t n = len > 12 + skip ? len - 12 - skip : 0;
            BOOL gpu_completes = NO;
            @try {
                if ((op == 0x43 || op == 0x45) && n >= 12) {
                    gpu_completes = [r exec:args length:n channel:ch stamp:cmd[2]];
                } else if ((op == 0x25 || op == 0x28) && n >= 8) { /* {task, ref} or {task, tag, size, ref} */
                    uint32_t ref = op == 0x25 ? args[1] : (n >= 16 ? args[3] : 0);
                    [r->tasks[@(args[0])] removeObjectForKey:@(ref)];
                } else if (op == 0x35 || op == 0x3e) {
                    [r synchronize:args length:n];
                } else if (op == 0x34) {
                    [r invalidate:args length:n];
                } else if (op == 0x3f) {
                    /* "May be thrown away": Apple's host only marks it purgeable, and the bytes stay
                       until the system needs the memory. Here they just stay. */
                } else if (op == 0x20 && n >= 4) {
                    [r->tasks removeObjectForKey:@(args[0] >> 1)];
                } else {
                    hv_pvg_lite_command(lite, cmd, len);
                    if (op == 8 && n >= 16) {
                        if (r->tracing) fprintf(stderr, "trace: swap to %llu\n", (uint64_t)args[3] << 32 | args[2]);
                        cp_note_swap(r, (uint64_t)args[3] << 32 | args[2], args, n);
                        [r present:(uint64_t)args[3] << 32 | args[2]];
                    }
                    if (op == 4) [r cursorGlyph:args length:n];
                    if (op == 5 && n >= 8) {
                        os_unfair_lock_lock(&r->screen_lock);
                        r->cursor_shown = args[1] & 1;
                        os_unfair_lock_unlock(&r->screen_lock);
                        if (getenv("HV_PVG_CURSOR_LOG")) fprintf(stderr, "pvgcursor show %u\n", args[1]);
                        [r drawCursor];
                    }
                }
            } @catch (NSException *x) {
                [r complain:[NSString stringWithFormat:@"exception: %@", x.reason]];
            }
            if (!gpu_completes) [r finishChannel:ch stamp:cmd[2]];
            free(cmd);
        }
    });
}

void hv_pvg_metal_set_layer(HvPvgMetal *m, void *layer) {
    if (!m) return;
    HvReplay *r = (__bridge HvReplay *)m;
    CAMetalLayer *l = (__bridge CAMetalLayer *)layer;
    // Invalidate immediately, even when the guest GPU worker has a backlog.
    os_unfair_lock_lock(&r->screen_lock);
    r->layer_generation++; /* reattaching the same layer also wakes a stopped presenter */
    r->layer = l;
    r->drawable_retry_pending = NO;
    r->drawable_retry_deadline = r->drawable_retry_cooldown = 0;
    r->screen_dirty = YES;
    [r kickPresenter:YES];
    os_unfair_lock_unlock(&r->screen_lock);
    dispatch_async(r->worker, ^{
        if (r->on_screen) [r present:r->on_screen];
    });
}

uint64_t hv_pvg_metal_gpu_frames(HvPvgMetal *m) {
    return m ? ((__bridge HvReplay *)m)->gpu_frames : 0;
}

void hv_pvg_metal_cursor_moved(HvPvgMetal *m) {
    if (!m) return;
    HvReplay *r = (__bridge HvReplay *)m;
    if ([r hasDisplayLayer]) { /* no need to queue behind the guest's GPU work */
        [r screenChanged];
        return;
    }
    static _Atomic int queued;
    if (atomic_exchange(&queued, 1)) return; /* one is already on its way, and it will read the newest position */
    dispatch_async(r->worker, ^{
        atomic_store(&queued, 0);
        [r drawCursor];
    });
}

void hv_pvg_metal_unmap(HvPvgMetal *m, uint32_t mapping) {
    HvReplay *r = (__bridge HvReplay *)m;
    dispatch_async(r->worker, ^{
        /* The last mapping of a shared multi-plane surface takes its table entry along: the entry holds the host
           IOSurface (4K video frames are 27-65 MB of Henge's own memory), and the guest reuses its memory for others. */
        HvObject *s = r->surfaces[@(mapping)];
        NSString *pk = s ? s->planar_key : nil;
        if (pk) {
            s->planar_key = nil;
            if ((__bridge IOSurfaceRef)r->planar[pk] == s->iosurface) { /* still this one's entry (the table may have been emptied since) */
                unsigned uses = r->planar_uses[pk].unsignedIntValue;
                if (uses > 1) r->planar_uses[pk] = @(uses - 1);
                else {
                    [r->planar removeObjectForKey:pk];
                    [r->planar_uses removeObjectForKey:pk];
                }
            }
        }
        [r->surfaces removeObjectForKey:@(mapping)];
        [r->mapped_textures removeObjectForKey:@(mapping)];
    });
}

/* A small picture of what the display shows, BGRA with no padding, for the app's home screen.
   Read straight from the surface's memory, nearest pixel: it may catch a frame being drawn, which
   does not matter at this size. -1 before anything has been shown. */
int hv_pvg_metal_snapshot(HvPvgMetal *m, uint8_t *bgra, uint32_t w, uint32_t h) {
    if (!m || !w || !h) return -1;
    HvReplay *r = (__bridge HvReplay *)m;
    os_unfair_lock_lock(&r->screen_lock);
    id<MTLTexture> tex = r->screen_tex;
    BOOL half = r->screen_half;
    uint32_t W = r->screen_w, H = r->screen_h;
    os_unfair_lock_unlock(&r->screen_lock);
    id<MTLBuffer> buffer = tex.buffer;
    if (!buffer || !W || !H || buffer.storageMode == MTLStorageModePrivate) return -1;
    const uint8_t *base = (const uint8_t *)buffer.contents + tex.bufferOffset;
    size_t row = tex.bufferBytesPerRow;
    /* Four samples to a pixel: one makes text look broken at this size. */
    for (uint32_t y = 0; y < h; y++) {
        uint8_t *out = bgra + (size_t)y * w * 4;
        for (uint32_t x = 0; x < w; x++, out += 4) {
            uint32_t sum[3] = {0, 0, 0};
            for (uint32_t k = 0; k < 4; k++) {
                size_t sx = (2 * x + (k & 1)) * (uint64_t)W / (2 * w), sy = (2 * y + (k >> 1)) * (uint64_t)H / (2 * h);
                const uint8_t *src = base + sy * row;
                if (!half) {
                    for (int c = 0; c < 3; c++) sum[c] += src[sx * 4 + c];
                    continue;
                }
                const __fp16 *p = (const __fp16 *)(src + sx * 8);
                for (int c = 0; c < 3; c++) {
                    float v = hv_half_shown((float)p[c], hv_half_mode_get()); /* as the screen shows it */
                    sum[2 - c] += (uint32_t)(v * 255 + 0.5f);
                }
            }
            for (int c = 0; c < 3; c++) out[c] = (uint8_t)((sum[c] + 2) / 4);
            out[3] = 255;
        }
    }
    return 0;
}

int hv_pvg_metal_frame(HvPvgMetal *m, const uint8_t **pixels, uint32_t *w, uint32_t *h) {
    if (!m) return -1;
    HvReplay *r = (__bridge HvReplay *)m;
    if (!r->frame) return -1;
    *pixels = r->frame, *w = r->frame_w, *h = r->frame_h;
    return 0;
}
