/* Metal libraries as macOS sends them, relabelled for an older iOS (devices/metallib.h).

   A metallib (as xcrun metallib writes it, file version 2.x) starts
     +0 "MTLB", +4 u16 platform (0x8001 macOS, 0x0001 iOS), +6 u16 file version major, +8 minor,
     +10 u8 library type, +11 u8 target OS (0x81 macOS, 0x82 iOS), +12 u16 OS version major, +14 minor,
     +16 u64 file size, +24 u64 function list offset, +32 its size, +40/+48 public metadata,
     +56/+64 private metadata, +72 u64 bitcode offset, +80 its size.
   The function list is a u32 count (not in the list size), then per function a u32 size (counted from that word) and tags
   {4-byte name, u16 length, data} up to ENDT: HASH (SHA-256 of the function's bitcode), MDSZ (its
   size), OFFT (u64 public, private and bitcode offsets, relative to the sections), VERS (u16 AIR
   major, minor, language major, minor). The same shader built for macOS 15 and iOS 17 differs in
   the header, VERS (AIR 2.7 and Metal 3.2 against 2.6 and 3.1), and the bitcode's target triple
   (air64_v27-apple-macosx15.0.0 against air64_v26-apple-ios17.0.0) and so its hash. */
#include "metallib.h"

#include <CommonCrypto/CommonDigest.h>
#include <string.h>

static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | p[1] << 8); }
static uint32_t rd32(const uint8_t *p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24; }
static uint64_t rd64(const uint8_t *p) { return (uint64_t)rd32(p) | (uint64_t)rd32(p + 4) << 32; }
static void wr16(uint8_t *p, uint16_t v) { p[0] = (uint8_t)v, p[1] = (uint8_t)(v >> 8); }

/* "air64_vNN-apple-macosxA.B.C" in [p, end) becomes "air64_v26-apple-ios17...." of the same length. */
static int relabel_triple(uint8_t *p, size_t n) {
    static const char key[] = "-apple-macosx";
    int changed = 0;
    for (size_t i = 9; i + sizeof key - 1 < n; i++) {
        if (memcmp(p + i, key, sizeof key - 1) != 0) continue;
        if (memcmp(p + i - 9, "air64_v", 7) != 0) continue;
        size_t v = i + sizeof key - 1, e = v; /* the version: digits and dots */
        while (e < n && ((p[e] >= '0' && p[e] <= '9') || p[e] == '.')) e++;
        size_t len = e - (i + 7); /* "macosx" + version */
        char out[64];
        size_t rest = len - 6; /* after "ios17.": "ios17.0.0" needs room for two parts */
        if (len < 12 || len >= sizeof out) continue;
        p[i - 2] = '2', p[i - 1] = '6';
        memcpy(out, "ios17.", 6);
        size_t a = rest / 2; /* "0..0" "." "0..0" */
        memset(out + 6, '0', rest);
        out[6 + a] = '.';
        memcpy(p + i + 7, out, len);
        changed++;
    }
    return changed;
}

int hv_metallib_relabel(uint8_t *buf, size_t len, int stage) {
    if (len < 88 || memcmp(buf, "MTLB", 4) != 0 || rd16(buf + 6) != 2) return -1;
    if (rd64(buf + 16) > len) return -1;
    int changed = 0;
    if (rd16(buf + 4) & 0x8000) wr16(buf + 4, rd16(buf + 4) & 0x7fff), changed++;
    if (rd16(buf + 8) > 7) wr16(buf + 8, 7), changed++;
    if (buf[11] == 0x81) buf[11] = 0x82, changed++;
    if (rd16(buf + 12) > 17) wr16(buf + 12, 17), wr16(buf + 14, 0), changed++;
    uint64_t fl = rd64(buf + 24), fls = rd64(buf + 32), bc = rd64(buf + 72), bcs = rd64(buf + 80);
    if (fl > len || fls > len - fl - 4 || bc > len || bcs > len - bc) return changed;
    const uint8_t *end = buf + fl + 4 + fls; /* the size counts the entries after the count */
    uint32_t count = rd32(buf + fl);
    uint8_t *f = buf + fl + 4;
    for (uint32_t k = 0; k < count && f + 4 <= end; k++) {
        uint32_t size = rd32(f);
        if (size < 4 || size > (size_t)(end - f)) break;
        uint8_t *hash = NULL, *t = f + 4, *fend = f + size;
        uint64_t mdsz = 0, bc_off = ~0ull;
        while (t + 6 <= fend && memcmp(t, "ENDT", 4) != 0) {
            uint16_t n = rd16(t + 4);
            uint8_t *d = t + 6;
            if (d + n > fend) break;
            if (!memcmp(t, "VERS", 4) && n >= 8) {
                if (rd16(d) == 2 && rd16(d + 2) > 6) wr16(d + 2, 6), changed++;           /* AIR 2.7 and up: 2.6 */
                if (rd16(d + 4) == 3 && rd16(d + 6) > 1) wr16(d + 6, 1), changed++;       /* Metal 3.2 and up: 3.1 */
            } else if (!memcmp(t, "HASH", 4) && n == 32) {
                hash = d;
            } else if (!memcmp(t, "MDSZ", 4) && n == 8) {
                mdsz = rd64(d);
            } else if (!memcmp(t, "OFFT", 4) && n == 24) {
                bc_off = rd64(d + 16);
            }
            t = d + n;
        }
        if (stage >= 2 && hash && bc_off != ~0ull && bc_off <= bcs && mdsz <= bcs - bc_off) {
            uint8_t *code = buf + bc + bc_off;
            int c = relabel_triple(code, (size_t)mdsz);
            if (c) {
                CC_SHA256(code, (CC_LONG)mdsz, hash);
                changed += c;
            }
        }
        f += size;
    }
    return changed;
}
