/* mountain_noise_neon.c — Mountain noise with 16-pixel unrolling
 *
 * Compile: cc -O3 -mcpu=apple-m4 -ffast-math -ffp-contract=fast -o mountain_noise_neon mountain_noise_neon.c
 * Run:     ./mountain_noise [width] [height] [freq] [seed] [output.pgm]
 */
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static inline __attribute__((always_inline)) uint32x4_t hash2d4(int32x4_t ix, int32x4_t iy, uint32x4_t sv) {
    uint32x4_t h = vreinterpretq_u32_s32(ix);
    h = vmulq_n_u32(h, 0x9E3779B9u);
    h = veorq_u32(h, veorq_u32(vreinterpretq_u32_s32(iy), sv));
    h = veorq_u32(h, vshrq_n_u32(h, 16));
    h = vmulq_n_u32(h, 0x85ebca6bu);
    h = veorq_u32(h, vshrq_n_u32(h, 13));
    return h;
}

static void fill(float * restrict out, int w, int h, float freq, int32_t seed) {
    const float32x4_t Cx  = vdupq_n_f32(0.211324865405f);
    const float32x4_t Cy  = vdupq_n_f32(0.366025403784f);
    const float32x4_t Ch  = vdupq_n_f32(0.5f);
    const float32x4_t C1  = vdupq_n_f32(1.0f);
    const float32x4_t Cx2 = vdupq_n_f32(0.42264973081f);
    const float32x4_t Cx_m1 = vdupq_n_f32(0.211324865405f - 1.0f);  /* Cx - 1 */
    const float32x4_t zero  = vdupq_n_f32(0.0f);
    const int32x4_t  one_i = vdupq_n_s32(1);
    const float32x4_t Cn_half = vdupq_n_f32(46.432f);
    const uint32x4_t sv = vdupq_n_u32((uint32_t)seed * 0x45d9f3bu);
    const float32x4_t C = vdupq_n_f32(3.14159265f/64.0f);
    const float32x4_t pi = vdupq_n_f32(3.14159265f);
    const float32x4_t tpi2 = vdupq_n_f32(19.7392088f);
    const uint32x4_t s = vdupq_n_u32(0x80000000u);
    const float32x4_t step8 = vdupq_n_f32(8.0f * freq);

    #define GRADIENT_AND_TWO_CONTRIB(hh, dx0, dy0, dx1, dy1, out0, out1) do { \
        float32x4_t _off = vcvtq_f32_u32(vandq_u32((hh), vdupq_n_u32(0x3f))); \
        float32x4_t _v   = vmulq_f32(_off, C); \
        float32x4_t _u   = vsubq_f32(pi, _v); \
        float32x4_t _u2  = vmulq_f32(_u, _u); \
        float32x4_t _v2  = vmulq_f32(_v, _v); \
        float32x4_t _rca = vrecpeq_f32(vsubq_f32(tpi2, _u2)); \
        float32x4_t _rsa = vrecpeq_f32(vsubq_f32(tpi2, _v2)); \
        float32x4_t _d0 = vmlaq_f32(vmulq_f32((dx0), (dx0)), (dy0), (dy0)); \
        float32x4_t _d1 = vmlaq_f32(vmulq_f32((dx1), (dx1)), (dy1), (dy1)); \
        float32x4_t _m0 = vmaxq_f32(vsubq_f32(Ch, _d0), zero); \
        float32x4_t _m1 = vmaxq_f32(vsubq_f32(Ch, _d1), zero); \
        _m0 = vmulq_f32(_m0, _m0); _m0 = vmulq_f32(_m0, _m0); \
        _m1 = vmulq_f32(_m1, _m1); _m1 = vmulq_f32(_m1, _m1); \
        float32x4_t _ca0 = vmulq_f32(_u2, _rca); \
        float32x4_t _sa0 = vmulq_f32(_v2, _rsa); \
        uint32x4_t _h23 = vshlq_n_u32((hh), 23); \
        _ca0 = vreinterpretq_f32_u32(veorq_u32( \
            vreinterpretq_u32_f32(_ca0), \
            vandq_u32(veorq_u32(_h23, vshlq_n_u32(_h23, 2)), s))); \
        _sa0 = vreinterpretq_f32_u32(veorq_u32( \
            vreinterpretq_u32_f32(_sa0), \
            vandq_u32(vshlq_n_u32(_h23, 1), s))); \
        (out0) = vmulq_f32(_m0, vmlaq_f32(vmulq_f32(_ca0, (dx0)), _sa0, (dy0))); \
        (out1) = vmulq_f32(_m1, vmlaq_f32(vmulq_f32(_ca0, (dx1)), _sa0, (dy1))); \
    } while(0)

    #define DOT_CONTRIB(ca, sa, dx, dy, out) do { \
        float32x4_t _d = vfmaq_f32(vmulq_f32((dx), (dx)), (dy), (dy)); \
        float32x4_t _m = vmaxq_f32(vsubq_f32(Ch, _d), zero); \
        _m = vmulq_f32(_m, _m); _m = vmulq_f32(_m, _m); \
        (out) = vmulq_f32(_m, vfmaq_f32(vmulq_f32((ca), (dx)), (sa), (dy))); \
    } while(0)

    #define GRADIENT_FROM_HASH(hh, out_ca, out_sa) do { \
        float32x4_t _off = vcvtq_f32_u32(vandq_u32((hh), vdupq_n_u32(0x3f))); \
        float32x4_t _v   = vmulq_f32(_off, C); \
        float32x4_t _u   = vsubq_f32(pi, _v); \
        float32x4_t _u2  = vmulq_f32(_u, _u); \
        float32x4_t _v2  = vmulq_f32(_v, _v); \
        float32x4_t _rca = vrecpeq_f32(vsubq_f32(tpi2, _u2)); \
        float32x4_t _rsa = vrecpeq_f32(vsubq_f32(tpi2, _v2)); \
        float32x4_t _ca0 = vmulq_f32(_u2, _rca); \
        float32x4_t _sa0 = vmulq_f32(_v2, _rsa); \
        uint32x4_t _h23 = vshlq_n_u32((hh), 23); \
        (out_ca) = vreinterpretq_f32_u32(veorq_u32( \
            vreinterpretq_u32_f32(_ca0), \
            vandq_u32(veorq_u32(_h23, vshlq_n_u32(_h23, 2)), s))); \
        (out_sa) = vreinterpretq_f32_u32(veorq_u32( \
            vreinterpretq_u32_f32(_sa0), \
            vandq_u32(vshlq_n_u32(_h23, 1), s))); \
    } while(0)

    for (int py = 0; py < h; py++) {
        float *row = out + (size_t)py * w;
        float32x4_t vsy = vdupq_n_f32(py * freq);
        int px = 0;

        /* 16-pixel loop: process A completely, then B */
        for (; px + 15 < w; px += 16) {
            /* === Block A: pixels px..px+7 === */
            float32x4_t vx0a = {px*freq,(px+1)*freq,(px+2)*freq,(px+3)*freq};
            float32x4_t vx1a = vaddq_f32(vx0a, vdupq_n_f32(4.0f*freq));
            float32x4_t s0a=vmulq_f32(vaddq_f32(vx0a,vsy),Cy);
            float32x4_t s1a=vmulq_f32(vaddq_f32(vx1a,vsy),Cy);
            float32x4_t iv0a=vrndmq_f32(vaddq_f32(vx0a,s0a)), jv0a=vrndmq_f32(vaddq_f32(vsy,s0a));
            float32x4_t iv1a=vrndmq_f32(vaddq_f32(vx1a,s1a)), jv1a=vrndmq_f32(vaddq_f32(vsy,s1a));
            float32x4_t ij0a=vaddq_f32(iv0a,jv0a), ij1a=vaddq_f32(iv1a,jv1a);
            float32x4_t x00a=vfmaq_f32(vsubq_f32(vx0a,iv0a),ij0a,Cx);
            float32x4_t y00a=vfmaq_f32(vsubq_f32(vsy,jv0a),ij0a,Cx);
            float32x4_t x01a=vfmaq_f32(vsubq_f32(vx1a,iv1a),ij1a,Cx);
            float32x4_t y01a=vfmaq_f32(vsubq_f32(vsy,jv1a),ij1a,Cx);
            uint32x4_t split0a = vcgtq_f32(x00a,y00a), split1a = vcgtq_f32(x01a,y01a);
            float32x4_t i1x0a=vbslq_f32(split0a,C1,zero), i1y0a=vbslq_f32(split0a,zero,C1);
            float32x4_t i1x1a=vbslq_f32(split1a,C1,zero), i1y1a=vbslq_f32(split1a,zero,C1);
            int32x4_t xi0a=vcvtq_s32_f32(iv0a), yi0a=vcvtq_s32_f32(jv0a);
            int32x4_t xi1a=vcvtq_s32_f32(iv1a), yi1a=vcvtq_s32_f32(jv1a);
            int32x4_t i1xi0a=vreinterpretq_s32_u32(vandq_u32(split0a, vdupq_n_u32(1)));
            int32x4_t i1yi0a=vreinterpretq_s32_u32(vandq_u32(vmvnq_u32(split0a), vdupq_n_u32(1)));
            int32x4_t i1xi1a=vreinterpretq_s32_u32(vandq_u32(split1a, vdupq_n_u32(1)));
            int32x4_t i1yi1a=vreinterpretq_s32_u32(vandq_u32(vmvnq_u32(split1a), vdupq_n_u32(1)));

            float32x4_t n0a=zero, n1a=zero;

            /* A Corner 0 */
            {uint32x4_t ha=hash2d4(xi1a,yi1a,sv);
             float32x4_t ca0,ca1;
             GRADIENT_AND_TWO_CONTRIB(ha, x00a, y00a, x01a, y01a, ca0, ca1);
             n0a=vaddq_f32(n0a,ca0); n1a=vaddq_f32(n1a,ca1);}

            /* A Corner 1 */
            {float32x4_t x10a=vaddq_f32(vsubq_f32(x00a,i1x0a),Cx), y10a=vaddq_f32(vsubq_f32(y00a,i1y0a),Cx);
             float32x4_t x11a=vaddq_f32(vsubq_f32(x01a,i1x1a),Cx), y11a=vaddq_f32(vsubq_f32(y01a,i1y1a),Cx);
             uint32x4_t h0a=hash2d4(vaddq_s32(xi0a,i1xi0a),vaddq_s32(yi0a,i1yi0a),sv);
             uint32x4_t h1a=hash2d4(vaddq_s32(xi1a,i1xi1a),vaddq_s32(yi1a,i1yi1a),sv);
             float32x4_t c0a,c1a,ca0,sa0,ca1,sa1;
             GRADIENT_FROM_HASH(h0a, ca0, sa0); DOT_CONTRIB(ca0, sa0, x10a, y10a, c0a);
             GRADIENT_FROM_HASH(h1a, ca1, sa1); DOT_CONTRIB(ca1, sa1, x11a, y11a, c1a);
             n0a=vaddq_f32(n0a,c0a); n1a=vaddq_f32(n1a,c1a);}

            /* A Corner 2 */
            {float32x4_t x20a=vaddq_f32(vsubq_f32(x00a,C1),Cx2), y20a=vaddq_f32(vsubq_f32(y00a,C1),Cx2);
             float32x4_t x21a=vaddq_f32(vsubq_f32(x01a,C1),Cx2), y21a=vaddq_f32(vsubq_f32(y01a,C1),Cx2);
             uint32x4_t ha=hash2d4(vaddq_s32(xi0a,one_i),vaddq_s32(yi0a,one_i),sv);
             float32x4_t ca0,ca1;
             GRADIENT_AND_TWO_CONTRIB(ha, x20a, y20a, x21a, y21a, ca0, ca1);
             n0a=vaddq_f32(n0a,ca0); n1a=vaddq_f32(n1a,ca1);}

            /* A normalize and store */
            n0a=vmlaq_f32(Ch,n0a,Cn_half); n1a=vmlaq_f32(Ch,n1a,Cn_half);
            vst1q_f32(row + px, n0a);
            vst1q_f32(row + px + 4, n1a);

            /* === Block B: pixels px+8..px+15 — fully independent === */
            float32x4_t vx0b = vaddq_f32(vx0a, step8);
            float32x4_t vx1b = vaddq_f32(vx1a, step8);
            float32x4_t s0b=vmulq_f32(vaddq_f32(vx0b,vsy),Cy);
            float32x4_t s1b=vmulq_f32(vaddq_f32(vx1b,vsy),Cy);
            float32x4_t iv0b=vrndmq_f32(vaddq_f32(vx0b,s0b)), jv0b=vrndmq_f32(vaddq_f32(vsy,s0b));
            float32x4_t iv1b=vrndmq_f32(vaddq_f32(vx1b,s1b)), jv1b=vrndmq_f32(vaddq_f32(vsy,s1b));
            float32x4_t ij0b=vaddq_f32(iv0b,jv0b), ij1b=vaddq_f32(iv1b,jv1b);
            float32x4_t x00b=vfmaq_f32(vsubq_f32(vx0b,iv0b),ij0b,Cx);
            float32x4_t y00b=vfmaq_f32(vsubq_f32(vsy,jv0b),ij0b,Cx);
            float32x4_t x01b=vfmaq_f32(vsubq_f32(vx1b,iv1b),ij1b,Cx);
            float32x4_t y01b=vfmaq_f32(vsubq_f32(vsy,jv1b),ij1b,Cx);
            uint32x4_t split0b = vcgtq_f32(x00b,y00b), split1b = vcgtq_f32(x01b,y01b);
            float32x4_t i1x0b=vbslq_f32(split0b,C1,zero), i1y0b=vbslq_f32(split0b,zero,C1);
            float32x4_t i1x1b=vbslq_f32(split1b,C1,zero), i1y1b=vbslq_f32(split1b,zero,C1);
            int32x4_t xi0b=vcvtq_s32_f32(iv0b), yi0b=vcvtq_s32_f32(jv0b);
            int32x4_t xi1b=vcvtq_s32_f32(iv1b), yi1b=vcvtq_s32_f32(jv1b);
            int32x4_t i1xi0b=vreinterpretq_s32_u32(vandq_u32(split0b, vdupq_n_u32(1)));
            int32x4_t i1yi0b=vreinterpretq_s32_u32(vandq_u32(vmvnq_u32(split0b), vdupq_n_u32(1)));
            int32x4_t i1xi1b=vreinterpretq_s32_u32(vandq_u32(split1b, vdupq_n_u32(1)));
            int32x4_t i1yi1b=vreinterpretq_s32_u32(vandq_u32(vmvnq_u32(split1b), vdupq_n_u32(1)));

            float32x4_t n0b=zero, n1b=zero;

            /* B Corner 0 */
            {uint32x4_t hb=hash2d4(xi1b,yi1b,sv);
             float32x4_t cb0,cb1;
             GRADIENT_AND_TWO_CONTRIB(hb, x00b, y00b, x01b, y01b, cb0, cb1);
             n0b=vaddq_f32(n0b,cb0); n1b=vaddq_f32(n1b,cb1);}

            /* B Corner 1 */
            {float32x4_t x10b=vaddq_f32(vsubq_f32(x00b,i1x0b),Cx), y10b=vaddq_f32(vsubq_f32(y00b,i1y0b),Cx);
             float32x4_t x11b=vaddq_f32(vsubq_f32(x01b,i1x1b),Cx), y11b=vaddq_f32(vsubq_f32(y01b,i1y1b),Cx);
             uint32x4_t h0b=hash2d4(vaddq_s32(xi0b,i1xi0b),vaddq_s32(yi0b,i1yi0b),sv);
             uint32x4_t h1b=hash2d4(vaddq_s32(xi1b,i1xi1b),vaddq_s32(yi1b,i1yi1b),sv);
             float32x4_t c0b,c1b,cb0,sa0b,cb1,sa1b;
             GRADIENT_FROM_HASH(h0b, cb0, sa0b); DOT_CONTRIB(cb0, sa0b, x10b, y10b, c0b);
             GRADIENT_FROM_HASH(h1b, cb1, sa1b); DOT_CONTRIB(cb1, sa1b, x11b, y11b, c1b);
             n0b=vaddq_f32(n0b,c0b); n1b=vaddq_f32(n1b,c1b);}

            /* B Corner 2 */
            {float32x4_t x20b=vaddq_f32(vsubq_f32(x00b,C1),Cx2), y20b=vaddq_f32(vsubq_f32(y00b,C1),Cx2);
             float32x4_t x21b=vaddq_f32(vsubq_f32(x01b,C1),Cx2), y21b=vaddq_f32(vsubq_f32(y01b,C1),Cx2);
             uint32x4_t hb=hash2d4(vaddq_s32(xi0b,one_i),vaddq_s32(yi0b,one_i),sv);
             float32x4_t cb0,cb1;
             GRADIENT_AND_TWO_CONTRIB(hb, x20b, y20b, x21b, y21b, cb0, cb1);
             n0b=vaddq_f32(n0b,cb0); n1b=vaddq_f32(n1b,cb1);}

            /* B normalize and store */
            n0b=vmlaq_f32(Ch,n0b,Cn_half); n1b=vmlaq_f32(Ch,n1b,Cn_half);
            vst1q_f32(row + px + 8, n0b);
            vst1q_f32(row + px + 12, n1b);
        }

        /* 8-pixel fallback */
        for (; px + 7 < w; px += 8) {
            float32x4_t vx0 = {px*freq,(px+1)*freq,(px+2)*freq,(px+3)*freq};
            float32x4_t vx1 = vaddq_f32(vx0, vdupq_n_f32(4.0f*freq));
            float32x4_t s0=vmulq_f32(vaddq_f32(vx0,vsy),Cy);
            float32x4_t s1=vmulq_f32(vaddq_f32(vx1,vsy),Cy);
            float32x4_t iv0=vrndmq_f32(vaddq_f32(vx0,s0)), jv0=vrndmq_f32(vaddq_f32(vsy,s0));
            float32x4_t iv1=vrndmq_f32(vaddq_f32(vx1,s1)), jv1=vrndmq_f32(vaddq_f32(vsy,s1));
            float32x4_t ij0=vaddq_f32(iv0,jv0), ij1=vaddq_f32(iv1,jv1);
            float32x4_t x00=vfmaq_f32(vsubq_f32(vx0,iv0),ij0,Cx);
            float32x4_t y00=vfmaq_f32(vsubq_f32(vsy,jv0),ij0,Cx);
            float32x4_t x01=vfmaq_f32(vsubq_f32(vx1,iv1),ij1,Cx);
            float32x4_t y01=vfmaq_f32(vsubq_f32(vsy,jv1),ij1,Cx);
            uint32x4_t split0 = vcgtq_f32(x00,y00), split1 = vcgtq_f32(x01,y01);
            float32x4_t i1x0=vbslq_f32(split0,C1,zero), i1y0=vbslq_f32(split0,zero,C1);
            float32x4_t i1x1=vbslq_f32(split1,C1,zero), i1y1=vbslq_f32(split1,zero,C1);
            int32x4_t xi0=vcvtq_s32_f32(iv0), yi0=vcvtq_s32_f32(jv0);
            int32x4_t xi1=vcvtq_s32_f32(iv1), yi1=vcvtq_s32_f32(jv1);
            int32x4_t i1xi0=vreinterpretq_s32_u32(vandq_u32(split0, vdupq_n_u32(1)));
            int32x4_t i1yi0=vreinterpretq_s32_u32(vandq_u32(vmvnq_u32(split0), vdupq_n_u32(1)));
            int32x4_t i1xi1=vreinterpretq_s32_u32(vandq_u32(split1, vdupq_n_u32(1)));
            int32x4_t i1yi1=vreinterpretq_s32_u32(vandq_u32(vmvnq_u32(split1), vdupq_n_u32(1)));
            float32x4_t n0=zero, n1=zero;
            {uint32x4_t h=hash2d4(xi1,yi1,sv);
             float32x4_t c0, c1;
             GRADIENT_AND_TWO_CONTRIB(h, x00, y00, x01, y01, c0, c1);
             n0=vaddq_f32(n0,c0); n1=vaddq_f32(n1,c1);}
            {float32x4_t x10=vaddq_f32(vsubq_f32(x00,i1x0),Cx), y10=vaddq_f32(vsubq_f32(y00,i1y0),Cx);
             float32x4_t x11=vaddq_f32(vsubq_f32(x01,i1x1),Cx), y11=vaddq_f32(vsubq_f32(y01,i1y1),Cx);
             uint32x4_t h0=hash2d4(vaddq_s32(xi0,i1xi0),vaddq_s32(yi0,i1yi0),sv);
             uint32x4_t h1=hash2d4(vaddq_s32(xi1,i1xi1),vaddq_s32(yi1,i1yi1),sv);
             float32x4_t ca0,sa0,ca1,sa1;
             GRADIENT_FROM_HASH(h0,ca0,sa0); DOT_CONTRIB(ca0,sa0,x10,y10,n0);
             GRADIENT_FROM_HASH(h1,ca1,sa1); DOT_CONTRIB(ca1,sa1,x11,y11,n1);
             n0=vaddq_f32(n0,n0); n1=vaddq_f32(n1,n1);}
            {float32x4_t x20=vaddq_f32(vsubq_f32(x00,C1),Cx2), y20=vaddq_f32(vsubq_f32(y00,C1),Cx2);
             float32x4_t x21=vaddq_f32(vsubq_f32(x01,C1),Cx2), y21=vaddq_f32(vsubq_f32(y01,C1),Cx2);
             uint32x4_t h=hash2d4(vaddq_s32(xi0,one_i),vaddq_s32(yi0,one_i),sv);
             float32x4_t c0, c1;
             GRADIENT_AND_TWO_CONTRIB(h, x20, y20, x21, y21, c0, c1);
             n0=vaddq_f32(n0,c0); n1=vaddq_f32(n1,c1);}
            n0=vmlaq_f32(Ch,n0,Cn_half); n1=vmlaq_f32(Ch,n1,Cn_half);
            vst1q_f32(row + px, n0); vst1q_f32(row + px + 4, n1);
        }

        /* Scalar tail */
        for (; px < w; px++) {
            float sx=px*freq, sy=py*freq, s2=(sx+sy)*0.366025403784f;
            float fi=__builtin_floorf(sx+s2), fj=__builtin_floorf(sy+s2);
            float t2=(fi+fj)*0.211324865405f, x0=sx-(fi-t2), y0=sy-(fj-t2);
            int i1=x0>y0, j1=!i1;
            float x1=x0-i1+0.211324865405f, y1=y0-j1+0.211324865405f;
            float x2=x0-1+0.42264973081f, y2=y0-1+0.42264973081f;
            uint32_t su=(uint32_t)seed*0x45d9f3bu;
            #define H(a,b) ({uint32_t _h=(uint32_t)(a);_h*=0x9E3779B9u; \
                _h^=(uint32_t)(b)^su;_h^=_h>>16;_h*=0x85ebca6bu;_h^=_h>>13;_h;})
            #define GG(hh,dx,dy) ({uint32_t _hh=(hh); \
                float _f=(_hh&0x3F)*(3.14159265f/64.0f); \
                float _v=_f,_u=3.14159265f-_v; \
                float _u2=_u*_u,_v2=_v*_v,_t=19.7392088f; \
                float _ca=_u2/(_t-_u2),_sa=_v2/(_t-_v2); \
                uint32_t _sign_ca=((_hh>>8)^(_hh>>6))&1u; \
                uint32_t _sign_sa=(_hh>>7)&1u; \
                if(_sign_ca)_ca=-_ca; if(_sign_sa)_sa=-_sa; \
                float _m=fmaxf(.5f-((dx)*(dx)+(dy)*(dy)),0); \
                _m*=_m; _m*=_m; _m*(_ca*(dx)+_sa*(dy));})
            row[px]=(GG(H((int)fi,(int)fj),x0,y0) \
                +GG(H((int)fi+i1,(int)fj+j1),x1,y1) \
                +GG(H((int)fi+1,(int)fj+1),x2,y2))*92.864f*0.5f+0.5f;
            #undef H
            #undef GG
        }
    }
    #undef GRADIENT_AND_TWO_CONTRIB
    #undef DOT_CONTRIB
    #undef GRADIENT_FROM_HASH
}

#define NITER 1000
#define WARMUP 500

int main(int argc, char **argv) {
    int w=1024, h=1024; float freq=0.04f; int seed=42;
    const char *out = "mountain_noise.pgm";
    if (argc>1) w=atoi(argv[1]); if (argc>2) h=atoi(argv[2]);
    if (argc>3) freq=atof(argv[3]); if (argc>4) seed=atoi(argv[4]);
    if (argc>5) out=argv[5];
    float *buf = malloc(w*h*sizeof(float));
    size_t npix = (size_t)w * h;

    /* Warm-up: stabilize caches and CPU frequency */
    for (int i=0; i<WARMUP; i++) fill(buf, w, h, freq, seed);

    /* Benchmark: collect individual iteration times */
    double times[NITER];
    volatile float sink = 0;
    for (int i=0; i<NITER; i++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        fill(buf, w, h, freq, seed);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        sink += buf[(i * 997) % npix];
        times[i] = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    }

    /* Sort for median/percentile */
    for (int i=0; i<NITER-1; i++)
        for (int j=i+1; j<NITER; j++)
            if (times[i] > times[j]) { double t=times[i]; times[i]=times[j]; times[j]=t; }

    double median = times[NITER/2];
    double p10    = times[NITER/10];
    double p90    = times[NITER*9/10];
    double mean   = 0;
    for (int i=0; i<NITER; i++) mean += times[i];
    mean /= NITER;

    printf("%dx%d freq=%.2f seed=%d  (%d iterations, %d warmup)\n",
           w, h, freq, seed, NITER, WARMUP);
    printf("  median: %.3f ms  %.0f Mpixels/s\n",
           median*1000, npix/median/1e6);
    printf("  mean:   %.3f ms  %.0f Mpixels/s\n",
           mean*1000, npix/mean/1e6);
    printf("  p10:    %.3f ms  %.0f Mp/s (best 10%%)\n",
           p10*1000, npix/p10/1e6);
    printf("  p90:    %.3f ms  %.0f Mp/s (worst 10%%)\n",
           p90*1000, npix/p90/1e6);
    printf("  spread: %.1f%% (p90/p10 - 1)\n", (p90/p10 - 1)*100);
    if (sink != sink) printf("  (sink check)\n");

    /* Generate output image */
    fill(buf, w, h, freq, seed);
    FILE *f=fopen(out,"wb"); fprintf(f,"P5\n%d %d\n255\n",w,h);
    for (size_t i=0; i<npix; i++) { int v=buf[i]*255; fputc(v<0?0:v>255?255:v,f); }
    fclose(f); printf("  saved -> %s\n", out);
    free(buf);
}
