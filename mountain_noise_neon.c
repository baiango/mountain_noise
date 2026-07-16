/* mountain_noise_standalone.c — Standalone peak noise generator
 *
 * Run:     ./mountain_noise [width] [height] [freq] [seed] [output.pgm]
 */
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static inline uint32x4_t hash2d4(int32x4_t ix, int32x4_t iy, uint32x4_t sv) {
    uint32x4_t h = vreinterpretq_u32_s32(ix);
    h = vmulq_u32(h, vdupq_n_u32(0x9E3779B9u));
    h = veorq_u32(h, veorq_u32(vreinterpretq_u32_s32(iy), sv));
    h = veorq_u32(h, vshrq_n_u32(h, 16));
    h = vmulq_u32(h, vdupq_n_u32(0x85ebca6bu));
    h = veorq_u32(h, vshrq_n_u32(h, 13));
    return h;
}

static inline void grad(uint32x4_t h, float32x4_t *gx, float32x4_t *gy) {
    const float32x4_t C = vdupq_n_f32(3.14159265f/64.0f);
    const float32x4_t pi = vdupq_n_f32(3.14159265f);
    const float32x4_t tpi2 = vdupq_n_f32(19.7392088f);
    const uint32x4_t s = vdupq_n_u32(0x80000000u);
    float32x4_t offf = vcvtq_f32_u32(vandq_u32(h, vdupq_n_u32(0x3F)));
    float32x4_t v = vmulq_f32(offf, C), u = vsubq_f32(pi, v);
    float32x4_t u2 = vmulq_f32(u,u), v2 = vmulq_f32(v,v);
    float32x4_t ca = vmulq_f32(u2, vrecpeq_f32(vsubq_f32(tpi2, u2)));
    float32x4_t sa = vmulq_f32(v2, vrecpeq_f32(vsubq_f32(tpi2, v2)));
    uint32x4_t h23 = vshlq_n_u32(h, 23);
    *gx = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(ca),
        vandq_u32(veorq_u32(h23, vshlq_n_u32(h23, 2)), s)));
    *gy = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(sa),
        vandq_u32(vshlq_n_u32(h23, 1), s)));
}

static void fill(float *out, int w, int h, float freq, int32_t seed) {
    const float32x4_t Cx = vdupq_n_f32(0.211324865405f);
    const float32x4_t Cy = vdupq_n_f32(0.366025403784f);
    const float32x4_t Ch = vdupq_n_f32(0.5f);
    const float32x4_t C1 = vdupq_n_f32(1.0f);
    const float32x4_t Cx2 = vdupq_n_f32(0.42264973081f);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const int32x4_t one_i = vdupq_n_s32(1);
    const float32x4_t Cn_half = vdupq_n_f32(46.432f);  /* 92.864 * 0.5 */
    const uint32x4_t sv = vdupq_n_u32((uint32_t)seed * 0x45d9f3bu);
    const float32x4_t C = vdupq_n_f32(3.14159265f/64.0f);
    const float32x4_t pi = vdupq_n_f32(3.14159265f);
    const float32x4_t tpi2 = vdupq_n_f32(19.7392088f);
    const uint32x4_t s = vdupq_n_u32(0x80000000u);

    /* Fused hash+gradient+attenuation: issue vrecpeq early, hide latency */
    #define HASHED_CORNER(hh, dx, dy, n) do { \
        float32x4_t _off = vcvtq_f32_u32(vandq_u32((hh), vdupq_n_u32(0x3f))); \
        float32x4_t _v   = vmulq_f32(_off, C); \
        float32x4_t _u   = vsubq_f32(pi, _v); \
        float32x4_t _u2  = vmulq_f32(_u, _u); \
        float32x4_t _v2  = vmulq_f32(_v, _v); \
        /* Issue both vrecpeq FIRST — 4 cycle latency */ \
        float32x4_t _rca = vrecpeq_f32(vsubq_f32(tpi2, _u2)); \
        float32x4_t _rsa = vrecpeq_f32(vsubq_f32(tpi2, _v2)); \
        /* Independent work while frecpe in flight */ \
        float32x4_t _d = vmlaq_f32(vmulq_f32((dx), (dx)), (dy), (dy)); \
        float32x4_t _m = vmaxq_f32(vsubq_f32(Ch, _d), zero); \
        _m = vmulq_f32(_m, _m); \
        _m = vmulq_f32(_m, _m); \
        /* Consume reciprocal results */ \
        float32x4_t _ca = vmulq_f32(_u2, _rca); \
        float32x4_t _sa = vmulq_f32(_v2, _rsa); \
        uint32x4_t _h23 = vshlq_n_u32((hh), 23); \
        _ca = vreinterpretq_f32_u32(veorq_u32( \
            vreinterpretq_u32_f32(_ca), \
            vandq_u32(veorq_u32(_h23, vshlq_n_u32(_h23, 2)), s))); \
        _sa = vreinterpretq_f32_u32(veorq_u32( \
            vreinterpretq_u32_f32(_sa), \
            vandq_u32(vshlq_n_u32(_h23, 1), s))); \
        /* Dot + accumulate */ \
        (n) = vmlaq_f32((n), _m, \
            vmlaq_f32(vmulq_f32(_ca, (dx)), _sa, (dy))); \
    } while(0)

    for (int py = 0; py < h; py++) {
        float *row = out + (size_t)py * w;
        float32x4_t vsy = vdupq_n_f32(py * freq);
        int px = 0;
        for (; px + 7 < w; px += 8) {
            float32x4_t vx0 = {px*freq,(px+1)*freq,(px+2)*freq,(px+3)*freq};
            float32x4_t vx1 = vaddq_f32(vx0, vdupq_n_f32(4.0f*freq));
            float32x4_t s0=vmulq_f32(vaddq_f32(vx0,vsy),Cy), s1=vmulq_f32(vaddq_f32(vx1,vsy),Cy);
            float32x4_t iv0=vrndmq_f32(vaddq_f32(vx0,s0)), jv0=vrndmq_f32(vaddq_f32(vsy,s0));
            float32x4_t iv1=vrndmq_f32(vaddq_f32(vx1,s1)), jv1=vrndmq_f32(vaddq_f32(vsy,s1));
            float32x4_t t0=vmulq_f32(vaddq_f32(iv0,jv0),Cx), t1=vmulq_f32(vaddq_f32(iv1,jv1),Cx);
            float32x4_t x00=vsubq_f32(vx0,vsubq_f32(iv0,t0)), y00=vsubq_f32(vsy,vsubq_f32(jv0,t0));
            float32x4_t x01=vsubq_f32(vx1,vsubq_f32(iv1,t1)), y01=vsubq_f32(vsy,vsubq_f32(jv1,t1));
            const int32x4_t zero_i = vdupq_n_s32(0);
            uint32x4_t split0 = vcgtq_f32(x00,y00);
            uint32x4_t split1 = vcgtq_f32(x01,y01);
            float32x4_t i1x0=vbslq_f32(split0,C1,zero);
            float32x4_t i1y0=vbslq_f32(split0,zero,C1);
            float32x4_t i1x1=vbslq_f32(split1,C1,zero);
            float32x4_t i1y1=vbslq_f32(split1,zero,C1);
            int32x4_t xi0=vcvtq_s32_f32(iv0), yi0=vcvtq_s32_f32(jv0);
            int32x4_t xi1=vcvtq_s32_f32(iv1), yi1=vcvtq_s32_f32(jv1);
            int32x4_t i1xi0=vreinterpretq_s32_u32(vandq_u32(split0, vdupq_n_u32(1)));
            int32x4_t i1yi0=vreinterpretq_s32_u32(vandq_u32(vmvnq_u32(split0), vdupq_n_u32(1)));
            int32x4_t i1xi1=vreinterpretq_s32_u32(vandq_u32(split1, vdupq_n_u32(1)));
            int32x4_t i1yi1=vreinterpretq_s32_u32(vandq_u32(vmvnq_u32(split1), vdupq_n_u32(1)));
            float32x4_t n0=zero, n1=zero;

            /* Corner 0: shared */
            {uint32x4_t h1=hash2d4(xi1,yi1,sv);
             HASHED_CORNER(h1, x00, y00, n0);
             HASHED_CORNER(h1, x01, y01, n1);}
            /* Corner 1: separate */
            {float32x4_t x10=vaddq_f32(vsubq_f32(x00,i1x0),Cx), y10=vaddq_f32(vsubq_f32(y00,i1y0),Cx);
             float32x4_t x11=vaddq_f32(vsubq_f32(x01,i1x1),Cx), y11=vaddq_f32(vsubq_f32(y01,i1y1),Cx);
             uint32x4_t h0=hash2d4(vaddq_s32(xi0,i1xi0),vaddq_s32(yi0,i1yi0),sv);
             uint32x4_t h1=hash2d4(vaddq_s32(xi1,i1xi1),vaddq_s32(yi1,i1yi1),sv);
             HASHED_CORNER(h0, x10, y10, n0);
             HASHED_CORNER(h1, x11, y11, n1);}
            /* Corner 2: shared */
            {float32x4_t x20=vaddq_f32(vsubq_f32(x00,C1),Cx2), y20=vaddq_f32(vsubq_f32(y00,C1),Cx2);
             float32x4_t x21=vaddq_f32(vsubq_f32(x01,C1),Cx2), y21=vaddq_f32(vsubq_f32(y01,C1),Cx2);
             uint32x4_t h0=hash2d4(vaddq_s32(xi0,one_i),vaddq_s32(yi0,one_i),sv);
             HASHED_CORNER(h0, x20, y20, n0);
             HASHED_CORNER(h0, x21, y21, n1);}

            n0=vmlaq_f32(Ch,n0,Cn_half);
            n1=vmlaq_f32(Ch,n1,Cn_half);
            vst1q_f32(row + px, n0);
            vst1q_f32(row + px + 4, n1);
        }
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
    #undef HASHED_CORNER
}

int main(int argc, char **argv) {
    int w=1024, h=1024; float freq=0.04f; int seed=42;
    const char *out = "mountain_noise.pgm";
    if (argc>1) w=atoi(argv[1]); if (argc>2) h=atoi(argv[2]);
    if (argc>3) freq=atof(argv[3]); if (argc>4) seed=atoi(argv[4]);
    if (argc>5) out=argv[5];
    float *buf = malloc(w*h*sizeof(float));
    fill(buf, w, h, freq, seed); /* warm up */
    struct timespec t0,t1;
    clock_gettime(CLOCK_MONOTONIC,&t0);
    volatile float sink = 0;
    for (int i=0; i<100; i++) { fill(buf, w, h, freq, seed); sink += buf[0]; }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double dt=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
    printf("%dx%d freq=%.2f seed=%d → %.3f ms  %.0f Mpixels/s\n",
           w, h, freq, seed, dt/100*1000, (double)w*h/(dt/100)/1e6);
    fill(buf, w, h, freq, seed);
    FILE *f=fopen(out,"wb"); fprintf(f,"P5\n%d %d\n255\n",w,h);
    for (int i=0;i<w*h;i++) { int v=buf[i]*255; fputc(v<0?0:v>255?255:v,f); }
    fclose(f); printf("  saved → %s\n", out);
    free(buf);
}
