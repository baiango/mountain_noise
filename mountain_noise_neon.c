/* mountain_noise_standalone.c — Standalone peak noise generator
 *
 * Compile: cc -O3 -march=native -ffast-math -o mountain_noise mountain_noise_standalone.c
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
    const float32x4_t Ch = vdupq_n_f32(0.5f), C1 = vdupq_n_f32(1.0f);
    const float32x4_t Cn = vdupq_n_f32(92.864f);
    const float32x4_t Cx2 = vdupq_n_f32(0.42264973081f);
    const uint32x4_t sv = vdupq_n_u32((uint32_t)seed * 0x45d9f3bu);
    for (int py = 0; py < h; py++) {
        float32x4_t vsy = vdupq_n_f32(py * freq);
        int px = 0;
        for (; px + 7 < w; px += 8) {
            float32x4_t vx0 = {px*freq,(px+1)*freq,(px+2)*freq,(px+3)*freq};
            float32x4_t vx1 = {(px+4)*freq,(px+5)*freq,(px+6)*freq,(px+7)*freq};
            float32x4_t s0=vmulq_f32(vaddq_f32(vx0,vsy),Cy), s1=vmulq_f32(vaddq_f32(vx1,vsy),Cy);
            float32x4_t iv0=vrndmq_f32(vaddq_f32(vx0,s0)), jv0=vrndmq_f32(vaddq_f32(vsy,s0));
            float32x4_t iv1=vrndmq_f32(vaddq_f32(vx1,s1)), jv1=vrndmq_f32(vaddq_f32(vsy,s1));
            float32x4_t t0=vmulq_f32(vaddq_f32(iv0,jv0),Cx), t1=vmulq_f32(vaddq_f32(iv1,jv1),Cx);
            float32x4_t x00=vsubq_f32(vx0,vsubq_f32(iv0,t0)), y00=vsubq_f32(vsy,vsubq_f32(jv0,t0));
            float32x4_t x01=vsubq_f32(vx1,vsubq_f32(iv1,t1)), y01=vsubq_f32(vsy,vsubq_f32(jv1,t1));
            float32x4_t i1x0=vbslq_f32(vcgtq_f32(x00,y00),C1,vdupq_n_f32(0));
            float32x4_t i1y0=vsubq_f32(C1,i1x0);
            float32x4_t i1x1=vbslq_f32(vcgtq_f32(x01,y01),C1,vdupq_n_f32(0));
            float32x4_t i1y1=vsubq_f32(C1,i1x1);
            int32x4_t xi0=vcvtq_s32_f32(iv0), yi0=vcvtq_s32_f32(jv0);
            int32x4_t xi1=vcvtq_s32_f32(iv1), yi1=vcvtq_s32_f32(jv1);
            float32x4_t n0=vdupq_n_f32(0), n1=vdupq_n_f32(0);
            #define CORNER(dx,dy,h,n) { \
                float32x4_t _d=vmlaq_f32(vmulq_f32(dx,dx),dy,dy); \
                float32x4_t _m=vmaxq_f32(vsubq_f32(Ch,_d),vdupq_n_f32(0)); \
                _m=vmulq_f32(_m,_m); _m=vmulq_f32(_m,_m); \
                n=vmlaq_f32(n,_m,vmlaq_f32(vmulq_f32(gx,dx),gy,dy)); }
            {float32x4_t gx,gy; uint32x4_t h0=hash2d4(xi0,yi0,sv),h1=hash2d4(xi1,yi1,sv);
             grad(h0,&gx,&gy); float32x4_t gx0=gx,gy0=gy; grad(h1,&gx,&gy);
             CORNER(x00,y00,h0,n0) float32x4_t gx_=gx,gy_=gy;
             CORNER(x01,y01,h1,n1)}
            {float32x4_t x10=vaddq_f32(vsubq_f32(x00,i1x0),Cx), y10=vaddq_f32(vsubq_f32(y00,i1y0),Cx);
             float32x4_t x11=vaddq_f32(vsubq_f32(x01,i1x1),Cx), y11=vaddq_f32(vsubq_f32(y01,i1y1),Cx);
             uint32x4_t h0=hash2d4(vaddq_s32(xi0,vcvtq_s32_f32(i1x0)),vaddq_s32(yi0,vcvtq_s32_f32(i1y0)),sv);
             uint32x4_t h1=hash2d4(vaddq_s32(xi1,vcvtq_s32_f32(i1x1)),vaddq_s32(yi1,vcvtq_s32_f32(i1y1)),sv);
             float32x4_t gx,gy; grad(h0,&gx,&gy); CORNER(x10,y10,h0,n0)
             grad(h1,&gx,&gy); CORNER(x11,y11,h1,n1)}
            {float32x4_t x20=vaddq_f32(vsubq_f32(x00,C1),Cx2), y20=vaddq_f32(vsubq_f32(y00,C1),Cx2);
             float32x4_t x21=vaddq_f32(vsubq_f32(x01,C1),Cx2), y21=vaddq_f32(vsubq_f32(y01,C1),Cx2);
             uint32x4_t h0=hash2d4(vaddq_s32(xi0,vdupq_n_s32(1)),vaddq_s32(yi0,vdupq_n_s32(1)),sv);
             uint32x4_t h1=hash2d4(vaddq_s32(xi1,vdupq_n_s32(1)),vaddq_s32(yi1,vdupq_n_s32(1)),sv);
             float32x4_t gx,gy; grad(h0,&gx,&gy); CORNER(x20,y20,h0,n0)
             grad(h1,&gx,&gy); CORNER(x21,y21,h1,n1)}
            #undef CORNER
            n0=vmulq_f32(n0,Cn); n0=vmlaq_f32(Ch,n0,Ch);
            n1=vmulq_f32(n1,Cn); n1=vmlaq_f32(Ch,n1,Ch);
            vst1q_f32(&out[py*w+px], n0);
            vst1q_f32(&out[py*w+px+4], n1);
        }
        for (; px < w; px++) {
            float sx=px*freq, sy=py*freq, s2=(sx+sy)*0.366025403784f;
            float fi=__builtin_floorf(sx+s2), fj=__builtin_floorf(sy+s2);
            float t2=(fi+fj)*0.211324865405f, x0=sx-(fi-t2), y0=sy-(fj-t2);
            int i1=x0>y0, j1=!i1;
            float x1=x0-i1+0.211324865405f, y1=y0-j1+0.211324865405f;
            float x2=x0-1+0.42264973081f, y2=y0-1+0.42264973081f;
            uint32_t su=(uint32_t)seed*0x45d9f3bu;
            #define H(a,b) ({uint32_t _h=(uint32_t)(a);_h*=0x9E3779B9u;_h^=(uint32_t)(b)^su;_h^=_h>>16;_h*=0x85ebca6bu;_h^=_h>>13;_h;})
            #define GG(hh,dx,dy) ({uint32_t _hh=(hh);float _f=(_hh&0x3F)*(3.14159265f/64.0f);float _v=2*_f,_u=3.14159265f-_v;float _u2=_u*_u,_v2=_v*_v,_t=19.7392088f;float _ca=_u2/(_t-_u2),_sa=_v2/(_t-_v2);uint32_t _h23=_hh<<23;if(_h23^(_h23<<2))_ca=-_ca;if(_h23<<1)_sa=-_sa;float _m=fmaxf(.5f-((dx)*(dx)+(dy)*(dy)),0);_m*=_m;_m*=_m;_m*(_ca*(dx)+_sa*(dy));})
            out[py*w+px]=(GG(H((int)fi,(int)fj),x0,y0)+GG(H((int)fi+i1,(int)fj+j1),x1,y1)+GG(H((int)fi+1,(int)fj+1),x2,y2))*92.864f*0.5f+0.5f;
            #undef H
            #undef GG
        }
    }
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
