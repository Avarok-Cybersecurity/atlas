// SPDX-License-Identifier: AGPL-3.0-only
//
// GPU Q4_K weight quantizer — faithful port of llama.cpp quantize_q4_K (null-imatrix path:
// make_qkx3_quants + make_qp_quants, default weights av_x+|x|). One thread per 256-weight
// superblock. Input bf16 [nrows, n_per_row] (n_per_row % 256 == 0), output GGML atlas_bq4k
// row-major [nrows][n_per_row/256] — the exact layout the vendored MMQ kernel reads.
// Validated bit-equivalent to ggml_quantize_chunk in test (q4k_quantize gate).
#include <cuda_bf16.h>
#include <cstdint>

#define QK_K 256
struct atlas_bq4k { uint16_t d; uint16_t dmin; uint8_t scales[12]; uint8_t qs[128]; }; // 144 B

__device__ __forceinline__ int nearest_int(float fval) {
    float val = fval + 12582912.f;
    int i = __float_as_int(val); // bit-reinterpret float->int (portable to MSVC/nvcc)
    return (i & 0x007fffff) - 0x00400000;
}
__device__ __forceinline__ float fmaxf_(float a, float b){ return a>b?a:b; }
__device__ __forceinline__ int imax_(int a,int b){return a>b?a:b;}
__device__ __forceinline__ int imin_(int a,int b){return a<b?a:b;}

// make_qkx3_quants(n, nmax, x, weights, L, &the_min, Laux, rmin, rdelta, nstep, use_mad=false)
__device__ float make_qkx3(int n, int nmax, const float* x, const float* w, uint8_t* L,
                           float* the_min, uint8_t* Laux, float rmin, float rdelta, int nstep) {
    float mn=x[0], mx=x[0], sum_w=w[0], sum_x=w[0]*x[0];
    for (int i=1;i<n;++i){ if(x[i]<mn)mn=x[i]; if(x[i]>mx)mx=x[i]; float wi=w[i]; sum_w+=wi; sum_x+=wi*x[i]; }
    if (mn>0) mn=0;
    if (mx<=mn){ for(int i=0;i<n;++i)L[i]=0; *the_min=-mn; return 0.f; }
    float iscale=nmax/(mx-mn), scale=1/iscale, best=0;
    for (int i=0;i<n;++i){ int l=nearest_int(iscale*(x[i]-mn)); L[i]=imax_(0,imin_(nmax,l));
        float diff=scale*L[i]+mn-x[i]; best+=w[i]*diff*diff; }
    if (nstep<1){ *the_min=-mn; return scale; }
    for (int is=0;is<=nstep;++is){
        iscale=(rmin+rdelta*is+nmax)/(mx-mn);
        float sl=0,sl2=0,sxl=0;
        for (int i=0;i<n;++i){ int l=imax_(0,imin_(nmax,nearest_int(iscale*(x[i]-mn)))); Laux[i]=l;
            float wi=w[i]; sl+=wi*l; sl2+=wi*l*l; sxl+=wi*l*x[i]; }
        float D=sum_w*sl2-sl*sl;
        if (D>0){
            float ts=(sum_w*sxl-sum_x*sl)/D, tm=(sl2*sum_x-sl*sxl)/D;
            if (tm>0){ tm=0; ts=sxl/sl2; }
            float mad=0;
            for (int i=0;i<n;++i){ float diff=ts*Laux[i]+tm-x[i]; mad+=w[i]*diff*diff; }
            if (mad<best){ for(int i=0;i<n;++i)L[i]=Laux[i]; best=mad; scale=ts; mn=tm; }
        }
    }
    *the_min=-mn; return scale;
}

// make_qp_quants(n, nmax=63, x, L, quant_weights=sw)
__device__ float make_qp(int n, int nmax, const float* x, uint8_t* L, const float* qw) {
    float mx=0; for(int i=0;i<n;++i) mx=fmaxf_(mx,x[i]);
    if (mx<1e-15f){ for(int i=0;i<n;++i)L[i]=0; return 0.f; }
    float iscale=nmax/mx;
    for(int i=0;i<n;++i) L[i]=nearest_int(iscale*x[i]);
    float scale=1/iscale, best=0;
    for(int i=0;i<n;++i){ float diff=x[i]-scale*L[i]; best+=qw[i]*diff*diff; }
    for(int is=-4;is<=4;++is){ if(is==0)continue;
        float isc=(0.1f*is+nmax)/mx, sc=1/isc, mse=0;
        for(int i=0;i<n;++i){ int l=imin_(nmax,nearest_int(isc*x[i])); float diff=x[i]-sc*l; mse+=qw[i]*diff*diff; }
        if (mse<best){ best=mse; iscale=isc; }
    }
    float sumlx=0,suml2=0;
    for(int i=0;i<n;++i){ int l=imin_(nmax,nearest_int(iscale*x[i])); L[i]=l; sumlx+=qw[i]*x[i]*l; suml2+=qw[i]*l*l; }
    for(int itry=0;itry<5;++itry){ int nch=0;
        for(int i=0;i<n;++i){ float wi=qw[i]; float slx=sumlx-wi*x[i]*L[i], sl2=suml2-wi*L[i]*L[i];
            if (slx>0&&sl2>0){ int nl=imin_(nmax,nearest_int(x[i]*sl2/slx));
                if (nl!=L[i]){ slx+=wi*x[i]*nl; sl2+=wi*nl*nl;
                    if (slx*slx*suml2 > sumlx*sumlx*sl2){ L[i]=nl; sumlx=slx; suml2=sl2; ++nch; } } } }
        if (!nch) break;
    }
    return suml2>0.f ? sumlx/suml2 : 0.f;
}

__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m){
    if (j<4){ *d=q[j]&63; *m=q[j+4]&63; }
    else { *d=(q[j+4]&0xF)|((q[j-4]>>6)<<4); *m=(q[j+4]>>4)|((q[j-0]>>6)<<4); }
}

extern "C" __global__ void q4k_quantize(const __nv_bfloat16* __restrict__ x, atlas_bq4k* __restrict__ y,
                                        int nrows, int n_per_row) {
    const long sb_per_row = n_per_row / QK_K;
    const long total = (long)nrows * sb_per_row;
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    const long row = t / sb_per_row, b = t % sb_per_row;
    const __nv_bfloat16* xb = x + row*(long)n_per_row + b*QK_K;

    float xf[QK_K]; uint8_t L[QK_K];
    float sum_x2=0;
    for (int l=0;l<QK_K;++l){ float v=__bfloat162float(xb[l]); xf[l]=v; sum_x2+=v*v; }
    float sigma2 = 2*sum_x2/QK_K, av_x = sqrtf(sigma2);

    float weights[32], scales[8], mins[8], sw[8]; uint8_t Laux[32], Ls[8], Lm[8];
    atlas_bq4k yb;
    for (int j=0;j<8;++j){
        float sumw=0;
        for (int l=0;l<32;++l){ float wl=av_x+fabsf(xf[32*j+l]); weights[l]=wl; sumw+=wl; }
        sw[j]=sumw;
        scales[j]=make_qkx3(32,15,xf+32*j,weights,L+32*j,&mins[j],Laux,-0.9f,0.05f,36);
    }
    float d_block=make_qp(8,63,scales,Ls,sw);
    float m_block=make_qp(8,63,mins,Lm,sw);
    for (int j=0;j<12;++j) yb.scales[j]=0;
    for (int j=0;j<8;++j){
        uint8_t ls=Ls[j], lm=Lm[j];
        if (j<4){ yb.scales[j]=ls; yb.scales[j+4]=lm; }
        else { yb.scales[j+4]=(ls&0xF)|((lm&0xF)<<4); yb.scales[j-4]|=((ls>>4)<<6); yb.scales[j-0]|=((lm>>4)<<6); }
    }
    yb.d=__half_as_ushort(__float2half(d_block)); yb.dmin=__half_as_ushort(__float2half(m_block));
    // requant L using packed scales
    for (int j=0;j<8;++j){
        uint8_t sc,m; get_scale_min_k4(j,yb.scales,&sc,&m);
        float d=__half2float(*(const __half*)&yb.d)*sc;
        if (d==0.f){ for(int ii=0;ii<32;++ii) L[32*j+ii]=0; continue; }
        float dm=__half2float(*(const __half*)&yb.dmin)*m;
        for (int ii=0;ii<32;++ii){ int l=nearest_int((xf[32*j+ii]+dm)/d); L[32*j+ii]=imax_(0,imin_(15,l)); }
    }
    for (int j=0,qi=0;j<QK_K;j+=64,qi+=32)
        for (int l=0;l<32;++l) yb.qs[qi+l]=L[j+l]|(L[j+l+32]<<4);
    y[t]=yb;
}
