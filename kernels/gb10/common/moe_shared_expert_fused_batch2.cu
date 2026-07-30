// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Fused MoE Expert+Shared GEMV — K=2 multi-token batch variant.
// Dual-token amortize (2026-07-24): grid y = top_k+1 (not 2*(top_k+1)).
// Shared expert always loads weights once for both tokens. Routed slot s
// dual-accumulates when expert_indices[s]==expert_indices[top_k+s].

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 128
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_BATCH2[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float atlas_dec_e4m3(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)               v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else                       v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}
#else
__device__ __forceinline__ float atlas_dec_e4m3(unsigned char b) {
    __nv_fp8_e4m3 f; *(unsigned char*)&f = b; return (float)f;
}
#endif

// Grid: (ceil(N/8), top_k+1, 2)  Block: (128,1,1)
// y: 0..top_k-1 routed slot; y==top_k shared dual-token
extern "C" __global__ void moe_expert_gate_up_shared_batch2(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,
    const unsigned char* __restrict__ sh_gate_packed,
    const unsigned char* __restrict__ sh_gate_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sh_up_packed,
    const unsigned char* __restrict__ sh_up_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int y = blockIdx.y;
    const unsigned int proj = blockIdx.z;
    const bool is_shared = (y == top_k);

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2u) + local_out * 2u;
    const unsigned int n2o = n1 + 1u;
    if (n1 >= N) return;
    const bool have_n2 = (n2o < N);

    const unsigned int half_K = K / 2u;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8u;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_BATCH2[threadIdx.x];
    __syncthreads();

    const __nv_bfloat16* A0 = A;
    const __nv_bfloat16* A1 = A + K;

    // Resolve weights + outputs for token0 and token1
    const unsigned char *Bp0 = 0, *Bs0 = 0, *Bp1 = 0, *Bs1 = 0;
    float s20 = 0.f, s21 = 0.f;
    __nv_bfloat16 *C0 = 0, *C1 = 0;
    bool same = false;

    if (is_shared) {
        if (proj == 0) {
            Bp0 = Bp1 = sh_gate_packed; Bs0 = Bs1 = sh_gate_scale; s20 = s21 = sh_gate_s2;
            C0 = sh_gate_out; C1 = sh_gate_out + N;
        } else {
            Bp0 = Bp1 = sh_up_packed; Bs0 = Bs1 = sh_up_scale; s20 = s21 = sh_up_s2;
            C0 = sh_up_out; C1 = sh_up_out + N;
        }
        same = true;
    } else {
        const unsigned int slot = y;
        const unsigned int e0 = expert_indices[slot];
        const unsigned int e1 = expert_indices[top_k + slot];
        same = (e0 == e1);
        if (proj == 0) {
            Bp0 = (const unsigned char*)gate_packed_ptrs[e0];
            Bs0 = (const unsigned char*)gate_scale_ptrs[e0];
            s20 = gate_scale2_vals[e0];
            Bp1 = (const unsigned char*)gate_packed_ptrs[e1];
            Bs1 = (const unsigned char*)gate_scale_ptrs[e1];
            s21 = gate_scale2_vals[e1];
            C0 = gate_out + (unsigned long long)slot * N;
            C1 = gate_out + (unsigned long long)(top_k + slot) * N;
        } else {
            Bp0 = (const unsigned char*)up_packed_ptrs[e0];
            Bs0 = (const unsigned char*)up_scale_ptrs[e0];
            s20 = up_scale2_vals[e0];
            Bp1 = (const unsigned char*)up_packed_ptrs[e1];
            Bs1 = (const unsigned char*)up_scale_ptrs[e1];
            s21 = up_scale2_vals[e1];
            C0 = up_out + (unsigned long long)slot * N;
            C1 = up_out + (unsigned long long)(top_k + slot) * N;
        }
    }

    // zero helper
    #define ZERO_C(Cptr) do { \
        const unsigned int nb = blockIdx.x * (N_PER_BLOCK * 2u); \
        for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2u && nb + i < N; i += BLOCK_SIZE) \
            (Cptr)[nb + i] = __float2bfloat16(0.0f); \
    } while (0)

    if (same) {
        if (Bp0 == 0) { ZERO_C(C0); ZERO_C(C1); return; }
        float a01=0,a02=0,a11=0,a12=0;
        for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
            uint4 q0 = ((const uint4*)A0)[k8];
            uint4 q1 = ((const uint4*)A1)[k8];
            const unsigned int r0[4]={q0.x,q0.y,q0.z,q0.w};
            const unsigned int r1[4]={q1.x,q1.y,q1.z,q1.w};
            const unsigned int base_k = k8 * 8u;
            unsigned int p1 = *(const unsigned int*)(Bp0 + (unsigned long long)n1 * half_K + k8 * 4u);
            unsigned int sg = base_k / GROUP_SIZE;
            float sc1 = atlas_dec_e4m3(Bs0[(unsigned long long)n1 * num_groups + sg]) * s20;
            unsigned int p2 = have_n2 ? *(const unsigned int*)(Bp0 + (unsigned long long)n2o * half_K + k8 * 4u) : 0u;
            float sc2 = have_n2 ? atlas_dec_e4m3(Bs0[(unsigned long long)n2o * num_groups + sg]) * s20 : 0.f;
            #pragma unroll
            for (int b=0;b<4;b++) {
                unsigned char bv1=(p1>>(b*8))&0xFF, bv2=(p2>>(b*8))&0xFF;
                float w1l=s_lut[bv1&0xF]*sc1, w1h=s_lut[bv1>>4]*sc1;
                float w2l=s_lut[bv2&0xF]*sc2, w2h=s_lut[bv2>>4]*sc2;
                __nv_bfloat16 al0,ah0,al1,ah1;
                *(unsigned short*)&al0=(unsigned short)(r0[b]&0xFFFF);
                *(unsigned short*)&ah0=(unsigned short)(r0[b]>>16);
                *(unsigned short*)&al1=(unsigned short)(r1[b]&0xFFFF);
                *(unsigned short*)&ah1=(unsigned short)(r1[b]>>16);
                float f0l=__bfloat162float(al0), f0h=__bfloat162float(ah0);
                float f1l=__bfloat162float(al1), f1h=__bfloat162float(ah1);
                a01 += f0l*w1l + f0h*w1h; a02 += f0l*w2l + f0h*w2h;
                a11 += f1l*w1l + f1h*w1h; a12 += f1l*w2l + f1h*w2h;
            }
        }
        #pragma unroll
        for (int o=WARP_SIZE/2;o>0;o>>=1) {
            a01+=__shfl_down_sync(0xffffffffu,a01,o);
            a11+=__shfl_down_sync(0xffffffffu,a11,o);
            if (have_n2){a02+=__shfl_down_sync(0xffffffffu,a02,o);a12+=__shfl_down_sync(0xffffffffu,a12,o);}
        }
        if (lane==0) {
            C0[n1]=__float2bfloat16(a01); C1[n1]=__float2bfloat16(a11);
            if (have_n2){C0[n2o]=__float2bfloat16(a02); C1[n2o]=__float2bfloat16(a12);}
        }
    } else {
        // sequential two experts
        for (int t=0;t<2;t++) {
            const unsigned char* Bp = (t==0)?Bp0:Bp1;
            const unsigned char* Bs = (t==0)?Bs0:Bs1;
            float s2v = (t==0)?s20:s21;
            __nv_bfloat16* Cout = (t==0)?C0:C1;
            const __nv_bfloat16* At = (t==0)?A0:A1;
            if (Bp==0) { ZERO_C(Cout); continue; }
            float acc1=0, acc2=0;
            for (unsigned int k8=lane; k8<K8; k8+=threads_per_out) {
                uint4 a_data=((const uint4*)At)[k8];
                const unsigned int ar[4]={a_data.x,a_data.y,a_data.z,a_data.w};
                const unsigned int base_k=k8*8u;
                unsigned int p1=*(const unsigned int*)(Bp+(unsigned long long)n1*half_K+k8*4u);
                unsigned int sg=base_k/GROUP_SIZE;
                float sc1=atlas_dec_e4m3(Bs[(unsigned long long)n1*num_groups+sg])*s2v;
                unsigned int p2=have_n2?*(const unsigned int*)(Bp+(unsigned long long)n2o*half_K+k8*4u):0u;
                float sc2=have_n2?atlas_dec_e4m3(Bs[(unsigned long long)n2o*num_groups+sg])*s2v:0.f;
                #pragma unroll
                for (int b=0;b<4;b++) {
                    unsigned char bv1=(p1>>(b*8))&0xFF, bv2=(p2>>(b*8))&0xFF;
                    float w1l=s_lut[bv1&0xF]*sc1, w1h=s_lut[bv1>>4]*sc1;
                    float w2l=s_lut[bv2&0xF]*sc2, w2h=s_lut[bv2>>4]*sc2;
                    __nv_bfloat16 al,ah;
                    *(unsigned short*)&al=(unsigned short)(ar[b]&0xFFFF);
                    *(unsigned short*)&ah=(unsigned short)(ar[b]>>16);
                    float afl=__bfloat162float(al), afh=__bfloat162float(ah);
                    acc1+=afl*w1l+afh*w1h; acc2+=afl*w2l+afh*w2h;
                }
            }
            #pragma unroll
            for (int o=WARP_SIZE/2;o>0;o>>=1) acc1+=__shfl_down_sync(0xffffffffu,acc1,o);
            if (lane==0) Cout[n1]=__float2bfloat16(acc1);
            if (have_n2) {
                #pragma unroll
                for (int o=WARP_SIZE/2;o>0;o>>=1) acc2+=__shfl_down_sync(0xffffffffu,acc2,o);
                if (lane==0) Cout[n2o]=__float2bfloat16(acc2);
            }
        }
    }
#undef ZERO_C
}

// Grid: (ceil(N/8), top_k+1, 1)  N=hidden, K=inter
extern "C" __global__ void moe_expert_silu_down_shared_batch2(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    const unsigned char* __restrict__ sh_down_packed,
    const unsigned char* __restrict__ sh_down_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int y = blockIdx.y;
    const bool is_shared = (y == top_k);

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2u) + local_out * 2u;
    const unsigned int n2o = n1 + 1u;
    if (n1 >= N) return;
    const bool have_n2 = (n2o < N);

    const unsigned int half_K = K / 2u;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8u;

    __shared__ float s_lut[16];
    extern __shared__ float s_act[]; // [2*K]: token0 | token1

    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_BATCH2[threadIdx.x];

    const __nv_bfloat16 *g0,*u0,*g1,*u1;
    const unsigned char *Bp0=0,*Bs0=0,*Bp1=0,*Bs1=0;
    float s20=0.f,s21=0.f;
    __nv_bfloat16 *out0,*out1;
    bool same=false;

    if (is_shared) {
        g0=sh_gate_in; u0=sh_up_in; g1=sh_gate_in+K; u1=sh_up_in+K;
        Bp0=Bp1=sh_down_packed; Bs0=Bs1=sh_down_scale; s20=s21=sh_down_s2;
        out0=sh_down_out; out1=sh_down_out+N; same=true;
    } else {
        const unsigned int slot=y;
        const unsigned int e0=expert_indices[slot], e1=expert_indices[top_k+slot];
        g0=gate_out+(unsigned long long)slot*K; u0=up_out+(unsigned long long)slot*K;
        g1=gate_out+(unsigned long long)(top_k+slot)*K; u1=up_out+(unsigned long long)(top_k+slot)*K;
        out0=C+(unsigned long long)slot*N; out1=C+(unsigned long long)(top_k+slot)*N;
        Bp0=(const unsigned char*)packed_ptrs[e0]; Bs0=(const unsigned char*)scale_ptrs[e0]; s20=scale2_vals[e0];
        Bp1=(const unsigned char*)packed_ptrs[e1]; Bs1=(const unsigned char*)scale_ptrs[e1]; s21=scale2_vals[e1];
        same=(e0==e1);
    }

    for (unsigned int i=threadIdx.x; i<K; i+=BLOCK_SIZE) {
        float gf0=__bfloat162float(g0[i]), uf0=__bfloat162float(u0[i]);
        s_act[i]=(gf0/(1.f+__expf(-gf0)))*uf0;
        float gf1=__bfloat162float(g1[i]), uf1=__bfloat162float(u1[i]);
        s_act[K+i]=(gf1/(1.f+__expf(-gf1)))*uf1;
    }
    __syncthreads();

    #define ZERO_O(op) do { \
        const unsigned int nb=blockIdx.x*(N_PER_BLOCK*2u); \
        for (unsigned int i=threadIdx.x; i<N_PER_BLOCK*2u && nb+i<N; i+=BLOCK_SIZE) \
            (op)[nb+i]=__float2bfloat16(0.f); \
    } while(0)

    if (same) {
        if (Bp0==0) { ZERO_O(out0); ZERO_O(out1); return; }
        float a01=0,a02=0,a11=0,a12=0;
        for (unsigned int k8=lane; k8<K8; k8+=threads_per_out) {
            const unsigned int base_k=k8*8u;
            unsigned int p1=*(const unsigned int*)(Bp0+(unsigned long long)n1*half_K+k8*4u);
            unsigned int sg=base_k/GROUP_SIZE;
            float sc1=atlas_dec_e4m3(Bs0[(unsigned long long)n1*num_groups+sg])*s20;
            unsigned int p2=have_n2?*(const unsigned int*)(Bp0+(unsigned long long)n2o*half_K+k8*4u):0u;
            float sc2=have_n2?atlas_dec_e4m3(Bs0[(unsigned long long)n2o*num_groups+sg])*s20:0.f;
            #pragma unroll
            for (int b=0;b<4;b++) {
                float a0l=s_act[base_k+b*2], a0h=s_act[base_k+b*2+1];
                float a1l=s_act[K+base_k+b*2], a1h=s_act[K+base_k+b*2+1];
                unsigned char bv1=(p1>>(b*8))&0xFF, bv2=(p2>>(b*8))&0xFF;
                float w1l=s_lut[bv1&0xF]*sc1, w1h=s_lut[bv1>>4]*sc1;
                float w2l=s_lut[bv2&0xF]*sc2, w2h=s_lut[bv2>>4]*sc2;
                a01+=a0l*w1l+a0h*w1h; a02+=a0l*w2l+a0h*w2h;
                a11+=a1l*w1l+a1h*w1h; a12+=a1l*w2l+a1h*w2h;
            }
        }
        #pragma unroll
        for (int o=WARP_SIZE/2;o>0;o>>=1) {
            a01+=__shfl_down_sync(0xffffffffu,a01,o);
            a11+=__shfl_down_sync(0xffffffffu,a11,o);
            if(have_n2){a02+=__shfl_down_sync(0xffffffffu,a02,o);a12+=__shfl_down_sync(0xffffffffu,a12,o);}
        }
        if (lane==0) {
            out0[n1]=__float2bfloat16(a01); out1[n1]=__float2bfloat16(a11);
            if(have_n2){out0[n2o]=__float2bfloat16(a02); out1[n2o]=__float2bfloat16(a12);}
        }
    } else {
        for (int t=0;t<2;t++) {
            const unsigned char* Bp=(t==0)?Bp0:Bp1;
            const unsigned char* Bs=(t==0)?Bs0:Bs1;
            float s2v=(t==0)?s20:s21;
            __nv_bfloat16* out=(t==0)?out0:out1;
            const float* act=(t==0)?s_act:(s_act+K);
            if (Bp==0) { ZERO_O(out); continue; }
            float acc1=0,acc2=0;
            for (unsigned int k8=lane; k8<K8; k8+=threads_per_out) {
                const unsigned int base_k=k8*8u;
                unsigned int p1=*(const unsigned int*)(Bp+(unsigned long long)n1*half_K+k8*4u);
                unsigned int sg=base_k/GROUP_SIZE;
                float sc1=atlas_dec_e4m3(Bs[(unsigned long long)n1*num_groups+sg])*s2v;
                unsigned int p2=have_n2?*(const unsigned int*)(Bp+(unsigned long long)n2o*half_K+k8*4u):0u;
                float sc2=have_n2?atlas_dec_e4m3(Bs[(unsigned long long)n2o*num_groups+sg])*s2v:0.f;
                #pragma unroll
                for (int b=0;b<4;b++) {
                    float al=act[base_k+b*2], ah=act[base_k+b*2+1];
                    unsigned char bv1=(p1>>(b*8))&0xFF, bv2=(p2>>(b*8))&0xFF;
                    float w1l=s_lut[bv1&0xF]*sc1, w1h=s_lut[bv1>>4]*sc1;
                    float w2l=s_lut[bv2&0xF]*sc2, w2h=s_lut[bv2>>4]*sc2;
                    acc1+=al*w1l+ah*w1h; acc2+=al*w2l+ah*w2h;
                }
            }
            #pragma unroll
            for (int o=WARP_SIZE/2;o>0;o>>=1) acc1+=__shfl_down_sync(0xffffffffu,acc1,o);
            if (lane==0) out[n1]=__float2bfloat16(acc1);
            if (have_n2) {
                #pragma unroll
                for (int o=WARP_SIZE/2;o>0;o>>=1) acc2+=__shfl_down_sync(0xffffffffu,acc2,o);
                if (lane==0) out[n2o]=__float2bfloat16(acc2);
            }
        }
    }
#undef ZERO_O
}

// Down projection for SwiGLU activations materialized once per route.
//
// Grid: (ceil(N/8), top_k+1, 1)  N=hidden, K=inter
extern "C" __global__ void moe_expert_down_shared_batch2_precomputed(
    const __nv_bfloat16* __restrict__ act,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    const __nv_bfloat16* __restrict__ sh_act,
    const unsigned char* __restrict__ sh_down_packed,
    const unsigned char* __restrict__ sh_down_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int y = blockIdx.y;
    const bool is_shared = (y == top_k);

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2u) + local_out * 2u;
    const unsigned int n2o = n1 + 1u;
    if (n1 >= N) return;
    const bool have_n2 = (n2o < N);

    const unsigned int half_K = K / 2u;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8u;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_BATCH2[threadIdx.x];
    __syncthreads();

    const __nv_bfloat16 *A0, *A1;
    const unsigned char *Bp0 = 0, *Bs0 = 0, *Bp1 = 0, *Bs1 = 0;
    float s20 = 0.f, s21 = 0.f;
    __nv_bfloat16 *out0, *out1;
    bool same;

    if (is_shared) {
        A0 = sh_act;
        A1 = sh_act + K;
        Bp0 = Bp1 = sh_down_packed;
        Bs0 = Bs1 = sh_down_scale;
        s20 = s21 = sh_down_s2;
        out0 = sh_down_out;
        out1 = sh_down_out + N;
        same = true;
    } else {
        const unsigned int slot = y;
        const unsigned int e0 = expert_indices[slot];
        const unsigned int e1 = expert_indices[top_k + slot];
        A0 = act + (unsigned long long)slot * K;
        A1 = act + (unsigned long long)(top_k + slot) * K;
        Bp0 = (const unsigned char*)packed_ptrs[e0];
        Bs0 = (const unsigned char*)scale_ptrs[e0];
        s20 = scale2_vals[e0];
        Bp1 = (const unsigned char*)packed_ptrs[e1];
        Bs1 = (const unsigned char*)scale_ptrs[e1];
        s21 = scale2_vals[e1];
        out0 = C + (unsigned long long)slot * N;
        out1 = C + (unsigned long long)(top_k + slot) * N;
        same = (e0 == e1);
    }

    #define ZERO_PRECOMPUTED_OUT(op) do { \
        const unsigned int nb = blockIdx.x * (N_PER_BLOCK * 2u); \
        for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2u && nb + i < N; i += BLOCK_SIZE) \
            (op)[nb + i] = __float2bfloat16(0.f); \
    } while (0)

    if (same) {
        if (Bp0 == 0) {
            ZERO_PRECOMPUTED_OUT(out0);
            ZERO_PRECOMPUTED_OUT(out1);
            return;
        }
        float a01 = 0.f, a02 = 0.f, a11 = 0.f, a12 = 0.f;
        for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
            uint4 q0 = ((const uint4*)A0)[k8];
            uint4 q1 = ((const uint4*)A1)[k8];
            const unsigned int r0[4] = {q0.x, q0.y, q0.z, q0.w};
            const unsigned int r1[4] = {q1.x, q1.y, q1.z, q1.w};
            const unsigned int base_k = k8 * 8u;
            unsigned int p1 = *(const unsigned int*)(
                Bp0 + (unsigned long long)n1 * half_K + k8 * 4u);
            const unsigned int sg = base_k / GROUP_SIZE;
            const float sc1 =
                atlas_dec_e4m3(Bs0[(unsigned long long)n1 * num_groups + sg]) * s20;
            unsigned int p2 = have_n2
                ? *(const unsigned int*)(
                    Bp0 + (unsigned long long)n2o * half_K + k8 * 4u)
                : 0u;
            const float sc2 = have_n2
                ? atlas_dec_e4m3(
                    Bs0[(unsigned long long)n2o * num_groups + sg]) * s20
                : 0.f;
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const unsigned char bv1 = (p1 >> (b * 8)) & 0xff;
                const unsigned char bv2 = (p2 >> (b * 8)) & 0xff;
                const float w1l = s_lut[bv1 & 0xf] * sc1;
                const float w1h = s_lut[bv1 >> 4] * sc1;
                const float w2l = s_lut[bv2 & 0xf] * sc2;
                const float w2h = s_lut[bv2 >> 4] * sc2;
                __nv_bfloat16 al0, ah0, al1, ah1;
                *(unsigned short*)&al0 = (unsigned short)(r0[b] & 0xffff);
                *(unsigned short*)&ah0 = (unsigned short)(r0[b] >> 16);
                *(unsigned short*)&al1 = (unsigned short)(r1[b] & 0xffff);
                *(unsigned short*)&ah1 = (unsigned short)(r1[b] >> 16);
                const float f0l = __bfloat162float(al0);
                const float f0h = __bfloat162float(ah0);
                const float f1l = __bfloat162float(al1);
                const float f1h = __bfloat162float(ah1);
                a01 += f0l * w1l + f0h * w1h;
                a02 += f0l * w2l + f0h * w2h;
                a11 += f1l * w1l + f1h * w1h;
                a12 += f1l * w2l + f1h * w2h;
            }
        }
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            a01 += __shfl_down_sync(0xffffffffu, a01, offset);
            a11 += __shfl_down_sync(0xffffffffu, a11, offset);
            if (have_n2) {
                a02 += __shfl_down_sync(0xffffffffu, a02, offset);
                a12 += __shfl_down_sync(0xffffffffu, a12, offset);
            }
        }
        if (lane == 0) {
            out0[n1] = __float2bfloat16(a01);
            out1[n1] = __float2bfloat16(a11);
            if (have_n2) {
                out0[n2o] = __float2bfloat16(a02);
                out1[n2o] = __float2bfloat16(a12);
            }
        }
    } else {
        for (int token = 0; token < 2; ++token) {
            const unsigned char* Bp = token == 0 ? Bp0 : Bp1;
            const unsigned char* Bs = token == 0 ? Bs0 : Bs1;
            const float s2 = token == 0 ? s20 : s21;
            const __nv_bfloat16* A = token == 0 ? A0 : A1;
            __nv_bfloat16* out = token == 0 ? out0 : out1;
            if (Bp == 0) {
                ZERO_PRECOMPUTED_OUT(out);
                continue;
            }
            float acc1 = 0.f, acc2 = 0.f;
            for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
                uint4 q = ((const uint4*)A)[k8];
                const unsigned int r[4] = {q.x, q.y, q.z, q.w};
                const unsigned int base_k = k8 * 8u;
                unsigned int p1 = *(const unsigned int*)(
                    Bp + (unsigned long long)n1 * half_K + k8 * 4u);
                const unsigned int sg = base_k / GROUP_SIZE;
                const float sc1 =
                    atlas_dec_e4m3(Bs[(unsigned long long)n1 * num_groups + sg]) * s2;
                unsigned int p2 = have_n2
                    ? *(const unsigned int*)(
                        Bp + (unsigned long long)n2o * half_K + k8 * 4u)
                    : 0u;
                const float sc2 = have_n2
                    ? atlas_dec_e4m3(
                        Bs[(unsigned long long)n2o * num_groups + sg]) * s2
                    : 0.f;
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const unsigned char bv1 = (p1 >> (b * 8)) & 0xff;
                    const unsigned char bv2 = (p2 >> (b * 8)) & 0xff;
                    const float w1l = s_lut[bv1 & 0xf] * sc1;
                    const float w1h = s_lut[bv1 >> 4] * sc1;
                    const float w2l = s_lut[bv2 & 0xf] * sc2;
                    const float w2h = s_lut[bv2 >> 4] * sc2;
                    __nv_bfloat16 al, ah;
                    *(unsigned short*)&al = (unsigned short)(r[b] & 0xffff);
                    *(unsigned short*)&ah = (unsigned short)(r[b] >> 16);
                    const float afl = __bfloat162float(al);
                    const float afh = __bfloat162float(ah);
                    acc1 += afl * w1l + afh * w1h;
                    acc2 += afl * w2l + afh * w2h;
                }
            }
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
            if (lane == 0) out[n1] = __float2bfloat16(acc1);
            if (have_n2) {
                #pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                    acc2 += __shfl_down_sync(0xffffffffu, acc2, offset);
                if (lane == 0) out[n2o] = __float2bfloat16(acc2);
            }
        }
    }

#undef ZERO_PRECOMPUTED_OUT
}

// ── Weighted sum + sigmoid blend — K=2 batch variant ──
//
// Combines routed expert outputs with shared expert via sigmoid gate.
// blockIdx.y = token index (0 or 1).
//
// Grid: (ceil(hidden/256), 2, 1)  Block: (256, 1, 1)
extern "C" __global__ void moe_weighted_sum_blend_batch2(
    __nv_bfloat16* __restrict__ output,              // [2, hidden] BF16
    const __nv_bfloat16* __restrict__ expert_out,    // [2*top_k, hidden] BF16
    const float* __restrict__ expert_weights,         // [2*top_k] f32
    const __nv_bfloat16* __restrict__ shared_out,    // [2, hidden] BF16
    const __nv_bfloat16* __restrict__ input,         // [2, K] BF16 (MoE input)
    const __nv_bfloat16* __restrict__ gate_weight,   // [1, K] BF16 (shared gate)
    unsigned int hidden,
    unsigned int top_k,
    unsigned int K
) {
    const unsigned int token = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane = tid % WARP_SIZE;

    // Per-token input pointer
    const __nv_bfloat16* my_input = input + (unsigned long long)token * K;
    const float* my_weights = expert_weights + token * top_k;
    const __nv_bfloat16* my_expert_out = expert_out + (unsigned long long)token * top_k * hidden;
    const __nv_bfloat16* my_shared_out = shared_out + (unsigned long long)token * hidden;
    __nv_bfloat16* my_output = output + (unsigned long long)token * hidden;

    // ── Phase 1: Compute gate scalar (dot product + sigmoid) ──
    // NULL gate_weight = ungated shared expert (DeepSeek-V4) → sigmoid=1.0
    // Missing this check was CUDA-700 on V4 EP MTP K2 (null deref).
    __shared__ float s_warp_sums[8];
    __shared__ float sigmoid_val;

    if (gate_weight == 0) {
        if (tid == 0) sigmoid_val = 1.0f;
        __syncthreads();
    } else {
        float dot_acc = 0.0f;
        unsigned int K8 = K / 8;
        for (unsigned int k8 = tid; k8 < K8; k8 += 256) {
            uint4 a_data = ((const uint4*)my_input)[k8];
            uint4 w_data = ((const uint4*)gate_weight)[k8];
            const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
            const unsigned int w_raw[4] = {w_data.x, w_data.y, w_data.z, w_data.w};

            #pragma unroll
            for (int b = 0; b < 4; b++) {
                __nv_bfloat16 a_lo, a_hi, w_lo, w_hi;
                *(unsigned short*)&a_lo = (unsigned short)(a_raw[b] & 0xFFFF);
                *(unsigned short*)&a_hi = (unsigned short)(a_raw[b] >> 16);
                *(unsigned short*)&w_lo = (unsigned short)(w_raw[b] & 0xFFFF);
                *(unsigned short*)&w_hi = (unsigned short)(w_raw[b] >> 16);
                dot_acc += __bfloat162float(a_lo) * __bfloat162float(w_lo);
                dot_acc += __bfloat162float(a_hi) * __bfloat162float(w_hi);
            }
        }

        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            dot_acc += __shfl_down_sync(0xFFFFFFFF, dot_acc, offset);
        }
        if (lane == 0) {
            s_warp_sums[warp_id] = dot_acc;
        }
        __syncthreads();

        if (tid == 0) {
            float gate_scalar = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; w++) {
                gate_scalar += s_warp_sums[w];
            }
            sigmoid_val = 1.0f / (1.0f + __expf(-gate_scalar));
        }
        __syncthreads();
    }

    // ── Phase 2: Weighted sum + blend ──
    unsigned int j = blockIdx.x * blockDim.x + tid;
    if (j >= hidden) return;

    float acc = 0.0f;
    for (unsigned int e = 0; e < top_k; e++) {
        acc += my_weights[e] * __bfloat162float(my_expert_out[(unsigned long long)e * hidden + j]);
    }
    acc += sigmoid_val * __bfloat162float(my_shared_out[j]);
    my_output[j] = __float2bfloat16(acc);
}
