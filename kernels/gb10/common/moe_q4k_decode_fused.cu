// SPDX-License-Identifier: AGPL-3.0-only
//
// Fused n=1 DECODE MoE GEMV for keep-packed GGUF Q4_K_M (Laguna-S-2.1).
// Replaces the grouped-MMQ arm's permute + q8_1-quantize + grouped gate/up +
// silu + grouped down at decode (te = n*top_k ~ 10). The grouped MMQ launches
// ~2048 CTAs (grid.z=256 experts) of which ~80 do work, each wasting 127/128 of
// a 128-row tensor-core tile on one real token.
//
// OCCUPANCY: a single block-per-slot fused kernel launches only ~te (~10) blocks
// and starves GB10's SMs (measured 3x SLOWER than the grouped path). So we tile
// the OUTPUT dimension across grid.y (many blocks) and split into TWO kernels —
// gate+up+silu (staging s_act to a global buffer) then down — mirroring the
// NVFP4 decode structure. Each block stages its operand row in shared memory and
// its warps each own one output channel (warp-reduce over K). Scalar dequant x
// bf16-FMA (decode is bandwidth-bound). CUDA-graph-legal: no alloc/sync.
//
// PARITY: dequant matches the in-repo bit-for-bit CPU reference
// dequant_gguf_bf16.cu (dequant_q4_k_to_bf16 :61-88, dequant_q6_k_to_bf16
// :93-127) + get_scale_min_k4 :28-37. SiLU replicates ops::silu_mul
// (moe_silu_mul.cu:28-31): clamp gate<=10, up in [-10,10], + BF16 round-trips of
// the grouped path's expert_gate_out/up_out/s_act buffers.
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define DEC_BLOCK 256
#define DEC_WARPS (DEC_BLOCK / 32)   // 8 warps => 8 output channels per block

__device__ __forceinline__ float dec_h2f(unsigned short bits) {
    return __half2float(__ushort_as_half(bits));
}
__device__ __forceinline__ float dec_bf16rt(float x) {
    return __bfloat162float(__float2bfloat16(x));
}
// == ggml get_scale_min_k4 (dequant_gguf_bf16.cu:28-37)
__device__ __forceinline__ void dec_scale_min_k4(int j, const unsigned char* q,
                                                 unsigned char* sc, unsigned char* mn) {
    if (j < 4) { *sc = q[j] & 63; *mn = q[j + 4] & 63; }
    else {
        *sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *mn = (q[j + 4] >> 4)   | ((q[j]     >> 6) << 4);
    }
}

// ── Kernel 1: gate+up+silu. grid=(total_expanded, ceil(inter/DEC_WARPS)),
// block=256. Each block stages the slot's activation row a[h] in shared, then
// its 8 warps each compute one gate output + one up output (A-reuse), clamp+silu,
// and write to gate_silu_out[slot*inter + n] (BF16, the arena expert_gate_out).
extern "C" __global__ void atlas_moe_q4k_decode_gate_up(
    const __nv_bfloat16* __restrict__ expert_input,   // [n, h] BF16
    const unsigned char* __restrict__ gate_base,      // expert0 Q4_K [E,inter,h/256]
    const unsigned char* __restrict__ up_base,        // expert0 Q4_K [E,inter,h/256]
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ sorted_expert_ids,
    __nv_bfloat16* __restrict__ gate_silu_out,        // [total_expanded, inter] BF16
    unsigned int h, unsigned int inter, unsigned int total_expanded)
{
    const unsigned int slot = blockIdx.x;
    if (slot >= total_expanded) return;
    const int e = sorted_expert_ids[slot];
    const int t = sorted_token_ids[slot];
    const unsigned int n = blockIdx.y * DEC_WARPS + (threadIdx.x >> 5); // output channel
    const unsigned int lane = threadIdx.x & 31;

    extern __shared__ float s_x[];   // [h] staged activation
    if (e < 0 || t < 0) {
        if (n < inter && lane == 0) gate_silu_out[(unsigned long long)slot * inter + n] = __float2bfloat16(0.0f);
        return;
    }
    const __nv_bfloat16* a_row = expert_input + (unsigned long long)t * h;
    for (unsigned int i = threadIdx.x; i < h; i += DEC_BLOCK)
        s_x[i] = __bfloat162float(a_row[i]);
    __syncthreads();
    if (n >= inter) return;

    const unsigned int gu_kblk = h / 256;
    const unsigned long long gu_row = (unsigned long long)e * inter * gu_kblk + (unsigned long long)n * gu_kblk;
    const unsigned char* grow = gate_base + gu_row * 144ULL;
    const unsigned char* urow = up_base   + gu_row * 144ULL;
    float acc_g = 0.f, acc_u = 0.f;
    for (unsigned int kb = 0; kb < gu_kblk; ++kb) {
        const unsigned char* gb = grow + (unsigned long long)kb * 144ULL;
        const unsigned char* ub = urow + (unsigned long long)kb * 144ULL;
        float gd = dec_h2f(*(const unsigned short*)(gb)), gdm = dec_h2f(*(const unsigned short*)(gb + 2));
        float ud = dec_h2f(*(const unsigned short*)(ub)), udm = dec_h2f(*(const unsigned short*)(ub + 2));
        const unsigned char* gsc = gb + 4; const unsigned char* gqs = gb + 16;
        const unsigned char* usc = ub + 4; const unsigned char* uqs = ub + 16;
        const unsigned int kbase = kb * 256u;
        #pragma unroll
        for (int e8 = 0; e8 < 8; ++e8) {
            unsigned int y = lane + (unsigned int)e8 * 32u;
            unsigned int c = (unsigned int)e8 >> 1, half = (unsigned int)e8 & 1u;
            unsigned char gsc8, gmn8, usc8, umn8;
            dec_scale_min_k4(e8, gsc, &gsc8, &gmn8);
            dec_scale_min_k4(e8, usc, &usc8, &umn8);
            unsigned char gbyte = gqs[c * 32u + lane], ubyte = uqs[c * 32u + lane];
            unsigned int gnib = half ? (gbyte >> 4) : (gbyte & 0x0F);
            unsigned int unib = half ? (ubyte >> 4) : (ubyte & 0x0F);
            float av = s_x[kbase + y];
            acc_g += av * (gd * (float)gsc8 * (float)gnib - gdm * (float)gmn8);
            acc_u += av * (ud * (float)usc8 * (float)unib - udm * (float)umn8);
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        acc_g += __shfl_down_sync(0xFFFFFFFFu, acc_g, o);
        acc_u += __shfl_down_sync(0xFFFFFFFFu, acc_u, o);
    }
    if (lane == 0) {
        float g = fminf(dec_bf16rt(acc_g), 10.0f);
        float u = fminf(fmaxf(dec_bf16rt(acc_u), -10.0f), 10.0f);
        float sil = g / (1.0f + __expf(-g));
        gate_silu_out[(unsigned long long)slot * inter + n] = __float2bfloat16(dec_bf16rt(sil * u));
    }
}

// ── Kernel 2: down. grid=(total_expanded, ceil(h/DEC_WARPS)), block=256. Each
// block stages the slot's s_act[inter] in shared, 8 warps each compute one down
// output -> expert_down_out[slot*h + n] (SORTED, feeds the unchanged unpermute).
extern "C" __global__ void atlas_moe_q4k_decode_down(
    const __nv_bfloat16* __restrict__ gate_silu_in,   // [total_expanded, inter] BF16 (s_act)
    const unsigned char* __restrict__ down_base,      // expert0 Q4_K/Q6_K [E,h,inter/256]
    const int* __restrict__ sorted_expert_ids,
    __nv_bfloat16* __restrict__ expert_down_out,      // [total_expanded, h] SORTED
    unsigned int h, unsigned int inter, unsigned int total_expanded,
    unsigned int down_is_q6k)
{
    const unsigned int slot = blockIdx.x;
    if (slot >= total_expanded) return;
    const int e = sorted_expert_ids[slot];
    const unsigned int n = blockIdx.y * DEC_WARPS + (threadIdx.x >> 5); // output feature (of h)
    const unsigned int lane = threadIdx.x & 31;

    extern __shared__ float s_act[];  // [inter]
    if (e < 0) {
        if (n < h && lane == 0) expert_down_out[(unsigned long long)slot * h + n] = __float2bfloat16(0.0f);
        return;
    }
    const __nv_bfloat16* a = gate_silu_in + (unsigned long long)slot * inter;
    for (unsigned int i = threadIdx.x; i < inter; i += DEC_BLOCK) s_act[i] = __bfloat162float(a[i]);
    __syncthreads();
    if (n >= h) return;

    const unsigned int dn_kblk = inter / 256u;
    const unsigned long long dn_row = (unsigned long long)e * h * dn_kblk + (unsigned long long)n * dn_kblk;
    float acc = 0.f;
    if (!down_is_q6k) {
        const unsigned char* drow = down_base + dn_row * 144ULL;
        for (unsigned int kb = 0; kb < dn_kblk; ++kb) {
            const unsigned char* db = drow + (unsigned long long)kb * 144ULL;
            float dd = dec_h2f(*(const unsigned short*)(db)), ddm = dec_h2f(*(const unsigned short*)(db + 2));
            const unsigned char* dsc = db + 4; const unsigned char* dqs = db + 16;
            const unsigned int kbase = kb * 256u;
            #pragma unroll
            for (int e8 = 0; e8 < 8; ++e8) {
                unsigned int y = lane + (unsigned int)e8 * 32u;
                unsigned int c = (unsigned int)e8 >> 1, half = (unsigned int)e8 & 1u;
                unsigned char sc8, mn8; dec_scale_min_k4(e8, dsc, &sc8, &mn8);
                unsigned char byte = dqs[c * 32u + lane];
                unsigned int nib = half ? (byte >> 4) : (byte & 0x0F);
                acc += s_act[kbase + y] * (dd * (float)sc8 * (float)nib - ddm * (float)mn8);
            }
        }
    } else {
        const unsigned char* drow = down_base + dn_row * 210ULL;
        for (unsigned int kb = 0; kb < dn_kblk; ++kb) {
            const unsigned char* db = drow + (unsigned long long)kb * 210ULL;
            const unsigned char* ql_all = db; const unsigned char* qh_all = db + 128;
            const signed char* sc_all = (const signed char*)(db + 192);
            float dd = dec_h2f(*(const unsigned short*)(db + 208));
            const unsigned int kbase = kb * 256u;
            #pragma unroll
            for (int e8 = 0; e8 < 8; ++e8) {
                unsigned int y = lane + (unsigned int)e8 * 32u;
                unsigned int hn = (unsigned int)e8 >> 2, gg = (unsigned int)e8 & 3u, is = lane >> 4;
                const unsigned char* ql = ql_all + hn * 64u; const unsigned char* qh = qh_all + hn * 32u;
                unsigned int sco = hn * 8u;
                int q;
                switch (gg) {
                    case 0: q = (int)(ql[lane]      & 0x0F) | (((int)(qh[lane] >> 0) & 3) << 4); break;
                    case 1: q = (int)(ql[lane + 32] & 0x0F) | (((int)(qh[lane] >> 2) & 3) << 4); break;
                    case 2: q = (int)(ql[lane]       >> 4)  | (((int)(qh[lane] >> 4) & 3) << 4); break;
                    default:q = (int)(ql[lane + 32]  >> 4)  | (((int)(qh[lane] >> 6) & 3) << 4); break;
                }
                q -= 32;
                float sc = (float)sc_all[sco + is + 2u * gg];
                acc += s_act[kbase + y] * (dd * sc * (float)q);
            }
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, o);
    if (lane == 0) expert_down_out[(unsigned long long)slot * h + n] = __float2bfloat16(acc);
}
