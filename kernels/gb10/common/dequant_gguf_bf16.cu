// SPDX-License-Identifier: AGPL-3.0-only

// GPU dequant: raw packed GGUF quant blocks -> BF16, on device.
//
// Scope: the hot P0 ggml types for flagship GGUFs -- Q8_0, Q4_K, Q6_K -- plus
// the PrismML-private Q2_0 group-N (id 42). Each kernel maps ONE CUDA block to
// ONE GGUF (super-)block and fans per-element work across threads. Input is the
// raw little-endian block bytes already uploaded h2d; output is contiguous BF16
// [n_blocks * QK]. Block byte-strides are passed as params (never hardcoded) so
// the Q2_0 group-128 (34 B) vs group-64 (18 B) variants share one kernel.
//
// Math mirrors the CPU reference dequant (ggml-quants.c `dequantize_row_*`)
// bit-for-bit; --fmad=false keeps CPU/GPU parity. These are load-time kernels:
// correctness > occupancy.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

// ---- helpers ---------------------------------------------------------------

// Little-endian IEEE fp16 (2 bytes) -> f32. Matches half::f16::to_f32 on host.
__device__ __forceinline__ float dq_rd_f16(const unsigned char* p) {
    unsigned short bits = (unsigned short)p[0] | ((unsigned short)p[1] << 8);
    return __half2float(__ushort_as_half(bits));
}

// Q4_K / Q5_K 6-bit packed scale+min unpack. Reproduces ggml get_scale_min_k4.
__device__ __forceinline__ void dq_scale_min_k4(
    int j, const unsigned char* q, unsigned char* sc, unsigned char* mn) {
    if (j < 4) {
        *sc = q[j] & 63;
        *mn = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *mn = (q[j + 4] >> 4)   | ((q[j]     >> 6) << 4);
    }
}

// ---- Q8_0 : { f16 d; i8 qs[32] }, QK=32, 34 B ------------------------------
// Grid (n_blocks,1,1)  Block (256,1,1). value = qs * d.
extern "C" __global__ void dequant_q8_0_to_bf16(
    const unsigned char* __restrict__ blocks,
    __nv_bfloat16* __restrict__ out,
    unsigned int n_blocks,
    unsigned int block_bytes)        // 34
{
    unsigned int b = blockIdx.x;
    if (b >= n_blocks) return;
    const unsigned char* blk = blocks + (unsigned long long)b * block_bytes;
    float d = dq_rd_f16(blk);
    const signed char* qs = (const signed char*)(blk + 2);
    __nv_bfloat16* o = out + (unsigned long long)b * 32u;
    for (unsigned int j = threadIdx.x; j < 32u; j += blockDim.x) {
        o[j] = __float2bfloat16((float)qs[j] * d);
    }
}

// ---- Q4_K : { f16 d; f16 dmin; u8 scales[12]; u8 qs[128] }, QK=256, 144 B ---
// 4 chunks of 64; each chunk c: is=2c (low nibbles, 32 elems) then is=2c+1
// (high nibbles, 32 elems). value = d*sc*nibble - dmin*m.
extern "C" __global__ void dequant_q4_k_to_bf16(
    const unsigned char* __restrict__ blocks,
    __nv_bfloat16* __restrict__ out,
    unsigned int n_blocks,
    unsigned int block_bytes)        // 144
{
    unsigned int b = blockIdx.x;
    if (b >= n_blocks) return;
    const unsigned char* blk = blocks + (unsigned long long)b * block_bytes;
    float d    = dq_rd_f16(blk);
    float dmin = dq_rd_f16(blk + 2);
    const unsigned char* scales = blk + 4;
    const unsigned char* qs     = blk + 16;
    __nv_bfloat16* o = out + (unsigned long long)b * 256u;

    for (unsigned int y = threadIdx.x; y < 256u; y += blockDim.x) {
        unsigned int c    = y >> 6;          // chunk 0..3  (y / 64)
        unsigned int half = (y >> 5) & 1u;   // 0 = low nibble, 1 = high
        unsigned int l    = y & 31u;         // 0..31
        int is = (int)(2u * c + half);
        unsigned char sc, mn;
        dq_scale_min_k4(is, scales, &sc, &mn);
        unsigned char byte = qs[c * 32u + l];
        unsigned int nib = half ? (byte >> 4) : (byte & 0x0F);
        float v = d * (float)sc * (float)nib - dmin * (float)mn;
        o[y] = __float2bfloat16(v);
    }
}

// ---- Q6_K : { u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d }, QK=256, 210 B --
// Two 128-elem halves; within a half, 4 groups of 32 pick scale sco+is+2*g,
// is=l/16. 6-bit quant centered by -32. value = d * sc(i8) * q.
extern "C" __global__ void dequant_q6_k_to_bf16(
    const unsigned char* __restrict__ blocks,
    __nv_bfloat16* __restrict__ out,
    unsigned int n_blocks,
    unsigned int block_bytes)        // 210
{
    unsigned int b = blockIdx.x;
    if (b >= n_blocks) return;
    const unsigned char* blk = blocks + (unsigned long long)b * block_bytes;
    const unsigned char* ql_all = blk;
    const unsigned char* qh_all = blk + 128;
    const signed char*   sc_all = (const signed char*)(blk + 192);
    float d = dq_rd_f16(blk + 208);
    __nv_bfloat16* o = out + (unsigned long long)b * 256u;

    for (unsigned int y = threadIdx.x; y < 256u; y += blockDim.x) {
        unsigned int n = y >> 7;             // half 0/1  (y / 128)
        unsigned int w = y & 127u;           // 0..127 within half
        unsigned int g = w >> 5;             // group 0..3
        unsigned int l = w & 31u;            // 0..31
        const unsigned char* ql = ql_all + n * 64u;
        const unsigned char* qh = qh_all + n * 32u;
        unsigned int sco = n * 8u;
        unsigned int is  = l >> 4;           // 0 or 1
        int q;
        switch (g) {
            case 0: q = (int)(ql[l]        & 0x0F) | (((int)(qh[l] >> 0) & 3) << 4); break;
            case 1: q = (int)(ql[l + 32]   & 0x0F) | (((int)(qh[l] >> 2) & 3) << 4); break;
            case 2: q = (int)(ql[l]         >> 4)  | (((int)(qh[l] >> 4) & 3) << 4); break;
            default:q = (int)(ql[l + 32]    >> 4)  | (((int)(qh[l] >> 6) & 3) << 4); break;
        }
        q -= 32;
        float sc = (float)sc_all[sco + is + 2u * g];
        o[y] = __float2bfloat16(d * sc * (float)q);
    }
}

// ---- Q2_0 group-N (PrismML id 42) : { f16 d; u8 qs[G/4] }, scale at FRONT ---
// Contiguous low-bits-first 2-bit codes. value = (code - 1) * d.
// group_size G in {128, 64}; block_bytes = 2 + G/4 (34 or 18). Parameterized.
extern "C" __global__ void dequant_q2_0_gn_to_bf16(
    const unsigned char* __restrict__ blocks,
    __nv_bfloat16* __restrict__ out,
    unsigned int n_blocks,
    unsigned int group_size,         // 128 or 64
    unsigned int block_bytes)        // 2 + group_size/4
{
    unsigned int b = blockIdx.x;
    if (b >= n_blocks) return;
    const unsigned char* blk = blocks + (unsigned long long)b * block_bytes;
    float d = dq_rd_f16(blk);         // scale at FRONT
    const unsigned char* qs = blk + 2;
    __nv_bfloat16* o = out + (unsigned long long)b * group_size;
    for (unsigned int j = threadIdx.x; j < group_size; j += blockDim.x) {
        int code = (qs[j >> 2] >> (2u * (j & 3u))) & 3;   // low-bits-first
        o[j] = __float2bfloat16((float)(code - 1) * d);
    }
}
