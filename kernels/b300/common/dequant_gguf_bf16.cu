// SPDX-License-Identifier: AGPL-3.0-only

// GPU dequant: raw packed GGUF quant blocks -> BF16, on device.
//
// Scope: the hot P0 ggml types for flagship GGUFs -- Q8_0, Q4_K, Q6_K -- plus
// the PrismML-private Q2_0 group-N (id 42), plus Q2_K and Q3_K for the K-quant
// checkpoints (DeepSeek-V4.1 Flash Q2_K: attention, embed and the engram tables
// are Q2_K, the routed down projections Q3_K). Each kernel maps ONE CUDA block to
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

// ---- Q2_K : { u8 scales[16]; u8 qs[64]; f16 d; f16 dmin }, QK=256, 84 B ----
// Two 128-elem halves (n); within a half, 4 shift groups j (2-bit codes at
// shift 2j), each of two 16-lane runs (sub) with their own 4-bit scale/min
// byte: is = n*8 + 2j + sub. value = d*(sc & 0xF)*code - dmin*(sc >> 4).
// Mirrors dequant_cpu::blocks::dequant_q2_k in element order and op order.
extern "C" __global__ void dequant_q2_k_to_bf16(
    const unsigned char* __restrict__ blocks,
    __nv_bfloat16* __restrict__ out,
    unsigned int n_blocks,
    unsigned int block_bytes)        // 84
{
    unsigned int b = blockIdx.x;
    if (b >= n_blocks) return;
    const unsigned char* blk    = blocks + (unsigned long long)b * block_bytes;
    const unsigned char* scales = blk;
    const unsigned char* qs     = blk + 16;
    float d    = dq_rd_f16(blk + 80);
    float dmin = dq_rd_f16(blk + 82);
    __nv_bfloat16* o = out + (unsigned long long)b * 256u;

    for (unsigned int y = threadIdx.x; y < 256u; y += blockDim.x) {
        unsigned int n   = y >> 7;           // half 0/1
        unsigned int w   = y & 127u;
        unsigned int j   = w >> 5;           // shift group 0..3
        unsigned int sub = (w >> 4) & 1u;    // 16-lane run 0/1
        unsigned int l   = w & 15u;
        unsigned int is  = n * 8u + 2u * j + sub;
        unsigned char sc = scales[is];
        float dl = d * (float)(sc & 0x0F);
        float ml = dmin * (float)(sc >> 4);
        unsigned char q = qs[n * 32u + sub * 16u + l];
        int code = (int)((q >> (2u * j)) & 3u);
        o[y] = __float2bfloat16(dl * (float)code - ml);
    }
}

// ---- Q3_K : { u8 hmask[32]; u8 qs[64]; u8 scales[12]; f16 d }, QK=256, 110 B --
// Same halves / shift groups / 16-lane runs as Q2_K. The 6-bit scales unpack
// from 12 bytes exactly as ggml (kmask1/kmask2 shuffle) into 16 int8, centred
// by -32. The high bit of each 3-bit code comes from hmask, one mask bit per
// (half, shift group): m = 1 << (4n + j). value = d*(sc-32)*(code - (hbit ? 0 : 4)).
// Mirrors dequant_cpu::blocks::dequant_q3_k.
extern "C" __global__ void dequant_q3_k_to_bf16(
    const unsigned char* __restrict__ blocks,
    __nv_bfloat16* __restrict__ out,
    unsigned int n_blocks,
    unsigned int block_bytes)        // 110
{
    unsigned int b = blockIdx.x;
    if (b >= n_blocks) return;
    const unsigned char* blk   = blocks + (unsigned long long)b * block_bytes;
    const unsigned char* hmask = blk;
    const unsigned char* qs    = blk + 32;
    const unsigned char* raw   = blk + 96;
    float d_all = dq_rd_f16(blk + 108);
    __nv_bfloat16* o = out + (unsigned long long)b * 256u;

    // scale unpack (ggml): aux[0..3] from three LE u32 of the 12 raw bytes
    const unsigned int KM1 = 0x03030303u, KM2 = 0x0f0f0f0fu;
    unsigned int a0 = (unsigned int)raw[0] | ((unsigned int)raw[1] << 8) | ((unsigned int)raw[2] << 16) | ((unsigned int)raw[3] << 24);
    unsigned int a1 = (unsigned int)raw[4] | ((unsigned int)raw[5] << 8) | ((unsigned int)raw[6] << 16) | ((unsigned int)raw[7] << 24);
    unsigned int tmp = (unsigned int)raw[8] | ((unsigned int)raw[9] << 8) | ((unsigned int)raw[10] << 16) | ((unsigned int)raw[11] << 24);
    unsigned int aux[4];
    aux[2] = ((a0 >> 4) & KM2) | (((tmp >> 4) & KM1) << 4);
    aux[3] = ((a1 >> 4) & KM2) | (((tmp >> 6) & KM1) << 4);
    aux[0] = (a0 & KM2) | (((tmp >> 0) & KM1) << 4);
    aux[1] = (a1 & KM2) | (((tmp >> 2) & KM1) << 4);

    for (unsigned int y = threadIdx.x; y < 256u; y += blockDim.x) {
        unsigned int n   = y >> 7;
        unsigned int w   = y & 127u;
        unsigned int j   = w >> 5;
        unsigned int sub = (w >> 4) & 1u;
        unsigned int l   = w & 15u;
        unsigned int is  = n * 8u + 2u * j + sub;
        int sc = (int)(signed char)((aux[is >> 2] >> (8u * (is & 3u))) & 0xFFu);
        float dl = d_all * (float)(sc - 32);
        unsigned char m = (unsigned char)(1u << (4u * n + j));
        unsigned int idx = sub * 16u + l;
        int h = (hmask[idx] & m) ? 0 : 4;
        unsigned char q = qs[n * 32u + idx];
        int code = (int)((q >> (2u * j)) & 3u) - h;
        o[y] = __float2bfloat16(dl * (float)code);
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
