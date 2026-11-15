// SPDX-License-Identifier: AGPL-3.0-only
// Algorithm after XAttention (antidiagonal block scoring for block-sparse prefill), described in its paper;
// independent implementation, no source copied (the reference repository has no detected license).
//
// Block-sparse paged PREFILL attention for Ada (sm_89). APPROXIMATE: it is
// reached only when the `[defaults] prefill_sparse_attn` row resolves on (off on every target) and this
// module is in the image (only kernels/ada compiles it). Host side:
// crates/spark-model/src/port/sparse_prefill/launch.rs; CPU reference: port/sparse_prefill/reference.rs,
// which restates every rule below in the same order.
//
// ── Geometry ──────────────────────────────────────────────────────────────────────────────────────────
// One chunked-prefill call: q_len query rows at absolute positions [q_offset, q_offset + q_len), keys
// [0, kv_len) in Atlas's paged cache (BF16, or FP8 E4M3 with a per-tensor scale), GQA head kvh = h / (nq/nkv).
// Selection blocks are BLOCK = 128 tokens on ABSOLUTE positions, for queries and keys alike: query block
// I = pos / BLOCK, key block J = pos / BLOCK. A query block attends to key blocks 0..=I (causal).
//
// ── Stage 1: score and select (one persistent CTA loops over (head, query block) items) ────────────
// Antidiagonal scoring with stride S (4, 8 or 16): a query block's B rows are read as R = B/S strided rows,
// strided row a = the S rows S*a .. S*a+S-1, and likewise for a key block. The strided logit of (a, b) is
// the sum of the products on the antidiagonals of their S x S sub-block, i.e. over the S pairs
//     q[S*a + S-1-r] . k[S*b + r],  r = 0..S-1,
// scaled by (1/sqrt(d)) / S (the mean of the S logits, so the strided logit is on a single token's scale).
// That is one GEMM per (I, J): the query rows reshaped to R x (S*d) with the S rows of a group in REVERSED
// order, times the key rows reshaped to R x (S*d) in natural order. It runs on BF16 mma.sync m16n8k16
// with FP32 accumulation (FP8 keys are widened exactly to BF16 and the logit takes the k scale).
// Masks: a strided query row counts only when all S of its rows are in this chunk (rows outside it are
// read as zero); a strided key row is masked when its first key is at or past kv_len (later rows of a
// partial group read as zero); on the diagonal block (J == I) strided key b is masked when b > a.
// Per strided row, a softmax over every unmasked strided key of every key block J <= I; the mass of
// block J is the mean over counted rows of the probability that falls in J (so the masses of one query
// block sum to 1). Stored as one log-sum-exp per (row, block) in a per-CTA scratch slot, so the logits
// are computed once.
// Selection per (head, query block): the sink block 0, the diagonal block I and its neighbor I-1 are
// always kept (F). The rest are kept largest mass first until kept mass >= threshold, implemented as a
// deterministic binary search on the FP32 bit pattern of the cut c: the largest c with
// sum_{J not in F, mass_J >= c} mass_J >= threshold - mass(F); every block with mass >= c is kept (ties
// at the cut are all kept). A query block with no counted strided row keeps every block. Output: one bit
// per key block, mask[(h * n_qb + qb) * mask_words + J / 32], plus kept and candidate block totals.
//
// ── Stage 2: block-sparse flash attention ─────────────────────────────────────────────────────────
// One CTA = 64 query rows (absolute-aligned, so every tile lies in one query block) of one head, 4 warps x
// 16 rows. It walks the set bits of its query block's mask in increasing J and, inside each kept block, key
// sub-tiles of BC rows: S = Q K^T on BF16 mma.sync, causal and range masks, online softmax in registers,
// O += P V on FP16 mma.sync (P and V in FP16, as the dense prefill does), FP32 accumulators throughout.
// With every bit set this is exact causal attention (the microtest's identity check against the dense
// kernel); the host never runs it at threshold 1.0, which takes the dense kernel instead.
//
// Shared memory is dynamic, through prefill_dyn_smem.cuh, with one exported launch floor per entry.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include "prefill_dyn_smem.cuh"

namespace psa {

constexpr unsigned BLOCK = 128;          // selection block, tokens (queries and keys)
constexpr unsigned TILE_Q = 64;          // query rows per attention CTA
constexpr unsigned SCORE_THREADS = 256;  // 8 warps per scoring CTA
constexpr unsigned ATTN_THREADS = 128;   // 4 warps x 16 query rows
constexpr unsigned PAD = 8;              // bf16 elements of padding per shared row (16 bytes)
constexpr unsigned MAX_R = 32;           // strided rows per block at the smallest stride (4)
constexpr float NEG = -1.0e30f;          // masked logit / empty log-sum-exp
constexpr float NEG_HALF = -1.0e29f;     // at or below: treated as masked
constexpr unsigned INF_BITS = 0x7F800000u;

template <unsigned HD>
struct ScoreTiles {
    __nv_bfloat16 q[BLOCK][HD + PAD];    // slot r * R + a holds query row S*a + S-1-r of the block
    float row_logl[MAX_R];               // per strided row: log of its softmax normalizer (NEG = not counted)
    float red[SCORE_THREADS / 32];
    float bcast[2];
};

// Key rows per attention sub-tile: 32 at head_dim 256 (the FP32 output accumulators take 128 registers
// per thread there), 64 at 128. Mirrored by `launch::attn_bc`.
template <unsigned HD>
struct AttnBc {
    static constexpr unsigned value = HD >= 256 ? 32u : 64u;
};

template <unsigned HD>
struct AttnTiles {
    __nv_bfloat16 q[TILE_Q][HD + PAD];
    __nv_bfloat16 k[AttnBc<HD>::value][HD + PAD];
    __nv_bfloat16 v[AttnBc<HD>::value][HD + PAD];
};

__device__ __forceinline__ float fp8_to_float(unsigned char b) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3)));
}

__device__ __forceinline__ unsigned bf16_bits(__nv_bfloat16 x) {
    return (unsigned)(*reinterpret_cast<const unsigned short*>(&x));
}

// Two FP8 E4M3 bytes (low byte = lower index) widened to a BF16 pair, exactly (E4M3 fits BF16).
__device__ __forceinline__ unsigned fp8x2_to_bf16x2(unsigned short v) {
    const __nv_bfloat16 lo = __float2bfloat16(fp8_to_float((unsigned char)(v & 0xFF)));
    const __nv_bfloat16 hi = __float2bfloat16(fp8_to_float((unsigned char)(v >> 8)));
    return bf16_bits(lo) | (bf16_bits(hi) << 16);
}

__device__ __forceinline__ unsigned half2_bits(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<unsigned*>(&h);
}

__device__ __forceinline__ unsigned bf16_pair_to_half2(__nv_bfloat16 lo, __nv_bfloat16 hi) {
    return half2_bits(__bfloat162float(lo), __bfloat162float(hi));
}

__device__ __forceinline__ void mma_bf16(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                         unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                 : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__ void mma_f16(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                        unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                 : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// Element offset of (pos, kv head) in a per-layer paged pool: phys * stride + (pos % bs) * row + kvh * HD.
__device__ __forceinline__ unsigned long long kv_offset(const unsigned* __restrict__ bt, unsigned pos,
                                                        unsigned bs, unsigned long long stride,
                                                        unsigned row_elems, unsigned head_off) {
    const unsigned phys = bt[pos / bs];
    return (unsigned long long)phys * stride + (unsigned long long)(pos % bs) * row_elems + head_off;
}

// Sum over the CTA in a fixed order (xor tree in each warp, then warps in index order): the same inputs
// give the same bits on every run. Every thread must call it.
__device__ __forceinline__ float block_sum(float v, float* red, float* bcast) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) red[warp] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (unsigned w = 0; w < (blockDim.x >> 5); ++w) s += red[w];
        bcast[0] = s;
    }
    __syncthreads();
    const float r = bcast[0];
    __syncthreads();
    return r;
}

// A strided query row counts only when all S of its rows are in this chunk.
template <unsigned S>
__device__ __forceinline__ bool row_counted(unsigned blk0, unsigned a, unsigned q_offset, unsigned q_end) {
    const unsigned first = blk0 + S * a;
    return first >= q_offset && first + S <= q_end;
}

// ── Stage 1 ─────────────────────────────────────────────────────────────────────────────────────────
template <unsigned HD, unsigned S, bool FP8>
__device__ __forceinline__ void score_body(
    const __nv_bfloat16* __restrict__ Q, const void* __restrict__ Kc, const unsigned* __restrict__ bt,
    float* __restrict__ scratch, unsigned* __restrict__ masks, unsigned long long* __restrict__ stats,
    unsigned q_len, unsigned kv_len, unsigned q_offset, unsigned nq, unsigned nkv, unsigned bs,
    unsigned long long stride, unsigned q_block_lo, unsigned n_qb, unsigned nkb, unsigned mask_words,
    float logit_scale, float threshold) {
    constexpr unsigned R = BLOCK / S;         // 32, 16, 8 strided rows per block
    constexpr unsigned MT = (R + 15) / 16;    // m16 tiles (R = 8 pads rows 8..15 with zeros)
    constexpr unsigned NT = R / 8;            // n8 tiles
    constexpr unsigned KD = HD / 16;          // k16 steps per antidiagonal phase r
    static_assert(R <= MAX_R && R % 8 == 0, "stride must be 4, 8 or 16");
    AVAROK_PREFILL_TILES(ScoreTiles<HD>, sm);

    const unsigned tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const unsigned g = lane >> 2, t = lane & 3;
    const unsigned nwarps = blockDim.x >> 5;
    const unsigned q_end = q_offset + q_len;
    const unsigned q_stride = nq * HD;
    const unsigned row_elems = nkv * HD;
    const unsigned group = nq / nkv;
    const unsigned items = n_qb * nq;
    float* __restrict__ lse = scratch + (size_t)blockIdx.x * (R + 1) * nkb;   // [R][nkb]
    float* __restrict__ mass = lse + (size_t)R * nkb;                          // [nkb]
    const __nv_bfloat16* __restrict__ kb16 = reinterpret_cast<const __nv_bfloat16*>(Kc);
    const unsigned char* __restrict__ kb8 = reinterpret_cast<const unsigned char*>(Kc);

    for (unsigned item = blockIdx.x; item < items; item += gridDim.x) {
        const unsigned h = item / n_qb, qb = item % n_qb;
        const unsigned I = q_block_lo + qb;
        const unsigned head_off = (h / group) * HD;
        const unsigned blk0 = I * BLOCK;
        const unsigned last_kb = (kv_len - 1) / BLOCK;
        const unsigned nJ = (I < last_kb ? I : last_kb) + 1;
        const unsigned jd = nJ - 1;
        unsigned* __restrict__ mask = masks + (size_t)item * mask_words;

        // Query block, strided-reordered: slot (r, a) <- row S*a + S-1-r; rows outside the chunk are zero.
        for (unsigned c = tid; c < BLOCK * (HD / 8); c += blockDim.x) {
            const unsigned s = c / (HD / 8), col = (c % (HD / 8)) * 8;
            const unsigned r = s / R, a = s % R;
            const unsigned pos = blk0 + S * a + (S - 1 - r);
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (pos >= q_offset && pos < q_end)
                v = *reinterpret_cast<const uint4*>(Q + (size_t)(pos - q_offset) * q_stride + h * HD + col);
            *reinterpret_cast<uint4*>(&sm.q[s][col]) = v;
        }
        unsigned n_counted = 0;
        for (unsigned a = 0; a < R; ++a) n_counted += row_counted<S>(blk0, a, q_offset, q_end) ? 1u : 0u;
        const bool keep_all = (n_counted == 0) || !(threshold < 1.0f);
        __syncthreads();

        if (!keep_all) {
            // Strided logits of (I, J) for each J this warp owns, reduced to one log-sum-exp per row.
            for (unsigned J = warp; J < nJ; J += nwarps) {
                float acc[MT][NT][4];
                #pragma unroll
                for (unsigned mt = 0; mt < MT; ++mt)
                    #pragma unroll
                    for (unsigned nt = 0; nt < NT; ++nt)
                        acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.0f;
                for (unsigned r = 0; r < S; ++r) {
                    unsigned long long koff[NT];
                    bool kok[NT];
                    #pragma unroll
                    for (unsigned nt = 0; nt < NT; ++nt) {
                        const unsigned pos = J * BLOCK + S * (8 * nt + g) + r;
                        kok[nt] = pos < kv_len;
                        koff[nt] = kok[nt] ? kv_offset(bt, pos, bs, stride, row_elems, head_off) : 0ull;
                    }
                    #pragma unroll
                    for (unsigned kd = 0; kd < KD; ++kd) {
                        const unsigned d0 = kd * 16 + 2 * t;
                        unsigned a[MT][4];
                        #pragma unroll
                        for (unsigned mt = 0; mt < MT; ++mt) {
                            const unsigned ra = 16 * mt + g, rb = ra + 8;
                            const unsigned sa = r * R + ra, sb = r * R + rb;
                            a[mt][0] = ra < R ? *reinterpret_cast<const unsigned*>(&sm.q[sa][d0]) : 0u;
                            a[mt][2] = ra < R ? *reinterpret_cast<const unsigned*>(&sm.q[sa][d0 + 8]) : 0u;
                            a[mt][1] = rb < R ? *reinterpret_cast<const unsigned*>(&sm.q[sb][d0]) : 0u;
                            a[mt][3] = rb < R ? *reinterpret_cast<const unsigned*>(&sm.q[sb][d0 + 8]) : 0u;
                        }
                        #pragma unroll
                        for (unsigned nt = 0; nt < NT; ++nt) {
                            unsigned b0 = 0u, b1 = 0u;
                            if (kok[nt]) {
                                if constexpr (FP8) {
                                    b0 = fp8x2_to_bf16x2(*reinterpret_cast<const unsigned short*>(kb8 + koff[nt] + d0));
                                    b1 = fp8x2_to_bf16x2(*reinterpret_cast<const unsigned short*>(kb8 + koff[nt] + d0 + 8));
                                } else {
                                    b0 = __ldg(reinterpret_cast<const unsigned*>(kb16 + koff[nt] + d0));
                                    b1 = __ldg(reinterpret_cast<const unsigned*>(kb16 + koff[nt] + d0 + 8));
                                }
                            }
                            #pragma unroll
                            for (unsigned mt = 0; mt < MT; ++mt)
                                mma_bf16(acc[mt][nt], a[mt][0], a[mt][1], a[mt][2], a[mt][3], b0, b1);
                        }
                    }
                }
                // Masks, then per strided row: max, sum of exponentials, log-sum-exp of this block.
                #pragma unroll
                for (unsigned mt = 0; mt < MT; ++mt) {
                    #pragma unroll
                    for (unsigned hr = 0; hr < 2; ++hr) {   // rows g (C regs 0, 1) and g + 8 (2, 3)
                        const unsigned arow = 16 * mt + g + 8 * hr;
                        float v[NT][2];
                        float m = NEG;
                        #pragma unroll
                        for (unsigned nt = 0; nt < NT; ++nt)
                            #pragma unroll
                            for (unsigned e = 0; e < 2; ++e) {
                                const unsigned b = 8 * nt + 2 * t + e;
                                const bool ok = (J * BLOCK + S * b < kv_len) && (J < I || b <= arow);
                                v[nt][e] = ok ? acc[mt][nt][2 * hr + e] * logit_scale : NEG;
                                m = fmaxf(m, v[nt][e]);
                            }
                        m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, 1));
                        m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, 2));
                        float s = 0.0f;
                        #pragma unroll
                        for (unsigned nt = 0; nt < NT; ++nt)
                            #pragma unroll
                            for (unsigned e = 0; e < 2; ++e)
                                s += v[nt][e] > NEG_HALF ? __expf(v[nt][e] - m) : 0.0f;
                        s += __shfl_xor_sync(0xFFFFFFFFu, s, 1);
                        s += __shfl_xor_sync(0xFFFFFFFFu, s, 2);
                        if (t == 0 && arow < R) {
                            const bool counted = row_counted<S>(blk0, arow, q_offset, q_end);
                            lse[arow * nkb + J] = (counted && m > NEG_HALF) ? m + __logf(s) : NEG;
                        }
                    }
                }
            }
            __syncthreads();
            // Per strided row: the log of its softmax normalizer over every block.
            for (unsigned a = warp; a < R; a += nwarps) {
                float m = NEG;
                for (unsigned J = lane; J < nJ; J += 32) m = fmaxf(m, lse[a * nkb + J]);
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, o));
                float s = 0.0f;
                for (unsigned J = lane; J < nJ; J += 32) {
                    const float x = lse[a * nkb + J];
                    s += x > NEG_HALF ? __expf(x - m) : 0.0f;
                }
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, o);
                if (lane == 0)
                    sm.row_logl[a] = (row_counted<S>(blk0, a, q_offset, q_end) && m > NEG_HALF) ? m + __logf(s) : NEG;
            }
            __syncthreads();
            // Block masses: mean over counted rows of the probability in each block.
            const float inv_counted = 1.0f / (float)n_counted;
            for (unsigned J = tid; J < nJ; J += blockDim.x) {
                float acc = 0.0f;
                for (unsigned a = 0; a < R; ++a) {
                    const float ll = sm.row_logl[a], x = lse[a * nkb + J];
                    if (ll > NEG_HALF && x > NEG_HALF) acc += __expf(x - ll);
                }
                mass[J] = acc * inv_counted;
            }
            __syncthreads();
        }

        // Selection: forced F = {0, I-1, I}; the cut c by binary search on the FP32 bit pattern.
        unsigned cut = 0u;   // 0 keeps every candidate
        if (!keep_all) {
            float mf = mass[0];
            if (jd >= 1) mf += mass[jd];
            if (jd >= 2) mf += mass[jd - 1];
            const float need = threshold - mf;
            if (need <= 0.0f) {
                cut = INF_BITS;   // the forced blocks already hold the threshold
            } else {
                float part = 0.0f;
                for (unsigned J = tid; J < nJ; J += blockDim.x)
                    if (J != 0 && J != jd && J + 1 != jd) part += mass[J];
                const float total = block_sum(part, sm.red, sm.bcast);
                if (total >= need) {
                    unsigned lo = 0u, hi = INF_BITS;
                    while (hi - lo > 1u) {
                        const unsigned mid = lo + ((hi - lo) >> 1);
                        float p = 0.0f;
                        for (unsigned J = tid; J < nJ; J += blockDim.x)
                            if (J != 0 && J != jd && J + 1 != jd && __float_as_uint(mass[J]) >= mid) p += mass[J];
                        if (block_sum(p, sm.red, sm.bcast) >= need) lo = mid; else hi = mid;
                    }
                    cut = lo;
                }
                // total < need (FP32 shortfall at a threshold near 1): cut stays 0, keep every candidate.
            }
        }
        unsigned kept = 0u;
        for (unsigned w = tid; w < mask_words; w += blockDim.x) {
            unsigned word = 0u;
            for (unsigned e = 0; e < 32; ++e) {
                const unsigned J = 32 * w + e;
                if (J >= nJ) break;
                const bool forced = J == 0 || J == jd || J + 1 == jd;
                if (keep_all || forced || __float_as_uint(mass[J]) >= cut) word |= 1u << e;
            }
            mask[w] = word;
            kept += __popc(word);
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) kept += __shfl_xor_sync(0xFFFFFFFFu, kept, o);
        if (lane == 0 && kept != 0u) atomicAdd(&stats[0], (unsigned long long)kept);
        if (tid == 0) {
            atomicAdd(&stats[1], (unsigned long long)nJ);
            atomicAdd(&stats[2], 1ull);
        }
        __syncthreads();   // the next item reuses the query tile, the row normalizers and the scratch slot
    }
}

// ── Stage 2 ─────────────────────────────────────────────────────────────────────────────────────────
template <unsigned HD, bool FP8>
__device__ __forceinline__ void load_kv_rows(__nv_bfloat16 (*dst)[HD + PAD], const void* __restrict__ cache,
                                             const unsigned* __restrict__ bt, unsigned kv0, unsigned k_hi,
                                             unsigned rows, unsigned bs, unsigned long long stride,
                                             unsigned row_elems, unsigned head_off, float scale) {
    if constexpr (FP8) {
        const unsigned char* __restrict__ c8 = reinterpret_cast<const unsigned char*>(cache);
        for (unsigned c = threadIdx.x; c < rows * (HD / 16); c += blockDim.x) {
            const unsigned row = c / (HD / 16), col = (c % (HD / 16)) * 16;
            const unsigned pos = kv0 + row;
            union { uint4 u[2]; __nv_bfloat16 h[16]; } cv;
            if (pos < k_hi) {
                const uint4 raw = *reinterpret_cast<const uint4*>(
                    c8 + kv_offset(bt, pos, bs, stride, row_elems, head_off) + col);
                const unsigned char* b = reinterpret_cast<const unsigned char*>(&raw);
                #pragma unroll
                for (int i = 0; i < 16; ++i) cv.h[i] = __float2bfloat16(fp8_to_float(b[i]) * scale);
            } else {
                cv.u[0] = make_uint4(0u, 0u, 0u, 0u);
                cv.u[1] = cv.u[0];
            }
            *reinterpret_cast<uint4*>(&dst[row][col]) = cv.u[0];
            *reinterpret_cast<uint4*>(&dst[row][col + 8]) = cv.u[1];
        }
    } else {
        const __nv_bfloat16* __restrict__ c16 = reinterpret_cast<const __nv_bfloat16*>(cache);
        for (unsigned c = threadIdx.x; c < rows * (HD / 8); c += blockDim.x) {
            const unsigned row = c / (HD / 8), col = (c % (HD / 8)) * 8;
            const unsigned pos = kv0 + row;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (pos < k_hi)
                v = *reinterpret_cast<const uint4*>(c16 + kv_offset(bt, pos, bs, stride, row_elems, head_off) + col);
            *reinterpret_cast<uint4*>(&dst[row][col]) = v;
        }
    }
}

template <unsigned HD, bool FP8>
__device__ __forceinline__ void attn_body(
    const __nv_bfloat16* __restrict__ Q, const void* __restrict__ Kc, const void* __restrict__ Vc,
    const unsigned* __restrict__ bt, const unsigned* __restrict__ masks, __nv_bfloat16* __restrict__ O,
    unsigned q_len, unsigned kv_len, unsigned q_offset, unsigned nq, unsigned nkv, unsigned bs,
    unsigned long long stride, unsigned q_block_lo, unsigned n_qb, unsigned mask_words, float inv_sqrt_d,
    float k_scale, float v_scale) {
    constexpr unsigned BC = AttnBc<HD>::value;
    constexpr unsigned NS = BC / 8;     // score n8 tiles
    constexpr unsigned NO = HD / 8;     // output n8 tiles
    AVAROK_PREFILL_TILES(AttnTiles<HD>, sm);

    const unsigned tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const unsigned g = lane >> 2, t = lane & 3;
    const unsigned q_end = q_offset + q_len;
    const unsigned q_stride = nq * HD;
    const unsigned row_elems = nkv * HD;
    const unsigned h = blockIdx.y;
    const unsigned head_off = (h / (nq / nkv)) * HD;
    const unsigned tile0 = (q_offset / TILE_Q) * TILE_Q + blockIdx.x * TILE_Q;
    const unsigned I = tile0 / BLOCK;
    const unsigned* __restrict__ mask = masks + (size_t)(h * n_qb + (I - q_block_lo)) * mask_words;
    const unsigned last_q = (tile0 + TILE_Q < q_end ? tile0 + TILE_Q : q_end) - 1;

    for (unsigned c = tid; c < TILE_Q * (HD / 8); c += blockDim.x) {
        const unsigned row = c / (HD / 8), col = (c % (HD / 8)) * 8;
        const unsigned pos = tile0 + row;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (pos >= q_offset && pos < q_end)
            v = *reinterpret_cast<const uint4*>(Q + (size_t)(pos - q_offset) * q_stride + h * HD + col);
        *reinterpret_cast<uint4*>(&sm.q[row][col]) = v;
    }

    const unsigned r0 = warp * 16 + g, r1 = r0 + 8;
    const unsigned p0 = tile0 + r0, p1 = tile0 + r1;
    const bool ok0 = p0 >= q_offset && p0 < q_end, ok1 = p1 >= q_offset && p1 < q_end;
    float m0 = NEG, m1 = NEG, l0 = 0.0f, l1 = 0.0f;
    float o[NO][4];
    #pragma unroll
    for (unsigned i = 0; i < NO; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;

    bool done = false;
    for (unsigned w = 0; w < mask_words && !done; ++w) {
        unsigned bits = mask[w];
        while (bits != 0u) {
            const unsigned J = 32 * w + (unsigned)(__ffs((int)bits) - 1);
            bits &= bits - 1u;
            const unsigned k_lo = J * BLOCK;
            if (k_lo > last_q) { done = true; break; }   // blocks ascend: nothing later is causal for this tile
            unsigned k_hi = k_lo + BLOCK;
            if (k_hi > kv_len) k_hi = kv_len;
            if (k_hi > last_q + 1) k_hi = last_q + 1;
            for (unsigned kv0 = k_lo; kv0 < k_hi; kv0 += BC) {
                __syncthreads();   // Q is loaded / the previous sub-tile is consumed
                load_kv_rows<HD, FP8>(sm.k, Kc, bt, kv0, k_hi, BC, bs, stride, row_elems, head_off, k_scale);
                load_kv_rows<HD, FP8>(sm.v, Vc, bt, kv0, k_hi, BC, bs, stride, row_elems, head_off, v_scale);
                __syncthreads();

                float s[NS][4];
                #pragma unroll
                for (unsigned nt = 0; nt < NS; ++nt) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
                #pragma unroll
                for (unsigned ks = 0; ks < HD / 16; ++ks) {
                    const unsigned c0 = ks * 16 + 2 * t;
                    const unsigned a0 = *reinterpret_cast<const unsigned*>(&sm.q[r0][c0]);
                    const unsigned a1 = *reinterpret_cast<const unsigned*>(&sm.q[r1][c0]);
                    const unsigned a2 = *reinterpret_cast<const unsigned*>(&sm.q[r0][c0 + 8]);
                    const unsigned a3 = *reinterpret_cast<const unsigned*>(&sm.q[r1][c0 + 8]);
                    #pragma unroll
                    for (unsigned nt = 0; nt < NS; ++nt) {
                        const unsigned n = nt * 8 + g;
                        const unsigned b0 = *reinterpret_cast<const unsigned*>(&sm.k[n][c0]);
                        const unsigned b1 = *reinterpret_cast<const unsigned*>(&sm.k[n][c0 + 8]);
                        mma_bf16(s[nt], a0, a1, a2, a3, b0, b1);
                    }
                }
                float mx0 = NEG, mx1 = NEG;
                #pragma unroll
                for (unsigned nt = 0; nt < NS; ++nt)
                    #pragma unroll
                    for (unsigned e = 0; e < 2; ++e) {
                        const unsigned kp = kv0 + nt * 8 + 2 * t + e;
                        const bool kin = kp < k_hi;
                        s[nt][e] = (ok0 && kin && kp <= p0) ? s[nt][e] * inv_sqrt_d : NEG;
                        s[nt][2 + e] = (ok1 && kin && kp <= p1) ? s[nt][2 + e] * inv_sqrt_d : NEG;
                        mx0 = fmaxf(mx0, s[nt][e]);
                        mx1 = fmaxf(mx1, s[nt][2 + e]);
                    }
                mx0 = fmaxf(mx0, __shfl_xor_sync(0xFFFFFFFFu, mx0, 1));
                mx0 = fmaxf(mx0, __shfl_xor_sync(0xFFFFFFFFu, mx0, 2));
                mx1 = fmaxf(mx1, __shfl_xor_sync(0xFFFFFFFFu, mx1, 1));
                mx1 = fmaxf(mx1, __shfl_xor_sync(0xFFFFFFFFu, mx1, 2));
                const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
                const float al0 = __expf(m0 - n0), al1 = __expf(m1 - n1);
                m0 = n0;
                m1 = n1;
                l0 *= al0;
                l1 *= al1;
                #pragma unroll
                for (unsigned i = 0; i < NO; ++i) {
                    o[i][0] *= al0; o[i][1] *= al0;
                    o[i][2] *= al1; o[i][3] *= al1;
                }
                #pragma unroll
                for (unsigned nt = 0; nt < NS; ++nt)
                    #pragma unroll
                    for (unsigned e = 0; e < 2; ++e) {
                        const float pa = s[nt][e] > NEG_HALF ? __expf(s[nt][e] - m0) : 0.0f;
                        const float pb = s[nt][2 + e] > NEG_HALF ? __expf(s[nt][2 + e] - m1) : 0.0f;
                        s[nt][e] = pa;
                        s[nt][2 + e] = pb;
                        l0 += pa;
                        l1 += pb;
                    }
                // O += P V: P straight from the score accumulators (C layout == A layout), FP16 operands.
                #pragma unroll
                for (unsigned kk = 0; kk < BC / 16; ++kk) {
                    const unsigned a0 = half2_bits(s[2 * kk][0], s[2 * kk][1]);
                    const unsigned a1 = half2_bits(s[2 * kk][2], s[2 * kk][3]);
                    const unsigned a2 = half2_bits(s[2 * kk + 1][0], s[2 * kk + 1][1]);
                    const unsigned a3 = half2_bits(s[2 * kk + 1][2], s[2 * kk + 1][3]);
                    const unsigned k0 = kk * 16 + 2 * t;
                    #pragma unroll
                    for (unsigned nt = 0; nt < NO; ++nt) {
                        const unsigned n = nt * 8 + g;
                        const unsigned b0 = bf16_pair_to_half2(sm.v[k0][n], sm.v[k0 + 1][n]);
                        const unsigned b1 = bf16_pair_to_half2(sm.v[k0 + 8][n], sm.v[k0 + 9][n]);
                        mma_f16(o[nt], a0, a1, a2, a3, b0, b1);
                    }
                }
            }
        }
    }

    l0 += __shfl_xor_sync(0xFFFFFFFFu, l0, 1);
    l0 += __shfl_xor_sync(0xFFFFFFFFu, l0, 2);
    l1 += __shfl_xor_sync(0xFFFFFFFFu, l1, 1);
    l1 += __shfl_xor_sync(0xFFFFFFFFu, l1, 2);
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
    #pragma unroll
    for (unsigned nt = 0; nt < NO; ++nt) {
        const unsigned col = h * HD + nt * 8 + 2 * t;
        if (ok0) {
            const __nv_bfloat162 v = __floats2bfloat162_rn(o[nt][0] * i0, o[nt][1] * i0);
            *reinterpret_cast<__nv_bfloat162*>(O + (size_t)(p0 - q_offset) * q_stride + col) = v;
        }
        if (ok1) {
            const __nv_bfloat162 v = __floats2bfloat162_rn(o[nt][2] * i1, o[nt][3] * i1);
            *reinterpret_cast<__nv_bfloat162*>(O + (size_t)(p1 - q_offset) * q_stride + col) = v;
        }
    }
}

}  // namespace psa

// ── Entry points ──────────────────────────────────────────────────────────────────────────────────────
// prefill_sparse_score_ada_{bf16,fp8}_hd{128,256}_s{4,8,16} and prefill_sparse_attn_ada_{bf16,fp8}_hd{128,256};
// the host table is `port/sparse_prefill/launch.rs` (ENTRY_SCORE, ENTRY_ATTN).

#define PSA_SCORE_ENTRY(NAME, HD, S, FP8)                                                                    \
    extern "C" __global__ void __launch_bounds__(psa::SCORE_THREADS) NAME(                                 \
        const __nv_bfloat16* __restrict__ Q, const void* __restrict__ Kc, const unsigned* __restrict__ bt, \
        float* __restrict__ scratch, unsigned* __restrict__ masks, unsigned long long* __restrict__ stats, \
        unsigned q_len, unsigned kv_len, unsigned q_offset, unsigned nq, unsigned nkv, unsigned bs,         \
        unsigned long long stride, unsigned q_block_lo, unsigned n_qb, unsigned nkb, unsigned mask_words,   \
        float logit_scale, float threshold) {                                                               \
        psa::score_body<HD, S, FP8>(Q, Kc, bt, scratch, masks, stats, q_len, kv_len, q_offset, nq, nkv, bs, \
                                    stride, q_block_lo, n_qb, nkb, mask_words, logit_scale, threshold);     \
    }                                                                                                       \
    AVAROK_EXPORT_DYN_SMEM(NAME, psa::ScoreTiles<HD>)

#define PSA_ATTN_ENTRY(NAME, HD, FP8)                                                                        \
    extern "C" __global__ void __launch_bounds__(psa::ATTN_THREADS, 1) NAME(                               \
        const __nv_bfloat16* __restrict__ Q, const void* __restrict__ Kc, const void* __restrict__ Vc,     \
        const unsigned* __restrict__ bt, const unsigned* __restrict__ masks, __nv_bfloat16* __restrict__ O, \
        unsigned q_len, unsigned kv_len, unsigned q_offset, unsigned nq, unsigned nkv, unsigned bs,         \
        unsigned long long stride, unsigned q_block_lo, unsigned n_qb, unsigned mask_words,                 \
        float inv_sqrt_d, float k_scale, float v_scale) {                                                   \
        psa::attn_body<HD, FP8>(Q, Kc, Vc, bt, masks, O, q_len, kv_len, q_offset, nq, nkv, bs, stride,      \
                                q_block_lo, n_qb, mask_words, inv_sqrt_d, k_scale, v_scale);                \
    }                                                                                                       \
    AVAROK_EXPORT_DYN_SMEM(NAME, psa::AttnTiles<HD>)

PSA_SCORE_ENTRY(prefill_sparse_score_ada_bf16_hd128_s4, 128, 4, false);
PSA_SCORE_ENTRY(prefill_sparse_score_ada_bf16_hd128_s8, 128, 8, false);
PSA_SCORE_ENTRY(prefill_sparse_score_ada_bf16_hd128_s16, 128, 16, false);
PSA_SCORE_ENTRY(prefill_sparse_score_ada_bf16_hd256_s4, 256, 4, false);
PSA_SCORE_ENTRY(prefill_sparse_score_ada_bf16_hd256_s8, 256, 8, false);
PSA_SCORE_ENTRY(prefill_sparse_score_ada_bf16_hd256_s16, 256, 16, false);
PSA_SCORE_ENTRY(prefill_sparse_score_ada_fp8_hd128_s4, 128, 4, true);
PSA_SCORE_ENTRY(prefill_sparse_score_ada_fp8_hd128_s8, 128, 8, true);
PSA_SCORE_ENTRY(prefill_sparse_score_ada_fp8_hd128_s16, 128, 16, true);
PSA_SCORE_ENTRY(prefill_sparse_score_ada_fp8_hd256_s4, 256, 4, true);
PSA_SCORE_ENTRY(prefill_sparse_score_ada_fp8_hd256_s8, 256, 8, true);
PSA_SCORE_ENTRY(prefill_sparse_score_ada_fp8_hd256_s16, 256, 16, true);

PSA_ATTN_ENTRY(prefill_sparse_attn_ada_bf16_hd128, 128, false);
PSA_ATTN_ENTRY(prefill_sparse_attn_ada_bf16_hd256, 256, false);
PSA_ATTN_ENTRY(prefill_sparse_attn_ada_fp8_hd128, 128, true);
PSA_ATTN_ENTRY(prefill_sparse_attn_ada_fp8_hd256, 256, true);
