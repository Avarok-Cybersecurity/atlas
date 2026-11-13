// SPDX-License-Identifier: AGPL-3.0-only
// Translated for Atlas from microsoft/RetrievalAttention (RetroInfer), MIT License,
// Copyright (c) Microsoft Corporation. Upstream: cache_hub/retroinfer_cache.py:765-770 (the
// centroid softmax, the GQA group sum, the empty-cluster mask and the top-k split into the
// retrieval and estimation zones) and the semantics of batch_gemm_softmax, commit 03f912c. The
// upstream CUTLASS kernels are not copied. Changes: written from the PyTorch reference for Ada
// (sm_89); scores, softmax and the group sum stay FP32 (upstream rounds them to BF16); empty
// clusters are left out of each query head's softmax as well as out of the selection (the Atlas
// contract, `state.rs` ZoneLists); torch.topk is replaced by an exact radix select on an
// order-preserving 32-bit key and a bitonic sort with a lowest-index tie-break; sizes, capacities
// and counts live on the device, so the three launches are capturable in a CUDA graph and replay
// unchanged for any index.

// Decode-time zone selection for one attention layer and one sequence.
//
//   retro_sel_stats  grid (tiles_cap, kv_heads), block 256: per 128-centroid tile, the logits
//                    scale * q_rg . c of the rows * group query heads of KV head h (in shared
//                    memory) and, per query head, the tile's max and sum of exp(logit - max).
//   retro_sel_prob   same grid: the logits again (bit-identical; the centroids are L2-hot), every
//                    query head's softmax denominator from the tile stats, and the group-summed
//                    probability p[c] = sum_rg exp(logit - M_rg) / L_rg into `scores` (-1 for an
//                    empty cluster).
//   retro_sel_pick   grid (kv_heads), block 1024: the order key bits(p) + 1 (0 = never), a radix
//                    select of the nprobe + es largest keys (four 8-bit digit passes), the stable
//                    compaction of that set, a bitonic sort by (key descending, id ascending), and
//                    the split into the retrieval zone (first nprobe) and the estimation zone.
//
// Inputs: Q rows [rows][num_q_heads][256] BF16 `q_row_stride` elements apart (query head
// h * group + g serves KV head h); centroids [kv_heads][cent_stride][256] BF16, cluster_size
// [kv_heads][cent_stride] i32 and the index header (N_CENTROIDS, NPROBE, ES at words 0, 1, 2).
// Outputs: scores [kv_heads][cent_stride] f32, retrieval [kv_heads][r_cap] i32, estimation
// [kv_heads][e_cap] i32, counts [kv_heads][2] i32 = (retrieval, estimation).
//
// Static shared memory: the two scoring kernels 32 * 256 * 2 + 32 * 128 * 4 + 256 = 33,024 bytes,
// retro_sel_pick about 34 KiB; all under ptxas's 48 KiB sm_89 ceiling (`select/tests.rs`).

#include <cuda_bf16.h>

#define RS_D 256
#define RS_TILE_C 128
#define RS_MAX_RG 32
#define RS_MAX_SELECTED 4096
#define RS_PICK_THREADS 1024
#define RS_FULL 0xffffffffu

#define RS_HDR_NC 0
#define RS_HDR_NPROBE 1
#define RS_HDR_ES 2

typedef unsigned long long u64;

__device__ __forceinline__ float rs_bf16(unsigned short s) {
    return __bfloat162float(__ushort_as_bfloat16(s));
}

__device__ __forceinline__ float rs_neg_inf() {
    return -__int_as_float(0x7f800000);
}

// Butterfly sum over the warp: every lane ends with the same bits.
__device__ __forceinline__ float rs_warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = __fadd_rn(v, __shfl_xor_sync(RS_FULL, v, off));
    return v;
}

__device__ __forceinline__ float rs_warp_max(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(RS_FULL, v, off));
    return v;
}

// Merge two (max, sum of exp(x - max)) pairs. Symmetric, so a butterfly gives every lane the same
// bits. A pair with max = -inf carries nothing.
__device__ __forceinline__ void rs_merge(float& m, float& l, float m2, float l2) {
    if (m2 == rs_neg_inf()) return;
    if (m == rs_neg_inf()) { m = m2; l = l2; return; }
    const float mm = fmaxf(m, m2);
    l = __fadd_rn(__fmul_rn(l, expf(__fsub_rn(m, mm))), __fmul_rn(l2, expf(__fsub_rn(m2, mm))));
    m = mm;
}

// Exclusive scan of two ints per thread over the CTA (every thread must call it).
__device__ __forceinline__ void rs_block_excl_scan2(int a, int b, int* out_a, int* out_b, int* wa, int* wb) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    const unsigned nwarps = blockDim.x >> 5;
    int x = a, y = b;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const int px = __shfl_up_sync(RS_FULL, x, off);
        const int py = __shfl_up_sync(RS_FULL, y, off);
        if (lane >= (unsigned)off) { x += px; y += py; }
    }
    if (lane == 31u) { wa[warp] = x; wb[warp] = y; }
    __syncthreads();
    if (warp == 0) {
        int u = (lane < nwarps) ? wa[lane] : 0;
        int v = (lane < nwarps) ? wb[lane] : 0;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const int pu = __shfl_up_sync(RS_FULL, u, off);
            const int pv = __shfl_up_sync(RS_FULL, v, off);
            if (lane >= (unsigned)off) { u += pu; v += pv; }
        }
        if (lane < nwarps) { wa[lane] = u; wb[lane] = v; }
    }
    __syncthreads();
    *out_a = ((warp > 0) ? wa[warp - 1] : 0) + x - a;
    *out_b = ((warp > 0) ? wb[warp - 1] : 0) + y - b;
    __syncthreads();
}

// The logits of one tile into slog[rg][j]: lane l sums dims 8l .. 8l+7 in order (products
// rounded), the warp butterfly adds the lanes, then the scale. Empty clusters and clusters past nc
// get -inf. Called by all 256 threads; ends with a __syncthreads.
__device__ __forceinline__ void rs_tile_logits(
    const __nv_bfloat16* Q, unsigned q_row_stride, unsigned rg_n, unsigned group, unsigned h,
    const __nv_bfloat16* cent, const int* csize, unsigned cent_stride, unsigned nc, unsigned c0,
    float scale, unsigned short (*sq)[RS_D], float (*slog)[RS_TILE_C]) {
    const unsigned tid = threadIdx.x;
    const unsigned warp = tid >> 5;
    const unsigned lane = tid & 31u;
    for (unsigned i = tid; i < rg_n * RS_D; i += blockDim.x) {
        const unsigned rg = i / RS_D;
        const unsigned d = i % RS_D;
        const unsigned qh = h * group + rg % group;
        sq[rg][d] = __bfloat16_as_ushort(Q[(u64)(rg / group) * q_row_stride + (u64)qh * RS_D + d]);
    }
    __syncthreads();
    for (unsigned k = 0; k < RS_TILE_C / 8; k++) {
        const unsigned j = warp * (RS_TILE_C / 8) + k;
        const unsigned c = c0 + j;
        const bool live = c < nc && csize[(u64)h * cent_stride + c] > 0;  // warp-uniform
        if (live) {
            const uint4 u = *(const uint4*)(cent + ((u64)h * cent_stride + c) * RS_D + lane * 8);
            const unsigned w[4] = {u.x, u.y, u.z, u.w};
            float x[8];
            #pragma unroll
            for (int e = 0; e < 8; e++) x[e] = rs_bf16((unsigned short)((e & 1) ? (w[e >> 1] >> 16) : (w[e >> 1] & 0xffffu)));
            for (unsigned rg = 0; rg < rg_n; rg++) {
                float p = 0.0f;
                #pragma unroll
                for (int e = 0; e < 8; e++) p = __fadd_rn(p, __fmul_rn(rs_bf16(sq[rg][lane * 8 + e]), x[e]));
                p = rs_warp_sum(p);
                if (lane == 0) slog[rg][j] = __fmul_rn(p, scale);
            }
        } else if (lane == 0) {
            for (unsigned rg = 0; rg < rg_n; rg++) slog[rg][j] = rs_neg_inf();
        }
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------------------------

extern "C" __global__ void __launch_bounds__(256) retro_sel_stats(
    const __nv_bfloat16* __restrict__ Q, unsigned q_row_stride, unsigned rows, unsigned group,
    const __nv_bfloat16* __restrict__ cent, const int* __restrict__ csize, unsigned cent_stride,
    const int* __restrict__ hdr, float* __restrict__ tmax, float* __restrict__ tsum, unsigned tiles_cap,
    float scale) {
    __shared__ __align__(16) unsigned short sq[RS_MAX_RG][RS_D];
    __shared__ float slog[RS_MAX_RG][RS_TILE_C];
    const unsigned tile = blockIdx.x;
    const unsigned h = blockIdx.y;
    const int nci = hdr[RS_HDR_NC];
    const unsigned nc = nci > 0 ? (unsigned)nci : 0u;
    const unsigned c0 = tile * RS_TILE_C;
    if (c0 >= nc) return;  // CTA-uniform
    const unsigned rg_n = rows * group;
    rs_tile_logits(Q, q_row_stride, rg_n, group, h, cent, csize, cent_stride, nc, c0, scale, sq, slog);
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    for (unsigned rg = warp; rg < rg_n; rg += blockDim.x >> 5) {
        float v[RS_TILE_C / 32];
        float m = rs_neg_inf();
        #pragma unroll
        for (int e = 0; e < RS_TILE_C / 32; e++) { v[e] = slog[rg][lane + 32 * e]; m = fmaxf(m, v[e]); }
        m = rs_warp_max(m);
        float l = 0.0f;
        #pragma unroll
        for (int e = 0; e < RS_TILE_C / 32; e++) {
            if (v[e] != rs_neg_inf()) l = __fadd_rn(l, expf(__fsub_rn(v[e], m)));
        }
        l = rs_warp_sum(l);
        if (lane == 0) {
            const u64 si = ((u64)h * rg_n + rg) * tiles_cap + tile;
            tmax[si] = m;
            tsum[si] = l;
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) retro_sel_prob(
    const __nv_bfloat16* __restrict__ Q, unsigned q_row_stride, unsigned rows, unsigned group,
    const __nv_bfloat16* __restrict__ cent, const int* __restrict__ csize, unsigned cent_stride,
    const int* __restrict__ hdr, const float* __restrict__ tmax, const float* __restrict__ tsum,
    unsigned tiles_cap, float scale, float* __restrict__ scores) {
    __shared__ __align__(16) unsigned short sq[RS_MAX_RG][RS_D];
    __shared__ float slog[RS_MAX_RG][RS_TILE_C];
    __shared__ float sM[RS_MAX_RG];
    __shared__ float sL[RS_MAX_RG];
    const unsigned tile = blockIdx.x;
    const unsigned h = blockIdx.y;
    const int nci = hdr[RS_HDR_NC];
    const unsigned nc = nci > 0 ? (unsigned)nci : 0u;
    const unsigned c0 = tile * RS_TILE_C;
    if (c0 >= nc) return;  // CTA-uniform
    const unsigned rg_n = rows * group;
    const unsigned n_tiles = (nc + RS_TILE_C - 1) / RS_TILE_C;
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    // Every CTA combines the same stats in the same order, so every CTA has the same M and L.
    for (unsigned rg = warp; rg < rg_n; rg += blockDim.x >> 5) {
        float m = rs_neg_inf(), l = 0.0f;
        const u64 sb = ((u64)h * rg_n + rg) * tiles_cap;
        for (unsigned t = lane; t < n_tiles; t += 32) rs_merge(m, l, tmax[sb + t], tsum[sb + t]);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float m2 = __shfl_xor_sync(RS_FULL, m, off);
            const float l2 = __shfl_xor_sync(RS_FULL, l, off);
            rs_merge(m, l, m2, l2);
        }
        if (lane == 0) { sM[rg] = m; sL[rg] = l; }
    }
    rs_tile_logits(Q, q_row_stride, rg_n, group, h, cent, csize, cent_stride, nc, c0, scale, sq, slog);
    for (unsigned j = threadIdx.x; j < RS_TILE_C; j += blockDim.x) {
        const unsigned c = c0 + j;
        if (c >= nc) continue;
        float p = -1.0f;
        if (slog[0][j] != rs_neg_inf()) {  // a live cluster has a finite logit for every head
            p = 0.0f;
            for (unsigned rg = 0; rg < rg_n; rg++) {
                p = __fadd_rn(p, __fdiv_rn(expf(__fsub_rn(slog[rg][j], sM[rg])), sL[rg]));
            }
        }
        scores[(u64)h * cent_stride + c] = p;
    }
}

// ---------------------------------------------------------------------------------------------

__device__ __forceinline__ unsigned rs_key(float p) {
    return p >= 0.0f ? __float_as_uint(p) + 1u : 0u;
}

extern "C" __global__ void __launch_bounds__(RS_PICK_THREADS) retro_sel_pick(
    const float* __restrict__ scores, unsigned cent_stride, const int* __restrict__ hdr,
    int* __restrict__ retrieval, int* __restrict__ estimation, int* __restrict__ counts,
    unsigned r_cap, unsigned e_cap) {
    __shared__ int hist[256];
    __shared__ int st[2];  // prefix, need
    __shared__ int wa[32];
    __shared__ int wb[32];
    __shared__ int n_valid_s;
    __shared__ unsigned rkey[RS_MAX_SELECTED];
    __shared__ int ridx[RS_MAX_SELECTED];

    const unsigned h = blockIdx.x;
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u;
    const int nci = hdr[RS_HDR_NC];
    if (nci <= 0) {
        if (tid == 0) { counts[h * 2] = 0; counts[h * 2 + 1] = 0; }
        return;
    }
    const unsigned nc = (unsigned)nci;
    const int nprobe = hdr[RS_HDR_NPROBE];
    const int es = hdr[RS_HDR_ES];
    const float* sc = scores + (u64)h * cent_stride;

    int local_valid = 0;
    for (unsigned c = tid; c < nc; c += blockDim.x) local_valid += rs_key(sc[c]) != 0u ? 1 : 0;
    int vbase = 0, ignore = 0;
    rs_block_excl_scan2(local_valid, 0, &vbase, &ignore, wa, wb);
    if (tid == blockDim.x - 1) n_valid_s = vbase + local_valid;
    __syncthreads();
    const int n_valid = n_valid_s;

    int kr = nprobe < n_valid ? nprobe : n_valid;
    if (kr > (int)r_cap) kr = (int)r_cap;
    if (kr > RS_MAX_SELECTED) kr = RS_MAX_SELECTED;
    if (kr < 0) kr = 0;
    int ks = nprobe + es;
    if (ks > n_valid) ks = n_valid;
    if (ks - kr > (int)e_cap) ks = kr + (int)e_cap;
    if (ks > RS_MAX_SELECTED) ks = RS_MAX_SELECTED;
    if (ks < kr) ks = kr;

    // 1. Radix select of the ks-th largest key, four 8-bit digits.
    if (tid == 0) { st[0] = 0; st[1] = ks; }
    __syncthreads();
    for (int pass = 0; pass < 4 && ks > 0; pass++) {
        const unsigned shift = 24u - 8u * (unsigned)pass;
        const unsigned hi_mask = pass == 0 ? 0u : (0xffffffffu << (shift + 8u));
        for (unsigned i = tid; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        const unsigned pre = (unsigned)st[0];
        for (unsigned c = tid; c < nc; c += blockDim.x) {
            const unsigned key = rs_key(sc[c]);
            if (key != 0u && (key & hi_mask) == pre) atomicAdd(&hist[(key >> shift) & 255u], 1);
        }
        __syncthreads();
        if (tid < 32) {
            // Lane l covers bins 255 - 8l down to 248 - 8l: lane order is descending key order.
            int bins[8];
            int sum = 0;
            #pragma unroll
            for (int e = 0; e < 8; e++) { bins[e] = hist[255 - lane * 8 - e]; sum += bins[e]; }
            int incl = sum;
            #pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                const int y = __shfl_up_sync(RS_FULL, incl, off);
                if (lane >= (unsigned)off) incl += y;
            }
            const int excl = incl - sum;
            const int k = st[1];
            __syncwarp();
            if (excl < k && k <= incl) {
                int cum = excl;
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    if (cum + bins[e] >= k) {
                        st[0] = (int)(pre | ((unsigned)(255 - lane * 8 - e) << shift));
                        st[1] = k - cum;
                        break;
                    }
                    cum += bins[e];
                }
            }
        }
        __syncthreads();
    }
    const unsigned thr = ks > 0 ? (unsigned)st[0] : 0xffffffffu;
    const int need = ks > 0 ? st[1] : 0;

    // 2. Stable compaction of the selected set over contiguous chunks (thread order = id order).
    const unsigned ch = (nc + blockDim.x - 1) / blockDim.x;
    unsigned c0 = tid * ch, c1 = c0 + ch;
    if (c0 > nc) c0 = nc;
    if (c1 > nc) c1 = nc;
    int eq = 0;
    for (unsigned c = c0; c < c1; c++) eq += rs_key(sc[c]) == thr ? 1 : 0;
    int eq_base = 0;
    rs_block_excl_scan2(eq, 0, &eq_base, &ignore, wa, wb);
    int nsel = 0;
    {
        int er = eq_base;
        for (unsigned c = c0; c < c1; c++) {
            const unsigned key = rs_key(sc[c]);
            const bool in = key > thr || (key == thr && er < need);
            if (key == thr) er++;
            nsel += in ? 1 : 0;
        }
    }
    int sel_base = 0;
    rs_block_excl_scan2(nsel, 0, &sel_base, &ignore, wa, wb);
    {
        int er = eq_base, pos = sel_base;
        for (unsigned c = c0; c < c1; c++) {
            const unsigned key = rs_key(sc[c]);
            const bool in = key > thr || (key == thr && er < need);
            if (key == thr) er++;
            if (in) {
                if (pos < RS_MAX_SELECTED) { rkey[pos] = key; ridx[pos] = (int)c; }
                pos++;
            }
        }
    }
    __syncthreads();

    // 3. Bitonic sort of the selected set: key descending, then cluster id ascending.
    unsigned p2 = 1;
    while (p2 < (unsigned)ks) p2 <<= 1;
    for (unsigned i = tid; i < p2; i += blockDim.x) {
        if (i >= (unsigned)ks) { rkey[i] = 0u; ridx[i] = 0x7fffffff; }
    }
    __syncthreads();
    for (unsigned k = 2; k <= p2; k <<= 1) {
        for (unsigned j = k >> 1; j > 0; j >>= 1) {
            for (unsigned i = tid; i < p2; i += blockDim.x) {
                const unsigned ixj = i ^ j;
                if (ixj > i) {
                    const unsigned ka = rkey[i], kb = rkey[ixj];
                    const int ia = ridx[i], ib = ridx[ixj];
                    const bool b_first = kb > ka || (kb == ka && ib < ia);
                    const bool a_first = ka > kb || (ka == kb && ia < ib);
                    if (((i & k) == 0) ? b_first : a_first) {
                        rkey[i] = kb; rkey[ixj] = ka;
                        ridx[i] = ib; ridx[ixj] = ia;
                    }
                }
            }
            __syncthreads();
        }
    }

    // 4. The first kr are the retrieval zone, the rest the estimation zone, both in rank order.
    for (unsigned i = tid; i < (unsigned)ks; i += blockDim.x) {
        if (i < (unsigned)kr) retrieval[(u64)h * r_cap + i] = ridx[i];
        else estimation[(u64)h * e_cap + (i - (unsigned)kr)] = ridx[i];
    }
    if (tid == 0) { counts[h * 2] = kr; counts[h * 2 + 1] = ks - kr; }
}
