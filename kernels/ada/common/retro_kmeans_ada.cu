// SPDX-License-Identifier: AGPL-3.0-only
// Translated for Atlas from microsoft/RetrievalAttention (RetroInfer), MIT License,
// Copyright (c) Microsoft Corporation. Upstream: cache_hub/kmeans.py (segment_k_means,
// _triton_assign_kernel, _triton_update_kernel, triton_index_add, triton_reverse_index) and
// cache_hub/retroinfer_cache.py:533-554 (key centering, the meta index), commit 03f912c.
// Changes: rewritten from Triton to CUDA for Ada (sm_89); keys and values are read in place from
// Atlas's paged BF16 or FP8 KV pool through the block table instead of a contiguous copy; the
// assignment is a BF16 mma.sync m16n8k16 tile with an exact lowest-index tie-break; every float
// reduction runs in a fixed order (the per-token atomics are replaced by a stable two-level counting
// sort into a CSR, and each centroid sums its members in ascending token order), so two builds are
// bitwise equal; the cluster-to-token lists are a CSR of absolute positions instead of a padded
// [n_centroids, max_cluster_size] table, written at caller-given offsets so a decode-time update
// appends to an existing index.

// RetroInfer index build: segmented spherical k-means over one attention layer's keys.
//
// Per KV head h and the build's tokens t in [0, N) at absolute positions start + t:
//   mean       = bf16(sum_t k[t] / N)                         (retro_km_mean_partial/_final)
//   c[t]       = bf16(k[t] - mean)                            (centered key, BF16 as upstream)
//   centroid i = c[trunc(f32(i) * f32(N / nc) + f32(N / nc / 2))]               (retro_km_init)
//   iters - 1 passes over segments (tokens [s*tps, (s+1)*tps) against centroids [s*cps, (s+1)*cps)):
//     assign t -> argmax_i c[t] . centroid_i (lowest i on a tie)                 (retro_km_assign)
//     stable CSR of the assignment (count, scan, scatter)        (retro_km_count/_scan/_scatter)
//     centroid_i = normalize(mean of its members), kept when empty               (retro_km_update)
//   1 global pass (all N tokens against all nc centroids): assign, CSR into the index,
//     centroid_i = bf16(bf16(mean of members) + mean), value_sum_i = bf16(sum of member values)
//
// Layouts: pool [num_blocks][block_size][kv_heads][256] BF16 or FP8 E4M3, blocks `block_stride`
// bytes apart. Every other array is per KV head with an explicit per-head stride (in elements), and
// the host passes index pointers already offset to the build's first centroid, so an update writes
// behind the clusters already there: centroids and value_sum rows of 256 BF16, cluster sizes i32,
// CSR starts i32 (one more entry than clusters: the end), members i32 absolute positions. Scratch
// (from the host's build scratch): assign i32 (-1 = not in this pass), hist [RK_TILES][hist_stride]
// i32 per head.
//
// Shared memory is static and under ptxas's 48 KiB ceiling for sm_89 (the largest, retro_km_assign,
// uses 64 * 264 * 2 + 256 * 4 = 34,816 bytes); `build/tests.rs` recomputes it from the defines below.
// Built with common/KERNEL.toml's --fmad=false; the arithmetic below is written for that, and the
// CPU mirror in `crates/spark-model/src/port/retro/build/reference.rs` reproduces it bit for bit
// everywhere except the tensor-core dot products of retro_km_assign.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define RK_D 256
#define RK_TILES 64
#define RK_MEAN_CHUNKS 64
#define RK_FULL 0xffffffffu
#define RK_ASSIGN_THREADS 256
#define RK_ASSIGN_ROWS 128
#define RK_ASSIGN_CHUNK 64
#define RK_SB_STRIDE (RK_D + 8)
#define RK_SCAN_THREADS 1024
#define RK_WARPS_PER_CTA 8

typedef unsigned long long u64;

// ---------------------------------------------------------------------------------------------
// Paged pool access
// ---------------------------------------------------------------------------------------------

__device__ __forceinline__ const unsigned char* rk_row(
    const unsigned char* pool, const int* bt, unsigned pos, unsigned block_size,
    u64 block_stride, unsigned kv_heads, unsigned h, unsigned esize) {
    const unsigned blk = (unsigned)bt[pos / block_size];
    return pool + (u64)blk * block_stride + ((u64)(pos % block_size) * kv_heads + h) * (u64)(RK_D * esize);
}

__device__ __forceinline__ float rk_fp8(unsigned b) {
    __half_raw r = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)(b & 0xffu), __NV_E4M3);
    return __half2float(__half(r));
}

__device__ __forceinline__ float rk_bf16(unsigned short s) {
    return __bfloat162float(__ushort_as_bfloat16(s));
}

__device__ __forceinline__ float rk_round_bf16(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

// Element d of a row, dequantized (FP8: value * scale; BF16: exact).
__device__ __forceinline__ float rk_load1(const unsigned char* row, unsigned d, unsigned fp8, float scale) {
    if (fp8) return __fmul_rn(rk_fp8(row[d]), scale);
    return rk_bf16(((const unsigned short*)row)[d]);
}

// Elements d0 .. d0 + 7 of a row (d0 a multiple of 8), dequantized.
__device__ __forceinline__ void rk_load8(const unsigned char* row, unsigned d0, unsigned fp8, float scale, float* x) {
    if (fp8) {
        const uint2 u = *(const uint2*)(row + d0);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const unsigned w = (j < 4) ? u.x : u.y;
            x[j] = __fmul_rn(rk_fp8(w >> ((j & 3) * 8)), scale);
        }
    } else {
        const uint4 u = *(const uint4*)(row + (u64)d0 * 2);
        const unsigned w[4] = {u.x, u.y, u.z, u.w};
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const unsigned v = w[j >> 1];
            x[j] = rk_bf16((unsigned short)((j & 1) ? (v >> 16) : (v & 0xffffu)));
        }
    }
}

// The BF16 pair (d, d + 1) of a centered key row, packed low element first (an mma A operand).
__device__ __forceinline__ unsigned rk_centered_pair(
    const unsigned char* row, unsigned d, unsigned fp8, float scale, const float* mean) {
    float x0, x1;
    if (fp8) {
        const unsigned short u = *(const unsigned short*)(row + d);
        x0 = __fmul_rn(rk_fp8(u), scale);
        x1 = __fmul_rn(rk_fp8((unsigned)u >> 8), scale);
    } else {
        const unsigned u = *(const unsigned*)(row + (u64)d * 2);
        x0 = rk_bf16((unsigned short)(u & 0xffffu));
        x1 = rk_bf16((unsigned short)(u >> 16));
    }
    const unsigned short c0 = __bfloat16_as_ushort(__float2bfloat16_rn(__fsub_rn(x0, mean[d])));
    const unsigned short c1 = __bfloat16_as_ushort(__float2bfloat16_rn(__fsub_rn(x1, mean[d + 1])));
    return (unsigned)c0 | ((unsigned)c1 << 16);
}

// Exclusive scan of one int per thread over the whole CTA (all threads must call it).
__device__ __forceinline__ int rk_block_excl_scan(int v, int* warp_sums) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    const unsigned nwarps = blockDim.x >> 5;
    int x = v;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const int y = __shfl_up_sync(RK_FULL, x, off);
        if (lane >= (unsigned)off) x += y;
    }
    if (lane == 31u) warp_sums[warp] = x;
    __syncthreads();
    if (warp == 0) {
        int w = (lane < nwarps) ? warp_sums[lane] : 0;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const int y = __shfl_up_sync(RK_FULL, w, off);
            if (lane >= (unsigned)off) w += y;
        }
        if (lane < nwarps) warp_sums[lane] = w;
    }
    __syncthreads();
    const int base = (warp > 0) ? warp_sums[warp - 1] : 0;
    __syncthreads();
    return base + x - v;
}

// ---------------------------------------------------------------------------------------------
// Key mean
// ---------------------------------------------------------------------------------------------

// Grid (RK_MEAN_CHUNKS, kv_heads), block 256 (one thread per dimension). Chunk c sums tokens
// [c * chunk_len, (c + 1) * chunk_len) in ascending order. partial: [kv_heads][RK_MEAN_CHUNKS][256].
extern "C" __global__ void __launch_bounds__(256) retro_km_mean_partial(
    const unsigned char* __restrict__ k_pool, const int* __restrict__ bt, unsigned block_size,
    u64 block_stride, unsigned kv_heads, unsigned fp8, float k_scale,
    float* __restrict__ partial, unsigned n_tok, unsigned start, unsigned chunk_len) {
    const unsigned chunk = blockIdx.x;
    const unsigned h = blockIdx.y;
    const unsigned d = threadIdx.x;
    const unsigned esize = fp8 ? 1u : 2u;
    const unsigned t_begin = chunk * chunk_len;
    unsigned t_end = t_begin + chunk_len;
    if (t_end > n_tok) t_end = n_tok;
    float acc = 0.0f;
    for (unsigned t = t_begin; t < t_end; t++) {
        const unsigned char* row = rk_row(k_pool, bt, start + t, block_size, block_stride, kv_heads, h, esize);
        acc = __fadd_rn(acc, rk_load1(row, d, fp8, k_scale));
    }
    partial[((u64)h * RK_MEAN_CHUNKS + chunk) * RK_D + d] = acc;
}

// Grid (kv_heads), block 256. mean[h][d] = bf16(sum of the chunk sums in order / n_tok), as f32.
extern "C" __global__ void __launch_bounds__(256) retro_km_mean_final(
    const float* __restrict__ partial, float* __restrict__ mean, unsigned n_tok) {
    const unsigned h = blockIdx.x;
    const unsigned d = threadIdx.x;
    float acc = 0.0f;
    for (unsigned c = 0; c < RK_MEAN_CHUNKS; c++) {
        acc = __fadd_rn(acc, partial[((u64)h * RK_MEAN_CHUNKS + c) * RK_D + d]);
    }
    mean[h * RK_D + d] = rk_round_bf16(__fdiv_rn(acc, (float)n_tok));
}

// ---------------------------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------------------------

// Grid (ceil(nc / 8), kv_heads), block 256 (one warp per centroid). cent: the build's first
// centroid row, cent_stride rows per head.
extern "C" __global__ void __launch_bounds__(256) retro_km_init(
    const unsigned char* __restrict__ k_pool, const int* __restrict__ bt, unsigned block_size,
    u64 block_stride, unsigned kv_heads, unsigned fp8, float k_scale,
    const float* __restrict__ mean, __nv_bfloat16* __restrict__ cent, unsigned cent_stride,
    unsigned nc, unsigned n_tok, unsigned start, float step, float half_step) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned i = blockIdx.x * RK_WARPS_PER_CTA + (threadIdx.x >> 5);
    const unsigned h = blockIdx.y;
    if (i >= nc) return;
    const float fi = __fadd_rn(__fmul_rn((float)i, step), half_step);
    unsigned t = (unsigned)fi;
    if (t >= n_tok) t = n_tok - 1;
    const unsigned char* row = rk_row(k_pool, bt, start + t, block_size, block_stride, kv_heads, h, fp8 ? 1u : 2u);
    float x[8];
    rk_load8(row, lane * 8, fp8, k_scale, x);
    const float* m = mean + h * RK_D + lane * 8;
    __nv_bfloat16* out = cent + ((u64)h * cent_stride + i) * RK_D + lane * 8;
    #pragma unroll
    for (int j = 0; j < 8; j++) out[j] = __float2bfloat16_rn(__fsub_rn(x[j], m[j]));
}

// ---------------------------------------------------------------------------------------------
// Assignment (tensor cores)
// ---------------------------------------------------------------------------------------------

// Grid (ceil(n_tok / 128), kv_heads), block 256: 8 warps x 16 tokens. Token t of a training pass
// belongs to segment t / seg_tokens and competes only for that segment's centroids
// [s * seg_cents, (s + 1) * seg_cents); tokens past n_seg * seg_tokens sit the pass out (-1). The
// final pass is n_seg = 1, seg_tokens = n_tok, seg_cents = nc. Scores are FP32-accumulated BF16 dot
// products of the centered key and the BF16 centroid; the lowest centroid index wins a tie.
extern "C" __global__ void __launch_bounds__(RK_ASSIGN_THREADS, 1) retro_km_assign(
    const unsigned char* __restrict__ k_pool, const int* __restrict__ bt, unsigned block_size,
    u64 block_stride, unsigned kv_heads, unsigned fp8, float k_scale,
    const float* __restrict__ mean, const __nv_bfloat16* __restrict__ cent, unsigned cent_stride,
    int* __restrict__ assign, unsigned asg_stride, unsigned n_tok, unsigned start, unsigned seg_tokens,
    unsigned seg_cents, unsigned n_seg) {
    __shared__ __align__(16) unsigned short sB[RK_ASSIGN_CHUNK][RK_SB_STRIDE];
    __shared__ float s_mean[RK_D];

    const unsigned h = blockIdx.y;
    const unsigned tid = threadIdx.x;
    const unsigned warp = tid >> 5;
    const unsigned lane = tid & 31u;
    const unsigned g = lane >> 2;
    const unsigned tig = lane & 3u;
    const unsigned t0 = blockIdx.x * RK_ASSIGN_ROWS;
    const unsigned trained = n_seg * seg_tokens;
    int* out = assign + (u64)h * asg_stride;

    if (t0 >= trained) {  // CTA-uniform: every row of this CTA sits the pass out
        if (tid < RK_ASSIGN_ROWS && t0 + tid < n_tok) out[t0 + tid] = -1;
        return;
    }
    for (unsigned d = tid; d < RK_D; d += RK_ASSIGN_THREADS) s_mean[d] = mean[h * RK_D + d];
    __syncthreads();

    const unsigned esize = fp8 ? 1u : 2u;
    const unsigned r0 = t0 + warp * 16 + g;
    const unsigned r1 = r0 + 8;
    const bool v0 = r0 < n_tok && r0 < trained;
    const bool v1 = r1 < n_tok && r1 < trained;
    const unsigned char* row0 = v0 ? rk_row(k_pool, bt, start + r0, block_size, block_stride, kv_heads, h, esize) : nullptr;
    const unsigned char* row1 = v1 ? rk_row(k_pool, bt, start + r1, block_size, block_stride, kv_heads, h, esize) : nullptr;

    unsigned a[16][4];
    #pragma unroll
    for (int ks = 0; ks < 16; ks++) {
        const unsigned d0 = ks * 16 + tig * 2;
        a[ks][0] = v0 ? rk_centered_pair(row0, d0, fp8, k_scale, s_mean) : 0u;
        a[ks][1] = v1 ? rk_centered_pair(row1, d0, fp8, k_scale, s_mean) : 0u;
        a[ks][2] = v0 ? rk_centered_pair(row0, d0 + 8, fp8, k_scale, s_mean) : 0u;
        a[ks][3] = v1 ? rk_centered_pair(row1, d0 + 8, fp8, k_scale, s_mean) : 0u;
    }

    unsigned t_last = t0 + RK_ASSIGN_ROWS - 1;
    const unsigned lim = (n_tok < trained ? n_tok : trained) - 1;
    if (t_last > lim) t_last = lim;
    const unsigned c_lo = (t0 / seg_tokens) * seg_cents;
    const unsigned c_hi = (t_last / seg_tokens + 1) * seg_cents;
    const unsigned lo0 = v0 ? (r0 / seg_tokens) * seg_cents : 0u;
    const unsigned lo1 = v1 ? (r1 / seg_tokens) * seg_cents : 0u;
    const unsigned hi0 = v0 ? lo0 + seg_cents : 0u;
    const unsigned hi1 = v1 ? lo1 + seg_cents : 0u;
    const __nv_bfloat16* ch = cent + (u64)h * cent_stride * RK_D;

    const float NEG_INF = -__int_as_float(0x7f800000);
    float best0 = NEG_INF, best1 = NEG_INF;
    int bi0 = -1, bi1 = -1;

    for (unsigned cb = c_lo; cb < c_hi; cb += RK_ASSIGN_CHUNK) {
        __syncthreads();
        for (unsigned i = tid; i < RK_ASSIGN_CHUNK * (RK_D / 8); i += RK_ASSIGN_THREADS) {
            const unsigned rr = i >> 5;
            const unsigned q = i & 31u;
            const unsigned c = cb + rr;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (c < c_hi) v = *(const uint4*)(ch + (u64)c * RK_D + q * 8);
            *(uint4*)&sB[rr][q * 8] = v;
        }
        __syncthreads();

        float acc[8][4];
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            acc[nt][0] = 0.0f; acc[nt][1] = 0.0f; acc[nt][2] = 0.0f; acc[nt][3] = 0.0f;
            #pragma unroll
            for (int ks = 0; ks < 16; ks++) {
                const unsigned b0 = *(const unsigned*)&sB[nt * 8 + g][ks * 16 + tig * 2];
                const unsigned b1 = *(const unsigned*)&sB[nt * 8 + g][ks * 16 + 8 + tig * 2];
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                    : "=f"(acc[nt][0]), "=f"(acc[nt][1]), "=f"(acc[nt][2]), "=f"(acc[nt][3])
                    : "r"(a[ks][0]), "r"(a[ks][1]), "r"(a[ks][2]), "r"(a[ks][3]), "r"(b0), "r"(b1),
                      "f"(acc[nt][0]), "f"(acc[nt][1]), "f"(acc[nt][2]), "f"(acc[nt][3]));
            }
        }
        // Columns ascend across chunks, n-tiles and the pair, so strict `>` keeps the lowest index.
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            #pragma unroll
            for (int e = 0; e < 2; e++) {
                const unsigned col = cb + nt * 8 + tig * 2 + e;
                if (v0 && col >= lo0 && col < hi0 && acc[nt][e] > best0) { best0 = acc[nt][e]; bi0 = (int)col; }
                if (v1 && col >= lo1 && col < hi1 && acc[nt][2 + e] > best1) { best1 = acc[nt][2 + e]; bi1 = (int)col; }
            }
        }
    }

    // The four lanes of a row group hold disjoint columns: keep the larger score, the lower index
    // on a tie (an unset -1 never beats a set index).
    #pragma unroll
    for (int off = 1; off <= 2; off <<= 1) {
        const float ob0 = __shfl_xor_sync(RK_FULL, best0, off);
        const int oi0 = __shfl_xor_sync(RK_FULL, bi0, off);
        const float ob1 = __shfl_xor_sync(RK_FULL, best1, off);
        const int oi1 = __shfl_xor_sync(RK_FULL, bi1, off);
        if (oi0 >= 0 && (bi0 < 0 || ob0 > best0 || (ob0 == best0 && oi0 < bi0))) { best0 = ob0; bi0 = oi0; }
        if (oi1 >= 0 && (bi1 < 0 || ob1 > best1 || (ob1 == best1 && oi1 < bi1))) { best1 = ob1; bi1 = oi1; }
    }
    if (tig == 0) {
        if (r0 < n_tok) out[r0] = v0 ? bi0 : -1;
        if (r1 < n_tok) out[r1] = v1 ? bi1 : -1;
    }
}

// ---------------------------------------------------------------------------------------------
// Stable CSR of an assignment
// ---------------------------------------------------------------------------------------------

// Grid (n_tiles, kv_heads), block 256. hist[h][tile][c] = members of c among the tile's tokens
// (zeroed by the host before the launch; integer atomics, so the counts are exact).
extern "C" __global__ void __launch_bounds__(256) retro_km_count(
    const int* __restrict__ assign, unsigned asg_stride, int* __restrict__ hist, unsigned hist_stride,
    unsigned n_tok, unsigned tile_tokens) {
    const unsigned tile = blockIdx.x;
    const unsigned h = blockIdx.y;
    const unsigned tb = tile * tile_tokens;
    unsigned te = tb + tile_tokens;
    if (te > n_tok) te = n_tok;
    int* row = hist + ((u64)h * RK_TILES + tile) * hist_stride;
    const int* a = assign + (u64)h * asg_stride;
    for (unsigned t = tb + threadIdx.x; t < te; t += blockDim.x) {
        const int c = a[t];
        if (c >= 0) atomicAdd(&row[c], 1);
    }
}

// Grid (kv_heads), block 1024. Thread i owns clusters [i * ch, (i + 1) * ch). Writes
// out_start[c] = base + exclusive prefix of the sizes in cluster order, out_start[nc] = base + n,
// out_size[c], and rewrites hist[h][tile][c] in place into the first slot tile `tile` writes for c.
extern "C" __global__ void __launch_bounds__(RK_SCAN_THREADS) retro_km_scan(
    int* __restrict__ hist, unsigned hist_stride, int* __restrict__ out_start, unsigned start_stride,
    int* __restrict__ out_size, unsigned size_stride, unsigned nc, unsigned n_tiles, int base) {
    __shared__ int warp_sums[32];
    const unsigned h = blockIdx.x;
    const unsigned ch = (nc + RK_SCAN_THREADS - 1) / RK_SCAN_THREADS;
    unsigned c0 = threadIdx.x * ch;
    unsigned c1 = c0 + ch;
    if (c0 > nc) c0 = nc;
    if (c1 > nc) c1 = nc;
    int* hh = hist + (u64)h * RK_TILES * hist_stride;
    int* os = out_start + (u64)h * start_stride;
    int* oz = out_size + (u64)h * size_stride;
    int local = 0;
    for (unsigned c = c0; c < c1; c++) {
        for (unsigned t = 0; t < n_tiles; t++) local += hh[(u64)t * hist_stride + c];
    }
    int run = base + rk_block_excl_scan(local, warp_sums);
    for (unsigned c = c0; c < c1; c++) {
        os[c] = run;
        int s = 0;
        for (unsigned t = 0; t < n_tiles; t++) {
            int* p = hh + (u64)t * hist_stride + c;
            const int v = *p;
            *p = run + s;
            s += v;
        }
        oz[c] = s;
        run += s;
    }
    if (c0 < c1 && c1 == nc) os[nc] = run;
}

// Grid (n_tiles, kv_heads), block 32: one warp walks its tile in token order, 32 tokens at a time,
// so every cluster's members land in ascending token order (stable). members[h][slot] = start + t.
extern "C" __global__ void __launch_bounds__(32) retro_km_scatter(
    const int* __restrict__ assign, unsigned asg_stride, int* __restrict__ hist, unsigned hist_stride,
    int* __restrict__ members, unsigned mem_stride, unsigned n_tok, unsigned tile_tokens, unsigned start) {
    const unsigned tile = blockIdx.x;
    const unsigned h = blockIdx.y;
    const unsigned lane = threadIdx.x;
    const unsigned tb = tile * tile_tokens;
    unsigned te = tb + tile_tokens;
    if (te > n_tok) te = n_tok;
    int* off = hist + ((u64)h * RK_TILES + tile) * hist_stride;
    const int* a = assign + (u64)h * asg_stride;
    int* m = members + (u64)h * mem_stride;
    for (unsigned b0 = tb; b0 < te; b0 += 32) {
        const unsigned t = b0 + lane;
        const int c = (t < te) ? a[t] : -1;
        const unsigned peers = __match_any_sync(RK_FULL, c);
        const int leader = __ffs(peers) - 1;
        const unsigned rank = __popc(peers & ((1u << lane) - 1u));
        int b = 0;
        if (c >= 0 && (int)lane == leader) {
            b = off[c];
            off[c] = b + (int)__popc(peers);
        }
        b = __shfl_sync(RK_FULL, b, leader);
        if (c >= 0) m[b + (int)rank] = (int)(start + t);
        __syncwarp();
    }
}

// ---------------------------------------------------------------------------------------------
// Centroid update (and, on the final pass, the value sums)
// ---------------------------------------------------------------------------------------------

// Grid (ceil(nc / 8), kv_heads), block 256 (one warp per centroid; lane l owns dims 8l .. 8l+7).
// Members are summed in CSR order (ascending positions) in FP32. Training pass: the mean is
// L2-normalized (the sum of squares is a fixed xor-butterfly over the lanes; an all-zero mean is
// stored as is) and stored as BF16; an empty cluster keeps its centroid. Final pass: centroid =
// bf16(bf16(member mean) + key mean), an empty cluster bf16(old + key mean); value_sum =
// bf16(sum of member values), 0 for an empty cluster. cent, vsum, cstart and csize point at the
// build's first cluster; the CSR starts are absolute slots in `members`.
extern "C" __global__ void __launch_bounds__(256) retro_km_update(
    const unsigned char* __restrict__ k_pool, const unsigned char* __restrict__ v_pool,
    const int* __restrict__ bt, unsigned block_size, u64 block_stride, unsigned kv_heads,
    unsigned fp8, float k_scale, float v_scale, const float* __restrict__ mean,
    __nv_bfloat16* __restrict__ cent, __nv_bfloat16* __restrict__ vsum, unsigned cent_stride,
    const int* __restrict__ cstart, unsigned start_stride, const int* __restrict__ csize,
    unsigned size_stride, const int* __restrict__ members, unsigned mem_stride, unsigned nc,
    unsigned final_pass) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned c = blockIdx.x * RK_WARPS_PER_CTA + (threadIdx.x >> 5);
    const unsigned h = blockIdx.y;
    if (c >= nc) return;  // warp-uniform
    const unsigned esize = fp8 ? 1u : 2u;
    const int first = cstart[(u64)h * start_stride + c];
    const int cnt = csize[(u64)h * size_stride + c];
    const int* mem = members + (u64)h * mem_stride;
    float m[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) m[j] = mean[h * RK_D + lane * 8 + j];
    float acc[8], vacc[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) { acc[j] = 0.0f; vacc[j] = 0.0f; }
    for (int k = 0; k < cnt; k++) {
        const unsigned pos = (unsigned)mem[first + k];
        float x[8];
        rk_load8(rk_row(k_pool, bt, pos, block_size, block_stride, kv_heads, h, esize), lane * 8, fp8, k_scale, x);
        #pragma unroll
        for (int j = 0; j < 8; j++) acc[j] = __fadd_rn(acc[j], rk_round_bf16(__fsub_rn(x[j], m[j])));
        if (final_pass) {
            rk_load8(rk_row(v_pool, bt, pos, block_size, block_stride, kv_heads, h, esize), lane * 8, fp8, v_scale, x);
            #pragma unroll
            for (int j = 0; j < 8; j++) vacc[j] = __fadd_rn(vacc[j], x[j]);
        }
    }
    const u64 ci = (u64)h * cent_stride + c;
    __nv_bfloat16* out = cent + ci * RK_D + lane * 8;
    if (cnt > 0) {
        float x[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) x[j] = __fdiv_rn(acc[j], (float)cnt);
        if (!final_pass) {
            float ss = 0.0f;
            #pragma unroll
            for (int j = 0; j < 8; j++) ss = __fadd_rn(ss, __fmul_rn(x[j], x[j]));
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(RK_FULL, ss, off));
            if (ss > 0.0f) {
                const float nrm = __fsqrt_rn(ss);
                #pragma unroll
                for (int j = 0; j < 8; j++) x[j] = __fdiv_rn(x[j], nrm);
            }
            #pragma unroll
            for (int j = 0; j < 8; j++) out[j] = __float2bfloat16_rn(x[j]);
        } else {
            #pragma unroll
            for (int j = 0; j < 8; j++) out[j] = __float2bfloat16_rn(__fadd_rn(rk_round_bf16(x[j]), m[j]));
        }
    } else if (final_pass) {
        #pragma unroll
        for (int j = 0; j < 8; j++) out[j] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(out[j]), m[j]));
    }
    if (final_pass) {
        __nv_bfloat16* vo = vsum + ci * RK_D + lane * 8;
        #pragma unroll
        for (int j = 0; j < 8; j++) vo[j] = __float2bfloat16_rn(cnt > 0 ? vacc[j] : 0.0f);
    }
}
