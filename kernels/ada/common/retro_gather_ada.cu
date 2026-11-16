// SPDX-License-Identifier: AGPL-3.0-only
// Translated for Atlas from microsoft/RetrievalAttention (RetroInfer), MIT License,
// Copyright (c) Microsoft Corporation. Upstream: cache_hub/retroinfer_cache.py:754-815
// (`sparse_attention`: steady zone plus retrieval-zone clusters assembled into one execution
// buffer) and library/retroinfer/retroinfer_kernels/src/gather_copy.cu:100-217
// (`gather_copy_and_concat`, the access pattern only). Changes: rewritten for Atlas's paged KV
// (block table, BF16 or per-tensor-scaled FP8 E4M3 pools, head_dim 256) instead of the
// upstream's head_dim 128, 8-vector cluster pages and pinned-host sources; cluster members
// read from the device index's CSR lists; a fixed-capacity BF16 scratch per kv head with a
// device valid length; no host involvement, so the launch is CUDA-graph capturable.

// Ada (sm_89) RetroInfer exact-zone gather (port lane L34b), two entries:
//
//   retro_gather_exact_bf16   BF16 paged KV source
//   retro_gather_exact_fp8    FP8 E4M3 paged KV source, dequantized to BF16 on the fly
//
// ── CONTRACT (crates/spark-model/src/port/retro/state.rs, `ExactBuf`) ──────
//
// Per kv head h, one logical sequence, len = max(row_lens[0 .. rows)):
//
//   [ sink  0 .. min(sink, indexed_end, len) )
//   [ the members < len of each retrieval cluster, in rank order, each ascending ]
//   [ tail  min(indexed_end, len) .. len )
//
// sink and indexed_end are index header words; the retrieval ids and their count are the
// selection's (`counts[h * 2]`). A retrieval cluster is stored whole or not at all: walking
// the clusters in rank order with `room = cap - (sink + tail)`, a cluster whose member count
// would take the running total past `room` is skipped and its member count added to
// dropped[h]; later, smaller clusters may still fit (exactly the oracle's
// `oracle_eval::exact_positions`). Should sink plus tail alone exceed cap (the host sizes cap
// so they never do), writes stop at cap and the overflow is also counted in dropped[h].
//
//   out_k, out_v   BF16 [kv_heads][cap][256]   entry e of head h at (h * cap + e) * 256
//   out_pos        i32  [kv_heads][cap]        absolute position, -1 for an unreadable token
//   out_valid      i32  [kv_heads]             entries written
//   out_dropped    i32  [kv_heads]
//
// Entries at or after out_valid[h] are never written. A token with no readable block
// (logical block >= max_blocks_per_seq or a negative physical id) is written as zeros with
// position -1, which the attention skips.
//
// ── NUMERICS ───────────────────────────────────────────────────────────────
//
// BF16: a byte copy. FP8: bf16_rn( float(e4m3) * scale ), K by k_scale and V by v_scale,
// exactly `spark_runtime::kv_dequant::dequant_fp8_to_bf16` (E4M3 to half is exact, one f32
// multiply, round to nearest even). Both paths are bit-equal to the CPU reference in
// `crates/spark-model/src/port/retro/gather_reference.rs`.
//
// ── LAUNCH ─────────────────────────────────────────────────────────────────
//
//   grid (kv_heads, grid_y)   block 256   dynamic shared (3 * retrieval_cap + 1) * 4 bytes
//
// Every CTA rebuilds its head's cluster table in shared memory: member counts and the members
// below len (a binary search in each ascending list) in parallel, then one thread walks the
// clusters in rank order (the store-or-skip decision is sequential by definition) and writes
// the exclusive prefix of stored members. Its eight warps then copy entries
// e = blockIdx.y * 8 + warp, stepping by gridDim.y * 8, one 256-element row of K and V per
// warp. Counts come from the device; the grid never depends on them. The host keeps
// retrieval_cap at or under 4,000 so shared memory stays under the 48 KiB default.
//
// Pool layout (both dtypes): token t of head h in physical block b starts at byte
//   b * block_stride + ((t % block_size) * kv_heads + h) * 256 * elem_bytes
// the layout `paged_decode_attn_bf16_gqa.cu` and `paged_decode_attn_fp8_gqa.cu` read
// (`block_stride` is the layer's `PagedKvCache::block_stride_bytes_for_layer`).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define RG_HDIM 256
#define RG_WARPS 8
#define RG_THREADS (RG_WARPS * 32)
// Index header words (state.rs `header`).
#define RG_HDR_INDEXED_END 3
#define RG_HDR_SINK 4

__device__ __forceinline__ unsigned int rg_fp8_to_bf16(unsigned int byte, float scale) {
    const __half_raw hr = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)byte, __NV_E4M3);
    const float f = __half2float(__half(hr));
    return (unsigned int)__bfloat16_as_ushort(__float2bfloat16(__fmul_rn(f, scale)));
}

// Eight FP8 bytes (little endian in a uint2) to eight BF16 values packed in a uint4.
__device__ __forceinline__ uint4 rg_fp8x8_to_bf16x8(uint2 raw, float scale) {
    unsigned int p[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const unsigned int word = (i < 2) ? raw.x : raw.y;
        const unsigned int shift = (i % 2) * 16;
        const unsigned int lo = rg_fp8_to_bf16((word >> shift) & 0xFF, scale);
        const unsigned int hi = rg_fp8_to_bf16((word >> (shift + 8)) & 0xFF, scale);
        p[i] = lo | (hi << 16);
    }
    return make_uint4(p[0], p[1], p[2], p[3]);
}

template <bool FP8>
__device__ __forceinline__ void rg_body(
    const unsigned char* __restrict__ k_pool,
    const unsigned char* __restrict__ v_pool,
    const int* __restrict__ block_table,   // [max_blocks_per_seq], the sequence's table
    const int* __restrict__ row_lens,      // [rows]
    unsigned int rows,
    unsigned int max_blocks_per_seq,
    unsigned int block_size,
    unsigned int kv_heads,
    unsigned long long block_stride,       // bytes from one block of the layer to the next
    float k_scale,
    float v_scale,
    const int* __restrict__ r_ids,         // [kv_heads][r_cap], rank order
    const int* __restrict__ counts,        // [kv_heads][2]: retrieval, estimation
    unsigned int r_cap,
    const int* __restrict__ header,        // the index header words
    const int* __restrict__ list_offsets,  // [kv_heads][cap_centroids + 1] CSR
    const int* __restrict__ list_tokens,   // [kv_heads][cap_tokens]
    unsigned int cap_centroids,
    unsigned int cap_tokens,
    __nv_bfloat16* __restrict__ out_k,
    __nv_bfloat16* __restrict__ out_v,
    int* __restrict__ out_pos,
    int* __restrict__ out_valid,
    int* __restrict__ out_dropped,
    unsigned int cap) {
    extern __shared__ int rg_smem[];
    int* s_out = rg_smem;                  // [r_cap + 1] exclusive prefix of stored members
    int* s_src = rg_smem + r_cap + 1;      // [r_cap] list start of each cluster
    int* s_full = s_src + r_cap;           // [r_cap] member count of each cluster
    __shared__ int s_meta[5];              // sink_end, tail_start, tail_len, n_ret, n_out

    const unsigned int h = blockIdx.x;
    if (h >= kv_heads) return;             // uniform per CTA
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid / 32;
    const unsigned int lane = tid % 32;

    if (tid == 0) {
        int len = 0;
        for (unsigned int r = 0; r < rows; r++) len = max(len, row_lens[r]);
        const int ie = max(header[RG_HDR_INDEXED_END], 0);
        const int sink = max(header[RG_HDR_SINK], 0);
        const int tail_start = min(ie, len);
        s_meta[0] = min(min(sink, ie), len);
        s_meta[1] = tail_start;
        s_meta[2] = len - tail_start;
        s_meta[3] = max(0, min(counts[h * 2], (int)r_cap));
        s_meta[4] = len;                   // temporarily: len
    }
    __syncthreads();
    const int sink_end = s_meta[0];
    const int tail_start = s_meta[1];
    const int tail_len = s_meta[2];
    const int nrc = s_meta[3];
    const int len = s_meta[4];
    const int* my_ids = r_ids + (unsigned long long)h * r_cap;
    const int* my_off = list_offsets + (unsigned long long)h * (cap_centroids + 1);
    const int* my_tok = list_tokens + (unsigned long long)h * cap_tokens;

    // Members and members below len of each retrieval cluster (lists are ascending).
    for (int i = (int)tid; i < nrc; i += RG_THREADS) {
        const int id = my_ids[i];
        int b = 0;
        int full = 0;
        int kept = 0;
        if (id >= 0 && (unsigned int)id < cap_centroids) {
            b = min(max(my_off[id], 0), (int)cap_tokens);
            const int e = min(max(my_off[id + 1], b), (int)cap_tokens);
            full = e - b;
            int lo = b;
            int hi = e;
            while (lo < hi) {
                const int mid = (lo + hi) >> 1;
                if (my_tok[mid] < len) lo = mid + 1; else hi = mid;
            }
            kept = lo - b;
        }
        s_src[i] = b;
        s_full[i] = full;
        s_out[i] = kept;
    }
    __syncthreads();

    // Store-or-skip in rank order (sequential by definition; nrc is a few hundred).
    if (tid == 0) {
        const long long fixed = (long long)sink_end + (long long)tail_len;
        const long long room = max((long long)cap - fixed, 0ll);
        long long used = 0;
        int acc = 0;
        long long dropped = 0;
        for (int i = 0; i < nrc; i++) {
            const int full = s_full[i];
            const int kept = s_out[i];
            s_out[i] = acc;
            if (used + full > room) {
                dropped += full;
                s_full[i] = 0;             // skipped: contributes no entry
            } else {
                used += full;
                acc += kept;
            }
        }
        s_out[nrc] = acc;
        const long long total = fixed + acc;
        const int n_out = (int)min(total, (long long)cap);
        dropped += total - n_out;
        s_meta[4] = n_out;
        if (blockIdx.y == 0) {
            out_valid[h] = n_out;
            out_dropped[h] = (int)min(dropped, (long long)0x7fffffff);
        }
    }
    __syncthreads();
    const int clustered = s_out[nrc];
    const int n_out = s_meta[4];

    const unsigned long long elem = FP8 ? 1ull : 2ull;
    const unsigned long long head_row_bytes = (unsigned long long)RG_HDIM * elem;
    for (int e = (int)(blockIdx.y * RG_WARPS + warp); e < n_out; e += (int)(gridDim.y * RG_WARPS)) {
        int pos;
        if (e < sink_end) {
            pos = e;
        } else if (e < sink_end + clustered) {
            // The cluster holding entry j: the largest i with s_out[i] <= j. Skipped and
            // empty clusters share their successor's offset, so the search lands on the one
            // that holds j.
            const int j = e - sink_end;
            int lo = 0;
            int hi = nrc - 1;
            while (lo < hi) {
                const int mid = (lo + hi + 1) >> 1;
                if (s_out[mid] <= j) lo = mid; else hi = mid - 1;
            }
            pos = my_tok[s_src[lo] + (j - s_out[lo])];
        } else {
            pos = tail_start + (e - sink_end - clustered);
        }

        // Warp-uniform: every lane computed the same position.
        bool ok = pos >= 0;
        const unsigned int logical = ok ? (unsigned int)pos / block_size : 0u;
        ok = ok && logical < max_blocks_per_seq;
        const int phys = ok ? block_table[logical] : -1;
        ok = ok && phys >= 0;

        uint4 kk = make_uint4(0u, 0u, 0u, 0u);
        uint4 vv = make_uint4(0u, 0u, 0u, 0u);
        if (ok) {
            const unsigned long long src = (unsigned long long)phys * block_stride
                + ((unsigned long long)((unsigned int)pos % block_size) * kv_heads + h)
                    * head_row_bytes;
            if (FP8) {
                const uint2 kr = *reinterpret_cast<const uint2*>(k_pool + src + lane * 8u);
                const uint2 vr = *reinterpret_cast<const uint2*>(v_pool + src + lane * 8u);
                kk = rg_fp8x8_to_bf16x8(kr, k_scale);
                vv = rg_fp8x8_to_bf16x8(vr, v_scale);
            } else {
                kk = *reinterpret_cast<const uint4*>(k_pool + src + lane * 16u);
                vv = *reinterpret_cast<const uint4*>(v_pool + src + lane * 16u);
            }
        }
        const unsigned long long dst = ((unsigned long long)h * cap + (unsigned int)e) * RG_HDIM
            + (unsigned long long)lane * 8u;
        *reinterpret_cast<uint4*>(out_k + dst) = kk;
        *reinterpret_cast<uint4*>(out_v + dst) = vv;
        if (lane == 0) out_pos[(unsigned long long)h * cap + (unsigned int)e] = ok ? pos : -1;
    }
}

#define RG_PARAMS                                                                              \
    const unsigned char* __restrict__ k_pool, const unsigned char* __restrict__ v_pool,        \
    const int* __restrict__ block_table, const int* __restrict__ row_lens, unsigned int rows,  \
    unsigned int max_blocks_per_seq, unsigned int block_size, unsigned int kv_heads,           \
    unsigned long long block_stride, float k_scale, float v_scale,                             \
    const int* __restrict__ r_ids, const int* __restrict__ counts, unsigned int r_cap,         \
    const int* __restrict__ header, const int* __restrict__ list_offsets,                      \
    const int* __restrict__ list_tokens, unsigned int cap_centroids, unsigned int cap_tokens,  \
    __nv_bfloat16* __restrict__ out_k, __nv_bfloat16* __restrict__ out_v,                      \
    int* __restrict__ out_pos, int* __restrict__ out_valid, int* __restrict__ out_dropped,     \
    unsigned int cap

#define RG_ARGS                                                                                \
    k_pool, v_pool, block_table, row_lens, rows, max_blocks_per_seq, block_size, kv_heads,     \
    block_stride, k_scale, v_scale, r_ids, counts, r_cap, header, list_offsets, list_tokens,   \
    cap_centroids, cap_tokens, out_k, out_v, out_pos, out_valid, out_dropped, cap

extern "C" __global__ void __launch_bounds__(RG_THREADS) retro_gather_exact_bf16(RG_PARAMS) {
    rg_body<false>(RG_ARGS);
}

extern "C" __global__ void __launch_bounds__(RG_THREADS) retro_gather_exact_fp8(RG_PARAMS) {
    rg_body<true>(RG_ARGS);
}
