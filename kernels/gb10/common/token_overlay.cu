// SPDX-License-Identifier: AGPL-3.0-only

// token_overlay: PEFT trainable-tokens / modules_to_save embed + lm_head row
// overlay for the main decoder (Feature 2), ported from the NLLB token overlay.
//
// All kernels are per-adapter-slot routed: a device pointer TABLE indexed by the
// adapter slot `s` yields that adapter's own slot_map / compact rows / ids. The
// only per-step argument is `seq_slot` (a device i32[n]; NULL ⇒ uniform `active`
// slot), so the tables are load-time-fixed addresses and the launch is
// CUDA-graph safe. `s < 0` ⇒ base request, row left untouched.
//
// Indexing uses (unsigned long long) casts on vocab×hidden products to avoid the
// 32-bit overflow at large vocab (~256k) × hidden (2048).

#include <cuda_bf16.h>

// ---- 1. Row-diff: which adapter base_layer rows differ from the served table.
// One thread per row; flags[r] = (max_i |base[r,i] - served[r,i]| > thresh).
// Grid: (ceil(rows/256),1,1)  Block: (256,1,1)
extern "C" __global__ void embed_rowdiff_bf16(
    const __nv_bfloat16* __restrict__ base,     // [rows, h] adapter base_layer
    const __nv_bfloat16* __restrict__ served,   // [rows, h] served embed table
    unsigned char* __restrict__ flags,          // [rows] out
    unsigned int rows,
    unsigned int h,
    float thresh
) {
    unsigned int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const __nv_bfloat16* a = base + (unsigned long long)r * h;
    const __nv_bfloat16* b = served + (unsigned long long)r * h;
    float maxd = 0.0f;
    for (unsigned int i = 0; i < h; ++i) {
        float d = fabsf(__bfloat162float(a[i]) - __bfloat162float(b[i]));
        if (d > maxd) maxd = d;
    }
    flags[r] = (maxd > thresh) ? 1u : 0u;
}

// ---- 2. Embed overlay: full row-replace of overridden vocab rows after gather.
// Per row r: pick slot s (seq_slot[r] or active); slot_map[s][token] gives the
// compact row index in rows[s]; if >= 0, copy that row over out[r].
// Grid: (num_tokens,1,1)  Block: (256,1,1)
extern "C" __global__ void embed_overlay_routed_bf16(
    const unsigned int* __restrict__ ids,        // [n] token id per row
    const int* __restrict__ seq_slot,            // [n] adapter slot, or NULL
    int active,                                  // fallback slot when seq_slot NULL
    const unsigned long long* __restrict__ slot_map_tab, // [L] -> const int* [vocab]
    const unsigned long long* __restrict__ rows_tab,     // [L] -> const bf16* [n_ov,h]
    const unsigned int* __restrict__ n_tab,      // [L] EMBED n_override per slot
    __nv_bfloat16* __restrict__ out,             // [n, h] residual stream (in place)
    unsigned int h,
    unsigned int vocab                           // slot_map length (served vocab)
) {
    const unsigned int r = blockIdx.x;
    int s = (seq_slot != nullptr) ? seq_slot[r] : active;
    if (s < 0) return;
    const int* slot_map = (const int*)slot_map_tab[s];
    if (slot_map == nullptr) return;
    // Bounds guard (CWE-125): `ids` carries the REQUEST's token stream, while
    // `slot_map` is `[vocab]`, sized to the served vocab snapshotted at adapter
    // load (overlay_build.rs). A token id at/beyond that vocab has BY
    // DEFINITION no override row in this sparse row-replacement overlay, so
    // skipping is the documented semantics (identical to `slot_map[id] == -1`:
    // the base embedding row already gathered stands) — NOT a silent default.
    // Validity of the id itself is the base embed gather's contract, not ours;
    // a hard error here would let one malformed request trap the whole serve.
    if (ids[r] >= vocab) return;
    const int slot = slot_map[ids[r]];
    // `slot` indexes rows_tab[s] = [n_tab[s], h]. The in-tree builder only
    // writes compact indices < n_override (host-unit-tested invariant), so
    // `slot >= n_tab[s]` can only mean a corrupt/foreign table: skip (fold
    // nothing) rather than read out of bounds (CWE-125 hardening).
    if (slot < 0 || (unsigned int)slot >= n_tab[s]) return;
    const __nv_bfloat16* src =
        (const __nv_bfloat16*)rows_tab[s] + (unsigned long long)slot * h;
    __nv_bfloat16* dst = out + (unsigned long long)r * h;
    for (unsigned int i = threadIdx.x; i < h; i += blockDim.x) {
        dst[i] = src[i];
    }
}

// Warp-reduced dot(hidden[row], override_row) over h; lane 0 returns the sum.
__device__ __forceinline__ float warp_dot_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    unsigned int h
) {
    float acc = 0.0f;
    for (unsigned int i = threadIdx.x; i < h; i += warpSize) {
        acc += __bfloat162float(x[i]) * __bfloat162float(w[i]);
    }
    for (int off = warpSize / 2; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    return acc;
}

// ---- 3. lm_head overlay: recompute the logit column for each overridden id.
// One warp per (row, j). j indexes the overridden-id slot of that row's adapter.
// Grid: (num_tokens, max_n_override, 1)  Block: (32,1,1)
extern "C" __global__ void lmhead_overlay_routed_bf16(
    const __nv_bfloat16* __restrict__ hidden,    // [m, h]
    const int* __restrict__ seq_slot,            // [m] or NULL
    int active,
    const unsigned long long* __restrict__ rows_tab, // [L] -> const bf16* [n_ov,h]
    const unsigned long long* __restrict__ ids_tab,  // [L] -> const u32* [n_ov]
    const unsigned int* __restrict__ n_tab,          // [L] n_override per slot
    __nv_bfloat16* __restrict__ logits,          // [m, vocab] (in place)
    unsigned int h,
    unsigned int vocab
) {
    const unsigned int row = blockIdx.x;
    const unsigned int j = blockIdx.y;
    int s = (seq_slot != nullptr) ? seq_slot[row] : active;
    if (s < 0) return;
    if (j >= n_tab[s]) return;
    const unsigned int id = ((const unsigned int*)ids_tab[s])[j];
    // Bounds guard (CWE-787): `id` selects a column of the `[m, vocab]` logits.
    // Loader-built ids are clamped `< vocab` (clamp_trainable_to_vocab); a
    // corrupt/foreign ids table must skip, not become an out-of-bounds WRITE.
    if (id >= vocab) return;
    const __nv_bfloat16* w =
        (const __nv_bfloat16*)rows_tab[s] + (unsigned long long)j * h;
    const __nv_bfloat16* x = hidden + (unsigned long long)row * h;
    float dot = warp_dot_bf16(x, w, h);
    if (threadIdx.x == 0) {
        logits[(unsigned long long)row * vocab + id] = __float2bfloat16(dot);
    }
}

// FP32-logits variant (single-token decode with use_fp32_logits).
extern "C" __global__ void lmhead_overlay_routed_f32(
    const __nv_bfloat16* __restrict__ hidden,
    const int* __restrict__ seq_slot,
    int active,
    const unsigned long long* __restrict__ rows_tab,
    const unsigned long long* __restrict__ ids_tab,
    const unsigned int* __restrict__ n_tab,
    float* __restrict__ logits,                  // [m, vocab] FP32
    unsigned int h,
    unsigned int vocab
) {
    const unsigned int row = blockIdx.x;
    const unsigned int j = blockIdx.y;
    int s = (seq_slot != nullptr) ? seq_slot[row] : active;
    if (s < 0) return;
    if (j >= n_tab[s]) return;
    const unsigned int id = ((const unsigned int*)ids_tab[s])[j];
    // Bounds guard (CWE-787): same as the bf16 variant — skip a corrupt id
    // rather than write out of bounds.
    if (id >= vocab) return;
    const __nv_bfloat16* w =
        (const __nv_bfloat16*)rows_tab[s] + (unsigned long long)j * h;
    const __nv_bfloat16* x = hidden + (unsigned long long)row * h;
    float dot = warp_dot_bf16(x, w, h);
    if (threadIdx.x == 0) {
        logits[(unsigned long long)row * vocab + id] = dot;
    }
}
