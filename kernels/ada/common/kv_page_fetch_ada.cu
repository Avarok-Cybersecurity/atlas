// SPDX-License-Identifier: AGPL-3.0-only
// Design after the TierKV pager of arczhi/llama-tierkv (kv-paging-design.md; MIT, Copyright (c) 2023-2026 The ggml authors): attention over the pages on the VRAM desk only, every key keeping the RoPE position it was written with, as that design describes it; no code copied or fetched. Atlas compacts its paged block table instead of staging pages into ordinary KV cells.
//
// kv_window_view_compact: the window desk's view of a paged block table
// (row `kv_window_desk`; host side `crates/spark-model/src/port/kv_window/selector.rs`).
//
// A block the desk moved to host RAM keeps its logical index: the sequence's block table holds
// the desk's HOLE block id there. Each decode or verify step, inside the captured graph, this
// kernel rewrites every row of the real table into the selector-owned view buffers:
//
//   out_table[row] = in_table[row][0 .. ceil(len / bs)] without the hole entries, in order;
//   out_lens[row]  = len - bs * (hole entries removed).
//
// Every removed entry is a FULL block (the desk never hides the open last block), so the new
// length counts exactly the positions the view keeps, and verify rows of one sequence (same
// table, lengths one apart) keep their offsets. The paged-decode attention then reads the view
// unchanged: the keys carry their original RoPE, so this is exact attention over the kept pages.
//
// The hole id is read from device memory (`*hole_ptr`), written by the host before any table
// holds it, so a captured graph needs no re-capture; a negative hole keeps every entry (the
// identity). Entries of out_table past the new length are left as they were; the attention
// reads only `out_lens[row]` positions. A row with len <= 0 (padding) gets out_lens = len.
//
// Grid: (n_rows, 1, 1)  Block: (32, 1, 1)  (one warp per row; ballot compaction, in order)

extern "C" __global__ void kv_window_view_compact(
    const int* __restrict__ in_table,    // [n_rows, max_blocks] physical block ids
    const int* __restrict__ in_lens,     // [n_rows] attention extent per row
    int* __restrict__ out_table,         // [n_rows, max_blocks] view (same stride)
    int* __restrict__ out_lens,          // [n_rows]
    const int* __restrict__ hole_ptr,    // [1] the hole block id, or -1
    unsigned int n_rows,
    unsigned int max_blocks,
    unsigned int block_size
) {
    const unsigned int row = blockIdx.x;
    if (row >= n_rows) {
        return;
    }
    const unsigned int lane = threadIdx.x & 31u;
    const int hole = *hole_ptr;
    const int len = in_lens[row];
    const unsigned int bs = block_size ? block_size : 1u;
    unsigned int nblk = len > 0 ? ((unsigned int)len + bs - 1u) / bs : 0u;
    if (nblk > max_blocks) {
        nblk = max_blocks;
    }
    const int* src = in_table + (size_t)row * max_blocks;
    int* dst = out_table + (size_t)row * max_blocks;
    const unsigned int below = (1u << lane) - 1u;
    unsigned int kept = 0u;
    for (unsigned int base = 0u; base < nblk; base += 32u) {
        const unsigned int i = base + lane;
        const bool live = i < nblk;
        const int v = live ? src[i] : 0;
        const bool keep = live && (hole < 0 || v != hole);
        const unsigned int mask = __ballot_sync(0xffffffffu, keep);
        if (keep) {
            dst[kept + __popc(mask & below)] = v;
        }
        kept += (unsigned int)__popc(mask);
    }
    if (lane == 0u) {
        if (len <= 0) {
            out_lens[row] = len;
        } else {
            const unsigned int removed = nblk - kept;
            const long long l = (long long)len - (long long)removed * (long long)bs;
            out_lens[row] = (int)(l > 0 ? l : 0);
        }
    }
}
