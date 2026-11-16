// SPDX-License-Identifier: AGPL-3.0-only
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license).
//
// kv_view_build_tables: the bounded KV view's per-step block tables (port lane
// L32, row `kv_view_selection`, off on every target).
//
// A view keeps, in chronological order, the selected pages that lie before its
// tail page, then every page from the tail page on. For every attention row of
// a decode or verify step this kernel writes the row's view table and length
// into fixed output buffers that the paged-decode attention then reads in
// place of the real ones. It runs inside the captured decode graph, so it
// finds each row's view by content, never by a host-side row index: a
// descriptor applies to a row whose physical blocks at logical pages 0,
// tail_page and last_page equal the descriptor's anchors, and among matching
// descriptors the one with the largest last_page wins (the lowest index on a
// tie). A row with no matching descriptor gets its real pages and length
// unchanged (the identity), and a padding row (length 0) gets length 0.
//
// Host twin and oracle: `spark-model/src/port/kv_view/view.rs::build_row`.
//
// Descriptor words (int32, 8 per entry): active, a0, a_tail, a_last,
// last_page, tail_page, n_sel, 0. Entry e's selected logical pages are
// sel_pool[e * sel_cap .. e * sel_cap + n_sel], ascending, all below tail_page.
// The host writes an entry only between steps, on the stream this kernel runs
// on, so a launch never sees a half-written entry.
//
// Bounds: a row is read up to min(ceil(len / block_size), row_cap) entries
// (row_cap is the stride for a batched step, the arena size for one row); the
// view of a row never has more pages than the row (n_sel <= tail_page).
//
// Grid: (n_rows, 1, 1)  Block: (128, 1, 1)

#define KV_VIEW_DESC_WORDS 8

extern "C" __global__ void kv_view_build_tables(
    const int* __restrict__ real_table,   // [n_rows * stride] physical block ids
    const int* __restrict__ real_lens,    // [n_rows] tokens per row, this step's included
    unsigned int n_rows,
    unsigned int stride,                  // entries from one row to the next (in and out)
    unsigned int row_cap,                 // most entries a row may hold
    unsigned int block_size,              // tokens per page
    const int* __restrict__ desc,         // [n_desc * KV_VIEW_DESC_WORDS]
    unsigned int n_desc,
    const int* __restrict__ sel_pool,     // [n_desc * sel_cap] logical pages
    unsigned int sel_cap,
    int* __restrict__ out_table,          // [n_rows * stride]
    int* __restrict__ out_lens            // [n_rows]
) {
    const unsigned int r = blockIdx.x;
    if (r >= n_rows) {
        return;
    }
    __shared__ int s_entry;
    const int* row = real_table + (size_t)r * stride;
    int* out = out_table + (size_t)r * stride;
    const int len = real_lens[r];
    const unsigned int bs = block_size > 0 ? block_size : 1u;
    unsigned int pages = len > 0 ? ((unsigned int)len + bs - 1u) / bs : 0u;
    if (pages > row_cap) {
        pages = row_cap;
    }

    if (threadIdx.x == 0) {
        int best = -1;
        int best_last = -1;
        for (unsigned int e = 0; e < n_desc; ++e) {
            const int* d = desc + (size_t)e * KV_VIEW_DESC_WORDS;
            if (d[0] != 1) {
                continue;
            }
            const int last = d[4];
            const int tail = d[5];
            const int n_sel = d[6];
            if (last < 0 || tail < 0 || tail > last || (unsigned int)last >= pages || n_sel < 0
                || n_sel > tail || (unsigned int)n_sel > sel_cap) {
                continue;
            }
            if (row[0] != d[1] || row[tail] != d[2] || row[last] != d[3]) {
                continue;
            }
            if (last > best_last) {
                best = (int)e;
                best_last = last;
            }
        }
        s_entry = best;
    }
    __syncthreads();

    const int e = s_entry;
    if (e < 0) {
        for (unsigned int j = threadIdx.x; j < pages; j += blockDim.x) {
            out[j] = row[j];
        }
        if (threadIdx.x == 0) {
            out_lens[r] = len > 0 ? len : 0;
        }
        return;
    }

    const int* d = desc + (size_t)e * KV_VIEW_DESC_WORDS;
    const unsigned int tail = (unsigned int)d[5];
    const unsigned int n_sel = (unsigned int)d[6];
    const int* sel = sel_pool + (size_t)e * sel_cap;
    const unsigned int total = n_sel + (pages - tail);
    for (unsigned int j = threadIdx.x; j < total; j += blockDim.x) {
        unsigned int src;
        if (j < n_sel) {
            const int p = sel[j];
            // A selected page is below the tail by construction; clamp so a
            // corrupt entry can never read outside the row.
            src = (p >= 0 && (unsigned int)p < tail) ? (unsigned int)p : 0u;
        } else {
            src = tail + (j - n_sel);
        }
        out[j] = row[src];
    }
    if (threadIdx.x == 0) {
        out_lens[r] = len - (int)((tail - n_sel) * bs);
    }
}
