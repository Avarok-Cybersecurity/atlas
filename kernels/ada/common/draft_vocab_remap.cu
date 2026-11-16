// SPDX-License-Identifier: AGPL-3.0-only
// Portions derived from Cinference (https://github.com/satellitedown/cinference, commit 383e5db), itself derived from NInfer by Neroued and contributors (https://github.com/Neroued/ninfer), Apache License 2.0. Source: src/ops/kernel/speculative_round.cuh:956-963 (proposal_remap_token_ids_kernel). Changes: u32 head-row ids from Atlas's argmax kernels instead of int32 proposal tokens, an optional copy of the remapped ids into a chain buffer, a launch per call site rather than per round, by Atlas contributors.
//
// draft_vocab_remap: map MTP draft-head ROW indices to TOKEN ids, in place.
//
// The frequency-ranked draft head (`mtp_vocab_rank = "freq"`) holds the rows
// of the LM head for the most frequent tokens, so the drafter's argmax
// returns a row index r, not a token id. This kernel rewrites
// ids[i] = row_to_id[ids[i]] on the device, right after the argmax and before
// `embed_from_argmax` / `batched_embed` read the id and before any readback,
// so every consumer downstream of the drafter sees token ids exactly as it
// does with the id-ranked head.
//
// An index at or past `rows` is left unchanged (cannot happen with a head of
// `rows` rows; kept as the upstream kernel's bound so a stray value never
// reads past the table).
//
// When `ids_copy` is non-null, the remapped id is also written to
// ids_copy[i]: the device-resident propose chain (`mtp_propose_graph`)
// collects every position's ids there and reads them back once per round.
//
// Each thread reads and writes only its own element, so the in-place update
// has no cross-thread hazard.
//
// Grid: (ceil(n / 128), 1, 1)  Block: (128, 1, 1)

extern "C" __global__ void draft_vocab_remap(
    unsigned int* __restrict__ ids,               // [n] head rows in, token ids out
    const unsigned int* __restrict__ row_to_id,   // [rows] head row -> token id
    unsigned int n,
    unsigned int rows,
    unsigned int* __restrict__ ids_copy           // [n] or null
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const unsigned int r = ids[i];
    const unsigned int t = (r < rows) ? row_to_id[r] : r;
    ids[i] = t;
    if (ids_copy != nullptr) {
        ids_copy[i] = t;
    }
}
