# Qwen3.8-Flash-Next: the K-row MoE decode arm, 4 to 8 rows then 16

Ronald R. Stesiak, Nonlinear Dynamics, 2026-09-12

<!-- provenance-id: 526f6e616c6420522e205374657369616b -->

The second rung of #974, on top of #1026 (lookup drafts into the wide verify) and #972 (the K-row verify body).

#1026 puts lookup drafts into the wide verify on Flash-Next. It runs at K=3 rows because that is where the small-M MoE FFN arms stop. This PR lifts that limit so a lookup step can commit 8 tokens, then 16. It is the rung that turns #1026 from an acceptance gain into an agentic-session gain, and it revives MTP-3 and MTP-4 as measurable options instead of a dead end.

## The cliff, exactly

`hc_ffn_dispatch` routes the MoE FFN by row count: 1 row, 2 rows, 3 rows, then `Prefill` for anything wider. `MoeLayer::forward_prefill` takes the grouped GEMM only above 64 tokens; below that it is `forward_batched`, the per-token expert loop with data-dependent routing. So a K=4 verify step runs four single-token MoE passes per layer, with the per-token launches and the routing round trips that come with them. Measured during the #972 series: MTP-3 at K=4 rows, LRU 14.9 tok/s, against 62.0 at K=3.

The K=3 arm itself is at the DRAM floor (09-09 census): `moe_expert_gate_up_shared_batch3_t` streams 57.5 MB in 188 us, 306 GB/s; `silu_down` at 239 GB/s; 15.0 ms of the step is MoE weight traffic. That number reproduces from the geometry: 512 experts, top-10 plus one shared, intermediate 640, hidden 2560, NVFP4 with FP8 block scales. Gate+up per expert is 1.84 MB, down is 0.92 MB. Three rows times ten experts plus the shared expert is 57 MB per gate-up launch. The batch3 kernel gives every (row, expert slot) its own block row (`blockIdx.y` over `3 * top_k`, shared expert blocks after), so it reads an expert once per row that routes to it. Weight traffic is linear in rows: about 27.6 MB per row per layer, 1.33 GB per row per step, 4.3 to 5.5 ms per row at the measured rates.

## Every K-row piece on this model, and where each one stops

| piece | today | cap | for K=8 | for K=16 |
|---|---|---|---|---|
| MoE FFN (routed + shared) | Single / K2 / K3 fused `_t` kernels; wider = per-token loop | **3** | new `batchN` arm | same arm, N to 16 |
| GDN (linear attention, 36 layers) | batched conv+GDN dispatch rows 2, 3, 4 (`wy2`/`wy3`/`wy4`); wider returns `Ok(false)` | **4** | wire `wy5`..`wy8`, already built (`gated_delta_rule_wyn`, loaded by `wyn_kernels`) | wire to `wy16` |
| attention projections (12 full-attention layers) | `dense_gemv_batchm`, weight read once for n rows | **8** (`DENSE_GEMV_BATCHM_MAX_M`) | covered | raise to 16 |
| hyper-connection sites | decode-shaped collapse `hc_dec_up` / `hc_dec_down`, T rows per pass | **8** (`QHC_DEC_MAX_T`) | covered | raise to 16; past 8 today the split arm reads the 13 MB per token per site again |
| attention core (paged decode) | per row, 38 launches per row across the 12 layers | none | covered, launch count grows | batch over K rows with a replicated block table (09-09 item 8) |
| QSA indexer | `qsa_score` grid (rows, heads) | none | covered | covered |
| PLE n-gram gather | one gather per row, NVMe miss gap on most steps | none | covered; verify-row prefetch (e8096d26) helps | same |
| LM head, verify rows | GEMM at M=K | none | covered | covered |
| router top-k | `moe_topk_*_batched` at 3 rows | 3 | N rows | N rows |
| host side | eager verify, D2H argmax per verify; #973 removes the per-row D2H on the greedy path | | covered | graph capture stays the structural item |

Two rungs fall out of that table. **K=8** needs the MoE arm and the GDN wiring, nothing else. **K=16** needs the same arm at N=16 plus three ceilings raised (projection GEMV, HC collapse, attention core batching). The PR lands K=8 first and certifies it; K=16 is the second commit series on the same branch.

## The arm

**A. `batchN`, slot-major.** The straight generalization of the batch3 `_t` family: `moe_expert_gate_up_shared_batchN_t`, `moe_expert_silu_down_shared_batchN_t`, `moe_weighted_sum_blend_batchN`, with N a launch parameter (grid.y = N * top_k + shared slots), the router GEMV through `dense_gemv_batchm`, top-k through the batched kernels at N rows. Same per-slot structure and the same per-row reduction order as batch3, so each row is bit-identical to the K=3 arm's row, which is what the parity test checks. Traffic linear in N at the floor: roughly 4.8 ms per row. This alone takes K=8 from the per-token loop (about 14.9 tok/s at K=4) to the floor.

**B. expert-major, deduplicated.** The N * top_k slots are sorted by expert on device (at most 160 entries, one block), and one block row runs per distinct expert, looping over the rows routed to it. Traffic becomes the number of distinct experts in the step, not rows times top_k. Under independent routing that is 512 * (1 - (1 - 10/512)^N): 29 at N=3, 76 at N=8, 139 at N=16, against 30 / 80 / 160 for the slot-major arm, so the random model gives 13% at N=16. Real routing overlaps more than random: the 07-19 measurement on Qwen3.6-35B-A3B (top-8 of 256, 17-token windows of generated text) found 59 distinct experts where the random model predicts 107 and the slot count is 136, on code and prose alike. Whether Flash-Next's router behaves the same on a copied block is measurable today with zero code: `ATLAS_MOE_UNION_STATS=1` samples exactly this union on every verify batch. That measurement decides whether B is built. A is built regardless.

**Width is per step, not per serve.** The verify already dispatches on `pending_drafts.len()`. With this arm the lookup gate proposes at `ATLAS_LOOKUP_WIDTH` (default 7 drafts, K=8 rows) while the MTP head stays at its own width (2 drafts, K=3). Fresh generation never pays for the wide arm. That is the difference from the MTP-3 census on 09-09, which was net negative because a third draft pass and 33% more expert traffic bought 13.6% more tokens on every step; here the wide step runs only when the index has the tokens already.

## Cost model, anchored and labeled

Measured: MoE at K=3 is 15.0 ms; per-row MoE traffic 1.33 GB at 239 to 306 GB/s; the record step at K=3 is 62.0 tok/s on LRU. Estimate from the census: the non-MoE part of the step is about 30 ms and mostly weight-read-once GEMVs, so it grows slowly with rows.

Step time model, estimate: T(K) = 30 ms + 4.8 ms * K. Copy-task tok/s at 95% acceptance (lookup on a repeated block), estimate:

| K rows | drafts | step (ms) | tokens committed | tok/s |
|---|---|---|---|---|
| 3 | 2 | 44 | 2.85 | 64 |
| 4 | 3 | 49 | 3.8 | 77 |
| 6 | 5 | 59 | 5.7 | 97 |
| 8 | 7 | 68 | 7.6 | 111 |
| 12 | 11 | 88 | 11.4 | 130 |
| 16 | 15 | 107 | 15.2 | 142 |

With arm B and Flash-Next routing overlapping like the 35B did (about 95 distinct experts of 160 at K=16), the K=16 step is about 76 ms and the copy-task number is about 200 tok/s. All of these are model numbers; the cells below replace them.

The competing number for the copy-task class stays what #974 lists: llama.cpp 97.4 tok/s on one Spark at 94.7% acceptance (NVIDIA forum, Flash-Next thread p.4, 26 Aug). K=8 at the floor sits above it on the model; K=16 sits well above it. Fresh-generation numbers (62.0, #973) are a different table and do not move with this PR.

## Plan

| step | what | GPU |
|---|---|---|
| 0 | `ATLAS_MOE_UNION_STATS=1` on the #1026 serve, copy task and MinHeap, K=3: distinct experts per layer-step vs 30. Decides A-only vs A+B. | yes, no code |
| 1 | `batchN` slot-major kernels (gate_up, silu_down, wsum_blend) for N in 4..=8, microtest against batch3 rows, `hc_ffn_dispatch` gets a `KN` arm, `MoeLayer::forward_kn`. | yes |
| 2 | GDN wide rows: wire `wy5`..`wy8` into the batched conv+GDN dispatch for this model; parity test `hc_rows_t8_rows_equal_t1_rows`. | yes |
| 3 | `ATLAS_LOOKUP_WIDTH` in the gate (#1026), default 7; MTP width unchanged. | no |
| 4 | Cells: copy task at K=3 vs K=8 (own table), MinHeap x3 and Volvo x3 unchanged vs base (byte-identical, the wide arm never fires there), `spark benchmark run agentic-webserver`. MTP-3 and MTP-4 re-measured on the new arm as a side row. | yes |
| 5 | K=16: N to 16 in the arm, `DENSE_GEMV_BATCHM_MAX_M` and `QHC_DEC_MAX_T` to 16, `wy9`..`wy16` wired, paged decode batched over rows. Arm B if step 0 says so. | yes |

Defaults ON at each rung once its cells are in. Kill switches: `ATLAS_FFN_SMALLM=0` already routes wide rows to the base path; `ATLAS_LOOKUP_WIDTH=2` restores #1026's shape.

## Not in this PR

The 27B: its dense FFN already takes K=5..16 through the batched GDN verify (#844, #845), and the lookup gate lands there as its own PR. Hopper: no NVFP4 path, the arm's FP8 twin is Initiative 2 work. EXL3 experts: Richard's, the `NativeBatched` arm stays as is.

Related: #974, #1026, #972, #973, #844, #845, #837, #834.
