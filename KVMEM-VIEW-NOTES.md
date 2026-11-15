# Bounded KV view: notes (port lanes L31 and L32)

Row `kv_view_selection` (`AVAROK_KV_VIEW`), flags `--kv-view-budget-tokens` (0 or absent = off),
`--kv-view-sink-tokens` (128), `--kv-view-recent-tokens` (2048), `--kv-view-query-tokens` (512), env knobs
`AVAROK_KV_VIEW_HISTORY_LAMBDA` and `AVAROK_KV_VIEW_GRAPH_ALPHA` (0 = off), diagnostic env
`AVAROK_KV_VIEW_SELECTION_LOG` (off unless set). APPROXIMATE attention: off on every
target and never default on in this sprint.

## Source and license

The technique is described in arXiv:2609.04852 (KVMem) and in the idea descriptions of the port research report
`76-kvmem.md` (sections 2.1, 2.2, B2 to B5). The kvmem and kvmem-llama.cpp repositories carry no license, so no file
from them was opened, read or copied. Every file here is an independent implementation from the published
description and carries the T-KV header:

> Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no
> source copied from kvmem/kvmem-llama.cpp (that repository carries no license).

The citation entry is lane L31's (`CITATIONS.md`). There is no NOTICE entry (nothing was copied).

## What lane L32 built (the wiring)

| Piece | File |
|---|---|
| Selector (`KvSelector` of the L29 seam) | `crates/spark-model/src/port/kv_view/selector.rs` |
| State, hook work, live views, selection | `port/kv_view/{state,plans,live,select}.rs` |
| View arithmetic (host twin of the kernel) | `port/kv_view/view.rs` |
| History and one-hop propagation | `port/kv_view/history.rs` |
| Device buffers and kernel launch | `port/kv_view/device.rs`, `kernels/ada/common/kv_view_table_ada.cu` |
| Adapter to L31's core | `port/kv_view/core_api.rs` |
| Knobs, counters | `port/kv_view/{config,metrics}.rs` |
| Scheduler side (turns, hooks, flags, metrics) | `crates/spark-server/src/scheduler/kv_view.rs` |
| Hook lines | `scheduler/phase_continue_prefills.rs` and its `phase_continue_prefills/` children, `prefill_a_step.rs`, `prefill_b_step.rs` |
| GPU oracle of the kernel | `crates/spark-model/examples/kv_view_table_microtest.rs` |

## How a turn works

1. Every prefill call is wrapped (`scheduler::kv_view::begin` / `finish`). `begin` classifies the prompt (ChatML: the
   block before the generation header is the final message; a `<tool_response>` or assistant block is a continuation),
   arms L31's query capture for the final message (it keeps the last `--kv-view-query-tokens` rows), records the call's
   streams and publishes their pool slots to the seam.
2. During the prefill every K write updates L31's per-page pre-RoPE mean-key sketch (`plan_jobs` + `update_jobs`), and
   the pre-RoPE Q rows of the armed span are captured.
3. After the final chunk `set_kv_view` scores the context's 128-token blocks (page-softmax, GQA mean), spreads one hop
   (alpha), ranks by the decayed history (lambda), picks (the tail block, the recent window, the sink, the question,
   then the best blocks, within the budget), and binds the view.
4. From the second output token on, each full-attention layer's decode and verify attention reads the view: the
   selected pages before the view's tail, then every page from the tail on (the recent window, the question and every
   token appended since). Dropping whole pages before the tail keeps the RoPE geometry, because K is stored rotated
   at its absolute position; the view length is the real length minus the dropped tokens.
5. The next turn of the conversation (same session hash, a prompt that repeats the selected prefix) prefills its paged
   chunks against the previous view plus the new pages, then selects again (new user message) or keeps it (tool
   result, assistant continuation; a kept view that grows past twice the budget selects again). An edited prefix
   starts over. A retry of the same round does not commit the round's history.

The first output token of a turn always comes from the prefill's own logits (the question is not replayed). Chunk 0
of a prefill runs through the cache-skip attention, which the seam leaves dense, so a short follow-up question is
prefilled against the full context.

## Graph safety

Decode steps replay CUDA graphs with no host code. While the view is on, every decode and verify plan is a `View` over
one fixed arena; `kv_view_build_tables` runs inside the graph and rebuilds the arena from the step's real tables. A
row finds its view by content: the descriptor whose anchors (the physical blocks at logical pages 0, the tail page
and the last page at selection) match the row, the largest last page winning. A row with no match gets its real
table and length, so sequences without a view, padding rows and every step before the first selection are exactly
dense. Descriptors are written only between steps, on the decode stream. A sketch or capture failure turns the view
off and clears every descriptor, so already captured graphs fall back to identity tables.

## Decisions and limits (all stated in REPORT.md)

- The MTP head's own attention stays dense (its KV is not the paged pool the view selects from);
  `atlas_kv_view_mtp_dense` says so. GDN layers keep their full recurrent state.
- The attention-edge sampler of the graph propagation is not built (it needs an attention-feedback kernel and the
  "under 1% of decode" gate is unmeasured), so `AVAROK_KV_VIEW_GRAPH_ALPHA` drives only the neighbor term
  (`alpha / 4` to each adjacent block); the edge share stays in place.
- The history residual is the round score minus the round mean; hits are the top 90% of the round (integer cut).
- The view refuses to run with the host KV tier (restored pages bypass the sketch), and the seam refuses KV dtypes
  other than BF16 and FP8, `--high-speed-swap` and vision input. Only one approximate selector runs per serve.
- The sketch undoes RoPE with the model's `rope_theta` and `rotary_dim`; a model with per-layer RoPE overrides or
  YaRN scaling would sketch inexactly (selection quality only; attention always reads the real K and V).
- The KV-budget charge covers the fixed buffers, the capture slots and the scorer scratch (an estimate, the actual
  size is logged); the per-page sketch (3.1% of BF16 KV, 6.25% of FP8 KV) is allocated after the build from what
  `--gpu-memory-utilization` left. A proportional charge needs a seam extension (REQUESTS.md R2).
- A batched chunk-0 prefill with no page tables leaves sketch gaps; that sequence's selection is skipped
  (`sketch_gap`).
- Selection debug log (lane L02's request): with `AVAROK_KV_VIEW_SELECTION_LOG=1` (off unless set) each selection
  logs one line per attention layer, `kv_view selection seq=<pool slot> layer=<n> view_blocks=.. tokens=..
  recall_mass=.. blocks=[b0,b1,...]`, the logical KV pages of the view (selected pages, then every page from the
  tail on), which is what `bench/niah_grid.py --selection-log` reads with its default regex. One selection serves
  every layer, so the lines of a selection differ only in `layer`. Decode steps replay graphs with no host code and
  only append tail pages, so the selection is the one point logged; a kept view (a continuation) logs nothing.

## Owed on the card (Q15 in the master plan)

The GPU lane script for this view (not in this repository) runs the kernel oracle, the identity gate (a budget above the context: greedy
output equal to the row off), the needle grid (5 depths x 16K to 128K with a 16K view: in-view recall and answer),
multi-needle with distractors, three questions on one 64K document, decode tok/s against context at batch 1 and 4,
and the VRAM ledger. Quality bar for any future default: needle recall at least 95% of dense and multi-needle within
2 points. Nothing here was measured yet.
