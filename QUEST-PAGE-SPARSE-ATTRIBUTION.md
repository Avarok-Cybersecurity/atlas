# Quest page-sparse decode attention (lane L30, row `attn_page_sparse`)

**Approximate, off by default on every target, never default on.** Long-context decode on the RTX 4060 Ti is
bound by reading the whole KV history once per generated token: the Qwen3.5-4B reads 32 KiB of BF16 KV per
context token, about 17 ms per step at 128K. Quest keeps a small summary
per page of keys and attends only to the pages whose summary says they can matter.

## Sources and license

| Source | What Atlas takes | Kind | License |
|---|---|---|---|
| Quest, J. Tang, Y. Zhao, K. Zhu, G. Xiao, B. Kasikci, S. Han, "Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference", ICML 2024, arXiv:2406.10774 | per-page key min/max, the bound `sum_c max(q_c * kmin_c, q_c * kmax_c)`, top-k page selection | algorithm (published method) | paper |
| arczhi/llama-tierkv, `scripts-5060ti/proto/proto_page_sparse.cu` (kernels `build_summaries`, `page_scores`, `topk`) | the correct per-channel bound (not the repository's pager, which scores `q * max(kmin, kmax)`, that is `q * kmax`); the sinks + top-k + recent-window budget; one selection shared by verify rows | idea only | MIT, Copyright (c) 2023-2026 The ggml authors, `LICENSES/llama.cpp-MIT.txt` (the file has no header of its own) |

No upstream source was fetched, read or copied: the lane worked from the formula and the design as
described, not from any upstream file. Every file carries an attribution header.

## Atlas files

- `kernels/ada/common/attn_page_sparse_ada.cu`: `attn_ps_summary_update`, `attn_ps_page_score`,
  `attn_ps_compact` (new CUDA, written for Atlas).
- `crates/spark-runtime/src/kv_cache/page_summary.rs`: the summary layout and arena.
- `crates/spark-model/src/port/sparse_quest/`: `mod.rs` (wiring), `config.rs` (flags), `budget.rs` (KV-budget
  charge), `arena.rs`, `launch.rs` (kernel ABI), `selector.rs` (the seam's `KvSelector`), `metrics.rs`,
  `reference.rs` (CPU reference), `tests.rs`, `mock_tests.rs`; `scorer.rs`, `scorer_rank.rs`,
  `scorer_store.rs`, `scorer_tests.rs` (the page scorer of the window desk, below).
- `crates/spark-model/src/layers/ops/port_rows/attn_page_sparse.rs`: the row resolver.
- `crates/spark-model/examples/page_sparse_microtest.rs`: the GPU oracle.

## Changes against the prototype's design

- **Summaries keyed by physical block.** A page's summary lives in the slot of the physical KV block holding
  its first token. Every index a kernel needs comes from the device block table and lengths, so a captured
  CUDA graph replays correctly as sequences grow, and sequences sharing a cached prefix share its summaries.
- **Trust, then verify.** Each (slot, head) record carries a completeness mask (one bit per page token), a
  fingerprint (8 bytes of the key at the page's first token) and a lock. The scorer trusts a summary only when
  the mask is full and the fingerprint matches the pool; otherwise it rebuilds the summary from the keys and
  writes it back. A missed update (a graph captured before allocation, a block restored by copy, a batched
  chunk-0 prefill without a table) costs one page read, never a wrong bound.
- **Directed rounding.** `kmin` is rounded toward minus infinity and `kmax` toward plus infinity when stored as
  f16, so the stored bound is still an upper bound of every key's logit.
- **GQA.** One page score per sequence, combining the per-head bounds of every query head by max (default) or
  sum (`AVAROK_ATTN_SPARSE_GQA=sum`); verify rows take the max over rows, so they share one selection.
- **Exact top-k** in one CTA per sequence: an 8-pass radix select over 64-bit keys (score, then page index, so
  ties go to the newer page), instead of the prototype's thread-0 merge that cost 3.77 ms.
- **No staging.** The selection becomes a compacted block table `[sink pages | top-k pages | recent window]`,
  in chronological order, in selector-owned fixed buffers that Atlas's unchanged paged decode kernel reads
  (`SparsePlan::View` through the L29 seam). Keys carry RoPE from the write, so skipped pages are position-safe.
- **The identity gate.** At or below `--attn-sparse-min-ctx`, or when the budget covers every page, the view is a
  verbatim copy of the dense table, so attention there is bit-identical to dense.

## Ranking the window desk's recall pages (request R3 of lane L36)

With `kv_window_desk` also on, the seam's one selector slot belongs to the desk, so `apply` registers a
`QuestScorer` with the desk (`kv_window::selector::set_page_scorer`) instead of the Quest selector. The Quest
selector still sizes and charges the page summaries and runs the summary update kernel (the scorer forwards the
seam's `on_kv_written`); attention is the desk's compacted table, and Quest only decides which demoted pages
come back. Both rows on is therefore "Quest-ranked recall into the desk", not a startup refusal.

- **Summaries leave with their block.** When a page's leading block moves to host RAM, the scorer copies that
  page's summary (every layer and KV head, only when its completeness mask is full) into a per-logical-page
  host store, because the arena is keyed by physical block and the block is reused after the copy.
- **The query is captured in the graph.** Decode and verify replay without host code, so `on_query` enqueues
  plain device copies (a memset, the first block-table column, the lengths and the query rows) into fixed
  per-layer buffers; they are captured with the graph and refreshed on every replay whatever batch it carries.
  A sequence is found by its first physical block, the desk's key. Zero lengths mark padding rows.
- **Ranking.** Between steps the host reads the rows back, scores each candidate page with the Quest bound per
  layer (GQA max or sum as for the selector), turns each layer's scores into z-scores over the candidates and
  averages them over layers, ties to the newer page. The desk gets whole pages (4 blocks of 16 tokens).
- **Limits.** The query is the last decode step's, not the step that will read the recalled pages; prefill
  queries are not captured (their hook runs before RoPE); a sequence absent from the last step has no query
  and that round recalls nothing. Series: `atlas_attn_sparse_desk_rank_total{outcome}`,
  `atlas_attn_sparse_desk_pages_ranked_total`, `atlas_attn_sparse_desk_summaries_total{result}` (only when the
  scorer is registered).

## Receipts (GPU, deferred)

`examples/page_sparse_microtest.rs` PASS (summary bytes equal to the CPU reference, scores within 1e-4, exact
top-k sets, the rebuild path, the identity gate bit-identical through `paged_decode_attn`); then the L02 NIAH grid
at 32K/64K/128K/262K, decode ms/step against context, perplexity drift and one agent task at 64K+. The row is
recorded, never flipped.
