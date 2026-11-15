<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# RetroInfer retrieval attention on Atlas (`retro_attn`)

_What it is, how to turn it on, what it costs, what is decided and what is still unmeasured._

**Status: written, not yet run.** The host side (lane L33) and the kernels (lanes L34a, L34b) are on the
`port/consumer-opt` branch. Nothing here is a measurement on the RTX 4060 Ti; the numbers marked "modeled"
come from a bandwidth model. The GPU pass runs the shadow gate first.

## What it does

For a long context, each full-attention layer's decode (8 of the 32 layers on Qwen3.5-4B) attends to three
zones per KV head instead of every token:

- **Exact:** the first `--retro-sink` tokens, the recent tail (at least `--retro-window` tokens, plus whatever
  was generated since the last index update), and every token of the `--retro-retrieval-budget` fraction of
  clusters whose centroids score highest for this query.
- **Estimated:** the next `--retro-estimation-budget` fraction of clusters, each standing in for its members
  through its centroid (the mean key) and the sum of its values, weighted by its size.
- **Dropped:** the rest.

The index is a segmented spherical k-means over the layer's keys (one centroid per `--retro-avg-cluster`
tokens, segments of `--retro-segment-tokens`), built once per sequence when its context first reaches
`--retro-min-context`, then grown every `--retro-update-interval` generated tokens. Selection scores every
centroid with the GQA group's query heads (softmax over centroids per head, summed over the group). Exact and
estimated parts are merged in one softmax, so the output is softmax attention over exact tokens plus
estimated clusters (arXiv 2505.02922, Eq 2-4).

**It is approximate attention.** Outputs differ from dense attention. The row is off on every target, and
this sprint never turns it on by default.

## Turning it on

```bash
spark serve ... --retro-attn                       # or AVAROK_RETRO_ATTN=1
spark serve ... --retro-shadow                     # measure only, output unchanged
```

| Flag | Default | Env twin | Meaning |
|---|---|---|---|
| `--retro-attn` | off | `AVAROK_RETRO_ATTN` (the row) | retro attention replaces dense decode attention |
| `--retro-retrieval-budget` | 0.018 | `AVAROK_RETRO_RETRIEVAL_BUDGET` | fraction of clusters attended exactly |
| `--retro-estimation-budget` | 0.232 | `AVAROK_RETRO_ESTIMATION_BUDGET` | fraction estimated through centroids |
| `--retro-segment-tokens` | 8192 | `AVAROK_RETRO_SEGMENT_TOKENS` | k-means segment size at build |
| `--retro-avg-cluster` | 16 | `AVAROK_RETRO_AVG_CLUSTER` | tokens per centroid |
| `--retro-sink` | 4 | `AVAROK_RETRO_SINK` | attention-sink tokens kept exact |
| `--retro-window` | 64 | `AVAROK_RETRO_WINDOW` | recent tokens kept exact |
| `--retro-min-context` | 16384 | `AVAROK_RETRO_MIN_CONTEXT` | dense below this context |
| `--retro-update-interval` | 1024 | `AVAROK_RETRO_UPDATE_INTERVAL` | generated tokens clustered at a time |
| `--retro-shadow` | off | `AVAROK_RETRO_SHADOW` | shadow mode (below) |
| `--retro-index-cache-mb` | 2048 | `AVAROK_RETRO_INDEX_CACHE_MB` | device arena for every retro buffer |
| (env only) | 64 | `AVAROK_RETRO_SHADOW_EVERY` | shadow samples every N-th decode step |
| (env only) | all | `AVAROK_RETRO_SHADOW_LAYERS` | comma list of attention-layer indexes |
| (env only) | off | `AVAROK_RETRO_SELECTION_LOG` | debug: one `retro selection: seq=<id> layer=<n> len=<tokens> blocks=[...]` log line per (decode step, layer), the KV blocks of the exact window (sink, retrieval-cluster members, tail); read by `niah_grid.py --selection-log`; reads index lists back from the device, so slow |

Every value is range-checked at startup; a bad value (CLI or env) is a startup error naming the flag.

Served: Qwen3-family full attention with head dim 256 and BF16 or FP8 KV. Refused at startup by the sparse
seam (`sparse_seam::validate_serve`): other KV dtypes (turbo and NVFP4 included), `--high-speed-swap`, and
vision input without `--no-vision`. At most one approximate selector runs per serve (`attn_page_sparse`,
`kv_view_selection` and `retro_attn` are mutually exclusive). Layers with another geometry stay dense.

## Shadow mode: the go or no-go

`--retro-shadow` changes nothing in the output. Every `AVAROK_RETRO_SHADOW_EVERY`-th decode step, for the
chosen layers, the layer's K and V for the whole context and the step's query are copied to the host and a
background thread runs the CPU oracle (`port/retro/oracle.rs`, `oracle_eval.rs`): it builds the index the
device path would build (once per sequence, then grown by the same update rule), selects the zones, runs the
merged attention, and compares it with dense attention. Per layer it publishes running means:

- `atlas_retro_shadow_mass_recall{layer,zone="retrieval"}`: dense softmax mass on the exact zone;
- `atlas_retro_shadow_mass_recall{layer,zone="retrieval_estimation"}`: plus the estimated clusters' members;
- `atlas_retro_shadow_out_rel_err{layer}`: relative L2 error of the retro output against dense;
- `atlas_retro_shadow_jaccard{layer}`: Jaccard of the exact tokens against the dense top tokens;

and one log line per sample. The copies are large (a 64K-token layer in BF16 is 256 MiB of K and V), one
sample is in flight at a time, and the first sample of a sequence pays the CPU k-means (seconds to minutes per
layer at 128K on 24 threads). That is why it is opt-in and sampled.

**Gate:** proceed to the timing rows only if, at 64K and 128K, on all 8 layers, the mean
`retrieval_estimation` mass recall is at least 0.97 and the mean output error at most 0.03, on the
needle-bearing prompts of the L02 grid. The hybrid model is the risk: only 8 layers carry KV, so each is a
long-range retrieval head the GDN layers cannot back up.

## How it runs

- **Eager decode.** The selector keeps per-sequence index state on the host and readies it on every step
  (build, update, cache install). While it is registered it vetoes decode CUDA graphs
  (`Qwen3AttentionLayer::decode_graph_unsupported`), and a step that is captured anyway plans dense (exact).
  Report 74's modeled gain (1.23x at 128K FP8 KV, 1.5x at 128K BF16 KV) has to beat the graph benefit it gives
  up; the timing rows measure both.
- **Builds.** At the last prefill chunk when the scheduler publishes it (`sparse_seam::publish_step`),
  otherwise at the sequence's first decode step. Prefill attention itself stays dense (TTFT unchanged).
- **Memory.** One arena of `--retro-index-cache-mb` MiB, allocated on the first eager hook after the model
  build and charged to the KV-block budget, holds live indexes, the prefix cache, build scratch, zone lists and
  the exact buffer. At 128K one sequence's eight layer indexes take about 290 MiB plus about 300 MiB of build
  scratch while building. When the arena is full, cached then idle indexes are evicted; a sequence that still
  does not fit runs dense (`atlas_retro_dense_fallback_total{reason="budget"}`).
- **Prefix reuse.** Each initial build is cached, keyed by the model instance, the layer, the build parameters
  and the physical KV blocks it covers. A new sequence whose block table starts with those blocks (a radix
  prefix-cache hit) reuses the index after a content fingerprint check, clusters only its own remainder, and
  copies the shared index only when it first changes it. A KV write into a covered block retires the index.
- **Speculative verify.** The K+1 rows of a verify select on their own queries; the retrieval zone is their
  union (every row's best cluster first), the estimation zone the first row's remaining clusters; each row then
  attends with its own causal extent. `atlas_retro_verify_union_size` reports the union. Draft and verify see
  the same approximate model; compare acceptance with retro on and off at 64K and 128K before using both.

## Metrics

`atlas_retro_active`, `atlas_retro_decode_total`, `atlas_retro_dense_fallback_total{reason}` (`short_context`,
`no_index`, `budget`, `mixed_batch`, `graph`, `kernel`, `shape`), `atlas_retro_index_build_seconds_total`,
`atlas_retro_index_bytes`, `atlas_retro_tokens_touched_ratio`, the three `atlas_retro_shadow_*` families,
`atlas_retro_update_total`, `atlas_retro_verify_union_size`. All zero with the row off.

## Known limits (v1)

- Decode graphs are off while it is on; a graph-captured variant needs device-side sequence resolution and a
  device-predicated update, which this version does not have.
- A batch that mixes served and unserved sequences runs dense for that step (`mixed_batch`).
- Without published sequence ids, sequences are matched by block table; a KV write the host cannot map to
  blocks (batched prefill) makes live indexes re-check before use.
- The MTP head's own attention is not routed through the seam and stays dense.
