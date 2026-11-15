# RetroInfer attribution: retrieval attention host side (lane L33)

Atlas's RetroInfer retrieval attention translates the index math of microsoft/RetrievalAttention (the
repository holds RetroInfer, arXiv 2505.02922, VLDB 2026) into a Rust CPU reference and Rust host logic.
The Python and Triton code was read and re-expressed; nothing was copied verbatim. The kernels of lanes
L34a and L34b (`kernels/ada/common/retro_{kmeans,select,gather,attn}_ada.cu`) carry their own entries.

- Upstream: https://github.com/microsoft/RetrievalAttention, commit `03f912c`
- License: MIT, Copyright (c) Microsoft Corporation (`LICENSES/RetroInfer-MIT.txt`). The repository
  `NOTICE` carries a RetroInfer entry; `CITATIONS.md` cites both papers.

## What was taken from where

| Atlas file | Upstream source | What it follows | Changes for Atlas |
|---|---|---|---|
| `crates/spark-model/src/port/retro/config.rs` | `config/config.py:35-74` (`generate_config`), `cache_hub/retroinfer_cache.py:57-121` | Segments `max(round(n/8192), 1)`; centroids `round(n/16)` moved to the nearer multiple of `lcm(8, segments)`; `nprobe`, estimation-zone, update-segment, `UPDATE_CENTROIDS`, `UPDATE_NPROBE`, `UPDATE_ES` sizes | Python's round-half-to-even kept; budgets in f64 like Python floats; capacities for fixed device buffers (upstream grows tensors); a build-parameter hash for cache keys |
| `crates/spark-model/src/port/retro/oracle.rs` | `cache_hub/kmeans.py:1-260` (`_triton_assign_kernel`, `_triton_update_kernel`, `_triton_k_means_train`, `triton_reverse_index`, `triton_index_add`, `segment_k_means`); `cache_hub/retroinfer_cache.py:540-556` | Mean centering and the mean added back; uniform-stride initialization in float32; nine L2-normalized training passes per segment on that segment's tokens; one global plain-mean pass; empty clusters keep their centroid; value sums of raw values; cluster member lists | A CPU reference in f64 with a fixed reduction order (upstream uses relaxed atomics); ties toward the lower centroid; member lists ascending; scoped threads over KV heads; an append form for updates |
| `crates/spark-model/src/port/retro/oracle_eval.rs` | `cache_hub/retroinfer_cache.py:754-815` (`sparse_attention`) | Softmax over centroids per query head, summed over the GQA group; empty clusters masked; top `nprobe` as the retrieval zone, the next clusters as the estimation zone; the estimation zone's size-weighted attention merged with the exact zone | The merge written as one softmax over the union (paper Eq 2-4), equal to upstream's LSE merge; quality measures (mass recall, output error, Jaccard) for shadow mode |
| `crates/spark-model/src/port/retro/update.rs` | `cache_hub/retroinfer_cache.py:595-728` (`_update_kv_cache`, `decode_update_kv_cache`) | Every 1,024 generated tokens past the window are clustered with one segment and appended; `nprobe += UPDATE_NPROBE`, `es += UPDATE_ES` | Positions instead of a moving steady zone (the tail is read in place from the paged KV cache); ranges aligned to the KV block size; refused (dense) instead of reallocated when a buffer is full |

Kernel lanes (same upstream and commit; their own headers and NOTICE entries carry the details):

| Atlas file | Lane | Upstream source | Notes |
|---|---|---|---|
| `kernels/ada/common/retro_kmeans_ada.cu`, `crates/spark-model/src/port/retro/build.rs`, `build/**` | L34a | `cache_hub/kmeans.py` (segment_k_means), `cache_hub/retroinfer_cache.py:505-561, 595-711` | device build from the paged KV pool into the host's fixed-capacity index, deterministic |
| `kernels/ada/common/retro_select_ada.cu`, `crates/spark-model/src/port/retro/select.rs`, `select/**` | L34a | `cache_hub/retroinfer_cache.py:765-770` (scoring and top-k); the semantics of `batch_gemm_softmax` | CUTLASS code not copied; device top-k with fixed capacities |
| `kernels/ada/common/retro_gather_ada.cu`, `crates/spark-model/src/port/retro/{gather.rs, gather_reference.rs}` | L34b | `cache_hub/retroinfer_cache.py:754-815` (execution buffer); `library/retroinfer/retroinfer_kernels/src/gather_copy.cu:100-217` (access pattern only) | rewritten for Atlas paged KV, BF16 and FP8 |
| `kernels/ada/common/retro_attn_ada.cu`, `crates/spark-model/src/port/retro/{attn.rs, attn_reference.rs}` | L34b | `cache_hub/retroinfer_cache.py:754-815` (estimation-zone weighted attention and merge) | forked from Atlas's own `kernels/gb10/common/paged_decode_attn_bf16_gqa.cu` (AGPL) |

Atlas-native (algorithm only, header `// Algorithm: RetroInfer, arXiv 2505.02922 (VLDB 2026).`):
`state.rs`, `device.rs`, `runtime.rs`, `runtime_build.rs`, `selector.rs`, `attend.rs`, `shadow.rs`, `verify.rs`,
`mod.rs`.
Atlas-native with no algorithm content: `api.rs`, `arena.rs`, `lifecycle.rs`, `metrics.rs`.

Not taken: the CPU wave buffer and its thread pool (`cache_hub/wave_buffer*`, the pinned-host KV store and
LRU GPU block cache; the 4B's KV fits on the card), the CUTLASS `batch_gemm_softmax`, the Triton kernels
themselves, and the RetrievalAttention CPU ANN design (no code for it in the repository). No upstream CUDA
code is copied: L34a and L34b write their own kernels (L34b's forked from Atlas's paged-decode kernel).
The upstream `weighted_flash_decoding` and `gather_copy*` CUDA files are cited as design references only:
`library/retroinfer/retroinfer_kernels/src/gather_copy.cu:100-217` for the gather's access pattern, and the
estimation-zone weighted attention and merge at `cache_hub/retroinfer_cache.py:754-815` for the attention.
L34b's file headers and `NOTICE` entry carry the same citations and their changes.

## The MIT notice

MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
