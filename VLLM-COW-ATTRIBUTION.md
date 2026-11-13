# vLLM attribution: copy-on-write partial-tail prefix hits (port sprint lane L28)

**Headline: a prefix hit that ends inside a KV block now gets a private copy of that block instead
of the donor's writable tail (Atlas issue #1193).** Row `prefix_cow_tail` (`AVAROK_PREFIX_COW_TAIL`,
off; `AVAROK_PREFIX_SUBBLOCK=cow` is the equivalent lever). With it off, every path is the pre-sprint
code.

## Upstream

- Project: vLLM, <https://github.com/vllm-project/vllm>, commit `8a36f08`.
- License: Apache License, Version 2.0 (`LICENSES/Apache-2.0.txt`). Copyright contributors to the
  vLLM project. The repository carries no NOTICE file.
- Scheme translated (no source copied; the partial-hit and copy-before-forward scheme re-expressed
  in Rust over Atlas's radix tree and paged KV cache):

| Upstream | Lines | Atlas |
|---|---|---|
| `vllm/v1/core/single_type_kv_cache_manager.py`, `_has_partial_local_hit` | 169-179, 324-330 | `crates/spark-runtime/src/radix_tree/inner/cow_walk.rs`, `walk_cow`: a hit that ends inside a block is recorded as a copy source and kept out of the shared blocks |
| `single_type_kv_cache_manager.py`, `allocate_new_blocks`, `_apply_cow` | 375-411, 460-480 | `crates/spark-model/src/model/trait_impl/prefill_b/prefix_cow.rs`, `copy_tail_block`: a fresh private block replaces the shared tail; both endpoints stay alive until the copy is enqueued |
| `vllm/v1/core/kv_cache_manager.py` (pending copies) and `vllm/v1/worker/gpu/model_runner.py` (copy before the forward reads) | 879-894; 1382-1389 | `prefix_cow.rs`, `prefill_b_apply_cow_tail`: the copies run on the prefill's own stream before the block table is used |

## Changes for Atlas

- The sub-block hit is found by Atlas's two existing sub-block arms (the prompt remainder is a
  prefix of a live child's key, or of the deepest matched node's partial suffix), not by a hash unit
  smaller than a block. Among several matching children the lowest block wins (deterministic).
- Only the CoW-aware lookup (`PrefixCache::lookup_cow`) returns a tail; plain `lookup` stays
  block-aligned for every other caller (single-shot prefill, batched reservations, peeks).
- The copy is one `copy_d2d_async` per attention layer and side (K, V), whole block, per-layer
  strides (mixed KV dtypes and per-layer dims included).
- The source is held by one KV reference across the allocation instead of a second block-pool
  retain; when that hold is the last reference (the allocation evicted the source), the stream is
  drained before the block can be reallocated.
- No copy is offered in a multi-rank world (ranks agree on a block-granular match) or under
  `--high-speed-swap` (a private copy has no disk image). A failed allocation falls back to the
  block-aligned match, so the SSM snapshot depth never exceeds the rows the sequence holds.
- An exact repeat of a non-block-aligned prompt can now reach the Marconi leaf snapshot at prompt
  end, because the snapshot lookup uses the CoW-extended match.

## Files carrying the translation

`crates/spark-runtime/src/radix_tree/inner/cow_walk.rs`,
`crates/spark-model/src/model/trait_impl/prefill_b/prefix_cow.rs` (and their tests,
`crates/spark-runtime/src/radix_tree/tests/cow_tail.rs`,
`crates/spark-model/src/model/trait_impl/prefill_b/prefix_cow_tests.rs`). Each derived file starts
with the vLLM attribution header. NOTICE carries the matching entry.

## The Apache-2.0 notice

Licensed under the Apache License, Version 2.0 (the "License"); you may not use these files except
in compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0 (also `LICENSES/Apache-2.0.txt` in this repository).
Unless required by applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied. See the License for the specific language governing permissions and limitations under
the License.
