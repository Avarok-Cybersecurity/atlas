# LMCache attribution: KV transfer basics (port sprint lane L25)

Atlas's batched KV block transfer, batched sequence swap, pinned host pool and LRU SSM blob store
follow designs in LMCache. No LMCache code was copied: every Atlas file below is an independent
Rust implementation written from the upstream design, for Atlas's own paged KV layout.

- Upstream: https://github.com/LMCache/LMCache, commit `3aefae1`
- License: Apache License, Version 2.0 (`LICENSES/Apache-2.0.txt`). LMCache ships no NOTICE file;
  the repository `NOTICE` carries an LMCache entry.

## What was taken from where

| Atlas file | Upstream design source | What it follows | Changes for Atlas |
|---|---|---|---|
| `crates/spark-runtime/src/kv_cache/block_io.rs` | `csrc/cuda/mp_mem_kernels.cu:358-849` | One batched submission of many (device, host, bytes) entries per object instead of one blocking copy per region; entries split into bounded batches | No layout transform (each layer's K or V block is one contiguous region); entries split at 4096 per backend call; chunk planner that splits a region across staging chunks |
| `crates/spark-runtime/src/kv_cache/block_io_swap.rs` | `csrc/cuda/mp_mem_kernels.cu:251-345` (staged DMA in plan steps), `csrc/cuda/phase_timing_recorder.cu` (per-phase GB/s) | Staged transfer through a pinned buffer, timing per phase | Two-buffer staging whose copy of chunk i overlaps the file write (or read) of chunk i-1; one `AVAROK_KV_SWAP_TIMING` line per image; `atlas_kv_swap_*` counters |
| `crates/spark-runtime/src/pinned_pool.rs` | `lmcache/v1/memory_allocators/lazy_memory_allocator.py:80-200`, `paged_tensor_memory_allocator.py`, `csrc/cuda/mem_alloc.cpp` | Reserve the whole range once, pin in 64 MB chunks, grow on a background thread while the first chunks already serve; page-granular runs; page-aligned base | `GpuBackend::host_register` (tier-owned in `pinned_hosts`), the pin thread binds the device context first, a refused pin stops growth with one log line, no hugepage path |
| `crates/spark-runtime/src/pinned_pool_free.rs` | `lmcache/v1/memory_allocators/tensor_memory_allocator.py` | Free list with coalescing and alignment | Best fit through a size index; extents never cross a pinning chunk |
| `crates/spark-model/src/model/trait_impl/sequence/state_io.rs` | as `block_io.rs` | Batched save and restore of the swap image | The file format is unchanged; the row-off path is the pre-sprint code |
| `crates/spark-model/src/model/ssm_tier/store.rs` | LMCache cache policies (LRU, touch on hit) and the pinned allocator above | LRU-on-get eviction, blobs in pinned page runs | One stamp index serving the old FIFO order and LRU; heap fallback for blobs the pool cannot place |

Not taken: LMCache's per-format offset kernels (`csrc/cuda/mem_kernels.cu`), its fused
multi-chunk kernel and the GPU gather kernel (`csrc/cuda/mp_mem_kernels_detail.cuh:175-215`). The
lane's optional `kernels/ada/common/kv_block_gather_ada.cu` was NOT written: the plan writes it only
if the batched copy path measures below 8 GB/s on the card, which is a GPU-pass decision.

## The Apache-2.0 notice

Licensed under the Apache License, Version 2.0 (the "License"); you may not use these files except
in compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0 (also `LICENSES/Apache-2.0.txt` in this repository).
Unless required by applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied. See the License for the specific language governing permissions and limitations under
the License.
