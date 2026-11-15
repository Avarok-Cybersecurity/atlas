# Attribution: host-RAM KV tier core (lane L26a)

Atlas's host-RAM KV tier (`crates/spark-runtime/src/kv_host_tier/`) follows designs in LMCache and
vLLM, applies an acceptance rule from Cinference, and implements two ideas published for KVMem. No
upstream code was copied: every Atlas file below is an independent Rust implementation written from
the upstream design or the published description, for Atlas's own paged KV layout and copy engine.

- LMCache: https://github.com/LMCache/LMCache, commit `3aefae1`, Apache License 2.0 (no NOTICE file).
- vLLM: https://github.com/vllm-project/vllm, commit `8a36f08`, Apache License 2.0, Copyright
  contributors to the vLLM project.
- Cinference: https://github.com/satellitedown/cinference, commit `383e5db`, Apache License 2.0,
  derived from NInfer by Neroued and contributors (https://github.com/Neroued/ninfer).
- KVMem: arXiv:2609.04852. The kvmem and kvmem-llama.cpp repositories carry no license; they were not
  read. Only the paper's description was used.
- ARC: N. Megiddo and D. S. Modha, "ARC: A Self-Tuning, Low Overhead Replacement Cache", USENIX FAST 2003.
- License texts: `LICENSES/Apache-2.0.txt`. The repository `NOTICE` carries the entry for this lane.

## What was taken from where

| Atlas file | Upstream design source | What it follows | Changes for Atlas |
|---|---|---|---|
| `mod.rs`, `control.rs` | LMCache `l1_manager/base.py`, `eviction_controller.py`, `prefetch_controller.py`; vLLM `kv_offload/cpu/gpu_worker.py`, `simple_kv_offload/manager.py` | A pinned host copy of prefix-cached blocks, stored after prefill and restored into fresh GPU blocks | One in-process tier behind one lock; per-step budget, lazy drain, cost gate and teardown for the spark-model glue |
| `index.rs` | LMCache `l1_manager/base.py:132-188` | Staging -> ready -> read-locked states; visible only after the write completed | Tokens stored and compared on every hit (a forged 128-bit key misses); byte totals kept for the watermark, the in-flight cap and the gauges |
| `writeback.rs` | vLLM `gpu_worker.py:543-731`; vLLM `simple_kv_offload/manager.py`; LMCache `base.py` | D2H on a dedicated stream that waits on the compute stream, one completion event per transfer, publish on completion | Atlas block regions (K then V per layer) as the copy unit, one batched `copy_batch_d2h_async` per call, failure synchronizes and returns every run |
| `restore.rs` | vLLM `gpu_worker.py:543-731`; LMCache `base.py:132-188`, `prefetch_controller.py:1114-1160` | H2D with source access order "any"; read locks with expiry; in-flight admission watermark | A lease is never dropped while its copy may run; TTL applies only after completion; tickets polled or waited on |
| `evict.rs` | LMCache `eviction_controller.py:160-195` | Watermark batch eviction (0.8 trigger, 0.2 ratio), skipping locked keys | Synchronous at write time (no daemon thread); one more block at a time when the pool cannot place a run; demotion to the cold store; a lookup's counted hits are never evicted by its own promotions |
| `policy.rs` | LMCache `eviction_policy/lru.py:72-95` | LRU where a request's later blocks leave first | Tail-first applied to every batch, for LRU, ARC and Heat alike |
| `policy_arc.rs` | vLLM `kv_offload/cpu/policies/arc.py:28-60`; the ARC paper | T1/T2, B1/B2 ghost lists, adaptive target | Admission decoupled from replacement (the byte watermark drives eviction); replacement skips leased and staging keys |
| `policy.rs` (Heat) | KVMem paper (selection frequency halved after N epochs) | Fewest accesses first, all heat halved every N accesses | Lazy halving by epoch; recency breaks ties |
| `lazy.rs` | LMCache `lazy_offload_policy/eviction_aware.py:60-140`; vLLM `simple_kv_offload/manager.py:644-690` | Buffer stores; copy only blocks within `horizon_steps` of the free queue's eviction head, capped per step, optional deadline; store ahead of the allocator | The near-head window is the larger of the horizon and the store-ahead target; blocks the budget could not start stay buffered; gone blocks are counted as dropped |
| `cost.rs` | LMCache `multiprocess/kv_load_policy.py`; Cinference `bench/README.md:103-175` | Restore only when the copy is estimated under half the recompute; trust a cost fit only after its held-out p95 relative error passes and no ordering inversion is seen | Applied online to live EWMA rate models, with prequential (predict before update) observations as the held-out set; until both models pass, the gate always restores |
| `fingerprint.rs` | LMCache `token_hasher.py:233-262` | Rolling chain of the previous key and this block's tokens | SHA-256 with domain separation over namespace (model fingerprint), adapter, parent key and tokens, truncated to 128 bits |
| `events.rs` | vLLM `gpu_worker.py:543-731` | Pooled events, completion polled without blocking | Device timing only where the backend has timing events, else the host clock |
| `cold_store.rs` | LMCache `eviction_controller.py` (L1 demotes to L2) | A slower tier below host RAM | Contract trait plus two defaulted token-aware methods |

Not taken: LMCache's daemon eviction thread, its remote and multi-process storage backends and its
per-format offset kernels; vLLM's connector and scheduler integration (Atlas's glue is lane L26b).

## The Apache-2.0 notice

Licensed under the Apache License, Version 2.0 (the "License"); you may not use these files except
in compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0 (also `LICENSES/Apache-2.0.txt` in this repository).
Unless required by applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied. See the License for the specific language governing permissions and limitations under
the License.
