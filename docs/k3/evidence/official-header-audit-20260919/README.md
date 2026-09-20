# Official K3 header and rank-memory audit — 2026-09-19

Pinned checkpoint: [`moonshotai/Kimi-K3@f831ab66814297da540d832a5235f8e904f29d06`](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06).

The audit fetched 96 safetensors headers (75,765,816 bytes) and only the 69 tiny `A_log` tensor payloads (35,328 bytes). It did not download the full checkpoint or run a GPU. Header metadata contains 497,220 tensors: all **497,052 text tensors** are accounted for by the core inventory and production TP planner; 168 non-text tensors are excluded. The pinned `text_config` equals the repository's official config fixture.

## TP8 device-weight budget

All ranks have the same planned size. The report executes the production `plan_tensor_bytes` and replicated-shape validator against actual header shapes/dtypes, including the guarded `A_log` normalization.

| Component | Bytes per rank | GiB per rank |
| --- | ---: | ---: |
| Resident text weights | 230,438,412,016 | 214.6125 |
| Of which replicated | 40,505,956,864 | 37.7241 |
| Largest local tensor / retained staging headroom | 2,348,810,240 | 2.1875 |
| Known extra GPU weight-binding allocations | 0 | 0 |
| Current admission before explicit reserve | 232,787,222,256 | 216.8000 |
| Previous 30% allowance admission | 301,918,745,861 | 281.1837 |

The old inherited 30% allowance added 69,131,523,605 bytes without matching the marked K3 binder's allocation behavior. The current bound counts additional BF16 copies only for FP32 embedding, LM head and final norm. These three official tensors are BF16, so the additional allocation count is zero. Marked dense binding reuses resident pointers; packed binding only creates pointer metadata; layer structs and metadata are on the host. The shared conversion predicate is enforced by the engine-facing binder. FP32 fixtures still reserve their exact extra copies and retain the source allocations.

This is **not a complete startup or inference memory certification**. Explicit reserve remains necessary for engine workspace, KDA/MLA state, KV, NCCL/CUDA, lazy kernel allocations, allocator overhead and concurrent activity. The largest local staging buffer is host memory; the existing admission retains an additional same-sized device margin conservatively. Do not call the full 216.8 GiB GPU payload: resident weights are 214.6 GiB.

A device reporting 288 GB decimal would have about 51.421 GiB left after current admission and before explicit reserve; a device reporting 288 GiB would have 71.200 GiB. Verify actual usable free memory on the provider's device. No B300 allocation or inference has been performed.

## A_log export exception

All 69 KDA tensors are stored as F32 `[128]`, while the pinned config and model implementation specify **96 active heads**. Every tensor's final 32 floats was fetched and verified equal to zero; each payload has a hash and source byte range in `receipt.json`. The core plan preserves 96 heads, checks the entire tail before any GPU allocation, then splits the active prefix by TP rank. It rejects nonzero, NaN or infinite tail values, alternate padding shapes, and unsupported head layouts.

This matches the checkpoint-side guard discussed in [Moonshot discussion #150](https://huggingface.co/moonshotai/Kimi-K3/discussions/150) and the [NVIDIA Megatron Bridge K3 documentation](https://docs.nvidia.com/nemo/megatron-bridge/nightly/models/kimi/kimi-k3.html). The actual bytes, rather than the discussion alone, establish this revision's zero-padding. No head-count changes or general shape relaxation are applied.

## Host memory and current performance limitation

Current `host_decode::host_layer` caches non-packed layer weights as FP32, including shared experts and attention projections, even when KDA/MLA recurrence and packed expert GEMMs use CUDA. The calculated logical FP32 payload is **89,861,658,352 bytes per rank (83.690 GiB)** or **718,893,266,816 bytes across eight ranks (669.521 GiB)**. Embedding/LM-head weights are not included in this host layer cache. Packed routed weights are not expanded to FP32.

`cpu_bind::pull` moves each vector out of the temporary map, so layer assembly does not duplicate every dense tensor. Conversion temporarily also owns the raw device-copy bytes; the largest such local copy is 88,080,384 bytes (84 MiB). KDA convolution concatenation can reallocate its vectors. Allocator capacity, metadata, state, temporary activations, page cache, and runtime overhead are additional to the logical payload. During checkpoint upload, each rank can also hold a 2.1875-GiB host staging vector and map a checkpoint shard; mapped address space is not a measured resident-memory bound. Eight ranks share the snapshot but own their FP32 layer caches.

Prefer **1.5–2 TB host RAM** for the current host-reference path. A 1-TB node is not demonstrated safe merely because the model weights fit GPUs. Measure RSS/PSS and node available memory through loading and first inference. Removing these host projections remains necessary for useful production throughput; this audit does not imply that full-model host matvec inference will be fast.

## Reproduce without GPU or full weight download

```sh
python3 scripts/k3/audit_headers.py --output /path/to/k3-header-audit
AVAROK_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo run -p avarok-core \
  --example k3_rank_memory -- 8 /path/to/k3-header-audit/headers /path/to/k3-header-audit/a-log
```

The Python tool refuses responses that ignore the requested byte range and caps individual header size. The core example checks the actual tensor geometry and zero tails. Without header arguments it produces an explicitly labeled config/BF16 estimate, which is retained separately as `tp8-config-estimate.json`; the official checkpoint uses F32 for several small KDA/router tensors.

Receipts contain metadata hashes and tiny tensor hashes, not full weights, private addresses or credentials. Production-shape reconstruction and corrupt-tail tests are in `avarok-core::kimi_k3::tp::tests`; runtime admission tests cover exact conversion copies and reserve rejection.
