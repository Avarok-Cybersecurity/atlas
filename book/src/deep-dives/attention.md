# Attention & Paged KV Cache

The attention path is where Atlas's biggest speedups land — up to **6.02×** vs PyTorch decode, up to **4.95×** vs PyTorch prefill. This chapter walks the kernels, the KV cache allocation model, and the pieces that make them fast on GB10.

## Two kernels, two shapes

Prefill and decode look like different problems:

- **Prefill** — `seq_len` is large (prompt length), batch is typically 1 per forward. The GEMM has one long axis; memory pressure is on the Q·K^T tile.
- **Decode** — `seq_len` for the new query is 1; the KV cache accumulated so far is long (thousands of tokens). The GEMM is matrix-vector; memory pressure is on the full-history K/V load.

Different kernel shapes:

| Kernel | Source | Role |
|---|---|---|
| `inferspark_prefill_v47.cu` | prefill | Flash Attention v2, `cp.async` 2-stage pipeline, 16×8×16 BF16 MMA, 2 CTAs/SM |
| `inferspark_prefill_fp8kv.cu` | prefill | Same structure, FP8 KV read path |
| `paged_decode_attn_nvfp4.cu` | decode | Online softmax, split-K parallelism, NVFP4 K/V dequant at fragment boundary |
| `paged_decode_attn_turbo3_128.cu` | decode | Optimised variant for `head_dim=128`, `turbo3` KV |
| `kv_cache_append.cu` | write | Per-token K/V write into paged cache |

## Prefill: Flash Attention v2 on SM121

The v47 prefill kernel is the "production" prefill path. It follows Flash Attention v2's tiled online-softmax pattern:

1. Load a tile of Q into shared memory once; it stays resident for the whole kernel.
2. Stream tiles of K and V through shared memory using `cp.async` (2 stages in flight).
3. For each K/V tile:
   - Compute partial Q·K^T into a register fragment.
   - Apply causal masking + softmax rescaling against the running max + running sum.
   - Multiply by V, accumulate.
4. After all tiles, normalize by the final sum.

Key design choices that matter on SM121:

- **2 CTAs/SM, not 1.** The shared-memory budget is generous enough that two blocks per SM fit comfortably, and the second block absorbs scheduler bubbles from the first. 1 CTA/SM was the baseline; going to 2 was a ~15% win.
- **`mma.sync.aligned.m16n8k16` for Q·K^T and for A·V.** These are the bread-and-butter BF16 MMA fragments on Blackwell.
- **Shared-memory xor-swizzle** on the Q tile to avoid bank conflicts during the `mma` load.
- **Causal masking as a predicate inside the mma loop**, not a separate kernel. A predicate add into the softmax scratch is almost free; a separate pre-mask would be a full kernel.

The `_fp8kv` variant adds one extra step: the K/V tiles are E4M3 in memory, not BF16. Loads are half the bytes; the first thing the consumer warp does is `cvt.rn.bf16.e4m3` on each fragment. Net effect: ~1.8× BW saved on the K/V load, at the cost of one extra instruction per fragment.

Prefill numbers on Qwen3-Next-80B shapes (hidden=2048, 16Q / 2KV heads, head_dim=256):

| seq_len | Atlas (ms) | PyTorch (ms) | Speedup |
|---:|---:|---:|---:|
| 32 | 0.0062 | 0.0077 | 1.26× |
| 128 | 0.0184 | 0.0205 | 1.11× |
| 256 | 0.0246 | 0.1217 | **4.95×** |
| 512 | 0.0494 | 0.0513 | 1.04× |

The dramatic win at seq=256 is the kernel hitting its sweet spot where the `cp.async` pipeline is fully saturated and shared memory is the bottleneck instead of DRAM. At small seq the fixed launch cost dominates; at large seq PyTorch's Flash-Attention-2 backend catches up.

## Decode: split-K online softmax

Decode attention is a different kernel because the shapes are different. For each new token, we need Q (one row) × K (full history). The hot axis is K — thousands of elements of history per head, dozens of heads.

Atlas's decode kernel parallelises across **two** axes:

1. **Head** — one warp per (Q-head, KV-head) pair.
2. **Split-K** — the full K/V history is chopped into `N` chunks; each chunk gets a CTA that produces a partial softmax + partial attention output. A second pass reduces across chunks.

The split count `N` is adaptive — chosen per call based on history length and current batch. Long history → more splits. This is the "adaptive split count" in the kernel table.

The online softmax pattern survives from prefill but with a different shape: each CTA maintains a running max and a running exp-sum for its K-chunk. The reduction across chunks at the end is a single warp's work.

**NVFP4 K/V in decode** is where the headline throughput comes from. The decode kernel reads packed E2M1 nibbles, unpacks two per byte in registers, multiplies by the block scale, and feeds BF16 to the Q·K^T and A·V MMAs. K and V in memory are half the bytes of BF16 → 2× BW saved → 2× decode throughput on BW-bound steps.

Decode numbers (same Qwen3-Next-80B shapes):

| history | Atlas (ms) | PyTorch (ms) | Speedup | Effective BW |
|---:|---:|---:|---:|---:|
| 64 | 0.0061 | 0.0077 | 1.25× | 22.7 GB/s |
| 256 | 0.0123 | 0.0164 | 1.33× | 43.3 GB/s |
| 1024 | 0.0205 | 0.0267 | 1.30× | 102.8 GB/s |
| 4096 | 0.0485 | 0.2924 | **6.02×** | 173 GB/s |

At 4k history we're at ~63% of GB10's 273 GB/s peak — tight but well below the roof. Further improvement here is the hot lane of kernel work.

## The paged KV cache

Atlas follows vLLM's paged-attention model: KV is allocated in fixed-size blocks (default 16 tokens per block) from a pool, not per-sequence. Key advantages:

- **No fragmentation.** A completed request returns its blocks to the pool; a new request claims fresh ones. No moving, no compaction.
- **Copy-on-write prefix sharing.** When prefix-caching hits, the prefix's blocks are shared between the cached and new sequence until divergence.
- **Scheduler-friendly.** Memory budget is expressible in "free blocks", a scalar — easy to reason about under load.

`spark-runtime::kv_cache::PagedKvCache` owns the pool. `KvCacheConfig` derives the pool size from:

- `--max-seq-len` × `--max-batch-size` → total token capacity
- Model's `num_hidden_layers` × `num_key_value_heads` × `head_dim` × 2 (K and V) × bytes-per-elt (from `KvCacheDtype`) → per-token storage
- Block size (16) → number of blocks

Block allocation is O(1) from a free list. Eviction is policy-driven — the scheduler picks victims (LRU by default) when the pool is full.

## Paged attention in the kernel

The decode kernel takes three extra arguments beyond a non-paged version:

- `block_tables` — `[batch, max_blocks_per_seq]` of block indices.
- `context_lens` — `[batch]` of valid token counts.
- `block_size` — compile-time constant for indexing arithmetic.

Inside the kernel, each CTA computes its K/V pointer for a given history position by indexing into the block table: `block_ptr = k_cache + block_tables[seq][pos/16] * block_size * head_dim * bytes_per_elt`. Gather-SMEM-MMA: gather the block pointers into shared memory, then run MMA against the resulting shared tile. Pattern from the FlashInfer paper (MLSys 2025 Best Paper, cited in the README).

## RadixAttention prefix caching

Prefix caching is a radix-tree lookup on the token prefix. The tree's nodes own KV blocks; a lookup that matches N tokens of prefix returns those N tokens' blocks already resident. `--enable-prefix-caching` turns it on.

Typical hit rates in production:

- **Agent workloads** (Claude Code, OpenCode, Cline): 90%+ on the system prompt + tool schemas.
- **Multi-turn chat**: 60–80% on the conversation context.
- **Cold workload** (single one-shot): 0%.

TTFT goes from ~400ms cold to ~40ms warm on Qwen3.5-35B. The prefix-cache chapter of the engine test suite validates the hit rate and the byte-identical output under warm vs cold.

For hybrid SSM+attention models, a matching "Marconi" SSM snapshot cache lives in `spark-runtime::prefix_cache` — the SSM state at the end of the prefix is checkpointed alongside the attention KV, so a warm hit reconstructs the full model state, not just the attention cache. Without this, prefix caching on an SSM model would produce silently incorrect output.

## Files to read

- `kernels/gb10/<model>/<quant>/inferspark_prefill_v47.cu` — the prefill kernel.
- `kernels/gb10/<model>/<quant>/paged_decode_attn_*.cu` — decode kernel variants.
- `kernels/gb10/<model>/<quant>/kv_cache_append.cu` — per-token KV write.
- `crates/spark-runtime/src/kv_cache.rs`, `prefix_cache.rs`, `radix_tree.rs` — Rust side.
- README "Citations" — links to the Flash Attention 2, Flash Attention 4, FlashInfer, SageAttention 3, and LeanAttention papers.
