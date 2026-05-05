# Single-Kernel-Launch Prefill for SSM Layers

## Problem Statement

During chunked prefill, each SSM layer's h_state (128×128 FP32 = 64KB per head) is:
1. Loaded from GPU DRAM → shared memory (kernel start)
2. Updated for `chunk_len` tokens in shared memory
3. Written back to GPU DRAM (kernel end)
4. Optionally clamped by `normalize_ssm_states()` (between chunks)
5. Repeat for next chunk

At 40K tokens with 8192 tokens/chunk = 5 chunks, this means:
- **5 global memory round-trips** per head per SSM layer (load + store × 5)
- **4 lossy normalization clamp events** (Frobenius norm → MAX_NORM=50)
- Each clamp event uniformly scales the entire state matrix, destroying relative magnitudes

vLLM avoids this entirely: their fla Triton kernel processes the **full sequence** in a single launch with FP32 registers carrying state. No inter-chunk serialization, no normalization needed.

## Proposed Design

### Core Idea

Replace the per-chunk kernel launch pattern with a **single persistent kernel launch per SSM layer** that processes ALL prefill tokens for one sequence. The kernel keeps h_state in shared memory for the entire sequence — identical to the current WY4-persistent design, but spanning all chunks instead of one.

### How It Works Today

```
Scheduler:
  for chunk in chunks:
    model.prefill_chunk(chunk_tokens, seq)    ← launches 30 SSM kernels
    model.normalize_ssm_states(seq)           ← launches 1 norm kernel

Per SSM layer per chunk:
  kernel_launch(h_state, Q[chunk], K[chunk], V[chunk], gate[chunk], beta[chunk], ...)
    1. Load H from DRAM → smem
    2. Process chunk_len tokens
    3. Store H from smem → DRAM
```

### How It Would Work

```
Scheduler:
  model.prefill_full(all_tokens, seq)         ← launches 30 SSM kernels ONCE

Per SSM layer (SINGLE launch):
  kernel_launch(h_state, Q_all, K_all, V_all, gate_all, beta_all, total_len, ...)
    1. Load H from DRAM → smem (once)
    2. Process ALL tokens (0..total_len) in tight loop
       - Optional: intra-kernel norm check every 4096 tokens
    3. Store H from smem → DRAM (once)
```

### Key Challenge: Upstream Data Dependencies

The SSM kernel can't process all tokens in one shot because it depends on **upstream layer outputs**. The forward pass is:

```
Layer 0 (SSM):   embed → QKVZ_GEMM → conv1d → GDN_prefill → gated_norm → out_proj → MoE
Layer 1 (SSM):   layer0_output → QKVZ_GEMM → conv1d → GDN_prefill → ...
Layer 2 (SSM):   layer1_output → ...
Layer 3 (Attn):  layer2_output → Q/K/V_proj → flash_attn → out_proj → MoE
...
```

Each layer needs the previous layer's output. We can't run all 30 SSM layers' GDN kernels on the full sequence simultaneously because the GDN input (Q, K, V, gate, beta) comes from earlier projections that depend on the previous layer's output.

### Solution: Two-Phase Architecture

**Phase 1 — Layer-by-layer QKVZ projection (chunked as today)**

The GEMM/conv1d/gate computations for each SSM layer are bandwidth-bound and don't involve h_state. Run these exactly as today, chunked, layer-by-layer. But **buffer the GDN inputs** (Q, K, V, gate, beta) for the full sequence instead of consuming them immediately.

**Phase 2 — Single-launch GDN kernel per layer**

After all chunks of a given layer's projections are complete, launch ONE GDN kernel with the full Q/K/V/gate/beta sequence. The kernel keeps H in shared memory throughout.

```
For each layer:
  Phase 1 (chunked, as today):
    for chunk in chunks:
      RMS_norm(hidden[chunk])
      QKVZ_GEMM(hidden[chunk]) → qkvz_buf[chunk_offset..chunk_offset+chunk_len]
      deinterleave(qkvz_buf[chunk])
      BA_GEMM(hidden[chunk]) → gates_buf[chunk_offset..chunk_offset+chunk_len]
      conv1d(qkvz_buf[chunk]) → conv_out_buf[chunk_offset..chunk_offset+chunk_len]
      L2_norm(conv_out_buf[chunk])

  Phase 2 (single launch):
    GDN_persistent(h_state, Q_full, K_full, V_full, gate_full, beta_full, total_len)
    gated_rms_norm(gdn_output_full)
    output_proj(gdn_output_full) → hidden_full   [or chunked]
    residual_add(hidden_full)
    MoE(hidden_full)                              [or chunked]
```

### Memory Cost

Buffering the full sequence's GDN inputs requires additional GPU memory:

Per SSM layer, per token:
- Q: 128 × 16 heads × 2 bytes = 4 KB
- K: 128 × 16 heads × 2 bytes = 4 KB
- V: 128 × 32 heads × 2 bytes = 8 KB
- gate: 32 × 4 bytes = 128 B
- beta: 32 × 4 bytes = 128 B
- **Total per token: ~16.3 KB**

For 40K tokens: 16.3 KB × 40,000 = **652 MB per SSM layer**

With 30 SSM layers processed sequentially (not simultaneously), we only need ONE buffer set:
- **Total additional memory: ~652 MB** (reused across layers)

On GB10 with 120 GB GPU memory, 652 MB is ~0.5%. Easily affordable.

### Simpler Alternative: Increase Chunk Size Dramatically

Instead of restructuring the entire prefill pipeline, we could simply set `--max-prefill-tokens 32768` or even `65536` (the full sequence). The current WY4 persistent kernel already handles arbitrary `seq_len`. The only constraint is activation buffer memory:

- hidden_states: 32K × 2048 × 2 = 128 MB
- ssm_qkvz: 32K × 12288 × 2 = 768 MB
- scratch/gates: ~200 MB
- **Total: ~1.1 GB for 32K chunk**

For 65K tokens (full sequence in one chunk):
- **Total: ~2.2 GB** — still well within memory

With a single chunk, there are **zero normalization events** and **one global memory round-trip** per SSM layer. This achieves the same goal without restructuring the code.

**Trade-off**: One giant chunk means the entire prefill must complete before decode can start. With multiple chunks, the scheduler can interleave prefill and decode for concurrent requests. For single-user Claude Code sessions, this doesn't matter.

### Recommendation

**Phase 0 (immediate)**: Try `--max-prefill-tokens 32768` or `65536`. Zero code changes. If this fixes the quality issue at 40K tokens, it validates the hypothesis that inter-chunk normalization is the problem.

**Phase 1 (if Phase 0 works)**: Make the large chunk size the default for single-sequence workloads. Add a CLI flag like `--single-sequence-mode` that maximizes chunk size.

**Phase 2 (if throughput matters)**: Implement the two-phase architecture for concurrent workloads where interleaving prefill/decode is important. This is a bigger change but the right long-term architecture.

### Implementation Estimate

| Phase | Effort | Risk | Impact |
|-------|--------|------|--------|
| Phase 0: Large chunk size | 0 (config change) | None | Validates hypothesis |
| Phase 1: Default large chunks | 1 day | Low | Production fix |
| Phase 2: Two-phase architecture | 1-2 weeks | Medium | Scalable solution |

### Open Questions

1. Does the WY4 kernel handle very long sequences (32K-65K tokens) correctly? Need to verify `unsigned int seq_len` doesn't overflow and shared memory access patterns remain correct.
2. At 65K tokens in one chunk, the kernel runs for ~65K/4 = 16K WY4 iterations. FP32 precision analysis from the research agents shows this is fine (~2e-3 relative error), but should be verified empirically.
3. The MoE GEMM and output projection use the same activation buffers. With 32K+ tokens, do these fit in the scratch arena?
