# SSM / Mamba / GDN Layers

State-Space Models (SSMs) are the non-attention half of every hybrid model in the support matrix: Qwen3.5 (GDN), Qwen3-Next (SSM), Nemotron-H (Mamba-2), Qwen3.6 (GDN + vision). Their speedups vs PyTorch are the largest in the whole kernel suite — up to **9.95× on conv1d prefill**, **7.89× on GDR prefill**.

## The three variants in Atlas

| Variant | Models | Core op |
|---|---|---|
| **Mamba-2** | Nemotron-3 Nano / Super | Selective state-space update with causal conv1d + linear recurrence |
| **SSM** (classical Mamba) | Qwen3-Next-80B-A3B | Similar; older parametrisation |
| **GDN** (Gated Delta Rule) | Qwen3.5-35B, Qwen3.5-122B, Qwen3.6 | Delta-net variant with learned gating + softplus/sigmoid |

All three share structure: a QKVZ projection (expanded 4-way linear), a causal conv1d, a selective linear recurrence, and a gated output normalisation. The differences are in the recurrence and the gating.

## Why SSMs are fast on Atlas

Three things:

1. **Fused QKVZ preprocess.** The `ssm_preprocess.cu` kernel deinterleaves the 4-way projection output (Q, K, V, Z) and computes the GDN gate (softplus + sigmoid) in one pass. PyTorch does this in three separate kernels plus a reshape; Atlas does it in one.
2. **Fused Gated Delta Rule (GDR).** The linear recurrence itself is hand-written — one warp per (batch, head) walks the token sequence, maintaining the hidden state in registers. Causal conv1d is fused into the same pass.
3. **Hand-rolled causal conv1d.** The conv1d kernel takes advantage of the fixed small filter size (kernel width 4) to hold the entire filter in registers and stream inputs through.

Benchmark numbers (dim=8192):

| Op | Atlas (ms) | PyTorch (ms) | Speedup |
|---|---:|---:|---:|
| Conv1d prefill seq=32 | 0.0112 | 0.0205 | 1.82× |
| Conv1d prefill seq=128 | 0.0143 | 0.0776 | 5.41× |
| Conv1d prefill seq=512 | 0.0532 | 0.5296 | **9.95×** |
| Conv1d decode | 0.0041 | 0.0364 | **8.89×** |
| GDR decode 32vh dim=128 | 0.0143 | 0.0732 | 5.11× |
| GDR prefill seq=32 | 0.3612 | 2.7849 | 7.71× |
| GDR prefill seq=128 | 1.4111 | 11.1267 | **7.89×** |

The 9.95× at conv1d seq=512 is the largest compute speedup in the whole repo. PyTorch's `causal_conv1d_fn` on this shape walks the full sequence per batch element; Atlas's kernel fuses the whole thing into one launch with shared-memory tiling.

## The SSM state

Unlike attention, SSMs carry a **compressed hidden state** across the sequence. For Mamba-2 with `d_inner = 8192` and `d_state = 128`, the state is `[8192, 128]` FP32 per layer — about 4 MB per layer per sequence. For a 36-layer model, that's 150 MB per sequence — comparable in size to a full attention KV cache.

## Chunked SSM prefill

A chronic issue: prefilling a long prompt through an SSM layer requires computing the full linear recurrence from scratch. The intermediate state can be gigabytes if the prompt is 16k tokens and the batch is moderate.

The **chunked prefill** path in `kernels/gb10/<model>/<quant>/` breaks the prefill into chunks of (typically) 1024 tokens. Each chunk:

1. Starts from the state at the end of the previous chunk.
2. Processes its tokens through the recurrence.
3. Writes the ending state for the next chunk.

Savings: ~7–9 GB of scratch memory for long-context prefill, at negligible perf cost. Before chunked prefill, 122B at 8k context wouldn't fit in 119.7 GB. After it, it does.

See `docs/design/chunked-ssm-prefill.md` (and the `project_chunked_ssm_prefill_fix` note — there was a BF16 paged-dispatch bug where the FP8 kernel was called on a BF16 cache, producing NaN; fixed in wave-4).

## Marconi: SSM state snapshots for prefix caching

A naive prefix cache on an SSM model reads the attention KV for the prefix and — because it doesn't have the SSM state — recomputes the SSM layers from scratch. That defeats most of the win.

Atlas's **Marconi** (inside-joke name for the SSM snapshot cache) stores the SSM state at the end of each cached prefix alongside the KV. A warm prefix-cache hit restores both the attention KV *and* the SSM state. Output is byte-identical to the cold run.

Costs: an extra ~GB of snapshot storage per top-level cache entry. Worth it for repeat agent workloads. Controlled by `--ssm-cache-slots` (default 16) and `--ssm-checkpoint-interval` (default 256). See the `marconi.md` note and `crates/spark-runtime/src/prefix_cache.rs`.

## GDN: softplus + sigmoid fusion

Gated Delta Rule does a per-token gate computation:

```
g = softplus(dt) * sigmoid(beta)
```

where `dt` is the delta-time projection. The gate is used both to attenuate the state update and to form the output. Atlas fuses this into `ssm_preprocess.cu` — the gate appears as a byproduct of the QKVZ deinterleave.

Numerically, softplus is the chronic problem: `softplus(large x) = x`, but `softplus(very large x)` can overflow in FP32 if you compute naively. Atlas uses the standard `max(0, x) + log1p(exp(-|x|))` stable form.

## Known gotchas (lessons from the bug sweeps)

- **SSM catastrophic forgetting** — an older version had a bug where the snapshot state was sometimes restored with a stale conv1d buffer, causing a slow drift of coherence over long agentic sessions. Fixed by also snapshotting the conv1d tail-buffer.
- **Upstream Mamba bug on chunked prefill** — the vLLM fix approach (different from Mamba-2 reference) is what Atlas uses; the reference implementation has a subtle boundary issue at chunk seams.
- **GDN register-tile experiments** — the bench `gdn_regtile_results.md` tracks a long tail of tile-shape experiments. The current production choice is a middle ground; alternate shapes win on specific seq_len ranges but fail the full regression suite.

## What the code looks like

`crates/spark-model/src/layers/qwen3_ssm.rs` and `nemotron_mamba2.rs` contain the layer-level state machines. They:

1. Run the fused preprocess kernel to produce Q, K, V, Z and gate.
2. Run the causal conv1d on the Q/K/V branches.
3. Run the selective linear recurrence (GDR or Mamba-2 variant) to produce the output.
4. Run `gated_rms_norm(output, Z, weight)` for the final gated normalisation.

Each step calls into `spark-runtime::GpuBackend` via the layer's cached `KernelHandle`s — the per-step Rust code is short because the real work is in the 3–4 kernel calls it issues.

## Files to read

- `kernels/gb10/<model>/<quant>/ssm_preprocess.cu`, `gdr.cu`, `causal_conv1d.cu`
- `crates/spark-model/src/layers/qwen3_ssm.rs`, `nemotron_mamba2.rs`
- `crates/spark-runtime/src/prefix_cache.rs` (Marconi SSM snapshot)
- README "Atlas Spark" section — the SSM/GDN story in narrative form
