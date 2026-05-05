# FP8 Native Serving

FP8 is the second-most-common quantization format in Atlas after NVFP4. It's also the format where the most recent engineering work has landed — Qwen3.6 ships FP8-native, Nemotron's checkpoints are FP8, and Atlas now runs them end-to-end without a BF16 upcast on the critical paths.

## The two FP8 checkpoint shapes

Atlas sees two layouts on disk, both covered by `atlas_quant::fp8::Fp8Format`:

1. **Per-tensor scaled** — `weight` (FP8 E4M3 bytes) + `weight_scale` (one `f32` scalar per tensor). Common in vLLM-exported checkpoints.
2. **Block-scaled** — `weight` (FP8 E4M3) + `weight_scale_inv` (BF16, one scale per `block_size × block_size` tile, typically `128 × 128`). Used by `compressed-tensors` FP8 checkpoints from Qwen and Nemotron.

```rust
pub struct Fp8Format {
    pub block_size: usize,             // 128 typical
    pub scale_dtype: ScaleDtype,       // Fp32 (per-tensor) or Bf16 (block)
}
```

Per-tensor scaled checkpoints can be read as a degenerate block case (`block_size = ∞`, one scale covers everything). The kernel code handles both with the same fragment-time dequant.

## FP8 E4M3: range and quirks

E4M3 is `sign(1) | exp(4) | mantissa(3)`, bias 7. Finite range is `[-448, +448]`. There is no infinity encoding; `0xFF` / `0x7F` are NaNs. The per-tensor scale maps the activation's dynamic range into E4M3's representable window.

Atlas ships a 256-entry `fp8_e4m3_to_f32_lut` in `atlas-quant/src/fp8.rs` for CPU sanity checks and scale-inversion arithmetic at weight-load time. The GPU hot path does not use the LUT — it uses the PTX instruction `cvt.rn.bf16.e4m3` (FP8 → BF16 on the fragment boundary), which is available on SM121 unlike the NVFP4 instruction.

## Native FP8 vs dequant-to-BF16

Two paths exist in the code today, selected per layer in `spark-model::quant_format`:

- **Dequant-to-BF16** — read FP8 from memory, convert to BF16 in the GEMM fragment, do MMA in BF16. Simple, always correct, works for every FP8 checkpoint.
- **Native FP8** — read FP8, feed directly to the FP8 MMA instructions (`mma.sync.aligned.m16n8k32.row.col.e4m3.e4m3.f32`). Keeps FP8 all the way through tensor cores, only converts to BF16 when writing activations back.

Native FP8 is the faster path and the one "FP8 native serving" refers to. The two gotchas:

1. **KV-cache interaction.** If the KV cache is FP8, the attention kernel needs to either keep it FP8 through the MMA (native) or convert it per-fragment (dequant). The `paged_decode_attn_fp8.cu` kernel does the former; the older `paged_decode_attn_fp8kv.cu` does the latter.
2. **Calibration scales.** Native FP8 demands well-conditioned scales. A calibration pass (`--fp8-kv-calibration-tokens N`) runs for the first N tokens, collecting online max-`‖K‖` / max-`‖V‖` stats and updating the KV scales before CUDA-graph capture. Without calibration, long-context FP8 KV drifts; *with* it, it matches BF16 to within measurement noise.

Typical deployment: `--kv-cache-dtype fp8 --fp8-kv-calibration-tokens 256`. 256 is enough warm-up; larger values cost prefill time without measurable quality lift.

## The Qwen3.6 FP8 story

Qwen3.6-35B-A3B is FP8-native: weights, KV, MTP head, vision tower all FP8. Atlas's support here was a sequence of fixes logged across several bug sweeps:

- **FP8 weight loading for native MTP** (wave-6) — the NVFP4 MTP loader was force-BF16 when `ignore_modules` listed `mtp.*`; fixed to fall through to FP8 dequant when the scales were BF16-block rather than NVFP4-group.
- **FP8 prefill shared-experts allreduce reorder** (wave-6) — the shared-expert path was all-reducing FP8 activations across EP=2 *before* the final BF16 downcast, which silently dropped precision. Reordered so the allreduce sees BF16.
- **FP8 KV calibration during CUDA-graph capture** — graph capture froze the scales at their t=0 values; now calibration runs before capture, and capture picks up the converged scales.
- **Spontaneous `<think>` fix** — Qwen3.6's FP8 path exposed a reasoning-parser bug where the model emitted `<think>` outside the template's expected position. Fixed across four codepaths.

End-to-end result: Qwen3.6-35B-A3B serves at ~90 tok/s on GB10 with full FP8 coherence including Claude Code-style tool use. See `project_coder_next_fp8.md` (in the history notes) for the before/after.

## When FP8 beats NVFP4

On GB10:

- **Qwen3.6 (FP8-native)** — obviously. Don't re-quantize.
- **Nemotron-3** — FP8 is the only quant that preserves the Mamba-2 A/B/C/D projections' numeric character; NVFP4 drifts on long-context Mamba.
- **Anything where you have the memory budget** — FP8 is 2× NVFP4 bytes but doesn't pay the software-E2M1 cost per fragment, which can matter at small batch × large K shapes.

When NVFP4 beats FP8:

- **122B-class models that only fit in NVFP4.** 76 GB NVFP4 weights leave room on a single GB10; 152 GB FP8 do not.
- **Short-context deployments where the compression just saves money.**
- **Qwen3.5** — the family has native NVFP4 checkpoints and calibrates well.

## KV cache dtypes (`--kv-cache-dtype`)

The full list:

| Dtype | Bytes/elt | Notes |
|---|---:|---|
| `bf16` | 2 | Baseline; no quantization |
| `fp8` | 1 | E4M3 + per-tensor scale (calibrated) |
| `nvfp4` | 0.5 | E2M1 + FP8 per-block scale |
| `turbo3` | 3/8 | 3-bit WHT + Lloyd-Max (TurboQuant) |
| `turbo4` | 0.5 | 4-bit WHT + Lloyd-Max — ~2× lower MSE than NVFP4 at same bit rate |
| `turbo8` | 1 | WHT + FP8 — outlier-resistant FP8 |

The Turbo family is Atlas-specific: Walsh-Hadamard rotates out the outlier structure typical of transformer K/V activations before quantizing with an optimally-placed codebook. For the same bit count, turbo4 gives measurably lower per-token error than NVFP4 on models with large RMSNorm weights. It is purely additive — you opt in via `--kv-cache-dtype turbo4`; the NVFP4 path is unchanged. See `docs/design/turboquant-nightjob-2026-03-31.md`.

## Files to read

- `kernels/gb10/<model>/fp8/` — per-model FP8 kernel sets (Qwen3.6 has its own leaf).
- `kernels/gb10/<model>/<quant>/paged_decode_attn_fp8.cu` — native FP8 KV attention.
- `crates/atlas-quant/src/fp8.rs` — `Fp8Format`, scale dtypes, LUT.
- `crates/spark-runtime/src/kv_cache.rs` — `KvCacheDtype::Fp8` sizing + calibration plumbing.
- `docs/design/fp8-native-design.md` — the authoritative design note.
