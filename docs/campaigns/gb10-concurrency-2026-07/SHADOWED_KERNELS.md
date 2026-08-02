# The decode-serialization root cause: shadowed kernels

**Date:** 2026-07-26 · **Box:** dgx1 · **Model:** `centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf`

## Symptom

Phase A measured Atlas losing to vLLM at every concurrency above 1:

| C | Atlas tok/s | vLLM tok/s | Atlas TPOT p50 | vLLM TPOT p50 |
|---|---|---|---|---|
| 1 | **27.3** | 14.2 | 36.6 ms | 70.1 ms |
| 2 | 20.5 | 27.8 | 96.9 ms | 71.4 ms |
| 4 | 20.1 | 53.3 | 198.1 ms | 74.8 ms |
| 8 | 22.7 | 98.8 | 350.7 ms | 80.6 ms |
| 16 | 23.5 | **168.9** | 673.5 ms | 93.8 ms |

Aggregate throughput is FLAT in C, and TPOT is `36.6 ms x C` to within a few
percent. Both say the same thing: the GPU does exactly one sequence's worth of
decode work at a time. This is not "slow at concurrency", it is **not
concurrent at all** on the decode path.

## Root cause

`crates/atlas-kernels/build.rs`, `collect_cu_files`:

```rust
// Override layer: model-specific kernel files shadow common ones
for f in find_cu_files(model_dir, source_ext) {
    let stem = f.file_stem().unwrap().to_str().unwrap().to_string();
    files.insert(stem, f);   // <-- whole-file replacement, keyed by stem
}
```

Shadowing is **whole-file, not per-symbol**. `kernels/gb10/qwen3.6-27b/nvfp4/
gated_delta_rule.cu` exists to add register-tiled prefill kernels (its header
calls it a "35B model shadow"). It was forked BEFORE `kernels/gb10/common/
gated_delta_rule.cu` gained four decode kernels, so for the 27B those four
never compile:

- `gated_delta_rule_decode_f32_strided`
- `gated_delta_rule_decode_f32_strided_norm`
- `gated_delta_rule_decode_f32_norm`
- `gated_delta_rule_decode_f32_conv_norm`

The first two ARE the N-sequence batched recurrent decode. The Rust path that
uses them is complete and wired (`trait_decode_multi_seq/ssm_batched_recurrent.rs`),
gated on:

```rust
ATLAS_SSM_BATCHED_RECURRENT == "1" && self.gdn_f32_strided_k.0 != 0 && n > 1
```

`try_kernel` returns `KernelHandle(0)` for a missing kernel and logs only at
`debug`, so the gate **failed closed, silently**, and every concurrent decode
fell back to the per-sequence loop.

### Runtime proof

With the merged-main image, per SSM layer:

```
DEBUG spark_model::layers: Optional kernel
  'gated_delta_rule::gated_delta_rule_decode_f32_strided' not loaded
```

(all four, x48 SSM layers). With the ported kernels: zero such lines.

This also explains why **Phase B moved nothing** (+3% at C=16). It A/B'd CUDA
graphs and scheduler changes while the actual lever was structurally
unreachable — it was never tested, because it could not run.

## Fix

Ported the four kernels into `kernels/gb10/qwen3.6-27b/nvfp4/gated_delta_rule.cu`
as an exact piecewise copy of `common/` lines 344-956, plus the helper block
they need (`gdn_unpack_bf16x2`, `gdn_pack_bf16x2`, `gdn_warp_reduce_sum`, the
`SSM_STATE_NORM_*` defines) under an include guard — the fork never carried
them. Verified: `nvcc -arch=sm_121a --fmad=false` compiles clean, `cuobjdump
-symbols` shows all four in the 27B cubin, and a live serve reports none
missing.

## Blast radius — this is fleet-wide, not a 27B quirk

Auditing every model dir that shadows a `common/` file for dropped entry
points:

| Model | File | Dropped |
|---|---|---|
| qwen3-next-80b-a3b, qwen3.5-{27b,35b-a3b,122b-a10b,397b-a17b}, gemma-4-26b-a4b | `gated_delta_rule.cu` | all **4** decode kernels |
| qwen3.6-35b-a3b, holo-3.1-{0.8b,4b,35b-a3b}, ornith-1.0-9b | `gated_delta_rule.cu` | `_conv_norm` |
| minimax-m2-229b, mistral-small-4, nemotron-*, step3p7-flash | `rms_norm.cu` | 8-10 norm kernels |
| ~every model | `w4a16_gemm.cu`, `moe_w4a16_grouped_gemm.cu` | `w4a16_dequant`, `moe_w4a16_grouped_gemm` |

Not every row is a bug — a model may genuinely not need a kernel, and some of
these are only reachable through paths that model never takes. The point is
that **today it is impossible to tell the difference**, because a dropped
kernel is indistinguishable from a deliberate omission.

## Guard

`collect_cu_files` now diffs `extern "C" __global__` entry points whenever a
model file shadows a common one and emits a `cargo:warning` naming each
dropped kernel. A warning, not an error: dropping a kernel can be deliberate.
The point is to make it a VISIBLE choice rather than a silent one.

## Verdict (A/B complete, 2026-07-26)

Both legs on the same `:msdecode` image, identical serve geometry to Phases
A/B, differing ONLY by `ATLAS_SSM_BATCHED_RECURRENT=1`.

* **Control validity:** the per-seq leg matches the Phase B baseline to
  <=0.4% at every C — the four ported kernels are bit-inert when off.
* **Engagement proven, lever small:** batched-recurrent is +1-2% tok/s /
  -1.0..-1.7% TPOT p50, monotone across every clean config and every C>=2 —
  too consistent to be noise, too small to matter.
* **The serialization root cause is NOT the GDN recurrent scan.** TPOT still
  scales ~36ms x C (96/193/344/662 at C=2/4/8/16). The dominant per-sequence
  cost lives elsewhere in the decode step (candidates: per-seq attention,
  FFN batched-arm eligibility, LM head, per-seq sample+D2H). Next probe:
  `ATLAS_SSM_MS_PROFILE=1` phase-split at C=4 to locate the ~160ms.
* balanced_long remains error-polluted (pool-exhaustion kills 1-5/leg) —
  latency numbers from that config are invalid, tracked separately.

**Disposition: the kernel port + audit guard land on their own merits**
(hygiene: the flag was structurally unrunnable; the guard prevents the drift
class fleet-wide; the strided conv1d is byte-identical with a negative-control
microtest). The performance hypothesis they enabled is refuted at this batch
geometry.

## Known remaining limits (this fix does not address them)

1. ~~conv1d stays per-sequence~~ FIXED in this change:
   `causal_conv1d_update_l2norm_f32_strided` takes explicit input/output row
   strides so the whole batch goes in one launch; byte-identical to the
   per-seq loop (`conv1d_strided_microtest`, 3 seeds, with a negative control
   proving the unstrided batch=N launch is corrupt). Older kernel sets fall
   back to the loop via the handle-0 gate.
2. **MTP is gated `active.len() == 1`** on every speculative path, so
   speculative decode is off entirely at C>1 regardless of this fix. That is
   the other half of the C=1 vs C>1 cliff.
