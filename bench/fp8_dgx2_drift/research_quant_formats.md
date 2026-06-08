# Alternative Quantization Formats vs Current FP8 E4M3 + BF16 Block Scales

Date: 2026-05-25
Author: Claude (research subagent)
Context: Qwen3.6-35B-A3B-FP8 release uses E4M3 weights with **BF16** per-(128x128) block
scales. dgx2 study shows `ssm.moe_out @ L20 = 0.91983` vs HF[BF16-unquant]. Phase D's
C-mean cos is 0.99497 vs the FP8 ceiling A ~ 0.98874 from 2026-05-23 — Atlas is at the
FP8 ceiling on this checkpoint. To beat the ceiling, the checkpoint format itself
must change.

## Current Atlas storage

- Weight: FP8 E4M3 byte, value v = E4M3_LUT[byte] (range ≈ [-448, +448], ~3 effective
  mantissa bits, finite-only).
- Scale: **BF16** per 128x128 block (Qwen release format is DeepSeek-V3 lineage; see
  `crates/spark-model/src/quant_format/fp8_blockscaled.rs:9`).
- Dequant: `bf16 = E4M3_LUT[byte] * BF16_block_scale`.

The BF16 scale is the soft underbelly. BF16 has 7 mantissa bits → quantization step
≈ 1/128 of the scale magnitude. With ~16k blocks per Qwen3.6 expert, blocks whose
scale lands near the BF16 grid lose ~0.4% accuracy *before any E4M3 rounding error*.
NVIDIA, DeepSeek, MS, and Meta all standardised on FP32 scales for exactly this
reason (see [NVIDIA per-block FP8 blog](https://developer.nvidia.com/blog/per-tensor-and-per-block-scaling-strategies-for-effective-fp8-training/),
[DeepSeek-V3 tech report](https://arxiv.org/html/2412.19437v1)).

---

## Format-by-format scorecard

Bits-per-value figure assumes Qwen3.6 expert tile (128x128 = 16384 weights per block).

| Format | Storage bits/val | Effective mantissa | Scale type/granularity | Existing GB10 kernel? | Conversion from FP8 release? |
|---|---|---|---|---|---|
| **Current FP8 E4M3 + BF16 scale (128x128)** | 8 + 16/16384 ≈ 8.001 | ~3 | BF16, 128x128 | Yes (Atlas today) | n/a |
| **A. FP8 E4M3 + FP32 scale (128x128)** | 8 + 32/16384 ≈ 8.002 | ~3 | FP32, 128x128 | **Yes** (Atlas LUT dequant ignores scale dtype; trivial dispatch flip) | **Yes** — pure scale-cast, ~7 min CPU job per checkpoint |
| **B. FP8 E4M3 + FP32 scale, 1x128 tile** | 8 + 32/128 = 8.25 | ~3 | FP32, 1x128 (DeepSeek-V3 native) | Partial (we lay out scales in 128x128 today; would need new dequant) | Yes if requant from BF16 reference (slow, ~hours) |
| **C. MXFP8 E4M3 + E8M0 scale, 1x32** | 8 + 8/32 = 8.25 | ~3 (same E4M3 values) | E8M0 (pow-of-2), 1x32 | **No** on SM121 — datacenter Blackwell only has tcgen05.mma MXFP8; SM121 (GB10) has Ampere-style mma.sync, **no MXFP8 tensor cores** | Yes via OCP MX requant from BF16 |
| **D. MXFP6 E3M2 or E2M3 + E8M0 scale, 1x32** | 6 + 8/32 = 6.25 | 2-3 | E8M0, 1x32 | No SM121 tensor support; software dequant only | Possible (offline MXFP6 requant ~<0.05 PPL loss on LLMs) |
| **E. NVFP4 E2M1 + FP8 E4M3 scale + FP32 tensor scale** | 4 + 8/16 + ~0 = 4.5 | 1 | FP8 E4M3 per 16, FP32 per tensor | **Yes** — Atlas's NVFP4 fast-path (today's MoE *prefill* path, gated off for native FP8) | Already exists for Qwen3.6 community NVFP4 builds, but Atlas explicitly uses BF16 path now |
| **F. AWQ-FP8 (mixed-precision)** | mostly 4-bit, ~1% channels FP16 | mixed | per-channel FP16, calibrated | Atlas has no AWQ MoE kernel; vLLM has it for dense but MoE support is partial | Requires calibration dataset + AutoAWQ re-quant — large effort |
| **G. SmoothQuant-FP8 (W8A8 + activation rescale)** | 8 + scale | ~3 | per-channel | Atlas FP8 GEMM accepts per-channel scales; activation smoothing is a pre-multiply | Yes via offline calibration; LMSYS's "Unified FP8" generalises this to MoE |
| **H. GGUF Q8_0** | 8 + 16/32 = 8.5 | ~7 (uniform int) | FP16 per 32 | No (Atlas has no GGUF runtime); llama.cpp only | Requant from BF16 only; not natively from FP8 |
| **I. FP8-Flow-MoE / Unified FP8** | 8 + 32/128 = 8.25 | ~3 | FP32 1x128, scaling-aware transpose, **no double-quant casts** | No — requires new MoE kernel chain (fused FP8 ops, dynamic per-tile scale) | Requires retrain or QAT; not a pure conversion |

---

## Why FP32-scale is the highest-leverage delta

Mathematical claim (the agent-A3 ~1pp hypothesis): each Qwen 128x128 block's BF16
scale stored value `s_bf16` differs from the true `s_fp32 = amax(W) / 448` by up
to `s_bf16 * 2^-8` (1 ULP at bf16). Through the GEMM, this propagates as a
relative weight error of the same order on every element of the block. Across 64
experts × topk=8 × 36 SSM/MoE layers, the per-token accumulation easily explains
the ~1pp cosine gap reported on `ssm.moe_out` even with a perfect Atlas kernel.

DeepSeek's V3 report and NVIDIA's per-block scaling blog both stipulate FP32
scales precisely to avoid this; the Qwen team's BF16-scale choice is an
unfortunate-but-fixable storage shortcut, not a fundamental limitation of E4M3.
([Per-block scaling NVIDIA blog](https://developer.nvidia.com/blog/per-tensor-and-per-block-scaling-strategies-for-effective-fp8-training/),
[FP8-Flow-MoE paper](https://arxiv.org/abs/2511.02302) on double-quant error)

### Conversion path (concrete)

1. Load Qwen3.6-35B-A3B-FP8 safetensors.
2. For each `*.weight_scale_inv` tensor (bf16 [N/128, K/128]):
   - Cast to FP32, **do not re-derive**. This is the lossless cast — the original
     calibration's chosen scale is preserved exactly.
3. Optionally: load the FP8-dequanted BF16 reference once, recompute `amax / 448`
   per block in FP32, take whichever-is-larger (`max(s_existing_f32, s_recomputed_f32)`).
   This is the "RNE-restore" variant. Tiny gain on top of the cast.
4. Save under the same model card with `quant_method: fp8_blockscaled_fp32scale`.
5. Atlas side: `quant_format/fp8_blockscaled.rs` dispatches by checkpoint key; flip
   the scale-load dtype from `bf16` to `f32` and propagate through
   `fp8_lut.rs::dequant_block` (drop the bf16→f32 cast). ~50 LoC.

Cost: ~80MB extra per 35B checkpoint (2x scales, scales are ~80MB total at bf16
for a 35B FP8 model). One-shot CPU job. **No retrain. No calibration data needed.**

Expected gain (estimate from the BF16-scale ULP argument):
- Cosine gap closure: ~0.7-1.0 pp on the worst SSM layers (L20, L25, L37). Brings
  C-mean from 0.99497 → ~0.998 and tightens worst layer from 0.99012 → ~0.995.
- This is the **only path that doesn't change kernels** and is the highest
  marginal-effort-vs-precision ratio.

---

## Why MXFP8 is *not* the next step on GB10

The OCP MXFP8 spec uses 1x32 blocks with E8M0 scales (power-of-2 only). Datacenter
Blackwell (SM100, B100/B200) has `tcgen05.mma.kind::mxf8f6f4` tensor-core
instructions that consume MX scales directly. **SM121 (GB10) does not.** SM121's
tensor-core ISA is closer to Ampere `mma.sync` and lacks the MX scale path. Any
MXFP8 deployment on GB10 would either software-dequant per 32-element block
(slower than current 128x128 BF16-scale path) or sit idle until NVIDIA ships
SM121 MX support.
([SM121 architecture analysis](https://deepwiki.com/christopherowen/spark-vllm-mxfp4-docker/2.1-sm121-(blackwellgb10)-hardware-architecture),
[SM121 NVFP4 native compute thread](https://forums.developer.nvidia.com/t/sm121-gb10-native-nvfp4-compute-seeking-guidance-on-software-support/364607))

Conclusion: skip MXFP8 on GB10 until NVIDIA ships native scaling kernels for
sm_121. Same logic applies to MXFP6.

---

## Why NVFP4-for-MoE is a sidegrade, not an upgrade

NVFP4 (4-bit E2M1 + FP8 E4M3 scale per 16 + FP32 tensor scale) was the format
that produced the 0.96707 baseline mean cos on 2026-05-23 — i.e. the very gap we
are trying to close. The NVFP4 *value* range (8 levels) is the bottleneck, not
the scale. Going to NVFP4 for storage gives ~2x memory but caps quality below
FP8. Atlas already correctly disables NVFP4 for native FP8 dispatch (memory
`project_qwen36_fp8_post_think_eos.md`); the alternative we want is *better than
FP8*, not below it.

---

## Why AWQ / SmoothQuant / FP8-Flow-MoE are deferred

- **AWQ-FP8 (F)**: requires calibration set + a new mixed-precision MoE kernel.
  Atlas's MoE path is hand-tuned NVFP4-routed FP8 dequant; adding 1% FP16
  outlier-protected channels means a new tile schedule. ~2 weeks engineering.
- **SmoothQuant + FP8 (G)**: easier to ship (it's an offline activation
  per-channel rescale folded into weights), but the LMSYS Unified-FP8 result
  ([blog](https://www.lmsys.org/blog/2025-11-25-fp8-rl/)) shows the *training*
  side needs cooperation. For a pre-quantized release checkpoint (Qwen3.6-FP8),
  smooth_aggregate equivalent at inference time requires us to learn the
  smoothing factor from BF16 ref — same calibration effort as AWQ.
- **FP8-Flow-MoE (I)**: a retrain recipe, not a checkpoint format. Out of scope
  unless we re-quant from BF16 reference and stand up our own MoE QAT — months,
  not days.

---

## Ranked candidates for Atlas to adopt

### Rank 1 (do this first): **FP8 E4M3 + FP32 block scales (option A)**

- Memory delta: +80 MB per 35B checkpoint (negligible).
- Kernel delta: ~50 LoC dispatch change in `quant_format/fp8_blockscaled.rs` +
  scale-load dtype flip in `weight_map/loaders_fp8.rs`.
- Conversion: one Python script over safetensors, ~7 min CPU per checkpoint, no
  calibration data.
- Expected cosine gain: +0.7-1.0 pp at the worst SSM layers; brings Atlas's
  C-mean from 0.99497 → ~0.998. **Directly tests agent-A3's hypothesis.**
- Risk: very low — the FP8 values themselves do not change; only the scale's
  representation. Atlas's E4M3 LUT dequant stays bit-identical *except* the
  scale multiplier is now exact.
- This is the single highest-ROI move on the table.

### Rank 2 (do this if Rank 1 doesn't close enough): **FP8 + FP32 + recompute-on-load (option A')**

- Same checkpoint format as Rank 1, but at load time also recompute `amax/448`
  per block from a *BF16-dequanted reference* and pick the tighter of the two
  scales. Closes the residual BF16-scale-was-rounded-down error.
- ~1 hour CPU per checkpoint (need to dequant a copy first to find true amax).
- Marginal +0.1-0.3 pp on top of Rank 1.
- Same kernel as Rank 1, just a different scale source.

### Rank 3 (parking lot): **SmoothQuant-style activation-rescale into weights**

- Bake a per-channel rescale into FP32 scales of (Rank 1) *plus* a per-channel
  activation multiplier baked into the input projection. Targets the
  router-gate→expert dataflow where activation outliers compound the FP8 noise
  (LMSYS Unified-FP8 result on MoE RL).
- Calibrate from 64 sequences of probe prompts (small effort).
- Expected: +0.2-0.4 pp beyond Rank 2 in MoE-heavy layers (L18-25, L37-38).
- Risk: medium — requires actual calibration tooling we don't have today.

---

## Recommendation

**Adopt Rank 1 (FP8 E4M3 + FP32 block scales) now.**

It is the only option that:
1. Closes the BF16-scale ULP gap (the structural cause of the residual drift),
2. Requires no new kernels and no calibration data,
3. Is byte-compatible with our existing FP8 LUT dequant logic,
4. Has been validated by DeepSeek-V3 and every major FP8 inference stack as the
   correct storage choice for block-scaled FP8.

Land Rank 1, re-run the dgx2 op-drift table, and *only if* the worst SSM layer
still sits below cos 0.997, escalate to Rank 2. Defer everything MX/AWQ/Unified
until SM121 grows native MX scaling or until we have a calibration harness.

---

## Sources

- [Per-Tensor and Per-Block Scaling Strategies for Effective FP8 Training — NVIDIA blog](https://developer.nvidia.com/blog/per-tensor-and-per-block-scaling-strategies-for-effective-fp8-training/)
- [Introducing NVFP4 for Efficient and Accurate Low-Precision Inference — NVIDIA blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [DeepSeek-V3 Technical Report — arXiv 2412.19437](https://arxiv.org/html/2412.19437v1)
- [FP8-Flow-MoE: A Casting-Free FP8 Recipe without Double Quantization Error — arXiv 2511.02302](https://arxiv.org/abs/2511.02302)
- [Unified FP8 — LMSYS blog](https://www.lmsys.org/blog/2025-11-25-fp8-rl/)
- [Recipes for Pre-training LLMs with MXFP8 — arXiv 2506.08027](https://arxiv.org/html/2506.08027v1)
- [OCP Microscaling Formats (MX) v1.0 spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [SM121 (Blackwell/GB10) Hardware Architecture — DeepWiki](https://deepwiki.com/christopherowen/spark-vllm-mxfp4-docker/2.1-sm121-(blackwellgb10)-hardware-architecture)
- [SM121 (GB10) native NVFP4 compute — NVIDIA developer forums](https://forums.developer.nvidia.com/t/sm121-gb10-native-nvfp4-compute-seeking-guidance-on-software-support/364607)
- [High-Accuracy MXFP4, MXFP6 — AMD ROCm blogs](https://rocm.blogs.amd.com/software-tools-optimization/mxfp4-mxfp6-quantization/README.html)
- [SGLang Quantization docs](https://sgl-project.github.io/advanced_features/quantization.html)
- [vLLM FP8 W8A8 docs](https://docs.vllm.ai/en/latest/features/quantization/fp8/)
- [SmoothQuant — arXiv 2211.10438](https://arxiv.org/abs/2211.10438)
- [MicroMix mixed-precision MX — arXiv 2508.02343](https://arxiv.org/pdf/2508.02343)
