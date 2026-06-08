# Arch-Phase #06 — Quantization Metadata Plumbing: vLLM vs Atlas

Compares how a single Qwen3.6-A3B FP8 checkpoint is parsed, dequantized,
and exposed to the GEMM kernels in the two stacks. The one-shot model-load
phase governs the in-memory scale layout that every subsequent forward
pass consumes — different storage dtype, granularity, or fusion point
propagates directly into per-token numerics.

## 1. Weight-loading sequence

| Step | vLLM | Atlas |
|---|---|---|
| Config dispatch | `Fp8Config.from_config()` reads `quantization_config.quant_method` → instantiates `Fp8LinearMethod` / `Fp8MoEMethod` (`vllm/model_executor/layers/quantization/fp8.py:242-257`) | `detect_quant_format()` reads `config.quantization_config` (`crates/spark-model/src/quant_format/mod.rs:95-175`) → returns `Box<dyn QuantFormat>` (`Fp8BlockScaledFormat`, `ModeloptFormat`, `CompressedTensorsFormat`); falls back to **tensor-name sniffing** when the config block is silent (`mod.rs:140-174`) |
| Empty parameters allocated | `create_weights()` registers `weight`, `weight_scale_inv` (block) or `weight_scale` (per-tensor), and optional `input_scale` on the `nn.Module` (`fp8.py:402-487`) | No `nn.Module` analogue — `load_fp8_block_scaled_as_fp8weight()` allocates raw `DevicePtr`s and packs them into a `Fp8Weight { weight, row_scale, n, k, scale_format }` struct (`crates/spark-model/src/weight_map/loaders_fp8.rs:23-66`) |
| Safetensors → host | `weight_utils.py::safetensors_weights_iterator` yields each `(name, tensor)` for `model.load_weights(weights)` (`weight_utils.py`) | `WeightStore` mmaps shards; `store.get(name)` returns a `DevicePtr` pre-uploaded by `spark_runtime::weights` (zero-copy device aliasing) |
| Host → GPU | Per-shard `weight_loader(param, tensor, shard_id)`; `Parameter` machinery (`ChannelQuantScaleParameter`, `BlockQuantScaleParameter`, `PerTensorScaleParameter`) handles TP slicing (`fp8.py:1059-1124` in `qwen3_next.py::load_weights`) | TP slicing in `tp_shard/quant_shard.rs`; `WeightStore` upload + per-loader explicit `gpu.copy_h2d` |
| Post-load transform | **`process_weights_after_loading(layer)`** runs once per layer after all shards are present (`fp8.py:489-545`, `870-1010` for MoE) — does requantization, weight-scale `T()`, normalize_e4m3fn_to_e4m3fnuz on ROCm, marlin packing, etc. | **No equivalent hook**. Per-loader functions (`load_fp8_block_scaled_as_fp8weight`, `quantize_to_nvfp4`, `build_linear_attention_fp8`) do their format-specific transforms inline at load time. SSM-arm concat is open-coded (`weight_loader/qwen35/load_layers/linear_attn_arms.rs:41-235`) |
| Format selection per-module | `is_layer_skipped(prefix, ignored_layers)` checks `quantization_config.ignored_layers` → returns `UnquantizedLinearMethod` for ignored modules (`fp8.py:296-302`) | `QuantFormat::variant_for(module_path)` returns `Nvfp4Variant::Bf16Raw` when path matches an ignore glob (`mod.rs:73-80`); pattern matcher `module_matches_pattern` (`mod.rs:185-213`) |

## 2. Scale storage — dtype × shape × layout

| Path | vLLM dtype/shape | Atlas dtype/shape | File:line |
|---|---|---|---|
| Per-tensor FP8 scale | `torch.float32` scalar, `PerTensorScaleParameter` of shape `[num_logical_shards]` (`fp8_utils.py:1091-1095`) | `f32` baked into kernel constant `448.0`; per-row scale: `DevicePtr → [N] f32` from `quantize_bf16_to_fp8` (`kernels/gb10/common/dense_gemv_fp8w.cu:36-79`); tagged `WeightQuantFormat::Fp8PerRow` (`quantized.rs:36`) | vllm `fp8_utils.py:1085`; atlas `quantized.rs:36-37,213-226` |
| Block-scaled FP8 (`weight_scale_inv`) | `torch.float32` of shape `[N//128, K//128]`, `BlockQuantScaleParameter` (`fp8_utils.py:1077-1090`) | **`bfloat16`** of shape `[N/128, K/128]`, `DevicePtr` stored in `Fp8Weight.row_scale`; tagged `WeightQuantFormat::Fp8BlockScaled` (`quantized.rs:42`) | vllm `fp8_utils.py:1085` (`dtype=torch.float32`); atlas `loaders_fp8.rs:43-65` + `quantized.rs:42` |
| Block size | `weight_block_size = [128,128]` from config, `validate_fp8_block_shape` enforces divisibility (`fp8.py:422-432`) | `const BS: usize = 128;` hard-coded in `linear_attn_arms.rs:97`, `moe_fp8_grouped_gemm.cu:24` (`#define FP8_BLOCK 128`) | vllm `fp8.py:202-211`; atlas `linear_attn_arms.rs:97-109` |
| NVFP4 per-group scale | `torch.uint8` E4M3 of shape `[N, K//16]` + per-tensor `weight_scale_2: f32`, modelopt or compressed-tensors variant (`modelopt.py:929-1020`) | `DevicePtr` packed E2M1 [N, K/2] + FP8 group scales `weight_scale: DevicePtr` + **`f32` scalar `weight_scale_2`** extracted by D2H copy at load (`quantized.rs:69-80`) | vllm `modelopt.py:923-1020`; atlas `quantized.rs:69-80`, loaders_fp8.rs:104-109 |
| NVFP4 activation scale | `input_scale: PerTensorScaleParameter f32`, max-reduced after load (`modelopt.py:307,318`) | `input_scale: DevicePtr` to f32 scalar **for the NVFP4 path only** (`quantized.rs:79`); always **`DevicePtr::NULL` on the FP8 path** (`loaders_fp8.rs:143`, `quantized.rs:89`, `moe.rs:58`) | vllm `modelopt.py:309-318`; atlas `loaders_fp8.rs:143` |
| MoE expert scale (block) | `w13_weight_scale_inv: f32 [num_experts, 2*N//block_n, K//block_k]` (`fp8.py:807-827`) | `[num_experts] → [N/128, K/128] BF16` device-pointer table (`kernels/gb10/common/moe_fp8_grouped_gemm.cu:163`) | vllm `fp8.py:807-827`; atlas `moe_fp8_grouped_gemm.cu:163,187` |
| MoE expert scale (per-tensor) | `f32 [num_experts, 2]` for w13, `f32 [num_experts]` for w2 (`fp8.py:799-806`) | NVFP4 path: `QuantizedWeight.weight_scale_2: f32` scalar per expert (`quantized.rs:77`) | vllm `fp8.py:799-806`; atlas `quantized.rs:77` |

**Atlas-specific quirk** (`linear_attn_arms.rs:111`):
`let scale_row_bytes = scale_cols * 2;`
The `* 2` hard-codes BF16 width. If vLLM-parity FP32 scales are ever
introduced, this open-coded width and **every kernel that does
`__bfloat162float(block_scale[...])`** must change in lock-step.

## 3. Per-GEMM scale dispatch — at the moment of dequant

| Kernel | Path | Scale dtype loaded in-kernel | Multiply precision |
|---|---|---|---|
| vLLM `fused_moe_kernel` (Triton block-FP8) | `model_executor/layers/fused_moe/fused_moe.py:316-516` | `tl.load(a_scales_ptr).to(tl.float32)` and same for `b_scales` (line ~500) | `tl.dot(...) * a_s * b_s` in FP32 (line 510) |
| vLLM CUTLASS SM100 block-scaled grouped GEMM | `csrc/quantization/blockwise_scaled_group_mm_sm100.cu:253-265` | FP32 scale tiles via UMMA-FP8 epilogue | FP32 in pipeline |
| vLLM DeepGEMM `m_grouped_fp8_gemm_nt_contiguous` | Hopper WGMMA-FP8 + UE8M0 scales | UE8M0 (8-bit exponent) loaded as FP32 | promote-every-4-WGMMA, FP32 |
| Atlas `w8a16_gemv` (FP8 decode GEMV) | `kernels/gb10/common/w8a16_gemv.cu:108-201` | `__bfloat162float(block_scale[n_block * k_blocks + k_block])` (line 141) — **BF16 read → FP32 use** | FP32 acc: `acc += __bfloat162float(a) * (s_lut[byte] * scale)` (line 160-174) |
| Atlas `moe_fp8_grouped_gemm` (decode MoE) | `kernels/gb10/common/moe_fp8_grouped_gemm.cu:160-340` | `__bfloat162float(S_exp[n_block * k_blocks + k_block])` (line 277) — **BF16 read → FP32 use** | Two-level promotion: `inner_acc` FP32 → `outer_acc += inner_acc * scale` FP32 (line 282) |
| Atlas `fp8_gemm_t` / `fp8_gemm_n128` (prefill) | `kernels/gb10/qwen3.5-35b-a3b/nvfp4/w4a16_gemm.cu:371-480` | **No scale argument** (line 371-375 signature); `bf16x4_to_e4m3x4(...)` straight bit conversion on activations (line 188-204) | FP32 MMA acc, **scale implicit = 1.0** at output |
| Atlas `fp8_fp8_gemm_t` (FP8×FP8 prefill) | `kernels/gb10/qwen3.5-35b-a3b/nvfp4/w4a16_gemm.cu:560-680` | **No scale argument** in kernel signature | FP32 MMA acc, **scale implicit = 1.0** at output |
| Atlas `w4a16_gemm_t` (NVFP4 prefill) | `kernels/gb10/qwen3.5-35b-a3b/nvfp4/w4a16_gemm.cu:206-369` | Per-group FP8 scale × `f32 scale2` (loaded from `QuantizedWeight.weight_scale_2`) | FP32 |

The Atlas BF16 scale storage is upcast to FP32 inside every kernel
that *takes* a scale; the multiply itself is FP32. The lossy step
is the **on-disk → in-memory** quantization of the scale from FP32
(disk: `weight_scale_inv` is FP32 in the Qwen3.6 HF release) to BF16,
which is a one-shot mantissa truncation from 23 bits to 7 bits
(~0.4% relative error per scale entry).

## 4. Activation-side quantization (the smoking gun)

| Path | vLLM | Atlas |
|---|---|---|
| FP8 W8A8 per-tensor static | `QuantFP8(static=True, group_shape=GroupShape.PER_TENSOR)` reads `input_scale` parameter, calls `ops.scaled_fp8_quant(x, scale=input_scale)` → quantized FP8 tensor + scalar `x_scale` (`w8a8_utils.py:435-491`) | **Not implemented.** `bf16_to_fp8` activation kernel (`kernels/gb10/qwen3.5-35b-a3b/nvfp4/w4a16_gemm.cu:529-547`) calls `cvt.rn.satfinite.e4m3x2.f32` directly on each BF16 element with **no scale**. Per-tensor scale ≡ 1.0; activations > 448 saturate, small activations lose 3-bit-mantissa precision |
| FP8 W8A8 per-tensor dynamic | `QuantFP8(static=False, group_shape=GroupShape.PER_TENSOR)` runs absmax over the tile, computes `x_scale = amax/448` on the fly, passes both as kernel args (`input_quant_fp8.py`, `w8a8_utils.py:445-491`) | **Not implemented.** Same straight-cast as above |
| FP8 W8A8 per-token group dynamic (DeepSeek/Qwen3 block release) | `W8A8BlockFp8LinearOp.apply()` → `per_token_group_quant_fp8(x, group_size=128, column_major=True)` → returns `(q_input, input_scale[M, K//128])`, then `cutlass_scaled_mm(q_input, weight, x_scale, weight_scale, …)` (`fp8_utils.py:266-403`, `_per_token_group_quant_fp8` line 483-535) | **Not implemented.** Activations are kept in BF16 for the decode `w8a16_gemv` and `moe_fp8_grouped_gemm` (which is W8A16, not W8A8). For the prefill `fp8_fp8_gemm_t` path, `bf16_to_fp8(BF16, FP8, total_elements, stream)` does an unscaled cast — Atlas-side activation per-token scale is permanently 1.0 |
| Static activation scale ingest | `process_fp8_weight_tensor_strategy` returns `(weight, weight_scale, input_scale)` together; `input_scale.max()` reduces logical-shard scales (`fp8.py:518-526`) | `QuantizedWeight.input_scale: DevicePtr` field exists for **NVFP4 only** (`quantized.rs:79`), populated for modelopt checkpoints. For FP8 checkpoints, `loaders_fp8.rs:143` hard-sets `input_scale: DevicePtr::NULL`; the field is never read by any FP8 kernel |
| MoE expert activation scale | `w13_input_scale: PerTensorScaleParameter f32[num_experts]`, max-reduced in `process_weights_after_loading` (`fp8.py:977-996`) — passed into `flashinfer_cutlass_moe_fp8` / fused_moe_kernel as `a1_scale` / `a2_scale` (`modelopt.py:576-579`) | No analogue. Atlas FP8 MoE consumes BF16 activations directly (`moe_fp8_grouped_gemm.cu:161`); the inner GEMM is mathematically W8A16, not W8A8 |

**This is the largest architectural divergence in the load phase.** vLLM
runs every FP8 GEMM as W8A8 with a runtime-computed per-token-group
activation scale that compresses BF16 activations to FP8 with the
exact dynamic range each row needs. Atlas runs FP8 weights × BF16
activations for decode (no activation quant required) and for prefill
collapses BF16 → FP8 with a fixed (implicit 1.0) scale that clips
anything above ±448 and loses all dynamic-range adaptation.

## 5. MoE-specific scale layouts

| Aspect | vLLM | Atlas |
|---|---|---|
| Expert weight tensor | `w13_weight: torch.float8_e4m3fn [E, 2I, H]`, `w2_weight: torch.float8_e4m3fn [E, H, I]` (`fp8.py:770-794`) | Per-expert `DevicePtr` table `B_weight_ptrs: [num_experts] → [N, K] FP8` (`moe_fp8_grouped_gemm.cu:162`); each expert owns its own contiguous allocation |
| Expert scale tensor | Block-quant: `w13_weight_scale_inv: f32 [E, 2*(I/BS), H/BS]` (single tensor, fused across experts) (`fp8.py:807-816`) | Per-expert `DevicePtr` table `B_scale_ptrs: [num_experts] → [N/128, K/128] BF16` (`moe_fp8_grouped_gemm.cu:163`); each expert's scale lives in its own allocation |
| Routed-expert dispatch | Token-sorted gather inside fused MoE kernel; `expert_offsets`, `topk_weights`, `topk_ids` (`fused_moe.py`, `modelopt.py:570-606`) | Token-sorted dispatch matching DeepGEMM convention: `expert_offsets: [num_experts+1]`, `sorted_token_ids: [total_expanded]` (`moe_fp8_grouped_gemm.cu:165-166`) |
| Shared expert | Same code path as routed under FusedMoE | **Separate kernels**: `moe_shared_expert_fused_fp8.cu`, `moe_shared_expert_fused_fp8_batch{2,3}{,_t}.cu` (gate_up & silu_down GEMV variants). **Apply scale eagerly per element** (no K_PROMOTE) — distinct from routed kernel which uses two-level promotion. Math equivalent only because of FP32 ops (see Arch-Diff #09 §1) |
| `weight_scale_2` (NVFP4 per-tensor) | `f32` scalar per expert in `weight_scale_2` parameter | `f32` scalar per expert in `QuantizedWeight.weight_scale_2` (`quantized.rs:77`); extracted at load via D2H copy |
| Activation prepare-and-finalize | `FusedMoEPrepareAndFinalize` infrastructure (`fp8.py:23-26`) — pluggable per backend (Triton, DeepGEMM, CUTLASS, FlashInfer) | Hand-rolled: `moe_prefill.rs`, `moe_grouped_a.rs`, `fp8_moe.rs`, `fp8_moe_batch_{a,b}.rs` — each kernel does its own gather/scatter inline |

## 6. NVFP4 vs FP8 auto-selection (same on-disk checkpoint)

| Stack | Selection mechanism | What Qwen3.6-A3B-FP8 picks |
|---|---|---|
| vLLM | `Fp8Config.from_config()` keys off `quant_method=="fp8"` with `weight_block_size=[128,128]` ⇒ `Fp8LinearMethod(block_quant=True)` path; backend further selected by `get_fp8_moe_backend(block_quant=True)` (`fp8.py:124-181`) — picks `MARLIN` (SM<89), `CUTLASS_BLOCK_SCALED_GROUPED_GEMM` (SM100 block), `DEEPGEMM` (Hopper block), or `TRITON` (default) | Hopper: DeepGEMM; SM100: CUTLASS block-scaled grouped GEMM; SM120/121: **falls back to Marlin or Triton** (no native FP8 path) |
| Atlas | Two-stage: (1) `detect_quant_format()` reads `quantization_config.quant_method` (`mod.rs:95-138`) → `Fp8BlockScaledFormat`; (2) `detect_nvfp4_variant()` → `Nvfp4Variant::Fp8Dequanted` (`nvfp4_detect.rs:27-160`). Then `weight_map::quantized_from_fp8` does runtime FP8→BF16→NVFP4 re-quant for prefill, while `load_fp8_block_scaled_as_fp8weight` keeps the FP8 byte buffer for decode (`loaders_fp8.rs:23-66`, `linear_attn_arms.rs:41-235`) | Decode: native FP8 via `w8a16_gemv` (BF16 scale, BF16 act). Prefill: **dequant-once into BF16 host then re-quant to either single-scale FP8 (`bf16_to_fp8` per-element straight cast) or NVFP4 (`quantize_to_nvfp4`)** — depends on per-tensor heuristic. Not the same as vLLM block-scaled FP8 at prefill. |

**Selection is NOT identical for the same on-disk checkpoint.** vLLM
keeps the block-scaled FP8 layout end-to-end (W8A8 with dynamic
per-token-group activation scales). Atlas re-quantizes through BF16
and then either single-scale FP8 (loses block granularity) or NVFP4
(introduces 4-bit weight quant on top of FP8 weight quant, double
attenuation — see `linear_attn_arms.rs:299-307` comment about
"triple-quant FP8→BF16→NVFP4→BF16 chain attenuates direction in
the k-channel").

## 7. Compressed-tensors variant (per-token static activation FP8)

vLLM also supports the `compressed-tensors` static-act-scale variant
(Neural Magic `llm-compressor` output: `weight_packed` +
`weight_global_scale` + `input_global_scale`). The activation scale
is baked into the checkpoint and loaded as `input_scale` parameter
(see `vllm/model_executor/layers/quantization/compressed_tensors/...`).
Atlas's `compressed_tensors.rs` `QuantFormat` impl recognises the
serialization tag, but the actual scale plumbing into the kernel
(`input_scale` from `QuantizedWeight`) is **only consumed by the NVFP4
kernels** — the FP8 path doesn't read it (`quant_helpers.rs:228-256`).
For an MTP-style FP8 checkpoint shipping a static `input_scale`,
Atlas silently drops it.

## Atlas-only quant ops

1. **Runtime FP8 → NVFP4 re-quantization** at load time: `quantize_to_nvfp4(bf16_weight, n, k, gpu, absmax_k, quantize_k, stream)` (`loaders_fp8.rs:73-145`) — two-phase absmax → per-group E2M1 encode. vLLM never does this; it consumes the on-disk format directly.
2. **Two-level FP32 promotion in routed MoE FP8 GEMM** (`moe_fp8_grouped_gemm.cu:201-219`). vLLM's Triton block-FP8 kernel applies the scale inside the K-loop per tile dot. Atlas's `K_PROMOTE=64` is half the FP8 block size (128) — meaning each scale is applied twice per K-block. Documented mathematical identity (`Σ s·a·w ≡ s·Σ a·w` in FP32) but increases inner-acc magnitude → risks FP32 overflow that vLLM avoids by scale-per-tile.
3. **BF16 scale storage** for block-scaled FP8 (`quantized.rs:42`, `Fp8Weight.row_scale: DevicePtr` to BF16 buffer). Justification: halves scale-tensor bandwidth at decode. vLLM uses FP32 throughout.
4. **Hard-coded BS=128** without runtime read from `quantization_config.weight_block_size` (`linear_attn_arms.rs:97`, `moe_fp8_grouped_gemm.cu:24`). vLLM reads block size from config (`fp8.py:202-211`).
5. **Separate-kernel shared expert** (`moe_shared_expert_fused_fp8*.cu`, 6 variants). vLLM uses the same FusedMoE codepath.
6. **Mathematically equivalent but not byte-equivalent** RNE BF16 conversion (`atlas-quant/src/fp8.rs:99-113`): Atlas added round-to-nearest-even in Phase 2b (commit `atlas-gb10:fp8-dequant-rne`, 2026-05-24). Without this, FP8 dequant accumulated ~3% per-layer cosine loss vs PyTorch reference. **vLLM's PyTorch path uses RNE natively** — Atlas is only at parity with this patch active.

## vLLM-only quant ops

1. **Per-token-group activation FP8 quantization** (`per_token_group_quant_fp8` in `fp8_utils.py:483-690`). The activation tensor is partitioned into K/128-element groups, absmax computed per group per token, then quantized to FP8 with the per-group scale carried alongside. This is the W8A8 path that matches the W8A8 weight quantization granularity. Atlas has **no analogue**.
2. **`requantize_with_max_scale`** for per-tensor mode (`w8a8_utils.py`): when a fused module (e.g. qkv_proj) has multiple logical shards with different scales, vLLM dequantizes-and-requantizes through the max scale so a single GEMM can serve all shards. Atlas keeps the logical shards as separate `Fp8Weight`s and concats them later (`linear_attn_arms.rs:82-138`).
3. **`process_weights_after_loading` two-pass model construction**: weights load first, then the post-hook does dtype conversions, transposes, marlin packing, etc. Atlas does these transforms inline at load.
4. **FP8 E4M3FNUZ normalization** (`normalize_e4m3fn_to_e4m3fnuz`, ROCm-only). N/A on GB10 but illustrates the abstraction depth.
5. **Pluggable MoE backend selection** (`Fp8MoeBackend.{TRITON,FLASHINFER_TRTLLM,FLASHINFER_CUTLASS,DEEPGEMM,CUTLASS_BLOCK_SCALED_GROUPED_GEMM,MARLIN}`). Atlas has a single hand-rolled kernel family per shape.
6. **DeepGEMM E8M0 (UE8M0) requantization**: weight scales pre-quantized to 8-bit exponent format for register-pressure savings. Atlas keeps BF16 throughout.
7. **Native `__scaled_mm` activation × weight scale fusion**: vLLM's CUTLASS path passes both `scale_a` and `scale_b` as device-side FP32 scalars into a kernel that applies them in the FP32 accumulator before the output cast.

## Assessment — is the BF16-scale-storage / no-act-scale a likely contributor to the 70pp gap?

**Yes, both contribute, in this order of magnitude:**

1. **No activation scale (largest factor)**. The Qwen3.6-A3B-FP8
   checkpoint is calibrated assuming per-token-group dynamic activation
   quantization in the `[N/128, K/128]` granularity the weights were
   quantized with. Without it:
   - Activations clip at ±448 instead of being range-compressed
     dynamically. For the late layers where post-RMSNorm activations
     routinely exceed ±10–±50 in magnitude, this is a hard ceiling.
   - The K-loop inside MMA accumulates BF16-act × FP8-weight products
     without the per-tile scale compression that makes FP8 accurate
     in the first place. The whole point of W8A8 block-quant is that
     each `[128,128]` weight block's scale matches the K-block of the
     activation it multiplies; with BF16 acts and FP8 weights, that
     coupling is broken.
   - **Decode**: `w8a16_gemv` and `moe_fp8_grouped_gemm` are W8A16
     by design; less affected, but still missing the FP32-scale
     precision.
   - **Prefill**: `fp8_fp8_gemm_t` (line 560) is W8A8 with **scale=1.0**.
     This is the worst case — both inputs are FP8 with no scale at
     all, MMA result is just raw FP8×FP8.
   - Drift signature: long-context low-margin argmax flips (matches
     `project_qwen36_c1_diagnostic` finding: 23.7% positions gap<1.5
     in long ctx, 0% in short).
2. **BF16 scale storage (smaller factor)**.
   - Per-scale relative error: 2⁻⁷ ≈ 0.78% (BF16 mantissa = 7 bits
     vs FP32 mantissa = 23 bits).
   - Two-level FP32 promotion in `moe_fp8_grouped_gemm.cu:273-286`
     partially recovers this for routed MoE: the scale is applied
     once per K-block to the inner FP32 sum, so its error doesn't
     compound across K. But the **scale itself is still BF16-truncated
     on the way to GPU**.
   - vLLM session memory note (Phase 2c day-3 follow-up,
     `quantized.rs:20-27`) records the attempt to upcast to FP32
     was reverted because "multiple `Fp8Weight` constructors bypass
     the upcast helper" — confirmed by direct inspection of
     `loaders_fp8.rs:59-65`, `linear_attn_arms.rs:127-133`,
     `weight_loader/qwen3.rs:152`, `weight_loader/qwen35/load_layers.rs:324`,
     `ssm_qwen35.rs:248`. Six independent construction sites all
     write `Fp8BlockScaled` with BF16 scale — there's no single
     point where an FP32-scale upgrade can be inserted.
3. **NVFP4-via-FP8 prefill triple-quant** (`linear_attn_arms.rs:299-307`)
   — Atlas's own code comment documents the issue: FP8 → BF16
   (dequant) → NVFP4 (re-quant) → BF16 (dequant in kernel) loses
   information in two stages. vLLM keeps block-scaled FP8 throughout.

**The combined effect is consistent with the observed 70pp opencode
gap.** Both the act-scale absence and the BF16 weight-scale storage
hit late layers hardest — exactly where the C1 diagnostic placed the
worst per-layer cosine (L20+) and where the Phase ζ MoE smoking-gun
analysis found 8/8 → 3/8 expert-routing divergence (L0 → L38).

**Recommended fix order** (matching the existing `project_fp8_ceiling_conclusive`
conclusion):
1. **Add per-token-group activation FP8 quant** to all FP8 GEMM
   kernels (`fp8_gemm_t`, `fp8_fp8_gemm_t`, `moe_fp8_grouped_gemm`,
   `moe_shared_expert_fused_fp8*`). This is the W8A8 path the
   checkpoint expects. Cost: new `per_token_group_quant_fp8` CUDA
   kernel + 7 kernel signature changes + load-path plumbing of
   per-token `x_scale` device buffer.
2. **Upcast scale storage to FP32** at every `Fp8Weight` construction
   site (six sites listed above). Cost: 6 loaders + 5 kernel
   signature changes (load `float*` instead of `__nv_bfloat16*`).
   Saves ~0.4% per-scale; ~1-2pp on opencode estimate.
3. **Stop the triple-quant prefill detour**: keep block-scaled FP8
   for prefill GEMM, drop the BF16-intermediate-then-NVFP4 path.
   Requires a true block-scaled FP8 prefill kernel — currently
   the only prefill FP8 path is single-scale (`fp8_gemm_n128`).
