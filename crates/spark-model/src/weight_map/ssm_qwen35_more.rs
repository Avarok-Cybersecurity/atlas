// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `weight_map.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::*;

/// Load MoE weights for MiniMax M2 family (M2.1, M2.7).
///
/// Differences from `load_moe_no_shared` (Qwen3-VL):
///   * Prefix `{layer}.block_sparse_moe` instead of `{layer}.mlp`.
///   * Expert matrices use the Mixtral convention `w1/w2/w3` (NOT
///     `gate_proj/up_proj/down_proj`): w1 = gate_proj, w2 = down_proj,
///     w3 = up_proj. This matches the MiniMax checkpoint layout verified
///     against yujiepan/minimax-m2.7-tiny-random.
///   * Loads `e_score_correction_bias: [num_experts]` — the DeepSeek-V3
///     / MiniMax-M2 loss-free-balancing bias — into
///     `MoeWeights.correction_bias`. Consumed by `moe_topk_sigmoid` when
///     the MoE layer is built via the sigmoid-routing constructor.
///
/// Shared expert handling matches `load_moe_no_shared`: MiniMax has
/// `shared_intermediate_size: 0`, so we allocate zero-filled dummy
/// shared expert buffers (the fused NVFP4 kernels always launch a
/// shared-expert block; zero weights with `weight_scale_2 = 0.0` make
/// that block contribute nothing).
#[allow(clippy::too_many_arguments)]
pub(crate) fn load_moe_minimax(
    store: &WeightStore,
    layer_prefix: &str,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    config: &atlas_core::config::ModelConfig,
    variant: Nvfp4Variant,
    absmax_k: spark_runtime::gpu::KernelHandle,
    quantize_k: spark_runtime::gpu::KernelHandle,
    stream: u64,
) -> Result<MoeWeights> {
    let p = format!("{layer_prefix}.block_sparse_moe");

    // Gate weight ships as F32 in the real MiniMax 229B checkpoint and
    // BF16 in tiny-random. Atlas's gate GEMM wants BF16 input, so
    // convert when F32.
    let gate = dense_f32_safe(store, &format!("{p}.gate.weight"), gpu)?;
    // Correction bias: F32 in the real 229B checkpoint (shape [256]),
    // BF16 on the tiny-random bring-up variant. The sigmoid kernel
    // reads `bias` as `const float*`, so F32 on-disk is the expected
    // canonical dtype. `dense` returns the raw pointer — tiny-random
    // BF16 bytes would be mis-interpreted here.
    //
    // M2.7-NVFP4 (lukealonso/MiniMax-M2.7-NVFP4) ships
    // `e_score_correction_bias` as BF16 [256] (not F32 like the
    // original 229B). Reading 256 BF16 elements as 256 F32 would
    // alias half the data and produce wrong routing scores. Convert
    // BF16 → F32 in that case via the existing helper.
    let bias_key = format!("{p}.e_score_correction_bias");
    let bias_t = store.get(&bias_key)?;
    let correction_bias = if bias_t.dtype == spark_runtime::weights::WeightDtype::BF16 {
        let n = bias_t.num_elements();
        let mut bf16_buf = vec![0u8; n * 2];
        gpu.copy_d2h(bias_t.ptr, &mut bf16_buf)?;
        // BF16 → F32: append 2 zero LSB bytes, treat upper as BF16 mantissa
        let mut f32_buf = vec![0u8; n * 4];
        for i in 0..n {
            // BF16 stored little-endian: low byte = mantissa LSB, high byte = sign+exp+mantissa MSB.
            // F32 layout is BF16 in upper 16 bits, lower 16 bits zero.
            f32_buf[i * 4 + 2] = bf16_buf[i * 2];
            f32_buf[i * 4 + 3] = bf16_buf[i * 2 + 1];
        }
        let ptr = gpu.alloc(f32_buf.len())?;
        gpu.copy_h2d(&f32_buf, ptr)?;
        DenseWeight { weight: ptr }
    } else {
        dense(store, &bias_key)?
    };

    // Dummy shared expert buffers — identical shape logic to
    // load_moe_no_shared (see that function for the rationale).
    //
    // Use ceil-division + a 1-byte minimum on each allocation to stay
    // resilient against tiny-random configs where h or inter < group_size
    // (e.g. yujiepan/minimax-m2.7-tiny-random has h=8, inter=32, and 8/16
    // floor-divides to 0 bytes → cuMemAlloc_v2 rejects the request). For
    // the real 229B (h=3072, inter=1536) the ceil path produces the same
    // byte count as floor.
    let h = config.hidden_size;
    let inter = config.moe_intermediate_size;
    let group_size = 16usize;
    let ceil_div = |n: usize, d: usize| -> usize { n.div_ceil(d) };
    let gu_packed_bytes = (inter * h).div_ceil(2).max(1);
    let gu_scale_bytes = (inter * ceil_div(h, group_size)).max(1);
    let d_packed_bytes = (h * inter).div_ceil(2).max(1);
    let d_scale_bytes = (h * ceil_div(inter, group_size)).max(1);

    let alloc_zero = |size: usize| -> Result<DevicePtr> {
        let sz = size.max(1);
        let ptr = gpu.alloc(sz)?;
        gpu.memset(ptr, 0, sz)?;
        Ok(ptr)
    };
    let mk_zero_quant = |packed_sz: usize, scale_sz: usize| -> Result<QuantizedWeight> {
        Ok(QuantizedWeight {
            weight: alloc_zero(packed_sz)?,
            weight_scale: alloc_zero(scale_sz)?,
            weight_scale_2: 0.0,
            input_scale: DevicePtr::NULL,
        })
    };
    let shared_expert = ExpertWeight {
        gate_proj: mk_zero_quant(gu_packed_bytes, gu_scale_bytes)?,
        up_proj: mk_zero_quant(gu_packed_bytes, gu_scale_bytes)?,
        down_proj: mk_zero_quant(d_packed_bytes, d_scale_bytes)?,
    };
    let shared_expert_gate = DenseWeight {
        weight: alloc_zero(h * 2)?,
    };

    // Per-expert w1/w2/w3 → gate/down/up in Atlas naming. Dispatch by
    // variant + what's actually on disk:
    //   * Pre-quantized NVFP4 (weight_packed + weight_scale) → zero-copy
    //     via `quantized_auto`.
    //   * FP8 block-scaled (weight + weight_scale_inv, DeepSeek-V3 /
    //     MiniMax M2 convention) → dequant via
    //     `dequant_fp8_blockscaled_to_bf16`, runtime-quantize to NVFP4.
    //     Note this is distinct from Nemotron's scalar-scale FP8
    //     (`dequant_fp8_to_bf16` + `weight_scale`) — different on-disk
    //     layout requires a different dequant kernel.
    //   * BF16 dense fallback (e.g. tiny-random bring-up checkpoint) →
    //     runtime-quantize via `quantize_to_nvfp4`.
    //
    // Shape (n, k) = rows × cols of the output matrix on disk. w1 and w3
    // are [moe_intermediate, hidden]; w2 is [hidden, moe_intermediate].
    let quant = |ep: &str, n: usize, k: usize| -> Result<QuantizedWeight> {
        // Pre-quantized NVFP4 fast path:
        //   * CompressedTensors variant (Sehyo/RedHat-style): tensor at
        //     `.weight_packed` plus `.weight_global_scale` companion.
        //   * Standard variant (lukealonso/MiniMax-M2.7-NVFP4 style): tensor
        //     at `.weight` (uint8 packed E2M1) plus `.weight_scale` (FP8
        //     per-group) and `.weight_scale_2` (F32 scalar) companions.
        // Both flow through `quantized_auto`, which dispatches by variant
        // (`quantized()` for Standard, `quantized_v2()` for CompressedTensors).
        let is_compressed = store.contains(&format!("{ep}.weight_packed"));
        let is_standard_nvfp4 = matches!(variant, Nvfp4Variant::Standard)
            && store.contains(&format!("{ep}.weight_scale_2"));
        if (matches!(
            variant,
            Nvfp4Variant::Standard | Nvfp4Variant::CompressedTensors
        ) && is_compressed)
            || is_standard_nvfp4
        {
            return quantized_auto(store, ep, gpu, variant);
        }
        // MiniMax M2 ships FP8 block-scaled (`weight` FP8E4M3 + `weight_scale_inv`
        // F32). We dequant FP8→BF16 (fresh alloc), runtime-quantize BF16→NVFP4
        // (fresh alloc), then free BOTH the transient BF16 AND the original
        // FP8 + scale_inv tensors on GPU. Without freeing the FP8 source,
        // the 109 GB FP8 store per EP=2 rank stays resident, OOMing model
        // construction when ~55 GB more NVFP4 weights are added.
        let wkey = format!("{ep}.weight");
        let scale_key = format!("{ep}.weight_scale_inv");
        let (src_ptr, src_is_fp8) = {
            let t = store.get(&wkey)?;
            (
                t.ptr,
                t.dtype == spark_runtime::weights::WeightDtype::FP8E4M3,
            )
        };
        let scale_ptr = if store.contains(&scale_key) {
            Some(store.get(&scale_key)?.ptr)
        } else {
            None
        };
        let (dense_w, owned_bf16, need_free_src) = if scale_ptr.is_some() {
            (dequant_fp8_blockscaled_to_bf16(store, ep, gpu)?, true, true)
        } else {
            (dense_auto(store, &wkey, gpu)?, src_is_fp8, src_is_fp8)
        };
        let nvfp4 = quantize_to_nvfp4(&dense_w, n, k, gpu, absmax_k, quantize_k, stream)?;
        if owned_bf16 {
            // Free the transient BF16 dequant buffer.
            gpu.free(dense_w.weight)?;
        }
        if need_free_src {
            // Free FP8 source and companion scale_inv — no longer needed
            // once NVFP4 has been produced. WeightStore retains stale
            // pointers; nothing reads them again.
            gpu.free(src_ptr)?;
            if let Some(sp) = scale_ptr {
                gpu.free(sp)?;
            }
        }
        Ok(nvfp4)
    };
    let inter_moe = config.moe_intermediate_size;
    let mut experts = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        if config.is_local_expert(e) {
            experts.push(ExpertWeight {
                gate_proj: quant(&format!("{p}.experts.{e}.w1"), inter_moe, h)?,
                down_proj: quant(&format!("{p}.experts.{e}.w2"), h, inter_moe)?,
                up_proj: quant(&format!("{p}.experts.{e}.w3"), inter_moe, h)?,
            });
        } else {
            experts.push(ExpertWeight::null());
        }
    }

    Ok(MoeWeights {
        gate,
        shared_expert,
        shared_expert_gate,
        experts,
        router_pre_norm: None,
        correction_bias: Some(correction_bias),
    })
}

/// Load MoE weights for Gemma-4 26B-A4B.
///
/// Differences from Qwen3.5:
/// - Router at `{lp}.router.proj.weight` (not `{lp}.mlp.gate.weight`)
/// - Router `scale` `[H]` and `per_expert_scale` `[E]` fused into gate at load time
/// - Experts at `{lp}.moe.experts.{e}` (not `{lp}.mlp.experts.{e}`)
/// - No shared expert (dense MLP is loaded separately as primary FFN)
pub(crate) fn load_moe_gemma4(
    store: &WeightStore,
    layer_prefix: &str,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    config: &atlas_core::config::ModelConfig,
    variant: Nvfp4Variant,
    qctx: QuantizeCtx,
) -> Result<MoeWeights> {
    let h = config.hidden_size;
    let inter = config.moe_intermediate_size;

    // ── Router gate with scale fusion ──
    // Load gate [E, H], input_scale [H], per_expert_scale [E] — all BF16 on GPU.
    // Fuse on CPU: gate[e,h] *= input_scale[h] * per_expert_scale[e]
    let gate_key = format!("{layer_prefix}.router.proj.weight");
    let scale_key = format!("{layer_prefix}.router.scale");
    let per_exp_key = format!("{layer_prefix}.router.per_expert_scale");

    let gate_wt = store.get(&gate_key)?;
    let gate_bytes = num_experts * h * 2; // BF16
    let mut gate_buf = vec![0u8; gate_bytes];
    gpu.copy_d2h(gate_wt.ptr, &mut gate_buf)?;

    // Router pre-normalization weight (new, correct path).
    // HF Gemma4TextRouter:
    //   x_normed = pure_rms_norm(x)                       [no learned weight]
    //   x_scaled = x_normed * scale * hidden_size^(-0.5)
    //   logits = proj @ x_scaled
    //
    // Prior versions fused `scale * root_size` into the gate weight but
    // SKIPPED the pure rms_norm. With the post-attention residual having
    // L2 norm ~20-30 on this hidden dim, the un-normalized input saturated
    // the gate softmax → near-top-1 routing → MoE collapse on tool-call
    // prompts. Correct fix: store `scale * root_size` as a DenseWeight and
    // call `rms_norm(input, weight, normed)` before the gate GEMV. The
    // existing rms_norm kernel applies `output[h] = (x[h] / rms) * w[h]`
    // which is exactly HF's `x_normed * scale * root_size` in one pass.
    // The gate weight stays as the raw unfused `router.proj.weight`.
    let router_pre_norm = if store.contains(&scale_key) {
        let scale_wt = store.get(&scale_key)?;
        let mut scale_buf = vec![0u8; h * 2];
        gpu.copy_d2h(scale_wt.ptr, &mut scale_buf)?;
        let scalar_root = 1.0f32 / (h as f32).sqrt();
        for dim in 0..h {
            let bits = u16::from_le_bytes([scale_buf[dim * 2], scale_buf[dim * 2 + 1]]);
            let f = f32::from_bits((bits as u32) << 16) * scalar_root;
            let out = (f.to_bits() >> 16) as u16;
            scale_buf[dim * 2] = out as u8;
            scale_buf[dim * 2 + 1] = (out >> 8) as u8;
        }
        let pre_norm_ptr = gpu.alloc(h * 2)?;
        gpu.copy_h2d(&scale_buf, pre_norm_ptr)?;
        tracing::info!("Gemma-4 MoE: router pre-norm weight = scale * hidden_size^(-0.5)");
        Some(DenseWeight {
            weight: pre_norm_ptr,
        })
    } else {
        None
    };

    // Upload unfused gate weight to GPU
    let gate_ptr = gpu.alloc(gate_bytes)?;
    gpu.copy_h2d(&gate_buf, gate_ptr)?;
    let gate = DenseWeight { weight: gate_ptr };

    // ── Dummy shared expert (zero-filled) ──
    let group_size = 16usize;
    let gu_packed = inter * h / 2;
    let gu_scale = inter * (h / group_size);
    let d_packed = h * inter / 2;
    let d_scale = h * (inter / group_size);

    let alloc_zero = |size: usize| -> Result<DevicePtr> {
        let ptr = gpu.alloc(size)?;
        gpu.memset(ptr, 0, size)?;
        Ok(ptr)
    };
    let mk_zero = |p: usize, s: usize| -> Result<QuantizedWeight> {
        Ok(QuantizedWeight {
            weight: alloc_zero(p)?,
            weight_scale: alloc_zero(s)?,
            weight_scale_2: 0.0,
            input_scale: DevicePtr::NULL,
        })
    };
    let shared_expert = ExpertWeight {
        gate_proj: mk_zero(gu_packed, gu_scale)?,
        up_proj: mk_zero(gu_packed, gu_scale)?,
        down_proj: mk_zero(d_packed, d_scale)?,
    };
    let shared_expert_gate = DenseWeight {
        weight: alloc_zero(h * 2)?,
    };

    // ── Load per_expert_scale for absorption into down_proj ──
    // HF Gemma4TextRouter applies per_expert_scale to routing WEIGHTS after topk.
    // Absorb into expert down_proj.weight_scale_2: output *= per_expert_scale[e].
    let per_expert_scales: Vec<f32> = if store.contains(&per_exp_key) {
        let per_exp_wt = store.get(&per_exp_key)?;
        let mut buf = vec![0u8; num_experts * 2];
        gpu.copy_d2h(per_exp_wt.ptr, &mut buf)?;
        (0..num_experts)
            .map(|e| {
                let bits = u16::from_le_bytes([buf[e * 2], buf[e * 2 + 1]]);
                f32::from_bits((bits as u32) << 16)
            })
            .collect()
    } else {
        vec![1.0f32; num_experts]
    };

    // ── Load 128 routed experts ──
    let mut experts = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let ep = format!("{layer_prefix}.moe.experts.{e}");
        if config.is_local_expert(e) {
            let mut down = quantized_any(
                store,
                &format!("{ep}.down_proj"),
                h,
                inter,
                gpu,
                variant,
                qctx,
            )?;
            // Absorb per_expert_scale into down_proj: scales all output elements.
            // NVFP4 dequant: val = E2M1 * block_scale * weight_scale_2.
            // Multiplying weight_scale_2 by per_expert_scale[e] effectively scales output.
            down.weight_scale_2 *= per_expert_scales[e];
            experts.push(ExpertWeight {
                gate_proj: quantized_any(
                    store,
                    &format!("{ep}.gate_proj"),
                    inter,
                    h,
                    gpu,
                    variant,
                    qctx,
                )?,
                up_proj: quantized_any(
                    store,
                    &format!("{ep}.up_proj"),
                    inter,
                    h,
                    gpu,
                    variant,
                    qctx,
                )?,
                down_proj: down,
            });
        } else {
            experts.push(ExpertWeight::null());
        }
    }

    Ok(MoeWeights {
        gate,
        shared_expert,
        shared_expert_gate,
        experts,
        router_pre_norm,
        correction_bias: None,
    })
}
