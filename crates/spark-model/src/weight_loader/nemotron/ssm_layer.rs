// SPDX-License-Identifier: AGPL-3.0-only

//! Nemotron-H Mamba-2 SSM layer construction: mixed-quant weight load plus the
//! FP8 / transposed-NVFP4 prefill weight copies. Split from `nemotron.rs`
//! (500-LoC cap).

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::weights::WeightStore;

use super::NemotronHWeightLoader;
use crate::layers::NemotronMamba2Layer;
use crate::weight_map::{
    DenseWeight, NemotronSsmQuant, dense, dequant_fp8_to_bf16_into, load_nemotron_ssm,
    quantize_to_nvfp4,
};

impl NemotronHWeightLoader {
    /// Build one Mamba-2 SSM layer (the `LayerType::LinearAttention` arm).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_ssm_layer(
        gpu: &dyn GpuBackend,
        store: &WeightStore,
        config: &ModelConfig,
        i: usize,
        h: usize,
        lp: &str,
        norm: DenseWeight,
        quantize_k: KernelHandle,
        absmax_k: KernelHandle,
        scratch: DevicePtr,
        stream: u64,
    ) -> Result<NemotronMamba2Layer> {
        // Mamba-2 SSM layer (mixed quant: NVFP4, FP8, or BF16)
        let (mut ssm, quant_kind) = load_nemotron_ssm(store, i, gpu, lp)?;
        tracing::info!(
            "L{i} SSM quant={quant_kind:?} in_proj_size={} d_inner={} h={h}",
            config.mamba2_in_proj_size(),
            config.mamba2_d_inner(),
        );
        // TODO: Fix 3 — FP8 direct load causes CUDA 700 (illegal address).
        // The WeightStore mmap pointers may be invalidated after loading.
        // For now, keep the double-quant path (FP8→BF16→NVFP4).
        if quant_kind != NemotronSsmQuant::Nvfp4 {
            let p = format!("{lp}.mixer");
            let in_proj_dense = if quant_kind == NemotronSsmQuant::Fp8 {
                dequant_fp8_to_bf16_into(store, &format!("{p}.in_proj"), gpu, scratch)?
            } else {
                dense(store, &format!("{p}.in_proj.weight"))?
            };
            ssm.in_proj = quantize_to_nvfp4(
                &in_proj_dense,
                config.mamba2_in_proj_size(),
                h,
                gpu,
                absmax_k,
                quantize_k,
                stream,
            )?;
            let out_fp8 = store.contains(&format!("{p}.out_proj.weight_scale"));
            let out_proj_dense = if out_fp8 {
                dequant_fp8_to_bf16_into(store, &format!("{p}.out_proj"), gpu, scratch)?
            } else {
                dense(store, &format!("{p}.out_proj.weight"))?
            };
            ssm.out_proj = quantize_to_nvfp4(
                &out_proj_dense,
                h,
                config.mamba2_d_inner(),
                gpu,
                absmax_k,
                quantize_k,
                stream,
            )?;
        }
        // Transposed NVFP4 copies of the two SSM projections. These
        // switch prefill from the base `w4a16_gemm` (M64/N64/K16, no
        // pipelining) to `w4a16_gemm_t` (N128/K32, FP8 MMA, 2-stage
        // cp.async) — see NemotronMamba2Layer::set_prefill_weights.
        // The 40 SSM layers are ~46% of prefill time on Puzzle, and
        // without this the fast kernel is compiled but unreachable.
        // Cost: ~2.1 GB extra weights. ATLAS_NO_SSM_PREFILL_T=1 keeps
        // the base GEMM (same-binary A/B + escape hatch).
        //
        // Two mutually exclusive prefill weight representations:
        //
        //   FP8  (default) — pre-dequantized E4M3 [N, K], consumed by
        //     `fp8_gemm_t`, which has NO dequant phase. The NVFP4
        //     path re-derives its B tile from FP4 on every K step of
        //     every M-block (cost N*K*(M/M_TILE), i.e. 8x over at 1k
        //     tokens); ablating that ALU alone cut a 1k prefill from
        //     557 ms to 424 ms, so it is worth removing outright.
        //     Cost: ~4.3 GB (vs ~2.1 GB for the transposed copies).
        //
        //   Transposed NVFP4 — `w4a16_gemm_t`/`_m128`. Kept as the
        //     escape hatch via ATLAS_NO_SSM_FP8_PREFILL=1.
        //
        // NVFP4 stays resident either way: decode uses w4a16_gemv.
        let fp8_prefill = std::env::var("ATLAS_NO_SSM_FP8_PREFILL").is_err();
        let prefill_t = !fp8_prefill && std::env::var("ATLAS_NO_SSM_PREFILL_T").is_err();
        let proj_t = if prefill_t {
            let in_t = ssm
                .in_proj
                .transpose_for_gemm(gpu, config.mamba2_in_proj_size(), h)?;
            let out_t = ssm
                .out_proj
                .transpose_for_gemm(gpu, h, config.mamba2_d_inner())?;
            Some((in_t, out_t))
        } else {
            None
        };
        let proj_fp8 = if fp8_prefill {
            let pdq_k = gpu.kernel("w4a16", "predequant_nvfp4_to_fp8")?;
            let in_fp8 = ssm.in_proj.predequant_to_fp8(
                gpu,
                pdq_k,
                config.mamba2_in_proj_size(),
                h,
                stream,
            )?;
            let out_fp8 =
                ssm.out_proj
                    .predequant_to_fp8(gpu, pdq_k, h, config.mamba2_d_inner(), stream)?;
            Some((in_fp8, out_fp8))
        } else {
            None
        };
        let mut layer = NemotronMamba2Layer::new(norm, ssm, config, gpu, i)?;
        if let Some((in_t, out_t)) = proj_t {
            layer.set_prefill_weights(Some(in_t), Some(out_t));
        }
        if let Some((in_fp8, out_fp8)) = proj_fp8 {
            layer.set_fp8_prefill_weights(in_fp8, out_fp8);
        }
        Ok(layer)
    }
}
