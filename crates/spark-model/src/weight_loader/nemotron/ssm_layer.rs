// SPDX-License-Identifier: AGPL-3.0-only

//! Nemotron-H Mamba-2 SSM layer construction: mixed-quant weight load plus the
//! FP8 / transposed-NVFP4 prefill weight copies. Split from `nemotron.rs`
//! (500-LoC cap).

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::weights::WeightStore;

use super::NemotronHWeightLoader;
use crate::layers::NemotronMamba2Layer;
use crate::weight_map::{
    DenseWeight, NemotronSsmQuant, dense, dequant_fp8_to_bf16_into,
    load_fp8_block_scaled_as_fp8weight, load_nemotron_ssm, quantize_to_nvfp4,
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
        let p = format!("{lp}.mixer");
        // ModelOpt ships this checkpoint's SSM projections as FP8 inside an
        // otherwise-NVFP4 repo. The legacy path dequantized them to BF16 and
        // re-quantized to NVFP4 under ONE global scale spanning the whole
        // [18048, 4096] tensor (measured at load: global_max=0.394531,
        // scale2=0.00014678 over 73.9M elements). Puzzle is Mamba-dominant —
        // only 8 attention layers — so these two projections carry the entire
        // sequence state, and the model answers noun-retrieval questions from
        // its training prior instead of from context. Native FP8 keeps the
        // checkpoint's own per-block scales end to end.
        //
        // The old "FP8 direct load causes CUDA 700 / mmap pointers may be
        // invalidated" TODO here was wrong: `WeightStore` holds *device*
        // pointers (the default fast loader uses O_DIRECT and never mmaps), and
        // the NVFP4 arm below has always aliased those pointers for the process
        // lifetime. The two mechanisms that genuinely fault are (1) the NULL
        // `QuantizedWeight`s `load_nemotron_ssm` returns on the FP8 arm being
        // dereferenced by the prefill weight copies, and (2) the checkpoint's
        // 4-byte scalar `weight_scale` being indexed as a [N/128, K/128] matrix
        // by w8a16. Both are closed below: the prefill copies are hard-gated off
        // under native, and `load_fp8_block_scaled_as_fp8weight` materializes a
        // real block-scale buffer. `ATLAS_NEMOTRON_NATIVE_FP8_SSM=0` restores
        // the legacy path exactly (same-binary A/B).
        //
        // DEFAULT ON. The requant is what broke this model, and the effect is
        // specific: language modelling was never damaged, but retrieval of a
        // proper noun from context was. Measured on a 977-token story
        // ("My dog is named Rufus ... mention his name often"):
        //   legacy requant  — calls the dog "Rover" / "Rex"; the given name
        //                     appears ZERO times
        //   native decode   — Rufus x3 / x8, with occasional "Rex"/"Buddy" slips
        //   native BOTH     — Rufus x6 / x5 / x2 / x2 over four trials, and
        //                     ZERO substitutions
        // Short direct questions ("what is my dog called?") passed either way,
        // which is why this hid for so long: only sustained generation exposes it.
        //
        // Safe to default on: the branch requires `quant_kind == Fp8`, and the
        // only other Nemotron checkpoint served here
        // (NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4) has no FP8 config group at all,
        // so the gate cannot fire on it. `=0` restores the legacy path exactly
        // for a same-binary A/B, and every failure to load native FP8 falls back
        // to the legacy arm with a warning rather than aborting.
        //
        // THREE modes, kept because they bisect decode from prefill — the native
        // path changes BOTH (`w8a16_gemv` for decode, `w8a16_gemm[_pipelined]`
        // for prefill), so a single on/off gate cannot say which one moved a
        // result:
        //   "0"            — legacy FP8→BF16→NVFP4 everywhere (baseline)
        //   "decode"       — native decode; legacy NVFP4 requant + prefill
        //                    copies still built, prefill keeps using them
        //   unset/"1"/"both" — native FP8 for decode and prefill (no NVFP4 copies)
        let mode =
            std::env::var("ATLAS_NEMOTRON_NATIVE_FP8_SSM").unwrap_or_else(|_| "1".to_string());
        let native_fp8_enabled = matches!(mode.as_str(), "1" | "both" | "decode");
        // Decode-only mode keeps the legacy NVFP4 weights resident for prefill.
        let native_prefill_wanted = matches!(mode.as_str(), "1" | "both");
        // Probe the kernels BEFORE committing to the path: `w8a16_gemm_pipelined`
        // has silently resolved to handle 0 in a shipped binary before (that is
        // why `kernel_audit` exists), and a 0 handle must degrade to the legacy
        // path rather than fault at the first token. Both projections must be
        // FP8 — a half-native layer would leave one NULL `QuantizedWeight` live.
        let out_is_fp8 = store.contains(&format!("{p}.out_proj.weight_scale"));
        let gemv_k = crate::layers::try_kernel(gpu, "w8a16_gemv", "w8a16_gemv");
        let gemm_pipe_k =
            crate::layers::try_kernel(gpu, "w8a16_gemm_pipelined", "w8a16_gemm_pipelined");
        let gemm_k = crate::layers::try_kernel(gpu, "w8a16_gemm", "w8a16_gemm");
        let native_fp8 = native_fp8_enabled
            && quant_kind == NemotronSsmQuant::Fp8
            && out_is_fp8
            && gemv_k.0 != 0
            && (gemm_pipe_k.0 != 0 || gemm_k.0 != 0);
        // NATIVE BF16 — the same argument as native FP8, for the other mixed-
        // precision case. A ModelOpt checkpoint may leave some Mamba layers
        // unquantized: Nemotron Nano-30B ships 6 of 23 in_proj/out_proj as plain
        // BF16 alongside 17 NVFP4 ones. The legacy arm below then DEQUANTIZES
        // nothing and simply quantizes BF16 -> NVFP4 under ONE global scale, i.e.
        // it invents a quantization the checkpoint never asked for and destroys
        // exactly what the FP8 comment above documents: retrieval of a token from
        // context. Keeping the checkpoint's own BF16 costs ~2x the bytes of those
        // few layers and no accuracy.
        // `ATLAS_NEMOTRON_NATIVE_BF16_SSM=0` restores the legacy requant (A/B).
        let out_is_bf16 = !store.contains(&format!("{p}.out_proj.weight_scale"));
        let dgemm_k = crate::layers::try_kernel(gpu, "gemm", "dense_gemm_bf16_pipelined");
        let dgemv_k = crate::layers::try_kernel(gpu, "gemv", "dense_gemv_bf16");
        let native_bf16 = std::env::var("ATLAS_NEMOTRON_NATIVE_BF16_SSM").as_deref() != Ok("0")
            && quant_kind == NemotronSsmQuant::Bf16
            && out_is_bf16
            && dgemm_k.0 != 0
            && dgemv_k.0 != 0;
        if native_fp8_enabled && quant_kind == NemotronSsmQuant::Fp8 && !native_fp8 {
            static NATIVE_FALLBACK_WARN: std::sync::Once = std::sync::Once::new();
            NATIVE_FALLBACK_WARN.call_once(|| {
                tracing::warn!(
                    "L{i} SSM: native FP8 unavailable (out_proj_fp8={out_is_fp8} \
                     w8a16_gemv={} w8a16_gemm_pipelined={} w8a16_gemm={}) — falling back \
                     to the FP8→BF16→NVFP4 double-quant path",
                    gemv_k.0 != 0,
                    gemm_pipe_k.0 != 0,
                    gemm_k.0 != 0,
                );
            });
        }
        // Native FP8 for prefill only when the native path is live AND the mode
        // asked for it. In "decode" mode the NVFP4 copies below are built as
        // usual and prefill keeps using them.
        let native_prefill = native_fp8 && native_prefill_wanted;
        tracing::info!(
            "L{i} SSM quant={quant_kind:?} native_fp8={native_fp8} native_bf16={native_bf16} \
             native_prefill={native_prefill} \
             in_proj_size={} d_inner={} h={h}",
            config.mamba2_in_proj_size(),
            config.mamba2_d_inner(),
        );
        let native = if native_fp8 {
            let mut in_fp8 =
                load_fp8_block_scaled_as_fp8weight(store, &format!("{p}.in_proj"), gpu)?;
            let mut out_fp8 =
                load_fp8_block_scaled_as_fp8weight(store, &format!("{p}.out_proj"), gpu)?;
            // OWN the FP8 weight bytes. `load_fp8_block_scaled_as_fp8weight`
            // ALIASES the `WeightStore` pointer (loaders_fp8.rs `let weight_ptr =
            // w.ptr`), which is fine for a loader that consumes the bytes during
            // load — the legacy arm below dequantizes them into a fresh NVFP4
            // buffer and never looks again. This arm is different: it keeps the
            // pointer and dereferences it on every token, for the process
            // lifetime. That is the real content of the old "FP8 direct load
            // causes CUDA 700 / mmap pointers may be invalidated" TODO.
            //
            // Measured: with the alias, `w8a16_gemv` is fed stale bytes and the
            // model stays fluent while losing its prompt ("Alice is 30. Bob is
            // 25. Who is older?" -> "Charlie is the oldest."). The kernel itself
            // is NOT at fault — `examples/w8a16_gemv_numeric` checks it at this
            // exact shape (N=18048, K=4096) under both a constant and a varying
            // per-block scale and it is correct to BF16 tolerance.
            //
            // Copying costs ~108 MB per SSM layer (in_proj 74 MB + out_proj
            // 34 MB), ~2.2 GB over the 20 SSM layers — well under the ~6.4 GB of
            // derived prefill copies this path no longer builds.
            for w in [&mut in_fp8, &mut out_fp8] {
                let bytes = (w.n as usize) * (w.k as usize);
                let owned = gpu.alloc(bytes)?;
                gpu.copy_d2d(w.weight, owned, bytes)?;
                w.weight = owned;
            }
            if i == 0 {
                // Prove the bytes the kernel will read ARE the checkpoint's. The
                // numeric test covers the kernel with synthetic weights, so this
                // is the one link it cannot check.
                let mut head = [0u8; 16];
                gpu.copy_d2h(in_fp8.weight, &mut head)?;
                tracing::debug!(
                    "L0 in_proj FP8 head: {}",
                    head.iter()
                        .map(|b| format!("{b:02x}"))
                        .collect::<Vec<_>>()
                        .join(" ")
                );
            }
            // Shape + alignment contract. `w8a16_gemv.cu` computes K16 = K/16 with
            // no tail and issues 16-byte uint4 loads of `B + n*K`, so a K that is
            // not a multiple of 16 reads garbage silently — fail loudly at load
            // instead (Puzzle is K=4096 / K=8192, both fine).
            ensure!(
                in_fp8.n as usize == config.mamba2_in_proj_size() && in_fp8.k as usize == h,
                "L{i} SSM in_proj FP8 shape [{},{}] != [{},{h}]",
                in_fp8.n,
                in_fp8.k,
                config.mamba2_in_proj_size(),
            );
            ensure!(
                out_fp8.n as usize == h && out_fp8.k as usize == config.mamba2_d_inner(),
                "L{i} SSM out_proj FP8 shape [{},{}] != [{h},{}]",
                out_fp8.n,
                out_fp8.k,
                config.mamba2_d_inner(),
            );
            ensure!(
                in_fp8.k % 16 == 0 && out_fp8.k % 16 == 0,
                "L{i} SSM: w8a16 requires K%16==0, got in_proj K={} out_proj K={}",
                in_fp8.k,
                out_fp8.k,
            );
            Some((in_fp8, out_fp8))
        } else {
            None
        };
        // Legacy FP8→BF16→NVFP4 requant. Runs whenever prefill is NOT native
        // (gate unset, or `decode` mode): the prefill arms and every derived
        // weight copy below read `ssm.in_proj`/`ssm.out_proj`, which
        // `load_nemotron_ssm` returns as `QuantizedWeight::null()` on the FP8
        // arm, so they must be materialized here or prefill derefs NULL.
        if !native_prefill && !native_bf16 && quant_kind != NemotronSsmQuant::Nvfp4 {
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
            let out_proj_dense = if out_is_fp8 {
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
        //
        // Both copies are derived FROM `ssm.in_proj`/`ssm.out_proj`, which are
        // `QuantizedWeight::null()` on the native-FP8 arm — `transpose_for_gemm`
        // would `copy_d2h` from NULL and `predequant_to_fp8` would launch on a
        // NULL B. Native prefill goes through w8a16_gemm instead, so neither
        // copy is built (and ~6.4 GB of derived weights is not allocated).
        // `native_bf16` joins `native_prefill` here for the same reason: both
        // leave `ssm.in_proj`/`ssm.out_proj` as NULL `QuantizedWeight`s, and both
        // derived copies read them (`transpose_for_gemm` copy_d2h's from NULL,
        // `predequant_to_fp8` launches on a NULL B).
        let fp8_prefill =
            !native_prefill && !native_bf16 && std::env::var("ATLAS_NO_SSM_FP8_PREFILL").is_err();
        let prefill_t = !native_prefill
            && !native_bf16
            && !fp8_prefill
            && std::env::var("ATLAS_NO_SSM_PREFILL_T").is_err();
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
        let bf16w = if native_bf16 {
            Some((
                dense(store, &format!("{p}.in_proj.weight"))?,
                dense(store, &format!("{p}.out_proj.weight"))?,
            ))
        } else {
            None
        };
        let mut layer = NemotronMamba2Layer::new(norm, ssm, config, gpu, i)?;
        if let Some((in_w, out_w)) = bf16w {
            layer.set_bf16_weights(in_w, out_w);
            ensure!(
                layer.bf16_native_ready(),
                "L{i} SSM: native BF16 selected but the dense kernels are missing"
            );
        }
        if let Some((in_fp8, out_fp8)) = native {
            layer.set_fp8_weights(Some(in_fp8), Some(out_fp8), native_prefill)?;
        }
        if let Some((in_t, out_t)) = proj_t {
            layer.set_prefill_weights(Some(in_t), Some(out_t));
        }
        if let Some((in_fp8, out_fp8)) = proj_fp8 {
            layer.set_fp8_prefill_weights(in_fp8, out_fp8);
        }
        Ok(layer)
    }
}
