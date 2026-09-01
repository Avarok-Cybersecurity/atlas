// SPDX-License-Identifier: AGPL-3.0-only

//! EXL3 checkpoint materialization: rewrite trellis-quantized linears into
//! the tensor layouts the existing model loaders already consume, so every
//! family's load path (qwen4_exp included) works on an EXL3 checkpoint with
//! ZERO per-arm changes.
//!
//! Runs ONCE, right after the weight store is loaded and before quant-format
//! detection / model construction. Idempotent: after the pass no `.trellis`
//! tensor remains, so a second call is a no-op.
//!
//! Per EXL3 linear (`{p}.trellis` + `.suh` + `.svh` [+ `.mul1`]), routed by
//! prefix:
//!
//!  * **MoE experts + shared experts** (`.mlp.experts.N.` /
//!    `.mlp.shared_expert.`) — reconstruct to BF16 (transient) and
//!    immediately runtime-quantize to the ModelOpt-style NVFP4 triplet
//!    (`.weight` U8 `[n, k/2]` + `.weight_scale` FP8 `[n, k/16]` +
//!    `.weight_scale_2` F32 scalar). `quantized_any` then takes its
//!    `Standard` arm verbatim. Quantizing INSIDE the pass is load-bearing:
//!    the routed experts are ~90% of the model's parameters, and holding
//!    them all as BF16 simultaneously (~4x the packed bytes) cannot fit —
//!    the transient here is one tensor at a time.
//!  * **Everything else** (attention, GDN, lm_head, MTP) — materialize as a
//!    plain BF16 `.weight` `[out, in]`. The Standard-variant arms read
//!    exactly that (`dense_auto` + runtime NVFP4 quantization per arm), and
//!    these tensors are small enough (~6 GB total on Qwen3.8-Flash-Next)
//!    that BF16 residency until construction is fine.
//!
//! The source trellis/suh/svh/mul1 tensors are freed as each linear lands,
//! so peak memory ≈ the packed checkpoint + one transient BF16 tensor.
//!
//! NOT covered here (documented gaps, each needs its own decode path):
//!  * `ngram_embedding.safetensors` (`exl3_ngram_trellis` row format) — the
//!    PLE n-gram tables. A qwen4_exp load will fail at the PLE loader's
//!    existing "no shard_* was deferred" check; this pass logs the reason
//!    up front.
//!  * `vision_k6.safetensors` — the EXL3-quantized vision encoder ships
//!    outside the weight index and is not loaded.
//!
//! Double-quantization note: EXL3(K bits) -> BF16 -> NVFP4 costs quality vs
//! a native-NVFP4 calibration AND vs native EXL3 serving. This pass is the
//! LOADING path (checkpoint compatibility); the native fused trellis-GEMM
//! (`.research/EXL3_DECODE_FINDINGS.md`) is the fidelity/memory path.

use anyhow::{Context, Result};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::exl3::{Exl3Weight, store_has_exl3};
use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

use super::{DenseWeight, quantize_to_nvfp4};

/// What the pass did — for the load log.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Exl3MaterializeStats {
    /// Linears rewritten as NVFP4 triplets (experts).
    pub quantized: usize,
    /// Linears rewritten as dense BF16 `.weight`.
    pub bf16: usize,
}

/// Expert-family prefixes get the NVFP4 triplet; everything else BF16.
fn wants_nvfp4_triplet(prefix: &str) -> bool {
    prefix.contains(".mlp.experts.") || prefix.contains(".mlp.shared_expert.")
}

/// Rewrite every EXL3 linear in `store` into loader-consumable tensors.
/// No-op (Ok, zero stats) when the store has no EXL3 tensors.
pub fn materialize_exl3(
    gpu: &dyn GpuBackend,
    store: &mut WeightStore,
) -> Result<Exl3MaterializeStats> {
    let mut stats = Exl3MaterializeStats::default();
    if !store_has_exl3(store) {
        return Ok(stats);
    }

    let prefixes: Vec<String> = store
        .names()
        .filter_map(|n| n.strip_suffix(".trellis").map(str::to_string))
        .collect();
    tracing::info!(
        "EXL3 checkpoint: materializing {} trellis-quantized linears \
         (experts -> NVFP4 triplet, rest -> BF16 dense)",
        prefixes.len()
    );
    // The PLE n-gram tables ship as a separate `exl3_ngram_trellis`-format
    // file that never enters the store; a model that requires them
    // (qwen4_exp) will fail at the PLE loader with its missing-shard error.
    // Say why up front rather than letting that error stand alone.
    if !store
        .names()
        .any(|n| n.contains("ple_embedding.ngram_embedding.shard_"))
    {
        tracing::warn!(
            "EXL3 checkpoint has no in-store PLE n-gram shards (the exl3 export keeps \
             them in ngram_embedding.safetensors, an exl3_ngram_trellis row-format file \
             Atlas does not decode yet). Models that REQUIRE PLE (qwen4_exp) will fail \
             at the PLE loader; models without PLE are unaffected."
        );
    }

    let absmax_k = gpu
        .kernel("quantize_nvfp4", "nvfp4_global_absmax")
        .context("EXL3 materialization needs the quantize_nvfp4 kernels")?;
    let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    let stream = gpu.default_stream();

    for p in &prefixes {
        let w = Exl3Weight::from_store(gpu, store, p)
            .with_context(|| format!("EXL3 materialization: resolving {p}"))?;
        let (n, k) = (w.out_dim, w.in_dim);
        let bf16 = w
            .to_bf16(gpu)
            .with_context(|| format!("EXL3 materialization: reconstructing {p}"))?;

        if wants_nvfp4_triplet(p) {
            let dense = DenseWeight { weight: bf16 };
            let q = quantize_to_nvfp4(&dense, n, k, gpu, absmax_k, quantize_k, stream)
                .with_context(|| format!("EXL3 materialization: quantizing {p}"))?;
            gpu.free(bf16)?;
            let scale2 = gpu.alloc(4)?;
            gpu.copy_h2d(&q.weight_scale_2.to_le_bytes(), scale2)?;
            store.insert(
                format!("{p}.weight"),
                WeightTensor {
                    ptr: q.weight,
                    shape: vec![n, k / 2],
                    dtype: WeightDtype::UInt8,
                },
            );
            store.insert(
                format!("{p}.weight_scale"),
                WeightTensor {
                    ptr: q.weight_scale,
                    shape: vec![n, k / 16],
                    dtype: WeightDtype::FP8E4M3,
                },
            );
            store.insert(
                format!("{p}.weight_scale_2"),
                WeightTensor {
                    ptr: scale2,
                    shape: vec![1],
                    dtype: WeightDtype::FP32,
                },
            );
            stats.quantized += 1;
        } else {
            store.insert(
                format!("{p}.weight"),
                WeightTensor {
                    ptr: bf16,
                    shape: vec![n, k],
                    dtype: WeightDtype::BF16,
                },
            );
            stats.bf16 += 1;
        }

        for suffix in ["trellis", "suh", "svh", "mul1"] {
            if let Some(t) = store.remove(&format!("{p}.{suffix}")) {
                gpu.free(t.ptr)?;
            }
        }
    }

    tracing::info!(
        "EXL3 materialization done: {} experts -> NVFP4 triplets, {} linears -> BF16 dense",
        stats.quantized,
        stats.bf16
    );
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use spark_runtime::gpu::mock::MockGpuBackend;

    use super::*;

    fn t(gpu: &MockGpuBackend, shape: Vec<usize>, dtype: WeightDtype) -> WeightTensor {
        let bytes: usize = shape.iter().product::<usize>() * dtype.byte_size().max(1);
        WeightTensor {
            ptr: gpu.alloc(bytes.max(4)).unwrap(),
            shape,
            dtype,
        }
    }

    fn exl3_linear(gpu: &MockGpuBackend, m: &mut HashMap<String, WeightTensor>, p: &str, k: u32) {
        // [2560 -> 640] geometry, K bits.
        m.insert(
            format!("{p}.trellis"),
            t(gpu, vec![160, 40, 16 * k as usize], WeightDtype::UInt16),
        );
        m.insert(format!("{p}.suh"), t(gpu, vec![2560], WeightDtype::F16));
        m.insert(format!("{p}.svh"), t(gpu, vec![640], WeightDtype::F16));
        m.insert(format!("{p}.mul1"), t(gpu, vec![], WeightDtype::Int32));
    }

    #[test]
    fn no_exl3_is_noop() {
        let gpu = MockGpuBackend::new();
        let mut m = HashMap::new();
        m.insert("a.weight".to_string(), t(&gpu, vec![8, 8], WeightDtype::BF16));
        let mut store = WeightStore::from_map(m);
        let stats = materialize_exl3(&gpu, &mut store).unwrap();
        assert_eq!(stats, Exl3MaterializeStats::default());
        assert!(store.contains("a.weight"));
    }

    #[test]
    fn routes_experts_to_triplet_and_attention_to_bf16() {
        let gpu = MockGpuBackend::new();
        let mut m = HashMap::new();
        exl3_linear(
            &gpu,
            &mut m,
            "model.layers.0.mlp.experts.3.gate_proj",
            4,
        );
        exl3_linear(
            &gpu,
            &mut m,
            "model.layers.0.mlp.shared_expert.up_proj",
            6,
        );
        exl3_linear(&gpu, &mut m, "model.layers.0.linear_attn.in_proj_qkv", 6);
        // Bystander that must survive untouched.
        m.insert(
            "model.layers.0.norm.weight".to_string(),
            t(&gpu, vec![2560], WeightDtype::BF16),
        );
        let mut store = WeightStore::from_map(m);

        let stats = materialize_exl3(&gpu, &mut store).unwrap();
        assert_eq!(stats.quantized, 2);
        assert_eq!(stats.bf16, 1);

        // Expert: ModelOpt-style NVFP4 triplet, [n=640, k=2560].
        let ep = "model.layers.0.mlp.experts.3.gate_proj";
        let w = store.get(&format!("{ep}.weight")).unwrap();
        assert_eq!(w.dtype, WeightDtype::UInt8);
        assert_eq!(w.shape, vec![640, 1280]); // [n, k/2]
        let s = store.get(&format!("{ep}.weight_scale")).unwrap();
        assert_eq!(s.dtype, WeightDtype::FP8E4M3);
        assert_eq!(s.shape, vec![640, 160]); // [n, k/16]
        let s2 = store.get(&format!("{ep}.weight_scale_2")).unwrap();
        assert_eq!(s2.dtype, WeightDtype::FP32);

        // Attention: dense BF16 [out, in].
        let ap = "model.layers.0.linear_attn.in_proj_qkv";
        let w = store.get(&format!("{ap}.weight")).unwrap();
        assert_eq!(w.dtype, WeightDtype::BF16);
        assert_eq!(w.shape, vec![640, 2560]);

        // Every EXL3 source tensor is gone; the bystander survived.
        for p in [ep, ap, "model.layers.0.mlp.shared_expert.up_proj"] {
            for sfx in ["trellis", "suh", "svh", "mul1"] {
                assert!(!store.contains(&format!("{p}.{sfx}")), "{p}.{sfx} remains");
            }
        }
        assert!(store.contains("model.layers.0.norm.weight"));

        // Idempotent: second call is a no-op.
        let again = materialize_exl3(&gpu, &mut store).unwrap();
        assert_eq!(again, Exl3MaterializeStats::default());
    }
}
