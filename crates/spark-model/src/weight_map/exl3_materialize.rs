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

// The native-serving predicates live in the child module (≤500 LoC split);
// the re-exports keep the `weight_map::exl3_native_*` paths unchanged.
#[path = "exl3_materialize_native.rs"]
mod native;
pub use native::{
    exl3_native_enabled, exl3_native_serves, exl3_native_serves_with, exl3_native_supported,
};

// `register_exl3_ngram_sidecar` lives in the child module (≤500 LoC split);
// the re-export keeps `weight_map::register_exl3_ngram_sidecar` unchanged.
#[path = "exl3_materialize_ngram.rs"]
mod ngram;
pub use ngram::register_exl3_ngram_sidecar;

/// What the pass did — for the load log.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Exl3MaterializeStats {
    /// Linears rewritten as NVFP4 triplets (experts).
    pub quantized: usize,
    /// Linears rewritten as dense BF16 `.weight`.
    pub bf16: usize,
    /// Linears kept packed for native serving (`ATLAS_EXL3_NATIVE=1`),
    /// routed experts included.
    pub kept_native: usize,
    /// The routed-expert subset of `kept_native` (`ATLAS_EXL3_NATIVE_MOE=1`).
    pub kept_native_experts: usize,
    /// Resident bytes of the kept-packed expert tensors.
    pub kept_packed_bytes: usize,
    /// What those same experts WOULD have cost as runtime NVFP4 triplets —
    /// the memory the keep saved, for the load log.
    pub nvfp4_equiv_bytes: usize,
    /// The GDN/attention dense subset of `kept_native`
    /// (`ATLAS_EXL3_NATIVE_DENSE=1`), with its per-family layer counts.
    pub dense: super::Exl3DenseKeepStats,
}

/// Expert-family prefixes get the NVFP4 triplet; everything else BF16.
fn wants_nvfp4_triplet(prefix: &str) -> bool {
    prefix.contains(".mlp.experts.") || prefix.contains(".mlp.shared_expert.")
}

/// Rewrite every EXL3 linear in `store` into loader-consumable tensors.
/// No-op (Ok, zero stats) when the store has no EXL3 tensors.
///
/// Gate validation runs FIRST, before the EXL3-store early-out:
/// `ATLAS_EXL3_NATIVE_MOE=1` without `ATLAS_EXL3_NATIVE=1` errors on every
/// load, EXL3 checkpoint or not — a misconfiguration must never silently
/// serve something else.
pub fn materialize_exl3(
    gpu: &dyn GpuBackend,
    store: &mut WeightStore,
) -> Result<Exl3MaterializeStats> {
    let native = exl3_native_enabled();
    let native_moe = super::exl3_native_moe_enabled();
    super::check_exl3_native_gates(native, native_moe)?;
    let dense_env = super::exl3_native_dense_enabled();
    let gdn_env = std::env::var("ATLAS_EXL3_NATIVE_GDN").ok();
    let attn_env = std::env::var("ATLAS_EXL3_NATIVE_ATTN").ok();
    super::check_exl3_native_dense_gates(
        native,
        dense_env,
        gdn_env.as_deref(),
        attn_env.as_deref(),
    )?;
    let dense =
        super::exl3_native_dense_families_with(dense_env, gdn_env.as_deref(), attn_env.as_deref());
    materialize_exl3_impl(gpu, store, native, native_moe, dense)
}

/// Env-independent body (tests exercise `native`/`native_moe`/`dense`
/// directly — `set_var` in parallel unit tests races).
pub(crate) fn materialize_exl3_impl(
    gpu: &dyn GpuBackend,
    store: &mut WeightStore,
    native: bool,
    native_moe: bool,
    dense: super::Exl3DenseFamilies,
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
        tracing::info!(
            "EXL3 checkpoint has no in-store PLE n-gram shards — the exl3 export keeps \
             them in ngram_embedding.safetensors; `register_exl3_ngram_sidecar` (called \
             right after this pass on the serve path) registers that file's trellis \
             tensors for the NVMe row cache. Models without PLE are unaffected either way."
        );
    }

    let absmax_k = gpu
        .kernel("quantize_nvfp4", "nvfp4_global_absmax")
        .context("EXL3 materialization needs the quantize_nvfp4 kernels")?;
    let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    let stream = gpu.default_stream();

    // ── Native MoE keep-set (ATLAS_EXL3_NATIVE_MOE=1) ──
    // Resolve every routed-expert linear ONCE up front (the resolve includes
    // the 4-byte `.mul1` codebook readback the uniformity check needs; the
    // main loop reuses these resolutions instead of re-reading), then decide
    // keep-vs-materialize ATOMICALLY per layer: per-(layer, projection) K/cb
    // uniformity + layer-uniform codebook + the MoE kernel envelope, or the
    // whole layer's experts materialize. No partial keeps — a half-kept
    // layer would double-hold memory (see exl3_materialize_moe.rs).
    let mut expert_weights: std::collections::BTreeMap<String, Exl3Weight> =
        std::collections::BTreeMap::new();
    if native && native_moe {
        for p in prefixes.iter().filter(|p| super::exl3_native_serves_moe(p)) {
            let w = Exl3Weight::from_store(gpu, store, p)
                .with_context(|| format!("EXL3 native MoE: resolving {p}"))?;
            expert_weights.insert(p.clone(), w);
        }
    }
    let keep_experts = super::expert_keep_set(&expert_weights);

    // ── Native dense keep-set (ATLAS_EXL3_NATIVE_DENSE=1) ──
    // Same shape as the expert set: resolve every gate-admitted GDN /
    // attention projection up front, then decide ATOMICALLY per (layer,
    // family) — all of a layer's `linear_attn.{qkv,z,out}` (or
    // `self_attn.{q,k,v,o}`) inside the K in {2,4} GEMV envelope, or the
    // whole family materializes to BF16 exactly as today (see
    // exl3_materialize_dense.rs). The loader arms re-derive "kept" from the
    // family's `.trellis` tensors still being in the store.
    let mut dense_weights: std::collections::BTreeMap<String, Exl3Weight> =
        std::collections::BTreeMap::new();
    if native && dense.any() {
        for p in prefixes
            .iter()
            .filter(|p| super::exl3_native_serves_dense(p, dense))
        {
            let w = Exl3Weight::from_store(gpu, store, p)
                .with_context(|| format!("EXL3 native dense: resolving {p}"))?;
            dense_weights.insert(p.clone(), w);
        }
    }
    let (keep_dense, dense_stats) = super::dense_keep_set(&dense_weights);
    stats.dense = dense_stats;

    for p in &prefixes {
        let is_expert = super::exl3_native_serves_moe(p);
        let is_dense = dense_weights.contains_key(p);
        let w = match expert_weights.get(p).or_else(|| dense_weights.get(p)) {
            Some(w) => *w,
            None => Exl3Weight::from_store(gpu, store, p)
                .with_context(|| format!("EXL3 materialization: resolving {p}"))?,
        };

        // Native serving (ATLAS_EXL3_NATIVE=1): leave the packed tensors in
        // the store for the model builder to resolve via
        // `Exl3Weight::from_store` — skip the rewrite AND the frees. Only
        // prefixes with a routed serving path (`exl3_native_serves`) AND a
        // compiled kernel envelope qualify (`exl3_native_supported` for the
        // lm_head; the atomic per-layer `keep_experts` / `keep_dense` sets —
        // which fold in the envelope + uniformity — for experts and the
        // dense families); an unsupported tensor falls through to
        // materialization with a log, so the same K/cb decision the builder
        // re-derives holds here.
        if native && exl3_native_serves_with(p, native_moe, dense) {
            let keep = if is_expert {
                keep_experts.contains(p)
            } else if is_dense {
                keep_dense.contains(p)
            } else {
                exl3_native_supported(&w)
            };
            if keep {
                if is_expert {
                    // Aggregate-logged after the loop: per-tensor logs at
                    // 73,728 expert projections would drown the load log.
                    stats.kept_native_experts += 1;
                    stats.kept_packed_bytes += w.packed_bytes();
                    stats.nvfp4_equiv_bytes += w.nvfp4_equiv_bytes();
                } else if is_dense {
                    // Aggregate-logged per family after the loop (the
                    // keep-set already accounted the bytes); the loader
                    // logs one line per installed layer family.
                } else {
                    tracing::info!(
                        "EXL3 native: keeping {p} packed ([{}x{}] K={} cb={:?}) for the \
                         fused trellis matmul path",
                        w.in_dim,
                        w.out_dim,
                        w.k_bits,
                        w.cb,
                    );
                }
                stats.kept_native += 1;
                continue;
            }
            // Expert / dense layers outside their keep-set were already
            // warned about (once per layer family, with the reason) by
            // `expert_keep_set` / `dense_keep_set`.
            if !is_expert && !is_dense {
                tracing::warn!(
                    "EXL3 native: {p} requested native serving but K={} cb={:?} \
                     [{}x{}] is outside the compiled kernel envelope — materializing \
                     to BF16 instead",
                    w.k_bits,
                    w.cb,
                    w.in_dim,
                    w.out_dim,
                );
            }
        }
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
        "EXL3 materialization done: {} experts -> NVFP4 triplets, {} linears -> \
         BF16 dense, {} kept packed for native serving",
        stats.quantized,
        stats.bf16,
        stats.kept_native,
    );
    if stats.kept_native_experts > 0 {
        tracing::info!(
            "EXL3 native MoE: {} routed-expert projections kept packed — \
             {:.2} GB resident vs {:.2} GB as runtime NVFP4 triplets \
             ({:.2} GB saved)",
            stats.kept_native_experts,
            stats.kept_packed_bytes as f64 / 1e9,
            stats.nvfp4_equiv_bytes as f64 / 1e9,
            (stats
                .nvfp4_equiv_bytes
                .saturating_sub(stats.kept_packed_bytes)) as f64
                / 1e9,
        );
    }
    if dense.any() {
        super::log_unrouted_dense_families(dense);
        let d = &stats.dense;
        tracing::info!(
            "EXL3 native dense: GDN routed set ({:?}) kept packed on {} layers ({} \
             materialized), attention routed set ({:?}) kept packed on {} layers ({} \
             materialized) — {:.2} GB resident vs {:.2} GB as BF16 dense ({:.2} GB saved)",
            super::Exl3DenseFamily::Gdn.leaves(),
            d.gdn_layers_kept,
            d.gdn_layers_materialized,
            super::Exl3DenseFamily::Attn.leaves(),
            d.attn_layers_kept,
            d.attn_layers_materialized,
            d.kept_packed_bytes as f64 / 1e9,
            d.bf16_equiv_bytes as f64 / 1e9,
            d.bf16_equiv_bytes.saturating_sub(d.kept_packed_bytes) as f64 / 1e9,
        );
    }
    Ok(stats)
}

#[cfg(test)]
#[path = "exl3_materialize_tests.rs"]
mod tests;
