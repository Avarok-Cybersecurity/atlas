// SPDX-License-Identifier: AGPL-3.0-only

//! Native EXL3 routed-expert loader (`ATLAS_EXL3_NATIVE_MOE=1`).
//!
//! The materialize pass (`exl3_materialize.rs` + `exl3_materialize_moe.rs`)
//! kept this layer's routed experts packed; this loader resolves them into
//! per-projection `Vec<Option<Exl3Weight>>` (index = GLOBAL expert id,
//! `None` = remote under EP — the EP load-skip drops remote experts'
//! `.trellis/.suh/.svh/.mul1` before they ever reach the store, pinned by
//! the test in `spark-runtime/src/weights/exl3.rs`). The layer then builds
//! DENSE local pointer tables from these vecs
//! (`layers/moe/ptr_table_build.rs::build_exl3_ptr_table`) and remote
//! experts are mapped to `-1` slot INDICES — never NULL table entries, which
//! the mgemm weighted reduction would silently sum stale scratch through.
//!
//! Mirrors `load_moe_qwen35_fp8_experts`'s shape: the caller pairs this with
//! `load_moe_qwen35(skip_routed_experts=true)` so `MoeWeights` keeps its
//! structure (null routed `ExpertWeight`s; shared expert + router still
//! materialized NVFP4).
//!
//! Codebook readback: `Exl3Weight::from_store` does a synchronous 4-byte
//! `.mul1` D2H per tensor — 73,728 of them across a 512-expert model. The
//! materialize pass already read AND validated every flag for its uniformity
//! keep-condition, so this loader reads ONE flag per (layer, projection) and
//! resolves the siblings with `from_store_with_cb`.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;
use spark_runtime::weights::exl3::{Exl3Codebook, Exl3Weight};

/// One MoE layer's routed experts as kept-packed EXL3 trellis, per
/// projection. `Vec` index = GLOBAL expert id; `None` = remote under EP.
pub struct Exl3MoeExperts {
    pub gate: Vec<Option<Exl3Weight>>,
    pub up: Vec<Option<Exl3Weight>>,
    pub down: Vec<Option<Exl3Weight>>,
}

/// Load one layer's kept-packed routed experts. Iterates ONLY the EP-local
/// expert range (remote experts' tensors are not in the store) and validates
/// per-projection `(K, cb)` uniformity + geometry against the config — a
/// failure here is a materialize/load divergence bug (the keep pass
/// guarantees uniformity), so it fails loudly rather than falling back.
pub(crate) fn load_moe_qwen4exp_exl3(
    store: &WeightStore,
    layer_prefix: &str,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    config: &atlas_core::config::ModelConfig,
) -> Result<Exl3MoeExperts> {
    let p = format!("{layer_prefix}.mlp");
    let (local_start, local_end) = config.local_expert_range();
    ensure!(
        local_end > local_start && local_end <= num_experts,
        "EXL3 native MoE {layer_prefix}: empty/invalid local expert range \
         [{local_start}, {local_end}) of {num_experts}"
    );
    let h = config.hidden_size;
    let inter = config.moe_intermediate_size;

    let load_proj =
        |proj: &str, in_dim: usize, out_dim: usize| -> Result<Vec<Option<Exl3Weight>>> {
            let mut v: Vec<Option<Exl3Weight>> = vec![None; num_experts];
            // One codebook readback per (layer, projection); siblings reuse it.
            let mut cb: Option<Exl3Codebook> = None;
            let mut first_kc: Option<(u32, Exl3Codebook)> = None;
            for e in local_start..local_end {
                let prefix = format!("{p}.experts.{e}.{proj}");
                let w = match cb {
                    None => {
                        let w = Exl3Weight::from_store(gpu, store, &prefix)
                            .with_context(|| format!("EXL3 native MoE: resolving {prefix}"))?;
                        cb = Some(w.cb);
                        w
                    }
                    Some(c) => Exl3Weight::from_store_with_cb(gpu, store, &prefix, c)
                        .with_context(|| format!("EXL3 native MoE: resolving {prefix}"))?,
                };
                ensure!(
                    w.in_dim == in_dim && w.out_dim == out_dim,
                    "EXL3 native MoE {prefix}: trellis geometry [{}x{}] does not \
                 match the config's [{in_dim}x{out_dim}]",
                    w.in_dim,
                    w.out_dim,
                );
                ensure!(
                    super::exl3_native_supported_moe(&w),
                    "EXL3 native MoE {prefix}: K={} cb={:?} is outside the MoE \
                 kernel envelope — the materialize pass should not have kept \
                 this layer (keep/load predicate divergence bug)",
                    w.k_bits,
                    w.cb,
                );
                let kc = (w.k_bits, w.cb);
                match first_kc {
                    None => first_kc = Some(kc),
                    Some(exp) => ensure!(
                        exp == kc,
                        "EXL3 native MoE {prefix}: K={}/cb={:?} differs from this \
                     projection's first expert (K={}/cb={:?}) — one mgemm \
                     launch decodes at ONE template; the materialize pass \
                     should have materialized this layer (divergence bug)",
                        kc.0,
                        kc.1,
                        exp.0,
                        exp.1,
                    ),
                }
                v[e] = Some(w);
            }
            Ok(v)
        };

    Ok(Exl3MoeExperts {
        gate: load_proj("gate_proj", h, inter)?,
        up: load_proj("up_proj", h, inter)?,
        down: load_proj("down_proj", inter, h)?,
    })
}
