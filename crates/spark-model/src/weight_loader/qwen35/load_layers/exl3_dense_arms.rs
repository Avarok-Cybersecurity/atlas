// SPDX-License-Identifier: AGPL-3.0-only

//! Native EXL3 dense installs (`ATLAS_EXL3_NATIVE_DENSE=1`) for the GDN and
//! full-attention arms of `load_layers`: resolve the kept-packed projections
//! from the store, validate them against the config geometry, probe their
//! kernels, bind the model-shared dense stage, install on the layer and log
//! ONE line per layer. Split from `linear_attn_arms.rs` /
//! `attention_arms.rs` (500-LoC cap).
//!
//! The decision "is this layer's family kept?" is NOT made here — the arms
//! call [`crate::weight_map::exl3_dense_family_kept`] first and skip their
//! BF16/NVFP4 loads; these helpers only run on a layer whose projection
//! slots were left null.

use std::sync::Arc;

use anyhow::{Context, Result};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use crate::layers::exl3_dense::{Exl3AttnWeights, Exl3GdnWeights};
use crate::layers::ops::Exl3DenseStage;
use crate::layers::{Qwen3AttentionLayer, Qwen3SsmLayer};
use crate::tp_shard::{TpAttentionDims, TpGdnDims};

/// The model-shared stage is created by the model loader (qwen4_exp) before
/// its first native layer; a caller that kept a family packed without one is
/// a wiring bug, not a runtime condition.
fn stage_or_bail(stage: Option<&Arc<Exl3DenseStage>>, what: &str) -> Result<Arc<Exl3DenseStage>> {
    stage.cloned().with_context(|| {
        format!(
            "EXL3 native {what}: the store kept this layer's projections packed but the \
             loader passed no model-shared dense stage (locks/fence/f16 staging) — this \
             model loader does not thread `Exl3DenseStage`; unset ATLAS_EXL3_NATIVE_DENSE"
        )
    })
}

/// Install the packed GDN family on a freshly built [`Qwen3SsmLayer`] whose
/// fused-QKVZ and `out_proj` slots are null.
#[allow(clippy::too_many_arguments)]
pub(super) fn install_native_gdn(
    layer: &mut Qwen3SsmLayer,
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    lp: &str,
    layer_idx: usize,
    h: usize,
    dims: &TpGdnDims,
    stage: Option<&Arc<Exl3DenseStage>>,
) -> Result<()> {
    let stage = stage_or_bail(stage, "GDN")?;
    // Resolve + validate against the config geometry and probe the
    // exl3_matmul instances the arm dispatches to (a missing module fails
    // HERE, not on the first request).
    let w = Exl3GdnWeights::from_store(
        gpu,
        store,
        lp,
        h,
        dims.full_conv_dim(),
        dims.full_value_dim(),
        stage,
    )
    .with_context(|| format!("Layer {layer_idx}: EXL3 native GDN family"))?;
    tracing::info!(
        "Layer {layer_idx}: EXL3 native GDN family installed — in_proj_qkv [{}->{}] K={} + \
         in_proj_z [{}->{}] K={} (shared-A pair into the fused [Q|K|V|Z] row of {}), \
         out_proj [{}->{}] K={} cb={}; {:.1} MB packed vs {:.1} MB BF16; decode-graph \
         capture vetoed",
        w.in_proj_qkv.in_dim,
        w.in_proj_qkv.out_dim,
        w.in_proj_qkv.k_bits,
        w.in_proj_z.in_dim,
        w.in_proj_z.out_dim,
        w.in_proj_z.k_bits,
        w.qkvz_row_elems(),
        w.out_proj.in_dim,
        w.out_proj.out_dim,
        w.out_proj.k_bits,
        if w.out_proj.cb == 2 { "MUL1" } else { "MCG" },
        w.packed_bytes() as f64 / 1e6,
        w.bf16_bytes() as f64 / 1e6,
    );
    layer.set_exl3_gdn_weights(w)
}

/// Install the packed attention family on a freshly built
/// [`Qwen3AttentionLayer`] whose q/k/v/o slots are null.
pub(super) fn install_native_attn(
    layer: &mut Qwen3AttentionLayer,
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    lp: &str,
    layer_idx: usize,
    dims: &TpAttentionDims,
    stage: Option<&Arc<Exl3DenseStage>>,
) -> Result<()> {
    let stage = stage_or_bail(stage, "attention")?;
    let w = Exl3AttnWeights::from_store(
        gpu,
        store,
        lp,
        dims.h,
        dims.full_q_n,
        dims.full_kv_n,
        dims.full_o_in,
        stage,
    )
    .with_context(|| format!("Layer {layer_idx}: EXL3 native attention family"))?;
    tracing::info!(
        "Layer {layer_idx}: EXL3 native attention family installed — q_proj [{}->{}] K={} \
         ({}), k/v_proj [{}->{}] K={}/{}, o_proj [{}->{}] K={} cb={}; {:.1} MB packed vs \
         {:.1} MB BF16 (no runtime NVFP4 requant, no transposed twins); decode-graph \
         capture vetoed",
        w.q_proj.in_dim,
        w.q_proj.out_dim,
        w.q_proj.k_bits,
        if dims.gated {
            "gated, [Q|gate] interleaved per head"
        } else {
            "ungated"
        },
        w.k_proj.in_dim,
        w.k_proj.out_dim,
        w.k_proj.k_bits,
        w.v_proj.k_bits,
        w.o_proj.in_dim,
        w.o_proj.out_dim,
        w.o_proj.k_bits,
        if w.o_proj.cb == 2 { "MUL1" } else { "MCG" },
        w.packed_bytes() as f64 / 1e6,
        w.bf16_bytes() as f64 / 1e6,
    );
    layer.set_exl3_attn_weights(w)
}
