// SPDX-License-Identifier: AGPL-3.0-only

//! Model-shared native-EXL3 state for the qwen4_exp loader — ONE
//! [`Exl3LaunchState`] (locks + fence + section mutex) that the MoE arm's
//! [`Exl3MoeState`] and the dense arms' [`Exl3DenseStage`] both hang off, so
//! the "one cooperative dispatch section at a time" invariant is GLOBAL
//! across every native launch that is not the LM head — plus the
//! `ATLAS_EXL3_NATIVE_DENSE=1` load-time tally and its summary line. Split
//! from `qwen4_exp.rs` (500-LoC cap).
//!
//! The GDN / attention arms decide per layer from the store
//! (`exl3_dense_family_kept`) and install the packed projections on the
//! layer; the tally only counts the same predicate so the summary cannot
//! disagree with what the arms did.

use std::sync::Arc;

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use crate::layers::moe::Exl3MoeState;
use crate::layers::ops::{EXL3_DENSE_STAGE_ROWS_DEFAULT, Exl3DenseStage, Exl3LaunchState};
use crate::weight_map::{Exl3DenseFamilies, Exl3DenseFamily, exl3_dense_family_kept};

pub(super) struct NativeExl3 {
    /// The per-model launch state; created by whichever arm needs it first.
    pub(super) launch: Option<Arc<Exl3LaunchState>>,
    /// Native MoE mgemm state (`ATLAS_EXL3_NATIVE_MOE=1`), ~140 MB of slabs.
    pub(super) moe: Option<Arc<Exl3MoeState>>,
    /// Native dense staging (`ATLAS_EXL3_NATIVE_DENSE=1`), sized once.
    stage: Option<Arc<Exl3DenseStage>>,
    families: Exl3DenseFamilies,
    gdn_layers: usize,
    attn_layers: usize,
}

impl NativeExl3 {
    pub(super) fn new() -> Self {
        Self {
            launch: None,
            moe: None,
            stage: None,
            families: crate::weight_map::exl3_native_dense_families(),
            gdn_layers: 0,
            attn_layers: 0,
        }
    }

    /// Count layer `lp` if either of its families is kept packed.
    pub(super) fn observe(&mut self, store: &WeightStore, lp: &str) {
        if exl3_dense_family_kept(store, lp, Exl3DenseFamily::Gdn) {
            self.gdn_layers += 1;
        }
        if exl3_dense_family_kept(store, lp, Exl3DenseFamily::Attn) {
            self.attn_layers += 1;
        }
    }

    /// The model-shared dense stage for the layer arms — `None` when the
    /// dense gate is off (the arms never reach their native branch then).
    /// Created on first use, inside the util pledge and before the KV budget
    /// (layer construction precedes both), from the model-wide maxima of the
    /// ROUTED projections (`Exl3DenseFamily::leaves`): the GDN family —
    /// in_proj_qkv `[hidden -> conv_dim]`, in_proj_z `[hidden ->
    /// value_dim]`, out_proj `[value_dim -> hidden]` — and the attention
    /// family — q_proj `[hidden -> q_n]` (12288 gated), k/v `[hidden ->
    /// kv_n]`, o_proj `[o_in -> hidden]`. Rows above the stage capacity
    /// are batched, so the default row count is a launch-count knob, not a
    /// correctness bound.
    pub(super) fn stage(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &ModelConfig,
    ) -> Result<Option<&Arc<Exl3DenseStage>>> {
        if !self.families.any() {
            return Ok(None);
        }
        if self.stage.is_none() {
            let gdn = crate::tp_shard::TpGdnDims::from_config(config);
            let h = config.hidden_size;
            let (mut max_in, mut max_out) = (128usize, 128usize);
            if self.families.gdn && Exl3DenseFamily::Gdn.routed() {
                // in_proj pair: A is `[m, hidden]`, the qkv block is the
                // widest C; out_proj: A is `[m, value_dim]`, C `[m, hidden]`.
                max_in = max_in.max(h).max(gdn.full_value_dim());
                max_out = max_out
                    .max(h)
                    .max(gdn.full_conv_dim())
                    .max(gdn.full_value_dim());
            }
            if self.families.attn && Exl3DenseFamily::Attn.routed() {
                let attn = crate::tp_shard::TpAttentionDims::from_config(config);
                max_in = max_in.max(h).max(attn.full_o_in);
                max_out = max_out.max(h).max(attn.full_q_n).max(attn.full_kv_n);
            }
            let launch = Exl3LaunchState::get_or_create(&mut self.launch, gpu)?;
            // fp32-C rows for the residual-bound out_proj / o_proj (out = h).
            Exl3DenseStage::get_or_create(
                &mut self.stage,
                gpu,
                &launch,
                EXL3_DENSE_STAGE_ROWS_DEFAULT,
                max_in,
                max_out,
                h,
            )?;
        }
        Ok(self.stage.as_ref())
    }

    /// One summary line (only under the gate), plus a warn when the gate is
    /// on but nothing was kept — the model is then serving every dense
    /// projection from its materialized BF16 copy, which must be visible.
    pub(super) fn log(&self) {
        if !self.families.any() {
            return;
        }
        tracing::info!(
            "EXL3 native dense: {} GDN + {} full-attention layers serve their routed \
             projections from packed trellis (gates: gdn={} attn={}; routed leaves: \
             gdn={:?} attn={:?})",
            self.gdn_layers,
            self.attn_layers,
            self.families.gdn,
            self.families.attn,
            Exl3DenseFamily::Gdn.leaves(),
            Exl3DenseFamily::Attn.leaves(),
        );
        if self.gdn_layers + self.attn_layers == 0 {
            tracing::warn!(
                "ATLAS_EXL3_NATIVE_DENSE=1 but NO layer family was kept packed — every \
                 GDN/attention projection is serving from its materialized BF16 copy \
                 (not an EXL3 checkpoint, or every family fell outside the K in {{2,4}} \
                 envelope; see the materialize-pass warnings)"
            );
        }
    }
}
