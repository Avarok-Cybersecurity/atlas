// SPDX-License-Identifier: AGPL-3.0-only

//! Pre-flight namespace audit for `qwen4_exp`.
//!
//! Runs BEFORE any upload and reports what the store actually holds, grouped
//! the way `bench/qwen4_exp/ns_audit.py` groups it. The point is that a
//! missing family should be named at load time — a loader that silently skips
//! one produces a model that runs and is wrong, which on this codebase has
//! repeatedly cost more time than a hard failure would have.

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::weights::WeightStore;

/// What the store holds, per family the loader cares about.
#[derive(Debug, Default, Clone, Copy)]
pub struct NamespaceReport {
    pub layers: usize,
    pub gdn_layers: usize,
    pub attn_layers: usize,
    pub expert_tensors: usize,
    pub hc_tensors: usize,
    pub indexer_tensors: usize,
    pub ple_shards: usize,
    pub vision_tensors: usize,
    pub has_embed: bool,
    pub has_lm_head: bool,
    /// Per-layer norms, which this architecture should NOT have.
    pub stray_layer_norms: usize,
}

pub fn audit_namespace(store: &WeightStore, config: &ModelConfig) -> NamespaceReport {
    let mut r = NamespaceReport {
        layers: config.num_hidden_layers,
        ..Default::default()
    };
    let pfx = if config.weight_prefix.is_empty() {
        "model".to_string()
    } else {
        config.weight_prefix.clone()
    };
    r.has_embed = store.contains(&format!("{pfx}.embed_tokens.weight"));
    // A kept-packed EXL3 lm_head (ATLAS_EXL3_NATIVE=1) counts: the native
    // head serves it without a dense `.weight` ever existing.
    r.has_lm_head = store.contains("lm_head.weight")
        || spark_runtime::weights::exl3::is_exl3_linear(store, "lm_head");

    // Like the lm_head and the experts, a kept-packed EXL3 dense family
    // (ATLAS_EXL3_NATIVE_DENSE=1) counts: the native GDN / attention arms
    // serve it without a dense `.weight` ever existing. The predicate is the
    // loader's own (`exl3_dense_family_kept`), so the audit and the arms
    // cannot disagree about which layers are native.
    use crate::weight_map::{Exl3DenseFamily, exl3_dense_family_kept};
    for i in 0..config.num_hidden_layers {
        let lp = config.layer_prefix(i);
        if store.contains(&format!("{lp}.linear_attn.in_proj_qkv.weight"))
            || exl3_dense_family_kept(store, &lp, Exl3DenseFamily::Gdn)
        {
            r.gdn_layers += 1;
        }
        if store.contains(&format!("{lp}.self_attn.q_proj.weight"))
            || exl3_dense_family_kept(store, &lp, Exl3DenseFamily::Attn)
        {
            r.attn_layers += 1;
        }
        if store.contains(&format!("{lp}.self_attn.indexer.index_qk_proj.weight")) {
            r.indexer_tensors += 1;
        }
        if store.contains(&format!("{lp}.attn_hyper_connection.hc_norm.weight")) {
            r.hc_tensors += 1;
        }
        if store.contains(&format!("{lp}.mlp_hyper_connection.hc_norm.weight")) {
            r.hc_tensors += 1;
        }
        // The FIRST LOCAL expert, not expert 0: under EP the store holds only
        // `local_expert_range()`, so probing a hardcoded 0 finds nothing on
        // every rank but 0 and the `expert_tensors == layers` check below
        // then refuses a perfectly good rank-1 load ("0 of 48 layers have
        // routed experts", measured on a 2-node EP=2 bring-up 2026-08-27).
        // Same first-local idiom as `weight_map::nemotron`.
        let first_local = (0..config.num_experts)
            .find(|e| config.is_local_expert(*e))
            .unwrap_or(0);
        // Like the lm_head above, kept-packed EXL3 experts
        // (ATLAS_EXL3_NATIVE_MOE=1) count: the native MoE path serves them
        // without a dense `.weight` ever existing.
        let first_local_gate = format!("{lp}.mlp.experts.{first_local}.gate_proj");
        if store.contains(&format!("{first_local_gate}.weight"))
            || (crate::weight_map::exl3_native_moe_enabled()
                && spark_runtime::weights::exl3::is_exl3_linear(store, &first_local_gate))
        {
            r.expert_tensors += 1;
        }
        if store.contains(&format!("{lp}.input_layernorm.weight"))
            || store.contains(&format!("{lp}.post_attention_layernorm.weight"))
        {
            r.stray_layer_norms += 1;
        }
    }

    for &l in &config.ple_layer_ids {
        let lp = config.layer_prefix(l);
        let base = format!("{lp}.ple.ple_embedding.ngram_embedding");
        let parts = if config.ngram_split_parts > 0 {
            config.ngram_split_parts
        } else {
            0
        };
        for s in 0..parts {
            if store.contains(&format!("{base}.shard_{s}.weight")) {
                r.ple_shards += 1;
            }
        }
    }

    for b in 0..64usize {
        if store.contains(&format!("model.visual.blocks.{b}.attn.qkv.weight")) {
            r.vision_tensors += 1;
        }
    }
    r
}

impl NamespaceReport {
    pub fn log(&self) {
        tracing::info!(
            "qwen4_exp namespace: {}/{} GDN + {} full-attention layers, \
             {} MoE blocks, {} hc blocks, {} indexers, {} PLE shards, \
             {} ViT blocks, embed={} lm_head={}",
            self.gdn_layers,
            self.layers,
            self.attn_layers,
            self.expert_tensors,
            self.hc_tensors,
            self.indexer_tensors,
            self.ple_shards,
            self.vision_tensors,
            self.has_embed,
            self.has_lm_head,
        );
        if self.stray_layer_norms == 0 {
            tracing::info!(
                "qwen4_exp: no per-layer input/post-attention norms, as expected — \
                 this architecture keeps normalization inside the \
                 hyper-connection blocks"
            );
        }
    }

    /// Refuse before uploading ~75 GB if the store is not what we think.
    pub fn ensure_loadable(&self) -> Result<()> {
        ensure!(
            self.has_embed && self.has_lm_head,
            "qwen4_exp: embed_tokens={} lm_head={} — checkpoint is not the \
             expected multimodal layout (is `weight_prefix` set to \
             `model.language_model`?)",
            self.has_embed,
            self.has_lm_head,
        );
        ensure!(
            self.gdn_layers + self.attn_layers == self.layers,
            "qwen4_exp: {} GDN + {} full-attention = {} but the config declares \
             {} layers — every layer must resolve to exactly one mixer",
            self.gdn_layers,
            self.attn_layers,
            self.gdn_layers + self.attn_layers,
            self.layers,
        );
        ensure!(
            self.expert_tensors == self.layers,
            "qwen4_exp: {} of {} layers have routed experts — a layer whose \
             experts silently fail to resolve would route into nothing",
            self.expert_tensors,
            self.layers,
        );
        // A stray norm means the architecture is NOT what this loader assumes
        // and the mHC reading is wrong. Better to stop than to load 75 GB
        // against a mistaken model of the model.
        ensure!(
            self.stray_layer_norms == 0,
            "qwen4_exp: {} layer(s) carry input/post-attention norms, which \
             this architecture should not have — normalization is supposed to \
             live inside the hyper-connection blocks. The mHC reading of this \
             checkpoint is wrong; refusing rather than loading on a bad premise",
            self.stray_layer_norms,
        );
        Ok(())
    }
}
