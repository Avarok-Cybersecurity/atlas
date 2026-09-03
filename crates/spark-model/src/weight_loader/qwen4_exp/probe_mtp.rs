// SPDX-License-Identifier: AGPL-3.0-only

//! Pre-flight audit of the `mtp.*` namespace, BY SHAPE.
//!
//! Split from `probe.rs` for the 500-LoC cap; the main-model audit next door
//! is the model this one is built on. The difference is that this one asserts
//! SHAPES rather than presence, because the MTP block's one genuine departure
//! from a mirror of a main layer is a pair of norms at DIFFERENT widths — and
//! a wrong width there is not a crash, it is plausible-wrong activations.
//!
//! It also resolves the routed-expert layout explicitly. Two real checkpoint
//! revisions ship two different layouts (fused BF16 stacks; 512 per-expert
//! NVFP4 triples) and `load_moe_qwen35` picks between them silently, so
//! "neither" and "both" are refusals here rather than a branch taken quietly
//! halfway through a multi-gigabyte upload.

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::weights::WeightStore;

/// The MTP block's single decoder layer. A top-level namespace, NOT under
/// `weight_prefix` — `config.layer_prefix(i)` can never produce it, which is
/// why the prefix comes down as a literal.
pub const MTP_LAYER_PREFIX: &str = "mtp.layers.0";

/// Which of the two shipped routed-expert layouts the MTP MoE uses.
///
/// Both are real: `RadixArk/...-NVFP4` ships FUSED BF16 stacks, the Inferact
/// re-quant ships 512 PER-EXPERT NVFP4 triples. `load_moe_qwen35` picks the
/// branch itself, but it picks it SILENTLY — so the layout is resolved here
/// and refused when it is neither or both, rather than discovered halfway
/// through a 5 GB upload.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum MtpExpertLayout {
    /// Neither `experts.gate_up_proj` nor `experts.{e}.gate_proj.weight`.
    #[default]
    Neither,
    /// `mtp.layers.0.mlp.experts.{gate_up_proj,down_proj}` stacks.
    Fused,
    /// `mtp.layers.0.mlp.experts.{e}.{gate,up,down}_proj.weight`.
    PerExpert,
    /// Both — ambiguous, and `is_fused` would pick one without saying so.
    Both,
}

/// What the store holds under `mtp.*`, checked by SHAPE rather than presence.
#[derive(Debug, Default, Clone)]
pub struct MtpNamespaceReport {
    /// Any `mtp.`-prefixed tensor at all. False is the normal state: the
    /// serve path skips uploading them unless MTP is armed.
    pub any_tensors: bool,
    pub expert_layout: MtpExpertLayout,
    /// The expert index actually probed (first LOCAL under EP, not 0).
    pub first_local_expert: usize,
    /// Tensors the module loader reads that the store does not have.
    pub missing: Vec<String>,
    /// Tensors present at the wrong shape, with both widths named.
    pub shape_errors: Vec<String>,
    /// Tensors present that must NOT be.
    pub unexpected: Vec<String>,
}

/// Accumulates presence + shape failures instead of bailing on the first, so
/// the operator sees every problem at once rather than one per re-run.
struct Chk<'a> {
    store: &'a WeightStore,
    missing: Vec<String>,
    shape_errors: Vec<String>,
}

impl Chk<'_> {
    fn need(&mut self, name: String, want: &[usize]) {
        match self.store.get(&name) {
            Err(_) => self.missing.push(name),
            Ok(t) => {
                if t.shape != want {
                    self.shape_errors.push(format!(
                        "{name}: checkpoint shape {:?}, loader expects {want:?}",
                        t.shape
                    ));
                }
            }
        }
    }
}

/// Pre-flight audit of the `mtp.*` namespace, by shape.
///
/// Must run BEFORE the first `build_moe`: `load_moe_qwen35` `gpu.free()`s the
/// fused source tensors after quantizing them, leaving the store holding freed
/// pointers for `experts.gate_up_proj` / `experts.down_proj`. An audit that ran
/// afterwards would be reading dangling allocations.
pub fn audit_mtp_namespace(store: &WeightStore, config: &ModelConfig) -> MtpNamespaceReport {
    let h = config.hidden_size;
    let hc_h = config.hc_mult * h;
    let hd = config.head_dim;
    let lp = MTP_LAYER_PREFIX;

    let mut c = Chk {
        store,
        missing: Vec::new(),
        shape_errors: Vec::new(),
    };
    let mut unexpected: Vec<String> = Vec::new();
    let any_tensors = store.names().any(|n| n.starts_with("mtp."));

    // ── Attention. Byte-identical to a main full-attention layer, including
    // the two traps: q_proj is DOUBLE width (`output_gate_type: sigmoid`
    // concatenates the query and its sigmoid gate) while o_proj's input is
    // SINGLE width. Sizing either from the other silently over- or
    // under-reads, and no name check catches it.
    let ap = format!("{lp}.self_attn");
    let q_rows = 2 * config.num_attention_heads * hd;
    let kv_rows = config.num_key_value_heads * hd;
    c.need(format!("{ap}.q_proj.weight"), &[q_rows, h]);
    c.need(format!("{ap}.k_proj.weight"), &[kv_rows, h]);
    c.need(format!("{ap}.v_proj.weight"), &[kv_rows, h]);
    c.need(
        format!("{ap}.o_proj.weight"),
        &[h, config.num_attention_heads * hd],
    );
    c.need(format!("{ap}.q_norm.weight"), &[hd]);
    c.need(format!("{ap}.k_norm.weight"), &[hd]);

    // ── QSA indexer. `(index_n_heads + 1) * index_head_dim` rows: the +1 is
    // the single indexer KV head (`indexer_kv_heads: 1`), which the config
    // parser does not carry as a field.
    if config.index_topk > 0 {
        let ip = format!("{ap}.indexer");
        let idx_rows = (config.index_n_heads + 1) * config.index_head_dim;
        let ihd = config.index_head_dim;
        c.need(format!("{ip}.index_qk_proj.weight"), &[idx_rows, h]);
        c.need(format!("{ip}.q_layernorm.weight"), &[ihd]);
        c.need(format!("{ip}.k_layernorm.weight"), &[ihd]);
    }

    // ── The two per-layer mHC sites, four tensors each, plus the module's OWN
    // collapse-only mixer — which has only THREE. A `block_inject_weight` on
    // the mixer would mean it is not collapse-only and
    // `hc::load_head_at(.., with_inject = false)` is reading the wrong model.
    if config.hc_mult > 0 {
        let r = config.hc_lowrank;
        for site in ["attn_hyper_connection", "mlp_hyper_connection"] {
            let sp = format!("{lp}.{site}");
            c.need(format!("{sp}.hc_norm.weight"), &[hc_h]);
            c.need(format!("{sp}.input_mix_weight_down.weight"), &[r, hc_h]);
            c.need(format!("{sp}.input_mix_weight_up.weight"), &[hc_h, r]);
            c.need(
                format!("{sp}.block_inject_weight.weight"),
                &[config.hc_mult, hc_h],
            );
        }
        let mx = "mtp.hyper_connection_mixer";
        c.need(format!("{mx}.hc_norm.weight"), &[hc_h]);
        c.need(format!("{mx}.input_mix_weight_down.weight"), &[r, hc_h]);
        c.need(format!("{mx}.input_mix_weight_up.weight"), &[hc_h, r]);
        if store.contains(&format!("{mx}.block_inject_weight.weight")) {
            unexpected.push(format!(
                "{mx}.block_inject_weight.weight: the module mixer is built \
                 use_combine=False and must only collapse"
            ));
        }
    }

    // ── Router + shared expert. The MTP block routes over the FULL expert set,
    // top-k, exactly like a main layer — it is a second model, not a light head.
    let mp = format!("{lp}.mlp");
    let si = config.shared_expert_intermediate_size;
    c.need(format!("{mp}.gate.weight"), &[config.num_experts, h]);
    c.need(format!("{mp}.shared_expert.gate_proj.weight"), &[si, h]);
    c.need(format!("{mp}.shared_expert.up_proj.weight"), &[si, h]);
    c.need(format!("{mp}.shared_expert.down_proj.weight"), &[h, si]);
    c.need(format!("{mp}.shared_expert_gate.weight"), &[1, h]);

    // ── The four MTP-ONLY tensors — the one place this namespace is not a
    // mirror of a main layer.
    //
    // THE WIDTHS ARE ASYMMETRIC AND THAT IS THE POINT:
    //   mtp.pre_fc_norm_embedding [hidden]           = 2560
    //   mtp.pre_fc_norm_hidden    [hc_mult * hidden] = 10240
    // The second norms the FOUR-STREAM mHC highway, not a collapsed hidden. A
    // loader that allocates both at `hidden_size` reads 2560 of 10240 elements
    // and produces plausible-wrong activations with nothing in the log.
    // `fc_embedding` and `fc_hidden` are both [hidden, hidden] — square, so
    // swapping THEM is silent too; only their names distinguish them.
    c.need("mtp.pre_fc_norm_embedding.weight".into(), &[h]);
    c.need("mtp.pre_fc_norm_hidden.weight".into(), &[hc_h]);
    c.need("mtp.fc_embedding.weight".into(), &[h, h]);
    c.need("mtp.fc_hidden.weight".into(), &[h, h]);

    // ── Routed experts: exactly one of the two shipped layouts. The
    // per-expert probe uses the first LOCAL expert, not a hardcoded 0 — the
    // same `weight_map::nemotron` idiom the main-model audit uses.
    let first_local = (0..config.num_experts)
        .find(|e| config.is_local_expert(*e))
        .unwrap_or(0);
    let fused = store.contains(&format!("{mp}.experts.gate_up_proj"))
        && store.contains(&format!("{mp}.experts.down_proj"));
    let per_expert = store.contains(&format!("{mp}.experts.{first_local}.gate_proj.weight"));
    let expert_layout = match (fused, per_expert) {
        (true, true) => MtpExpertLayout::Both,
        (true, false) => MtpExpertLayout::Fused,
        (false, true) => MtpExpertLayout::PerExpert,
        (false, false) => MtpExpertLayout::Neither,
    };
    if expert_layout == MtpExpertLayout::Fused {
        // BF16 stacks: [E, 2*moe_inter, hidden] and [E, hidden, moe_inter].
        // The per-expert arm is deliberately NOT shape-checked: those tensors
        // are NVFP4-packed, so their on-disk width is half the logical one and
        // the packing is the MoE loader's business, not this audit's.
        let mi = config.moe_intermediate_size;
        let e = config.num_experts;
        c.need(format!("{mp}.experts.gate_up_proj"), &[e, 2 * mi, h]);
        c.need(format!("{mp}.experts.down_proj"), &[e, h, mi]);
    }

    MtpNamespaceReport {
        any_tensors,
        expert_layout,
        first_local_expert: first_local,
        missing: c.missing,
        shape_errors: c.shape_errors,
        unexpected,
    }
}

impl MtpNamespaceReport {
    pub fn log(&self) {
        tracing::info!(
            "qwen4_exp MTP namespace: tensors={} experts={:?} (probed expert {}), \
             {} missing, {} shape mismatches, {} unexpected",
            self.any_tensors,
            self.expert_layout,
            self.first_local_expert,
            self.missing.len(),
            self.shape_errors.len(),
            self.unexpected.len(),
        );
        for m in &self.missing {
            tracing::warn!("qwen4_exp MTP: missing {m}");
        }
        for s in &self.shape_errors {
            tracing::warn!("qwen4_exp MTP: {s}");
        }
        for u in &self.unexpected {
            tracing::warn!("qwen4_exp MTP: unexpected {u}");
        }
    }

    /// Refuse before building anything.
    ///
    /// Takes `config` (unlike `NamespaceReport::ensure_loadable` next door)
    /// because the EP refusal is a config property, not a store one.
    pub fn ensure_loadable(&self, config: &ModelConfig) -> Result<()> {
        if config.num_mtp_modules == 0 {
            // Nothing declared, nothing to load — an empty report is correct.
            return Ok(());
        }
        ensure!(
            config.ep_world_size <= 1,
            "qwen4_exp MTP: ep_world_size={} but the MTP MoE has no \
             force_all_experts path — `load_moe_qwen35` honours \
             `is_local_expert`, while the weight upload never shards `mtp.*`, \
             so a rank>0 draft would route into NULL experts. Serve without EP \
             or leave MTP off.",
            config.ep_world_size,
        );
        ensure!(
            self.any_tensors,
            "qwen4_exp MTP: num_mtp_modules={} but the store holds no `mtp.*` \
             tensors at all",
            config.num_mtp_modules,
        );
        ensure!(
            self.missing.is_empty(),
            "qwen4_exp MTP: {} tensor(s) the module loader reads are absent: {}",
            self.missing.len(),
            self.missing.join(", "),
        );
        ensure!(
            self.shape_errors.is_empty(),
            "qwen4_exp MTP: {} tensor(s) at unexpected shapes: {}",
            self.shape_errors.len(),
            self.shape_errors.join("; "),
        );
        ensure!(
            self.unexpected.is_empty(),
            "qwen4_exp MTP: {} tensor(s) present that this architecture should \
             not have: {}",
            self.unexpected.len(),
            self.unexpected.join("; "),
        );
        match self.expert_layout {
            MtpExpertLayout::Fused | MtpExpertLayout::PerExpert => Ok(()),
            MtpExpertLayout::Neither => anyhow::bail!(
                "qwen4_exp MTP: no routed experts — neither the FUSED \
                 `{MTP_LAYER_PREFIX}.mlp.experts.gate_up_proj`/`.down_proj` \
                 stacks nor a PER-EXPERT \
                 `{MTP_LAYER_PREFIX}.mlp.experts.{}.gate_proj.weight`. The MoE \
                 would route into nothing.",
                self.first_local_expert,
            ),
            MtpExpertLayout::Both => anyhow::bail!(
                "qwen4_exp MTP: BOTH expert layouts present under \
                 `{MTP_LAYER_PREFIX}.mlp.experts.*` — `load_moe_qwen35` would \
                 silently pick the fused branch. Refusing rather than guessing \
                 which one the checkpoint means."
            ),
        }
    }
}

#[cfg(test)]
#[path = "probe_mtp_tests.rs"]
mod probe_mtp_tests;
