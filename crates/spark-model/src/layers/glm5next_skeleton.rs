// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3-Flash **45-layer text-model skeleton** — Slice 9.
//!
//! The topology, the residual/norm/mHC wiring, the structural weight contract and the state
//! plumbing of the text stack, in one place that can be checked against the checkpoint without
//! a GPU and without executing anything.
//!
//! Scope is deliberate: **no MoE, no dense FFN, no MTP speculation, no forward pass.** The MLP
//! site is represented as a hole in the residual plan (`ResidualStep::Mlp`) with its kind
//! recorded, so the shape of what is missing is explicit rather than implied.
//!
//! # Why a skeleton is its own artifact
//!
//! "Every attention block executes" and "the model is assembled correctly" are different
//! statements. Slices 1–8 proved the first for all 34 KDA and all 12 DSA blocks. A stack that
//! binds every tensor and still orders its layers wrong, or hangs the hyper-connection off the
//! wrong site, produces perfectly plausible output — the failure mode this campaign keeps
//! paying for. So the ordering, the wiring and the binding are asserted as data.
//!
//! # Measured facts this module encodes (checkpoint `LibertAIDAI/GLM-5.3-Flash-NVFP4@9e0d74e3`)
//!
//! * The structural (non-MLP) surface is **1,047 tensors** in exactly **3 signatures**:
//!   34 KDA layers × 23, 11 DSA layers × 22, layer 45 × 20, plus 3 non-layer tensors.
//! * 🪤 **Layer 45 has NO hyper-connection.** All 270 `hc_*` tensors live on layers 0..=44.
//!   A skeleton that gives the MTP layer an `attn_hc`/`ffn_hc` looks for six tensors that do
//!   not exist.
//! * 🪤 **The final collapse is an UNWEIGHTED MEAN.** `Glm5NextTextHyperHead` has no
//!   parameters and the checkpoint carries **zero** `hc_head` tensors — unlike DeepSeek-V4,
//!   whose `hc_head` is a learned sigmoid-weighted sum. Atlas's `hc_head` CUDA kernel is the
//!   DeepSeek one; for GLM it is **ADAPT, not REUSE**.
//! * 🪤 **`hc_*_fn` is BF16 on disk**; only `base`/`scale` are F32. The `hc_pre` kernel takes
//!   `f32*`, so binding must upcast.
//! * Layers **0..=2** carry a dense MLP (`first_k_dense_replace = 3`); 42 text layers and the
//!   MTP layer route to experts.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};

/// Which mixer a layer runs. Narrower than [`LayerType`] on purpose: the skeleton refuses the
/// kinds GLM-5.3 does not have rather than carrying them as unreachable arms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mixer {
    /// Kimi Delta Attention — recurrent, carries state, no KV cache.
    Kda,
    /// NoPE sparse MLA behind a kpool top-k indexer — attends a KV cache.
    Dsa,
}

/// Which MLP a layer runs. Not executed by this slice; recorded so the hole is named.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mlp {
    Dense,
    RoutedMoe,
}

/// One decoder layer's structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SkeletonLayer {
    pub index: usize,
    pub mixer: Mixer,
    pub mlp: Mlp,
    /// False only for the MTP layer — see the module trap note.
    pub hyper_connection: bool,
    /// True for layer 45. It is NOT part of the text stack.
    pub is_mtp: bool,
}

/// One step of a layer's residual path, in execution order.
///
/// This is the wiring, as data. HF's `Glm5NextTextDecoderLayer::forward` runs, per site:
/// `hc_pre` → norm → sublayer → `hc_post`, twice; the residual streams entering `hc_pre` are the
/// ones `hc_post` mixes through `comb`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidualStep {
    /// Snapshot the `hc_mult` streams as the residual for this site's `hc_post`.
    SaveResidual,
    /// `hc_pre`: collapse the streams to one sequence, emit `post`/`comb`.
    HcPre(Site),
    /// RMSNorm on the collapsed sequence.
    Norm(&'static str),
    /// The attention mixer.
    Mixer,
    /// The MLP site. **Not implemented by this slice.**
    Mlp,
    /// `hc_post`: `out[j] = post[j]*block_out + Σ_i comb[i][j]*residual[i]`.
    HcPost(Site),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Site {
    Attn,
    Ffn,
}

/// What the model does once the 45 text layers are done.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinalStep {
    /// 🪤 UNWEIGHTED mean over the `hc_mult` streams. Not DeepSeek-V4's learned collapse.
    HyperHeadMean,
    Norm(&'static str),
    LmHead,
}

/// Which cache a layer needs. KDA and DSA are mutually exclusive here, and admission needs
/// BOTH kinds satisfied — a KDA slot is not interchangeable with KV blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateKind {
    /// Recurrent `[H/ep, 128, 128]` fp32 state + a bf16 causal-conv window.
    KdaRecurrent,
    /// Paged KV blocks + the indexer's own key state.
    SparseKv,
}

#[derive(Debug, Clone)]
pub struct Glm5NextTextSkeleton {
    pub hidden_size: usize,
    pub hc_mult: usize,
    /// Text stack, indices 0..=44. Length is `num_hidden_layers` and never includes MTP.
    pub layers: Vec<SkeletonLayer>,
    /// Layer 45. Held separately so every "iterate the text stack" loop keeps meaning what it
    /// says — the same discipline `ModelConfig::mtp_layer_types` established in Slice 8.
    pub mtp: Option<SkeletonLayer>,
}

const NON_LAYER_TENSORS: [&str; 3] = [
    "model.language_model.embed_tokens.weight",
    "model.language_model.norm.weight",
    "lm_head.weight",
];

/// The 15 `self_attn` tensors of a KDA block (Slice 7: one signature across all 34).
const KDA_ATTN: [&str; 15] = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.b_proj.weight",
    "self_attn.f_a_proj.weight",
    "self_attn.f_b_proj.weight",
    "self_attn.g_a_proj.weight",
    "self_attn.g_b_proj.weight",
    "self_attn.q_conv1d.weight",
    "self_attn.k_conv1d.weight",
    "self_attn.v_conv1d.weight",
    "self_attn.A_log",
    "self_attn.dt_bias",
    "self_attn.o_norm.weight",
    "self_attn.o_proj.weight",
];

/// The 14 `self_attn` tensors of a DSA block (Slice 8: one signature across all 12,
/// layer 45 included).
///
/// 🪤 `indexer.k_norm` is an `nn.LayerNorm` and carries a **bias**. Every other norm in this
/// model is a bias-free RMSNorm; a binder that takes only `.weight` drops it silently.
const DSA_ATTN: [&str; 14] = [
    "self_attn.q_a_proj.weight",
    "self_attn.q_a_layernorm.weight",
    "self_attn.q_b_proj.weight",
    "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.kv_a_layernorm.weight",
    "self_attn.kv_b_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.indexer.wq_b.weight",
    "self_attn.indexer.wk.weight",
    "self_attn.indexer.k_norm.weight",
    "self_attn.indexer.k_norm.bias",
    "self_attn.indexer.weights_proj.weight",
    "self_attn.indexer.index_kpool_compress_ape",
    "self_attn.indexer.index_kpool_compress_gate",
];

const LAYER_NORMS: [&str; 2] = ["input_layernorm.weight", "post_attention_layernorm.weight"];

const HC_PARAMS: [&str; 6] = [
    "hc_attn_fn",
    "hc_attn_base",
    "hc_attn_scale",
    "hc_ffn_fn",
    "hc_ffn_base",
    "hc_ffn_scale",
];

/// MTP-head tensors. A head, not attention — layer 45's mixer is a plain DSA block, so MTP
/// needs no attention implementation of its own.
const MTP_HEAD: [&str; 4] = [
    "eh_proj.weight",
    "enorm.weight",
    "hnorm.weight",
    "shared_head.norm.weight",
];

fn qualify(layer: usize, leaf: &str) -> String {
    format!("model.language_model.layers.{layer}.{leaf}")
}

impl Glm5NextTextSkeleton {
    /// Derive the topology from a parsed config. Every kind is read, never defaulted: an
    /// unexpected `LayerType` is a hard error, because a sparse layer silently bound as dense
    /// attends the whole cache and produces plausible output.
    pub fn from_config(cfg: &ModelConfig) -> Result<Self> {
        if cfg.model_type != "glm5_next" {
            bail!(
                "Glm5NextTextSkeleton built from a {:?} config",
                cfg.model_type
            );
        }
        if cfg.layer_types.len() != cfg.num_hidden_layers {
            bail!(
                "layer_types has {} entries, num_hidden_layers is {}",
                cfg.layer_types.len(),
                cfg.num_hidden_layers
            );
        }
        if cfg.hc_mult == 0 {
            bail!("glm5_next skeleton needs hc_mult > 0; got 0 (mHC is not optional here)");
        }
        let dense: BTreeSet<usize> = cfg.mlp_only_layers.iter().copied().collect();

        let mixer_of = |t: LayerType, i: usize| -> Result<Mixer> {
            Ok(match t {
                LayerType::LinearAttention => Mixer::Kda,
                LayerType::SparseAttention => Mixer::Dsa,
                other => {
                    bail!("layer {i}: GLM-5.3-Flash has no {other:?} layers; refusing to bind one")
                }
            })
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for (i, t) in cfg.layer_types.iter().enumerate() {
            layers.push(SkeletonLayer {
                index: i,
                mixer: mixer_of(*t, i)?,
                mlp: if dense.contains(&i) {
                    Mlp::Dense
                } else {
                    Mlp::RoutedMoe
                },
                hyper_connection: true,
                is_mtp: false,
            });
        }

        // MTP sits PAST the text stack and is reached through `layer_type_at`, never appended.
        let mtp = match cfg.mtp_layer_types.len() {
            0 => None,
            1 => {
                let i = cfg.num_hidden_layers;
                Some(SkeletonLayer {
                    index: i,
                    mixer: mixer_of(cfg.mtp_layer_types[0], i)?,
                    mlp: Mlp::RoutedMoe,
                    // 🪤 Measured: zero `hc_*` tensors on layer 45.
                    hyper_connection: false,
                    is_mtp: true,
                })
            }
            n => bail!("glm5_next skeleton expects 0 or 1 MTP layers, config declares {n}"),
        };

        Ok(Self {
            hidden_size: cfg.hidden_size,
            hc_mult: cfg.hc_mult,
            layers,
            mtp,
        })
    }

    /// Text stack plus the MTP layer, in checkpoint index order.
    pub fn all_layers(&self) -> Vec<SkeletonLayer> {
        let mut v = self.layers.clone();
        v.extend(self.mtp);
        v
    }

    /// The structural (non-MLP, non-MoE) tensors this layer needs, fully qualified.
    pub fn structural_tensors(&self, l: &SkeletonLayer) -> Vec<String> {
        let mut v: Vec<String> = match l.mixer {
            Mixer::Kda => KDA_ATTN.iter().map(|t| qualify(l.index, t)).collect(),
            Mixer::Dsa => DSA_ATTN.iter().map(|t| qualify(l.index, t)).collect(),
        };
        v.extend(LAYER_NORMS.iter().map(|t| qualify(l.index, t)));
        if l.hyper_connection {
            v.extend(HC_PARAMS.iter().map(|t| qualify(l.index, t)));
        }
        if l.is_mtp {
            v.extend(MTP_HEAD.iter().map(|t| qualify(l.index, t)));
        }
        v
    }

    /// Every structural tensor the whole skeleton needs, including the three non-layer ones.
    pub fn structural_tensor_set(&self) -> BTreeSet<String> {
        let mut s: BTreeSet<String> = NON_LAYER_TENSORS.iter().map(|t| t.to_string()).collect();
        for l in self.all_layers() {
            s.extend(self.structural_tensors(&l));
        }
        s
    }

    /// The residual/norm/mHC wiring for one layer, in execution order.
    ///
    /// The MTP layer has no hyper-connection, so its residual path is the ordinary
    /// `x = x + sublayer(norm(x))` shape and the mHC steps drop out entirely.
    pub fn residual_plan(&self, l: &SkeletonLayer) -> Vec<ResidualStep> {
        let mut p = Vec::new();
        for (site, norm, sub) in [
            (Site::Attn, LAYER_NORMS[0], ResidualStep::Mixer),
            (Site::Ffn, LAYER_NORMS[1], ResidualStep::Mlp),
        ] {
            if l.hyper_connection {
                p.push(ResidualStep::SaveResidual);
                p.push(ResidualStep::HcPre(site));
                p.push(ResidualStep::Norm(norm));
                p.push(sub);
                p.push(ResidualStep::HcPost(site));
            } else {
                p.push(ResidualStep::SaveResidual);
                p.push(ResidualStep::Norm(norm));
                p.push(sub);
            }
        }
        p
    }

    /// What runs after the last text layer.
    pub fn final_plan(&self) -> [FinalStep; 3] {
        [
            FinalStep::HyperHeadMean,
            FinalStep::Norm("model.language_model.norm.weight"),
            FinalStep::LmHead,
        ]
    }

    /// Per-layer cache requirement. Admission needs every kind satisfied at once.
    pub fn state_plan(&self) -> BTreeMap<usize, StateKind> {
        self.all_layers()
            .iter()
            .map(|l| {
                (
                    l.index,
                    match l.mixer {
                        Mixer::Kda => StateKind::KdaRecurrent,
                        Mixer::Dsa => StateKind::SparseKv,
                    },
                )
            })
            .collect()
    }

    /// Layers whose recurrent state must be carried between steps.
    pub fn kda_state_layers(&self) -> Vec<usize> {
        self.state_plan()
            .into_iter()
            .filter(|(_, k)| *k == StateKind::KdaRecurrent)
            .map(|(i, _)| i)
            .collect()
    }

    /// Layers that consume paged KV blocks.
    pub fn kv_cache_layers(&self) -> Vec<usize> {
        self.state_plan()
            .into_iter()
            .filter(|(_, k)| *k == StateKind::SparseKv)
            .map(|(i, _)| i)
            .collect()
    }

    /// The per-sequence state contract, from the config's real geometry.
    ///
    /// 🪤 `kda_recurrent` is **fp32 and not negotiable**: HF casts the recurrent state to
    /// float32 and vLLM hardcodes `kda_state_dtype`, so `--ssm-h-dtype f16` is unavailable.
    /// Sizing it as bf16 halves the number and is wrong.
    pub fn state_budget(&self, cfg: &ModelConfig, num_spec: usize) -> StateBudget {
        let kda_layers = self.kda_state_layers().len();
        let kv_layers = self.layers.iter().filter(|l| l.mixer == Mixer::Dsa).count();
        let heads = cfg.linear_num_value_heads.max(1);
        let hd = cfg.linear_value_head_dim.max(1);
        let conv_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim * 2
            + cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        StateBudget {
            kda_recurrent: kda_layers * heads * hd * hd * 4,
            kda_conv: kda_layers * conv_dim * (cfg.linear_conv_kernel_dim - 1 + num_spec) * 2,
            dsa_kv_per_token: kv_layers * cfg.kv_lora_rank * 2,
            // 🔴 TWO buffers of `index_head_dim` BF16 (`k_normed` AND the compress `gate`)
            // plus the 1 B validity flag — `Glm5NextDsaState::alloc`. Counting only
            // `k_normed` halved this, which did not bite while the cache was pinned at a
            // fixed 16,384 rows and does bite the moment it scales with --max-seq-len.
            dsa_indexer_per_token: kv_layers * (cfg.index_head_dim * 2 * 2 + 1),
            mhc_highway_per_token: self.hc_mult * self.hidden_size * 4,
            moe_routing_per_token: cfg.num_experts * 4 + cfg.num_experts_per_tok * 8,
        }
    }

    /// Account the skeleton's structural contract against a checkpoint's tensor names.
    ///
    /// `available` is the FULL name list; MLP/MoE and vision names are expected to be present
    /// and are reported as `deferred`, not as errors — this slice does not bind them. Anything
    /// structural that is missing, and any non-MLP text tensor the skeleton did not ask for,
    /// is a hard failure. Zero unknown, zero silent skips.
    pub fn account(&self, available: &BTreeSet<String>) -> StructuralAccounting {
        let required = self.structural_tensor_set();
        let mut missing = Vec::new();
        for r in &required {
            if !available.contains(r) {
                missing.push(r.clone());
            }
        }
        let mut unexpected = Vec::new();
        let mut deferred = 0usize;
        for a in available {
            if required.contains(a) {
                continue;
            }
            let is_mlp = a.contains(".mlp.");
            let is_vision = a.starts_with("model.visual.") || a.starts_with("model.vision");
            if is_mlp || is_vision {
                deferred += 1;
            } else {
                unexpected.push(a.clone());
            }
        }
        StructuralAccounting {
            required: required.len(),
            bound: required.len() - missing.len(),
            missing,
            unexpected,
            deferred,
        }
    }
}

/// Per-sequence state contract (Slice 11 gate 3).
///
/// Derived from the checkpoint config, not from a formula chosen to look tidy. Every field is
/// bytes for ONE sequence at EP=1; `per_rank` halves only what EP actually shards.
#[derive(Debug, Clone, Copy)]
pub struct StateBudget {
    /// KDA recurrent state — `[heads, head_dim, head_dim]` **fp32 mandatory** (HF casts to
    /// float32 and vLLM hardcodes it), 34 layers. FIXED: does not grow with sequence length.
    pub kda_recurrent: usize,
    /// KDA causal-conv window, bf16, `conv_dim x (kernel - 1 + num_spec)`. FIXED.
    pub kda_conv: usize,
    /// DSA MLA KV per TOKEN across the 11 text layers — `kv_lora_rank` bf16, NoPE so there is
    /// no rope section. GROWS with sequence length.
    pub dsa_kv_per_token: usize,
    /// Indexer key + gate state per TOKEN across the 11 text layers. GROWS.
    /// 🪤 REPLICATED, not sharded: the indexer is `DsaShard::Replicated` (`dsa/tp.rs`), so
    /// [`Self::per_rank`] must not divide it.
    pub dsa_indexer_per_token: usize,
    /// mHC highway per TOKEN — `hc_mult x hidden` fp32 in Atlas (bf16 in HF; see the OPEN
    /// highway-dtype item). Activation-lifetime, not persistent across steps.
    pub mhc_highway_per_token: usize,
    /// MoE routing scratch per token: logits + top-k ids + weights.
    pub moe_routing_per_token: usize,
}

impl StateBudget {
    /// Fixed (sequence-length-independent) bytes per sequence.
    pub fn fixed(&self) -> usize {
        self.kda_recurrent + self.kda_conv
    }
    /// Bytes that grow with every token of context.
    pub fn per_token(&self) -> usize {
        self.dsa_kv_per_token + self.dsa_indexer_per_token
    }
    /// Total persistent state for a sequence of `tokens`.
    pub fn for_sequence(&self, tokens: usize) -> usize {
        self.fixed() + tokens * self.per_token()
    }
    /// EP shards the KDA head dimension and the KV heads; the DSA INDEXER cache, the mHC
    /// highway and the routing scratch are replicated. Only what EP actually shards is
    /// divided.
    pub fn per_rank(&self, ep: usize) -> StateBudget {
        StateBudget {
            kda_recurrent: self.kda_recurrent / ep,
            kda_conv: self.kda_conv / ep,
            dsa_kv_per_token: self.dsa_kv_per_token / ep,
            // 🔴 NOT divided: the indexer's `wk`/gate projections are replicated on every
            // rank (`DsaShard::Replicated`), so every rank holds the whole cache.
            dsa_indexer_per_token: self.dsa_indexer_per_token,
            mhc_highway_per_token: self.mhc_highway_per_token,
            moe_routing_per_token: self.moe_routing_per_token,
        }
    }
}

#[derive(Debug)]
pub struct StructuralAccounting {
    pub required: usize,
    pub bound: usize,
    /// Structural tensors the checkpoint does not have. MUST be empty.
    pub missing: Vec<String>,
    /// Text tensors that are neither structural nor MLP/vision — i.e. names the skeleton was
    /// never taught. MUST be empty.
    pub unexpected: Vec<String>,
    /// MLP / MoE / vision tensors, deliberately not bound by this slice.
    pub deferred: usize,
}

impl StructuralAccounting {
    pub fn is_complete(&self) -> bool {
        self.missing.is_empty() && self.unexpected.is_empty() && self.bound == self.required
    }
}
