// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! DeepSeek-V4.1 **full tiny-graph forward** (`Block.forward`, `Transformer.forward`): embed,
//! expand to `hc_mult` copies, the one-hot initial pre-mix, engram before layers 1 and 4, six
//! blocks with the delayed mixes, collapse, final RMSNorm, f32 logits. Every component is the
//! CPU reference already checked on its own; this file wires them in the reference's order and
//! carries every per-layer cache and the shared slots across prefill and decode, so the golden's
//! sampled captures for layers 1 to 5 and the logits become checkable.

use super::attn::{AttnWeights, freqs_cis};
use super::compress::{
    CompAttnCfg, CompAttnRun, CompressorWeights, IndexerCfg, IndexerWeights, LayerAttnState, SharedRuntime, attention_any,
    yarn_freqs_cis,
};
use super::engram::{EngramTables, NgramHashState, embed_rows, gate_and_add, regen_bf16_matrix, regen_bf16_qk, regen_table};
use super::hc::{hc_mixes, hc_post, hc_pre, rms_norm};
use super::moe::{MoeCfg, MoeWeights, moe};
use super::{Golden, regen_param};

pub struct ModelCfg {
    pub dim: usize,
    pub hc: usize,
    pub iters: usize,
    pub hc_eps: f32,
    pub eps: f32,
    pub n_layers: usize,
    pub vocab: usize,
    pub max_seq: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub q_rank: usize,
    pub o_rank: usize,
    pub groups: usize,
    pub window: usize,
    pub ratios: Vec<usize>,
    pub kv_sources: Vec<usize>,
    pub index_sources: Vec<usize>,
    pub cand_src: i64,
    pub cand_topk_blocks: usize,
    pub cand_block: usize,
    pub index_heads: usize,
    pub index_hd: usize,
    pub index_topk: usize,
    pub rope_theta: f32,
    pub compress_rope_theta: f32,
    pub rope_factor: f32,
    pub orig_seq: usize,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub moe: MoeCfg,
}

impl ModelCfg {
    pub fn from_golden(g: &Golden) -> Self {
        let u = |k: &str| g.fixture_u64(k) as usize;
        let f = |k: &str| g.fixture_f64(k) as f32;
        let list = |k: &str| g.fixture_usize_list(k);
        let dim = u("dim");
        ModelCfg {
            dim,
            hc: u("hc_mult"),
            iters: u("hc_sinkhorn_iters"),
            hc_eps: f("hc_eps"),
            eps: f("norm_eps"),
            n_layers: u("n_layers"),
            vocab: u("vocab_size"),
            max_seq: u("max_seq_len"),
            n_heads: u("n_heads"),
            head_dim: u("head_dim"),
            rope_dim: u("rope_head_dim"),
            q_rank: u("q_lora_rank"),
            o_rank: u("o_lora_rank"),
            groups: u("o_groups"),
            window: u("window_size"),
            ratios: list("compress_ratios"),
            kv_sources: list("kv_source_layers"),
            index_sources: list("index_source_layers"),
            cand_src: g.fixture_i64("candidate_source_layer"),
            cand_topk_blocks: u("candidate_topk_blocks"),
            cand_block: u("candidate_block_size"),
            index_heads: u("index_n_heads"),
            index_hd: u("index_head_dim"),
            index_topk: u("index_topk"),
            rope_theta: f("rope_theta"),
            compress_rope_theta: f("compress_rope_theta"),
            rope_factor: f("rope_factor"),
            orig_seq: u("original_seq_len"),
            beta_fast: f("beta_fast"),
            beta_slow: f("beta_slow"),
            moe: MoeCfg {
                dim,
                inter: u("moe_inter_dim"),
                n_routed: u("n_routed_experts"),
                topk: u("n_activated_experts"),
                gate_temp: f("gate_temp"),
                norm_topk_prob: true,
                route_scale: f("route_scale"),
                swiglu_limit: f("swiglu_limit"),
            },
        }
    }

    pub fn attn_cfg(&self, layer: usize) -> CompAttnCfg {
        let ratio = self.ratios[layer];
        CompAttnCfg {
            dim: self.dim,
            n_heads: self.n_heads,
            head_dim: self.head_dim,
            rope_dim: self.rope_dim,
            q_rank: self.q_rank,
            o_rank: self.o_rank,
            groups: self.groups,
            window: self.window,
            eps: self.eps,
            ratio,
            is_kv_source: self.kv_sources.contains(&layer),
            is_index_source: self.index_sources.contains(&layer),
            is_candidate_source: self.cand_src == layer as i64,
            uses_candidates: self.cand_src >= 0 && (self.cand_src as usize) < layer,
        }
    }

    pub fn indexer_cfg(&self) -> IndexerCfg {
        IndexerCfg {
            n_heads: self.index_heads,
            index_hd: self.index_hd,
            rope_dim: self.rope_dim,
            q_rank: self.q_rank,
            dim: self.dim,
            hd: self.head_dim,
            index_topk: self.index_topk,
            cand_topk_blocks: self.cand_topk_blocks,
            cand_block: self.cand_block,
            eps: self.eps,
        }
    }
}

pub struct EngramWeights {
    pub table: Vec<f32>,
    pub wkv: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub hash_index: usize,
}

pub struct LayerWeights {
    pub hc_attn_fn: Vec<f32>,
    pub hc_attn_scale: Vec<f32>,
    pub hc_attn_base: Vec<f32>,
    pub hc_ffn_fn: Vec<f32>,
    pub hc_ffn_scale: Vec<f32>,
    pub hc_ffn_base: Vec<f32>,
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub sink: Vec<f32>,
    pub wq_a: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub wq_b: Vec<f32>,
    pub wkv: Vec<f32>,
    pub kv_norm: Vec<f32>,
    pub wo_a: Vec<f32>,
    pub wo_b: Vec<f32>,
    pub comp_wkv: Option<Vec<f32>>,
    pub comp_wgate: Option<Vec<f32>>,
    pub comp_norm: Option<Vec<f32>>,
    pub idx_wq_b: Option<Vec<f32>>,
    pub idx_weights_proj: Option<Vec<f32>>,
    pub idx_wk: Option<Vec<f32>>,
    pub idx_k_norm: Option<Vec<f32>>,
    pub gate_w: Vec<f32>,
    pub gate_bias: Vec<f32>,
    pub experts: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)>,
    pub shared: (Vec<f32>, Vec<f32>, Vec<f32>),
    pub engram: Option<EngramWeights>,
}

pub struct ModelWeights {
    pub embed: Vec<f32>,
    pub norm: Vec<f32>,
    pub head: Vec<f32>,
    pub layers: Vec<LayerWeights>,
}

impl ModelWeights {
    pub fn from_golden(g: &Golden, c: &ModelCfg, t: &EngramTables) -> Self {
        let layers = (0..c.n_layers)
            .map(|l| {
                let p = |n: &str| regen_param(g, &format!("layers.{l}.{n}"));
                let opt = |n: &str, on: bool| if on { Some(p(n)) } else { None };
                let ac = c.attn_cfg(l);
                let has_comp = ac.is_kv_source;
                let has_idx = ac.is_index_source;
                let engram = t.layer_ids.iter().position(|&x| x == l).map(|hi| EngramWeights {
                    table: regen_table(t, hi),
                    wkv: regen_bf16_matrix(&format!("layers.{l}.engram.wkv.weight"), c.dim * (c.hc + 1), t.n_hash_cols() * t.head_dim),
                    q: regen_bf16_qk(&format!("layers.{l}.engram.q_weight"), c.hc * c.dim),
                    k: regen_bf16_qk(&format!("layers.{l}.engram.k_weight"), c.hc * c.dim),
                    hash_index: hi,
                });
                LayerWeights {
                    hc_attn_fn: p("hc_attn_fn"),
                    hc_attn_scale: p("hc_attn_scale"),
                    hc_attn_base: p("hc_attn_base"),
                    hc_ffn_fn: p("hc_ffn_fn"),
                    hc_ffn_scale: p("hc_ffn_scale"),
                    hc_ffn_base: p("hc_ffn_base"),
                    attn_norm: p("attn_norm.weight"),
                    ffn_norm: p("ffn_norm.weight"),
                    sink: p("attn.attn_sink"),
                    wq_a: p("attn.wq_a.weight"),
                    q_norm: p("attn.q_norm.weight"),
                    wq_b: p("attn.wq_b.weight"),
                    wkv: p("attn.wkv.weight"),
                    kv_norm: p("attn.kv_norm.weight"),
                    wo_a: p("attn.wo_a.weight"),
                    wo_b: p("attn.wo_b.weight"),
                    comp_wkv: opt("attn.compressor.wkv.weight", has_comp),
                    comp_wgate: opt("attn.compressor.wgate.weight", has_comp && ac.ratio > 1),
                    comp_norm: opt("attn.compressor.norm.weight", has_comp),
                    idx_wq_b: opt("attn.indexer.wq_b.weight", has_idx),
                    idx_weights_proj: opt("attn.indexer.weights_proj.weight", has_idx),
                    idx_wk: opt("attn.indexer.wk.weight", has_idx && has_comp),
                    idx_k_norm: opt("attn.indexer.k_norm.weight", has_idx && has_comp),
                    gate_w: p("ffn.gate.weight"),
                    gate_bias: p("ffn.gate.bias"),
                    experts: (0..c.moe.n_routed)
                        .map(|e| (p(&format!("ffn.experts.{e}.w1.weight")), p(&format!("ffn.experts.{e}.w2.weight")), p(&format!("ffn.experts.{e}.w3.weight"))))
                        .collect(),
                    shared: (p("ffn.shared_experts.w1.weight"), p("ffn.shared_experts.w2.weight"), p("ffn.shared_experts.w3.weight")),
                    engram,
                }
            })
            .collect();
        ModelWeights { embed: regen_param(g, "embed.weight"), norm: regen_param(g, "norm.weight"), head: regen_param(g, "head.weight"), layers }
    }
}

pub struct ModelState<'a> {
    pub hash: NgramHashState<'a>,
    pub layers: Vec<LayerAttnState>,
    pub shared: SharedRuntime,
    pub fcs: Vec<Vec<(f32, f32)>>,
}

impl<'a> ModelState<'a> {
    pub fn new(c: &ModelCfg, t: &'a EngramTables) -> Self {
        ModelState {
            hash: NgramHashState::new(t, 1, c.max_seq),
            layers: (0..c.n_layers).map(|l| LayerAttnState::new(&c.attn_cfg(l), c.max_seq, c.index_hd)).collect(),
            shared: SharedRuntime::default(),
            fcs: (0..c.n_layers)
                .map(|l| {
                    if c.ratios[l] == 0 {
                        freqs_cis(c.rope_dim, c.max_seq, c.rope_theta)
                    } else {
                        yarn_freqs_cis(c.rope_dim, c.max_seq, c.orig_seq, c.compress_rope_theta, c.rope_factor, c.beta_fast, c.beta_slow)
                    }
                })
                .collect(),
        }
    }
}

pub struct LayerTrace {
    pub engram_out: Option<Vec<f32>>,
    pub h_in: Vec<f32>,
    pub pre_mix_in: Vec<f32>,
    pub attn_pre: Vec<f32>,
    pub attn_post: Vec<f32>,
    pub attn_comb: Vec<f32>,
    pub attn_in: Vec<f32>,
    pub attn: CompAttnRun,
    pub ffn_pre: Vec<f32>,
    pub ffn_post: Vec<f32>,
    pub ffn_comb: Vec<f32>,
    pub ffn_in: Vec<f32>,
    pub moe_weights: Vec<f32>,
    pub moe_indices: Vec<usize>,
    pub ffn_out: Vec<f32>,
    pub h_out: Vec<f32>,
    /// snapshot of the shared slots after this layer, for the source-layer captures
    pub shared_compress_kv: Vec<f32>,
    pub shared_index_k: Vec<f32>,
    pub shared_topk: Vec<i32>,
    pub shared_topk_w: usize,
    pub shared_cand: Vec<bool>,
    pub shared_cand_w: usize,
}

pub struct StepTrace {
    pub tokens: usize,
    pub embed: Vec<f32>,
    pub layers: Vec<LayerTrace>,
    pub h_final: Vec<f32>,
    pub head_in: Vec<f32>,
    pub logits: Vec<f32>,
}

/// `Transformer.forward` for one chunk (`[seqlen]` ids at `start_pos`), text only.
pub fn forward(ids: &[i64], start_pos: usize, c: &ModelCfg, w: &ModelWeights, st: &mut ModelState, t: &EngramTables) -> StepTrace {
    let (dim, hc, s) = (c.dim, c.hc, ids.len());
    let hashes = st.hash.forward(ids, s, start_pos);
    let (nl, cols) = (t.layer_ids.len(), t.n_hash_cols());

    let embed: Vec<f32> = ids.iter().flat_map(|&id| w.embed[id as usize * dim..(id as usize + 1) * dim].iter().copied()).collect();
    let mut h: Vec<f32> = (0..s).flat_map(|tk| std::iter::repeat(embed[tk * dim..(tk + 1) * dim].to_vec()).take(hc).flatten()).collect();
    let mut pre_mix: Vec<f32> = (0..s).flat_map(|_| (0..hc).map(|cpy| if cpy == 0 { 1.0 } else { 0.0 })).collect();
    let icfg = c.indexer_cfg();
    let mut traces = Vec::with_capacity(c.n_layers);

    for l in 0..c.n_layers {
        let lw = &w.layers[l];
        let hs = &hashes;
        let engram_out = lw.engram.as_ref().map(|e| {
            let hi = e.hash_index;
            let ids_l: Vec<i64> = (0..s).flat_map(|tk| (0..cols).map(move |cc| hs[(tk * nl + hi) * cols + cc])).collect();
            let emb = embed_rows(&e.table, t.head_dim, &ids_l);
            let in_f = cols * t.head_dim;
            let kv = super::engram::linear_bf16(&emb, s, in_f, &e.wkv, dim * (hc + 1));
            let out = gate_and_add(&h, &kv, &e.q, &e.k, s, hc, dim, c.eps);
            h = out.clone();
            out
        });
        let h_in = h.clone();
        let pre_mix_in = pre_mix.clone();

        let (attn_pre, attn_post, attn_comb) = hc_mixes(&h, s, hc, dim, &lw.hc_attn_fn, &lw.hc_attn_scale, &lw.hc_attn_base, c.iters, c.hc_eps, c.eps);
        let attn_in = rms_norm(&hc_pre(&h, &pre_mix, s, hc, dim), &lw.attn_norm, s, dim, c.eps);
        let aw = AttnWeights { sink: &lw.sink, wq_a: &lw.wq_a, q_norm: &lw.q_norm, wq_b: &lw.wq_b, wkv: &lw.wkv, kv_norm: &lw.kv_norm, wo_a: &lw.wo_a, wo_b: &lw.wo_b };
        let comp = lw.comp_wkv.as_ref().map(|wkv| CompressorWeights { wkv, wgate: lw.comp_wgate.as_deref(), norm: lw.comp_norm.as_deref().expect("comp norm") });
        let idxw = lw.idx_wq_b.as_ref().map(|wq_b| IndexerWeights { wq_b, weights_proj: lw.idx_weights_proj.as_deref().expect("weights_proj"), wk: lw.idx_wk.as_deref(), k_norm: lw.idx_k_norm.as_deref() });
        let ac = c.attn_cfg(l);
        let attn = attention_any(&attn_in, s, start_pos, &aw, comp.as_ref(), idxw.as_ref(), &icfg, &ac, &st.fcs[l], &mut st.layers[l], &mut st.shared);
        let h_mid = hc_post(&attn.out, &h, &attn_post, &attn_comb, s, hc, dim);

        let (ffn_pre, ffn_post, ffn_comb) = hc_mixes(&h_mid, s, hc, dim, &lw.hc_ffn_fn, &lw.hc_ffn_scale, &lw.hc_ffn_base, c.iters, c.hc_eps, c.eps);
        let ffn_in = rms_norm(&hc_pre(&h_mid, &attn_pre, s, hc, dim), &lw.ffn_norm, s, dim, c.eps);
        let mw = MoeWeights {
            gate_w: &lw.gate_w,
            gate_bias: &lw.gate_bias,
            experts: lw.experts.iter().map(|(a, b, cc)| (a.as_slice(), b.as_slice(), cc.as_slice())).collect(),
            shared: (&lw.shared.0, &lw.shared.1, &lw.shared.2),
        };
        let (ffn_out, moe_weights, moe_indices) = moe(&ffn_in, s, &mw, &c.moe);
        let h_out = hc_post(&ffn_out, &h_mid, &ffn_post, &ffn_comb, s, hc, dim);

        h = h_out.clone();
        pre_mix = ffn_pre.clone();
        traces.push(LayerTrace {
            engram_out,
            h_in,
            pre_mix_in,
            attn_pre,
            attn_post,
            attn_comb,
            attn_in,
            attn,
            ffn_pre,
            ffn_post,
            ffn_comb,
            ffn_in,
            moe_weights,
            moe_indices,
            ffn_out,
            h_out,
            shared_compress_kv: st.shared.compress_kv.clone(),
            shared_index_k: st.shared.index_k.clone(),
            shared_topk: st.shared.topk_idxs.clone(),
            shared_topk_w: st.shared.topk,
            shared_cand: st.shared.candidates.clone(),
            shared_cand_w: st.shared.cand_width,
        });
    }

    let h_final = hc_pre(&h, &pre_mix, s, hc, dim);
    let head_in = rms_norm(&h_final, &w.norm, s, dim, c.eps);
    let mut logits = vec![0f32; s * c.vocab];
    for tk in 0..s {
        for v in 0..c.vocab {
            logits[tk * c.vocab + v] = head_in[tk * dim..(tk + 1) * dim].iter().zip(&w.head[v * dim..(v + 1) * dim]).map(|(a, b)| a * b).sum();
        }
    }
    StepTrace { tokens: s, embed, layers: traces, h_final, head_in, logits }
}

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;
