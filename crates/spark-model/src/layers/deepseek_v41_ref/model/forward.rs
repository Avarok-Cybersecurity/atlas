// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! `Transformer.forward` for one chunk of the DeepSeek-V4.1 tiny graph: embed, expand to
//! `hc_mult` copies, the one-hot initial pre-mix, engram, the blocks with the delayed mixes,
//! collapse, final RMSNorm, f32 logits. Split out of `model.rs` for the file-size cap; the
//! config, weights, state and trace types it fills stay there.

use super::super::attn::AttnWeights;
use super::super::compress::{CompressorWeights, IndexerWeights, attention_any};
use super::super::engram::{EngramTables, embed_rows, gate_and_add};
use super::super::hc::{hc_mixes, hc_post, hc_pre, rms_norm};
use super::super::moe::{MoeWeights, moe};
use super::{LayerTrace, ModelCfg, ModelState, ModelWeights, StepTrace};

/// `Transformer.forward` for one chunk (`[seqlen]` ids at `start_pos`), text only.
pub fn forward(
    ids: &[i64],
    start_pos: usize,
    c: &ModelCfg,
    w: &ModelWeights,
    st: &mut ModelState,
    t: &EngramTables,
) -> StepTrace {
    let (dim, hc, s) = (c.dim, c.hc, ids.len());
    let hashes = st.hash.forward(ids, s, start_pos);
    let (nl, cols) = (t.layer_ids.len(), t.n_hash_cols());

    let embed: Vec<f32> = ids
        .iter()
        .flat_map(|&id| {
            w.embed[id as usize * dim..(id as usize + 1) * dim]
                .iter()
                .copied()
        })
        .collect();
    let mut h: Vec<f32> = (0..s)
        .flat_map(|tk| std::iter::repeat_n(embed[tk * dim..(tk + 1) * dim].to_vec(), hc).flatten())
        .collect();
    let mut pre_mix: Vec<f32> = (0..s)
        .flat_map(|_| (0..hc).map(|stream| if stream == 0 { 1.0 } else { 0.0 }))
        .collect();
    let icfg = c.indexer_cfg();
    let mut traces = Vec::with_capacity(c.n_layers);

    for l in 0..c.n_layers {
        let lw = &w.layers[l];
        let hs = &hashes;
        let engram_out = lw.engram.as_ref().map(|e| {
            let hi = e.hash_index;
            let ids_l: Vec<i64> = (0..s)
                .flat_map(|tk| (0..cols).map(move |cc| hs[(tk * nl + hi) * cols + cc]))
                .collect();
            let emb = embed_rows(&e.table, t.head_dim, &ids_l);
            let in_f = cols * t.head_dim;
            let kv = super::super::engram::linear_bf16(&emb, s, in_f, &e.wkv, dim * (hc + 1));
            let out = gate_and_add(&h, &kv, &e.q, &e.k, s, hc, dim, c.eps);
            h = out.clone();
            out
        });
        let h_in = h.clone();
        let pre_mix_in = pre_mix.clone();

        let (attn_pre, attn_post, attn_comb) = hc_mixes(
            &h,
            s,
            hc,
            dim,
            &lw.hc_attn_fn,
            &lw.hc_attn_scale,
            &lw.hc_attn_base,
            c.iters,
            c.hc_eps,
            c.eps,
        );
        let attn_in = rms_norm(
            &hc_pre(&h, &pre_mix, s, hc, dim),
            &lw.attn_norm,
            s,
            dim,
            c.eps,
        );
        let aw = AttnWeights {
            sink: &lw.sink,
            wq_a: &lw.wq_a,
            q_norm: &lw.q_norm,
            wq_b: &lw.wq_b,
            wkv: &lw.wkv,
            kv_norm: &lw.kv_norm,
            wo_a: &lw.wo_a,
            wo_b: &lw.wo_b,
        };
        let comp = lw.comp_wkv.as_ref().map(|wkv| CompressorWeights {
            wkv,
            wgate: lw.comp_wgate.as_deref(),
            norm: lw.comp_norm.as_deref().expect("comp norm"),
        });
        let idxw = lw.idx_wq_b.as_ref().map(|wq_b| IndexerWeights {
            wq_b,
            weights_proj: lw.idx_weights_proj.as_deref().expect("weights_proj"),
            wk: lw.idx_wk.as_deref(),
            k_norm: lw.idx_k_norm.as_deref(),
        });
        let ac = c.attn_cfg(l);
        let attn = attention_any(
            &attn_in,
            s,
            start_pos,
            &aw,
            comp.as_ref(),
            idxw.as_ref(),
            &icfg,
            &ac,
            &st.fcs[l],
            &mut st.layers[l],
            &mut st.shared,
        );
        let h_mid = hc_post(&attn.out, &h, &attn_post, &attn_comb, s, hc, dim);

        let (ffn_pre, ffn_post, ffn_comb) = hc_mixes(
            &h_mid,
            s,
            hc,
            dim,
            &lw.hc_ffn_fn,
            &lw.hc_ffn_scale,
            &lw.hc_ffn_base,
            c.iters,
            c.hc_eps,
            c.eps,
        );
        let ffn_in = rms_norm(
            &hc_pre(&h_mid, &attn_pre, s, hc, dim),
            &lw.ffn_norm,
            s,
            dim,
            c.eps,
        );
        let mw = MoeWeights {
            gate_w: &lw.gate_w,
            gate_bias: &lw.gate_bias,
            experts: lw
                .experts
                .iter()
                .map(|(a, b, cc)| (a.as_slice(), b.as_slice(), cc.as_slice()))
                .collect(),
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
            logits[tk * c.vocab + v] = head_in[tk * dim..(tk + 1) * dim]
                .iter()
                .zip(&w.head[v * dim..(v + 1) * dim])
                .map(|(a, b)| a * b)
                .sum();
        }
    }
    StepTrace {
        tokens: s,
        embed,
        layers: traces,
        h_final,
        head_in,
        logits,
    }
}
