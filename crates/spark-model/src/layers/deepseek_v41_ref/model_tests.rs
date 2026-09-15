// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! The full forward against the golden, prefill then both decode steps through one model state.
//! Each test walks one capture family across every layer and regime so a failure names the first
//! layer that diverged.

use super::super::testutil::*;
use super::super::{Golden, fixed_int};
use super::super::engram::EngramTables;
use super::*;

struct Fx {
    g: Golden,
    c: ModelCfg,
    t: EngramTables,
    regimes: Vec<String>,
    ids: Vec<i64>,
    prefill: usize,
}

fn fx() -> Fx {
    let g = Golden::load();
    let c = ModelCfg::from_golden(&g);
    let t = EngramTables::from_golden(&g);
    let prefill = g.fixture_u64("prefill_len") as usize;
    let ids: Vec<i64> = (0..(prefill + 2) as u64).map(|i| fixed_int("input_ids", i, c.vocab as u64) as i64).collect();
    Fx { regimes: g.regimes(), ids, prefill, g, c, t }
}

fn run_all(f: &Fx) -> Vec<(String, StepTrace)> {
    let w = ModelWeights::from_golden(&f.g, &f.c, &f.t);
    let mut st = ModelState::new(&f.c, &f.t);
    let p = f.prefill;
    let chunks = [(&f.ids[..p], 0usize), (&f.ids[p..p + 1], p), (&f.ids[p + 1..p + 2], p + 1)];
    f.regimes.iter().zip(chunks).map(|(r, (ids, start))| (r.clone(), forward(ids, start, &f.c, &w, &mut st, &f.t))).collect()
}

fn check_bf16(g: &Golden, r: &str, name: &str, got: &[f32]) {
    let gt = g.tensor(r, name);
    check_capture(&format!("{r}.{name}"), &as_f64(got), &gt, bf16_tol(&gt), BF16_CK_REL);
}

fn check_f32(g: &Golden, r: &str, name: &str, got: &[f32]) {
    let gt = g.tensor(r, name);
    check_capture(&format!("{r}.{name}"), &as_f64(got), &gt, F32_TOL, F32_TOL);
}

fn check_int<I: Copy + Into<i64>>(g: &Golden, r: &str, name: &str, got: &[I]) {
    let gt = g.tensor(r, name);
    let v: Vec<f64> = got.iter().map(|&i| i.into() as f64).collect();
    check_capture(&format!("{r}.{name}"), &v, &gt, EXACT, EXACT);
}

#[test]
fn config_and_geometry() {
    let f = fx();
    assert_eq!(f.c.n_layers, 6);
    assert_eq!(f.c.ratios, vec![0, 0, 2, 2, 1, 1]);
    assert_eq!((f.c.kv_sources.clone(), f.c.index_sources.clone(), f.c.cand_src), (vec![2, 4], vec![2, 4, 5], 4));
    let w = ModelWeights::from_golden(&f.g, &f.c, &f.t);
    assert_eq!(w.embed.len(), f.c.vocab * f.c.dim);
    assert_eq!(w.head.len(), f.c.vocab * f.c.dim);
    assert!(w.layers[1].engram.is_some() && w.layers[4].engram.is_some() && w.layers[0].engram.is_none());
    assert!(w.layers[2].comp_wgate.is_some() && w.layers[4].comp_wgate.is_none() && w.layers[5].comp_wkv.is_none());
    assert!(w.layers[5].idx_wq_b.is_some() && w.layers[5].idx_wk.is_none() && w.layers[3].idx_wq_b.is_none());
}

#[test]
fn embedding_and_layer0_reproduce_the_standalone_checks() {
    let f = fx();
    for (r, s) in run_all(&f) {
        check_bf16(&f.g, &r, "embed", &s.embed);
        let l0 = &s.layers[0];
        check_bf16(&f.g, &r, "L0.h_in", &l0.h_in);
        check_f32(&f.g, &r, "L0.pre_mix_in", &l0.pre_mix_in);
        check_bf16(&f.g, &r, "L0.attn_in", &l0.attn_in);
        check_bf16(&f.g, &r, "L0.attn_out", &l0.attn.out);
        check_bf16(&f.g, &r, "L0.ffn_in", &l0.ffn_in);
        check_bf16(&f.g, &r, "L0.ffn_out", &l0.ffn_out);
        check_bf16(&f.g, &r, "L0.h_out", &l0.h_out);
    }
}

#[test]
fn engram_layers_match_in_the_forward() {
    let f = fx();
    for (r, s) in run_all(&f) {
        for l in [1usize, 4] {
            let e = s.layers[l].engram_out.as_ref().expect("engram layer");
            check_bf16(&f.g, &r, &format!("L{l}.engram_out"), e);
            check_bf16(&f.g, &r, &format!("L{l}.h_in"), &s.layers[l].h_in);
        }
    }
}

#[test]
fn every_layer_attention_matches() {
    let f = fx();
    for (r, s) in run_all(&f) {
        for (l, lt) in s.layers.iter().enumerate() {
            let p = |n: &str| format!("L{l}.{n}");
            check_f32(&f.g, &r, &p("attn_pre"), &lt.attn_pre);
            check_f32(&f.g, &r, &p("attn_post"), &lt.attn_post);
            check_f32(&f.g, &r, &p("attn_comb"), &lt.attn_comb);
            check_bf16(&f.g, &r, &p("attn_in"), &lt.attn_in);
            check_bf16(&f.g, &r, &p("sa_q"), &lt.attn.q);
            check_bf16(&f.g, &r, &p("sa_kv"), &lt.attn.kv_rows);
            assert_eq!(f.g.tensor(&r, &p("sa_topk_idxs")).shape[2], lt.attn.topk, "{r}.L{l}: topk width");
            check_int(&f.g, &r, &p("sa_topk_idxs"), &lt.attn.idx);
            check_bf16(&f.g, &r, &p("sa_o"), &lt.attn.o);
            check_bf16(&f.g, &r, &p("attn_out"), &lt.attn.out);
        }
    }
}

#[test]
fn shared_slots_match_after_each_source_layer() {
    let f = fx();
    let hd = f.c.head_dim;
    let ihd = f.c.index_hd;
    for (r, s) in run_all(&f) {
        for l in [2usize, 4, 5] {
            let lt = &s.layers[l];
            let names = f.g.capture_names(&r);
            let has = |n: &str| names.iter().any(|x| x == n);
            let ckv = format!("shared.L{l}.compress_kv");
            if has(&ckv) {
                let used = f.g.tensor(&r, &ckv).shape[1];
                check_bf16(&f.g, &r, &ckv, &lt.shared_compress_kv[..used * hd]);
            }
            let ik = format!("shared.L{l}.index_k");
            if has(&ik) {
                let used = f.g.tensor(&r, &ik).shape[1];
                check_bf16(&f.g, &r, &ik, &lt.shared_index_k[..used * ihd]);
            }
            let tk = format!("shared.L{l}.topk_idxs");
            if has(&tk) {
                assert_eq!(f.g.tensor(&r, &tk).shape[2], lt.shared_topk_w, "{r}.{tk}: width");
                check_int(&f.g, &r, &tk, &lt.shared_topk);
            }
            let cd = format!("shared.L{l}.candidates");
            if has(&cd) {
                let v: Vec<i32> = lt.shared_cand.iter().map(|&b| b as i32).collect();
                assert_eq!(f.g.tensor(&r, &cd).shape[2], lt.shared_cand_w, "{r}.{cd}: width");
                check_int(&f.g, &r, &cd, &v);
            }
        }
    }
}

#[test]
fn every_layer_ffn_and_block_output_match() {
    let f = fx();
    for (r, s) in run_all(&f) {
        for (l, lt) in s.layers.iter().enumerate() {
            let p = |n: &str| format!("L{l}.{n}");
            check_f32(&f.g, &r, &p("ffn_pre"), &lt.ffn_pre);
            check_f32(&f.g, &r, &p("ffn_post"), &lt.ffn_post);
            check_f32(&f.g, &r, &p("ffn_comb"), &lt.ffn_comb);
            check_bf16(&f.g, &r, &p("ffn_in"), &lt.ffn_in);
            check_int(&f.g, &r, &p("moe_indices"), &lt.moe_indices.iter().map(|&i| i as i64).collect::<Vec<_>>());
            check_f32(&f.g, &r, &p("moe_weights"), &lt.moe_weights);
            check_bf16(&f.g, &r, &p("ffn_out"), &lt.ffn_out);
            check_bf16(&f.g, &r, &p("h_out"), &lt.h_out);
            check_f32(&f.g, &r, &p("pre_mix_out"), &lt.ffn_pre);
        }
    }
}

#[test]
fn final_collapse_norm_and_logits_match() {
    let f = fx();
    for (r, s) in run_all(&f) {
        check_bf16(&f.g, &r, "h_final", &s.h_final);
        check_bf16(&f.g, &r, "head_in", &s.head_in);
        let gt = f.g.tensor(&r, "logits_full");
        // f32 logits from bf16 inputs: allow bf16-scale noise from accumulation order
        check_capture(&format!("{r}.logits_full"), &as_f64(&s.logits), &gt, bf16_tol(&gt), BF16_CK_REL);
        assert_eq!(s.logits.len(), s.tokens * f.c.vocab);
    }
}
