// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! Layer 0's attention against the golden, the three regimes run IN ORDER through one window
//! cache: `sa_q`, `sa_kv` (the rows attention read: the chunk in prefill, the ring in decode),
//! `sa_topk_idxs` exactly, `sa_o`, and `attn_out` in full.

use super::super::testutil::*;
use super::super::{Golden, regen_param};
use super::*;

struct Fx {
    g: Golden,
    regimes: Vec<String>,
    cfg: AttnCfg,
    fc: Vec<(f32, f32)>,
    prefill_len: usize,
}

fn fx() -> Fx {
    let g = Golden::load();
    let cfg = AttnCfg {
        dim: g.fixture_u64("dim") as usize,
        n_heads: g.fixture_u64("n_heads") as usize,
        head_dim: g.fixture_u64("head_dim") as usize,
        rope_dim: g.fixture_u64("rope_head_dim") as usize,
        q_rank: g.fixture_u64("q_lora_rank") as usize,
        o_rank: g.fixture_u64("o_lora_rank") as usize,
        groups: g.fixture_u64("o_groups") as usize,
        window: g.fixture_u64("window_size") as usize,
        eps: g.fixture_f64("norm_eps") as f32,
    };
    let fc = freqs_cis(
        cfg.rope_dim,
        g.fixture_u64("max_seq_len") as usize,
        g.fixture_f64("rope_theta") as f32,
    );
    Fx {
        regimes: g.regimes(),
        cfg,
        fc,
        prefill_len: g.fixture_u64("prefill_len") as usize,
        g,
    }
}

struct Params {
    sink: Vec<f32>,
    wq_a: Vec<f32>,
    q_norm: Vec<f32>,
    wq_b: Vec<f32>,
    wkv: Vec<f32>,
    kv_norm: Vec<f32>,
    wo_a: Vec<f32>,
    wo_b: Vec<f32>,
}

fn params(f: &Fx) -> Params {
    let p = |n: &str| regen_param(&f.g, &format!("layers.0.attn.{n}"));
    Params {
        sink: p("attn_sink"),
        wq_a: p("wq_a.weight"),
        q_norm: p("q_norm.weight"),
        wq_b: p("wq_b.weight"),
        wkv: p("wkv.weight"),
        kv_norm: p("kv_norm.weight"),
        wo_a: p("wo_a.weight"),
        wo_b: p("wo_b.weight"),
    }
}

fn weights(p: &Params) -> AttnWeights<'_> {
    AttnWeights {
        sink: &p.sink,
        wq_a: &p.wq_a,
        q_norm: &p.q_norm,
        wq_b: &p.wq_b,
        wkv: &p.wkv,
        kv_norm: &p.kv_norm,
        wo_a: &p.wo_a,
        wo_b: &p.wo_b,
    }
}

/// Run every regime in order through one cache; returns (regime, start_pos, run).
fn run_all(f: &Fx, w: &AttnWeights) -> Vec<(String, usize, AttnRun)> {
    assert_eq!(
        f.regimes[0], "prefill12",
        "regimes must start with the prefill"
    );
    let mut cache = WindowCache::new(&f.cfg);
    let mut start = 0usize;
    let mut out = Vec::new();
    for r in &f.regimes {
        let x = full_f32(&f.g.tensor(r, "L0.attn_in"), "L0.attn_in");
        let seqlen = x.len() / f.cfg.dim;
        let run = attention(&x, seqlen, start, w, &f.cfg, &f.fc, &mut cache);
        out.push((r.clone(), start, run));
        start += seqlen;
    }
    assert_eq!(start, f.prefill_len + f.regimes.len() - 1);
    out
}

#[test]
fn params_have_the_expected_geometry() {
    let f = fx();
    let p = params(&f);
    let c = &f.cfg;
    assert_eq!(
        (
            c.dim, c.n_heads, c.head_dim, c.rope_dim, c.q_rank, c.o_rank, c.groups, c.window
        ),
        (64, 4, 32, 8, 16, 16, 2, 8)
    );
    assert_eq!(p.sink.len(), c.n_heads);
    assert_eq!(p.wq_a.len(), c.q_rank * c.dim);
    assert_eq!(p.wq_b.len(), c.n_heads * c.head_dim * c.q_rank);
    assert_eq!(p.wkv.len(), c.head_dim * c.dim);
    assert_eq!(
        p.wo_a.len(),
        c.groups * c.o_rank * (c.n_heads * c.head_dim / c.groups)
    );
    assert_eq!(p.wo_b.len(), c.dim * c.groups * c.o_rank);
    assert_eq!(
        c.head_dim, FP8_BLOCK,
        "the fp8 block must cover one latent row"
    );
}

#[test]
fn window_indices_match_exactly() {
    let f = fx();
    let p = params(&f);
    let w = weights(&p);
    for (r, _, run) in run_all(&f, &w) {
        let gt = f.g.tensor(&r, "L0.sa_topk_idxs");
        let got: Vec<f64> = run.idx.iter().map(|&i| i as f64).collect();
        assert_eq!(gt.shape[2], run.topk, "{r}: topk width");
        check_capture(&format!("{r}.L0.sa_topk_idxs"), &got, &gt, EXACT, EXACT);
    }
}

#[test]
fn query_and_kv_rows_match_within_bf16() {
    let f = fx();
    let p = params(&f);
    let w = weights(&p);
    for (r, _, run) in run_all(&f, &w) {
        let gq = f.g.tensor(&r, "L0.sa_q");
        check_capture(
            &format!("{r}.L0.sa_q"),
            &as_f64(&run.q),
            &gq,
            bf16_tol(&gq),
            BF16_CK_REL,
        );
        // prefill: the chunk's own rows; decode: the whole ring, so this checks the cache too
        let gk = f.g.tensor(&r, "L0.sa_kv");
        check_capture(
            &format!("{r}.L0.sa_kv"),
            &as_f64(&run.kv_rows),
            &gk,
            bf16_tol(&gk),
            BF16_CK_REL,
        );
    }
}

#[test]
fn sparse_attention_output_matches_within_bf16() {
    let f = fx();
    let p = params(&f);
    let w = weights(&p);
    for (r, _, run) in run_all(&f, &w) {
        let go = f.g.tensor(&r, "L0.sa_o");
        check_capture(
            &format!("{r}.L0.sa_o"),
            &as_f64(&run.o),
            &go,
            bf16_tol(&go),
            BF16_CK_REL,
        );
    }
}

#[test]
fn attention_output_matches_within_bf16() {
    let f = fx();
    let p = params(&f);
    let w = weights(&p);
    for (r, _, run) in run_all(&f, &w) {
        let gt = f.g.tensor(&r, "L0.attn_out");
        check_capture(
            &format!("{r}.L0.attn_out"),
            &as_f64(&run.out),
            &gt,
            bf16_tol(&gt),
            BF16_CK_REL,
        );
    }
}

#[test]
fn fp8_round_trip_is_bf16_exact_and_pow2_ceil_matches_the_kernel() {
    assert_eq!(pow2_ceil(1.0), 1.0);
    assert_eq!(pow2_ceil(1.0001), 2.0);
    assert_eq!(pow2_ceil(0.75), 1.0);
    assert_eq!(pow2_ceil(0.5), 0.5);
    let mut v: Vec<f32> = (0..FP8_BLOCK).map(|i| (i as f32 - 16.0) * 0.37).collect();
    act_quant_inplace(&mut v);
    for x in &v {
        assert_eq!(*x, to_bf16_rne(*x), "dequantised fp8 must already be bf16");
    }
}
