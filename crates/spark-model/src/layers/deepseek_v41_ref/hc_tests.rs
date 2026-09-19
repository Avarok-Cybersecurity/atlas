// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! Layer 0's scaffold against the golden, in the reference's own order, so each failure names
//! the first stage that diverged: attention mixes -> attn_in -> (hc_post) -> FFN mixes -> ffn_in
//! -> h_out / pre_mix_out. Attention and MoE outputs are taken from the golden.

use super::super::testutil::*;
use super::super::{Golden, regen_param};
use super::*;

struct Fx {
    g: Golden,
    regimes: Vec<String>,
    hc: usize,
    dim: usize,
    iters: usize,
    hc_eps: f32,
    norm_eps: f32,
}

fn fx() -> Fx {
    let g = Golden::load();
    Fx {
        regimes: g.regimes(),
        hc: g.fixture_u64("hc_mult") as usize,
        dim: g.fixture_u64("dim") as usize,
        iters: g.fixture_u64("hc_sinkhorn_iters") as usize,
        hc_eps: g.fixture_f64("hc_eps") as f32,
        norm_eps: g.fixture_f64("norm_eps") as f32,
        g,
    }
}

struct Layer0 {
    attn_fn: Vec<f32>,
    attn_scale: Vec<f32>,
    attn_base: Vec<f32>,
    ffn_fn: Vec<f32>,
    ffn_scale: Vec<f32>,
    ffn_base: Vec<f32>,
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
}

fn layer0(f: &Fx) -> Layer0 {
    let p = |n: &str| regen_param(&f.g, &format!("layers.0.{n}"));
    Layer0 {
        attn_fn: p("hc_attn_fn"),
        attn_scale: p("hc_attn_scale"),
        attn_base: p("hc_attn_base"),
        ffn_fn: p("hc_ffn_fn"),
        ffn_scale: p("hc_ffn_scale"),
        ffn_base: p("hc_ffn_base"),
        attn_norm: p("attn_norm.weight"),
        ffn_norm: p("ffn_norm.weight"),
    }
}

/// Everything the scaffold produces for one regime, computed in the reference's order.
struct Run {
    tokens: usize,
    attn_pre: Vec<f32>,
    attn_post: Vec<f32>,
    attn_comb: Vec<f32>,
    attn_in: Vec<f32>,
    h_mid: Vec<f32>,
    ffn_pre: Vec<f32>,
    ffn_post: Vec<f32>,
    ffn_comb: Vec<f32>,
    ffn_in: Vec<f32>,
    h_out: Vec<f32>,
}

fn run(f: &Fx, w: &Layer0, r: &str) -> Run {
    let g = &f.g;
    let h_in = full_f32(&g.tensor(r, "L0.h_in"), "L0.h_in");
    let pre_mix_in = full_f32(&g.tensor(r, "L0.pre_mix_in"), "L0.pre_mix_in");
    let attn_out = full_f32(&g.tensor(r, "L0.attn_out"), "L0.attn_out");
    let ffn_out = full_f32(&g.tensor(r, "L0.ffn_out"), "L0.ffn_out");
    let tokens = h_in.len() / (f.hc * f.dim);

    let (attn_pre, attn_post, attn_comb) = hc_mixes(
        &h_in,
        tokens,
        f.hc,
        f.dim,
        &w.attn_fn,
        &w.attn_scale,
        &w.attn_base,
        f.iters,
        f.hc_eps,
        f.norm_eps,
    );
    let attn_in = rms_norm(
        &hc_pre(&h_in, &pre_mix_in, tokens, f.hc, f.dim),
        &w.attn_norm,
        tokens,
        f.dim,
        f.norm_eps,
    );
    let h_mid = hc_post(
        &attn_out, &h_in, &attn_post, &attn_comb, tokens, f.hc, f.dim,
    );
    let (ffn_pre, ffn_post, ffn_comb) = hc_mixes(
        &h_mid,
        tokens,
        f.hc,
        f.dim,
        &w.ffn_fn,
        &w.ffn_scale,
        &w.ffn_base,
        f.iters,
        f.hc_eps,
        f.norm_eps,
    );
    // the FFN collapses with the mix THIS block's attention produced
    let ffn_in = rms_norm(
        &hc_pre(&h_mid, &attn_pre, tokens, f.hc, f.dim),
        &w.ffn_norm,
        tokens,
        f.dim,
        f.norm_eps,
    );
    let h_out = hc_post(&ffn_out, &h_mid, &ffn_post, &ffn_comb, tokens, f.hc, f.dim);
    Run {
        tokens,
        attn_pre,
        attn_post,
        attn_comb,
        attn_in,
        h_mid,
        ffn_pre,
        ffn_post,
        ffn_comb,
        ffn_in,
        h_out,
    }
}

#[test]
fn params_have_the_expected_geometry() {
    let f = fx();
    let w = layer0(&f);
    let mix_hc = (2 + f.hc) * f.hc;
    assert_eq!(w.attn_fn.len(), mix_hc * f.hc * f.dim);
    assert_eq!(w.attn_scale.len(), 3);
    assert_eq!(w.attn_base.len(), mix_hc);
    assert_eq!(w.attn_norm.len(), f.dim);
    assert_eq!((f.hc, f.dim, f.iters), (4, 64, 20));
}

#[test]
fn attention_mixes_match() {
    let f = fx();
    let w = layer0(&f);
    for r in &f.regimes {
        let s = run(&f, &w, r);
        check_capture(
            &format!("{r}.L0.attn_pre"),
            &as_f64(&s.attn_pre),
            &f.g.tensor(r, "L0.attn_pre"),
            F32_TOL,
            F32_TOL,
        );
        check_capture(
            &format!("{r}.L0.attn_post"),
            &as_f64(&s.attn_post),
            &f.g.tensor(r, "L0.attn_post"),
            F32_TOL,
            F32_TOL,
        );
        check_capture(
            &format!("{r}.L0.attn_comb"),
            &as_f64(&s.attn_comb),
            &f.g.tensor(r, "L0.attn_comb"),
            F32_TOL,
            F32_TOL,
        );
    }
}

#[test]
fn attention_input_matches_within_bf16() {
    let f = fx();
    let w = layer0(&f);
    for r in &f.regimes {
        let s = run(&f, &w, r);
        let gt = f.g.tensor(r, "L0.attn_in");
        check_capture(
            &format!("{r}.L0.attn_in"),
            &as_f64(&s.attn_in),
            &gt,
            bf16_tol(&gt),
            BF16_CK_REL,
        );
    }
}

#[test]
fn ffn_mixes_match_through_hc_post() {
    // hc_post has no capture of its own; the FFN mixes are computed from its output, so this
    // checks hc_post's comb orientation as much as the mixes
    let f = fx();
    let w = layer0(&f);
    for r in &f.regimes {
        let s = run(&f, &w, r);
        assert_eq!(s.h_mid.len(), s.tokens * f.hc * f.dim);
        check_capture(
            &format!("{r}.L0.ffn_pre"),
            &as_f64(&s.ffn_pre),
            &f.g.tensor(r, "L0.ffn_pre"),
            F32_TOL,
            F32_TOL,
        );
        check_capture(
            &format!("{r}.L0.ffn_post"),
            &as_f64(&s.ffn_post),
            &f.g.tensor(r, "L0.ffn_post"),
            F32_TOL,
            F32_TOL,
        );
        check_capture(
            &format!("{r}.L0.ffn_comb"),
            &as_f64(&s.ffn_comb),
            &f.g.tensor(r, "L0.ffn_comb"),
            F32_TOL,
            F32_TOL,
        );
    }
}

#[test]
fn ffn_input_matches_within_bf16() {
    let f = fx();
    let w = layer0(&f);
    for r in &f.regimes {
        let s = run(&f, &w, r);
        let gt = f.g.tensor(r, "L0.ffn_in");
        check_capture(
            &format!("{r}.L0.ffn_in"),
            &as_f64(&s.ffn_in),
            &gt,
            bf16_tol(&gt),
            BF16_CK_REL,
        );
    }
}

#[test]
fn block_output_and_next_pre_mix_match() {
    let f = fx();
    let w = layer0(&f);
    for r in &f.regimes {
        let s = run(&f, &w, r);
        let gt = f.g.tensor(r, "L0.h_out");
        check_capture(
            &format!("{r}.L0.h_out"),
            &as_f64(&s.h_out),
            &gt,
            bf16_tol(&gt),
            BF16_CK_REL,
        );
        // Block.forward returns ffn_pre as the NEXT block's pre_mix
        check_capture(
            &format!("{r}.L0.pre_mix_out"),
            &as_f64(&s.ffn_pre),
            &f.g.tensor(r, "L0.pre_mix_out"),
            F32_TOL,
            F32_TOL,
        );
    }
}
