// SPDX-License-Identifier: AGPL-3.0-only

//! The recurrence half of the Slice 2B/2C numeric gate: sub-ops 8-10, the full-layer
//! comparison, and the pre-norm/internal-norm laws.
//!
//! Split out of `tests.rs` for the 500-LoC cap. A CHILD of that module, not a sibling, so
//! `Golden`, `assert_close`, `TOL` and `GOLDEN` stay in one place and both halves read the
//! same fixture through the same accessors.

use super::super::*;
use super::{Golden, TOL, assert_close};

// ---------------------------------------------------------------- 8. recurrent state update

#[test]
fn sub_op_8_recurrent_multi_token() {
    let g = Golden::load();
    let d = g.dims();
    let mut state = vec![0.0f32; d.heads * d.head_dim * d.head_dim];
    let out = kda_recurrent(
        &g.get("inputs", "q_in"),
        &g.get("inputs", "k_in"),
        &g.get("inputs", "v_in"),
        &g.get("outputs", "gate"),
        &g.get("outputs", "beta"),
        d,
        &mut state,
    );
    assert_close(
        "core_recurrent",
        &out,
        &g.get("outputs", "core_recurrent"),
        2e-6,
    );
    assert_close(
        "state_recurrent",
        &state,
        &g.get("outputs", "state_recurrent"),
        2e-6,
    );
}

/// Prefill formulation must reproduce the decode formulation over the same tokens.
#[test]
fn sub_op_8_chunked_prefill_matches_recurrent() {
    let g = Golden::load();
    let d = g.dims();
    let chunk = g.0["fixture"]["chunk_size"].as_u64().unwrap() as usize;
    let mut state = vec![0.0f32; d.heads * d.head_dim * d.head_dim];
    let out = kda_chunked(
        &g.get("inputs", "q_in"),
        &g.get("inputs", "k_in"),
        &g.get("inputs", "v_in"),
        &g.get("outputs", "gate"),
        &g.get("outputs", "beta"),
        d,
        chunk,
        &mut state,
    );
    assert_close(
        "core_chunked",
        &out,
        &g.get("outputs", "core_chunked"),
        2e-5,
    );
    assert_close(
        "state_chunked",
        &state,
        &g.get("outputs", "state_chunked"),
        2e-5,
    );
    assert_close(
        "chunked_vs_recurrent",
        &out,
        &g.get("outputs", "core_recurrent"),
        2e-5,
    );
}

/// `tokens % chunk != 0` — the zero-pad path.
#[test]
fn sub_op_8_chunked_prefill_pads() {
    let g = Golden::load();
    let d = g.dims();
    assert_ne!(
        d.tokens % 4,
        0,
        "fixture must not divide evenly by 4 or this tests nothing"
    );
    let mut state = vec![0.0f32; d.heads * d.head_dim * d.head_dim];
    let out = kda_chunked(
        &g.get("inputs", "q_in"),
        &g.get("inputs", "k_in"),
        &g.get("inputs", "v_in"),
        &g.get("outputs", "gate"),
        &g.get("outputs", "beta"),
        d,
        4,
        &mut state,
    );
    assert_close(
        "core_chunked_c4",
        &out,
        &g.get("outputs", "core_chunked_c4_padded"),
        2e-5,
    );
    assert_close(
        "state_chunked_c4",
        &state,
        &g.get("outputs", "state_chunked_c4_padded"),
        2e-5,
    );
}

/// Prefill 4 tokens, then decode tokens 4 and 5 one at a time off the carried state. This is the
/// transition Atlas's scheduler will actually drive, and it is the one a single-step gate test
/// cannot catch.
#[test]
fn sub_op_8_prefill_then_decode_continuation() {
    let g = Golden::load();
    let d = g.dims();
    let (h_n, hd) = (d.heads, d.head_dim);
    let (q, k, v) = (
        g.get("inputs", "q_in"),
        g.get("inputs", "k_in"),
        g.get("inputs", "v_in"),
    );
    let (gate, beta) = (g.get("outputs", "gate"), g.get("outputs", "beta"));
    let chunk = g.0["fixture"]["chunk_size"].as_u64().unwrap() as usize;
    let split = 4usize;
    let per_tok = h_n * hd;

    let pre_dims = KdaDims { tokens: split, ..d };
    let mut state = vec![0.0f32; h_n * hd * hd];
    let mut out = kda_chunked(
        &q[..split * per_tok],
        &k[..split * per_tok],
        &v[..split * per_tok],
        &gate[..split * per_tok],
        &beta[..split * h_n],
        pre_dims,
        chunk,
        &mut state,
    );

    let step_dims = KdaDims { tokens: 1, ..d };
    for t in split..d.tokens {
        let (a, b) = (t * per_tok, (t + 1) * per_tok);
        out.extend_from_slice(&kda_recurrent(
            &q[a..b],
            &k[a..b],
            &v[a..b],
            &gate[a..b],
            &beta[t * h_n..(t + 1) * h_n],
            step_dims,
            &mut state,
        ));
    }
    assert_close(
        "split_prefill_then_decode",
        &out,
        &g.get("outputs", "core_split_prefill_then_decode"),
        2e-5,
    );
    assert_close(
        "split_state_matches_full",
        &state,
        &g.get("outputs", "state_recurrent"),
        2e-5,
    );
}

// ---------------------------------------------------------------- 9. RMSNormGated

#[test]
fn sub_op_9_rms_norm_gated() {
    let g = Golden::load();
    let d = g.dims();
    let out = rms_norm_gated(
        &g.get("outputs", "core_recurrent"),
        &g.get("inputs", "o_norm_w"),
        &g.get("outputs", "out_gate"),
        d.head_dim,
        g.scalar("rms_eps"),
    );
    assert_close("o_norm_out", &out, &g.get("outputs", "o_norm_out"), 2e-6);
}

/// `rms_norm_eps` comes from config (1e-5 here). vLLM never passes it and gets the same value from
/// a library default — a coincidence Atlas must not inherit.
#[test]
fn sub_op_9_eps_is_load_bearing() {
    let g = Golden::load();
    let d = g.dims();
    let args = (
        g.get("outputs", "core_recurrent"),
        g.get("inputs", "o_norm_w"),
        g.get("outputs", "out_gate"),
    );
    let with_cfg = rms_norm_gated(&args.0, &args.1, &args.2, d.head_dim, g.scalar("rms_eps"));
    let with_other = rms_norm_gated(&args.0, &args.1, &args.2, d.head_dim, 1e-2);
    let diff = with_cfg
        .iter()
        .zip(&with_other)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(diff > 1e-4, "eps must measurably matter, saw {diff:e}");
}

// ---------------------------------------------------------------- 10. low-rank output gate

#[test]
fn sub_op_10_lowrank_output_gate() {
    let g = Golden::load();
    let d = g.dims();
    let g_a = linear(
        &g.get("inputs", "hidden_states"),
        d.tokens,
        d.hidden,
        &g.get("inputs", "W_g_a"),
        d.head_dim,
    );
    let out_gate = linear(
        &g_a,
        d.tokens,
        d.head_dim,
        &g.get("inputs", "W_g_b"),
        d.heads * d.head_dim,
    );
    assert_close("out_gate", &out_gate, &g.get("outputs", "out_gate"), TOL);
}

// ---------------------------------------------------------------- end to end

#[test]
fn full_layer_matches_hf() {
    let g = Golden::load();
    let d = g.dims();
    let (w_f_a, w_f_b, dt_bias) = (
        g.get("inputs", "W_f_a"),
        g.get("inputs", "W_f_b"),
        g.get("inputs", "dt_bias"),
    );
    let (a_log, w_b) = (g.get("inputs", "A_log"), g.get("inputs", "W_b"));
    let (w_g_a, w_g_b) = (g.get("inputs", "W_g_a"), g.get("inputs", "W_g_b"));
    let (o_norm_w, w_o) = (g.get("inputs", "o_norm_w"), g.get("inputs", "W_o"));
    let w = KdaWeights {
        w_f_a: &w_f_a,
        w_f_b: &w_f_b,
        dt_bias: &dt_bias,
        a_log: &a_log,
        w_b: &w_b,
        w_g_a: &w_g_a,
        w_g_b: &w_g_b,
        o_norm_w: &o_norm_w,
        w_o: &w_o,
    };
    let mut state = vec![0.0f32; d.heads * d.head_dim * d.head_dim];
    let out = kda_reference_layer(
        &g.get("inputs", "hidden_states"),
        &g.get("inputs", "q_in"),
        &g.get("inputs", "k_in"),
        &g.get("inputs", "v_in"),
        &w,
        d,
        g.scalar("lower_bound"),
        g.scalar("rms_eps"),
        &mut state,
    );
    assert_close("layer_out", &out, &g.get("outputs", "layer_out"), 5e-6);
}

/// The goldens carry HF's own prefill-vs-decode agreement. If HF ever stops agreeing with itself,
/// every tolerance above is meaningless.
#[test]
fn hf_self_checks_were_clean() {
    let g = Golden::load();
    for (k, v) in g.0["hf_self_checks"]
        .as_object()
        .expect("self checks present")
    {
        let d = v.as_f64().unwrap();
        assert!(d < 1e-6, "HF self-check {k} = {d:e} is not clean");
    }
}

// ─────────────────────────── prenorm contract (Slice 4 trap)

/// `kda_recurrent` normalises q/k internally (HF's contract); `kda_recurrent_prenorm`
/// does not (Atlas's contract, where the conv fuses the L2). Passing already-normalised
/// vectors to the former is *nearly* a no-op in fp32, which is exactly what makes it
/// dangerous: on bf16-rounded inputs the second normalisation RESTORES the norm the
/// rounding destroyed, so the reference silently disagrees with a kernel that consumes
/// pre-normalised q/k — and it looks like a kernel bug.
#[test]
fn prenorm_and_internal_norm_agree_only_on_unit_input() {
    let g = Golden::load();
    let d = g.dims();
    let (q, k, v) = (
        g.get("inputs", "q_in"),
        g.get("inputs", "k_in"),
        g.get("inputs", "v_in"),
    );
    let (gate, beta) = (g.get("outputs", "gate"), g.get("outputs", "beta"));
    let n = d.heads * d.head_dim * d.head_dim;

    // Routed through the internal-norm entry point == prenorm on the normalised vectors.
    let mut s1 = vec![0.0f32; n];
    let a = kda_recurrent(&q, &k, &v, &gate, &beta, d, &mut s1);
    let mut s2 = vec![0.0f32; n];
    let b = kda_recurrent_prenorm(
        &g.get("outputs", "q_l2"),
        &g.get("outputs", "k_l2"),
        &v,
        &gate,
        &beta,
        d,
        &mut s2,
    );
    assert_close("prenorm == internal-norm on unit input", &b, &a, 2e-6);

    // Double-normalising a bf16-rounded unit vector is NOT a no-op: it measurably moves.
    let round = |x: &[f32]| -> Vec<f32> {
        x.iter()
            .map(|v| f32::from_bits((v.to_bits() + 0x8000) & 0xFFFF_0000))
            .collect()
    };
    let (qr, kr) = (
        round(&g.get("outputs", "q_l2")),
        round(&g.get("outputs", "k_l2")),
    );
    let mut s3 = vec![0.0f32; n];
    let pre = kda_recurrent_prenorm(&qr, &kr, &v, &gate, &beta, d, &mut s3);
    let mut s4 = vec![0.0f32; n];
    let dbl = kda_recurrent(&qr, &kr, &v, &gate, &beta, d, &mut s4);
    let spread = pre
        .iter()
        .zip(&dbl)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        spread > 1e-6,
        "double-normalisation must be detectable on bf16-rounded input, saw {spread:e}"
    );
}

/// Slice 6: the chunked path needs the same prenorm split the recurrent path got in Slice 4.
///
/// Atlas's prefill L2 (`l2_norm_bf16`) writes **bf16**, so the chunked kernel is always fed
/// already-normalised bf16 vectors. Routing those through [`kda_chunked`] would re-normalise
/// them and silently restore the rounding the bf16 write destroyed — the Slice-4 hazard, which
/// looked exactly like a kernel bug when it appeared on the recurrent path.
#[test]
fn chunked_prenorm_and_internal_norm_agree_only_on_unit_input() {
    let g = Golden::load();
    let d = g.dims();
    let (q, k, v) = (
        g.get("inputs", "q_in"),
        g.get("inputs", "k_in"),
        g.get("inputs", "v_in"),
    );
    let (gate, beta) = (g.get("outputs", "gate"), g.get("outputs", "beta"));
    let n = d.heads * d.head_dim * d.head_dim;
    let chunk = 2;

    let mut s1 = vec![0.0f32; n];
    let a = kda_chunked(&q, &k, &v, &gate, &beta, d, chunk, &mut s1);
    let mut s2 = vec![0.0f32; n];
    let b = kda_chunked_prenorm(
        &g.get("outputs", "q_l2"),
        &g.get("outputs", "k_l2"),
        &v,
        &gate,
        &beta,
        d,
        chunk,
        &mut s2,
    );
    assert_close(
        "chunked prenorm == internal-norm on unit input",
        &b,
        &a,
        2e-6,
    );

    let round = |x: &[f32]| -> Vec<f32> {
        x.iter()
            .map(|v| f32::from_bits((v.to_bits() + 0x8000) & 0xFFFF_0000))
            .collect()
    };
    let (qr, kr) = (
        round(&g.get("outputs", "q_l2")),
        round(&g.get("outputs", "k_l2")),
    );
    let mut s3 = vec![0.0f32; n];
    let pre = kda_chunked_prenorm(&qr, &kr, &v, &gate, &beta, d, chunk, &mut s3);
    let mut s4 = vec![0.0f32; n];
    let dbl = kda_chunked(&qr, &kr, &v, &gate, &beta, d, chunk, &mut s4);
    let spread = pre
        .iter()
        .zip(&dbl)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        spread > 1e-6,
        "double-normalisation must be detectable on bf16-rounded input, saw {spread:e}"
    );
}
