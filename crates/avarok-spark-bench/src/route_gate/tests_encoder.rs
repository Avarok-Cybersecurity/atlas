// SPDX-License-Identifier: AGPL-3.0-only
//! Layer-level tests of the CPU encoder with random weights at tiny dimensions (no model file needed): the gated
//! DeltaNet recurrence against a naive f64 loop, gated attention against a naive softmax, the operators against hand
//! values, causality of both token mixers, and a two-layer stack end to end.

use super::encoder::{
    AttnWeights, EmbedSource, Encoder, EncoderOptions, GdnWeights, Layer, LayerType, Lin, Mixer,
    RopeConfig, RopeTables, RopeVariant, TextConfig, attn_forward, causal_attention,
    delta_rule_head, gdn_forward, maybe_round, rms_norm_1p, rms_norm_gated, round_bf16,
};
use super::mt::NumpyRandom;

fn rnd(rng: &mut NumpyRandom, n: usize, scale: f64) -> Vec<f32> {
    rng.randn_vec(n)
        .into_iter()
        .map(|v| (v * scale) as f32)
        .collect()
}

fn lin(rng: &mut NumpyRandom, out: usize, inp: usize, scale: f64) -> Lin {
    Lin::from_f32(out, inp, &rnd(rng, out * inp, scale))
}

fn bf(v: u16) -> f64 {
    f64::from(f32::from_bits(u32::from(v) << 16))
}

#[test]
fn matmul_matches_a_naive_loop_and_bf16_rounding_is_applied_on_request() {
    let mut rng = NumpyRandom::new(1);
    let (o, i, t) = (3, 11, 4);
    let l = lin(&mut rng, o, i, 0.5);
    let x = rnd(&mut rng, t * i, 1.0);
    let y = l.forward(&x, t, false);
    for tt in 0..t {
        for oo in 0..o {
            let want: f64 = (0..i)
                .map(|k| f64::from(x[tt * i + k]) * bf(l.w[oo * i + k]))
                .sum();
            assert!((f64::from(y[tt * o + oo]) - want).abs() < 1e-4, "{tt} {oo}");
        }
    }
    let r = l.forward(&x, t, true);
    for (a, b) in r.iter().zip(&y) {
        assert_eq!(*a, half::bf16::from_f32(*b).to_f32());
    }
    let mut v = vec![1.0f32 + 2f32.powi(-9), 1.0 + 3.0 * 2f32.powi(-9)];
    round_bf16(&mut v); // spacing 2^-7: the first rounds down, the second up (round to nearest even)
    assert_eq!(v, vec![1.0, 1.0 + 2f32.powi(-7)]);
    assert_eq!(
        maybe_round(vec![1.0 + 2f32.powi(-9)], false),
        vec![1.0 + 2f32.powi(-9)]
    );
}

#[test]
fn norms_by_hand() {
    // x = [3, 4]: mean square 12.5, so the scale is 1 / sqrt(12.5)
    let inv = 1.0 / 12.5f32.sqrt();
    let y = rms_norm_1p(&[3.0, 4.0], &[0.0, 1.0], 0.0, false);
    assert!(
        (y[0] - 3.0 * inv).abs() < 1e-6 && (y[1] - 4.0 * inv * 2.0).abs() < 1e-6,
        "{y:?}"
    );
    let silu1 = 1.0 / (1.0 + (-1.0f32).exp());
    let g = rms_norm_gated(&[3.0, 4.0], &[1.0, 1.0], &[1.0, 1.0], 0.0, false);
    assert!(
        (g[0] - 3.0 * inv * silu1).abs() < 1e-6 && (g[1] - 4.0 * inv * silu1).abs() < 1e-6,
        "{g:?}"
    );
    let zero_gate = rms_norm_gated(&[3.0, 4.0], &[0.0, 0.0], &[1.0, 1.0], 0.0, false);
    assert_eq!(zero_gate, vec![0.0, 0.0]);
}

#[test]
fn delta_rule_matches_a_naive_f64_loop() {
    let mut rng = NumpyRandom::new(2);
    let (dk, dv, t) = (5, 3, 7);
    let q = rnd(&mut rng, t * dk, 0.5);
    let k = rnd(&mut rng, t * dk, 0.5);
    let v = rnd(&mut rng, t * dv, 1.0);
    let g: Vec<f32> = (0..t).map(|i| -0.1 - 0.05 * i as f32).collect();
    let beta: Vec<f32> = (0..t).map(|i| 0.3 + 0.1 * i as f32).collect();
    let got = delta_rule_head(&q, &k, &v, &g, &beta, dk, dv);
    let mut s = vec![vec![0.0f64; dv]; dk];
    for i in 0..t {
        for row in s.iter_mut() {
            for e in row.iter_mut() {
                *e *= f64::from(g[i]).exp();
            }
        }
        let kv: Vec<f64> = (0..dv)
            .map(|j| (0..dk).map(|a| s[a][j] * f64::from(k[i * dk + a])).sum())
            .collect();
        for j in 0..dv {
            let d = (f64::from(v[i * dv + j]) - kv[j]) * f64::from(beta[i]);
            for a in 0..dk {
                s[a][j] += f64::from(k[i * dk + a]) * d;
            }
        }
        for j in 0..dv {
            let o: f64 = (0..dk).map(|a| s[a][j] * f64::from(q[i * dk + a])).sum();
            assert!(
                (f64::from(got[i * dv + j]) - o).abs() < 1e-4,
                "step {i} col {j}"
            );
        }
    }
    // a zero write strength never changes the state, so the output is zero
    let zero = delta_rule_head(&q, &k, &v, &g, &vec![0.0; t], dk, dv);
    assert!(zero.iter().all(|&x| x == 0.0));
}

#[test]
fn causal_attention_matches_a_naive_softmax() {
    let mut rng = NumpyRandom::new(3);
    let (t, hd) = (6, 4);
    let q = rnd(&mut rng, t * hd, 1.0);
    let k = rnd(&mut rng, t * hd, 1.0);
    let v = rnd(&mut rng, t * hd, 1.0);
    let got = causal_attention(&q, &k, &v, t, hd);
    for i in 0..t {
        let scores: Vec<f64> = (0..=i)
            .map(|s| {
                (0..hd)
                    .map(|d| f64::from(q[i * hd + d]) * f64::from(k[s * hd + d]))
                    .sum::<f64>()
                    / (hd as f64).sqrt()
            })
            .collect();
        let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let total: f64 = scores.iter().map(|s| (s - m).exp()).sum();
        for d in 0..hd {
            let want: f64 = scores
                .iter()
                .enumerate()
                .map(|(s, sc)| (sc - m).exp() / total * f64::from(v[s * hd + d]))
                .sum();
            assert!((f64::from(got[i * hd + d]) - want).abs() < 1e-5, "{i} {d}");
        }
    }
    // the first token attends only to itself
    for d in 0..hd {
        assert!((got[d] - v[d]).abs() < 1e-6);
    }
}

fn tiny_config() -> TextConfig {
    TextConfig {
        hidden: 8,
        layer_types: vec![LayerType::Linear, LayerType::Full],
        inter: 12,
        eps: 1e-6,
        vocab: 20,
        lin_k_heads: 2,
        lin_v_heads: 2,
        lin_k_dim: 4,
        lin_v_dim: 4,
        conv_kernel: 3,
        heads: 2,
        kv_heads: 1,
        head_dim: 8,
        attn_gate: true,
        rope: RopeConfig {
            theta: 1e4,
            partial_rotary_factor: 0.5,
            factor: 4.0,
            original_max_position: 64.0,
            max_position: 64.0,
            mrope_section: [1, 1, 1],
        },
    }
}

fn tiny_gdn(rng: &mut NumpyRandom, c: &TextConfig) -> GdnWeights {
    let (key, value) = (c.lin_k_heads * c.lin_k_dim, c.lin_v_heads * c.lin_v_dim);
    let conv_dim = 2 * key + value;
    GdnWeights {
        in_qkv: lin(rng, conv_dim, c.hidden, 0.4),
        in_z: lin(rng, value, c.hidden, 0.4),
        in_b: lin(rng, c.lin_v_heads, c.hidden, 0.4),
        in_a: lin(rng, c.lin_v_heads, c.hidden, 0.4),
        conv: rnd(rng, conv_dim * c.conv_kernel, 0.5),
        a_log: vec![-1.0, -0.5],
        dt_bias: vec![0.1, -0.1],
        norm: vec![1.0; c.lin_v_dim],
        out: lin(rng, c.hidden, value, 0.4),
    }
}

fn tiny_attn(rng: &mut NumpyRandom, c: &TextConfig) -> AttnWeights {
    AttnWeights {
        q: lin(rng, c.heads * c.head_dim * 2, c.hidden, 0.4),
        k: lin(rng, c.kv_heads * c.head_dim, c.hidden, 0.4),
        v: lin(rng, c.kv_heads * c.head_dim, c.hidden, 0.4),
        o: lin(rng, c.hidden, c.heads * c.head_dim, 0.4),
        q_norm: rnd(rng, c.head_dim, 0.1),
        k_norm: rnd(rng, c.head_dim, 0.1),
    }
}

fn close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= tol, "index {i}: {x} vs {y}");
    }
}

#[test]
fn gdn_and_attention_layers_are_causal_and_finite() {
    let c = tiny_config();
    let mut rng = NumpyRandom::new(4);
    let (gdn, attn) = (tiny_gdn(&mut rng, &c), tiny_attn(&mut rng, &c));
    let x = rnd(&mut rng, 6 * c.hidden, 1.0);
    let rope = RopeTables::for_text(&c.rope, c.head_dim, 6, RopeVariant::Yarn);
    for bf16 in [false, true] {
        let full_g = gdn_forward(&c, &gdn, &x, 6, bf16);
        let part_g = gdn_forward(&c, &gdn, &x[..4 * c.hidden], 4, bf16);
        assert!(full_g.iter().all(|v| v.is_finite()));
        close(&full_g[..4 * c.hidden], &part_g, 1e-5);
        let rope4 = RopeTables::for_text(&c.rope, c.head_dim, 4, RopeVariant::Yarn);
        let full_a = attn_forward(&c, &attn, &x, 6, &rope, bf16);
        let part_a = attn_forward(&c, &attn, &x[..4 * c.hidden], 4, &rope4, bf16);
        assert!(full_a.iter().all(|v| v.is_finite()));
        close(&full_a[..4 * c.hidden], &part_a, 1e-5);
    }
    // the layers do something: changing a later token moves later outputs only
    let mut x2 = x.clone();
    x2[5 * c.hidden] += 1.0;
    let g2 = gdn_forward(&c, &gdn, &x2, 6, false);
    let g1 = gdn_forward(&c, &gdn, &x, 6, false);
    close(&g1[..5 * c.hidden], &g2[..5 * c.hidden], 1e-5);
    assert!(
        g1[5 * c.hidden..]
            .iter()
            .zip(&g2[5 * c.hidden..])
            .any(|(a, b)| (a - b).abs() > 1e-4)
    );
}

fn tiny_encoder(rng: &mut NumpyRandom, c: &TextConfig, opts: EncoderOptions) -> Encoder {
    let mut layers = Vec::new();
    for ty in &c.layer_types {
        layers.push(Layer {
            input_norm: rnd(rng, c.hidden, 0.1),
            post_norm: rnd(rng, c.hidden, 0.1),
            mixer: match ty {
                LayerType::Linear => Mixer::Gdn(tiny_gdn(rng, c)),
                LayerType::Full => Mixer::Attn(tiny_attn(rng, c)),
            },
            gate: lin(rng, c.inter, c.hidden, 0.4),
            up: lin(rng, c.inter, c.hidden, 0.4),
            down: lin(rng, c.hidden, c.inter, 0.4),
        });
    }
    let table = rnd(rng, c.vocab * c.hidden, 1.0);
    let bits: Vec<u16> = table
        .iter()
        .map(|&v| half::bf16::from_f32(v).to_bits())
        .collect();
    Encoder {
        cfg: c.clone(),
        opts,
        embed: EmbedSource::Owned(bits),
        layers,
        final_norm: rnd(rng, c.hidden, 0.1),
    }
}

#[test]
fn two_layer_stack_runs_end_to_end() {
    let c = tiny_config();
    let mut rng = NumpyRandom::new(5);
    let enc = tiny_encoder(&mut rng, &c, EncoderOptions::default());
    let ids = [3u32, 7, 7, 19, 0];
    let h = enc.hidden(&ids).expect("forward");
    assert_eq!(h.len(), ids.len() * c.hidden);
    assert!(h.iter().all(|v| v.is_finite()));
    // causal: the prefix gives the same rows
    let prefix = enc.hidden(&ids[..3]).expect("prefix");
    close(&h[..3 * c.hidden], &prefix, 1e-4);
    // the same token at different positions gives different rows (the stack mixes context)
    assert!(
        h[c.hidden..2 * c.hidden]
            .iter()
            .zip(&h[2 * c.hidden..3 * c.hidden])
            .any(|(a, b)| (a - b).abs() > 1e-4)
    );
    // errors: empty input and an id outside the vocabulary
    assert!(enc.hidden(&[]).is_err());
    assert!(enc.hidden(&[20]).is_err());
    // bf16 emulation changes values slightly but not the shape or finiteness
    let mut enc16 = tiny_encoder(
        &mut NumpyRandom::new(5),
        &c,
        EncoderOptions {
            bf16: true,
            rope: RopeVariant::Yarn,
        },
    );
    let h16 = enc16.hidden(&ids).expect("bf16 forward");
    assert!(h16.iter().all(|v| v.is_finite()));
    assert!(h16.iter().zip(&h).any(|(a, b)| a != b));
    enc16.opts.rope = RopeVariant::Plain;
    assert!(enc16.hidden(&ids).is_ok());
}
