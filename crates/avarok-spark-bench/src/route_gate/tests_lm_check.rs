// SPDX-License-Identifier: AGPL-3.0-only
//! Tests of the full-LM oracle pieces at tiny dimensions (no model file needed): the tied LM head against a naive
//! loop, residual taps against a truncated stack, the next-token loss against a naive log-softmax, greedy decoding,
//! the weights-file lookup, the RoPE settings and the SHA-256 helper.

use super::encoder::{
    AttnWeights, EmbedSource, Encoder, EncoderOptions, GdnWeights, Layer, LayerType, Lin, Mixer,
    RopeConfig, RopeVariant, TextConfig, weights_file,
};
use super::lm_check::{
    RopeSetting, argmax, greedy, log_sum_exp, rel_l2_and_cosine, rms, sha_hex, token_nll, top_k,
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

fn bits(v: &[f32]) -> Vec<u16> {
    v.iter()
        .map(|&x| half::bf16::from_f32(x).to_bits())
        .collect()
}

fn widen(b: u16) -> f64 {
    f64::from(f32::from_bits(u32::from(b) << 16))
}

fn cfg() -> TextConfig {
    TextConfig {
        hidden: 8,
        layer_types: vec![LayerType::Linear, LayerType::Full, LayerType::Linear],
        inter: 12,
        eps: 1e-6,
        vocab: 24,
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
            factor: 1.0,
            original_max_position: 64.0,
            max_position: 64.0,
            mrope_section: [1, 1, 0],
        },
    }
}

fn encoder(seed: u64) -> Encoder {
    let c = cfg();
    let mut rng = NumpyRandom::new(seed);
    let (key, value) = (c.lin_k_heads * c.lin_k_dim, c.lin_v_heads * c.lin_v_dim);
    let mut layers = Vec::new();
    for ty in &c.layer_types {
        let mixer = match ty {
            LayerType::Linear => Mixer::Gdn(GdnWeights {
                in_qkv: lin(&mut rng, 2 * key + value, c.hidden, 0.4),
                in_z: lin(&mut rng, value, c.hidden, 0.4),
                in_b: lin(&mut rng, c.lin_v_heads, c.hidden, 0.4),
                in_a: lin(&mut rng, c.lin_v_heads, c.hidden, 0.4),
                conv: rnd(&mut rng, (2 * key + value) * c.conv_kernel, 0.5),
                a_log: vec![-1.0, -0.5],
                dt_bias: vec![0.1, -0.1],
                norm: vec![1.0; c.lin_v_dim],
                out: lin(&mut rng, c.hidden, value, 0.4),
            }),
            LayerType::Full => Mixer::Attn(AttnWeights {
                q: lin(&mut rng, c.heads * c.head_dim * 2, c.hidden, 0.4),
                k: lin(&mut rng, c.kv_heads * c.head_dim, c.hidden, 0.4),
                v: lin(&mut rng, c.kv_heads * c.head_dim, c.hidden, 0.4),
                o: lin(&mut rng, c.hidden, c.heads * c.head_dim, 0.4),
                q_norm: rnd(&mut rng, c.head_dim, 0.1),
                k_norm: rnd(&mut rng, c.head_dim, 0.1),
            }),
        };
        layers.push(Layer {
            input_norm: rnd(&mut rng, c.hidden, 0.1),
            post_norm: rnd(&mut rng, c.hidden, 0.1),
            mixer,
            gate: lin(&mut rng, c.inter, c.hidden, 0.4),
            up: lin(&mut rng, c.inter, c.hidden, 0.4),
            down: lin(&mut rng, c.hidden, c.inter, 0.4),
        });
    }
    let table = rnd(&mut rng, c.vocab * c.hidden, 1.0);
    Encoder {
        opts: EncoderOptions {
            bf16: false,
            rope: RopeVariant::Plain,
        },
        embed: EmbedSource::Owned(bits(&table)),
        final_norm: rnd(&mut rng, c.hidden, 0.1),
        layers,
        cfg: c,
    }
}

#[test]
fn tied_logits_match_a_naive_loop() {
    let mut rng = NumpyRandom::new(11);
    let (vocab, hid, t) = (13, 9, 3);
    let table = bits(&rnd(&mut rng, vocab * hid, 1.0));
    let x = rnd(&mut rng, t * hid, 1.0);
    let src = EmbedSource::Owned(table.clone());
    let got = src.logits(&x, t, hid, vocab).expect("logits");
    assert_eq!(got.len(), t * vocab);
    for tt in 0..t {
        for r in 0..vocab {
            let want: f64 = (0..hid)
                .map(|k| f64::from(x[tt * hid + k]) * widen(table[r * hid + k]))
                .sum();
            assert!(
                (f64::from(got[tt * vocab + r]) - want).abs() < 1e-4,
                "{tt} {r}"
            );
        }
    }
    assert!(src.logits(&x[..hid - 1], 1, hid, vocab).is_err());
    assert!(src.logits(&x[..hid], 1, hid, vocab + 1).is_err());
}

#[test]
fn residual_taps_equal_a_truncated_stack_and_hidden_is_the_final_norm() {
    let enc = encoder(3);
    let ids = [1u32, 5, 9, 2, 23, 0];
    let (h, snaps) = enc.residual(&ids, &[1, 2, 3]).expect("forward");
    assert_eq!(snaps.len(), 3);
    assert_eq!(snaps[2], h);
    for keep in 1..=2 {
        let mut short = encoder(3);
        short.layers.truncate(keep);
        let (hs, _) = short.residual(&ids, &[]).expect("short forward");
        assert_eq!(hs, snaps[keep - 1], "after {keep} layers");
    }
    assert_eq!(enc.hidden(&ids).expect("hidden"), enc.final_normed(&h));
    assert!(enc.residual(&ids, &[0]).is_err());
    assert!(enc.residual(&ids, &[4]).is_err());
    assert!(rms(&h) > 0.0 && rms(&h).is_finite());
}

#[test]
fn next_token_loss_matches_a_naive_log_softmax() {
    let enc = encoder(4);
    let ids = [3u32, 7, 7, 19, 0, 11];
    let (nll, hit) = token_nll(&enc, &ids).expect("nll");
    assert_eq!((nll.len(), hit.len()), (5, 5));
    let h = enc.hidden(&ids).expect("hidden");
    let (hid, v) = (enc.cfg.hidden, enc.cfg.vocab);
    let logits = enc.logits(&h, ids.len()).expect("logits");
    for i in 0..ids.len() - 1 {
        let row: Vec<f64> = logits[i * v..(i + 1) * v]
            .iter()
            .map(|&x| f64::from(x))
            .collect();
        let z: f64 = row.iter().map(|x| x.exp()).sum();
        let want = -(row[ids[i + 1] as usize].exp() / z).ln();
        assert!((nll[i] - want).abs() < 1e-4, "{i}: {} vs {want}", nll[i]);
        let best = argmax(&logits[i * v..(i + 1) * v]).0;
        assert_eq!(hit[i], best == ids[i + 1] as usize);
    }
    assert_eq!(h.len(), ids.len() * hid);
    assert!(token_nll(&enc, &ids[..1]).is_err());
}

#[test]
fn greedy_takes_the_argmax_extends_its_own_prefix_and_stops() {
    let enc = encoder(5);
    let prompt = [2u32, 4, 8];
    let three = greedy(&enc, &prompt, 3, &[]).expect("greedy");
    assert_eq!(three.len(), 3);
    let h = enc.hidden(&prompt).expect("hidden");
    let hid = enc.cfg.hidden;
    let first = argmax(&enc.logits(&h[2 * hid..], 1).expect("logits")).0 as u32;
    assert_eq!(three[0], first);
    assert_eq!(greedy(&enc, &prompt, 2, &[]).expect("two")[..], three[..2]);
    assert_eq!(
        greedy(&enc, &prompt, 3, &[first]).expect("stop"),
        vec![first]
    );
    assert_eq!(argmax(&[1.0, 3.0, 3.0, -1.0]), (1, 3.0));
}

#[test]
fn weights_file_takes_the_single_file_or_the_only_shard() {
    let d = tempfile::tempdir().expect("tempdir");
    assert!(weights_file(d.path()).is_err());
    let shard = d
        .path()
        .join("model.safetensors-00001-of-00001.safetensors");
    std::fs::write(&shard, b"x").expect("write");
    assert_eq!(weights_file(d.path()).expect("shard"), shard);
    std::fs::write(
        d.path()
            .join("model.safetensors-00002-of-00002.safetensors"),
        b"x",
    )
    .expect("write");
    assert!(weights_file(d.path()).is_err());
    let single = d.path().join("model.safetensors");
    std::fs::write(&single, b"x").expect("write");
    assert_eq!(weights_file(d.path()).expect("single"), single);
}

#[test]
fn rope_settings_map_to_the_stock_config_and_yarn_factor_four() {
    let mut enc = encoder(6);
    let base = enc.cfg.rope.clone();
    RopeSetting::YarnF4.apply(&mut enc, &base);
    assert_eq!(enc.opts.rope, RopeVariant::Yarn);
    assert_eq!(enc.cfg.rope.factor, 4.0);
    assert_eq!(enc.cfg.rope.original_max_position, 262_144.0);
    RopeSetting::Stock.apply(&mut enc, &base);
    assert_eq!(enc.opts.rope, RopeVariant::Plain);
    assert_eq!(enc.cfg.rope.factor, 1.0);
    let mut yarn_base = base.clone();
    yarn_base.factor = 4.0;
    RopeSetting::Stock.apply(&mut enc, &yarn_base);
    assert_eq!(enc.opts.rope, RopeVariant::Yarn);
    assert_eq!(RopeSetting::YarnF4.name(), "yarn-f4");
}

#[test]
fn top_k_and_log_sum_exp_by_hand() {
    let row = [0.5f32, 2.0, -1.0, 2.0, 1.0];
    assert_eq!(top_k(&row, 3), vec![(1, 2.0), (3, 2.0), (4, 1.0)]);
    let want = row.iter().map(|&x| f64::from(x).exp()).sum::<f64>().ln();
    assert!((log_sum_exp(&row) - want).abs() < 1e-12);
    assert!((log_sum_exp(&[1000.0, 1000.0]) - (1000.0 + 2f64.ln())).abs() < 1e-9);
    let (r, c) = rel_l2_and_cosine(&[3.0, 4.0], &[3.0, 4.0]);
    assert!(r.abs() < 1e-12 && (c - 1.0).abs() < 1e-12);
    let (r, c) = rel_l2_and_cosine(&[0.0, 5.0], &[3.0, 4.0]);
    assert!((r - (10f64).sqrt() / 5.0).abs() < 1e-12 && (c - 0.8).abs() < 1e-12);
}

#[test]
fn sha_hex_is_lowercase_sha256() {
    assert_eq!(
        sha_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}
