// SPDX-License-Identifier: AGPL-3.0-only
//! Tests of the rotary embedding (YaRN inverse frequencies, interleaved M-RoPE, cos/sin tables) against hand values,
//! and of the text-config parser on an ARC-1.0 shaped `text_config`.

use serde_json::json;

use super::encoder::{
    LayerType, RopeConfig, RopeTables, RopeVariant, TextConfig, inv_freq_for, mrope_angles,
    plain_inv_freq, yarn_inv_freq, yarn_mscale,
};

#[test]
fn plain_frequencies_by_hand() {
    let f = plain_inv_freq(10_000.0, 4);
    assert_eq!(f.len(), 2);
    assert!((f[0] - 1.0).abs() < 1e-15);
    assert!((f[1] - 0.01).abs() < 1e-15);
}

#[test]
fn yarn_frequencies_by_hand() {
    // dim 4, theta 1e4, factor 4, original max 2048, beta 32 and 1: correction dims 0.504 and 1.257, so low 0 and
    // high 2; the ramp is [0, 0.5], giving [1 (extrapolated), 0.5 * 0.0025 + 0.5 * 0.01 = 0.00625].
    let f = yarn_inv_freq(10_000.0, 4, 4.0, 2048.0, 32.0, 1.0);
    assert!((f[0] - 1.0).abs() < 1e-12, "{f:?}");
    assert!((f[1] - 0.006_25).abs() < 1e-12, "{f:?}");
    assert!((yarn_mscale(4.0) - (1.0 + 0.1 * 4f64.ln())).abs() < 1e-15);
    assert!((yarn_mscale(4.0) - 1.138_629_4).abs() < 1e-6);
    assert_eq!(yarn_mscale(1.0), 1.0);
    // a factor of 1 is plain RoPE
    let flat = yarn_inv_freq(10_000.0, 4, 1.0, 2048.0, 32.0, 1.0);
    assert!((flat[0] - 1.0).abs() < 1e-12 && (flat[1] - 0.01).abs() < 1e-12);
}

fn arc_rope() -> RopeConfig {
    RopeConfig {
        theta: 1e7,
        partial_rotary_factor: 0.25,
        factor: 4.0,
        original_max_position: 262_144.0,
        max_position: 262_144.0,
        mrope_section: [11, 11, 10],
    }
}

#[test]
fn variants_of_the_arc_config() {
    let cfg = arc_rope();
    let (yarn, ms) = inv_freq_for(&cfg, 64, RopeVariant::Yarn);
    let (plain, ms_plain) = inv_freq_for(&cfg, 64, RopeVariant::Plain);
    let (ratio, ms_ratio) = inv_freq_for(&cfg, 64, RopeVariant::YarnRatio);
    assert_eq!(yarn.len(), 32);
    assert!((ms - 1.138_629_4).abs() < 1e-6);
    assert_eq!(ms_plain, 1.0);
    // max == original, so the ratio variant has factor 1: plain frequencies and no attention scaling
    assert_eq!(ms_ratio, 1.0);
    for (a, b) in plain.iter().zip(&ratio) {
        assert!((a - b).abs() < 1e-18 * a.abs().max(1.0));
    }
    // the highest-frequency dimension is untouched by YaRN, the lowest is divided by the factor
    assert!((yarn[0] - plain[0]).abs() < 1e-15);
    assert!((yarn[31] - plain[31] / 4.0).abs() < 1e-3 * plain[31]);
}

#[test]
fn interleaved_mrope_picks_the_right_stream() {
    // section [3, 2, 2]: height overwrites indexes 1 and 4, width indexes 2 and 5; the rest stays temporal
    let pos = [vec![1.0], vec![10.0], vec![100.0]];
    let angles = mrope_angles(&pos, &[1.0; 8], [3, 2, 2]);
    assert_eq!(
        angles[0],
        vec![1.0, 10.0, 100.0, 1.0, 10.0, 100.0, 1.0, 1.0]
    );
    // equal streams (text) change nothing
    let equal = [vec![3.0], vec![3.0], vec![3.0]];
    let inv = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03];
    let a = mrope_angles(&equal, &inv, [2, 2, 1]);
    for (x, f) in a[0].iter().zip(&inv) {
        assert!((x - 3.0 * f).abs() < 1e-15);
    }
}

#[test]
fn rotation_by_hand() {
    // one token, rotary width 4, angles [pi/2, 0]: cos = [0, 1, 0, 1], sin = [1, 0, 1, 0]
    let tables = RopeTables::from_angles(&[vec![std::f64::consts::FRAC_PI_2, 0.0]], 4, 1.0);
    let mut head = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    tables.apply(&mut head, 0);
    // pairs (0, 2) and (1, 3): (1, 3) rotates a quarter turn to (-3, 1); (2, 4) is not rotated; index 4 passes through
    assert!((head[0] + 3.0).abs() < 1e-6);
    assert!((head[2] - 1.0).abs() < 1e-6);
    assert!((head[1] - 2.0).abs() < 1e-6);
    assert!((head[3] - 4.0).abs() < 1e-6);
    assert_eq!(head[4], 5.0);
    // the attention scaling multiplies cos and sin
    let scaled = RopeTables::from_angles(&[vec![0.0, 0.0]], 4, 2.0);
    assert!(
        scaled.cos.iter().all(|&c| (c - 2.0).abs() < 1e-6)
            && scaled.sin.iter().all(|&s| s.abs() < 1e-6)
    );
}

#[test]
fn text_tables_have_the_partial_rotary_width_and_position_zero_is_identity() {
    let t = RopeTables::for_text(&arc_rope(), 256, 5, RopeVariant::Yarn);
    assert_eq!(t.rotary_dim, 64);
    assert_eq!(t.cos.len(), 5 * 64);
    let mut head: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
    let before = head.clone();
    t.apply(&mut head, 0);
    // position 0: angle 0, so only the attention scaling applies to the rotated part and the rest is untouched
    let ms = yarn_mscale(4.0) as f32;
    for i in 0..32 {
        assert!((head[i] - before[i] * ms).abs() < 1e-5);
    }
    assert_eq!(&head[64..], &before[64..]);
}

#[test]
fn text_config_parses_an_arc_shaped_text_config() {
    let v = json!({
        "attn_output_gate": true, "head_dim": 256, "hidden_size": 1024, "intermediate_size": 3584,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
        "linear_num_value_heads": 16, "linear_value_head_dim": 128, "max_position_embeddings": 262144,
        "num_attention_heads": 8, "num_key_value_heads": 2, "rms_norm_eps": 1e-6, "vocab_size": 248320,
        "rope_parameters": {"factor": 4.0, "mrope_interleaved": true, "mrope_section": [11, 11, 10],
            "original_max_position_embeddings": 262144, "partial_rotary_factor": 0.25, "rope_theta": 10000000,
            "rope_type": "yarn"}
    });
    let c = TextConfig::from_json(&v).expect("parses");
    assert_eq!(
        c.layer_types,
        vec![
            LayerType::Linear,
            LayerType::Linear,
            LayerType::Linear,
            LayerType::Full
        ]
    );
    assert_eq!(
        (
            c.hidden,
            c.heads,
            c.kv_heads,
            c.head_dim,
            c.lin_k_heads,
            c.conv_kernel
        ),
        (1024, 8, 2, 256, 16, 4)
    );
    assert!(c.attn_gate);
    assert_eq!(c.rope.theta, 1e7);
    assert_eq!(c.rope.factor, 4.0);
    assert_eq!(c.rope.mrope_section, [11, 11, 10]);
    let mut bad = v.clone();
    bad["layer_types"][0] = json!("mystery");
    assert!(TextConfig::from_json(&bad).is_err());
    let mut missing = v;
    missing
        .as_object_mut()
        .expect("object")
        .remove("hidden_size");
    assert!(TextConfig::from_json(&missing).is_err());
}
