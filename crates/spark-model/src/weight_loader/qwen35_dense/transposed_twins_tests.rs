// SPDX-License-Identifier: AGPL-3.0-only

//! Exact-integer pins for the transposed-twin projection and the lever table.
//!
//! The numbers asserted below are the ones `docs/porting/r9700-residency.md`
//! reports from the R9700 ledger sweep of `unsloth/Qwen3.8-27B-NVFP4`
//! (8.96 GiB dense FFN, 2.90 GiB SSM, 0.88 GiB attention, 12.74 GiB total).
//! This test is the join between that measurement and the arithmetic the
//! `auto` probe decides on: if the projection drifts from the ledger, the probe
//! is deciding about a model that is not the one being loaded.
//!
//! No GPU, no checkpoint, no environment.

use super::*;
use atlas_core::config::{LayerType, ModelConfig};

const MIB: usize = 1024 * 1024;

/// `unsloth/Qwen3.8-27B-NVFP4` at the shapes
/// `kernels/r9700/qwen3.8-27b/MODEL.toml` declares: 64 layers on a 4-cycle
/// (16 full attention, 48 GDN), hidden 5120, intermediate 17408, head_dim 256,
/// 24 q heads, 4 kv heads, output-gated attention.
///
/// The GDN head geometry (16x128 key heads, 48x128 value heads) is from the
/// checkpoint's own `config.json`, as `predicted_residency_tests.rs` explains:
/// no MODEL.toml carries those fields.
fn qwen38_27b() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "qwen3_5".to_string();
    c.num_experts = 0;
    c.num_experts_per_tok = 0;
    c.moe_intermediate_size = 0;
    c.hidden_size = 5120;
    c.intermediate_size = 17408;
    c.num_hidden_layers = 64;
    c.num_attention_heads = 24;
    c.num_key_value_heads = 4;
    c.head_dim = 256;
    c.attn_gated = true;
    c.linear_num_key_heads = 16;
    c.linear_key_head_dim = 128;
    c.linear_num_value_heads = 48;
    c.linear_value_head_dim = 128;
    c.full_attention_interval = 4;
    c.layer_types = cycle4(64);
    c
}

/// `deepreinforce-ai/Ornith-1.0-9B` at the shapes
/// `kernels/r9700/ornith-1.0-9b/MODEL.toml` declares: 32 layers on a 4-cycle
/// (8 full attention, 24 GDN), hidden 4096, intermediate 12288, head_dim 256,
/// 16 q heads, 4 kv heads, `attn_output_gate = true`, GDN 16x128 key heads and
/// 32x128 value heads.
fn ornith_9b() -> ModelConfig {
    let mut c = qwen38_27b();
    c.hidden_size = 4096;
    c.intermediate_size = 12288;
    c.num_hidden_layers = 32;
    c.num_attention_heads = 16;
    c.linear_num_value_heads = 32;
    c.layer_types = cycle4(32);
    c
}

/// The 3:1 (linear, linear, linear, full) pattern both models declare.
fn cycle4(n: usize) -> Vec<LayerType> {
    (0..n)
        .map(|i| {
            if (i + 1).is_multiple_of(4) {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            }
        })
        .collect()
}

/// ★ The join with the measured ledger. Every term is asserted in whole MiB
/// because every term IS a whole number of MiB at these shapes — a projection
/// that lands on a fraction has taken a wrong dimension somewhere.
#[test]
fn qwen38_27b_twins_reproduce_the_r9700_ledger() {
    let c = qwen38_27b();
    let b = projected_bytes(&c, &c.layer_types);

    // 3 x nvfp4_bytes(17408, 5120) = 143.4375 MiB per layer, x64 layers.
    assert_eq!(b.ffn, 9180 * MIB, "dense FFN twins are 8.96 GiB");
    // qkvz [16384, 5120] = 45 MiB + out_proj [5120, 6144] = 16.875 MiB, x48.
    assert_eq!(b.ssm, 2970 * MIB, "SSM twins are 2.90 GiB");
    // q [12288, 5120] = 33.75 + k,v [1024, 5120] = 2.8125 each
    // + o [5120, 6144] = 16.875, x16.
    assert_eq!(b.attn, 900 * MIB, "attention twins are 0.88 GiB");
    assert_eq!(
        b.total(),
        13050 * MIB,
        "12.74 GiB, the doc's headline number"
    );

    // The doc's GiB figures, to the hundredth it prints them at.
    let gib = |x: usize| (x as f64 / (1024.0 * 1024.0 * 1024.0) * 100.0).round() / 100.0;
    assert_eq!(gib(b.ffn), 8.96);
    assert_eq!(gib(b.ssm), 2.90);
    assert_eq!(gib(b.attn), 0.88);
    assert_eq!(gib(b.total()), 12.74);
}

/// The small sibling this target serves while the 27B does not fit. Not a
/// measured number — nothing has been served on this board — but it is the
/// same arithmetic, and it is what the `auto` probe will compare against
/// 31.9 GB, so it is worth being able to read off a test rather than a serve.
#[test]
fn ornith_9b_twins_are_a_third_of_the_27b() {
    let c = ornith_9b();
    let b = projected_bytes(&c, &c.layer_types);
    // 3 x nvfp4_bytes(12288, 4096) = 81 MiB per layer, x32.
    assert_eq!(b.ffn, 2592 * MIB);
    // qkvz [12288, 4096] = 27 MiB + out_proj [4096, 4096] = 9 MiB, x24.
    assert_eq!(b.ssm, 864 * MIB);
    // q [8192, 4096] = 18 + k,v [1024, 4096] = 2.25 each + o [4096, 4096] = 9, x8.
    assert_eq!(b.attn, 252 * MIB);
    assert_eq!(b.total(), 3708 * MIB, "3.62 GiB");
}

/// A model with no GDN layers must price no GDN twins, and one with no full
/// attention no attention twins. The FFN term counts EVERY layer either way —
/// this architecture carries a dense FFN on every mixer.
#[test]
fn the_projection_follows_the_layer_type_list() {
    let mut c = qwen38_27b();
    c.layer_types = vec![LayerType::FullAttention; 64];
    let b = projected_bytes(&c, &c.layer_types);
    assert_eq!(
        b.ffn,
        9180 * MIB,
        "the FFN term is per LAYER, not per mixer"
    );
    assert_eq!(b.ssm, 0);
    assert_eq!(b.attn, 3600 * MIB, "56.25 MiB x64 full-attention layers");

    c.layer_types = vec![LayerType::LinearAttention; 64];
    let b = projected_bytes(&c, &c.layer_types);
    assert_eq!(b.ffn, 9180 * MIB);
    assert_eq!(b.ssm, 3960 * MIB, "61.875 MiB x64 GDN layers");
    assert_eq!(b.attn, 0);
}

/// `intermediate_size` unset falls back to `moe_intermediate_size`, exactly as
/// `load_dense_ffn` does. A projection that read 0 here would price the whole
/// FFN family at nothing and let `auto` build twins that do not fit.
#[test]
fn the_ffn_width_falls_back_the_way_the_loader_does() {
    let mut c = qwen38_27b();
    c.intermediate_size = 0;
    c.moe_intermediate_size = 17408;
    assert_eq!(projected_bytes(&c, &c.layer_types).ffn, 9180 * MIB);
}

/// ★ SCALE's unset default is `Never`, not the probe. The R9700 measurement of
/// 2026-09-17 has the twin GEMM arm SLOWER than the untransposed one on gfx1201
/// (~1 TFLOP/s against ~4), so there is no residency-versus-speed trade left for
/// `auto` to weigh: the twins are only 12.74 GiB. Every non-SCALE target is
/// untouched and still builds them.
#[test]
fn the_default_follows_the_target_and_the_knob_overrides_it() {
    assert_eq!(
        decide(None, true),
        TwinPolicy::Never,
        "the twin arm measured slower than the plain one on gfx1201"
    );
    assert_eq!(
        decide(None, false),
        TwinPolicy::Always,
        "an NVIDIA build with the variable unset must build every twin"
    );
    // Explicit wins on both targets, in all three directions.
    assert_eq!(decide(Some("1"), true), TwinPolicy::Always);
    assert_eq!(decide(Some("0"), false), TwinPolicy::Never);
    assert_eq!(decide(Some("auto"), false), TwinPolicy::Auto);
    assert_eq!(decide(Some("auto"), true), TwinPolicy::Auto);
    // Anything else is "unset", never a silent skip.
    assert_eq!(decide(Some("true"), false), TwinPolicy::Always);
    assert_eq!(decide(Some("yes"), true), TwinPolicy::Never);
    assert_eq!(decide(Some(""), false), TwinPolicy::Always);
}

/// The lever string the load line prints has to be the one an operator types.
#[test]
fn every_policy_names_itself_as_an_assignment() {
    for p in [TwinPolicy::Always, TwinPolicy::Never, TwinPolicy::Auto] {
        let s = p.lever();
        assert!(
            s.starts_with("ATLAS_LOAD_TRANSPOSED_TWINS="),
            "{s} is not something an operator can type"
        );
        assert_eq!(
            decide(s.split_once('=').map(|(_, v)| v), false),
            p,
            "{s} does not round-trip through decide()"
        );
    }
}

/// The `auto` arithmetic, without a GPU: the reserve is part of the need, and
/// the comparison is strict. A board with exactly `twins + reserve` free has
/// nothing left over and must not build.
#[test]
fn auto_requires_the_twins_plus_the_serve_reserve() {
    let c = qwen38_27b();
    let need = projected_bytes(&c, &c.layer_types).total() + SERVE_RESERVE_BYTES;
    // 12.74 GiB of twins plus a 4 GiB reserve is 16.74 GiB. The R9700 has
    // ~9.2 GB free at this point on this checkpoint, so it cannot.
    assert!(
        9_200_000_000usize <= need,
        "the R9700's measured post-checkpoint free must not clear the bar"
    );
    // A GB10 with 121 GB of unified memory clears it with room to spare.
    assert!(110_000_000_000usize > need);
}
