// SPDX-License-Identifier: AGPL-3.0-only

//! Exact-integer pins for the pre-load derived-residency prediction (#915).
//!
//! The shapes are `kernels/gb10/qwen3.8-27b/MODEL.toml`'s — that is the only
//! MODEL.toml for this checkpoint on `main`, and its architecture block is
//! target-independent: the Hopper copy that the H100 rounds ran under repeats
//! those fields verbatim and adds nothing but `expected_absent` entries. So
//! the fixture is the checkpoint the H100 rounds served, cited at a path that
//! exists in this tree. The number the prediction is judged against is the
//! loader's OWN summary line from round 6 (`h100-round6-report.md`, serve I):
//! `native FP8 dense residency: weights 28.75 GB, derived 4.24 GB`.
//!
//! No GPU, no checkpoint, no environment: `Fp8RouteInputs` is built by hand so
//! the decision table is pinned rather than the machine the test runs on.

use super::*;
use atlas_core::config::{LayerType, ModelConfig, QuantizationConfig};

use crate::layers::ops::GemmDispatch;

/// `Qwen/Qwen3.8-27B-FP8` at the shapes `kernels/gb10/qwen3.8-27b/MODEL.toml`
/// declares: 64 layers on a 4-cycle (16 full attention, 48 GDN), hidden 5120,
/// head_dim 256, 24 q heads, 4 kv heads, output-gated attention.
///
/// The GDN head geometry below — 16x128 key heads, 48x128 value heads — is
/// NOT from that file. No MODEL.toml carries the linear-attention head
/// fields; `ModelConfig` reads them from the checkpoint's `config.json`, and
/// they are reproduced here because the 83,886,080-byte fused `[QKV|Z]` term
/// is arithmetic over them.
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
    c.layer_types = (0..64)
        .map(|i| {
            if (i + 1) % 4 == 0 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            }
        })
        .collect();
    c.quantization_config = Some(QuantizationConfig {
        quant_method: "fp8".to_string(),
        quant_algo: String::new(),
        format: String::new(),
        ignore_modules: Vec::new(),
    });
    c
}

/// The route the round-6 serve booted in: `ATLAS_DENSE_FP8=1`, tp 1, a
/// config-declared FP8 checkpoint, no NVFP4 lever, both W8A8 prefill kernels
/// present in the hopper kernel set.
///
/// `ffn_gateup_fused` is FALSE here: round 6 predates the lever (#927), and
/// the round-6 receipt below is the join between this arithmetic and a serve
/// log that was printed before the arm existed.
fn round6_route() -> Fp8RouteInputs {
    Fp8RouteInputs {
        dense_fp8: true,
        tp_size: 1,
        declared_variant: Some(Nvfp4Variant::Fp8Dequanted),
        gdn_fp8: true,
        w8a8_prefill_kernels: true,
        route: RouteEnv {
            keep_nvfp4: false,
            dispatch: GemmDispatch::defaults(),
            attn_w4a4: false,
            attn_prefill_q_t: false,
        },
        ffn_gateup_fused: false,
        attn_qkv_fused: false,
    }
}

fn predicted(route: &Fp8RouteInputs) -> PredictedDerived {
    match predicted_derived_bytes(&qwen38_27b(), route) {
        DerivedBytesEstimate::NativeFp8Dense(p) => p,
        DerivedBytesEstimate::Unavailable(why) => panic!("expected a prediction, got: {why}"),
    }
}

/// The load-bearing assertion: the prediction reproduces the number the H100
/// serve log printed, to well inside the 5 % the brief allows.
///
/// 16 x (K twin + V twin) + 48 x (fused `[QKV|Z]` + its scales + out_proj
/// scales + interleaved `in_proj_ba`) = 4,242,882,560 B = 4.243 GB, against
/// round 6's `derived 4.24 GB`.
#[test]
fn the_27b_prediction_matches_the_round6_residency_summary() {
    let p = predicted(&round6_route());
    assert_eq!(
        p.attn_fp8_twins, 167_813_120,
        "16 layers x (K twin + V twin)"
    );
    assert_eq!(
        p.ssm_fp8_concat, 4_075_069_440,
        "48 GDN layers x 84,897,280"
    );
    assert_eq!(p.total(), 4_242_882_560);

    let measured = 4.24_f64;
    let predicted_gb = p.total() as f64 / 1e9;
    let err = (predicted_gb - measured).abs() / measured;
    assert!(
        err < 0.05,
        "predicted {predicted_gb:.3} GB vs the round-6 log's {measured} GB ({:.2}% off)",
        err * 100.0,
    );
}

/// Every term is the loader's own arithmetic, recomputed here independently
/// so a change to `ssm_concat_bytes` cannot silently redefine what is being
/// predicted.
#[test]
fn each_term_is_the_loaders_shape_arithmetic() {
    let c = qwen38_27b();
    // Attention: `fp8_twin_bytes(kv_n, hidden)` twice, K and V only.
    let kv_twin = fp8_residency::fp8_twin_bytes(4 * 256, 5120);
    assert_eq!(kv_twin, 5_244_160);
    assert_eq!(
        predicted(&round6_route()).attn_fp8_twins,
        16 * 2 * kv_twin as u64
    );

    // SSM: the fused weight dominates at 16384 x 5120 = 83,886,080 B/layer.
    assert_eq!(c.ssm_qkvz_size(), 16384);
    assert_eq!(c.ssm_qkv_size(), 10240);
    assert_eq!(c.ssm_z_size(), 6144);
    let per_layer = 83_886_080  // fused [QKV|Z] E4M3
        + (80 * 40 * 4 + 48 * 40 * 4)  // the two concatenated block-scale grids
        + 40 * 48 * 4   // out_proj block-scale grid
        + 48 * 2 * 5120 * 2; // in_proj_ba, [2*nv, hidden] BF16
    assert_eq!(per_layer, 84_897_280);
    assert_eq!(
        predicted(&round6_route()).ssm_fp8_concat,
        48 * per_layer as u64,
    );
}

/// THE RESIDENCY CONSTRAINT, as an equation (#927). The fused gate+up weight
/// is 178.3 MB x 64 layers = 11.4 GB — more than the bs32 KV budget has to
/// give — and it is spent ONLY because the same 11.4 GB comes back off the
/// checkpoint at `prune_after_load`. The prediction feeds `headroom.rs`, whose
/// `weights` term is the on-disk size and therefore still counts the two
/// tensors the loader released, so the two must cancel EXACTLY. If they ever
/// do not, an H100 serve quietly picks a smaller decode-rollback ring, or is
/// refused with `No memory left for KV cache`, and nothing says why.
#[test]
fn the_gateup_fusion_is_residency_neutral() {
    let mut route = round6_route();
    route.ffn_gateup_fused = true;
    let p = predicted(&route);
    let base = predicted(&round6_route());

    // Qwen3.8-27B: 2 x 17408 x 5120 = 178,257,920 B per layer, 64 layers.
    assert_eq!(p.ffn_gateup_fused, 64 * 178_257_920);
    assert_eq!(p.ffn_gateup_fused, 11_408_506_880, "11.4 GB, as briefed");
    assert_eq!(
        p.ffn_gateup_fused, p.ffn_gateup_pruned,
        "the fused weight IS the two store tensors copied side by side"
    );
    assert_eq!(
        p.total(),
        base.total(),
        "arming the fusion must not move the preflight yardstick by one byte"
    );
    assert!(p.twins.ffn_gateup_fused, "but the log still names it");
    assert!(!base.twins.ffn_gateup_fused);
}

/// THE RESIDENCY CONSTRAINT for the attention fusion (#927), as an equation.
/// The fused `[14336, 5120]` weight is 73.4 MB x 16 full-attention layers =
/// 1.17 GB, and it is spent ONLY because the same 1.17 GB comes back off the
/// checkpoint at `prune_after_load`. The prediction feeds `headroom.rs`, whose
/// `weights` term is the on-disk size and therefore still counts the three
/// tensors the loader released, so the two must cancel EXACTLY — otherwise an
/// H100 serve quietly picks a smaller decode-rollback ring, or is refused with
/// `No memory left for KV cache`, and nothing says why.
#[test]
fn the_qkv_fusion_is_residency_neutral() {
    let mut route = round6_route();
    route.attn_qkv_fused = true;
    let p = predicted(&route);
    let base = predicted(&round6_route());

    // Qwen3.8-27B: (12288 + 2 x 1024) x 5120 = 73,400,320 B per layer, over
    // the 16 FULL-attention layers (not all 64 — the other 48 are GDN).
    assert_eq!(p.attn_qkv_fused, 16 * 73_400_320);
    assert_eq!(p.attn_qkv_fused, 1_174_405_120, "1.17 GB, as briefed");
    assert_eq!(
        p.attn_qkv_fused, p.attn_qkv_pruned,
        "the fused weight IS the three store tensors copied side by side"
    );
    assert_eq!(
        p.total(),
        base.total(),
        "arming the fusion must not move the preflight yardstick by one byte"
    );
    assert!(p.twins.attn_qkv_fused, "but the log still names it");
    assert!(!base.twins.attn_qkv_fused);
}

/// The two fusions are priced independently and BOTH net to zero — the state
/// an H100 serve actually boots in once round 17 lands.
#[test]
fn both_fusions_together_still_move_nothing() {
    let mut route = round6_route();
    route.attn_qkv_fused = true;
    route.ffn_gateup_fused = true;
    let p = predicted(&route);
    assert_eq!(p.total(), predicted(&round6_route()).total());
    assert!(p.twins.attn_qkv_fused && p.twins.ffn_gateup_fused);
    assert!(p.twins.describe().contains("attn-qkv-fp8"));
}

/// A head geometry whose Q or K/V width is not a whole number of 128-blocks is
/// NOT priced for the attention fusion, because the loader will not build it:
/// the concat appends the `[N/128, K/128]` scale grids, and that is only the
/// fused grid when BOTH seams fall on a block boundary.
#[test]
fn a_kv_width_that_is_not_a_whole_block_grid_is_never_priced() {
    let mut c = qwen38_27b();
    // 4 kv heads x 100 = 400, not a multiple of 128.
    c.head_dim = 100;
    let mut route = round6_route();
    route.attn_qkv_fused = true;
    let p = match predicted_derived_bytes(&c, &route) {
        DerivedBytesEstimate::NativeFp8Dense(p) => p,
        DerivedBytesEstimate::Unavailable(why) => panic!("expected a prediction, got: {why}"),
    };
    assert_eq!(p.attn_qkv_fused, 0);
    assert_eq!(p.attn_qkv_pruned, 0);
    assert!(!p.twins.attn_qkv_fused);
}

/// An `intermediate_size` that is not a whole number of 128-blocks is NOT
/// priced for the fusion, because the loader will not build it: the concat
/// appends the two `[N/128, K/128]` scale grids, and that is only the fused
/// grid when the seam falls on a block boundary. The prediction and
/// `qwen35_dense::ffn_gateup_fused_selected` must agree on which models fuse —
/// a prediction that priced a fusion the loader declines would under-state the
/// KV headroom by 11.4 GB.
#[test]
fn a_width_that_is_not_a_whole_block_grid_is_never_priced() {
    let mut c = qwen38_27b();
    c.intermediate_size = 17408 - 64;
    let mut route = round6_route();
    route.ffn_gateup_fused = true;
    let p = match predicted_derived_bytes(&c, &route) {
        DerivedBytesEstimate::NativeFp8Dense(p) => p,
        DerivedBytesEstimate::Unavailable(why) => panic!("expected a prediction, got: {why}"),
    };
    assert_eq!(p.ffn_gateup_fused, 0);
    assert_eq!(p.ffn_gateup_pruned, 0);
    assert!(!p.twins.ffn_gateup_fused);
}

/// A target missing either W8A8 prefill kernel builds the Q and O twins too —
/// which is why the kernel set is an input and not an assumption. The 27B
/// pays 16 x (Q twin + O twin) = 1.51 GB more.
#[test]
fn a_target_without_the_w8a8_prefill_kernels_pays_for_the_q_and_o_twins() {
    let mut route = round6_route();
    route.w8a8_prefill_kernels = false;
    let p = predicted(&route);
    assert!(
        p.attn_twin_set.q && p.attn_twin_set.o,
        "{:?}",
        p.attn_twin_set
    );
    let q_twin = fp8_residency::fp8_twin_bytes(24 * 256 * 2, 5120) as u64;
    let o_twin = fp8_residency::fp8_twin_bytes(5120, 24 * 256) as u64;
    assert_eq!(
        p.attn_fp8_twins,
        predicted(&round6_route()).attn_fp8_twins + 16 * (q_twin + o_twin),
    );
    assert!(p.total() > predicted(&round6_route()).total());
}

/// `ATLAS_ATTN_PREFILL_Q_T=1` adds the Q twin on the W8A8-covered route: the
/// `cache_skip_qkv.rs:142` dispatch reads it per prefill.
#[test]
fn the_q_transpose_lever_adds_exactly_the_q_twin() {
    let mut route = round6_route();
    route.route.attn_prefill_q_t = true;
    let p = predicted(&route);
    assert!(p.attn_twin_set.q && !p.attn_twin_set.o);
    let q_twin = fp8_residency::fp8_twin_bytes(24 * 256 * 2, 5120) as u64;
    assert_eq!(
        p.attn_fp8_twins,
        predicted(&round6_route()).attn_fp8_twins + 16 * q_twin,
    );
}

/// `ATLAS_NO_GDN_FP8` takes the GDN layers off the fused-concat arm, so the
/// 4.03 GB that dominates the prediction is not spent.
#[test]
fn disabling_the_gdn_fp8_arm_drops_the_fused_concat_term() {
    let mut route = round6_route();
    route.gdn_fp8 = false;
    let p = predicted(&route);
    assert_eq!(p.ssm_fp8_concat, 0);
    assert!(!p.twins.ssm_fp8_concat);
    assert_eq!(p.total(), 167_813_120);
}

/// Every gate that must DECLINE rather than guess. Each of these routes
/// either does not reach the native-FP8 dense loader at all, or reaches it
/// with NVFP4 copies alive whose bytes `DerivedResidency` tallies through
/// `skip` — a quantity a prediction built on `keep` cannot price.
#[test]
fn the_prediction_declines_rather_than_guessing() {
    let cases: Vec<(&str, Box<dyn Fn(&mut Fp8RouteInputs)>, &str)> = vec![
        (
            "flag off",
            Box::new(|r: &mut Fp8RouteInputs| r.dense_fp8 = false),
            "ATLAS_DENSE_FP8 is not 1",
        ),
        (
            "tp > 1",
            Box::new(|r: &mut Fp8RouteInputs| r.tp_size = 2),
            "--tp-size > 1 takes the NVFP4 route",
        ),
        (
            "config declares nothing",
            Box::new(|r: &mut Fp8RouteInputs| r.declared_variant = None),
            "config.json does not declare a block-scaled FP8 checkpoint",
        ),
        (
            "config declares NVFP4",
            Box::new(|r: &mut Fp8RouteInputs| {
                r.declared_variant = Some(Nvfp4Variant::CompressedTensors)
            }),
            "config.json does not declare a block-scaled FP8 checkpoint",
        ),
        (
            "keep-nvfp4 escape hatch",
            Box::new(|r: &mut Fp8RouteInputs| r.route.keep_nvfp4 = true),
            "ATLAS_DENSE_FP8_KEEP_NVFP4 restores the pre-#915 fallback copies",
        ),
        (
            "W4A4 o_proj lever",
            Box::new(|r: &mut Fp8RouteInputs| r.route.attn_w4a4 = true),
            "an NVFP4 fallback lever (ATLAS_CUTLASS_NVFP4_* / ATLAS_ATTN_W4A4) is set",
        ),
    ];
    for (name, mutate, want) in cases {
        let mut route = round6_route();
        mutate(&mut route);
        assert_eq!(
            predicted_derived_bytes(&qwen38_27b(), &route),
            DerivedBytesEstimate::Unavailable(want),
            "case: {name}",
        );
    }
}

/// A model that is not the Qwen3.5-dense loader's is declined on the FIRST
/// gate — the arithmetic above is specific to this loader's derived copies.
#[test]
fn another_architecture_is_declined_on_the_loader_gate() {
    let other = ModelConfig::qwen3_next_80b_nvfp4();
    assert_eq!(
        predicted_derived_bytes(&other, &round6_route()),
        DerivedBytesEstimate::Unavailable("not the Qwen3.5-dense loader"),
    );
}

/// `bytes()` / `reason()` are what the preflight log reads; neither may
/// invent a value for the other's case.
#[test]
fn the_estimate_reports_either_bytes_or_a_reason_never_both() {
    let ok = predicted_derived_bytes(&qwen38_27b(), &round6_route());
    assert_eq!(ok.bytes(), Some(4_242_882_560));
    assert_eq!(ok.reason(), None);
    let no = DerivedBytesEstimate::Unavailable("because");
    assert_eq!(no.bytes(), None);
    assert_eq!(no.reason(), Some("because"));
}
