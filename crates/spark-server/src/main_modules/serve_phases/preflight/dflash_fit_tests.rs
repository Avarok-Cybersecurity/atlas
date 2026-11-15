// SPDX-License-Identifier: AGPL-3.0-only

//! The DFlash fit, pinned at the Qwen3.5-4B + `Qwen3.5-4B-DFlash` numbers of
//! an earlier GPU run on an RTX 4060 Ti 16 GB: the default
//! reserve it refused, the arithmetic that makes the same card hold eight
//! streams with the row on, and the proof that the row off changes nothing.
//! CPU only: every figure comes from the SSOT functions the allocators use.

use super::*;
use crate::cli;
use avarok_core::config::{LayerType, ModelConfig};
use clap::Parser as _;
use spark_model::ssm_reserve::{SsmRollbackMode, mtp_state_slots_with, ssm_pool_reserve_bytes};

use super::super::spec_headroom::{
    DFLASH_CTX_CAP, dflash_ctx_accumulator_bytes_with_cap, dflash_headroom_with_cap,
};

/// Pre-load free memory in the log: "14.84 GB free" (GiB).
const FREE_PRE_LOAD: usize = 15_196 * MIB;
/// "8.68 GB checkpoint - 2.11 GB the load-time routes free = 6.57 GB resident".
const RESIDENT: usize = 6_727 * MIB;
/// `Qwen3.5-4B-DFlash`'s `model.safetensors`: "69 tensors, 1268851712 bytes".
const DRAFTER_WEIGHTS: usize = 1_268_851_712;

fn args(extra: &[&str]) -> cli::ServeArgs {
    let mut v = vec!["spark", "Qwen/Qwen3.5-4B", "--dflash"];
    if !extra.contains(&"--max-seq-len") {
        v.extend_from_slice(&["--max-seq-len", "8192"]);
    }
    if !extra.contains(&"--max-batch-size") {
        v.extend_from_slice(&["--max-batch-size", "8"]);
    }
    v.extend_from_slice(extra);
    cli::ServeArgs::parse_from(v)
}

/// Qwen3.5-4B as the preflight and the buffer arena see it (config.json of
/// the checkpoint): hidden 2,560, 32 layers in the 3:1 GDN/full pattern, 16
/// query and 4 KV heads of 256 with a gated Q, FFN 9,216, vocab 248,320.
fn qwen35_4b() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "qwen3_5".to_string();
    c.num_experts = 0;
    c.num_experts_per_tok = 0;
    c.hidden_size = 2560;
    c.intermediate_size = 9216;
    c.vocab_size = 248_320;
    c.num_hidden_layers = 32;
    c.layer_types = (0..32)
        .map(|i| {
            if i % 4 == 3 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            }
        })
        .collect();
    c.num_attention_heads = 16;
    c.num_key_value_heads = 4;
    c.head_dim = 256;
    c.attn_gated = true;
    c.mrope_interleaved = true;
    c.linear_num_key_heads = 16;
    c.linear_key_head_dim = 128;
    c.linear_num_value_heads = 32;
    c.linear_value_head_dim = 128;
    c.linear_conv_kernel_dim = 4;
    c.mamba_num_heads = 0;
    c.mamba_head_dim = 0;
    c
}

/// `preflight_reserve`'s `gdn_two_phase_bytes` at `tokens` prefill tokens.
fn gdn_two_phase(c: &ModelConfig, tokens: usize) -> usize {
    let key_dim = c.linear_num_key_heads * c.linear_key_head_dim;
    let value_dim = c.linear_num_value_heads * c.linear_value_head_dim;
    let nv = c.linear_num_value_heads;
    let conv_dim = key_dim * 2 + value_dim;
    tokens * conv_dim * 2 + tokens * nv * 2 * 4 + tokens * value_dim * 2 + tokens * value_dim * 2
}

fn arena(a: &cli::ServeArgs, c: &ModelConfig, tokens: usize) -> usize {
    spark_runtime::buffers::BufferSizes::from_config(
        c,
        tokens,
        a.max_seq_len,
        a.block_size,
        a.max_batch_size,
    )
    .total_bytes()
}

/// The SSM pools `preflight_reserve` prices for a DFlash serve at gamma `g`.
fn pools(a: &cli::ServeArgs, c: &ModelConfig, g: usize) -> usize {
    ssm_pool_reserve_bytes(
        a.max_batch_size,
        c.num_ssm_layers() * c.ssm_h_state_bytes(),
        c.num_ssm_layers() * c.ssm_conv_state_bytes(),
        true,
        g,
        mtp_state_slots_with(a.max_batch_size, 32, false),
        true,
        false,
        SsmRollbackMode::Snapshot,
    )
}

/// The drafter's own numbers: 6 layers, 8 KV heads of 128, hidden 2,560.
fn drafter() -> usize {
    drafter_bytes_from(DRAFTER_WEIGHTS, 6, 8, 128, 2560, 2560, 8192, true)
}

// ── The row is off unless asked ──────────────────────────────────────────

#[test]
fn the_row_is_off_unless_the_env_asks() {
    assert!(!enabled_from(None));
    for off in ["", "0", "false", "off", "no", " OFF "] {
        assert!(!enabled_from(Some(off)), "{off:?}");
    }
    for on in ["1", "true", "on", "yes"] {
        assert!(enabled_from(Some(on)), "{on:?}");
    }
    assert_eq!(ENV, "AVAROK_DFLASH_FIT");
}

#[test]
fn the_default_pricing_is_untouched_by_the_cap_parameter() {
    // The compiled-in cap is what the preflight priced before the row existed;
    // an 8,192-token serve is the whole window under either spelling.
    let (a, c) = (args(&[]), qwen35_4b());
    assert_eq!(
        dflash_ctx_accumulator_bytes_with_cap(&a, &c, DFLASH_CTX_CAP),
        8 * 8192 * 8 * 2560 * 2
    );
    assert_eq!(DFLASH_CTX_CAP, 16 * 1024);
}

// ── What the refused serve asked for ─────────────────────────────────────

#[test]
fn the_default_8_stream_reserve_is_the_13_32_gib_the_boot_log_refused() {
    let (a, c) = (args(&[]), qwen35_4b());
    // Default gamma: the drafter's block 16 + 2, clamped to 15.
    let gamma = spark_model::layers::qwen3_ssm::default_dflash_gamma(16);
    assert_eq!(gamma, 15);
    // 8 base blobs of 51 MiB, then 8 slots of 15 x 48 MiB h + 16 x 3 MiB conv
    // + a 51 MiB checkpoint = 819 MiB each.
    let base = 8 * 51 * MIB;
    let per_slot = 15 * 48 * MIB + 16 * 3 * MIB + 51 * MIB;
    assert_eq!(per_slot, 819 * MIB);
    assert_eq!(pools(&a, &c, gamma), base + 8 * per_slot);
    assert_eq!(pools(&a, &c, gamma) / MIB, 6960);
    // Headroom: 512 + 921 + 2,560 + 1,024 = 5,017, clamped to the flat 4,096.
    let (headroom, why) = dflash_headroom_with_cap(&a, &c, DFLASH_CTX_CAP);
    assert_eq!(headroom / MIB, 4096, "{why}");
    assert!(why.contains("clamped"), "{why}");
    assert_eq!(gdn_two_phase(&c, 8192) / MIB, 258);
    // The log's `inference=11314 MB`, to the megabyte.
    let inference = pools(&a, &c, gamma) + headroom + gdn_two_phase(&c, 8192);
    assert_eq!(inference / MIB, 11_314);
    // Its `buffer_arena=2327 MB`: the calibration of this file's config.
    let arena = arena(&a, &c, 8192);
    assert!(
        (2327 * MIB - 24 * MIB..=2327 * MIB + 24 * MIB).contains(&arena),
        "the 4B arena at 8,192 tokens is {} MiB, the boot log says 2,327",
        arena / MIB
    );
    // Against the 8.08 GiB the audit found: nowhere near.
    assert!((inference + arena) / MIB > 13_600);
    assert!(inference + arena > FREE_PRE_LOAD - RESIDENT);
}

// ── The same card with the row on ────────────────────────────────────────

/// The fit's terms at 8 streams with `--max-prefill-tokens 1024` and a
/// 1,024-token ctx window (`AVAROK_DFLASH_CTX_CAP=1024`), bf16 KV as the bench
/// serves: everything the real serve allocates beside the weights.
fn terms_at_8_streams(a: &cli::ServeArgs, c: &ModelConfig) -> Terms {
    // KV floor: 8 sequences x 4,096 tokens x (8 attention layers x K and V x 4
    // heads x 256 x 2 B) = 1 GiB.
    let kv_per_token = c.num_attention_layers() * 2 * c.num_key_value_heads * c.head_dim * 2;
    assert_eq!(kv_per_token, 32 * 1024);
    Terms {
        beside_weights: FREE_PRE_LOAD - RESIDENT,
        fixed: gdn_two_phase(c, 1024),
        arena: arena(a, c, 1024),
        drafter: drafter(),
        kv_floor: a.max_batch_size * 4096 * kv_per_token,
    }
}

#[test]
fn eight_streams_fit_a_16_gib_card_with_the_row_on() {
    let (a, c) = (args(&["--max-prefill-tokens", "1024"]), qwen35_4b());
    let t = terms_at_8_streams(&a, &c);
    let cap = 1024;
    let d = pick(
        15,
        &t,
        |g| pools(&a, &c, g),
        |g| {
            let mut ag = a.clone();
            ag.dflash_gamma = Some(g);
            dflash_headroom_with_cap(&ag, &c, cap).0
        },
    );
    // The default width needs more than the card has, the chosen one fits,
    // and the next width up does not (it is the widest that fits).
    let g = d.gamma.expect("some gamma fits at 8 streams");
    assert!((MIN_GAMMA..15).contains(&g), "gamma {g}");
    assert!(d.need_default > t.beside_weights, "{d:?}");
    assert!(d.need_chosen <= t.beside_weights, "{d:?}");
    let at = |g: usize| {
        let mut ag = a.clone();
        ag.dflash_gamma = Some(g);
        t.need(pools(&a, &c, g), dflash_headroom_with_cap(&ag, &c, cap).0)
    };
    assert_eq!(at(g), d.need_chosen);
    assert!(at(g + 1) > t.beside_weights, "gamma {} would fit", g + 1);
    // Print the arithmetic once, for the report (cargo test -- --nocapture).
    println!(
        "dflash fit at 8 streams: gamma {g}; need {} MiB of {} MiB beside the weights \
         (default 15 needs {} MiB): pools {} + headroom {} + fixed {} + arena {} + drafter {} \
         + KV floor {} + margin {}",
        d.need_chosen / MIB,
        t.beside_weights / MIB,
        d.need_default / MIB,
        pools(&a, &c, g) / MIB,
        {
            let mut ag = a.clone();
            ag.dflash_gamma = Some(g);
            dflash_headroom_with_cap(&ag, &c, cap).0 / MIB
        },
        t.fixed / MIB,
        t.arena / MIB,
        t.drafter / MIB,
        t.kv_floor / MIB,
        FIT_MARGIN / MIB,
    );
}

#[test]
fn the_other_two_levers_are_needed_at_8_streams() {
    // Gamma alone is not enough: with the 16,384-token ctx cap and the
    // 8,192-token prefill arena (2,327 MiB, 258 MiB of GDN prefill) even the
    // floor gamma needs more than the card has, so the fit chooses nothing
    // and says so rather than pinning a width that still cannot boot.
    let (a, c) = (args(&[]), qwen35_4b());
    let t = Terms {
        fixed: gdn_two_phase(&c, 8192),
        arena: arena(&a, &c, 8192),
        ..terms_at_8_streams(&a, &c)
    };
    let d = pick(
        15,
        &t,
        |g| pools(&a, &c, g),
        |g| {
            let mut ag = a.clone();
            ag.dflash_gamma = Some(g);
            dflash_headroom_with_cap(&ag, &c, DFLASH_CTX_CAP).0
        },
    );
    assert_eq!(d.gamma, None, "{d:?}");
    assert!(d.need_chosen > t.beside_weights + 3 * 1024 * MIB, "{d:?}");
}

#[test]
fn a_one_stream_serve_keeps_the_default_gamma() {
    // At one stream the default (gamma 15) fits with the row on, so the fit
    // leaves it alone and pins nothing.
    let (a, c) = (
        args(&["--max-batch-size", "1", "--max-prefill-tokens", "1024"]),
        qwen35_4b(),
    );
    let t = terms_at_8_streams(&a, &c);
    let d = pick(
        15,
        &t,
        |g| pools(&a, &c, g),
        |g| {
            let mut ag = a.clone();
            ag.dflash_gamma = Some(g);
            dflash_headroom_with_cap(&ag, &c, 1024).0
        },
    );
    assert_eq!(d.gamma, Some(15), "{d:?}");
}

#[test]
fn a_card_that_holds_no_width_says_so_and_leaves_the_default() {
    let (a, c) = (args(&["--max-prefill-tokens", "1024"]), qwen35_4b());
    let mut t = terms_at_8_streams(&a, &c);
    t.beside_weights = 3 * 1024 * MIB;
    let d = pick(15, &t, |g| pools(&a, &c, g), |_| 0);
    assert_eq!(d.gamma, None);
    assert!(d.need_chosen > t.beside_weights);
}

#[test]
fn the_widest_width_that_fits_is_chosen_and_never_below_the_floor() {
    // need(g) = 100 g + the margin, so 700 bytes beside the margin hold gamma 7.
    let t = Terms {
        beside_weights: FIT_MARGIN + 700,
        fixed: 0,
        arena: 0,
        drafter: 0,
        kv_floor: 0,
    };
    let d = pick(15, &t, |g| 100 * g, |_| 0);
    assert_eq!(d.gamma, Some(7), "{d:?}");
    assert_eq!(d.need_chosen, FIT_MARGIN + 700);
    let tight = Terms {
        beside_weights: FIT_MARGIN + 250,
        ..t
    };
    // Gamma 2 would fit but 3 is the floor: nothing is chosen, and the need
    // reported is the floor's.
    let d = pick(15, &tight, |g| 100 * g, |_| 0);
    assert_eq!(d.gamma, None, "{d:?}");
    assert_eq!(d.need_chosen, FIT_MARGIN + 100 * MIN_GAMMA);
    // A default narrower than the floor is itself the only candidate.
    let d = pick(2, &t, |g| 100 * g, |_| 0);
    assert_eq!(d.gamma, Some(2));
}

#[test]
fn the_pools_follow_the_gamma_they_are_asked_for() {
    let (a, c) = (args(&[]), qwen35_4b());
    // gamma 7: 8 base blobs + 8 slots x (7 x 48 + 8 x 3 + 51) = 411 MiB.
    assert_eq!(pools(&a, &c, 7), (8 * 51 + 8 * 411) * MIB);
    assert!(pools(&a, &c, 7) < pools(&a, &c, 15));
    assert_eq!(pools(&a, &c, 15) - pools(&a, &c, 14), 8 * 51 * MIB);
}

#[test]
fn the_ctx_window_cap_prices_what_the_head_allocates() {
    let (a, c) = (args(&[]), qwen35_4b());
    // 8 sequences x window x 8 taps x 2,560 x 2 B.
    assert_eq!(
        dflash_ctx_accumulator_bytes_with_cap(&a, &c, 1024) / MIB,
        8 * 1024 * 8 * 2560 * 2 / MIB
    );
    assert_eq!(
        dflash_ctx_accumulator_bytes_with_cap(&a, &c, 1024) / MIB,
        320
    );
    // Uncapped (`AVAROK_DFLASH_CTX_CAP=0`) is the whole --max-seq-len.
    assert_eq!(
        dflash_ctx_accumulator_bytes_with_cap(&a, &c, usize::MAX) / MIB,
        2560
    );
    // And a cap below the headroom's clamp lets the headroom fall under 4 GiB.
    let (h, why) = dflash_headroom_with_cap(&a, &c, 1024);
    assert!(h < 4096 * MIB, "{why}");
}

#[test]
fn the_drafter_footprint_is_what_the_kv_budget_stage_reserves() {
    // factory::build's dflash_reserve (drafter KV 201 MiB, fused KV 60,
    // capture 40, FP8 mirrors half the weights, 300 MiB scratch) plus the
    // weights themselves, which load before the KV sizing.
    let kv = 8192 * 6 * 2 * (8 * 128) * 2;
    let fused = 6 * 2 * (8 * 128) * 2560 * 2;
    let capture = 8192 * 2560 * 2;
    let want = DRAFTER_WEIGHTS + kv + fused + capture + DRAFTER_WEIGHTS / 2 + 300 * MIB;
    assert_eq!(drafter(), want);
    assert_eq!(
        drafter_bytes_from(DRAFTER_WEIGHTS, 6, 8, 128, 2560, 2560, 8192, false),
        want - DRAFTER_WEIGHTS / 2
    );
}

#[test]
fn the_row_off_plans_nothing() {
    // The env is unset under the test runner; the preflight then takes the
    // branch that skips the fit before it reads anything.
    if enabled() {
        return;
    }
    let (a, c) = (args(&[]), qwen35_4b());
    let i = PlanInputs {
        beside_weights: 8 * 1024 * MIB,
        kv_floor: 1024 * MIB,
        arena: 2327 * MIB,
        fixed: 258 * MIB,
        h_blob: c.num_ssm_layers() * c.ssm_h_state_bytes(),
        conv_blob: c.num_ssm_layers() * c.ssm_conv_state_bytes(),
        mtp_state_slots: 8,
        h_f16_pool: false,
        rollback: SsmRollbackMode::Snapshot,
        default_gamma: 15,
    };
    assert_eq!(plan(&a, &c, &i), None);
}
