// SPDX-License-Identifier: AGPL-3.0-only
//! Second half of the serve-arg validation tests, split from `validate_tests.rs`.
//!
//! Mechanical: this stack took the file from 441 to 506 lines and the cap is
//! 500. Every test below is a byte-exact copy.

// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for [`super`] (cross-flag CLI validation). A sibling file via
//! `#[path]` — the `helpers.rs`/`helpers_tests.rs` idiom — so `validate.rs`
//! stays under the 500-line cap; module position (child of `validate`) is
//! unchanged, so `super::*` paths are untouched.
use super::*;
use clap::Parser;

/// Parse a `spark serve ...` command line into `ServeArgs` for testing.
fn parse(extra: &[&str]) -> ServeArgs {
    let mut argv = vec!["spark", "serve", "dummy/model", "--model-name", "dummy"];
    argv.extend_from_slice(extra);
    match super::super::Cli::parse_from(argv).command {
        super::super::Command::Serve(a) => a,
        super::super::Command::Benchmark(_)
        | super::super::Command::DumpServeOptions
        | super::super::Command::SyncRecipes
        | super::super::Command::Doctor => {
            unreachable!("this test parses a serve command")
        }
    }
}

#[test]
fn flagship_recipe_is_accepted() {
    // The canonical 35B flagship serve recipe (PR #278) passes
    // `--kv-cache-dtype bf16 --kv-high-precision-layers auto` together —
    // redundant but valid. The validator must NOT reject it.
    assert!(
        validate_serve_args(&parse(&[
            "--kv-cache-dtype",
            "bf16",
            "--lm-head-dtype",
            "nvfp4",
            "--kv-high-precision-layers",
            "auto",
            "--scheduling-policy",
            "slai",
            "--speculative",
            "--num-drafts",
            "1",
            "--mtp-quantization",
            "bf16",
            "--enable-prefix-caching",
        ]))
        .is_ok()
    );
}

#[test]
fn enum_typos_are_rejected() {
    let err = validate_serve_args(&parse(&["--scheduling-policy", "fifoo"])).unwrap_err();
    assert!(err.contains("--scheduling-policy"));
    assert!(err.contains("fifo, slai"));
}

#[test]
fn multiple_violations_all_reported() {
    let err = validate_serve_args(&parse(&[
        "--require-auth",
        "--num-drafts",
        "3",
        "--rank",
        "5",
        "--world-size",
        "2",
    ]))
    .unwrap_err();
    assert!(err.contains("[1]"));
    assert!(err.contains("[2]"));
    assert!(err.contains("[3]"));
}

#[test]
fn gpu_mem_util_range_enforced() {
    assert!(validate_serve_args(&parse(&["--gpu-memory-utilization", "1.5"])).is_err());
    assert!(validate_serve_args(&parse(&["--gpu-memory-utilization", "0.0"])).is_err());
    assert!(validate_serve_args(&parse(&["--gpu-memory-utilization", "0.9"])).is_ok());
}

/// The dgx2 silent-flag bug class: a MODEL.toml-backed flag whose clap
/// declaration carries a `default_value` makes an explicitly passed
/// engine-default value ("--num-drafts 1", "--kv-cache-dtype fp8")
/// indistinguishable from an omitted flag, so the MODEL.toml default silently
/// wins over the user's pin. These flags must parse to `None` when omitted
/// and `Some` when passed — re-adding a clap default resurrects the bug.
#[test]
fn model_toml_backed_flags_distinguish_omitted_from_explicit() {
    let omitted = parse(&[]);
    assert_eq!(omitted.num_drafts, None);
    assert_eq!(omitted.kv_cache_dtype, None);
    assert_eq!(omitted.fp8_kv_calibration_tokens, None);

    let explicit = parse(&[
        "--num-drafts",
        "1",
        "--kv-cache-dtype",
        "fp8",
        "--fp8-kv-calibration-tokens",
        "0",
    ]);
    assert_eq!(explicit.num_drafts, Some(1));
    assert_eq!(explicit.kv_cache_dtype.as_deref(), Some("fp8"));
    // Explicit 0 must survive parsing: it force-disables calibration on a
    // model whose MODEL.toml enables it, which the old `usize` field with
    // `default_value_t = 0` could not express.
    assert_eq!(explicit.fp8_kv_calibration_tokens, Some(0));
}

#[test]
fn f16_pool_is_a_published_dtype_and_inherits_every_f16_rule() {
    use spark_model::layers::qwen3_ssm::ssm_h_dtype_bits;

    // SSOT: the validator and `publish_kernel_flags` decode through the SAME
    // function, so a spelling accepted here publishes exactly these bits.
    assert_eq!(ssm_h_dtype_bits(None), (false, false));
    assert_eq!(ssm_h_dtype_bits(Some("f32")), (false, false));
    assert_eq!(ssm_h_dtype_bits(Some("f16")), (true, false));
    // f16-pool is f16 PLUS the narrow pool — never the pool without the bits,
    // which would be an FP32 write into a 2-byte-sized slot.
    assert_eq!(ssm_h_dtype_bits(Some("f16-pool")), (true, true));

    // Accepted by the enum check, and serveable with the fused-norm arm.
    assert!(
        validate_serve_args(&parse(&["--ssm-h-dtype", "f16-pool", "--gdn-fused-norm"])).is_ok(),
        "the supported stage-3 pairing"
    );
    // Every f16 rule binds on it too, because it IS f16.
    assert!(validate_serve_args(&parse(&["--ssm-h-dtype", "f16-pool"])).is_err());
    assert!(
        validate_serve_args(&parse(&[
            "--exact-verify",
            "--ssm-h-dtype",
            "f16-pool",
            "--gdn-fused-norm",
        ]))
        .is_err()
    );
    // An unknown spelling is still rejected by the enum check, and decodes to
    // FP32 rather than to something half-enabled.
    assert!(validate_serve_args(&parse(&["--ssm-h-dtype", "f16pool"])).is_err());
    assert_eq!(ssm_h_dtype_bits(Some("f16pool")), (false, false));
}

#[test]
fn dflash_refuses_the_f16_h_state_only_at_the_untwinned_width() {
    // WAS a blanket refusal of `--dflash` under f16, correct while the only
    // DFlash verify width was gamma+1 = 17 and `gated_delta_rule_wy17` had no
    // FP16 twin. #817 added twins for K=5..16, so the refusal narrowed to the
    // width that still lacks one. What did NOT change is why it exists: an
    // FP32 kernel over an FP16 h-state emits fluent garbage, not a fault.
    for dtype in ["f16", "f16-pool"] {
        let err = validate_serve_args(&parse(&[
            "--dflash",
            "--dflash-gamma",
            "16",
            "--ssm-h-dtype",
            dtype,
            "--gdn-fused-norm",
        ]))
        .unwrap_err();
        assert!(err.contains("dflash-gamma"), "{dtype}: {err}");
        // ...and the twinned widths are now allowed under the same dtype,
        // which is the half of this contract that #817 could not reach.
        assert!(
            validate_serve_args(&parse(&[
                "--dflash",
                "--dflash-gamma",
                "10",
                "--ssm-h-dtype",
                dtype,
                "--gdn-fused-norm",
            ]))
            .is_ok(),
            "{dtype}: gamma 10 (width 11) has a twin and must be allowed"
        );
    }
    // POSITIVE controls: each side alone stays valid.
    assert!(validate_serve_args(&parse(&["--dflash"])).is_ok());
    assert!(validate_serve_args(&parse(&["--ssm-h-dtype", "f16", "--gdn-fused-norm"])).is_ok());
    // f32 (the default) is unaffected.
    assert!(
        validate_serve_args(&parse(&["--dflash", "--ssm-h-dtype", "f32"])).is_ok(),
        "the FP32 h-state has always been DFlash's supported pairing"
    );
}

// ── DFlash under the f16 h-state pool ────────────────────────────────────
//
// The width band matters to the byte, and it is easy to get wrong in the
// direction that does not fault: the DFlash verify width is K = gamma + 1,
// FP16 h-state twins exist for K = 5..16, so the last SAFE gamma is 15.
// Gamma 16 is K=17 — `gated_delta_rule_wy17`, the one family with no twin and
// an FP32-element intermediate stride. An FP32 kernel over an FP16 h-state
// emits fluent garbage rather than an error, which is the whole reason this
// pair is validated at all.

fn f16_dflash(gamma: Option<&str>) -> ServeArgs {
    let mut argv = vec![
        "--ssm-h-dtype",
        "f16-pool",
        "--gdn-fused-norm",
        "--dflash",
        "--draft-model",
        "some/drafter",
    ];
    if let Some(g) = gamma {
        argv.push("--dflash-gamma");
        argv.push(g);
    }
    parse(&argv)
}

#[test]
fn dflash_under_the_f16_pool_is_allowed_up_to_the_last_twinned_width() {
    // The regression this pins: a blanket refusal of `--dflash` under f16
    // made the FP16 twins unreachable through the CLI — three gamma values
    // measured on 2026-08-30 produced three refusals and zero tokens, on the
    // branch that ADDED the twins.
    for g in ["4", "8", "10", "15"] {
        assert!(
            validate_serve_args(&f16_dflash(Some(g))).is_ok(),
            "gamma {g} (verify width {}) has an FP16 twin and must be allowed",
            g.parse::<usize>().unwrap() + 1
        );
    }
}

#[test]
fn dflash_gamma_16_is_refused_because_width_17_has_no_twin() {
    // The off-by-one: a `> 16` test admits gamma 16, whose width is 17.
    let v = validate_serve_args(&f16_dflash(Some("16")));
    let msg = format!(
        "{:#}",
        v.expect_err("gamma 16 is verify width 17 — no FP16 twin")
    );
    assert!(
        msg.contains("f16") && msg.contains("dflash-gamma"),
        "the refusal must name both halves of the incompatible pair: {msg}"
    );
}

#[test]
fn an_unset_gamma_is_left_to_the_runtime_backstop() {
    // Gamma resolves from the drafter checkpoint, which is not readable at
    // validate time. Refusing here would reject working configurations;
    // a width with no twin is caught at the first verify step instead, where
    // the sequential fallback bails rather than reading FP16 bits as FP32.
    assert!(validate_serve_args(&f16_dflash(None)).is_ok());
}
