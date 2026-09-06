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
fn defaults_are_valid() {
    assert!(validate_serve_args(&parse(&[])).is_ok());
}

/// `--warmup-prompt` parses, is documented in QUICKSTART.md,
/// book/src/operations/server.md and docs/GB10_DEPLOYMENT_GUIDE.md, and its own
/// `--help` promises it "eliminates the cold-start TTFT penalty (~196ms)" — and
/// NOTHING in the workspace reads `args.warmup_prompt`. An operator following
/// the quickstart passed a path, measured an unchanged cold TTFT and had no
/// signal at all that the flag was inert.
#[test]
fn warmup_prompt_is_refused_because_nothing_implements_it() {
    let err = validate_serve_args(&parse(&["--warmup-prompt", "/tmp/warm.txt"]))
        .expect_err("an inert flag must not be accepted in silence");
    assert!(err.contains("--warmup-prompt"), "{err}");
    assert!(
        err.contains("not implemented"),
        "the operator must be told the flag does nothing, not merely that it is \
         disallowed: {err}"
    );
    assert!(
        err.contains("fix:"),
        "a diagnostic without a fix is half of one: {err}"
    );
    // The remedy has to be something they can actually do instead.
    assert!(
        err.contains("throwaway request"),
        "must name the way to warm the server: {err}"
    );
}

/// The guard above must not fire on a serve that never asked for it.
#[test]
fn omitting_warmup_prompt_is_fine() {
    assert!(parse(&[]).warmup_prompt.is_none());
    assert!(validate_serve_args(&parse(&[])).is_ok());
}

/// `--kv-high-precision-layers` is free-form, so a typo used to reach the
/// resolve site — AFTER the multi-minute weight load — and be swallowed by
/// `.parse().unwrap_or(0)` behind a single `warn!`. And `0` is not the
/// default: it is the value that defers to `auto_high_precision_layers`, so a
/// typo bought a third configuration rather than the documented one.
#[test]
fn kv_high_precision_layers_typo_is_refused_before_the_weight_load() {
    let err = validate_serve_args(&parse(&["--kv-high-precision-layers", "atuo"]))
        .expect_err("a typo must be refused, not silently resolved to 0");
    assert!(err.contains("--kv-high-precision-layers"), "{err}");
    assert!(err.contains("atuo"), "must quote the rejected value: {err}");
    assert!(err.contains("fix:"), "{err}");
    assert!(
        err.contains("auto") && err.contains("max"),
        "must name the accepted keywords: {err}"
    );
}

/// Every documented form still has to be accepted — a validator that rejects
/// valid input is worse than the silent default it replaced.
#[test]
fn every_documented_kv_high_precision_layers_form_is_accepted() {
    for form in ["0", "2", "64", "auto", "max", "all", "AUTO", "Max"] {
        assert!(
            validate_serve_args(&parse(&["--kv-high-precision-layers", form])).is_ok(),
            "{form} is documented as valid but was refused"
        );
    }
}

#[test]
fn fp8_calibration_requires_fp8_kv() {
    let err = validate_serve_args(&parse(&[
        "--kv-cache-dtype",
        "bf16",
        "--fp8-kv-calibration-tokens",
        "256",
    ]))
    .unwrap_err();
    assert!(err.contains("--fp8-kv-calibration-tokens"));
    assert!(err.contains("fix:"));
    // The same flags with an fp8 cache are fine.
    assert!(
        validate_serve_args(&parse(&[
            "--kv-cache-dtype",
            "fp8",
            "--fp8-kv-calibration-tokens",
            "256",
        ]))
        .is_ok()
    );
}

#[test]
fn an_absent_lever_flag_parses_as_unspecified() {
    // `None` is not the same as the default VALUE, and the difference is
    // load-bearing: publishing a default seals the cell these two flags
    // write to, which turned `ATLAS_SSM_TAIL_MIDCHUNK=0` and
    // `ATLAS_MTP_GATE_FORCE=1` into documented, echoed, silent no-ops under
    // `spark serve`. Absent has to stay absent all the way to
    // `publish_kernel_flags` for the fallback to be reachable.
    let a = parse(&[]);
    assert!(a.ssm_tail_midchunk.is_none(), "ATLAS_SSM_TAIL_MIDCHUNK");
    assert!(a.mtp_gate.is_none(), "ATLAS_MTP_GATE_FORCE");
    assert!(a.ssm_h_dtype.is_none(), "ATLAS_SSM_H_FP16");
    assert!(a.gdn_fused_norm.is_none(), "ATLAS_GDN_FUSED_NORM");
    assert!(
        a.ssm_batched_recurrent.is_none(),
        "ATLAS_SSM_BATCHED_RECURRENT"
    );
    // #435: absent must stay absent so publish_kernel_flags does not seal
    // the GDN cell; the resolved default (the legacy WY arms — exact verify
    // is OPT-IN) is asserted in gdn_flags' own tests.
    assert!(a.exact_verify.is_none(), "--exact-verify");
    assert!(a.prefill_varlen_batch.is_none(), "ATLAS_PREFILL_VARLEN");

    let a = parse(&["--ssm-tail-midchunk", "false", "--mtp-gate", "force"]);
    assert_eq!(a.ssm_tail_midchunk, Some(false), "given, it still wins");
    assert_eq!(a.mtp_gate.as_deref(), Some("force"));
}

#[test]
fn the_bare_gdn_switches_still_mean_on() {
    // `Option<bool>` must not turn a presence switch into one that DEMANDS
    // a value: every recipe and frozen command line writes them bare, and
    // silently requiring `--gdn-fused-norm true` would break all of them.
    let a = parse(&["--gdn-fused-norm", "--ssm-batched-recurrent"]);
    assert_eq!(a.gdn_fused_norm, Some(true));
    assert_eq!(a.ssm_batched_recurrent, Some(true));
    // And an explicit off is now expressible, which it was not before.
    let a = parse(&["--gdn-fused-norm", "false"]);
    assert_eq!(a.gdn_fused_norm, Some(false));
    // The #435 exact-verify opt-in follows the same convention: bare means
    // on (select the exact chain), explicit false is expressible.
    let a = parse(&["--exact-verify"]);
    assert_eq!(a.exact_verify, Some(true));
    let a = parse(&["--exact-verify", "false"]);
    assert_eq!(a.exact_verify, Some(false));
    // `--prefill-varlen-batch` follows the same convention.
    let a = parse(&["--prefill-varlen-batch"]);
    assert_eq!(a.prefill_varlen_batch, Some(true));
    let a = parse(&["--prefill-varlen-batch", "false"]);
    assert_eq!(a.prefill_varlen_batch, Some(false));
}

#[test]
fn exact_verify_refuses_the_f16_h_state() {
    // #435: the exact arm's kernels are FP32 readers; pairing the opt-in with
    // an FP16 h-state pool would silently drop the exact request, so the
    // validator refuses the combination outright.
    let err = validate_serve_args(&parse(&[
        "--exact-verify",
        "--ssm-h-dtype",
        "f16",
        "--gdn-fused-norm",
    ]))
    .unwrap_err();
    assert!(err.contains("--exact-verify"), "{err}");
    // POSITIVE controls: each side alone stays valid.
    assert!(validate_serve_args(&parse(&["--exact-verify"])).is_ok());
    assert!(validate_serve_args(&parse(&["--ssm-h-dtype", "f16", "--gdn-fused-norm"])).is_ok());
    // And an explicit `--exact-verify false` beside f16 is NOT a request for
    // exact verify, so it must not be refused.
    assert!(
        validate_serve_args(&parse(&[
            "--exact-verify",
            "false",
            "--ssm-h-dtype",
            "f16",
            "--gdn-fused-norm"
        ]))
        .is_ok()
    );
}

#[test]
fn f16_h_state_still_needs_the_fused_norm_arm() {
    // Absent counts as off: one GDN flag publishes all three, so
    // `--ssm-h-dtype f16` alone reaches the FP32-only kernel with an FP16
    // pool — fluent garbage, not a fault.
    assert!(validate_serve_args(&parse(&["--ssm-h-dtype", "f16"])).is_err());
    assert!(
        validate_serve_args(&parse(&["--ssm-h-dtype", "f16", "--gdn-fused-norm"])).is_ok(),
        "the supported pairing"
    );
    assert!(
        validate_serve_args(&parse(&[
            "--ssm-h-dtype",
            "f16",
            "--gdn-fused-norm",
            "false"
        ]))
        .is_err(),
        "and an explicit off is refused just as an absent one is"
    );
}

#[test]
fn ssm_rollback_mode_values_and_typos() {
    // The explicit default parses and validates (PCND: published on every
    // serve), and both recognized values round-trip.
    let a = parse(&[]);
    assert_eq!(a.ssm_rollback_mode, "snapshot");
    assert!(validate_serve_args(&a).is_ok());
    assert!(validate_serve_args(&parse(&["--ssm-rollback-mode", "replay"])).is_ok());
    // A typo is refused through the model-side FromStr (SSOT with the
    // publication parse) — never published, never silently defaulted.
    let err = validate_serve_args(&parse(&["--ssm-rollback-mode", "Replay"])).unwrap_err();
    assert!(err.contains("--ssm-rollback-mode"), "{err}");
    assert!(err.contains("snapshot"), "{err}");
}

#[test]
fn a_mistyped_mtp_gate_is_still_caught() {
    // Making the flag optional must not make its typo check optional.
    let err = validate_serve_args(&parse(&["--mtp-gate", "always"])).unwrap_err();
    assert!(err.contains("--mtp-gate"), "{err}");
    assert!(err.contains("auto, force"), "names the valid values: {err}");
}

#[test]
fn require_auth_needs_a_token() {
    assert!(validate_serve_args(&parse(&["--require-auth"])).is_err());
    assert!(validate_serve_args(&parse(&["--require-auth", "--auth-token", "sk-x"])).is_ok());
}

#[test]
fn num_drafts_needs_speculative() {
    assert!(validate_serve_args(&parse(&["--num-drafts", "2"])).is_err());
    assert!(validate_serve_args(&parse(&["--num-drafts", "2", "--speculative"])).is_ok());
}

#[test]
fn rank_must_be_below_world_size() {
    assert!(validate_serve_args(&parse(&["--rank", "2", "--world-size", "2"])).is_err());
    assert!(validate_serve_args(&parse(&["--rank", "1", "--world-size", "2"])).is_ok());
}

#[test]
fn ep_size_cannot_exceed_world_size() {
    assert!(validate_serve_args(&parse(&["--ep-size", "2"])).is_err());
    assert!(validate_serve_args(&parse(&["--ep-size", "2", "--world-size", "2"])).is_ok());
}

#[test]
fn disable_thinking_conflicts_with_budget() {
    assert!(
        validate_serve_args(&parse(&[
            "--disable-thinking",
            "--max-thinking-budget",
            "2048"
        ]))
        .is_err()
    );
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
