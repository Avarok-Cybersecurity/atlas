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
        | super::super::Command::SyncRecipes => {
            unreachable!("this test parses a serve command")
        }
    }
}

#[test]
fn defaults_are_valid() {
    assert!(validate_serve_args(&parse(&[])).is_ok());
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
fn dflash_refuses_the_f16_h_state() {
    // gamma+1 = 17 verify rows dispatch `gated_delta_rule_wy17`: FP32-only,
    // and the one WY family with an explicit FP32-element intermediate
    // stride. Over an FP16 h-state that is fluent garbage, not a fault —
    // the same class the `--num-drafts > 3` preflight refusal exists for.
    for dtype in ["f16", "f16-pool"] {
        let err = validate_serve_args(&parse(&[
            "--dflash",
            "--ssm-h-dtype",
            dtype,
            "--gdn-fused-norm",
        ]))
        .unwrap_err();
        assert!(err.contains("--dflash"), "{dtype}: {err}");
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
