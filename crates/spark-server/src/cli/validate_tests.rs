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
        super::super::Command::Benchmark(_) => unreachable!("this test parses a serve command"),
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
