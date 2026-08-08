// SPDX-License-Identifier: AGPL-3.0-only

//! Publishing the command line's kernel-path selections.
//!
//! Each of these was an `ATLAS_*` variable read at its own call site. They are
//! CONFIGURATION — the concurrency campaign's best recipe needs three of them —
//! so they belong on the command line, where `--help` lists them, `ps` shows
//! them and a recipe can be read without a ten-line env preamble. The variables
//! stay honoured as a fallback for scripts that predate the flags.
//!
//! ## Absent is not a value
//!
//! Every cell below is a `OnceLock` whose fallback closure reads the
//! environment on first touch. Publishing a clap DEFAULT into one seals it
//! before anything asks, which makes the fallback unreachable — so five
//! documented `ATLAS_*` variables were inert under `spark serve` while
//! `--help` still described them and the startup log still echoed the flag back.
//! Every flag here is therefore an `Option`, and an absent flag publishes
//! NOTHING.

use crate::cli;

/// Publish the command line's kernel-path selections to the crates that
/// dispatch on them. Called once, before any model is built.
///
/// Logged, not silent: a run's kernel selection is the first thing anyone
/// reproducing a number needs, and reading it back out of the process is not
/// possible after the fact.
pub(crate) fn publish_kernel_flags(args: &cli::ServeArgs) {
    // The three GDN flags are ONE cell in `spark-model`, so they are published
    // together or not at all. Publishing them unconditionally — which is what
    // passing clap defaults amounted to — sealed that cell on every boot and
    // made `ATLAS_SSM_H_FP16`, `ATLAS_GDN_FUSED_NORM` and
    // `ATLAS_SSM_BATCHED_RECURRENT` inert while `--help` still documented them.
    // A frozen config that set one of those ran with the OPPOSITE setting and
    // nothing said so.
    //
    // With none of the three given there is nothing to publish, and
    // `gdn_flags::flags()` resolves from the environment on first touch as
    // documented. With any of them given the command line owns the whole cell,
    // as it always has; `warn_shadowed_env` names the variables that decision
    // overrides.
    let gdn_from_cli = args.ssm_h_dtype.is_some()
        || args.gdn_fused_norm.is_some()
        || args.ssm_batched_recurrent.is_some();
    if gdn_from_cli {
        let flags = spark_model::layers::qwen3_ssm::GdnFlags {
            h_f16: args.ssm_h_dtype.as_deref() == Some("f16"),
            fused_norm: args.gdn_fused_norm.unwrap_or(false),
            batched_recurrent: args.ssm_batched_recurrent.unwrap_or(false),
        };
        let in_force = spark_model::layers::qwen3_ssm::gdn_flags::set_from_cli(flags);
        if in_force != flags {
            tracing::warn!(
                "GDN flags were already resolved from the environment ({in_force:?}); \
                 the command line's ({flags:?}) did NOT take effect"
            );
        }
        warn_shadowed_env();
    }
    // `None` where the flag was not given, so the documented `ATLAS_*` fallback
    // still decides. Passing the clap default instead sealed both cells on
    // every boot and made those variables silent no-ops.
    spark_runtime::set_ssm_tail_midchunk(args.ssm_tail_midchunk);
    crate::scheduler::levers::set_mtp_gate_force(
        args.mtp_gate.as_deref().map(|gate| gate == "force"),
    );
    // Every value RESOLVED, none of them the raw argument. Each of these five
    // may now come from the environment, and a log that echoes what was asked
    // for rather than what is in force is exactly how a dead knob stays
    // invisible for a campaign.
    let gdn = spark_model::layers::qwen3_ssm::gdn_flags::flags();
    tracing::info!(
        "kernel flags: ssm_h_dtype={} gdn_fused_norm={} ssm_batched_recurrent={} \
         ssm_tail_midchunk={} mtp_gate={}",
        if gdn.h_f16 { "f16" } else { "f32" },
        gdn.fused_norm,
        gdn.batched_recurrent,
        spark_runtime::ssm_tail_midchunk_enabled(),
        if crate::scheduler::levers::mtp_gate_force() {
            "force"
        } else {
            "auto"
        },
    );
}

/// Legacy `ATLAS_*` variables that a GDN flag on the command line overrides.
///
/// `gdn_flags::set_from_cli` publishes all three of its fields together — there
/// is no per-field "unspecified" to express — so ONE of these flags takes the
/// whole decision away from the environment, including for the two the operator
/// did not write. That is the long-standing "the flag wins" rule and not a
/// change; what is new is saying so. An operator whose variable was overruled is
/// running a different configuration than they believe they are, and every
/// number from that run is mislabelled.
///
/// Warned only when a GDN flag was actually given: with none of them given
/// nothing is published and these variables decide, which is the documented
/// behaviour and needs no warning.
const SHADOWED_BY_CLI: &[(&str, &str)] = &[
    ("ATLAS_SSM_H_FP16", "--ssm-h-dtype f16"),
    ("ATLAS_GDN_FUSED_NORM", "--gdn-fused-norm"),
    ("ATLAS_SSM_BATCHED_RECURRENT", "--ssm-batched-recurrent"),
];

fn warn_shadowed_env() {
    for (var, flag) in SHADOWED_BY_CLI {
        if std::env::var_os(var).is_some() {
            tracing::warn!(
                "{var} is set but was OVERRIDDEN: a GDN flag on the command line publishes \
                 all three kernel selections at once, so this variable did not decide \
                 anything. Pass `{flag}` instead, or drop the GDN flags entirely to let the \
                 environment decide."
            );
        }
    }
}
