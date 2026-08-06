// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 7 — the kernel-resolution audit and the fail-closed boot gate.
//!
//! Every kernel lookup in Atlas is EAGER: each `.kernel(…)` / `try_kernel(…)`
//! site sits in a constructor on the `build_model` path. By the time this runs,
//! the audit therefore holds the COMPLETE `(module, func)` set this model asks
//! for — which is what makes ONE BOOT yield the whole list for a target, and
//! what makes `--check-kernels` a usable fleet sweep rather than a sampler.
//!
//! The gate this replaces was a no-op. It intersected the failed lookups with
//! `shadowed_dropped`, which for `qwen3.6-27b/nvfp4` is exactly the two
//! `[shadow_exempt]` entries that have no dispatch site anywhere in the repo —
//! a provably empty intersection. It could not have fired for that model under
//! any circumstances, which is how the 27B shipped with concurrent decode
//! silently disabled while every gate stayed green.

use anyhow::Result;

use crate::cli;

/// POSIX exit statuses are 8 bits. An unclamped count of exactly 256 would be
/// reported as 0 — a catastrophically broken model reading as a clean pass,
/// which is the worst possible failure for a tool whose only job is to be
/// trustworthy. The clamp is announced in the output whenever it bites, so the
/// number in `$?` is never silently wrong.
const MAX_EXIT_CODE: usize = 255;

/// Print the audit, gate on it, and seal the audit for the rest of the run.
///
/// Under `--check-kernels` this function does NOT return: it exits the process
/// with the unresolved count as the status. Owning the exit here keeps the
/// count and the status in one place — routing it back through `Result` would
/// collapse every count to anyhow's 1.
pub(crate) fn audit_and_gate(
    args: &cli::ServeArgs,
    ptx_set: &atlas_kernels::TargetPtxSet,
) -> Result<()> {
    tracing::info!(
        "{}",
        spark_runtime::kernel_audit::render_kernel_table(
            &ptx_set.modules,
            atlas_kernels::KERNEL_SET_HASH,
            ptx_set.shadowed_dropped,
            ptx_set.expected_absent,
        )
    );

    let rows = spark_runtime::kernel_audit::audit_rows();
    let split = spark_runtime::kernel_audit::split_failures(&rows, ptx_set.expected_absent);
    let allowed = args.dangerously_allow_unresolved_kernel_lookups;
    let target = &ptx_set.target;

    // Seal BEFORE the decision so a late lookup is loud on every path,
    // including the one where the operator chose to serve anyway.
    spark_runtime::kernel_audit::seal(split.required.len() as u64, allowed);

    if args.check_kernels {
        check_and_exit(&rows, &split, ptx_set);
    }

    if split.required.is_empty() {
        return Ok(());
    }
    let report = spark_runtime::kernel_audit::unresolved_report(
        &split,
        ptx_set.shadowed_dropped,
        target.model,
        target.arch,
        target.quant,
        allowed,
    );
    if allowed {
        // No suppression, ever. A flag that mutes the warning recreates the bug.
        tracing::warn!("{report}");
        return Ok(());
    }
    Err(anyhow::anyhow!("{report}"))
}

/// `--check-kernels`: print the report, print the JSON line, exit with the
/// unresolved count (clamped to [`MAX_EXIT_CODE`]). Never returns.
fn check_and_exit(
    rows: &[spark_runtime::kernel_audit::AuditRow],
    split: &spark_runtime::kernel_audit::FailureSplit,
    ptx_set: &atlas_kernels::TargetPtxSet,
) -> ! {
    use std::io::Write as _;

    let target = &ptx_set.target;
    let n = split.required.len();
    if n == 0 {
        tracing::info!(
            "kernel check PASSED for ({}, {}, {}): {} lookups, {} expected-absent",
            target.model,
            target.arch,
            target.quant,
            rows.len(),
            split.expected.len(),
        );
    } else {
        // `--check-kernels` reports the TRUTH, so the remediation text never
        // switches to the "already allowed" form and the exit status below
        // ignores `--dangerously-allow-unresolved-kernel-lookups` entirely. A
        // check whose answer another flag can silence is worth nothing.
        tracing::error!(
            "{}",
            spark_runtime::kernel_audit::unresolved_report(
                split,
                ptx_set.shadowed_dropped,
                target.model,
                target.arch,
                target.quant,
                false,
            )
        );
    }
    let code = exit_code_for(n);
    if code != n {
        // Unmissable, on both streams: `$?` is about to under-report.
        let msg = format!("{n} unresolved kernels (exit code clamped to {MAX_EXIT_CODE})");
        tracing::error!("{msg}");
        println!("{msg}");
    }
    // Machine-readable result on ONE line, after the human report, so a sweep
    // across every target aggregates without parsing prose.
    println!("{}", check_json(rows, split, ptx_set, code));
    // `exit` runs no destructors, so flush what a pipe would otherwise lose.
    let _ = std::io::stdout().flush();
    std::process::exit(code as i32);
}

/// The process status for `n` unresolved kernels.
///
/// The contract is "the exit code IS the count", so this is identity up to the
/// 8-bit POSIX ceiling. The clamp exists because 256 would be reported as 0 —
/// a catastrophically broken model reading as a clean pass. Clamping to 255
/// keeps a broken target non-zero, and the caller announces whenever the clamp
/// bit so `$?` is never silently wrong.
fn exit_code_for(n: usize) -> usize {
    n.min(MAX_EXIT_CODE)
}

/// One compact JSON object summarising the check. `ok` is the exit-code twin.
fn check_json(
    rows: &[spark_runtime::kernel_audit::AuditRow],
    split: &spark_runtime::kernel_audit::FailureSplit,
    ptx_set: &atlas_kernels::TargetPtxSet,
    exit_code: usize,
) -> String {
    let unresolved: Vec<serde_json::Value> = split
        .required
        .iter()
        .map(|r| {
            serde_json::json!({
                "kernel": r.name(),
                "site": format!("{}:{}", r.site.file(), r.site.line()),
            })
        })
        .collect();
    serde_json::json!({
        "atlas_kernel_check": {
            "model": ptx_set.target.model,
            "arch": ptx_set.target.arch,
            "quant": ptx_set.target.quant,
            "kernel_set_hash": atlas_kernels::KERNEL_SET_HASH,
            "modules_embedded": ptx_set.modules.len(),
            "lookups": rows.len(),
            "unresolved": split.required.len(),
            "expected_absent": split.expected.len(),
            "ok": split.required.is_empty(),
            // The status this process is about to exit with. Differs from
            // `unresolved` only when the 8-bit ceiling clamped it.
            "exit_code": exit_code,
            "unresolved_kernels": unresolved,
        }
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::{MAX_EXIT_CODE, exit_code_for};

    #[test]
    fn the_exit_code_is_the_unresolved_count() {
        // The stated contract: `$?` equals the number of unresolved kernels.
        for n in [0usize, 1, 2, 15, 42, 254, 255] {
            assert_eq!(exit_code_for(n), n, "exit code must equal the count");
        }
    }

    #[test]
    fn a_count_of_256_does_not_report_as_a_clean_pass() {
        // ★ The reason the clamp exists. POSIX statuses are 8 bits, so an
        // unclamped 256 arrives as 0 — the most broken possible target reading
        // as "every lookup resolved". Anything at or above the ceiling must
        // stay non-zero.
        assert_eq!(exit_code_for(256), MAX_EXIT_CODE);
        assert_eq!(exit_code_for(1000), MAX_EXIT_CODE);
        for n in [256usize, 512, 4096] {
            assert_ne!(exit_code_for(n) % 256, 0, "{n} must not read as success");
        }
    }
}
