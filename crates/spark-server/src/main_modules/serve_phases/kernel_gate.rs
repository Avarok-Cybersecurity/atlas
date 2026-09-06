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

/// Retain the preflight error while reporting an early `--check-kernels` refusal.
/// No kernel-audit count is available before the backend has been constructed.
#[cfg(feature = "cuda")]
pub(crate) fn gate_arch_preflight(
    checking: bool,
    ptx_set: &atlas_kernels::TargetPtxSet,
    result: Result<()>,
) -> Result<()> {
    if let Err(error) = &result
        && let Some(line) = arch_refusal_json(
            checking,
            ptx_set.target.model,
            ptx_set.target.quant,
            ptx_set.modules.len(),
            error,
        )
    {
        use std::io::Write as _;
        println!("{line}");
        let _ = std::io::stdout().flush();
    }
    result
}

#[cfg(any(feature = "cuda", test))]
fn arch_refusal_json(
    checking: bool,
    model: &str,
    quant: &str,
    modules_embedded: usize,
    error: &anyhow::Error,
) -> Option<String> {
    let mismatch = error.downcast_ref::<atlas_core::arch::ArchMismatch>()?;
    checking.then(|| {
        serde_json::json!({
            "atlas_kernel_check": {
                "model": model,
                "arch": mismatch.compiled_arch,
                "compiled_arch": mismatch.compiled_arch,
                "device_cc": mismatch.device_cc,
                "quant": quant,
                "kernel_set_hash": atlas_kernels::KERNEL_SET_HASH,
                "modules_embedded": modules_embedded,
                "lookups": null,
                "unresolved": null,
                "expected_absent": null,
                "unresolved_kernels": null,
                "ok": false,
                "exit_code": 1,
                "failing_stage": "arch_preflight",
                "error": error.to_string(),
            }
        })
        .to_string()
    })
}

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

/// Everything the one-line `--check-kernels` result reports.
///
/// A struct rather than ten positional arguments, so the JSON shape can be
/// asserted without booting a GPU or building a `TargetPtxSet`.
struct CheckSummary<'a> {
    model: &'a str,
    /// The arch this binary's kernels were COMPILED for — verbatim
    /// `kernels/<hw>/HARDWARE.toml` `[hardware].arch`.
    compiled_arch: &'a str,
    quant: &'a str,
    modules_embedded: usize,
    lookups: usize,
    expected_absent: usize,
    exit_code: usize,
    unresolved: Vec<serde_json::Value>,
    /// `(major, minor)` of the GPU in the box, when there is one to ask.
    /// `None` on a host with no CUDA device — the check still reports the rest.
    device_cc: Option<(u32, u32)>,
}

/// One compact JSON object summarising the check. `ok` is the exit-code twin.
///
/// `arch` and `compiled_arch` carry the SAME value: `arch` is the name the
/// field shipped under and is kept for existing fleet sweeps, `compiled_arch`
/// is the unambiguous one now that a second architecture — the DEVICE's — sits
/// beside it in the same object.
fn check_json_from(summary: &CheckSummary) -> String {
    serde_json::json!({
        "atlas_kernel_check": {
            "model": summary.model,
            "arch": summary.compiled_arch,
            "compiled_arch": summary.compiled_arch,
            "device_cc": summary.device_cc,
            "quant": summary.quant,
            "kernel_set_hash": atlas_kernels::KERNEL_SET_HASH,
            "modules_embedded": summary.modules_embedded,
            "lookups": summary.lookups,
            "unresolved": summary.unresolved.len(),
            "expected_absent": summary.expected_absent,
            "ok": summary.unresolved.is_empty(),
            // The status this process is about to exit with. Differs from
            // `unresolved` only when the 8-bit ceiling clamped it.
            "exit_code": summary.exit_code,
            "unresolved_kernels": summary.unresolved,
        }
    })
    .to_string()
}

/// The GPU's compute capability, or `None` when nothing can be asked.
///
/// `--check-kernels` runs after the backend is up, so on a served target this
/// is the real device. It stays `None` for a metal build and for any CUDA host
/// whose driver refuses the query, because a check that reported a guessed
/// compute capability would be worse than one that reported none.
fn current_device_cc() -> Option<(u32, u32)> {
    #[cfg(feature = "cuda")]
    {
        spark_runtime::cuda_backend::arch_preflight::device_compute_capability().ok()
    }
    #[cfg(not(feature = "cuda"))]
    {
        None
    }
}

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
    check_json_from(&CheckSummary {
        model: ptx_set.target.model,
        // `ptx_arch`, not `target.arch`: this field's contract (above) is the
        // VERBATIM `[hardware].arch`, and `target.arch` is that string with
        // its feature suffix stripped. Reporting `sm_90` for a build nvcc
        // compiled as `sm_90a` describes PTX that travels forward, which this
        // PTX does not.
        compiled_arch: ptx_set.ptx_arch,
        quant: ptx_set.target.quant,
        modules_embedded: ptx_set.modules.len(),
        lookups: rows.len(),
        expected_absent: split.expected.len(),
        exit_code,
        unresolved,
        device_cc: current_device_cc(),
    })
}

#[cfg(test)]
mod tests {
    use super::{CheckSummary, MAX_EXIT_CODE, arch_refusal_json, check_json_from, exit_code_for};

    fn a_clean_gb10_check() -> CheckSummary<'static> {
        CheckSummary {
            model: "qwen3.6-27b",
            compiled_arch: "sm_121f",
            quant: "nvfp4",
            modules_embedded: 42,
            lookups: 300,
            expected_absent: 2,
            exit_code: 0,
            unresolved: Vec::new(),
            device_cc: Some((12, 1)),
        }
    }

    /// A sweep must be able to tell which kernels these are and which GPU
    /// answered, from the one machine-readable line and nothing else.
    ///
    /// Oracle: `kernels/gb10/HARDWARE.toml` — `arch = "sm_121f"`,
    /// `compute_capability = "12.1"`.
    #[test]
    fn the_check_line_reports_the_compiled_arch_and_the_device() {
        let v: serde_json::Value =
            serde_json::from_str(&check_json_from(&a_clean_gb10_check())).expect("valid JSON");
        let c = &v["atlas_kernel_check"];
        assert_eq!(c["compiled_arch"], "sm_121f");
        assert_eq!(c["device_cc"], serde_json::json!([12, 1]));
        // The pre-existing name for the same value, kept for existing sweeps.
        assert_eq!(c["arch"], "sm_121f");
    }

    /// A GPU-free `--check-kernels` still reports; it says "no device" rather
    /// than inventing one. A guessed compute capability in this line would be
    /// worse than an absent one.
    #[test]
    fn a_host_with_no_device_reports_a_null_device_cc() {
        let summary = CheckSummary {
            device_cc: None,
            ..a_clean_gb10_check()
        };
        let v: serde_json::Value =
            serde_json::from_str(&check_json_from(&summary)).expect("valid JSON");
        assert!(v["atlas_kernel_check"]["device_cc"].is_null());
        assert_eq!(v["atlas_kernel_check"]["compiled_arch"], "sm_121f");
    }

    /// The fields that were already there must survive the shape change —
    /// `ok` is the exit-code twin the whole gate contract rests on.
    #[test]
    fn the_pre_existing_fields_are_unchanged() {
        let summary = CheckSummary {
            exit_code: 3,
            unresolved: vec![serde_json::json!({"kernel": "m::k", "site": "a.rs:1"})],
            ..a_clean_gb10_check()
        };
        let v: serde_json::Value =
            serde_json::from_str(&check_json_from(&summary)).expect("valid JSON");
        let c = &v["atlas_kernel_check"];
        assert_eq!(c["model"], "qwen3.6-27b");
        assert_eq!(c["quant"], "nvfp4");
        assert_eq!(c["modules_embedded"], 42);
        assert_eq!(c["lookups"], 300);
        assert_eq!(c["unresolved"], 1);
        assert_eq!(c["expected_absent"], 2);
        assert_eq!(c["ok"], false);
        assert_eq!(c["exit_code"], 3);
        assert_eq!(c["unresolved_kernels"][0]["kernel"], "m::k");
        assert_eq!(c["device_cc"], serde_json::json!([12, 1]));
        assert_eq!(c["compiled_arch"], "sm_121f");
    }

    /// Oracle: the actual runtime compatibility decision, before any GPU or
    /// kernel lookup. A mismatch must survive as structured failure evidence.
    #[test]
    #[cfg(feature = "cuda")]
    fn early_arch_refusals_report_the_same_numeric_device_contract() {
        for arch in ["sm_90a", "sm_100a"] {
            let error = spark_runtime::cuda_backend::arch_preflight::check_arch(arch, (12, 1))
                .expect_err("architecture-specific PTX must refuse GB10");
            let line = arch_refusal_json(true, "nano", "nvfp4", 42, &error)
                .expect("--check-kernels must retain preflight mismatch JSON");
            let value: serde_json::Value = serde_json::from_str(&line).unwrap();
            let check = &value["atlas_kernel_check"];
            assert_eq!(check["compiled_arch"], arch);
            assert_eq!(check["arch"], arch);
            assert_eq!(check["device_cc"], serde_json::json!([12, 1]));
            assert_eq!(check["ok"], false);
            assert_eq!(check["exit_code"], 1);
            assert_eq!(check["failing_stage"], "arch_preflight");
            assert!(check["lookups"].is_null());
            assert!(check["unresolved"].is_null());
            assert!(check["error"].as_str().unwrap().contains("12.1"));
            assert!(check["error"].as_str().unwrap().contains(arch));
            assert!(arch_refusal_json(false, "nano", "nvfp4", 42, &error).is_none());
        }
    }

    #[test]
    fn an_unrelated_error_does_not_invent_device_facts() {
        let error = anyhow::anyhow!("CUDA initialization failed before the device query");
        assert!(arch_refusal_json(true, "nano", "nvfp4", 42, &error).is_none());
    }

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
