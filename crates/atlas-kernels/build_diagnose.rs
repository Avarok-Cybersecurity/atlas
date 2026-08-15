// SPDX-License-Identifier: AGPL-3.0-only
//
// Diagnostics for hardware-target selection.
//
// `ATLAS_TARGET_HW` selects which `kernels/<hw>/` tree gets compiled, and it
// defaults to `gb10` — an NVIDIA GB10 target. That default is right for the
// machines this repo was born on and silently wrong everywhere else: someone
// on an AMD box who runs the documented `cargo build --release -p spark-server`
// builds NVIDIA kernels, and finds out somewhere deep in nvcc, in an error
// that never mentions the env var that actually chose their hardware.
//
// The required triple is currently spelled out only in workflow comments and
// porting docs. Nothing in the build tells you it exists. Everything here
// exists to make the build say so itself.
//
// These functions run on the failure path and inside panic messages, so they
// are strictly best-effort: they never panic, and a missing or malformed
// HARDWARE.toml degrades to a less specific message rather than replacing the
// user's real error with one of ours.

use std::path::Path;
use std::sync::Once;

/// Hardware target compiled when `ATLAS_TARGET_HW` is unset.
pub const DEFAULT_HW: &str = "gb10";

/// Read `vendor = "..."` out of a HARDWARE.toml by hand.
///
/// Deliberately not a TOML parse: this is used to build panic messages, so it
/// must not itself fail on a file that is malformed — which is exactly the
/// situation some of these messages are reporting.
fn vendor_of(hw_dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(hw_dir.join("HARDWARE.toml")).ok()?;
    text.lines()
        .map(str::trim)
        .find_map(|l| l.strip_prefix("vendor"))
        .and_then(|rest| rest.trim_start().strip_prefix('='))
        .map(|v| v.trim().trim_matches('"').to_string())
}

/// Immediate subdirectories of `dir`, sorted, ignoring anything unreadable.
fn subdirs(dir: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<String> = entries
        .flatten()
        .filter(|e| e.path().is_dir())
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();
    out.sort();
    out
}

/// Hardware targets that exist, as `name (vendor)`.
pub fn available_hw(kernels_root: &Path) -> Vec<String> {
    subdirs(kernels_root)
        .into_iter()
        .filter(|name| kernels_root.join(name).join("HARDWARE.toml").is_file())
        .map(|name| match vendor_of(&kernels_root.join(&name)) {
            Some(v) => format!("{name} ({v})"),
            None => name,
        })
        .collect()
}

/// Models under a hardware target: subdirs carrying a MODEL.toml.
pub fn available_models(hw_dir: &Path) -> Vec<String> {
    subdirs(hw_dir)
        .into_iter()
        .filter(|d| hw_dir.join(d).join("MODEL.toml").is_file())
        .collect()
}

fn or_none(items: Vec<String>) -> String {
    if items.is_empty() {
        "(none found)".to_string()
    } else {
        items.join(", ")
    }
}

/// One ready-to-paste invocation per hardware target that actually exists.
///
/// Every line is built from real directory names, so anything listed here is
/// known to resolve. Listing all of them rather than picking one avoids
/// teaching an AMD user the Metal incantation just because `metal` happens to
/// sort first.
fn examples(kernels_root: &Path) -> String {
    let lines: Vec<String> = subdirs(kernels_root)
        .into_iter()
        .filter(|h| kernels_root.join(h).join("HARDWARE.toml").is_file())
        .map(|hw| {
            let hw_dir = kernels_root.join(&hw);
            let model = available_models(&hw_dir)
                .into_iter()
                .next()
                .unwrap_or_else(|| "<model>".to_string());
            let quant = subdirs(&hw_dir.join(&model))
                .into_iter()
                .next()
                .unwrap_or_else(|| "<quant>".to_string());
            format!("   ATLAS_TARGET_HW={hw} ATLAS_TARGET_MODEL={model} ATLAS_TARGET_QUANT={quant} cargo build --release -p spark-server")
        })
        .collect();
    if lines.is_empty() {
        "   (no kernels/<hw>/HARDWARE.toml found - is this a full checkout?)".to_string()
    } else {
        lines.join("\n")
    }
}

/// Panic text for an `ATLAS_TARGET_HW` naming no `kernels/<hw>/` directory.
pub fn unknown_hw(kernels_root: &Path, hw: &str) -> String {
    format!(
        "\n\
         ATLAS_TARGET_HW={hw} selects kernels/{hw}/, which does not exist.\n\
         \n\
         Available hardware targets: {}\n\
         \n\
         Build a target that exists:\n{}\n",
        or_none(available_hw(kernels_root)),
        examples(kernels_root),
    )
}

/// Panic text for an `ATLAS_TARGET_MODEL` with no directory under this target.
pub fn unknown_model(hw_dir: &Path, hw: &str, model: &str) -> String {
    format!(
        "\n\
         ATLAS_TARGET_MODEL={model} selects kernels/{hw}/{model}/, which does not exist.\n\
         \n\
         Models available for {hw}: {}\n",
        or_none(available_models(hw_dir)),
    )
}

/// Panic text for a target selection that resolved to nothing at all.
pub fn no_targets(kernels_root: &Path, hw: &str, model: &str, quant: &str) -> String {
    format!(
        "\n\
         No kernel targets resolved for ATLAS_TARGET_HW={hw} \
         ATLAS_TARGET_MODEL={model} ATLAS_TARGET_QUANT={quant}.\n\
         \n\
         Models available for {hw}: {}\n\
         \n\
         Build a target that exists:\n{}\n",
        or_none(available_models(&kernels_root.join(hw))),
        examples(kernels_root),
    )
}

/// Say out loud what is being built whenever any of the triple was implicit.
///
/// The `/atlas-release` skill states the rule this serves: "no stage runs on an
/// implicit default... a bare `cargo build` is a bug, not a shortcut". The
/// build could not previously say which hardware it picked, so the rule was
/// unenforceable from inside it.
///
/// A `cargo:warning=` rather than a hard error, for two reasons: changing or
/// requiring the value would break every existing NVIDIA build, and the
/// problem was never the default itself — it is that the default was
/// invisible. Emitted once even though resolution reads these from more than
/// one place.
pub fn warn_if_defaulted(kernels_root: &Path, hw: &str, model: &str, quant: &str) {
    static ONCE: Once = Once::new();

    let implicit: Vec<&str> = [
        ("ATLAS_TARGET_HW", hw),
        ("ATLAS_TARGET_MODEL", model),
        ("ATLAS_TARGET_QUANT", quant),
    ]
    .iter()
    .filter(|(var, _)| std::env::var_os(var).is_none())
    .map(|(var, _)| *var)
    .collect();

    if implicit.is_empty() {
        return;
    }

    // "a, b and c" rather than join(" and "), which produces "a and b and c".
    let unset = match implicit.split_last() {
        Some((last, [])) => (*last).to_string(),
        Some((last, rest)) => format!("{} and {last}", rest.join(", ")),
        None => unreachable!("empty case returned above"),
    };

    ONCE.call_once(|| {
        let vendor = vendor_of(&kernels_root.join(hw)).unwrap_or_else(|| "unknown vendor".into());
        println!(
            "cargo:warning=Building kernels for {hw} ({vendor}), model={model}, quant={quant} \
             - {unset} not set. Hardware targets available: {}. Set all three explicitly \
             to choose.",
            or_none(available_hw(kernels_root)),
        );
    });
}
