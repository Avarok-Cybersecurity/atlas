// SPDX-License-Identifier: AGPL-3.0-only
//
// The ONE line a build prints per kernel target. Included via
// `#[path = "build_summary.rs"] mod build_summary;`.
//
// WHY THIS FILE EXISTS. Round 11 of the H100 campaign asked for the resolved
// kernel count to be visible in a normal build, and round 13 (2026-09-11)
// recorded it still open: `build-r13.log` carried no kernel line and the count
// — 196 — had to be dug out of `target/release/build/atlas-kernels-*/output`
// and then independently confirmed by running `check_kernels.sh` against the
// finished binary. A number that takes a second tool to read is a number a
// campaign log does not carry, and every performance claim in that campaign is
// attributed to a binary by its kernel set.
//
// `cargo:warning=` is the documented way for a build script to reach the
// terminal, so that is what build.rs emits — ONE line per target, and this
// module is the only place its text is spelled. Its own file, with no `super::`
// dependencies, so `tests/build_summary.rs` can compile the SAME code: cargo
// never runs a build script's `#[cfg(test)]` modules, so a rule that lives only
// inside build.rs is a rule nothing tests. Same posture as `build_flags.rs`,
// `build_arch.rs` and `build_defaults.rs`.

/// `atlas-kernels: <N> kernels (<hw>, <model>, <quant>), <M> model-dir kernels, <K> declared overrides`
///
/// THREE counts, because the line used to print two and label one of them
/// wrong. H100 round 15 read `14 declared overrides` and went looking for
/// hopper's `[kernels] overrides` list; the printed field was
/// `find_cu_files(model_kernel_dir).len()` — the per-model/quant directory's
/// own `.cu` count — and had been 14 in round 14 too, while the list was a
/// different number. It never tracked that list, so it could never be the
/// receipt for "did my override land". Now both numbers print, each named for
/// what it is:
///
/// * `n_kernels` — sources compiled for the target, the same count the boot
///   preflight reports as `modules_embedded`. The join that lets a build log
///   and a running server be compared without a third tool.
/// * `n_model_dir` — how many of them came from the target's OWN
///   `<model>/<quant>/` directory rather than from `common/`. The old
///   mislabelled field, kept because it is a real fact about the resolution.
/// * `n_overrides` — the length of `kernels/<hw>/HARDWARE.toml`
///   `[kernels] overrides`: which `common/` entries this hardware TUNES for
///   itself rather than inheriting from gb10.
///
/// Zero prints as `0` in every field rather than being omitted: the
/// predecessor line suppressed the overrides clause entirely at zero, so "this
/// target declares none" and "this build did not say" looked identical in a
/// log. One shape, always, is what makes the line greppable.
pub(crate) fn summary(
    n_kernels: usize,
    hw: &str,
    model: &str,
    quant: &str,
    n_model_dir: usize,
    n_overrides: usize,
) -> String {
    format!(
        "atlas-kernels: {n_kernels} kernels ({hw}, {model}, {quant}), \
         {n_model_dir} model-dir kernels, {n_overrides} declared overrides"
    )
}

/// `[kernels] overrides` from `kernels/<hw>/HARDWARE.toml` — the entries this
/// hardware owns rather than inherits.
///
/// The SAME declaration `scripts/check_kernel_shadows.py` and
/// `tests/support/inherited.rs::kernel_overrides` read. A build script cannot
/// call a `tests/` helper, so the TOML is parsed here rather than a second
/// list being kept: the file is the SSOT, and that is the whole reason round
/// 15's mismatch was a reporting bug and not a build one.
///
/// A missing or unparseable file counts ZERO rather than panicking, matching
/// `build_defaults::read_defaults` — this also runs on the `ATLAS_SKIP_BUILD`
/// path, where the `kernels/` tree may not be present at all.
pub(crate) fn count_declared_overrides(kernels_root: &std::path::Path, hw: &str) -> usize {
    let path = kernels_root.join(hw).join("HARDWARE.toml");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return 0;
    };
    let Ok(value) = text.parse::<toml::Value>() else {
        return 0;
    };
    declared_overrides(&value).len()
}

/// The `[kernels] overrides` entries of an already-parsed HARDWARE.toml.
///
/// Split from [`count_declared_overrides`] so `tests/build_summary.rs` can
/// grade the PARSE — an `overrides` array that is absent, empty, or holds a
/// non-string — without a fixture directory on disk.
pub(crate) fn declared_overrides(hw_toml: &toml::Value) -> Vec<String> {
    hw_toml
        .get("kernels")
        .and_then(|k| k.get("overrides"))
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}
