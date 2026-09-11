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

/// `atlas-kernels: <N> kernels (<hw>, <model>, <quant>), <K> declared overrides`
///
/// `n_kernels` is the number of sources compiled for the target — the same
/// count the boot preflight reports as `modules_embedded`, which is the join
/// that lets a build log and a running server be compared without a third tool.
/// `n_overrides` is how many of them came from the target's OWN model/quant
/// directory rather than from `common/`.
///
/// Zero overrides prints as `0 declared overrides` rather than being omitted:
/// the previous line suppressed the clause entirely when the count was zero, so
/// "this target declares none" and "this build did not say" looked identical in
/// a log. One shape, always, is what makes the line greppable.
pub(crate) fn summary(
    n_kernels: usize,
    hw: &str,
    model: &str,
    quant: &str,
    n_overrides: usize,
) -> String {
    format!(
        "atlas-kernels: {n_kernels} kernels ({hw}, {model}, {quant}), \
         {n_overrides} declared overrides"
    )
}
