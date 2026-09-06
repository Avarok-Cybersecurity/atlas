// SPDX-License-Identifier: AGPL-3.0-only
//
// HARDWARE.toml `arch` → `KernelTarget.arch` mapping for build.rs. Included
// via `#[path = "build_arch.rs"] mod build_arch;` and reached from the codegen
// module as `super::build_arch::kernel_target_arch`.
//
// Its own file, with no `super::` dependencies, so
// `tests/kernel_shadow_detector.rs`-style integration tests can compile the
// SAME code: cargo never runs a build script's `#[cfg(test)]` modules, so a
// rule that lives only inside build.rs is a rule nothing tests.

/// The base SM string a compiled target is recorded under, given the `arch`
/// its HARDWARE.toml declares.
///
/// nvcc's `-arch=` takes a *feature* architecture: a base SM number plus an
/// optional one-letter suffix selecting an extended instruction set —
/// `a` = arch-specific (Hopper `sm_90a` wgmma/TMA, Blackwell `sm_100a`),
/// `f` = family-specific (`sm_121f`). The suffix steers the COMPILER; it is
/// not part of the architecture's identity, and `KernelTarget.arch` spells
/// GB10 `sm_121` (see `crates/atlas-core/src/target.rs`). So strip it.
///
/// Only `sm_<digits>` strings are touched. SCALE/HIP `gfx*` names select a
/// per-arch toolchain directory verbatim (`gfx90a` is a whole architecture,
/// not `gfx90` plus a suffix) and Metal forwards `metal3.1` to `-std=`;
/// rewriting either would break the build it feeds.
pub(crate) fn kernel_target_arch(arch: &str) -> String {
    let Some(digits) = arch.strip_prefix("sm_") else {
        return arch.to_string();
    };
    let trimmed = digits.trim_end_matches(['a', 'f']);
    if trimmed.is_empty() || !trimmed.bytes().all(|b| b.is_ascii_digit()) {
        return arch.to_string();
    }
    format!("sm_{trimmed}")
}
/// The two arch strings one HARDWARE.toml `arch` declaration produces, as
/// `(KernelTarget.arch, ptx_arch)`.
///
/// `TargetPtxSet` records both because they answer different questions and
/// only one of them is safe for each:
///
/// * `KernelTarget.arch` is the base SM (`sm_90`, `sm_121`). It is an
///   IDENTITY — the key `crates/atlas-core/src/target.rs` spells its constants
///   with, and what gate baselines and existing records are keyed by. It must
///   not change.
/// * `ptx_arch` is the string nvcc was handed, verbatim (`sm_90a`,
///   `sm_121f`). It is the only one that can answer a COMPATIBILITY question,
///   because the feature suffix IS the rule: `a` never runs forward onto a
///   later architecture, `f` stays inside one major family, and a bare
///   `sm_XY` JIT-compiles forward from X.Y. Strip the suffix and every
///   arch-specific build looks portable.
///
/// Returned as a pair, from one function, so the two readings of a single
/// declaration cannot drift apart in codegen. They drifted once: the GPU
/// preflight judged `KernelTarget.arch`, so `sm_90a` reached it as `sm_90`,
/// the plain forward-compat rule applied, and Hopper-only PTX PASSED the
/// preflight on a CC 10.0 device — the exact `cuModuleLoadData` failure the
/// preflight was added to replace.
pub(crate) fn target_arch_fields(arch: &str) -> (String, &str) {
    (kernel_target_arch(arch), arch)
}
