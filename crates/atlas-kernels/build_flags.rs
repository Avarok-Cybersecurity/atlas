// SPDX-License-Identifier: AGPL-3.0-only
//
// The extra-compiler-flag layers a kernel target is built with, and the rule
// that merges them. Included via `#[path = "build_flags.rs"] mod build_flags;`.
//
// Its own file, with no `super::` dependencies, so
// `tests/kernel_build_flags.rs` can compile the SAME code: cargo never runs a
// build script's `#[cfg(test)]` modules, so a rule that lives only inside
// build.rs is a rule nothing tests. Same posture as `build_arch.rs`.

/// The `[build]` key a vendor's extra flags are declared under.
///
/// NVIDIA (and SCALE/HIP, which consumes nvcc-shaped flags) reads
/// `extra_nvcc_flags`; Apple reads `extra_metal_flags`. A file may declare
/// both — only the vendor-matching list is forwarded, so flags do not bleed
/// across toolchains (nvcc's `--fmad=false` is not valid for `xcrun metal`).
///
/// SSOT for the key: `build_parse::parse_kernel_toml` calls this rather than
/// repeating the match, so HARDWARE.toml and KERNEL.toml can never disagree
/// about which key a vendor reads.
pub(crate) fn flag_key(vendor: &str) -> &'static str {
    match vendor {
        "apple" | "metal" => "extra_metal_flags",
        _ => "extra_nvcc_flags",
    }
}

/// `[build] extra_*_flags` from a parsed `kernels/<hw>/HARDWARE.toml`.
///
/// The HARDWARE layer states facts about the ARCHITECTURE, not the model: a
/// define that switches off a datapath whose instructions this ISA does not
/// have. `kernels/hopper` (sm_90a) and `kernels/b200` (sm_100a) use it to
/// define `ATLAS_NO_WARP_BLOCKSCALE_MMA`, because neither has the warp-level
/// `mma.sync ... .kind::mxf4nvf4.block_scale` that GB10's sm_121 does.
///
/// Why here and not in a KERNEL.toml: on those two targets every KERNEL.toml
/// is a relative symlink into `kernels/gb10`, and `tests/inherited_targets.rs`
/// requires that. Declaring an arch fact per model would mean forking a shared
/// file once per model and again for every model added later — the drift that
/// test exists to forbid. One line in the file that already describes the
/// architecture says it once, for every model on it.
///
/// Panics on a non-string entry, naming the key: a mistyped flag would
/// otherwise reach the compiler as nothing at all.
pub(crate) fn hardware_extra_flags(hw_toml: &toml::Value, vendor: &str) -> Vec<String> {
    let key = flag_key(vendor);
    let Some(arr) = hw_toml
        .get("build")
        .and_then(|b| b.get(key))
        .and_then(|f| f.as_array())
    else {
        return Vec::new();
    };
    arr.iter()
        .map(|v| {
            v.as_str()
                .unwrap_or_else(|| panic!("HARDWARE.toml: [build] {key} entries must be strings"))
                .to_string()
        })
        .collect()
}

/// The flag list a target is compiled with, from its three declaration layers.
///
/// Least specific first — hardware, then `common/KERNEL.toml`, then the
/// model's quant-dir KERNEL.toml — appended in order and deduped, so a flag
/// declared more than once is passed once, at the position its FIRST layer
/// gave it, and the most specific layer is what a reader sees last.
///
/// The two-layer half of this is not new behaviour: `resolve_targets` merged
/// common-then-model inline, after a prefer-model-else-common selection
/// silently dropped common's flags (gb10 model targets lost
/// `-DTQ_PLUS_SIGNS`; the Metal per-quant toml lost `-ffast-math`). Keeping
/// the merge in one tested function is what stops the third layer from
/// re-opening that.
pub(crate) fn merge_extra_flags(
    hardware: &[String],
    common: &[String],
    model: &[String],
) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for flag in hardware.iter().chain(common).chain(model) {
        if !out.contains(flag) {
            out.push(flag.clone());
        }
    }
    out
}
