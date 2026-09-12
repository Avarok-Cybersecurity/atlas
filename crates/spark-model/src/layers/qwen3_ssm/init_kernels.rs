// SPDX-License-Identifier: AGPL-3.0-only

//! Kernel lookups the SSM layer's constructor makes conditionally.
//!
//! Split from `init.rs` for the 500-LoC cap, which the file crossed by one
//! line when the LoRA `out_proj` slot joined the layer. Exact piecewise copy.

use super::*;

/// Resolve one `hyper_connection` entry point, but ONLY for a model that
/// carries the highway. Skipping the lookup rather than discarding its result
/// is the point: an un-issued lookup leaves no failed row in the fail-closed
/// startup audit, so what remains there is what someone has to act on.
#[track_caller]
pub(super) fn hc_kernel(
    config: &atlas_core::config::ModelConfig,
    gpu: &dyn GpuBackend,
    func: &str,
) -> KernelHandle {
    if config.hc_mult > 0 {
        crate::layers::try_kernel(gpu, "hyper_connection", func)
    } else {
        KernelHandle(0)
    }
}

/// Chain-verify K=5..16 WY kernels (one templated gb10-common module;
/// K=9..16 arrived 2026-08-29 with the gamma>8 window). Index = K-5; a NULL
/// handle means the target lacks the module, in which case that width keeps
/// the sequential per-token path.
///
/// Split out of `init.rs` with its FP16 twin for the 500-LoC cap. Exact
/// piecewise copy — the index contract is the load-bearing part and is
/// unchanged.
pub(super) fn wyn_kernels(gpu: &dyn GpuBackend) -> [KernelHandle; 12] {
    [
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy5"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy6"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy7"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy8"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy9"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy10"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy11"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy12"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy13"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy14"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy15"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy16"),
    ]
}

/// FP16 h-state twins (K=5..16), same module and the SAME index contract as
/// [`wyn_kernels`] — a mismatch between the two would silently pair a width
/// with another width's twin.
pub(super) fn wyn_f16_kernels(gpu: &dyn GpuBackend) -> [KernelHandle; 12] {
    [
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy5_f16"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy6_f16"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy7_f16"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy8_f16"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy9_f16"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy10_f16"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy11_f16"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy12_f16"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy13_f16"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy14_f16"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy15_f16"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy16_f16"),
    ]
}

/// The tensor-core GDN chunked-PREFILL spine's handle
/// ([`ops::GDN_TC_SPINE_ENTRY`](crate::layers::ops::GDN_TC_SPINE_ENTRY) —
/// the SAME constant the serve's route line prints, so the log cannot name a
/// kernel other than the one bound here), GATED on the same bit that
/// launches it: `[defaults] gdn_prefill_tc`, with `ATLAS_GDN_PREFILL_TC`
/// overriding (`layers::ops::target_defaults`).
///
/// A probe that runs unconditionally asks the kernel audit about a module no
/// target enables, which is how a lever nobody set comes to be the reason a
/// boot failed. Off yields `KernelHandle(0)`, and `ops::gdn_tc_spine_reject`
/// then answers "not requested" — which is what it would have answered anyway.
///
/// The `_x2` entry (two bf16 limbs of S_c in Phase A) is the one the lever
/// ships: the single-limb `..._tcfuse` entry is in the image for the oracle's
/// A/B, but its measured deviation on the FP32 state is ~2.0e-3, over the
/// 1e-3 contract. Both entries are ABI-, grid-, block- and smem-identical, so
/// nothing downstream changes with the choice.
pub(super) fn gdn_prefill_tc_kernel(gpu: &dyn GpuBackend) -> KernelHandle {
    if !crate::layers::ops::target_defaults::resolved()
        .gdn_prefill_tc
        .value
    {
        return KernelHandle(0);
    }
    crate::layers::try_kernel(
        gpu,
        crate::layers::ops::GDN_TC_SPINE_MODULE,
        crate::layers::ops::GDN_TC_SPINE_ENTRY,
    )
}

// ── The two HOPPER-ONLY prefill twin probes, one function each ─────────────
//
// `try_kernel` and not `kernel` in both: these modules exist only under
// `kernels/hopper` (declared in that target's `[kernels] overrides`), so on
// gb10/b200/strix the lookup must MISS quietly and leave the launcher on the
// parent kernel. A handle of 0 IS the "not on this target" answer; nothing
// downstream needs a second way to ask.
//
// One named function per handle rather than one helper taking two strings:
// the module/entry pair is the whole content of the probe, and a call site
// that passes them as arguments has simply moved the thing being reviewed
// back into `init.rs`. They live here for the 500-LoC cap, beside
// `gdn_prefill_tc_kernel`, which is the same shape for the third kernel of
// the same prefill family.

/// Prefill kernel 1's twin: the two forward substitutions on tensor cores
/// (#928). Selected by `[defaults] gdn_prefill_tc`, the same family lever as
/// [`gdn_prefill_tc_kernel`] above; unlike the spine, the probe is NOT gated on
/// it, because `gated_delta_rule_fla`'s parent is always loaded and a twin that
/// is merely absent costs nothing to have looked for.
pub(super) fn prefill_wu_hopper_k(gpu: &dyn GpuBackend) -> KernelHandle {
    crate::layers::try_kernel(
        gpu,
        "gdn_recompute_wu_hopper",
        "gated_delta_rule_recompute_wu_hopper",
    )
}

/// Prefill kernel 3's twin: the masked `tril(kq).uc` square on tensor cores
/// (#928). Same family lever and the same reasoning as the `wu` twin above.
pub(super) fn prefill_fwd_o_hopper_k(gpu: &dyn GpuBackend) -> KernelHandle {
    crate::layers::try_kernel(
        gpu,
        "gdn_fwd_o_hopper",
        "gated_delta_rule_chunk_fwd_o_hopper",
    )
}
