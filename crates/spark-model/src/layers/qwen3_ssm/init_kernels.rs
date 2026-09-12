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

/// Resolve one `fp8_scale_transpose` entry point, but ONLY when a cuBLASLt SSM
/// arm could launch it (`[defaults] cublas_gemm_scope` naming `ssm`, or
/// `ATLAS_CUBLAS_GEMM` overriding it).
///
/// Same rule as [`hc_kernel`], for a second reason on top of the audit one:
/// since the 2026-09-11 arch separation, `fp8_scale_transpose.cu` is a
/// HOPPER-TUNED source (`kernels/hopper/common`, declared in that target's
/// `[kernels] overrides`) and GB10 does not compile it. An unconditional probe
/// would leave a failed row the fail-closed startup audit refuses the boot on,
/// for a kernel this target is correct not to have.
#[track_caller]
pub(super) fn cublas_ssm_kernel(gpu: &dyn GpuBackend, func: &str) -> KernelHandle {
    if crate::layers::ops::target_defaults::resolved()
        .cublas
        .value
        .ssm
    {
        crate::layers::try_kernel(gpu, "fp8_scale_transpose", func)
    } else {
        KernelHandle(0)
    }
}

/// The tensor-core GDN chunked-PREFILL spine's handle
/// ([`ops::GDN_TC_SPINE_ENTRY`](crate::layers::ops::GDN_TC_SPINE_ENTRY) —
/// the SAME constant the serve's route line prints, so the log cannot name a
/// kernel other than the one bound here), GATED on the same bit that
/// launches it — `[defaults] gdn_prefill_tc`, with `ATLAS_GDN_PREFILL_TC`
/// overriding (`layers::ops::target_defaults`).
///
/// A probe that runs unconditionally asks the kernel audit about a module the
/// target may not enable, which is how a lever nobody set comes to be the
/// reason a boot failed. Off yields `KernelHandle(0)`, and
/// `ops::gdn_tc_spine_reject` then answers "not requested" — which is what it
/// would have answered anyway. Since round 13 `kernels/hopper` declares the row
/// TRUE, so on that target the probe runs by default and
/// `ATLAS_GDN_PREFILL_TC=0` is what silences it again.
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

/// The SCALAR fused GDN state-spine handle, and the one route line
/// `qwen3_ssm::init` prints per layer while binding it.
///
/// DEFAULT is `..._vfused` (SPLIT=2 / 256 threads): 2.01x over ksplit and 12/12
/// byte-identical on the ssm-poisoning tripwire. `ATLAS_GDN_VTILE=1` swaps in
/// the SPLIT=4 / 512-thread build, which is 2.15x but scores 1/12 there and
/// fails two accuracy gates — kept reachable for whoever diagnoses it, never
/// default. The two are ABI-identical apart from block size, which the launcher
/// derives from the same env, so nothing else downstream changes.
///
/// LOGGED, not silent: which spine ran is the single most consequential fact
/// about a GDN measurement, and a run record that cannot say which one it used
/// cannot be compared to another. An A/B on this kernel is otherwise
/// unfalsifiable — both arms produce a number either way.
///
/// `tc_spine` is the handle [`gdn_prefill_tc_kernel`] resolved just above, and
/// the line is built from it by
/// [`ops::gdn_init_spine_line`](crate::layers::ops::gdn_init_spine_line), so
/// what this prints is the entry the PREFILL will launch rather than the
/// fallback sitting underneath it — round 14 caught 48 of these lines naming
/// the scalar parent while all 14 400 dispatches went to the tensor-core entry.
pub(super) fn fused_spine_kernel(gpu: &dyn GpuBackend, tc_spine: KernelHandle) -> KernelHandle {
    use crate::layers::ops::{
        GDN_SCALAR_SPINE_PIPE, GDN_SCALAR_SPINE_VFUSED, GDN_SCALAR_SPINE_VTILE, gdn_init_spine_line,
    };
    let scalar = match (
        std::env::var("ATLAS_GDN_PIPE").ok().as_deref(),
        std::env::var("ATLAS_GDN_VTILE").ok().as_deref(),
    ) {
        (Some("1"), _) => GDN_SCALAR_SPINE_PIPE,
        (_, Some("1")) => GDN_SCALAR_SPINE_VTILE,
        _ => GDN_SCALAR_SPINE_VFUSED,
    };
    tracing::info!("{}", gdn_init_spine_line(tc_spine.0 != 0, scalar));
    crate::layers::try_kernel(gpu, "gated_delta_rule_fla", scalar)
}

// ── The four HOPPER-ONLY twin probes, one function each ───────────────────
//
// `try_kernel` and not `kernel` in all four: these modules exist only under
// `kernels/hopper` (declared in that target's `[kernels] overrides`), so on
// gb10/b200/strix the lookup must MISS quietly and leave the launcher on the
// parent kernel — exactly as `lib_tests.rs`'s exact-verify pins prescribe for
// a target-scoped kernel. A handle of 0 IS the "not on this target" answer;
// nothing downstream needs a second way to ask.
//
// One named function per handle rather than one helper taking two strings:
// the module/entry pair is the whole content of the probe, and a call site
// that passes them as arguments has simply moved the thing being reviewed
// back into `init.rs`. They live here for the 500-LoC cap, beside
// `gdn_prefill_tc_kernel`, which is the same shape for the third kernel of
// the same prefill family.

/// GDN decode recurrence, unstrided (#927).
pub(super) fn decode_hopper_k(gpu: &dyn GpuBackend) -> KernelHandle {
    crate::layers::try_kernel(
        gpu,
        "gdn_decode_hopper",
        "gated_delta_rule_decode_f32_hopper",
    )
}

/// GDN decode recurrence, one strided launch per batch (#927).
pub(super) fn decode_hopper_strided_k(gpu: &dyn GpuBackend) -> KernelHandle {
    crate::layers::try_kernel(
        gpu,
        "gdn_decode_hopper",
        "gated_delta_rule_decode_f32_strided_hopper",
    )
}

/// GDN decode recurrence, one strided launch per batch, state read ONCE for
/// 96 of its 128 rows (#927). A SECOND Hopper twin of the same parent as
/// [`decode_hopper_strided_k`] above, under its own `[defaults]` row, because
/// the two are different claims: that one re-partitions columns for the n=1
/// underfill and lost, this one keeps the partition and cuts state traffic on
/// the n >= 4 batched arm. The module and entry are named ONCE, in
/// `ops::ssm_gdn_strided_hopper`, so the probe and the route line cannot spell
/// different kernels.
pub(super) fn decode_hopper_strided_smem_k(gpu: &dyn GpuBackend) -> KernelHandle {
    crate::layers::try_kernel(
        gpu,
        crate::layers::ops::GDN_STRIDED_SMEM_MODULE,
        crate::layers::ops::GDN_STRIDED_SMEM_ENTRY,
    )
}

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

/// The SSM BA-gates twin: one CTA per token, bit-identical to the gb10 parent
/// (#928). Hopper-only source, so the lookup MISSES quietly everywhere else and
/// `ops::ba_gates_pick` keeps the launcher on `ssm_preprocess`'s parent.
///
/// NOT gated on `[defaults] ssm_ba_gates_hopper`, unlike the spine probe above:
/// the parent is always loaded and is always a valid launch, so a twin that is
/// merely absent costs nothing to have looked for, and the lever is read at the
/// dispatch site where the token-count guard is read too.
pub(super) fn ba_gates_hopper_k(gpu: &dyn GpuBackend) -> KernelHandle {
    crate::layers::try_kernel(
        gpu,
        "ssm_ba_gates_hopper",
        "dense_gemm_ba_gates_prefill_hopper",
    )
}
