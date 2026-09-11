// SPDX-License-Identifier: AGPL-3.0-only

//! The `gdn_decode_hopper` LEVER — the three answers the dispatch can give,
//! graded on the pure resolver rather than on the process environment.
//!
//! Why this file exists at all: #927 selected the Hopper GDN decode twins by
//! KERNEL PRESENCE, so "does the binary have the kernel" and "should the
//! binary run the kernel" were the same question and neither could be changed
//! without the other. H100 round 12 answered the second one no
//! (`GDN-DECODE-ATTRIBUTION.md`: 0.83x at contiguous n=1, +6.8% per C=1 nsys
//! step, -0.4% on the serve A/B, bit-identical throughout) while leaving the
//! first one yes — the kernel stays compiled and stays in
//! `kernels/hopper/HARDWARE.toml`'s `[kernels] overrides`, because the receipt
//! is per-target and the next Hopper part gets to be measured.
//!
//! `resolve` is driven with an explicit variable map: the production accessor
//! is a process-global `OnceLock`, so a test that set real variables would
//! seal it for every other test in this binary and grade whoever ran it.

use super::{gdn_decode_hopper_selected, gdn_decode_strided_hopper_selected};
use crate::layers::ops::target_defaults::{Source, TargetLevers, format_levers, resolve};
use atlas_kernels::TargetDefaults;
use spark_runtime::gpu::KernelHandle;

/// `kernels/hopper/HARDWARE.toml` `[defaults]`, abbreviated to the row under
/// test — every other field is irrelevant to this resolver and is spelled out
/// in `target_defaults_tests`.
const HOPPER: TargetDefaults = TargetDefaults {
    hw: "hopper",
    cublas_gemm_scope: "ffn,ssm,attn",
    ffn_batch16_tier: false,
    ffn_m16_tc: false,
    attn_m16_tc: true,
    attn_ncol_gemv: false,
    lm_head_m16_tc: true,
    lm_head_batchm_max: 16,
    ssm_batched_recurrent: true,
    gdn_decode_hopper: false,
    gdn_prefill_tc: true,
    ssm_ba_gates_hopper: true,
    ffn_gateup_fused: true,
    decode_split_silu: true,
    ssm_decode_ring_slots: "auto",
    w8a8_prefill_max_m_widening: u32::MAX,
    w8a8_prefill_max_m_narrowing: u32::MAX,
    attn_decode_splitk: "auto",
};

fn with(env: &[(&str, &str)]) -> TargetLevers {
    let env: Vec<(String, String)> = env
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect();
    resolve(&HOPPER, |name| {
        env.iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.to_owned())
    })
}

/// THE DEFAULT, and the deliverable: an H100 serve with NOTHING in its
/// environment leaves the twins OFF and stays on the gb10 parents. Before
/// round 12 this cell was ON and no variable could express otherwise.
#[test]
fn hopper_with_an_empty_environment_leaves_the_twins_off() {
    let l = with(&[]);
    assert!(!l.gdn_decode_hopper.value);
    assert_eq!(
        l.gdn_decode_hopper.source,
        Source::Target,
        "an untouched environment must attribute the value to the target"
    );
}

/// The POSITIVE lever — how the next Hopper part gets measured without an
/// edit to this repository.
#[test]
fn the_positive_lever_turns_the_twins_on_and_says_it_came_from_the_environment() {
    let l = with(&[("ATLAS_GDN_DECODE_HOPPER", "1")]);
    assert!(l.gdn_decode_hopper.value);
    assert!(l.gdn_decode_hopper.from_env());
}

/// The legacy kill switch OUTRANKS the positive, exactly as
/// `ATLAS_FFN_NO_BATCH16` outranks `ATLAS_FFN_BATCH16`: a hatch a stale
/// variable in the same shell could veto is not a hatch. Round 12's cell F was
/// run with this spelling and it still means what it meant then.
#[test]
fn the_legacy_kill_switch_outranks_the_positive_lever() {
    for env in [
        &[("ATLAS_NO_GDN_HOPPER", "1")][..],
        &[
            ("ATLAS_NO_GDN_HOPPER", "1"),
            ("ATLAS_GDN_DECODE_HOPPER", "1"),
        ][..],
    ] {
        let l = with(env);
        assert!(!l.gdn_decode_hopper.value, "{env:?}");
        assert!(l.gdn_decode_hopper.from_env(), "{env:?}");
    }
}

/// ⚠️ AND ITS POLARITY IS UNCHANGED. `ATLAS_NO_GDN_HOPPER` shipped documented
/// as `== "1"` and NOT presence, so `=0` never disabled the tier; it must not
/// start disabling it now that the tier is a resolved lever. `=0` is simply
/// not the kill switch, which leaves the positive lever (or the target's
/// declaration) to answer.
#[test]
fn the_kill_switch_keeps_its_equals_one_spelling() {
    assert!(
        with(&[
            ("ATLAS_NO_GDN_HOPPER", "0"),
            ("ATLAS_GDN_DECODE_HOPPER", "1")
        ])
        .gdn_decode_hopper
        .value,
        "ATLAS_NO_GDN_HOPPER=0 is not the kill switch and must not veto the \
         positive lever"
    );
    assert!(
        !with(&[("ATLAS_NO_GDN_HOPPER", "0")])
            .gdn_decode_hopper
            .value,
        "…and it is not an enable either: the target's declaration answers"
    );
}

/// `ATLAS_GDN_DECODE_HOPPER` is ONE lever and reaches ONE row. The prefill
/// family (`gdn_prefill_tc`) shares a kernel-family name and nothing else —
/// and since round 13 the two rows point OPPOSITE ways on this target, which
/// is the sharpest available statement that they are separate receipts: the
/// decode twins are a measured loss here and the prefill family a measured win.
#[test]
fn the_decode_lever_does_not_reach_the_prefill_spine() {
    let l = with(&[("ATLAS_GDN_DECODE_HOPPER", "1")]);
    assert!(l.gdn_decode_hopper.value);
    assert!(
        l.gdn_prefill_tc.value && l.gdn_prefill_tc.source == Source::Target,
        "the prefill family keeps the TARGET's value and is not marked (env): \
         a decode lever that silently re-sourced the prefill row would make a \
         serve log attribute the prefill win to the operator's shell"
    );
    assert!(l.ssm_batched_recurrent.value, "and nothing else moved");
}

/// The kill switch for the shipped default. `ATLAS_GDN_PREFILL_TC=0` turns the
/// WHOLE family off — the spine here, and both remnant twins because they read
/// this same resolved bit (`ssm_gdn_remnants_tests`) — and the boot line says
/// the environment did it, so an A/B leg is legible in its own log.
#[test]
fn the_prefill_family_kill_switch_turns_it_off_and_is_visible() {
    let l = with(&[("ATLAS_GDN_PREFILL_TC", "0")]);
    assert!(!l.gdn_prefill_tc.value && l.gdn_prefill_tc.source == Source::Env);
    assert!(
        format_levers(&l).contains("gdn_prefill_tc=off (env)"),
        "{}",
        format_levers(&l)
    );
    assert!(
        !l.gdn_decode_hopper.value && l.ssm_batched_recurrent.value,
        "and it reaches nothing else"
    );
}

/// THE BOOT LINE. An operator reading a serve log must be able to tell which
/// GDN decode kernel ran without also having the launch script — that is the
/// whole point of the `target defaults (<hw>): …` line, and round 12 had to
/// settle this question from an nsys trace because the twins had no route
/// line at all. Both provenances are graded: the target's declaration prints
/// bare, an override prints ` (env)`.
#[test]
fn the_boot_line_names_the_lever_and_marks_an_override() {
    let line = format_levers(&with(&[]));
    assert!(
        line.contains("gdn_decode_hopper=off"),
        "the default must be visible in the line:\n{line}"
    );
    assert!(
        line.contains("gdn_prefill_tc=on") && !line.contains("gdn_prefill_tc=on (env)"),
        "…and so must the prefill family, which this target now ships ON from \
         its own declaration:\n{line}"
    );
    assert!(
        !line.contains("gdn_decode_hopper=off (env)"),
        "an untouched environment marks nothing:\n{line}"
    );

    let line = format_levers(&with(&[("ATLAS_GDN_DECODE_HOPPER", "1")]));
    assert!(line.contains("gdn_decode_hopper=on (env)"), "{line}");

    // …and the kill switch is an override too, so cell F's recipe is legible
    // in its own log rather than only in the operator's notes.
    let line = format_levers(&with(&[("ATLAS_NO_GDN_HOPPER", "1")]));
    assert!(line.contains("gdn_decode_hopper=off (env)"), "{line}");
}

// ── the handle/geometry half of the same decision ──
//
// `gdn_decode_hopper_selected` folds the lever together with a resolved handle
// and the kernel's dimension contract. The lever itself reads the process
// `OnceLock`, which under `ATLAS_SKIP_BUILD` resolves to the default target's
// table — false on every target — so what is assertable here without touching
// the environment is the SHORT-CIRCUIT: a zero handle and an illegal shape are
// refused no matter what the lever says.

/// A NULL handle is the "this target does not have the twin" answer, and it is
/// checked before anything else. Every non-hopper build takes this arm.
#[test]
fn a_null_handle_never_selects_the_twin() {
    assert!(!gdn_decode_hopper_selected(KernelHandle(0), 128, 128));
    assert!(!gdn_decode_strided_hopper_selected(
        KernelHandle(0),
        128,
        128
    ));
}

/// …and the dimension contract is enforced on the same call, so a shape the
/// kernel would read out of bounds cannot be reached by setting a variable.
#[test]
fn an_illegal_shape_never_selects_the_twin() {
    // k_dim not a multiple of 4, k_dim over the 128 smem bound, and — for the
    // strided twin only — a v_dim its head-wide norm reduction cannot split.
    assert!(!gdn_decode_hopper_selected(KernelHandle(0x9D), 130, 128));
    assert!(!gdn_decode_hopper_selected(KernelHandle(0x9D), 132, 128));
    assert!(!gdn_decode_strided_hopper_selected(
        KernelHandle(0x9D),
        128,
        64
    ));
}
