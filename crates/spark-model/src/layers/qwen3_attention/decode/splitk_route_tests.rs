// SPDX-License-Identifier: AGPL-3.0-only

//! The dispatch-side paged-decode route line, graded as TEXT.
//!
//! H100 round 15 anomaly 3: "there is no dispatch-side route line naming the
//! chosen split count at all". The kernel-selection table printed
//! `paged_decode_fp8_splitk_hopper … used`, which proves the entry RESOLVED —
//! not that eleven splits reached the launch. The `(24,11,1)` grid was only
//! visible in an nsys capture, which is not a reasonable standing cost for
//! confirming a shipped lever from a serve log (#928).
//!
//! Pinned WHOLE, for the reason `tests/build_summary.rs` pins its line whole: a
//! format assembled from separately asserted pieces can be reordered without a
//! test noticing, and the value of this line is that it is greppable out of a
//! campaign log months later.

use super::splitk_dispatch::{
    ROUTE_NONSPLIT_BF16, ROUTE_NONSPLIT_FP8, ROUTE_SPLITK_BF16, ROUTE_SPLITK_FP8, route_line,
};
use avarok_kernels::attn_splitk::SplitkPolicy;

/// THE line, as round 15 asked for it, for the FP8 arm on Hopper under `auto`.
///
/// `sm_count` is read from `avarok_kernels::TARGET_SM_COUNT` — the compiled
/// target's, the same number the policy divided by — so this test states the
/// expectation against that constant rather than hardcoding 132 and passing on
/// a gb10 build for the wrong reason.
#[test]
fn the_fp8_route_line_names_the_kernel_the_split_count_and_the_policy() {
    let sm = avarok_kernels::TARGET_SM_COUNT;
    assert_eq!(
        route_line(ROUTE_SPLITK_FP8, 11, SplitkPolicy::Auto),
        format!(
            "paged decode attention: paged_decode_attn_splitk_fp8_hopper num_splits=11 \
             sm_count={sm} policy=auto (AVAROK_ATTN_DECODE_SPLITK)"
        ),
    );
}

/// The BF16 twin says the same line about its own kernel. Qwen3.8-27B runs FP8
/// KV on 44 layers and BF16 on the four `--kv-high-precision-layers auto` ones,
/// and both took the split-K path in round 15 — nsys priced them separately at
/// 32.7 and 35.0 µs/launch — so a log that named only one arm would be as
/// incomplete as no line at all.
#[test]
fn the_bf16_twin_reports_its_own_arm() {
    let line = route_line(ROUTE_SPLITK_BF16, 11, SplitkPolicy::Auto);
    assert!(
        line.contains("paged_decode_attn_splitk_bf16_hopper"),
        "{line}"
    );
    assert!(line.contains("num_splits=11"), "{line}");
    assert_ne!(line, route_line(ROUTE_SPLITK_FP8, 11, SplitkPolicy::Auto));
}

/// `AVAROK_ATTN_DECODE_SPLITK=0` is the pre-#928 control, and the line must say
/// so honestly: the NON-split kernel, at one split, under a policy that renders
/// as `1`. Round 15's S0 cell booted `attn_decode_splitk=1 (env)` from a raw
/// `0`, which is correct (`0`/`off` resolves to one split) and confusing
/// without this second line naming the kernel that then ran.
#[test]
fn the_zero_control_reports_the_non_split_kernel_at_one_split() {
    assert_eq!(
        route_line(ROUTE_NONSPLIT_FP8, 1, SplitkPolicy::Pinned(1)),
        format!(
            "paged decode attention: paged_decode_attn_fp8 num_splits=1 sm_count={} \
             policy=1 (AVAROK_ATTN_DECODE_SPLITK)",
            avarok_kernels::TARGET_SM_COUNT,
        ),
    );
    // And the policy field round-trips the boot line's own spelling, so the two
    // lines in one log cannot disagree about the lever.
    for (policy, label) in [
        (SplitkPolicy::Legacy, "legacy"),
        (SplitkPolicy::Auto, "auto"),
        (SplitkPolicy::Pinned(6), "6"),
    ] {
        assert!(
            route_line(ROUTE_NONSPLIT_BF16, 1, policy).contains(&format!("policy={label} ")),
            "policy {policy:?} must render as {label}",
        );
    }
}

/// The environment variable that moves it is NAMED, the way every other route
/// line in this crate names its lever — an operator who reads the line has the
/// knob without grepping the source.
#[test]
fn the_line_names_its_lever_and_leads_with_the_kernel() {
    let line = route_line(ROUTE_SPLITK_FP8, 11, SplitkPolicy::Auto);
    assert!(line.ends_with("(AVAROK_ATTN_DECODE_SPLITK)"), "{line}");
    assert!(
        line.starts_with("paged decode attention: paged_decode_attn_splitk_fp8_hopper "),
        "{line}"
    );
}

// ════════════════════════════════════════════════════════════════════════════
// The GQA-packed non-split arms.
// ════════════════════════════════════════════════════════════════════════════

use super::splitk_dispatch::{ROUTE_GQA_BF16, ROUTE_GQA_FP8, gqa_pack_kernel, gqa_pack_route};
use spark_runtime::gpu::KernelHandle;

/// The packed arms name THEIR kernel in the route line, at `num_splits=1`.
///
/// A campaign log has to be able to say which of the two non-split kernels
/// ran: they are bit-identical by construction, so the OUTPUT cannot tell them
/// apart and this line is the only receipt that the lever reached the launch.
/// Same failure `route_line` was added for (#928, round 15).
#[test]
fn the_packed_arms_report_their_own_kernel_at_one_split() {
    for kernel in [ROUTE_GQA_FP8, ROUTE_GQA_BF16] {
        let line = route_line(kernel, 1, SplitkPolicy::Legacy);
        assert!(line.contains(kernel), "{line}");
        assert!(line.contains("num_splits=1"), "{line}");
    }
    // …and each is distinguishable from the unpacked kernel whose place it
    // takes, which is the whole point of logging it.
    assert_ne!(ROUTE_GQA_FP8, ROUTE_NONSPLIT_FP8);
    assert_ne!(ROUTE_GQA_BF16, ROUTE_NONSPLIT_BF16);
}

/// ★ The packed route needs ALL THREE of the lever, the shape and the handle.
///
/// Stated as a conjunction because each leg fails differently and two of them
/// fail SILENTLY: a wrong GQA ratio makes the kernel index the wrong query
/// heads (right output shape, wrong contents), and a missing handle is a
/// launch into nothing.
///
/// Graded through `gqa_pack_route` with `armed` passed in, not through
/// `gqa_pack_kernel`: the lever is a process-wide `OnceLock` that a default
/// test run always resolves `false`, so every shape assertion made through the
/// public entry would pass at the FIRST line and prove nothing about the shape
/// check at all.
#[test]
fn the_packed_route_needs_the_lever_the_shape_and_the_handle() {
    let handle = Some(KernelHandle(7));

    // Leg 1 — the lever. Declared off, so a default build takes the unpacked
    // kernel and nothing about the shipped path changes.
    assert!(!avarok_kernels::attn_splitk::DECODE_GQA_PACK_DECLARED);
    assert!(gqa_pack_route(false, handle, 24, 4, 256).is_none());
    assert_eq!(
        gqa_pack_kernel(handle, 24, 4, 256).is_some(),
        avarok_kernels::attn_splitk::gqa_pack_enabled(),
        "the public entry must be the route function at the resolved lever"
    );

    // Leg 2 — the shape, with the lever ARMED so the assertion is real.
    // `(25, 4)` is the truncation trap: 25 / 4 == 6 in integer division while
    // 25 != 4 * 6.
    for (nq, nkv, hd) in [
        (32u32, 4u32, 256u32),
        (16, 4, 256),
        (24, 24, 256),
        (24, 1, 256),
        (25, 4, 256),
        (24, 4, 128),
        (24, 4, 512),
        (0, 0, 256),
    ] {
        assert!(
            gqa_pack_route(true, handle, nq, nkv, hd).is_none(),
            "nq={nq} nkv={nkv} hd={hd} must not reach the packed kernel"
        );
    }
    // …and the one shape it does serve, so the loop above is not passing
    // because the function refuses everything.
    assert_eq!(
        gqa_pack_route(true, handle, 24, 4, 256).map(|h| h.0),
        handle.map(|h| h.0)
    );

    // Leg 3 — the handle. Armed and correctly shaped is still not enough on a
    // build whose kernel tree does not carry the packed sources.
    assert!(gqa_pack_route(true, None, 24, 4, 256).is_none());
}

/// `kernels/gb10/common/<name>`, from this crate's manifest dir.
fn kernel_src(name: &str) -> String {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../kernels/gb10/common")
        .join(name);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

/// Parameters of an `extern "C" __global__` entry, by name.
///
/// Line comments are stripped first: the signatures carry `// [num_seqs,
/// num_q_heads, head_dim]` shape notes whose commas would otherwise be
/// counted as parameter separators.
fn cuda_param_count(src: &str, kernel: &str) -> usize {
    let at = src
        .find(&format!("{kernel}(\n"))
        .unwrap_or_else(|| panic!("no entry `{kernel}(` in source"));
    let mut lines = src[at..].lines();
    lines.next(); // the `<kernel>(` line itself
    let mut params = String::new();
    let mut closed = false;
    for line in lines {
        let code = line.split("//").next().unwrap_or("");
        // The closing paren is `) {` at column 0 in the unpacked kernels and
        // indented in the packed ones (they carry `__launch_bounds__`, which
        // rustfmt-style continuation indents). Match either.
        if code.trim_start().starts_with(')') {
            closed = true;
            break;
        }
        params.push_str(code);
        params.push('\n');
    }
    assert!(closed, "`{kernel}`: parameter list has no closing paren");
    params.matches(',').count() + 1
}

/// `.arg_*` calls in a launcher function in `ops/prefill_attn_a.rs`.
fn launcher_arg_count(name: &str) -> usize {
    let src = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/layers/ops/prefill_attn_a.rs"),
    )
    .expect("read prefill_attn_a.rs");
    let at = src
        .find(&format!("pub fn {name}(\n"))
        .unwrap_or_else(|| panic!("no launcher `{name}`"));
    let body = &src[at..];
    let end = body[1..].find("\npub fn ").map_or(body.len(), |e| e + 1);
    body[..end].matches(".arg_").count()
}

/// ★ Launcher arg count == compiled kernel parameter count.
///
/// `cuLaunchKernel`'s `void**` param form reads one host word per COMPILED
/// parameter, so a launcher that passes fewer args makes the DRIVER read past
/// the end of the arg array — `CUDA_ERROR_INVALID_VALUE` or a host SIGSEGV
/// depending on the neighbouring heap word, and no runtime API catches it.
/// That is how `w4a16_gemm_t_m128_bf16_v2` shipped broken
/// (`avarok-kernels/tests/kernel_arity.rs`); that pin needs a non-stub build
/// with nvcc and is `#[ignore]`d, so these two new entry points are pinned
/// here instead, against the SOURCES, where CI runs it every time.
///
/// The packed kernels deliberately take the SAME argument list as the
/// unpacked ones they stand in for — only the grid differs — so each pair is
/// asserted equal too: a parameter added to one and not the other is the
/// drift this catches.
#[test]
fn every_packed_launcher_passes_exactly_the_kernel_parameter_count() {
    let cases = [
        (
            "paged_decode_attn_fp8_gqa.cu",
            "paged_decode_attn_fp8_gqa",
            "paged_decode_attn_fp8.cu",
            "paged_decode_attn_fp8",
        ),
        (
            "paged_decode_attn_bf16_gqa.cu",
            "paged_decode_attn_bf16_gqa",
            "paged_decode_attn.cu",
            "paged_decode_attn",
        ),
    ];
    for (packed_file, packed_entry, base_file, base_entry) in cases {
        let params = cuda_param_count(&kernel_src(packed_file), packed_entry);
        let args = launcher_arg_count(packed_entry);
        assert_eq!(
            args, params,
            "{packed_entry}: launcher passes {args} args, kernel declares {params} params"
        );
        assert_eq!(
            params,
            cuda_param_count(&kernel_src(base_file), base_entry),
            "{packed_entry} must take the same argument list as {base_entry} — \
             it is a drop-in for it, differing only in grid"
        );
    }
}
