// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: CPU tests for the Ada tensor-core chunked GDN prefill:
//!   1. the FP64 chunk algebra against the FP64 recurrence (T = 1 to 4096,
//!      partial last chunks, GQA head mapping, the gate floor);
//!   2. the acceptance criteria: both emulated route profiles meet them, the
//!      two routes agree within them, and corruption fails them;
//!   3. the host grammar: eligibility verdicts, scratch carving against the
//!      FLA formula, shared-memory budgets, the output grid, the launch ABI on
//!      the mock backend and lazy resolution.
//!
//! The kernel's index maps are simulated in `gdn_chunk_tc_ada/maps_tests.rs`.

// Index loops mirror the kernel's index arithmetic.
#![allow(clippy::needless_range_loop)]

use std::collections::HashSet;

use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};

use super::emulate::{scalar_route, tc_route, tf32_round};
use super::oracle::{self, OUTPUT, STATE, compare, generate};
use super::*;

// ── 1. FP64 chunk algebra ────────────────────────────────────────────────────

#[test]
fn chunked_fp64_matches_recurrent() {
    // (t, nk, nv, kd, vd): T from one token to 4096, chunk boundaries at
    // b-1 / b / b+1, a GQA ratio of 2, and narrow heads for the long runs (the
    // algebra is width-generic; the kernel's 128 is exercised up to T = 150).
    let cases = [
        (1, 1, 1, 128, 128),
        (63, 1, 2, 128, 128),
        (64, 1, 2, 128, 128),
        (65, 1, 2, 128, 128),
        (150, 2, 4, 128, 128),
        (1000, 1, 2, 32, 32),
        (4096, 1, 2, 16, 16),
    ];
    for (i, &(t, nk, nv, kd, vd)) in cases.iter().enumerate() {
        let p = generate(0x5EED_0000 + i as u64, t, nk, nv, kd, vd);
        let r = oracle::recurrent(&p);
        let c = oracle::chunked(&p, 64);
        let eo = compare(&c.out, &r.out);
        let es = compare(&c.state, &r.state);
        assert!(eo.finite && es.finite, "t={t}: non-finite");
        assert!(eo.normwise < 1e-9, "t={t} kd={kd}: output normwise {eo:?}");
        assert!(es.normwise < 1e-9, "t={t} kd={kd}: state normwise {es:?}");
    }
}

#[test]
fn chunked_fp64_is_chunk_size_invariant() {
    let p = generate(0xC0FFEE, 200, 1, 1, 32, 32);
    let a = oracle::chunked(&p, 64);
    for chunk in [1, 7, 16, 200, 512] {
        let b = oracle::chunked(&p, chunk);
        assert!(compare(&b.out, &a.out).normwise < 1e-9, "chunk={chunk}");
        assert!(compare(&b.state, &a.state).normwise < 1e-9, "chunk={chunk}");
    }
}

#[test]
fn gate_floor_keeps_the_chunk_form_finite() {
    // A gate of exactly 0 (a deep layer's underflow) is log(0) = -inf without
    // the floor; with it, exp(gc_i - gc_l) stays defined and full decay holds.
    let mut p = generate(0x0F10, 130, 1, 2, 32, 32);
    for t in [0usize, 5, 63, 64, 100] {
        p.gate[t * p.nv] = 0.0;
    }
    let r = oracle::recurrent(&p);
    let c = oracle::chunked(&p, 64);
    let eo = compare(&c.out, &r.out);
    let es = compare(&c.state, &r.state);
    assert!(eo.finite && es.finite);
    assert!(eo.normwise < 1e-6 && es.normwise < 1e-6, "{eo:?} {es:?}");
}

// ── 2. Criteria ──────────────────────────────────────────────────────────────

#[test]
fn route_profiles_meet_the_criteria_and_agree() {
    for (i, t) in [1usize, 63, 64, 65, 150, 256].into_iter().enumerate() {
        let p = generate(0xA11CE + i as u64, t, 1, 2, 128, 128);
        let r = oracle::recurrent(&p);
        let tc = tc_route(&p);
        let sc = scalar_route(&p);
        let (to, ts) = (compare(&tc.out, &r.out), compare(&tc.state, &r.state));
        let (so, ss) = (compare(&sc.out, &r.out), compare(&sc.state, &r.state));
        assert!(OUTPUT.accepts(&to), "t={t}: TC output {to:?}");
        assert!(STATE.accepts(&ts), "t={t}: TC state {ts:?}");
        assert!(OUTPUT.accepts(&so), "t={t}: scalar output {so:?}");
        assert!(STATE.accepts(&ss), "t={t}: scalar state {ss:?}");
        // Agreement with the scalar spine within the same bound.
        let agree = compare(&tc.out, &sc.out);
        assert!(OUTPUT.accepts(&agree), "t={t}: TC vs scalar {agree:?}");
    }
}

#[test]
fn criteria_reject_corruption() {
    let p = generate(0xBAD, 70, 1, 1, 128, 128);
    let r = oracle::recurrent(&p);
    assert!(OUTPUT.accepts(&compare(&r.out, &r.out)));
    let ref_max = r.out.iter().fold(0f64, |m, x| m.max(x.abs()));
    // One gross element: invisible in the norm of 8960 values, caught by the cap.
    let mut one = r.out.clone();
    one[1234] += ref_max;
    assert!(!OUTPUT.accepts(&compare(&one, &r.out)));
    // A NaN anywhere.
    let mut nan = r.out.clone();
    nan[7] = f64::NAN;
    assert!(!OUTPUT.accepts(&compare(&nan, &r.out)));
    // A uniform 5% scale error.
    let scaled: Vec<f64> = r.out.iter().map(|x| x * 1.05).collect();
    assert!(!OUTPUT.accepts(&compare(&scaled, &r.out)));
    // Two token rows swapped (a store-map bug).
    let mut swapped = r.out.clone();
    let row = p.nv * p.vd;
    for c in 0..row {
        swapped.swap(3 * row + c, 40 * row + c);
    }
    assert!(!OUTPUT.accepts(&compare(&swapped, &r.out)));
}

#[test]
fn tf32_rounding_is_nearest_ties_away() {
    assert_eq!(tf32_round(1.0), 1.0);
    assert_eq!(tf32_round(1.0 + 2f32.powi(-11)), 1.0 + 2f32.powi(-10));
    assert_eq!(tf32_round(-(1.0 + 2f32.powi(-11))), -(1.0 + 2f32.powi(-10)));
    assert_eq!(tf32_round(1.0 + 2f32.powi(-12)), 1.0);
    for x in [0.1f32, -3.7, 12345.678, 1e-20] {
        let y = tf32_round(x);
        assert_eq!(y.to_bits() & 0x1FFF, 0, "{x}");
        assert!(((y - x) / x).abs() <= 2f32.powi(-11), "{x}");
    }
}

// ── 3. Host grammar ──────────────────────────────────────────────────────────

fn args(seq_len: u32) -> TcAdaArgs {
    TcAdaArgs {
        h_state: DevicePtr(0x1000_0000),
        h_state_is_table: false,
        q: DevicePtr(0x2000_0000),
        k: DevicePtr(0x2000_1000),
        v: DevicePtr(0x2000_2000),
        gate: DevicePtr(0x3000_0000),
        beta: DevicePtr(0x3000_0080),
        out: DevicePtr(0x4000_0000),
        // Qwen3.5-4B: 16 key heads, 32 value heads, conv_dim 8192.
        shape: TcAdaShape::packed(1, seq_len, 16, 32, 128, 128, 8192),
        midchunk_capture: false,
    }
}

const BIG: usize = usize::MAX / 4;

#[test]
fn packed_shape_is_the_prefill_layout() {
    let s = TcAdaShape::packed(2, 150, 16, 32, 128, 128, 8192);
    assert_eq!((s.qk_stride, s.v_stride, s.gb_stride), (8192, 8192, 64));
    assert_eq!(s.num_chunks(), 3);
    assert_eq!(
        TcAdaShape::packed(1, 64, 1, 1, 128, 128, 384).num_chunks(),
        1
    );
}

#[test]
fn reject_names_each_guard() {
    assert_eq!(reject(&args(3921), BIG, false), None);
    assert_eq!(reject(&args(0), BIG, false), Some(Reject::Empty));
    assert_eq!(reject(&args(3921), BIG, true), Some(Reject::ExactReplay));
    let mut a = args(3921);
    a.midchunk_capture = true;
    assert_eq!(reject(&a, BIG, false), Some(Reject::MidchunkCapture));
    let mut a = args(3921);
    a.shape.kd = 64;
    assert_eq!(reject(&a, BIG, false), Some(Reject::HeadDims));
    let mut a = args(3921);
    a.shape.nk = 12;
    assert_eq!(reject(&a, BIG, false), Some(Reject::HeadRatio));
    let mut a = args(3921);
    a.shape.nk = 0;
    assert_eq!(reject(&a, BIG, false), Some(Reject::HeadRatio));
    let mut a = args(3921);
    a.shape.qk_stride = 8196;
    assert_eq!(reject(&a, BIG, false), Some(Reject::Strides));
    let mut a = args(3921);
    a.k = a.k.offset(2);
    assert_eq!(reject(&a, BIG, false), Some(Reject::Alignment));
    assert_eq!(reject(&args(3921), 0, false), Some(Reject::NoScratch));
    let need = ScratchLayout::bytes(1, 3921u32.div_ceil(64), 32);
    assert_eq!(reject(&args(3921), need, false), None);
    assert_eq!(
        reject(&args(3921), need - 1, false),
        Some(Reject::ScratchTooSmall)
    );
    let mut a = args(64);
    a.shape.batch = 70_000;
    assert_eq!(reject(&a, BIG, false), Some(Reject::GridLimit));
    // Precedence: routing-by-design reasons come first.
    let mut a = args(3921);
    a.shape.kd = 64;
    assert_eq!(reject(&a, 0, true), Some(Reject::ExactReplay));
}

#[test]
fn reject_keys_are_distinct_and_classified() {
    let all = [
        Reject::Empty,
        Reject::ExactReplay,
        Reject::MidchunkCapture,
        Reject::HeadDims,
        Reject::HeadRatio,
        Reject::Strides,
        Reject::Alignment,
        Reject::NoScratch,
        Reject::ScratchTooSmall,
        Reject::GridLimit,
    ];
    let keys: HashSet<_> = all.iter().map(|r| r.key()).collect();
    assert_eq!(keys.len(), all.len());
    for r in all {
        assert!(r.key().starts_with("log:gdn_tc_ada_"));
        assert!(!r.why().is_empty() && !r.why().contains('\u{2014}'));
    }
    assert_eq!(all.iter().filter(|r| r.expected()).count(), 3);
}

#[test]
fn scratch_is_carved_like_the_fla_path() {
    // trait_prefill_recur.rs: w, u = w + nt*nv*64*kd*2, s = u + nt*nv*64*vd*2,
    // uc = s + nt*nv*kd*vd*2, gc = uc + nt*nv*64*vd*2; sizes.rs adds gc's
    // nt*nv*64*4. nt counts batch * chunks.
    let (batch, nc, nv, kd, vd) = (3u32, 7u32, 32usize, 128usize, 128usize);
    let nt = (batch * nc) as usize;
    let base = DevicePtr(0x7000_0000);
    let l = ScratchLayout::carve(base, batch, nc, nv as u32);
    assert_eq!(l.w, base);
    assert_eq!(l.u, base.offset(nt * nv * 64 * kd * 2));
    assert_eq!(l.h_chunk, l.u.offset(nt * nv * 64 * vd * 2));
    assert_eq!(l.v_new, l.h_chunk.offset(nt * nv * kd * vd * 2));
    assert_eq!(l.gc, l.v_new.offset(nt * nv * 64 * vd * 2));
    let fla = nt * nv * 64 * kd * 2
        + nt * nv * 64 * vd * 2
        + nt * nv * kd * vd * 2
        + nt * nv * 64 * vd * 2
        + nt * nv * 64 * 4;
    assert_eq!(ScratchLayout::bytes(batch, nc, nv as u32), fla);
    assert_eq!((l.gc.0 - base.0) as usize + nt * nv * 64 * 4, fla);
}

#[test]
fn shared_memory_fits_sm89_without_opt_in() {
    assert_eq!(PREPARE_SMEM, 24_320);
    assert_eq!(STATE_SMEM, 45_312);
    assert_eq!(OUTPUT_SMEM, 24_576);
    // Under the 48 KiB default, and the resident CTAs fit the SM's 100 KiB.
    for s in [PREPARE_SMEM, STATE_SMEM, OUTPUT_SMEM] {
        assert!(s <= 48 * 1024);
    }
    // Compile-time checks: the operands are constants.
    const { assert!(4 * PREPARE_SMEM <= 100 * 1024) };
    const { assert!(2 * STATE_SMEM <= 100 * 1024) };
    const { assert!(OUTPUT_CTAS_PER_SM * OUTPUT_SMEM <= 100 * 1024) };
}

#[test]
fn output_grid_covers_every_chunk_once() {
    assert_eq!(output_grid_chunks(61, 32, 1, 34), 5);
    assert_eq!(output_grid_chunks(1, 32, 1, 34), 1);
    assert_eq!(output_grid_chunks(2, 1, 1, 34), 2);
    assert_eq!(output_grid_chunks(0, 32, 1, 34), 1);
    let cases = [
        (61u32, 32u32, 1u32, 34u32),
        (2048, 32, 1, 34),
        (7, 16, 4, 128),
        (500, 8, 2, 1),
    ];
    for (nc, nv, b, sm) in cases {
        let gx = output_grid_chunks(nc, nv, b, sm);
        assert!((1..=nc.max(1)).contains(&gx));
        let mut seen = vec![0u32; nc as usize];
        for x in 0..gx {
            let mut c = x;
            while c < nc {
                seen[c as usize] += 1;
                c += gx;
            }
        }
        assert!(seen.iter().all(|&n| n == 1), "nc={nc} gx={gx}");
        // About one resident wave: at most one grid column of CTAs past it.
        let ctas = gx as u64 * nv as u64 * b as u64;
        assert!(
            ctas <= (sm * OUTPUT_CTAS_PER_SM) as u64 + nv as u64 * b as u64,
            "nc={nc}"
        );
    }
}

fn u32_arg(a: &MockArg) -> u32 {
    match a {
        MockArg::Bytes(b) => u32::from_le_bytes(b[..4].try_into().unwrap()),
        MockArg::Buffer(p) => panic!("expected a u32, got buffer {p:?}"),
    }
}

#[test]
fn launch_matches_the_kernel_abi() {
    let gpu = MockGpuBackend::new();
    let kern = Kernels {
        prepare: KernelHandle(11),
        state: KernelHandle(22),
        output: KernelHandle(33),
        sm_count: 34,
    };
    let mut a = args(150);
    a.shape = TcAdaShape::packed(2, 150, 2, 4, 128, 128, 1024);
    a.h_state_is_table = true;
    let scratch = DevicePtr(0x9000_0000);
    launch(&gpu, &kern, &a, scratch, 7).unwrap();
    let l = gpu.launches_snapshot();
    assert_eq!(l.len(), 3);
    let carve = ScratchLayout::carve(scratch, 2, 3, 4);
    let geo = |i: usize| (l[i].func, l[i].grid, l[i].block, l[i].shared_mem);
    // Stage 1.
    assert_eq!(geo(0), (11, [3, 4, 2], [128, 1, 1], PREPARE_SMEM));
    assert_eq!(l[0].args.len(), 14);
    assert_eq!(l[0].args[0], MockArg::Buffer(a.k));
    assert_eq!(l[0].args[4], MockArg::Buffer(carve.w));
    assert_eq!(l[0].args[6], MockArg::Buffer(carve.gc));
    let s1: Vec<u32> = l[0].args[7..].iter().map(u32_arg).collect();
    assert_eq!(s1, vec![150, 3, 2, 4, 1024, 1024, 8]);
    // Stage 2.
    assert_eq!(geo(1), (22, [32, 2, 1], [256, 1, 1], STATE_SMEM));
    assert_eq!(l[1].args.len(), 13);
    assert_eq!(l[1].args[0], MockArg::Buffer(a.h_state));
    assert_eq!(l[1].args[5], MockArg::Buffer(carve.v_new));
    assert_eq!(l[1].args[6], MockArg::Buffer(carve.h_chunk));
    let s2: Vec<u32> = l[1].args[7..].iter().map(u32_arg).collect();
    assert_eq!(s2, vec![150, 3, 2, 4, 1024, 1]);
    // Stage 3.
    let gx = output_grid_chunks(3, 4, 2, 34);
    assert_eq!(geo(2), (33, [gx, 4, 2], [128, 1, 1], OUTPUT_SMEM));
    assert_eq!(l[2].args.len(), 11);
    assert_eq!(l[2].args[0], MockArg::Buffer(a.q));
    assert_eq!(l[2].args[5], MockArg::Buffer(a.out));
    let s3: Vec<u32> = l[2].args[6..].iter().map(u32_arg).collect();
    assert_eq!(s3, vec![150, 3, 2, 4, 1024]);
    assert!(l.iter().all(|x| x.stream == 7));
}

#[test]
fn handles_resolve_only_when_the_module_is_present() {
    // Every target but Ada: no module, no lookup at all.
    let gpu = MockGpuBackend::new();
    gpu.mark_module_absent(MODULE);
    let h = Handles::default();
    assert!(h.resolve(&gpu).is_none());
    assert!(gpu.kernel_lookups_snapshot().is_empty());
    // Ada: the three entries, the machine width, cached after the first call.
    let gpu = MockGpuBackend::new();
    let h = Handles::default();
    let k = h.resolve(&gpu).expect("module present");
    assert_eq!(k.sm_count, 48);
    assert!(h.resolve(&gpu).is_some());
    let lookups = gpu.kernel_lookups_snapshot();
    assert_eq!(lookups.len(), 3);
    for (m, f) in &lookups {
        assert_eq!(m, MODULE);
        assert!([ENTRY_PREPARE, ENTRY_STATE, ENTRY_OUTPUT].contains(&f.as_str()));
    }
    // A missing entry disables the arm instead of launching a null handle.
    let gpu = MockGpuBackend::new();
    gpu.deny_kernel(MODULE, ENTRY_STATE);
    assert!(Handles::default().resolve(&gpu).is_none());
}
