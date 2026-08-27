// SPDX-License-Identifier: AGPL-3.0-only

//! Equivalence pins for the batched SSM verify-state copies.
//!
//! The property under test is the one the `ssm-state-poisoning-gate` exists
//! for: **the same bytes must land in the same slots as the per-layer
//! `copy_d2d_async` loop**, for representative per-sequence rollback plans.
//! These are executor tests: the plans deliberately mirror the dispatcher,
//! but do not instantiate `TransformerModel` or invoke the dispatcher itself.
//! So every
//! test here runs BOTH executors over the SAME plan on two identically-built
//! mock devices and compares the destination regions byte-for-byte — a test
//! that only asserted "one launch instead of 48" would pass on a batched form
//! that copied the wrong rows.

use super::super::ssm_pool::SsmStatePool;
use super::*;
use crate::ssm_reserve::SsmRollbackMode;
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::{DevicePtr, GpuBackend, mock::MockGpuBackend};

/// The 27B/80B hybrid layer pattern (3 GDN : 1 attention, 48 layers → 36 SSM
/// layers) at doll-house blob widths, so a test allocates kilobytes instead
/// of the gigabytes the production head dims imply. The GEOMETRY under test
/// is the pool's layer/slot/intermediate striding, which is independent of
/// the blob width; the width only has to be uniform and non-trivial.
fn tiny_config() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.linear_num_key_heads = 2;
    c.linear_key_head_dim = 4;
    c.linear_num_value_heads = 2;
    c.linear_value_head_dim = 4;
    c.linear_conv_kernel_dim = 4;
    c
}

const MAX_SLOTS: usize = 4;
const NUM_INTERMEDIATES: usize = 4;
const NUM_DRAFTS: usize = 3;

fn pool(gpu: &MockGpuBackend) -> SsmStatePool {
    SsmStatePool::new(
        &tiny_config(),
        MAX_SLOTS,
        true,
        NUM_INTERMEDIATES,
        NUM_DRAFTS,
        false,
        SsmRollbackMode::Snapshot,
        gpu,
    )
    .unwrap()
}

/// Distinct, position-encoding fill for every source region, so restoring
/// from the WRONG layer / slot / token index is visible in the bytes rather
/// than hidden behind a uniform pattern.
fn seed(gpu: &MockGpuBackend, p: &SsmStatePool) {
    for l in 0..p.num_ssm_layers {
        for s in 0..=p.mtp_slots {
            let tag = |k: usize| ((l * 131 + s * 17 + k * 7) % 251 + 1) as u8;
            gpu.copy_h2d(&vec![tag(0); p.h_stored_bytes], p.h_state(l, s))
                .unwrap();
            gpu.copy_h2d(&vec![tag(1); p.conv_bytes], p.conv_state(l, s))
                .unwrap();
            gpu.copy_h2d(&vec![tag(2); p.h_stored_bytes], p.h_checkpoint(l, s))
                .unwrap();
            gpu.copy_h2d(&vec![tag(3); p.conv_bytes], p.conv_checkpoint(l, s))
                .unwrap();
            for t in 0..p.h_inter_count(s) {
                gpu.copy_h2d(
                    &vec![tag(4 + t); p.h_stored_bytes],
                    p.h_intermediate(l, s, t),
                )
                .unwrap();
            }
            for t in 0..p.num_intermediates {
                gpu.copy_h2d(
                    &vec![tag(9 + t); p.conv_bytes],
                    p.conv_intermediate(l, s, t),
                )
                .unwrap();
            }
        }
    }
}

/// The LIVE state of every slot of every layer — the only thing a rollback
/// is allowed to change, and the thing the two executors must agree on.
fn live_state(gpu: &MockGpuBackend, p: &SsmStatePool) -> Vec<u8> {
    let mut out = Vec::new();
    for l in 0..p.num_ssm_layers {
        for s in 0..=p.mtp_slots {
            let mut h = vec![0u8; p.h_stored_bytes];
            gpu.copy_d2h(p.h_state(l, s), &mut h).unwrap();
            out.extend_from_slice(&h);
            let mut c = vec![0u8; p.conv_bytes];
            gpu.copy_d2h(p.conv_state(l, s), &mut c).unwrap();
            out.extend_from_slice(&c);
        }
    }
    out
}

/// The h/conv plans `commit_accepted_prefix_dispatch` builds for one
/// sequence: `h_intermediate[na-1] → h_state`, same for conv, one row per
/// SSM layer in ascending layer order.
fn commit_plans(
    p: &SsmStatePool,
    slot: usize,
    num_accepted: usize,
) -> (Vec<StateCopy>, Vec<StateCopy>) {
    let cfg = tiny_config();
    let conv_bytes = (cfg.linear_num_key_heads * cfg.linear_key_head_dim * 2
        + cfg.linear_num_value_heads * cfg.linear_value_head_dim)
        * cfg.linear_conv_kernel_dim
        * 4;
    let mut h = Vec::new();
    let mut c = Vec::new();
    let mut ssm_layer_idx = 0usize;
    for i in 0..cfg.num_hidden_layers {
        if cfg.layer_type(i) != LayerType::LinearAttention {
            continue;
        }
        let idx = num_accepted - 1;
        h.push(StateCopy {
            src: p.h_intermediate(ssm_layer_idx, slot, idx),
            dst: p.h_state(ssm_layer_idx, slot),
            bytes: p.h_stored_bytes,
        });
        c.push(StateCopy {
            src: p.conv_intermediate(ssm_layer_idx, slot, idx),
            dst: p.conv_state(ssm_layer_idx, slot),
            bytes: conv_bytes,
        });
        ssm_layer_idx += 1;
    }
    (h, c)
}

/// The full-reject arm of `rollback_ssm_states_dispatch`: restore the
/// pre-verify checkpoint (`num_accepted == 0`).
fn reject_plans(p: &SsmStatePool, slot: usize) -> (Vec<StateCopy>, Vec<StateCopy>) {
    let (mut h, mut c) = commit_plans(p, slot, 1);
    for (l, row) in h.iter_mut().enumerate() {
        row.src = p.h_checkpoint(l, slot);
    }
    for (l, row) in c.iter_mut().enumerate() {
        row.src = p.conv_checkpoint(l, slot);
    }
    (h, c)
}

/// Run one plan pair both ways on two identically-built devices and assert
/// the live state is byte-identical.
fn assert_equivalent(make_plans: impl Fn(&SsmStatePool) -> (Vec<StateCopy>, Vec<StateCopy>)) {
    let looped = MockGpuBackend::new();
    let lp = pool(&looped);
    seed(&looped, &lp);
    let (lh, lc) = make_plans(&lp);
    run_state_copies_with(&looped, &lh, false, 0).unwrap();
    run_state_copies_with(&looped, &lc, false, 0).unwrap();

    let batched = MockGpuBackend::new();
    let bp = pool(&batched);
    seed(&batched, &bp);
    let (bh, bc) = make_plans(&bp);
    run_state_copies_with(&batched, &bh, true, 0).unwrap();
    run_state_copies_with(&batched, &bc, true, 0).unwrap();
    // Both plans must actually have COLLAPSED, or this whole comparison is
    // vacuously "the loop equals the loop".
    assert_eq!(
        (batched.d2d_count(), batched.d2d_2d_count()),
        (0, 2),
        "batched run fell back to the per-layer loop"
    );

    // The two devices are built by the same deterministic bump allocator, so
    // the plans are address-identical too — a drift there would mean the
    // comparison below is comparing different geometries.
    assert_eq!(lh, bh, "h plan differs between runs");
    assert_eq!(lc, bc, "conv plan differs between runs");

    let want = live_state(&looped, &lp);
    let got = live_state(&batched, &bp);
    assert_eq!(want.len(), got.len());
    assert!(
        want == got,
        "batched rollback wrote different bytes than the per-layer loop"
    );
    // And the rollback must actually have DONE something — two no-op
    // executors would agree byte-for-byte for the wrong reason.
    let untouched = MockGpuBackend::new();
    let up = pool(&untouched);
    seed(&untouched, &up);
    assert!(
        want != live_state(&untouched, &up),
        "live state unchanged — the plan moved nothing"
    );
}

#[test]
fn commit_is_byte_identical_at_every_accepted_depth() {
    let probe = MockGpuBackend::new();
    let p = pool(&probe);
    let k = p.h_inter_count(0);
    assert!(k >= 2, "fixture must offer more than one rollback depth");
    // num_accepted == 1 is the shallowest commit (index 0); k is the deepest
    // representable (index k-1). Both ends plus the interior are covered.
    for na in 1..=k {
        assert_equivalent(|pool| commit_plans(pool, 0, na));
    }
}

#[test]
fn full_reject_restore_is_byte_identical() {
    // `num_accepted == 0`: the checkpoint family is the source, which is a
    // DIFFERENT pool with a different layer stride from the intermediates.
    assert_equivalent(|p| reject_plans(p, 0));
}

#[test]
fn every_slot_rolls_back_to_its_own_state() {
    // Slot addressing is where a batched form silently poisons a neighbour:
    // the h intermediates are TIERED, so slot s's rows are not at `s * k`.
    let probe = MockGpuBackend::new();
    let p = pool(&probe);
    for s in 0..MAX_SLOTS {
        let k = p.h_inter_count(s);
        assert!(k >= 1, "slot {s} has no intermediates");
        assert_equivalent(|pool| commit_plans(pool, s, k));
    }
}

#[test]
fn single_ssm_layer_still_copies_the_right_bytes() {
    // A one-row plan cannot collapse (no pitch to derive) and must fall
    // through to the single `copy_d2d_async` — the n=1-layer degenerate case.
    let gpu = MockGpuBackend::new();
    let p = pool(&gpu);
    seed(&gpu, &p);
    let (h, _) = commit_plans(&p, 0, 1);
    let one = [h[0]];
    assert!(copy_plan_as_strided_run(&one).is_none());
    run_state_copies_with(&gpu, &one, true, 0).unwrap();
    assert_eq!(gpu.d2d_count(), 1, "single row = one plain d2d");
    assert_eq!(gpu.d2d_2d_count(), 0);
    let mut got = vec![0u8; p.h_stored_bytes];
    gpu.copy_d2h(p.h_state(0, 0), &mut got).unwrap();
    let mut want = vec![0u8; p.h_stored_bytes];
    gpu.copy_d2h(p.h_intermediate(0, 0, 0), &mut want).unwrap();
    assert_eq!(got, want);
}

#[test]
fn batching_collapses_the_layer_loop_to_one_launch() {
    let gpu = MockGpuBackend::new();
    let p = pool(&gpu);
    seed(&gpu, &p);
    let (h, c) = commit_plans(&p, 0, 1);
    let layers = p.num_ssm_layers;
    assert_eq!(h.len(), layers);

    const STREAM: u64 = 0x5a17;
    run_ssm_state_copies(&gpu, &h, &c, STREAM).unwrap();
    assert_eq!(
        (gpu.d2d_count(), gpu.d2d_2d_count()),
        (0, 2),
        "{layers} h + {layers} conv copies must become 2 pitched launches"
    );
    assert_eq!(gpu.d2d_2d_async_streams(), [STREAM, STREAM]);

    // Kill switch: the same plan, unbatched, is the original loop.
    let off = MockGpuBackend::new();
    let op = pool(&off);
    let (oh, oc) = commit_plans(&op, 0, 1);
    run_state_copies_with(&off, &oh, false, STREAM).unwrap();
    run_state_copies_with(&off, &oc, false, STREAM).unwrap();
    assert_eq!((off.d2d_count(), off.d2d_2d_count()), (2 * layers, 0));
    assert_eq!(off.d2d_async_streams(), vec![STREAM; 2 * layers]);
}

#[test]
fn kill_switch_default_is_batched() {
    // The env var is a PRESENCE check latched in a `OnceLock`; under `cargo
    // test` (no var set) the production reader must say "batched", or the
    // optimization is inert in production too.
    assert!(
        batched_ssm_copy_enabled(),
        "ATLAS_NO_BATCHED_SSM_ROLLBACK is set in this environment"
    );
}

// ── copy_plan_as_strided_run: the collapse predicate itself ──

fn row(src: u64, dst: u64, bytes: usize) -> StateCopy {
    StateCopy {
        src: DevicePtr(src),
        dst: DevicePtr(dst),
        bytes,
    }
}

#[test]
fn a_plan_that_a_2d_copy_cannot_reproduce_is_refused() {
    let base: Vec<StateCopy> = (0..4)
        .map(|l| row(1000 + l * 64, 9000 + l * 64, 16))
        .collect();
    assert!(copy_plan_as_strided_run(&base).is_some(), "control");

    // Ragged width: no single `width_bytes`.
    let mut ragged = base.clone();
    ragged[2].bytes = 8;
    assert!(copy_plan_as_strided_run(&ragged).is_none());

    // A relocated layer (the fragmented-pool fallback) breaks the src stride.
    let mut hole = base.clone();
    hole[2].src = DevicePtr(500_000);
    assert!(copy_plan_as_strided_run(&hole).is_none());

    // ...and the dst stride, independently.
    let mut dhole = base.clone();
    dhole[3].dst = DevicePtr(500_000);
    assert!(copy_plan_as_strided_run(&dhole).is_none());

    // Descending family: a negative pitch has no 2-D representation.
    let down: Vec<StateCopy> = (0..4)
        .map(|l| row(1000 - l * 64, 9000 - l * 64, 16))
        .collect();
    assert!(copy_plan_as_strided_run(&down).is_none());

    // Pitch below width: rows would overlap, so the 2-D form is NOT the plan.
    let tight: Vec<StateCopy> = (0..4)
        .map(|l| row(1000 + l * 8, 9000 + l * 8, 16))
        .collect();
    assert!(copy_plan_as_strided_run(&tight).is_none());

    // Zero-width and empty plans are not 2-D shapes either.
    assert!(copy_plan_as_strided_run(&[]).is_none());
    assert!(copy_plan_as_strided_run(&[row(1, 2, 0), row(3, 4, 0)]).is_none());
}

#[test]
fn the_collapsed_run_reproduces_the_plan_row_for_row() {
    // The equivalence argument in one assertion: expanding the 2-D run by its
    // own definition (`row r = width bytes at base + r*pitch`) must give back
    // exactly the plan that produced it.
    let plan: Vec<StateCopy> = (0..7)
        .map(|l| row(4096 + l * 256, 1_000_000 + l * 128, 96))
        .collect();
    let r = copy_plan_as_strided_run(&plan).unwrap();
    let expanded: Vec<StateCopy> = (0..r.height)
        .map(|i| StateCopy {
            src: r.src.offset(i * r.src_pitch),
            dst: r.dst.offset(i * r.dst_pitch),
            bytes: r.width_bytes,
        })
        .collect();
    assert_eq!(expanded, plan);
}
