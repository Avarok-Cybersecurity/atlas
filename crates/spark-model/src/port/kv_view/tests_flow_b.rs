// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: second group of the mock-backend view wiring tests (split from
//! `tests_flow.rs` to keep each file under the 500-line cap): the freed-sequence
//! release, the unmarked prompt, and the zero-cost state while nothing is armed.

use spark_runtime::gpu::DevicePtr;
use spark_runtime::gpu::mock::MockGpuBackend;

use super::tests_flow::{SCORES, batch, ctx, needle_scores, prompt, run_turn, selector, table};
use super::{TurnKind, ViewOutcome};
use crate::layers::qwen3_attention::sparse_seam::{KvSelector, SeamPhase};

#[test]
fn freeing_a_sequence_drops_its_view() {
    *SCORES.lock().unwrap() = needle_scores();
    let gpu = MockGpuBackend::new();
    let sel = selector(1024, None);
    let tbl = table(256, 1000);
    run_turn(&sel, &gpu, 3, &prompt(4096, None), TurnKind::NewUser, &tbl);
    assert_eq!(sel.inner_for_test().live.iter().flatten().count(), 1);
    // Another sequence's table does not match.
    sel.release_sequence(&gpu, 9, &table(256, 50_000), 4096);
    assert_eq!(sel.inner_for_test().live.iter().flatten().count(), 1);
    sel.release_sequence(&gpu, 3, &tbl, 4096);
    assert_eq!(sel.inner_for_test().live.iter().flatten().count(), 0);
    // Discarding an unknown slot is harmless.
    assert_eq!(
        sel.set_kv_view(&gpu, 42, None).unwrap(),
        ViewOutcome::Discarded
    );
}

#[test]
fn an_unmarked_prompt_selects_without_a_conversation() {
    *SCORES.lock().unwrap() = needle_scores();
    let gpu = MockGpuBackend::new();
    let sel = selector(1024, None);
    let out = run_turn(
        &sel,
        &gpu,
        2,
        &prompt(4096, None),
        TurnKind::Unmarked,
        &table(256, 1000),
    );
    assert!(matches!(out, ViewOutcome::Selected { .. }), "{out:?}");
    assert!(sel.inner_for_test().convs.is_empty());
}

#[test]
fn nothing_armed_means_no_capture_work() {
    let gpu = MockGpuBackend::new();
    let sel = selector(1024, None);
    let (hl, f, n) = ([100u32], [0u32], [100u32]);
    let s = batch(1, DevicePtr::NULL, DevicePtr::NULL, &hl, &f, &n, &[], &[]);
    let q = DevicePtr(0x9000);
    sel.on_query(&ctx(&gpu, SeamPhase::Prefill, false), &s, Some(q), q)
        .unwrap();
    assert!(gpu.launches_snapshot().is_empty());
    assert!(
        sel.inner_for_test().dev.is_none(),
        "nothing allocated before a prefill call"
    );
}
