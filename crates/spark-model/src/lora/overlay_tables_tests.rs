// SPDX-License-Identifier: AGPL-3.0-only

//! Host tests for the `[max_loras]` overlay device pointer tables. Uses
//! `MockGpuBackend` (records `copy_h2d`) to read back the packed tables and
//! assert the per-slot pointer/count layout and the tied/untied lm_head wiring.

use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::*;
use crate::lora::overlay_build::{EmbedOverlay, LmHeadOverlay};

fn u64s(gpu: &MockGpuBackend, p: DevicePtr, n: usize) -> Vec<u64> {
    let mut b = vec![0u8; n * 8];
    gpu.copy_d2h(p, &mut b).unwrap();
    b.chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn u32s(gpu: &MockGpuBackend, p: DevicePtr, n: usize) -> Vec<u32> {
    let mut b = vec![0u8; n * 4];
    gpu.copy_d2h(p, &mut b).unwrap();
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn embed(rows: u64, ids: u64, sm: u64, n: u32, lm: Option<LmHeadOverlay>) -> EmbedOverlay {
    EmbedOverlay {
        rows: DevicePtr(rows),
        ids_dev: DevicePtr(ids),
        slot_map: DevicePtr(sm),
        n_override: n,
        vocab: 640,
        lmhead: lm,
    }
}

#[test]
fn tied_reuses_embed_rows_and_ids() {
    let gpu = MockGpuBackend::new();
    let ov = embed(0xA0, 0xB0, 0xC0, 3, None);
    let set = TokenOverlaySet::from_slots(&gpu, vec![Some(ov), None], 2, true).unwrap();
    assert_eq!(u64s(&gpu, set.embed_rows_table, 2), vec![0xA0, 0]);
    assert_eq!(u64s(&gpu, set.embed_slot_map_table, 2), vec![0xC0, 0]);
    // Tied ⇒ lm_head reuses the embed rows/ids and the embed n_override.
    assert_eq!(u64s(&gpu, set.lmhead_rows_table, 2), vec![0xA0, 0]);
    assert_eq!(u64s(&gpu, set.lmhead_ids_table, 2), vec![0xB0, 0]);
    assert_eq!(u32s(&gpu, set.n_override_table, 2), vec![3, 0]);
    assert_eq!(set.max_n_override, 3);
    assert!(set.any_active());
}

#[test]
fn untied_without_lmhead_zeroes_lmhead_count() {
    let gpu = MockGpuBackend::new();
    let ov = embed(0xA0, 0xB0, 0xC0, 3, None);
    let set = TokenOverlaySet::from_slots(&gpu, vec![Some(ov)], 1, false).unwrap();
    // Embed still overrides; lm_head is skipped (embed-only correction).
    assert_eq!(u64s(&gpu, set.embed_rows_table, 1), vec![0xA0]);
    assert_eq!(u32s(&gpu, set.n_override_table, 1), vec![0]);
    assert_eq!(u64s(&gpu, set.lmhead_rows_table, 1), vec![0]);
    assert_eq!(set.max_n_override, 0);
}

#[test]
fn untied_with_lmhead_uses_distinct_rows() {
    let gpu = MockGpuBackend::new();
    let lm = LmHeadOverlay {
        rows: DevicePtr(0xD0),
        ids_dev: DevicePtr(0xE0),
        n_override: 2,
    };
    let ov = embed(0xA0, 0xB0, 0xC0, 3, Some(lm));
    let set = TokenOverlaySet::from_slots(&gpu, vec![Some(ov)], 1, false).unwrap();
    assert_eq!(u64s(&gpu, set.lmhead_rows_table, 1), vec![0xD0]);
    assert_eq!(u64s(&gpu, set.lmhead_ids_table, 1), vec![0xE0]);
    assert_eq!(u32s(&gpu, set.n_override_table, 1), vec![2]);
    assert_eq!(set.max_n_override, 2);
    // The embed bound table must carry the EMBED count (3), never the
    // lm_head count (2) — they diverge exactly on this untied shape, so this
    // is the assert that catches a table mix-up.
    assert_eq!(u32s(&gpu, set.embed_n_table, 1), vec![3]);
}

// CWE-125 guard plumbing: the embed kernel's two bounds come from HERE, so pin
// them. Positive: a resident slot publishes its embed n_override and the
// build-time vocab. Negative: an empty slot stays 0 (kernel `slot_map == NULL`
// / `slot >= 0` skip), and an empty pool records vocab 0.
#[test]
fn embed_bound_table_and_vocab_are_published() {
    let gpu = MockGpuBackend::new();
    let ov = embed(0xA0, 0xB0, 0xC0, 3, None);
    let set = TokenOverlaySet::from_slots(&gpu, vec![Some(ov), None], 2, true).unwrap();
    assert_eq!(u32s(&gpu, set.embed_n_table, 2), vec![3, 0]);
    assert_eq!(set.vocab, 640);

    let empty = TokenOverlaySet::from_slots(&gpu, vec![None, None], 2, true).unwrap();
    assert_eq!(empty.vocab, 0);
    assert_eq!(u32s(&gpu, empty.embed_n_table, 2), vec![0, 0]);
}

// Two resident slots built against DIFFERENT vocabs cannot share the kernel's
// single `ids[r] < vocab` bound — from_slots must refuse, not pick one.
#[test]
fn mixed_vocab_slots_are_refused() {
    let gpu = MockGpuBackend::new();
    let a = embed(0xA0, 0xB0, 0xC0, 3, None);
    let mut b = embed(0xA1, 0xB1, 0xC1, 1, None);
    b.vocab = 768;
    let err = match TokenOverlaySet::from_slots(&gpu, vec![Some(a), Some(b)], 2, true) {
        Err(e) => e,
        Ok(_) => panic!("mixed-vocab overlay slots must be refused"),
    };
    assert!(err.to_string().contains("mixed-vocab"), "{err}");
}

#[test]
fn empty_pool_is_inert() {
    let gpu = MockGpuBackend::new();
    let set = TokenOverlaySet::from_slots(&gpu, vec![None, None], 2, true).unwrap();
    assert!(!set.any_active());
    assert_eq!(set.max_n_override, 0);
    assert_eq!(u64s(&gpu, set.embed_rows_table, 2), vec![0, 0]);
}
