// SPDX-License-Identifier: AGPL-3.0-only

//! DSA aux-state codec — round-trip exactness and every refusal.
//!
//! CPU-only. `MockGpuBackend` keeps real bytes behind its `DevicePtr`s, so a
//! snapshot → mutate → restore here moves the same bytes the CUDA backend would;
//! what it cannot prove is that the *kernels* then read them correctly. That is
//! the GPU phase. What it does prove is the part that would otherwise be found
//! at 2 a.m.: the layout arithmetic and the fail-closed boundary.

use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;

use crate::layers::glm5next_dsa::Glm5NextDsaConfig;
use crate::layers::glm5next_dsa::state::{Glm5NextDsaState, indexer_state_bytes};

/// GLM-5.3-Flash geometry, small `max_context` so the tests stay cheap.
/// `index_head_dim = 128` and `index_kpool = 4` are the shipped values.
fn cfg(max_context: usize) -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: 4096,
        index_heads: 32,
        index_head_dim: 128,
        index_kpool: 4,
        index_topk: 2048,
        always_select_tail: true,
        local_heads: 64,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 256,
        qk_rope_head_dim: 0,
        v_head_dim: 256,
        max_context,
    }
}

/// Fill the three buffers with a position-dependent pattern and set the cursor,
/// standing in for what `indexer_forward` would have written over `len` tokens.
fn ingest(gpu: &MockGpuBackend, st: &mut Glm5NextDsaState, len: usize, salt: u8) {
    let d = st.index_head_dim();
    let keys: Vec<u8> = (0..len * d * 2)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(salt))
        .collect();
    let gates: Vec<u8> = (0..len * d * 2)
        .map(|i| {
            (i as u8)
                .wrapping_mul(17)
                .wrapping_add(salt)
                .wrapping_add(7)
        })
        .collect();
    let valid: Vec<u8> = (0..len).map(|i| ((i % 251) as u8) ^ salt).collect();
    gpu.copy_h2d(&keys, st.k_normed).unwrap();
    gpu.copy_h2d(&gates, st.gate).unwrap();
    gpu.copy_h2d(&valid, st.valid).unwrap();
    st.rewind_to(0).unwrap();
    st.advance(len).unwrap();
}

/// Read back the reachable `[0, len)` rows as `(k_normed, gate, valid)`.
fn readback(gpu: &MockGpuBackend, st: &Glm5NextDsaState) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (len, d) = (st.len(), st.index_head_dim());
    let mut k = vec![0u8; len * d * 2];
    let mut g = vec![0u8; len * d * 2];
    let mut v = vec![0u8; len];
    gpu.copy_d2h(st.k_normed, &mut k).unwrap();
    gpu.copy_d2h(st.gate, &mut g).unwrap();
    gpu.copy_d2h(st.valid, &mut v).unwrap();
    (k, g, v)
}

/// The whole point: a snapshot taken at N tokens, applied to a state that has
/// since been overwritten by an unrelated sequence, reproduces the original
/// bytes exactly — not approximately, and not only the keys.
#[test]
fn snapshot_mutate_restore_is_byte_exact() {
    let gpu = MockGpuBackend::new();
    let c = cfg(4_096);
    let mut st = Glm5NextDsaState::alloc(&gpu, &c).unwrap();

    ingest(&gpu, &mut st, 1_000, 0xA5);
    let before = readback(&gpu, &st);
    let blob = st.snapshot_blob(&gpu, 0).unwrap();

    // Another sequence lands on this state: different content, different length.
    ingest(&gpu, &mut st, 2_048, 0x5A);
    assert_ne!(
        readback(&gpu, &st).0,
        before.0,
        "mutation must actually move"
    );

    st.restore_blob(&blob, &gpu, 0).unwrap();
    assert_eq!(st.len(), 1_000, "cursor restored");
    assert_eq!(
        readback(&gpu, &st),
        before,
        "k_normed / gate / valid all exact"
    );
}

/// Prefix length is a free variable, including the boundaries — an empty
/// snapshot (nothing ingested yet) and one filling the whole reservation.
#[test]
fn round_trip_holds_at_every_prefix_length() {
    let gpu = MockGpuBackend::new();
    let c = cfg(4_096);
    let mut st = Glm5NextDsaState::alloc(&gpu, &c).unwrap();

    for len in [0usize, 1, 4, 255, 256, 1_023, 4_096] {
        ingest(&gpu, &mut st, len, len as u8);
        let want = readback(&gpu, &st);
        let blob = st.snapshot_blob(&gpu, 0).unwrap();
        assert_eq!(
            blob.len(),
            16 + len * (c.index_head_dim * 4 + 1),
            "blob carries len rows, not capacity, at len={len}"
        );

        ingest(&gpu, &mut st, 4_096, 0xFF);
        st.restore_blob(&blob, &gpu, 0).unwrap();
        assert_eq!(st.len(), len);
        assert_eq!(readback(&gpu, &st), want, "exact at len={len}");
    }
}

/// The blob carries what was WRITTEN, not what was RESERVED. A 1K prefix on a
/// serve declaring a huge context must not cost the huge context.
#[test]
fn blob_scales_with_prefix_not_with_reservation() {
    let gpu = MockGpuBackend::new();
    let mut small = Glm5NextDsaState::alloc(&gpu, &cfg(4_096)).unwrap();
    let mut huge = Glm5NextDsaState::alloc(&gpu, &cfg(131_072)).unwrap();
    ingest(&gpu, &mut small, 1_000, 1);
    ingest(&gpu, &mut huge, 1_000, 1);
    assert_eq!(
        small.snapshot_blob(&gpu, 0).unwrap().len(),
        huge.snapshot_blob(&gpu, 0).unwrap().len(),
        "same prefix, same cost, whatever --max-seq-len claims"
    );
    // And the device reservation is the one that tracks the cap.
    assert!(indexer_state_bytes(huge.capacity(), 128) > indexer_state_bytes(small.capacity(), 128));
}

/// 🔴 A truncated blob is an error, never a partial apply. Both the header cut
/// and a body cut are covered — the second is the one a naive `len >= 16` check
/// would wave through.
#[test]
fn a_truncated_blob_is_refused() {
    let gpu = MockGpuBackend::new();
    let mut st = Glm5NextDsaState::alloc(&gpu, &cfg(4_096)).unwrap();
    ingest(&gpu, &mut st, 64, 3);
    let blob = st.snapshot_blob(&gpu, 0).unwrap();

    for cut in [0usize, 8, 15, 16, blob.len() - 1] {
        let err = st
            .restore_blob(&blob[..cut], &gpu, 0)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("truncated") || err.contains("size mismatch"),
            "cut to {cut} must be refused, got: {err}"
        );
    }
    assert_eq!(
        st.len(),
        64,
        "a refused restore leaves the cursor where it was"
    );
}

/// A blob whose header claims a different indexer width is a different model.
/// Caught before any copy, because the size check alone would let a
/// coincidentally-consistent geometry through.
#[test]
fn a_wrong_geometry_blob_is_refused() {
    let gpu = MockGpuBackend::new();
    let mut st = Glm5NextDsaState::alloc(&gpu, &cfg(4_096)).unwrap();
    ingest(&gpu, &mut st, 32, 9);
    let mut blob = st.snapshot_blob(&gpu, 0).unwrap();
    blob[8..16].copy_from_slice(&64u64.to_le_bytes());

    let err = st.restore_blob(&blob, &gpu, 0).unwrap_err().to_string();
    assert!(err.contains("index_head_dim"), "got: {err}");
    assert_eq!(st.len(), 32, "refused restores do not move the cursor");
}

/// A header `len` that disagrees with the body length is corruption, not a
/// shorter snapshot. Both directions.
#[test]
fn a_len_body_disagreement_is_refused() {
    let gpu = MockGpuBackend::new();
    let mut st = Glm5NextDsaState::alloc(&gpu, &cfg(4_096)).unwrap();
    ingest(&gpu, &mut st, 100, 4);
    let blob = st.snapshot_blob(&gpu, 0).unwrap();

    for claimed in [99u64, 101, 0] {
        let mut bad = blob.clone();
        bad[..8].copy_from_slice(&claimed.to_le_bytes());
        let err = st.restore_blob(&bad, &gpu, 0).unwrap_err().to_string();
        assert!(err.contains("size mismatch"), "claimed {claimed}: {err}");
    }
}

/// 🔴 A snapshot longer than this sequence's reservation is REFUSED, not
/// clamped. A clamped restore would select over a prefix while the MLA cache
/// held the full context — a wrong answer under HTTP 200, which is exactly the
/// failure `advance` already refuses to produce.
#[test]
fn a_blob_longer_than_the_reservation_is_refused() {
    let gpu = MockGpuBackend::new();
    let mut big = Glm5NextDsaState::alloc(&gpu, &cfg(8_192)).unwrap();
    ingest(&gpu, &mut big, 8_192, 2);
    let blob = big.snapshot_blob(&gpu, 0).unwrap();

    let mut small = Glm5NextDsaState::alloc(&gpu, &cfg(4_096)).unwrap();
    let err = small.restore_blob(&blob, &gpu, 0).unwrap_err().to_string();
    assert!(err.contains("exceeds"), "got: {err}");
    assert_eq!(small.len(), 0, "nothing was applied");
}

/// Rewind is exact where the code claims it: the rows in `[n, len)` stay put,
/// so re-advancing over them without rewriting reproduces the original bytes.
/// This is the property `decode_k`'s rejected-draft path depends on, and it is
/// why the codec only ever carries `[0, len)`.
#[test]
fn rewind_is_exact_within_a_live_sequence() {
    let gpu = MockGpuBackend::new();
    let mut st = Glm5NextDsaState::alloc(&gpu, &cfg(4_096)).unwrap();
    ingest(&gpu, &mut st, 500, 11);
    let full = readback(&gpu, &st);

    st.rewind_to(300).unwrap();
    let at_300 = st.snapshot_blob(&gpu, 0).unwrap();
    assert_eq!(at_300.len(), 16 + 300 * (128 * 4 + 1));

    st.advance(200).unwrap();
    assert_eq!(
        readback(&gpu, &st),
        full,
        "rewind moved the cursor, not the rows"
    );

    // And the 300-token blob is a strict prefix of the 500-token one.
    let at_500 = st.snapshot_blob(&gpu, 0).unwrap();
    assert_eq!(
        &at_300[16..16 + 300 * 128 * 2],
        &at_500[16..16 + 300 * 128 * 2]
    );
}

/// Forward "rewind" stays refused — the codec must not have opened a way to
/// claim a length the sequence never reached.
#[test]
fn the_cursor_cannot_be_moved_forward_by_a_restore() {
    let gpu = MockGpuBackend::new();
    let mut st = Glm5NextDsaState::alloc(&gpu, &cfg(4_096)).unwrap();
    ingest(&gpu, &mut st, 10, 1);
    assert!(st.rewind_to(11).is_err(), "forward rewind is still a bug");
}
