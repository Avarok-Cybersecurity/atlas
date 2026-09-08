// SPDX-License-Identifier: AGPL-3.0-only

//! The aux codec's failure mode is a WRONG restore, not a crash — rows laid over the wrong
//! layer, the wrong width, or a stale counter select over the wrong context and produce a
//! plausible answer. So every refusal arm is pinned here, and the round-trip is checked byte
//! for byte against the mock backend (which really moves bytes, offsets included).

use spark_runtime::gpu::GpuBackend;
use spark_runtime::gpu::mock::MockGpuBackend;

use super::*;
use crate::layers::glm5next_dsa::Glm5NextDsaConfig;

const D: usize = 128;

fn cfg(max_context: usize) -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: 4096,
        index_heads: 32,
        index_head_dim: D,
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

/// A state whose first `len` rows hold a recognisable per-row pattern, so a misaligned or
/// permuted restore shows up as a byte difference rather than passing on zeros.
fn filled_state(
    gpu: &MockGpuBackend,
    capacity: usize,
    len: usize,
    seed: u8,
) -> Glm5NextDsaState {
    let mut st = Glm5NextDsaState::alloc(gpu, &cfg(capacity)).unwrap();
    let rows: Vec<u8> = (0..len * D * 2)
        .map(|i| (i as u8).wrapping_add(seed))
        .collect();
    gpu.copy_h2d(&rows, st.k_normed).unwrap();
    let gate: Vec<u8> = rows.iter().map(|b| b.wrapping_mul(3)).collect();
    gpu.copy_h2d(&gate, st.gate).unwrap();
    let valid: Vec<u8> = (0..len).map(|i| (i % 2) as u8).collect();
    gpu.copy_h2d(&valid, st.valid).unwrap();
    st.advance(len).unwrap();
    st
}

fn read(gpu: &MockGpuBackend, st: &Glm5NextDsaState) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut k = vec![0u8; st.len() * D * 2];
    let mut g = vec![0u8; st.len() * D * 2];
    let mut v = vec![0u8; st.len()];
    gpu.copy_d2h(st.k_normed, &mut k).unwrap();
    gpu.copy_d2h(st.gate, &mut g).unwrap();
    gpu.copy_d2h(st.valid, &mut v).unwrap();
    (k, g, v)
}

#[test]
fn header_round_trips_and_sizes_the_blob() {
    let h = AuxHeader {
        len: 4_097,
        index_head_dim: D,
        layer_idx: 7,
    };
    let bytes = h.encode();
    assert_eq!(bytes.len(), AUX_HEADER_BYTES);
    assert_eq!(&bytes[..4], b"GDSA", "magic reads as GDSA in a hex dump");
    assert_eq!(AuxHeader::decode(&bytes).unwrap(), h);
    // 513 B/row at D=128 is the figure `indexer_state_bytes` and the memory ledger quote.
    assert_eq!(h.blob_bytes(), AUX_HEADER_BYTES + 4_097 * 513);
}

#[test]
fn every_header_refusal_arm_fires() {
    let h = AuxHeader {
        len: 8,
        index_head_dim: D,
        layer_idx: 3,
    };
    let good = h.encode();

    let err = AuxHeader::decode(&good[..23]).unwrap_err().to_string();
    assert!(err.contains("truncated"), "{err}");

    let mut magic = good;
    magic[0] ^= 0xff;
    let err = AuxHeader::decode(&magic).unwrap_err().to_string();
    assert!(err.contains("magic"), "{err}");

    let mut ver = good;
    ver[4..8].copy_from_slice(&(AUX_VERSION + 1).to_le_bytes());
    let err = AuxHeader::decode(&ver).unwrap_err().to_string();
    assert!(err.contains("version"), "{err}");

    let ok_len = h.blob_bytes();
    assert!(h.validate_for(3, D, 8, ok_len).is_ok());
    // Wrong layer: the silent-permutation trap.
    let err = h.validate_for(4, D, 8, ok_len).unwrap_err().to_string();
    assert!(err.contains("layer 3"), "{err}");
    // Wrong row width.
    let err = h.validate_for(3, 64, 8, ok_len).unwrap_err().to_string();
    assert!(err.contains("row width"), "{err}");
    // A blob from a larger --max-seq-len than this sequence reserved.
    let err = h.validate_for(3, D, 7, ok_len).unwrap_err().to_string();
    assert!(err.contains("reserved 7"), "{err}");
    // Exact length, both directions.
    assert!(h.validate_for(3, D, 8, ok_len - 1).is_err());
    assert!(h.validate_for(3, D, 8, ok_len + 1).is_err());
}

#[test]
fn snapshot_restore_round_trips_byte_for_byte_into_a_fresh_state() {
    let gpu = MockGpuBackend::new();
    let src = filled_state(&gpu, 64, 13, 0x11);
    let blob = snapshot_with_cap(&src, 5, 32, &gpu, 0)
        .unwrap()
        .expect("13 <= cap 32");
    assert_eq!(blob.len(), AUX_HEADER_BYTES + 13 * 513);

    let mut dst = Glm5NextDsaState::alloc(&gpu, &cfg(64)).unwrap();
    assert_eq!(dst.len(), 0);
    restore(&mut dst, 5, &blob, &gpu, 0).unwrap();
    assert_eq!(
        dst.len(),
        13,
        "the counter is the blob's, not the fresh state's 0"
    );
    assert_eq!(read(&gpu, &src), read(&gpu, &dst));
}

/// A reused slot is the common case on a warm hit: the sequence that held this state before
/// left rows behind. Overwriting is sound (rows past `len` are unreachable) and must not be
/// refused — refusing turns a cache hit into a request error.
#[test]
fn restore_over_a_live_state_overwrites_and_takes_the_blobs_len() {
    let gpu = MockGpuBackend::new();
    let src = filled_state(&gpu, 64, 5, 0x22);
    let blob = snapshot_with_cap(&src, 9, 32, &gpu, 0).unwrap().unwrap();

    let mut live = filled_state(&gpu, 64, 20, 0x77);
    restore(&mut live, 9, &blob, &gpu, 0).unwrap();
    assert_eq!(live.len(), 5);
    assert_eq!(read(&gpu, &src), read(&gpu, &live));
}

/// An empty image is a COMPLETE statement, not an absence: the restore gate counts one blob
/// per aux-carrying layer, so `len == 0` must still produce `Some`.
#[test]
fn an_empty_state_snapshots_to_some_and_restores_to_zero() {
    let gpu = MockGpuBackend::new();
    let src = Glm5NextDsaState::alloc(&gpu, &cfg(64)).unwrap();
    let blob = snapshot_with_cap(&src, 0, 32, &gpu, 0)
        .unwrap()
        .expect("empty is Some");
    assert_eq!(blob.len(), AUX_HEADER_BYTES);
    let mut dst = filled_state(&gpu, 64, 9, 0x33);
    restore(&mut dst, 0, &blob, &gpu, 0).unwrap();
    assert_eq!(dst.len(), 0);
}

/// Above the cap the layer declines rather than truncates. A truncated image would restore a
/// prefix while the KV blocks hold the full context — exactly the "select over the wrong
/// context" failure this whole codec exists to prevent.
#[test]
fn above_the_cap_the_snapshot_is_none_never_truncated() {
    let gpu = MockGpuBackend::new();
    let src = filled_state(&gpu, 64, 33, 0x44);
    assert!(snapshot_with_cap(&src, 1, 32, &gpu, 0).unwrap().is_none());
    assert!(
        snapshot_with_cap(&src, 1, 33, &gpu, 0).unwrap().is_some(),
        "cap is inclusive"
    );
    // Cap 0: only an EMPTY state snapshots — the negative-control setting.
    assert!(snapshot_with_cap(&src, 1, 0, &gpu, 0).unwrap().is_none());
    let empty = Glm5NextDsaState::alloc(&gpu, &cfg(64)).unwrap();
    assert!(snapshot_with_cap(&empty, 1, 0, &gpu, 0).unwrap().is_some());
}

/// A refused blob must leave the receiving state EXACTLY as it was — counter and rows — so
/// a request that fails validation cannot resume half-restored on a retry.
#[test]
fn a_refused_restore_touches_nothing() {
    let gpu = MockGpuBackend::new();
    let src = filled_state(&gpu, 64, 6, 0x55);
    let blob = snapshot_with_cap(&src, 2, 32, &gpu, 0).unwrap().unwrap();
    let mut dst = filled_state(&gpu, 64, 3, 0x66);
    let before = read(&gpu, &dst);

    // Wrong layer.
    assert!(restore(&mut dst, 3, &blob, &gpu, 0).is_err());
    // Off-by-one body.
    assert!(restore(&mut dst, 2, &blob[..blob.len() - 1], &gpu, 0).is_err());
    assert_eq!(dst.len(), 3);
    assert_eq!(read(&gpu, &dst), before);

    // A 6-row blob into a 4-row reservation (a smaller --max-seq-len than the save).
    let mut small = Glm5NextDsaState::alloc(&gpu, &cfg(4)).unwrap();
    let err = restore(&mut small, 2, &blob, &gpu, 0)
        .unwrap_err()
        .to_string();
    assert!(err.contains("reserved 4"), "{err}");
    assert_eq!(small.len(), 0);
}

#[test]
fn a_released_state_refuses_a_restore() {
    let gpu = MockGpuBackend::new();
    let src = Glm5NextDsaState::alloc(&gpu, &cfg(64)).unwrap();
    let blob = snapshot_with_cap(&src, 0, 32, &gpu, 0).unwrap().unwrap();
    let mut dead = Glm5NextDsaState::alloc(&gpu, &cfg(64)).unwrap();
    dead.free(&gpu).unwrap();
    let err = restore(&mut dead, 0, &blob, &gpu, 0)
        .unwrap_err()
        .to_string();
    assert!(err.contains("released"), "{err}");
}

/// `set_len_restored` is a restore primitive, not a general setter: it stays inside the
/// reservation, and moving BACK is still `rewind_to`'s job (the lockstep contract `decode_k`
/// relies on when a restored counter is ahead of the sequence).
#[test]
fn set_len_restored_only_moves_within_the_reservation() {
    let gpu = MockGpuBackend::new();
    let mut st = filled_state(&gpu, 16, 10, 0x01);
    st.set_len_restored(16).unwrap();
    assert_eq!(st.len(), 16);
    assert!(st.set_len_restored(17).is_err(), "past the reservation");
    st.rewind_to(12).unwrap();
    assert_eq!(st.len(), 12);
}
