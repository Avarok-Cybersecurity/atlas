// SPDX-License-Identifier: AGPL-3.0-only

//! Keep-packed ternary Q2_0 sizing tests, split out of `weights.rs`
//! (≤500 LoC cap).

use super::*;
use crate::gpu::DevicePtr;

/// A packed Q2_0 tensor's on-GPU footprint is block-based, not per-element:
/// `n_blocks * (2 + group/4)` bytes. Locks the group-128 (34 B) and
/// group-64 (18 B) sizing so `WeightStore::total_bytes` reflects the real
/// ~2.1 bpw resident, not a bogus per-element multiply.
#[test]
fn packed_q2_byte_size_is_block_based() {
    // [n=2, k=256] @ group 128 → 2 rows × 2 blocks × 34 B = 136 B.
    let t = WeightTensor {
        ptr: DevicePtr::NULL,
        shape: vec![2, 256],
        dtype: WeightDtype::PackedQ2_0 { group: 128 },
    };
    assert_eq!(t.num_elements(), 512);
    assert_eq!(t.byte_size(), (512 / 128) * (2 + 128 / 4));
    assert_eq!(t.byte_size(), 136);
    assert_eq!(t.q2_group(), Some(128));
    assert!(t.is_packed_q2());

    // group-64 → 18 B blocks: [n=1, k=128] → 2 blocks × 18 = 36 B.
    let t64 = WeightTensor {
        ptr: DevicePtr::NULL,
        shape: vec![1, 128],
        dtype: WeightDtype::PackedQ2_0 { group: 64 },
    };
    assert_eq!(t64.byte_size(), (128 / 64) * (2 + 64 / 4));
    assert_eq!(t64.byte_size(), 36);

    // Per-element size is undefined for packed; must be 0 so no caller
    // silently multiplies numel by it.
    assert_eq!(WeightDtype::PackedQ2_0 { group: 128 }.byte_size(), 0);

    // BF16 sizing of the SAME shape is 4× larger — the memory win.
    let bf16 = WeightTensor {
        ptr: DevicePtr::NULL,
        shape: vec![2, 256],
        dtype: WeightDtype::BF16,
    };
    assert!(bf16.byte_size() > t.byte_size() * 3);
    assert_eq!(bf16.q2_group(), None);
    assert!(!bf16.is_packed_q2());
}
