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

/// Keep-packed GGUF K-quant sizing: Q4_K = 144 B / 256-elem super-block,
/// Q6_K = 210 B. Per-element size must stay 0 (block-based), and the
/// `WeightTensor` footprint must be the real packed size — this is what the
/// keep-packed loader's per-expert view offsets and the pre-flight OOM
/// estimate are built on. Fails without the PackedQ4K/PackedQ6K arms in
/// `WeightTensor::byte_size`.
#[test]
fn packed_k_quant_byte_size_is_block_based() {
    // [n=2, k=512] → 1024 elems → 4 super-blocks.
    let q4 = WeightTensor {
        ptr: DevicePtr::NULL,
        shape: vec![2, 512],
        dtype: WeightDtype::PackedQ4K,
    };
    assert_eq!(q4.num_elements(), 1024);
    assert_eq!(q4.byte_size(), (1024 / 256) * 144);
    assert!(q4.is_packed_q4k());
    assert!(!q4.is_packed_q6k());

    let q6 = WeightTensor {
        ptr: DevicePtr::NULL,
        shape: vec![2, 512],
        dtype: WeightDtype::PackedQ6K,
    };
    assert_eq!(q6.byte_size(), (1024 / 256) * 210);
    assert!(q6.is_packed_q6k());
    assert!(!q6.is_packed_q4k());

    // Per-element size is undefined for packed; must be 0 so no caller
    // silently multiplies numel by it.
    assert_eq!(WeightDtype::PackedQ4K.byte_size(), 0);
    assert_eq!(WeightDtype::PackedQ6K.byte_size(), 0);

    // BF16 sizing of the SAME shape shows the memory win (>3.5x for Q4_K).
    let bf16 = WeightTensor {
        ptr: DevicePtr::NULL,
        shape: vec![2, 512],
        dtype: WeightDtype::BF16,
    };
    assert!(bf16.byte_size() > q4.byte_size() * 3);
    assert!(!bf16.is_packed_q4k() && !bf16.is_packed_q6k());
}
