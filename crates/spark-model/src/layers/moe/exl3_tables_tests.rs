// SPDX-License-Identifier: AGPL-3.0-only

//! Mock-GPU tests for the native EXL3 expert tables
//! (`ATLAS_EXL3_NATIVE_MOE=1`): dense-local ptr-table build from
//! `Vec<Option<Exl3Weight>>` (None = remote under EP), the `-1` slot-index
//! mapping, and build determinism. Split from `ptr_table_build.rs` for the
//! 500-LoC cap.

use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::exl3::{Exl3Codebook, Exl3Weight};

use super::ptr_table_build::build_exl3_ptr_table;
use super::tables::exl3_expert_slot_index;

fn w(e: usize, k_bits: u32, cb: Exl3Codebook) -> Exl3Weight {
    // Distinct, recognizable pointer values per expert so the uploaded
    // tables can be asserted byte-for-byte.
    Exl3Weight {
        trellis: DevicePtr(0x1000 + e as u64),
        suh: DevicePtr(0x2000 + e as u64),
        svh: DevicePtr(0x3000 + e as u64),
        in_dim: 2560,
        out_dim: 640,
        k_bits,
        cb,
    }
}

fn read_u64s(gpu: &MockGpuBackend, ptr: DevicePtr, n: usize) -> Vec<u64> {
    let mut bytes = vec![0u8; n * 8];
    gpu.copy_d2h(ptr, &mut bytes).unwrap();
    bytes
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn dense_local_table_from_ep_rank1_shaped_vec() {
    let gpu = MockGpuBackend::new();
    // 6 experts, local range [2, 5) — leading and trailing remotes are None.
    let experts: Vec<Option<Exl3Weight>> = (0..6)
        .map(|e| (2..5).contains(&e).then(|| w(e, 2, Exl3Codebook::Mul1)))
        .collect();
    let t = build_exl3_ptr_table(&experts, &gpu).unwrap();
    assert_eq!(t.num_local, 3);
    assert_eq!(t.local_start, 2);
    assert_eq!(t.k_bits, 2);
    assert_eq!(t.cb, 2, "MUL1 -> kernel codebook index 2");
    assert_eq!((t.in_dim, t.out_dim), (2560, 640));
    // DENSE local tables: exactly the local run, in order, no null entries.
    assert_eq!(
        read_u64s(&gpu, t.trellis_ptrs, 3),
        vec![0x1002, 0x1003, 0x1004]
    );
    assert_eq!(read_u64s(&gpu, t.suh_ptrs, 3), vec![0x2002, 0x2003, 0x2004]);
    assert_eq!(read_u64s(&gpu, t.svh_ptrs, 3), vec![0x3002, 0x3003, 0x3004]);

    // Deterministic: a rebuild from the same vec produces identical tables.
    let t2 = build_exl3_ptr_table(&experts, &gpu).unwrap();
    assert_eq!(
        read_u64s(&gpu, t.trellis_ptrs, 3),
        read_u64s(&gpu, t2.trellis_ptrs, 3)
    );
    assert_eq!((t2.num_local, t2.local_start), (3, 2));

    t.release(&gpu).unwrap();
    t2.release(&gpu).unwrap();
}

#[test]
fn mcg_maps_to_kernel_index_1() {
    let gpu = MockGpuBackend::new();
    let experts = vec![Some(w(0, 3, Exl3Codebook::Mcg))];
    let t = build_exl3_ptr_table(&experts, &gpu).unwrap();
    assert_eq!((t.cb, t.k_bits, t.local_start), (1, 3, 0));
    t.release(&gpu).unwrap();
}

#[test]
fn rejects_gap_in_local_run() {
    let gpu = MockGpuBackend::new();
    // Some, None, Some — a hole inside the "local" range would make dense
    // indexing address the wrong expert.
    let experts = vec![
        Some(w(0, 2, Exl3Codebook::Mul1)),
        None,
        Some(w(2, 2, Exl3Codebook::Mul1)),
    ];
    let err = build_exl3_ptr_table(&experts, &gpu).unwrap_err();
    assert!(err.to_string().contains("contiguous"), "{err:#}");
}

#[test]
fn rejects_all_remote_and_mixed_template() {
    let gpu = MockGpuBackend::new();
    let none: Vec<Option<Exl3Weight>> = vec![None, None];
    assert!(build_exl3_ptr_table(&none, &gpu).is_err());

    let mixed_k = vec![
        Some(w(0, 2, Exl3Codebook::Mul1)),
        Some(w(1, 4, Exl3Codebook::Mul1)),
    ];
    let err = build_exl3_ptr_table(&mixed_k, &gpu).unwrap_err();
    assert!(err.to_string().contains("ONE template"), "{err:#}");

    let mixed_cb = vec![
        Some(w(0, 2, Exl3Codebook::Mul1)),
        Some(w(1, 2, Exl3Codebook::Mcg)),
    ];
    assert!(build_exl3_ptr_table(&mixed_cb, &gpu).is_err());

    // cb0 has no compiled kernel instances.
    let inst3 = vec![Some(w(0, 2, Exl3Codebook::Inst3))];
    assert!(build_exl3_ptr_table(&inst3, &gpu).is_err());
}

#[test]
fn slot_index_maps_local_dense_and_remote_negative() {
    // EP rank with local range [256, 512): global 256 -> slot 0, global 511
    // -> slot 255, everything outside -> -1 (the mgemm reduction's skip
    // value; NULL table entries are NOT a substitute — they sum stale
    // scratch).
    assert_eq!(exl3_expert_slot_index(256, 256, 256), 0);
    assert_eq!(exl3_expert_slot_index(511, 256, 256), 255);
    assert_eq!(exl3_expert_slot_index(0, 256, 256), -1);
    assert_eq!(exl3_expert_slot_index(255, 256, 256), -1);
    assert_eq!(exl3_expert_slot_index(512, 256, 256), -1);
    // Single-node (no EP): identity over the full range.
    assert_eq!(exl3_expert_slot_index(0, 0, 512), 0);
    assert_eq!(exl3_expert_slot_index(511, 0, 512), 511);
}

/// The prefill tier's sizing constants agree: the default fused row cap
/// equals the overflow chunk (an expert that still overflows fills at least
/// one whole chunk) and fits inside the default token batch (so the default
/// is never clamped at the canonical geometry).
#[test]
fn prefill_row_cap_default_matches_overflow_chunk_and_fits_the_batch() {
    use super::tables::{EXL3_MOE_OVERFLOW_CHUNK_ROWS, EXL3_MOE_PREFILL_BATCH_TOKENS_DEFAULT};
    use crate::layers::ops::{
        EXL3_MOE_ROWS_PER_EXPERT_DEFAULT, EXL3_MOE_ROWS_PER_EXPERT_LEGACY, Exl3MoeRowCapGeometry,
        resolve_exl3_moe_row_cap,
    };
    assert_eq!(
        EXL3_MOE_ROWS_PER_EXPERT_DEFAULT,
        EXL3_MOE_OVERFLOW_CHUNK_ROWS
    );
    const _: () =
        assert!(EXL3_MOE_ROWS_PER_EXPERT_DEFAULT <= EXL3_MOE_PREFILL_BATCH_TOKENS_DEFAULT);
    const _: () = assert!(EXL3_MOE_ROWS_PER_EXPERT_LEGACY < EXL3_MOE_ROWS_PER_EXPERT_DEFAULT);
    // qwen4_exp on one GB10 (48 SMs -> C = 6): the default resolves unclamped.
    let r = resolve_exl3_moe_row_cap(
        false,
        None,
        Exl3MoeRowCapGeometry {
            t_cap: EXL3_MOE_PREFILL_BATCH_TOKENS_DEFAULT,
            hidden: 2560,
            inter: 640,
            concurrency: 6,
        },
    );
    assert_eq!(r.rows, EXL3_MOE_ROWS_PER_EXPERT_DEFAULT);
    assert!(r.warnings.is_empty(), "{:?}", r.warnings);
}
