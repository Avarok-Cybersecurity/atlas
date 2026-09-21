// SPDX-License-Identifier: AGPL-3.0-only

//! Host tests for the token-overlay Stage-2 build. Uses `MockGpuBackend` (no
//! GPU): the row-diff kernel is a recorded no-op there, so `flags` stay zero and
//! the override set reduces to the trainable ids — which is exactly the delta
//! (`trainable_tokens`) path we assert compacts correctly (delta rows f32→bf16,
//! `slot_map` populated, ascending ids).

use spark_runtime::gpu::GpuBackend;
use spark_runtime::gpu::KernelHandle;
use spark_runtime::gpu::mock::MockGpuBackend;

use super::*;
use crate::layers::ops::token_overlay::OverlayKernels;

fn f32_le(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn f32_to_bf16_round_nearest_even() {
    // Exact bf16-representable values round-trip.
    assert_eq!(super::f32_to_bf16(1.0), (1.0f32.to_bits() >> 16) as u16);
    assert_eq!(super::f32_to_bf16(0.0), 0);
    // A value whose low 16 bits force a round-up.
    let x = f32::from_bits(0x3F80_8000 | 0x0000_8001); // mantissa tail set
    let got = super::f32_to_bf16(x);
    let trunc = (x.to_bits() >> 16) as u16;
    assert_eq!(got, trunc + 1, "round-to-nearest bumps the bf16 mantissa");
    // Exact halfway values resolve to an even retained mantissa.
    assert_eq!(super::f32_to_bf16(f32::from_bits(0x3F80_8000)), 0x3F80);
    assert_eq!(super::f32_to_bf16(f32::from_bits(0x3F81_8000)), 0x3F82);
}

#[test]
fn build_overlay_delta_path_compacts_and_maps() {
    let gpu = MockGpuBackend::new();
    let h = 4usize;
    let vocab = 8usize;
    let embed_r = 8u32;
    // Served + adapter base tables (zeroed on the mock; row-diff is a no-op).
    let base = gpu.alloc(embed_r as usize * h * 2).unwrap();
    let served = gpu.alloc(vocab * h * 2).unwrap();
    // One trainable delta row (id 5), f32.
    let delta = gpu.alloc(h * 4).unwrap();
    let dvals = [0.5f32, -1.0, 2.0, 0.0];
    gpu.copy_h2d(&f32_le(&dvals), delta).unwrap();

    let slot = OverlayRawSlot {
        raw: OverlayRaw {
            embed_base: Some(base),
            embed_delta: Some(delta),
            embed_r,
            embed_t: 1,
            ..Default::default()
        },
        trainable: vec![5],
    };
    let kernels = OverlayKernels::new(&gpu);
    let ov = build_overlay(&gpu, &kernels, &slot, served, served, vocab, h, true, 0)
        .unwrap()
        .expect("delta trainable id 5 overrides one row");

    assert_eq!(ov.n_override, 1);
    // CWE-125 guard SSOT: `vocab` must be the slot_map length build_overlay
    // actually allocated — the embed kernel's `ids[r] < vocab` bound.
    assert_eq!(ov.vocab, vocab as u32);
    // Builder invariant the kernel's `slot < n_tab[s]` guard leans on: every
    // non-negative slot_map entry is a compact index < n_override.
    // (Checked below over the full map once `smi` is read back.)
    // ids buffer = [5].
    let mut idb = vec![0u8; 4];
    gpu.copy_d2h(ov.ids_dev, &mut idb).unwrap();
    assert_eq!(u32::from_le_bytes([idb[0], idb[1], idb[2], idb[3]]), 5);
    // slot_map[5] = 0, everything else -1.
    let mut sm = vec![0u8; vocab * 4];
    gpu.copy_d2h(ov.slot_map, &mut sm).unwrap();
    let smi: Vec<i32> = sm
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(smi[5], 0);
    assert_eq!(smi[0], -1);
    assert_eq!(smi[7], -1);
    assert!(
        smi.iter().all(|&v| v < ov.n_override as i32),
        "slot_map entries must be compact indices < n_override: {smi:?}"
    );
    // rows[0] = f32→bf16(dvals).
    let mut rb = vec![0u8; h * 2];
    gpu.copy_d2h(ov.rows, &mut rb).unwrap();
    for (i, &x) in dvals.iter().enumerate() {
        let got = u16::from_le_bytes([rb[i * 2], rb[i * 2 + 1]]);
        assert_eq!(got, super::f32_to_bf16(x), "row col {i}");
    }
    // Tied head: no distinct lm_head overlay.
    assert!(ov.lmhead.is_none());
}

#[test]
fn build_overlay_no_override_is_none() {
    // No trainable ids and an all-equal base (row-diff no-op ⇒ empty) ⇒ None.
    let gpu = MockGpuBackend::new();
    let base = gpu.alloc(8 * 4 * 2).unwrap();
    let served = gpu.alloc(8 * 4 * 2).unwrap();
    let slot = OverlayRawSlot {
        raw: OverlayRaw {
            embed_base: Some(base),
            embed_r: 8,
            ..Default::default()
        },
        trainable: vec![],
    };
    let kernels = OverlayKernels::new(&gpu);
    let ov = build_overlay(&gpu, &kernels, &slot, served, served, 8, 4, true, 0).unwrap();
    assert!(ov.is_none());
}

#[test]
fn build_overlay_null_kernels_bails() {
    let gpu = MockGpuBackend::new();
    let base = gpu.alloc(8 * 4 * 2).unwrap();
    let served = gpu.alloc(8 * 4 * 2).unwrap();
    let slot = OverlayRawSlot {
        raw: OverlayRaw {
            embed_base: Some(base),
            embed_r: 8,
            ..Default::default()
        },
        trainable: vec![1],
    };
    let live = KernelHandle(1);
    for kernels in [
        OverlayKernels::default(),
        OverlayKernels {
            rowdiff: KernelHandle(0),
            embed_overlay: live,
            lmhead_overlay_bf16: live,
            lmhead_overlay_f32: live,
        },
        OverlayKernels {
            rowdiff: live,
            embed_overlay: KernelHandle(0),
            lmhead_overlay_bf16: live,
            lmhead_overlay_f32: live,
        },
    ] {
        let err = build_overlay(&gpu, &kernels, &slot, served, served, 8, 4, true, 0).unwrap_err();
        assert!(
            err.to_string().contains("overlay-kernels-missing"),
            "got: {err}"
        );
    }
}
