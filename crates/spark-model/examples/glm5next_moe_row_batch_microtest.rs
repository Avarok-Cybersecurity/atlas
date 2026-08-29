// SPDX-License-Identifier: AGPL-3.0-only
//! The row-batched routed-MoE gate: `w4a16_gemv_sw_moe_batchm_mR` must be BIT-IDENTICAL to
//! the per-row `w4a16_gemv_sw_moe` loop it replaces, for every (row, slot) either computes.
//!
//! The batched kernel hoists the weight load out of the row loop so an expert two rows both
//! selected is streamed ONCE. Nothing else moves: same `w4a16_gemv_partial` walk per
//! orig-lane, same `fmaf` chain, same `fmaf(scale, part, acc)` regroup, same two-term
//! combine. So "close" is a FAILURE here — the assert is byte equality.
//!
//! Covered, because each is a way the union table can be wrong rather than merely imprecise:
//!   * rows that share experts (the whole point) and rows that share none,
//!   * remote experts (`packed_ptrs == 0`) — those rows must be left untouched,
//!   * an expert selected by row 1 but not row 0, and vice versa,
//!   * the `input_stride = 0` (gate/up) and slot-major (down) input layouts.
//!
//!   cargo run -p spark-model --release --example glm5next_moe_row_batch_microtest \
//!       --features cuda,gpu-examples

use anyhow::{Result, bail};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

const N: usize = 2048; // one expert's width
const K: usize = 1024; // input width (K/16 = 64 chunks, exercises the k16 tail path)
const NUM_EXPERTS: usize = 16;
const TOP_K: usize = 4;

/// Deterministic byte soup — a real NVFP4 packing is irrelevant to a bit-equality gate, but
/// the values must be varied enough that a dropped term cannot cancel.
fn lcg(seed: &mut u64) -> u8 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*seed >> 33) as u8
}

fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}

fn dn(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

struct Table {
    packed: DevicePtr,
    scale: DevicePtr,
    scale2: DevicePtr,
}

#[allow(clippy::too_many_arguments)]
fn per_row(
    g: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    t: &Table,
    c: DevicePtr,
    ids: DevicePtr,
    n: usize,
    kk: usize,
    input_stride: usize,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([n.div_ceil(8) as u32, TOP_K as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(t.packed)
        .arg_ptr(t.scale)
        .arg_ptr(t.scale2)
        .arg_ptr(c)
        .arg_ptr(ids)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .arg_u32(NUM_EXPERTS as u32)
        .arg_u32(input_stride as u32)
        .launch(0)
}

#[allow(clippy::too_many_arguments)]
fn batched(
    g: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    t: &Table,
    c: DevicePtr,
    u_eid: DevicePtr,
    u_slot: DevicePtr,
    n: usize,
    kk: usize,
    rows: usize,
    a_row_stride: usize,
    a_slot_stride: usize,
    c_row_stride: usize,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([n.div_ceil(8) as u32, (rows * TOP_K) as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(t.packed)
        .arg_ptr(t.scale)
        .arg_ptr(t.scale2)
        .arg_ptr(c)
        .arg_ptr(u_eid)
        .arg_ptr(u_slot)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .arg_u32(NUM_EXPERTS as u32)
        .arg_u32(a_row_stride as u32)
        .arg_u32(a_slot_stride as u32)
        .arg_u32(c_row_stride as u32)
        .launch(0)
}

fn main() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let k_row = gpu.kernel("w4a16_gemv", "w4a16_gemv_sw_moe")?;
    let k_union = gpu.kernel("w4a16_gemv", "glm5next_moe_row_union")?;
    let k_b = [
        gpu.kernel("w4a16_gemv", "w4a16_gemv_sw_moe_batchm_m2")?,
        gpu.kernel("w4a16_gemv", "w4a16_gemv_sw_moe_batchm_m3")?,
        gpu.kernel("w4a16_gemv", "w4a16_gemv_sw_moe_batchm_m4")?,
    ];

    // ── expert weights + the pointer table (experts 3 and 11 are "remote") ──
    let mut seed = 0x51ed_5eedu64;
    let mut packed_ptrs = Vec::new();
    let mut scale_ptrs = Vec::new();
    let mut scale2 = Vec::new();
    for e in 0..NUM_EXPERTS {
        let remote = e == 3 || e == 11;
        if remote {
            packed_ptrs.push(0u64);
            scale_ptrs.push(0u64);
            scale2.push(0.0f32);
            continue;
        }
        let w: Vec<u8> = (0..N * K / 2).map(|_| lcg(&mut seed)).collect();
        // FP8-E4M3 group scales, kept away from 0/inf so a dropped term cannot hide.
        let s: Vec<u8> = (0..N * (K / 16)).map(|_| 0x38 | (lcg(&mut seed) & 0x07)).collect();
        packed_ptrs.push(up(&gpu, &w)?.0);
        scale_ptrs.push(up(&gpu, &s)?.0);
        scale2.push(1.0 + (e as f32) * 0.01);
    }
    let t = Table {
        packed: up(&gpu, &packed_ptrs.iter().flat_map(|p| p.to_le_bytes()).collect::<Vec<_>>())?,
        scale: up(&gpu, &scale_ptrs.iter().flat_map(|p| p.to_le_bytes()).collect::<Vec<_>>())?,
        scale2: up(&gpu, &scale2.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?,
    };

    // Routing cases. Row 0 vs row 1 vs row 2 overlap differently on purpose.
    let cases: Vec<(&str, Vec<Vec<i32>>)> = vec![
        ("2 rows, full overlap", vec![vec![0, 1, 2, 4], vec![0, 1, 2, 4]]),
        ("2 rows, no overlap", vec![vec![0, 1, 2, 4], vec![5, 6, 7, 8]]),
        ("2 rows, partial + remote", vec![vec![0, 3, 5, 9], vec![3, 5, 11, 12]]),
        ("3 rows, mixed", vec![vec![1, 2, 3, 4], vec![2, 4, 6, 8], vec![1, 8, 11, 15]]),
        ("4 rows, heavy overlap", vec![
            vec![0, 1, 2, 3], vec![0, 1, 2, 5], vec![0, 1, 6, 7], vec![0, 9, 10, 11],
        ]),
    ];

    let mut failures = 0usize;
    for (tag, ids) in &cases {
        let rows = ids.len();
        let flat: Vec<i32> = ids.iter().flatten().copied().collect();
        let d_ids = up(&gpu, &flat.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
        let d_ueid = gpu.alloc(rows * TOP_K * 4)?;
        let d_uslot = gpu.alloc(rows * TOP_K * rows * 4)?;

        KernelLaunch::new(&gpu, k_union)
            .grid([1, 1, 1])
            .block([(rows * TOP_K) as u32, 1, 1])
            .arg_ptr(d_ids)
            .arg_ptr(d_ueid)
            .arg_ptr(d_uslot)
            .arg_u32(rows as u32)
            .arg_u32(TOP_K as u32)
            .launch(0)?;
        gpu.synchronize(0)?;

        // ── the union table itself: every (row, slot) must be reachable exactly once ──
        let ueid: Vec<i32> = dn(&gpu, d_ueid, rows * TOP_K * 4)?
            .chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
        let uslot: Vec<i32> = dn(&gpu, d_uslot, rows * TOP_K * rows * 4)?
            .chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
        let mut seen = vec![vec![false; TOP_K]; rows];
        for (u, &e) in ueid.iter().enumerate() {
            if e < 0 { continue; }
            for r in 0..rows {
                let s = uslot[u * rows + r];
                if s < 0 { continue; }
                assert_eq!(ids[r][s as usize], e, "{tag}: union entry {u} claims row {r} slot {s}");
                assert!(!seen[r][s as usize], "{tag}: row {r} slot {s} claimed twice");
                seen[r][s as usize] = true;
            }
        }
        for r in 0..rows {
            for s in 0..TOP_K {
                assert!(seen[r][s], "{tag}: row {r} slot {s} never claimed");
            }
        }
        let n_union = ueid.iter().filter(|e| **e >= 0).count();
        let distinct = {
            let mut v: Vec<i32> = flat.clone();
            v.sort_unstable(); v.dedup(); v.len()
        };
        assert_eq!(n_union, distinct, "{tag}: union size");

        // ── shared-input layout (gate/up): a_slot_stride = 0 ──
        // ── slot-major layout (down): a_slot_stride = one expert's width ──
        for (layout, kk, nn, a_slot_stride) in
            [("gate/up", K, N, 0usize), ("down", N, K, N)]
        {
            let a_row_stride = if a_slot_stride == 0 { kk } else { TOP_K * kk };
            let a: Vec<u8> = (0..rows * a_row_stride * 2).map(|_| lcg(&mut seed)).collect();
            let d_a = up(&gpu, &a)?;

            let bytes = rows * TOP_K * nn * 2;
            let d_ref = gpu.alloc(bytes)?;
            let d_new = gpu.alloc(bytes)?;
            gpu.memset_async(d_ref, 0, bytes, 0)?;
            gpu.memset_async(d_new, 0, bytes, 0)?;

            for r in 0..rows {
                per_row(
                    &gpu, k_row,
                    d_a.offset(r * a_row_stride * 2),
                    &t,
                    d_ref.offset(r * TOP_K * nn * 2),
                    d_ids.offset(r * TOP_K * 4),
                    nn, kk, a_slot_stride,
                )?;
            }
            batched(
                &gpu, k_b[rows - 2], d_a, &t, d_new, d_ueid, d_uslot,
                nn, kk, rows, a_row_stride, a_slot_stride, TOP_K * nn,
            )?;
            gpu.synchronize(0)?;

            let r_ref = dn(&gpu, d_ref, bytes)?;
            let r_new = dn(&gpu, d_new, bytes)?;
            if r_ref == r_new {
                println!("  PASS  {tag:28} [{layout:7}] rows={rows} union={n_union}/{}", rows * TOP_K);
            } else {
                let diff = r_ref.iter().zip(&r_new).filter(|(a, b)| a != b).count();
                println!("  FAIL  {tag:28} [{layout:7}] {diff}/{bytes} bytes differ");
                failures += 1;
            }
        }
    }

    if failures > 0 {
        bail!("{failures} arm(s) are not bit-identical to the per-row path");
    }
    println!("\nrow-batched MoE is bit-identical to the per-row path on every arm.");
    Ok(())
}
