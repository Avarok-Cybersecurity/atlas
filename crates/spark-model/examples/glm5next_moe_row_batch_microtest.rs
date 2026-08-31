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
//!   * the `input_stride = 0` (gate/up) and slot-major (down) input layouts,
//!   * **every tier 2..=8** (widened 2026-08-31 from 2..=4), at **both `top_k = 4` and
//!     `top_k = 8`** — the latter is GLM-5.3's real routing, where `rows = 8` puts
//!     `rows * top_k` at exactly 64, the single block `glm5next_moe_row_union` gets. That
//!     edge is the reason the sweep runs to 8 x 8 and not just to "wide enough".
//!
//! 🪤 The union table is checked SEPARATELY from the arithmetic (every (row, slot) claimed
//! exactly once, union size == distinct id count). A tier that silently dropped ids past the
//! block would still produce a self-consistent-looking output otherwise.
//!
//!   cargo run -p spark-model --release --example glm5next_moe_row_batch_microtest \
//!       --features cuda,gpu-examples

use anyhow::{Result, bail};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

const N: usize = 2048; // one expert's width
const K: usize = 1024; // input width (K/16 = 64 chunks, exercises the k16 tail path)
/// 72, not 16: at `rows = 8`, `top_k = 8` the disjoint-routing case needs 64 distinct ids
/// before it can also reserve the two "remote" ones.
const NUM_EXPERTS: usize = 72;
/// Both routings are exercised. 8 is GLM-5.3's `num_experts_per_tok`.
const TOP_KS: [usize; 2] = [4, 8];
/// Widest compiled tier — mirror of `ATLAS_MOE_BATCHM_ENTRY` in `w4a16_gemv.cu`.
const MAX_ROWS: usize = 8;
/// `glm5next_moe_row_union` is ONE block; `rows * top_k` past this would drop entries.
const MAX_UNION_IDS: usize = 64;

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
    top_k: usize,
    input_stride: usize,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([n.div_ceil(8) as u32, top_k as u32, 1])
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
    top_k: usize,
    a_row_stride: usize,
    a_slot_stride: usize,
    c_row_stride: usize,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([n.div_ceil(8) as u32, (rows * top_k) as u32, 1])
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
    let k_b: Vec<KernelHandle> = (2..=MAX_ROWS)
        .map(|r| gpu.kernel("w4a16_gemv", &format!("w4a16_gemv_sw_moe_batchm_m{r}")))
        .collect::<Result<_, _>>()?;

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

    // Routing cases, generated per (rows, top_k) rather than hand-listed: the sweep now runs
    // 2..=8 rows at two `top_k`, and a hand-listed table cannot express both widths.
    // Experts 3 and 11 are the REMOTE pair (null pointers) — `partial+remote` puts them in
    // deliberately, so a tier that touched a remote row would show up as a diff.
    fn cases(rows: usize, top_k: usize) -> Vec<(String, Vec<Vec<i32>>)> {
        let m = NUM_EXPERTS as i32;
        let disjoint: Vec<Vec<i32>> = (0..rows)
            .map(|r| (0..top_k).map(|i| ((r * top_k + i) as i32) % m).collect())
            .collect();
        let full: Vec<Vec<i32>> = (0..rows)
            .map(|_| (0..top_k).map(|i| (i as i32 * 2) % m).collect())
            .collect();
        // Two shared ids at the front, the rest fanning out; both remote experts present.
        let partial: Vec<Vec<i32>> = (0..rows)
            .map(|r| {
                let mut v = vec![3i32, 11];
                let mut n = 0i32;
                while v.len() < top_k {
                    let c = ((r as i32 * 5) + n * 3 + 20) % m;
                    if !v.contains(&c) {
                        v.push(c);
                    }
                    n += 1;
                }
                v.truncate(top_k);
                v
            })
            .collect();
        // Half shared, half private — the case the union is actually meant to win on.
        let heavy: Vec<Vec<i32>> = (0..rows)
            .map(|r| {
                let mut v: Vec<i32> = (0..top_k / 2).map(|i| i as i32).collect();
                let mut n = 0i32;
                while v.len() < top_k {
                    let c = (30 + r as i32 * 7 + n) % m;
                    if !v.contains(&c) {
                        v.push(c);
                    }
                    n += 1;
                }
                v.truncate(top_k);
                v
            })
            .collect();
        vec![
            (format!("{rows}r k{top_k} disjoint"), disjoint),
            (format!("{rows}r k{top_k} full overlap"), full),
            (format!("{rows}r k{top_k} partial+remote"), partial),
            (format!("{rows}r k{top_k} heavy overlap"), heavy),
        ]
    }

    let all: Vec<(String, Vec<Vec<i32>>, usize)> = TOP_KS
        .iter()
        .flat_map(|&tk| {
            (2..=MAX_ROWS)
                .filter(move |r| r * tk <= MAX_UNION_IDS)
                .flat_map(move |r| cases(r, tk).into_iter().map(move |(t, c)| (t, c, tk)))
        })
        .collect();

    let mut failures = 0usize;
    for (tag, ids, top_k) in &all {
        let (rows, top_k) = (ids.len(), *top_k);
        let flat: Vec<i32> = ids.iter().flatten().copied().collect();
        let d_ids = up(&gpu, &flat.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
        let d_ueid = gpu.alloc(rows * top_k * 4)?;
        let d_uslot = gpu.alloc(rows * top_k * rows * 4)?;

        KernelLaunch::new(&gpu, k_union)
            .grid([1, 1, 1])
            .block([(rows * top_k) as u32, 1, 1])
            .arg_ptr(d_ids)
            .arg_ptr(d_ueid)
            .arg_ptr(d_uslot)
            .arg_u32(rows as u32)
            .arg_u32(top_k as u32)
            .launch(0)?;
        gpu.synchronize(0)?;

        // ── the union table itself: every (row, slot) must be reachable exactly once ──
        let ueid: Vec<i32> = dn(&gpu, d_ueid, rows * top_k * 4)?
            .chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
        let uslot: Vec<i32> = dn(&gpu, d_uslot, rows * top_k * rows * 4)?
            .chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
        let mut seen = vec![vec![false; top_k]; rows];
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
            for s in 0..top_k {
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
            let a_row_stride = if a_slot_stride == 0 { kk } else { top_k * kk };
            let a: Vec<u8> = (0..rows * a_row_stride * 2).map(|_| lcg(&mut seed)).collect();
            let d_a = up(&gpu, &a)?;

            let bytes = rows * top_k * nn * 2;
            let d_ref = gpu.alloc(bytes)?;
            let d_new = gpu.alloc(bytes)?;
            gpu.memset_async(d_ref, 0, bytes, 0)?;
            gpu.memset_async(d_new, 0, bytes, 0)?;

            for r in 0..rows {
                per_row(
                    &gpu, k_row,
                    d_a.offset(r * a_row_stride * 2),
                    &t,
                    d_ref.offset(r * top_k * nn * 2),
                    d_ids.offset(r * top_k * 4),
                    nn, kk, top_k, a_slot_stride,
                )?;
            }
            batched(
                &gpu, k_b[rows - 2], d_a, &t, d_new, d_ueid, d_uslot,
                nn, kk, rows, top_k, a_row_stride, a_slot_stride, top_k * nn,
            )?;
            gpu.synchronize(0)?;

            let r_ref = dn(&gpu, d_ref, bytes)?;
            let r_new = dn(&gpu, d_new, bytes)?;
            if r_ref == r_new {
                println!("  PASS  {tag:28} [{layout:7}] rows={rows} union={n_union}/{}", rows * top_k);
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
    println!("\nrow-batched MoE is bit-identical to the per-row path on every arm, tiers 2..=8.");
    Ok(())
}
