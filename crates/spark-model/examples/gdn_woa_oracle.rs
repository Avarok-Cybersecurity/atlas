// SPDX-License-Identifier: AGPL-3.0-only
//! Kernel-level oracle for the GDN write-on-accept K=4 pair.
//!
//! One GDN layer, random H / q / k / v / gate / beta over `SEQS` sequences.
//! Runs the PARENT `gated_delta_rule_wy4` (table form) and the twin
//! `gated_delta_rule_wy4_woa` + `gated_delta_rule_wy4_fold`, then asserts:
//!
//! * `output` bit-equal between parent and twin;
//! * for na in 1..=4: H after the fold bit-equal to the parent's Hi(na-1)
//!   (na < 4) or its final H (na == 4);
//! * the flag=0 branch (parent ran): the fold performs the parent's
//!   partial-accept restore (H = Hi(na-1)) and is a no-op at na == 4.
//!
//! Every mismatch is counted and the maximum ULP delta reported, so a
//! non-zero result is a measurement, not a guess. Exit code 1 on any
//! mismatch.
//!
//!   cargo run -p spark-model --release --features gpu-examples \
//!       --example gdn_woa_oracle
//!
//! Env: SEQS (default 4), SEED (default 1), NK/NV (default 16/48).
//!
//! provenance-id: 526f6e616c6420522e205374657369616b

use anyhow::{Context, Result};
use half::bf16;
use spark_model::layers::ops;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

const KD: usize = 128;
const VD: usize = 128;
/// Matches `spark_model::layer::VERIFY_WY_TABLE_SEQS`: pointer entries per slab.
const SLAB_ENTRIES: usize = 32;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

/// Deterministic LCG in [-1, 1).
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn upload_f32(g: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(bytes.len())?;
    g.copy_h2d(&bytes, p)?;
    Ok(p)
}
fn upload_bf16(g: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    let bytes: Vec<u8> = v
        .iter()
        .flat_map(|x| bf16::from_f32(*x).to_bits().to_le_bytes())
        .collect();
    let p = g.alloc(bytes.len())?;
    g.copy_h2d(&bytes, p)?;
    Ok(p)
}
fn read_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
fn read_u16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}
fn ptr_table(g: &dyn GpuBackend, ptrs: &[DevicePtr]) -> Result<DevicePtr> {
    let mut bytes = vec![0u8; SLAB_ENTRIES * 8];
    for (i, p) in ptrs.iter().enumerate() {
        bytes[i * 8..i * 8 + 8].copy_from_slice(&p.0.to_le_bytes());
    }
    let t = g.alloc(bytes.len())?;
    g.copy_h2d(&bytes, t)?;
    Ok(t)
}

/// (mismatching elements, max ULP distance) between two f32 buffers.
fn diff_f32(a: &[f32], b: &[f32]) -> (usize, u32) {
    let mut n = 0usize;
    let mut ulp = 0u32;
    for (x, y) in a.iter().zip(b) {
        if x.to_bits() != y.to_bits() {
            n += 1;
            let d = (x.to_bits() as i64 - y.to_bits() as i64).unsigned_abs();
            ulp = ulp.max(d.min(u32::MAX as u64) as u32);
        }
    }
    (n, ulp)
}

fn main() -> Result<()> {
    let seqs = env_usize("SEQS", 4);
    let seed = env_usize("SEED", 1) as u64;
    let nk = env_usize("NK", 16);
    let nv = env_usize("NV", 48);
    anyhow::ensure!((1..=SLAB_ENTRIES).contains(&seqs) && nv.is_multiple_of(nk));
    let key_dim = nk * KD;
    let value_dim = nv * VD;
    let conv_dim = 2 * key_dim + value_dim; // qk_stride == v_stride == conv_dim
    let gb_stride = nv * 2;
    let hv = KD * VD;
    let h_numel = nv * hv;

    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-27b", "nvfp4")
        .context("no compiled qwen3.8-27b/nvfp4 kernel set (ATLAS_TARGET_MODEL=qwen3.8-27b)")?;
    let backend = AtlasCudaBackend::new(0, &set.modules)?;
    let g: &dyn GpuBackend = &backend;
    let wy4 = g.kernel("gated_delta_rule_wy4", "gated_delta_rule_wy4")?;
    let woa = g.kernel("gated_delta_rule_wy4_woa", "gated_delta_rule_wy4_woa")?;
    let fold = g.kernel("gated_delta_rule_wy4_woa", "gated_delta_rule_wy4_fold")?;
    let clear = g.kernel(
        "gated_delta_rule_wy4_woa",
        "gated_delta_rule_wy4_flag_clear",
    )?;
    let stream = g.default_stream();

    // ── Inputs (same bytes for both arms) ──
    let mut rng = Rng(seed);
    let rows = seqs * 4;
    let mut qkv = vec![0f32; rows * conv_dim];
    for x in qkv.iter_mut() {
        *x = rng.next_f32() * 0.5;
    }
    let mut gb = vec![0f32; rows * gb_stride];
    for r in 0..rows {
        for h in 0..nv {
            gb[r * gb_stride + h] = 0.5 + 0.49 * rng.next_f32(); // gate in (0.01, 0.99)
            gb[r * gb_stride + nv + h] = 0.5 + 0.5 * rng.next_f32().abs(); // beta
        }
    }
    let h_init: Vec<Vec<f32>> = (0..seqs)
        .map(|_| (0..h_numel).map(|_| rng.next_f32() * 0.1).collect())
        .collect();
    let qkv_dev = upload_bf16(g, &qkv)?;
    let gb_dev = upload_f32(g, &gb)?;
    let q_ptr = qkv_dev;
    let k_ptr = qkv_dev.offset(key_dim * 2);
    let v_ptr = qkv_dev.offset(key_dim * 2 * 2);
    let gate_ptr = gb_dev;
    let beta_ptr = gb_dev.offset(nv * 4);

    // ── Parent arm: H + Hi0..Hi2 per sequence, table form ──
    let h_p: Vec<DevicePtr> = h_init
        .iter()
        .map(|h| upload_f32(g, h))
        .collect::<Result<_>>()?;
    let mut hi_p: Vec<Vec<DevicePtr>> = Vec::new();
    for _ in 0..3 {
        let mut v = Vec::new();
        for _ in 0..seqs {
            let p = g.alloc(h_numel * 4)?;
            g.memset(p, 0, h_numel * 4)?;
            v.push(p);
        }
        hi_p.push(v);
    }
    let t_h_p = ptr_table(g, &h_p)?;
    let t_hi_p: Vec<DevicePtr> = hi_p
        .iter()
        .map(|v| ptr_table(g, v))
        .collect::<Result<_>>()?;
    let out_p = g.alloc(rows * value_dim * 2)?;
    ops::gdn_decode_wy4(
        g,
        wy4,
        t_h_p,
        q_ptr,
        k_ptr,
        v_ptr,
        gate_ptr,
        beta_ptr,
        out_p,
        t_hi_p[0],
        t_hi_p[1],
        t_hi_p[2],
        seqs as u32,
        nk as u32,
        nv as u32,
        KD as u32,
        VD as u32,
        conv_dim as u32,
        conv_dim as u32,
        gb_stride as u32,
        true,
        stream,
    )?;
    g.synchronize(stream)?;
    let out_parent = read_u16(g, out_p, rows * value_dim)?;
    let h_parent: Vec<Vec<f32>> = h_p
        .iter()
        .map(|&p| read_f32(g, p, h_numel))
        .collect::<Result<_>>()?;
    let hi_parent: Vec<Vec<Vec<f32>>> = hi_p
        .iter()
        .map(|v| {
            v.iter()
                .map(|&p| read_f32(g, p, h_numel))
                .collect::<Result<_>>()
        })
        .collect::<Result<_>>()?;

    // ── Twin arm: one contiguous table buffer [h | Hi0 | Hi1 | Hi2] of
    // SLAB_ENTRIES pointers each, exactly the verify's staging layout, so
    // the fold's `hi_tables` indexing is exercised as shipped. ──
    let h_w: Vec<DevicePtr> = h_init
        .iter()
        .map(|h| upload_f32(g, h))
        .collect::<Result<_>>()?;
    let mut table_bytes = vec![0u8; 4 * SLAB_ENTRIES * 8];
    for (slab, ptrs) in [&h_w, &hi_p[0], &hi_p[1], &hi_p[2]].into_iter().enumerate() {
        for (i, p) in ptrs.iter().enumerate() {
            let o = (slab * SLAB_ENTRIES + i) * 8;
            table_bytes[o..o + 8].copy_from_slice(&p.0.to_le_bytes());
        }
    }
    let tables = g.alloc(table_bytes.len())?;
    g.copy_h2d(&table_bytes, tables)?;
    let hi_tables = tables.offset(SLAB_ENTRIES * 8);
    let seq_floats = 4 * (nv * VD + nv + nk * KD);
    let stash = g.alloc(seqs * seq_floats * 4)?;
    let flag = g.alloc(4)?;
    g.memset(flag, 0, 4)?;
    let na_tab = g.alloc(SLAB_ENTRIES * 4)?;
    let out_w = g.alloc(rows * value_dim * 2)?;

    let reset_h = |g: &dyn GpuBackend| -> Result<()> {
        for (p, h) in h_w.iter().zip(&h_init) {
            let bytes: Vec<u8> = h.iter().flat_map(|x| x.to_le_bytes()).collect();
            g.copy_h2d(&bytes, *p)?;
        }
        Ok(())
    };
    let set_na = |g: &dyn GpuBackend, na: u32| -> Result<()> {
        let mut b = vec![0u8; SLAB_ENTRIES * 4];
        for i in 0..seqs {
            b[i * 4..i * 4 + 4].copy_from_slice(&na.to_le_bytes());
        }
        g.copy_h2d(&b, na_tab)
    };
    let run_fold = |g: &dyn GpuBackend| -> Result<()> {
        ops::gdn_wy4_fold(
            g,
            fold,
            tables,
            stash,
            na_tab,
            hi_tables,
            SLAB_ENTRIES as u32,
            flag,
            4,
            seqs as u32,
            nk as u32,
            nv as u32,
            KD as u32,
            VD as u32,
            seq_floats as u32,
            stream,
        )?;
        g.synchronize(stream)
    };

    let mut failures = 0usize;
    let mut report = |label: &str, n: usize, ulp: u32, total: usize| {
        let ok = n == 0;
        if !ok {
            failures += 1;
        }
        println!(
            "{:<44} {}  mismatches={n}/{total}  max_ulp={ulp}",
            label,
            if ok { "PASS" } else { "FAIL" }
        );
    };

    // Twin: output must be byte-identical to the parent.
    ops::gdn_wy4_flag_clear(g, clear, flag, stream)?;
    ops::gdn_decode_wy4_woa(
        g,
        woa,
        tables,
        q_ptr,
        k_ptr,
        v_ptr,
        gate_ptr,
        beta_ptr,
        out_w,
        stash,
        seqs as u32,
        nk as u32,
        nv as u32,
        KD as u32,
        VD as u32,
        conv_dim as u32,
        conv_dim as u32,
        gb_stride as u32,
        seq_floats as u32,
        flag,
        stream,
    )?;
    g.synchronize(stream)?;
    let out_twin = read_u16(g, out_w, rows * value_dim)?;
    let out_mis = out_parent
        .iter()
        .zip(&out_twin)
        .filter(|(a, b)| a != b)
        .count();
    report(
        "output: twin vs parent (bf16 bits)",
        out_mis,
        0,
        out_parent.len(),
    );
    // The twin writes no state: H must still be the initial value.
    for (s, p) in h_w.iter().enumerate() {
        let now = read_f32(g, *p, h_numel)?;
        let (n, ulp) = diff_f32(&now, &h_init[s]);
        report(&format!("twin wrote no state (seq {s})"), n, ulp, h_numel);
    }
    let flag_now = read_f32(g, flag, 1)?[0].to_bits();
    report(
        "engaged word set by the twin",
        usize::from(flag_now != 1),
        0,
        1,
    );

    // Fold at na = 1..=4 against Hi(na-1) / final H of the parent.
    for na in 1..=4u32 {
        reset_h(g)?;
        set_na(g, na)?;
        run_fold(g)?;
        for s in 0..seqs {
            let now = read_f32(g, h_w[s], h_numel)?;
            let want = if na == 4 {
                &h_parent[s]
            } else {
                &hi_parent[na as usize - 1][s]
            };
            let (n, ulp) = diff_f32(&now, want);
            report(
                &format!("fold na={na} vs parent state (seq {s})"),
                n,
                ulp,
                h_numel,
            );
        }
    }

    // flag = 0: the parent ran; the fold must perform the parent's
    // partial-accept restore (H = Hi(na-1)) and nothing at na == 4.
    ops::gdn_wy4_flag_clear(g, clear, flag, stream)?;
    g.synchronize(stream)?;
    for na in 1..=4u32 {
        // Start from garbage so a no-op is visible.
        for p in &h_w {
            g.memset(*p, 0x7f, h_numel * 4)?;
        }
        set_na(g, na)?;
        run_fold(g)?;
        for s in 0..seqs {
            let now = read_f32(g, h_w[s], h_numel)?;
            if na == 4 {
                let untouched = now.iter().all(|x| x.to_bits() == 0x7f7f7f7f);
                report(
                    &format!("flag=0 na=4 is a no-op (seq {s})"),
                    usize::from(!untouched),
                    0,
                    h_numel,
                );
            } else {
                let (n, ulp) = diff_f32(&now, &hi_parent[na as usize - 1][s]);
                report(
                    &format!("flag=0 na={na} restores Hi(na-1) (seq {s})"),
                    n,
                    ulp,
                    h_numel,
                );
            }
        }
    }

    println!(
        "gdn_woa_oracle: seqs={seqs} nk={nk} nv={nv} seed={seed}: {}",
        if failures == 0 {
            "ALL PASS (bit-equal)"
        } else {
            "FAILURES"
        }
    );
    if failures > 0 {
        std::process::exit(1);
    }
    Ok(())
}
