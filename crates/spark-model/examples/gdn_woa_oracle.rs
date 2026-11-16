// SPDX-License-Identifier: AGPL-3.0-only
//! Kernel-level oracle for the GDN write-on-accept K=4 pair.
//!
//! One GDN layer, random H / q / k / v / gate / beta over `seqs` sequences,
//! for every batch width in `BATCHES` (default 1, 2 and 8: the MTP batched
//! verify's C=1..8 shapes). Runs the PARENT `gated_delta_rule_wy4` (table
//! form) and the twin `gated_delta_rule_wy4_woa` + `gated_delta_rule_wy4_fold`,
//! then asserts:
//!
//! * `output` bit-equal between parent and twin, and the twin wrote no state;
//! * for na in 1..=4, the SAME na for every sequence: H after the fold
//!   bit-equal to the parent's Hi(na-1) (na < 4) or its final H (na == 4);
//! * RAGGED accepted counts (sequence b folds na = 1 + (b + r) % 4 for four
//!   rotations r, so every sequence meets every count inside one launch, the
//!   shape a real batched verdict produces);
//! * the flag=0 branch (parent ran): the fold performs the parent's
//!   partial-accept restore (H = Hi(na-1)) and is a no-op at na == 4, also
//!   with ragged counts;
//! * na = 0 (a padded table entry) leaves H untouched.
//!
//! Every mismatch is counted and the maximum ULP delta reported, so a
//! non-zero result is a measurement, not a guess. Exit code 1 on any
//! mismatch.
//!
//!   cargo run -p spark-model --release --features cuda,gpu-examples \
//!       --example gdn_woa_oracle
//!
//! Env: BATCHES (default "1,2,8"), SEED (default 1), NK/NV (default 16/48;
//! 16/32 is the Qwen3.5-4B shape). Kernels: the exact qwen3.8-27b/nvfp4 set
//! when the build carries it, else the build's own target set (an
//! `AVAROK_TARGET_HW=ada` build, whose common/ links the same pair).
//!
//! provenance-id: 526f6e616c6420522e205374657369616b

#[path = "gdn_woa_oracle_parts/util.rs"]
mod util;

use anyhow::Result;
use spark_model::layers::ops;
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use util::{
    Rng, diff_f32, env_list, env_usize, ptr_table, read_f32, read_u16, upload_bf16, upload_f32,
};

const KD: usize = 128;
const VD: usize = 128;
/// Matches `spark_model::layer::VERIFY_WY_TABLE_SEQS`: pointer entries per slab.
const SLAB_ENTRIES: usize = 32;

struct Kernels {
    wy4: KernelHandle,
    woa: KernelHandle,
    fold: KernelHandle,
    clear: KernelHandle,
}

/// PASS/FAIL lines and the failure count.
#[derive(Default)]
struct Report {
    failures: usize,
}
impl Report {
    fn line(&mut self, label: &str, n: usize, ulp: u32, total: usize) {
        if n != 0 {
            self.failures += 1;
        }
        println!(
            "{:<58} {}  mismatches={n}/{total}  max_ulp={ulp}",
            label,
            if n == 0 { "PASS" } else { "FAIL" }
        );
    }
}

/// State after row `na - 1` of sequence `s`, as the parent wrote it (its
/// final H at na == 4).
fn parent_state<'a>(h: &'a [Vec<f32>], hi: &'a [Vec<Vec<f32>>], s: usize, na: u32) -> &'a [f32] {
    if na == 4 {
        &h[s]
    } else {
        &hi[na as usize - 1][s]
    }
}

/// One batch width: parent, twin, uniform and ragged folds, flag=0 restore.
fn run_batch(
    g: &dyn GpuBackend,
    k: &Kernels,
    seqs: usize,
    nk: usize,
    nv: usize,
    seed: u64,
    rep: &mut Report,
) -> Result<()> {
    let key_dim = nk * KD;
    let value_dim = nv * VD;
    let conv_dim = 2 * key_dim + value_dim; // qk_stride == v_stride == conv_dim
    let gb_stride = nv * 2;
    let h_numel = nv * KD * VD;
    let stream = g.default_stream();
    let tag = format!("[b={seqs}]");

    // ── Inputs (same bytes for both arms) ──
    let mut rng = Rng(seed ^ (seqs as u64).wrapping_mul(0x9e37_79b9));
    let rows = seqs * 4;
    let qkv: Vec<f32> = (0..rows * conv_dim).map(|_| rng.next_f32() * 0.5).collect();
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
    let (q_ptr, k_ptr, v_ptr) = (
        qkv_dev,
        qkv_dev.offset(key_dim * 2),
        qkv_dev.offset(key_dim * 4),
    );
    let (gate_ptr, beta_ptr) = (gb_dev, gb_dev.offset(nv * 4));

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
        k.wy4,
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
    // SLAB_ENTRIES pointers each, exactly the verify's staging layout, so the
    // fold's `hi_tables` indexing is exercised as shipped. ──
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

    let reset_h = |fill: Option<u8>| -> Result<()> {
        for (p, h) in h_w.iter().zip(&h_init) {
            match fill {
                Some(b) => g.memset(*p, b, h_numel * 4)?,
                None => {
                    let bytes: Vec<u8> = h.iter().flat_map(|x| x.to_le_bytes()).collect();
                    g.copy_h2d(&bytes, *p)?;
                }
            }
        }
        Ok(())
    };
    let set_na = |na: &[u32]| -> Result<()> {
        let mut b = vec![0u8; SLAB_ENTRIES * 4];
        for (i, v) in na.iter().enumerate() {
            b[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        g.copy_h2d(&b, na_tab)
    };
    let run_fold = || -> Result<()> {
        ops::gdn_wy4_fold(
            g,
            k.fold,
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
    let ragged = |r: usize| -> Vec<u32> { (0..seqs).map(|b| 1 + ((b + r) % 4) as u32).collect() };

    // Twin: output byte-identical to the parent, no state written.
    ops::gdn_wy4_flag_clear(g, k.clear, flag, stream)?;
    ops::gdn_decode_wy4_woa(
        g,
        k.woa,
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
    rep.line(
        &format!("{tag} output: twin vs parent (bf16 bits)"),
        out_mis,
        0,
        out_parent.len(),
    );
    for (s, p) in h_w.iter().enumerate() {
        let (n, ulp) = diff_f32(&read_f32(g, *p, h_numel)?, &h_init[s]);
        rep.line(
            &format!("{tag} twin wrote no state (seq {s})"),
            n,
            ulp,
            h_numel,
        );
    }
    let flag_now = read_f32(g, flag, 1)?[0].to_bits();
    rep.line(
        &format!("{tag} engaged word set by the twin"),
        usize::from(flag_now != 1),
        0,
        1,
    );

    // Uniform folds at na = 1..=4, then ragged folds (four rotations).
    let mut cases: Vec<(String, Vec<u32>)> = (1..=4u32)
        .map(|na| (format!("na={na}"), vec![na; seqs]))
        .collect();
    cases.extend((0..4).map(|r| (format!("ragged r={r}"), ragged(r))));
    for (label, na) in &cases {
        reset_h(None)?;
        set_na(na)?;
        run_fold()?;
        for s in 0..seqs {
            let want = parent_state(&h_parent, &hi_parent, s, na[s]);
            let (n, ulp) = diff_f32(&read_f32(g, h_w[s], h_numel)?, want);
            rep.line(
                &format!("{tag} fold {label} vs parent (seq {s}, na={})", na[s]),
                n,
                ulp,
                h_numel,
            );
        }
    }

    // na = 0 (a padded entry) never touches H.
    reset_h(Some(0x7f))?;
    set_na(&vec![0; seqs])?;
    run_fold()?;
    for s in 0..seqs {
        let untouched = read_f32(g, h_w[s], h_numel)?
            .iter()
            .all(|x| x.to_bits() == 0x7f7f7f7f);
        rep.line(
            &format!("{tag} na=0 is a no-op (seq {s})"),
            usize::from(!untouched),
            0,
            h_numel,
        );
    }

    // flag = 0: the parent ran; the fold restores H = Hi(na-1) and does
    // nothing at na == 4, also with ragged counts.
    ops::gdn_wy4_flag_clear(g, k.clear, flag, stream)?;
    g.synchronize(stream)?;
    for (label, na) in &cases {
        reset_h(Some(0x7f))?; // garbage, so a no-op is visible
        set_na(na)?;
        run_fold()?;
        for s in 0..seqs {
            let now = read_f32(g, h_w[s], h_numel)?;
            let (n, ulp) = if na[s] == 4 {
                (
                    usize::from(!now.iter().all(|x| x.to_bits() == 0x7f7f7f7f)),
                    0,
                )
            } else {
                diff_f32(&now, &hi_parent[na[s] as usize - 1][s])
            };
            rep.line(
                &format!(
                    "{tag} flag=0 {label} parent restore (seq {s}, na={})",
                    na[s]
                ),
                n,
                ulp,
                h_numel,
            );
        }
    }
    for p in h_p.iter().chain(&h_w).chain(hi_p.iter().flatten()) {
        g.free(*p)?;
    }
    for p in [
        qkv_dev, gb_dev, t_h_p, out_p, tables, stash, flag, na_tab, out_w,
    ]
    .into_iter()
    .chain(t_hi_p)
    {
        g.free(p)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let batches = env_list("BATCHES", &[1, 2, 8]);
    let seed = env_usize("SEED", 1) as u64;
    let nk = env_usize("NK", 16);
    let nv = env_usize("NV", 48);
    anyhow::ensure!(
        !batches.is_empty()
            && batches.iter().all(|b| (1..=SLAB_ENTRIES).contains(b))
            && nv.is_multiple_of(nk),
        "BATCHES must list widths in 1..={SLAB_ENTRIES}; NV must be a multiple of NK"
    );
    let modules = match avarok_kernels::ptx_for_exact_target("qwen3.8-27b", "nvfp4") {
        Some(set) => set.modules,
        None => {
            println!(
                "gdn_woa_oracle: no qwen3.8-27b/nvfp4 set in this build; using its own target set"
            );
            avarok_kernels::ptx_modules()
        }
    };
    let backend = AvarokCudaBackend::new(0, &modules)?;
    let g: &dyn GpuBackend = &backend;
    let m = "gated_delta_rule_wy4_woa";
    let k = Kernels {
        wy4: g.kernel("gated_delta_rule_wy4", "gated_delta_rule_wy4")?,
        woa: g.kernel(m, "gated_delta_rule_wy4_woa")?,
        fold: g.kernel(m, "gated_delta_rule_wy4_fold")?,
        clear: g.kernel(m, "gated_delta_rule_wy4_flag_clear")?,
    };
    let mut rep = Report::default();
    for &seqs in &batches {
        run_batch(g, &k, seqs, nk, nv, seed, &mut rep)?;
    }
    println!(
        "gdn_woa_oracle: batches={batches:?} nk={nk} nv={nv} seed={seed}: {}",
        if rep.failures == 0 {
            "ALL PASS (bit-equal)"
        } else {
            "FAILURES"
        }
    );
    if rep.failures > 0 {
        std::process::exit(1);
    }
    Ok(())
}
