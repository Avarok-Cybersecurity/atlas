// SPDX-License-Identifier: AGPL-3.0-only
//! Kernel-level oracle for the Ada K-generic GDN write-on-accept pair
//! (`kernels/ada/common/gated_delta_rule_wyn_woa.cu`, port lane L10, row
//! `gdn_woa_mtp`): `gated_delta_rule_wyn_woa_<K>` + `gated_delta_rule_wyn_fold`
//! against the parent each width stands in for, in the table form the
//! batched verify stages:
//!
//! * K = 2: `gated_delta_rule_wy2`, and `gated_delta_rule_wy2_resident` when
//!   linked (the parent `wy2_kernel` picks at batch >= 16);
//! * K = 3: `gated_delta_rule_wy3`, and `gated_delta_rule_wy3_resident`;
//! * K = 5..16: `gated_delta_rule_wy<K>_table` (the wyN template).
//!
//! For every width in `KS` and every batch in `BATCHES` it asserts, bit for
//! bit: the twin's output equals the parent's; the twin wrote no state; the
//! fold at every accepted count na = 1..=K (the same count for every
//! sequence, then ragged counts that rotate so every sequence meets several
//! counts inside one launch) leaves H equal to the parent's Hi(na-1), or its
//! final H at na == K; the fold's flag=0 arm restores Hi(na-1) and is a
//! no-op at na == K; na = 0 never touches H. Exit code 1 on any mismatch.
//!
//!   AVAROK_TARGET_HW=ada cargo run -p spark-model --release \
//!       --features cuda,gpu-examples --example gdn_wyn_woa_oracle
//!
//! Env: KS (default "2,3,5,6,7,8,12,16"), BATCHES (default "1,2,8,16"), SEED
//! (default 1), NK/NV (default 16/32, the Qwen3.5-4B shape). Uses the build's
//! own target set (`avarok_kernels::ptx_modules`), which must carry the Ada
//! module.
//!
//! Design after the ReplaySSM record-and-fold idea (Cinference / NInfer,
//! `docs/maintainer/replayssm-gdn.md`, Apache-2.0). Idea only: no upstream
//! code is copied.

use anyhow::{Context, Result};
use half::bf16;
use spark_model::layers::ops;
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

const KD: usize = 128;
const VD: usize = 128;
/// `spark_model::layer::VERIFY_WY_TABLE_SEQS`: pointer entries per slab.
const SLAB: usize = 32;
/// Byte pattern a no-op must leave in place.
const POISON: u8 = 0x7f;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn env_list(k: &str, d: &[usize]) -> Vec<usize> {
    match std::env::var(k) {
        Ok(v) => v.split(',').filter_map(|s| s.trim().parse().ok()).collect(),
        Err(_) => d.to_vec(),
    }
}

/// Deterministic LCG in [-1, 1).
struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn upload(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len())?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn read(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

fn read_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let b = read(g, p, n * 4)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// (mismatching elements, max ULP distance) between two f32 buffers.
fn diff(a: &[f32], b: &[f32]) -> (usize, u32) {
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

/// The verify's staging layout: slab 0 = H pointers, slab t+1 = Hi_t
/// pointers, `SLAB` entries each.
fn tables(g: &dyn GpuBackend, h: &[DevicePtr], hi: &[Vec<DevicePtr>]) -> Result<DevicePtr> {
    let mut b = vec![0u8; (1 + hi.len()) * SLAB * 8];
    let slabs = std::iter::once(h).chain(hi.iter().map(|v| v.as_slice()));
    for (slab, ptrs) in slabs.enumerate() {
        for (i, p) in ptrs.iter().enumerate() {
            let o = (slab * SLAB + i) * 8;
            b[o..o + 8].copy_from_slice(&p.0.to_le_bytes());
        }
    }
    upload(g, &b)
}

#[derive(Default)]
struct Report {
    failures: usize,
}
impl Report {
    fn line(&mut self, label: &str, (n, ulp): (usize, u32), total: usize) {
        self.failures += usize::from(n != 0);
        let verdict = if n == 0 { "PASS" } else { "FAIL" };
        println!("{label:<64} {verdict}  mismatches={n}/{total}  max_ulp={ulp}");
    }
}

/// Shapes and handles of one (K, parent) case.
struct Case {
    k: usize,
    parent: KernelHandle,
    parent_name: String,
    twin: KernelHandle,
    fold: KernelHandle,
    clear: KernelHandle,
    nk: usize,
    nv: usize,
}

/// One launch's activation and output buffers.
struct Io {
    q: DevicePtr,
    k: DevicePtr,
    v: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
}

/// The parents a width is checked against: (name, handle).
fn parents(g: &dyn GpuBackend, k: usize) -> Result<Vec<(String, KernelHandle)>> {
    let names: Vec<(&str, String)> = match k {
        2 => vec![
            ("gated_delta_rule_wy", "gated_delta_rule_wy2".into()),
            (
                "gated_delta_rule_wy2_resident",
                "gated_delta_rule_wy2_resident".into(),
            ),
        ],
        3 => vec![
            ("gated_delta_rule_wy3", "gated_delta_rule_wy3".into()),
            (
                "gated_delta_rule_wy3_resident",
                "gated_delta_rule_wy3_resident".into(),
            ),
        ],
        5..=16 => vec![(
            "gated_delta_rule_wyn",
            format!("gated_delta_rule_wy{k}_table"),
        )],
        _ => anyhow::bail!("K={k}: no K-generic twin (K=4 is gdn_woa_oracle's)"),
    };
    let out: Vec<(String, KernelHandle)> = names
        .into_iter()
        .filter_map(|(m, n)| g.kernel(m, &n).ok().map(|h| (n, h)))
        .collect();
    anyhow::ensure!(!out.is_empty(), "K={k}: no parent kernel in this build");
    Ok(out)
}

/// The parent at width `c.k`, table form, over `t` (`[h | Hi0 | ..]`).
/// Kept as a dense dispatch table (one argument row per kernel signature);
/// a vertical layout would push this file over the 500-line cap.
#[rustfmt::skip]
fn launch_parent(
    g: &dyn GpuBackend,
    c: &Case,
    t: DevicePtr,
    x: &Io,
    out: DevicePtr,
    n: usize,
) -> Result<()> {
    let conv = (2 * c.nk * KD + c.nv * VD) as u32;
    let hi = |i: usize| t.offset(i * SLAB * 8);
    let (nk, nv, gb, n) = (c.nk as u32, c.nv as u32, (c.nv * 2) as u32, n as u32);
    let (kd, vd, s) = (KD as u32, VD as u32, g.default_stream());
    let (q, k, v, ga, be) = (x.q, x.k, x.v, x.gate, x.beta);
    match c.k {
        2 => ops::gdn_decode_wy2(
            g, c.parent, t, q, k, v, ga, be, out, hi(1), n, nk, nv, kd, vd, conv, conv, gb, true, s,
        ),
        3 => ops::gdn_decode_wy3(
            g, c.parent, t, q, k, v, ga, be, out, hi(1), hi(2), n, nk, nv, kd, vd, conv, conv, gb,
            true, s,
        ),
        _ => ops::gdn_decode_wyn_table(
            g, c.parent, t, q, k, v, ga, be, out, hi(1), SLAB as u32, n, nk, nv, kd, vd, conv,
            conv, gb, s,
        ),
    }
}

/// One (K, parent, batch) case.
fn run(g: &dyn GpuBackend, c: &Case, n: usize, seed: u64, rep: &mut Report) -> Result<()> {
    let (k, nk, nv) = (c.k, c.nk, c.nv);
    let key_dim = nk * KD;
    let conv = 2 * key_dim + nv * VD;
    let gb_stride = nv * 2;
    let h_numel = nv * KD * VD;
    let rows = n * k;
    let out_bytes = rows * nv * VD * 2;
    let stream = g.default_stream();
    let tag = format!("[K={k} {} b={n}]", c.parent_name);

    let mut rng = Rng(seed ^ ((k * 131 + n) as u64).wrapping_mul(0x9e37_79b9));
    let qkv: Vec<u8> = (0..rows * conv)
        .flat_map(|_| bf16::from_f32(rng.f() * 0.5).to_bits().to_le_bytes())
        .collect();
    let mut gbv = vec![0f32; rows * gb_stride];
    for r in 0..rows {
        for h in 0..nv {
            gbv[r * gb_stride + h] = 0.5 + 0.49 * rng.f(); // gate in (0.01, 0.99)
            gbv[r * gb_stride + nv + h] = 0.5 + 0.5 * rng.f().abs(); // beta
        }
    }
    let h0: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..h_numel).map(|_| rng.f() * 0.1).collect())
        .collect();
    let (qkv_d, gb_d) = (upload(g, &qkv)?, upload(g, &f32_bytes(&gbv))?);
    let x = Io {
        q: qkv_d,
        k: qkv_d.offset(key_dim * 2),
        v: qkv_d.offset(key_dim * 4),
        gate: gb_d,
        beta: gb_d.offset(nv * 4),
    };

    // Parent arm: H and K-1 Hi states per sequence.
    let h_p: Vec<DevicePtr> = h0
        .iter()
        .map(|h| upload(g, &f32_bytes(h)))
        .collect::<Result<_>>()?;
    let mut hi_p: Vec<Vec<DevicePtr>> = Vec::new();
    for _ in 0..k - 1 {
        hi_p.push(
            (0..n)
                .map(|_| upload(g, &vec![0u8; h_numel * 4]))
                .collect::<Result<_>>()?,
        );
    }
    let t_p = tables(g, &h_p, &hi_p)?;
    let out_p = g.alloc(out_bytes)?;
    launch_parent(g, c, t_p, &x, out_p, n)?;
    g.synchronize(stream)?;
    let out_parent = read(g, out_p, out_bytes)?;
    // want[na-1][s] = state after row na-1 (Hi for na < K, final H at na == K).
    let mut want: Vec<Vec<Vec<f32>>> = Vec::new();
    for slab in hi_p.iter().chain(std::iter::once(&h_p)) {
        want.push(
            slab.iter()
                .map(|&p| read_f32(g, p, h_numel))
                .collect::<Result<_>>()?,
        );
    }

    // Twin arm: fresh H, the parent's Hi pointers (the flag=0 restore reads them).
    let h_w: Vec<DevicePtr> = h0
        .iter()
        .map(|h| upload(g, &f32_bytes(h)))
        .collect::<Result<_>>()?;
    let t_w = tables(g, &h_w, &hi_p)?;
    let seq_floats = spark_model::port::gdn_woa::stash_seq_floats_k(k, nk, nv, KD, VD);
    let stash = g.alloc(n * seq_floats * 4)?;
    let flag = upload(g, &[0u8; 4])?;
    let na_tab = g.alloc(SLAB * 4)?;
    let out_w = g.alloc(out_bytes)?;

    ops::gdn_wy4_flag_clear(g, c.clear, flag, stream)?;
    ops::gdn_decode_wy4_woa(
        g,
        c.twin,
        t_w,
        x.q,
        x.k,
        x.v,
        x.gate,
        x.beta,
        out_w,
        stash,
        n as u32,
        nk as u32,
        nv as u32,
        KD as u32,
        VD as u32,
        conv as u32,
        conv as u32,
        gb_stride as u32,
        seq_floats as u32,
        flag,
        stream,
    )?;
    g.synchronize(stream)?;
    let out_twin = read(g, out_w, out_bytes)?;
    let mis = out_parent
        .chunks(2)
        .zip(out_twin.chunks(2))
        .filter(|(a, b)| a != b)
        .count();
    rep.line(
        &format!("{tag} output twin vs parent (bf16 bits)"),
        (mis, 0),
        out_bytes / 2,
    );
    for (s, &p) in h_w.iter().enumerate() {
        let d = diff(&read_f32(g, p, h_numel)?, &h0[s]);
        rep.line(&format!("{tag} twin wrote no state (seq {s})"), d, h_numel);
    }
    let engaged = u32::from_le_bytes(read(g, flag, 4)?[..4].try_into()?);
    rep.line(
        &format!("{tag} engaged word set"),
        (usize::from(engaged != 1), 0),
        1,
    );

    let reset = |poison: bool| -> Result<()> {
        for (&p, h) in h_w.iter().zip(&h0) {
            if poison {
                g.memset(p, POISON, h_numel * 4)?;
            } else {
                g.copy_h2d(&f32_bytes(h), p)?;
            }
        }
        Ok(())
    };
    let set_na = |na: &[u32]| -> Result<()> {
        let mut b = vec![0u8; SLAB * 4];
        for (i, x) in na.iter().enumerate() {
            b[i * 4..i * 4 + 4].copy_from_slice(&x.to_le_bytes());
        }
        g.copy_h2d(&b, na_tab)
    };
    let fold = || -> Result<()> {
        let hi_tables = t_w.offset(SLAB * 8);
        let (nk, nv, kd, vd, n) = (nk as u32, nv as u32, KD as u32, VD as u32, n as u32);
        let sf = seq_floats as u32;
        let slab = SLAB as u32;
        ops::gdn_wy4_fold(
            g, c.fold, t_w, stash, na_tab, hi_tables, slab, flag, k as u32, n, nk, nv, kd, vd, sf,
            stream,
        )?;
        g.synchronize(stream)
    };
    let poisoned = |v: &[f32]| {
        v.iter()
            .all(|x| x.to_bits() == u32::from_ne_bytes([POISON; 4]))
    };

    let mut cases: Vec<(String, Vec<u32>)> = (1..=k as u32)
        .map(|na| (format!("na={na}"), vec![na; n]))
        .collect();
    for r in 0..k.min(4) {
        let step = r * (k / 4).max(1);
        let na: Vec<u32> = (0..n).map(|b| 1 + ((b + step) % k) as u32).collect();
        cases.push((format!("ragged r={r}"), na));
    }

    // Engaged word 1 (the twin ran): the fold applies the stash. Then word 0
    // (the parent ran): the fold restores Hi(na-1) and leaves H at na == K.
    for engaged in [true, false] {
        if !engaged {
            ops::gdn_wy4_flag_clear(g, c.clear, flag, stream)?;
            g.synchronize(stream)?;
        }
        for (label, na) in &cases {
            reset(!engaged)?;
            set_na(na)?;
            fold()?;
            let (mut m, mut u) = (0usize, 0u32);
            for s in 0..n {
                let now = read_f32(g, h_w[s], h_numel)?;
                let (dm, du) = if !engaged && na[s] as usize == k {
                    (usize::from(!poisoned(&now)), 0)
                } else {
                    diff(&now, &want[na[s] as usize - 1][s])
                };
                (m, u) = (m + dm, u.max(du));
            }
            let arm = if engaged { "fold" } else { "flag=0 restore" };
            rep.line(
                &format!("{tag} {arm} {label} vs parent state"),
                (m, u),
                n * h_numel,
            );
        }
    }

    // na = 0 (a padded table entry) never touches H, on either arm.
    reset(true)?;
    set_na(&vec![0; n])?;
    fold()?;
    let mut touched = 0usize;
    for &p in &h_w {
        touched += usize::from(!poisoned(&read_f32(g, p, h_numel)?));
    }
    rep.line(&format!("{tag} na=0 is a no-op"), (touched, 0), n);

    let owned = [qkv_d, gb_d, t_p, out_p, t_w, stash, flag, na_tab, out_w];
    for p in owned
        .into_iter()
        .chain(h_p)
        .chain(h_w)
        .chain(hi_p.into_iter().flatten())
    {
        g.free(p)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let ks = env_list("KS", &[2, 3, 5, 6, 7, 8, 12, 16]);
    let batches = env_list("BATCHES", &[1, 2, 8, 16]);
    let seed = env_usize("SEED", 1) as u64;
    let nk = env_usize("NK", 16);
    let nv = env_usize("NV", 32);
    anyhow::ensure!(
        nv.is_multiple_of(nk) && batches.iter().all(|b| (1..=SLAB).contains(b)),
        "NV must be a multiple of NK; BATCHES in 1..={SLAB}"
    );
    let backend = AvarokCudaBackend::new(0, &avarok_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let m = "gated_delta_rule_wyn_woa";
    let fold = g
        .kernel(m, "gated_delta_rule_wyn_fold")
        .context("the Ada module gated_delta_rule_wyn_woa is not in this build")?;
    let clear = g.kernel(
        "gated_delta_rule_wy4_woa",
        "gated_delta_rule_wy4_flag_clear",
    )?;
    let mut rep = Report::default();
    for &k in &ks {
        let twin = g.kernel(m, &format!("gated_delta_rule_wyn_woa_{k}"))?;
        for (parent_name, parent) in parents(g, k)? {
            let c = Case {
                k,
                parent,
                parent_name,
                twin,
                fold,
                clear,
                nk,
                nv,
            };
            for &n in &batches {
                run(g, &c, n, seed, &mut rep)?;
            }
        }
    }
    let verdict = if rep.failures == 0 {
        "ALL PASS (bit-equal)"
    } else {
        "FAILURES"
    };
    println!(
        "gdn_wyn_woa_oracle: ks={ks:?} batches={batches:?} nk={nk} nv={nv} seed={seed}: {verdict}"
    );
    if rep.failures > 0 {
        std::process::exit(1);
    }
    Ok(())
}
