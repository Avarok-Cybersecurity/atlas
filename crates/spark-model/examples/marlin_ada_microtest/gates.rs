// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the kernel-side gates of the Marlin microtest: every one of the 30 entry points
//! once against the CPU oracle, the decode rows M = 1, 8, 16, 17, 32, 64 with a planted control, row
//! invariance across M with its vLLM-selector contrast, a five-run byte repeat, the prompt path
//! (`prefill`) and the locks returning to zero.

use anyhow::{Result, ensure};
use spark_model::port::marlin::config::{Chunk, ThreadConfig};
use spark_model::port::marlin::kernels::{self, KernelKey, entry_name};
use spark_model::port::marlin::w4linear::{DecodePolicy, MarlinLinear};
use spark_model::port::marlin::{LogicalInt4, MarlinType, build};
use spark_model::port::w4::{W4Linear, W4Scratch};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::util::{Lcg, Tally, download, download_u16, oracle, upload_u16};

/// Column stride of the sampled oracle (prime, so every N tile and warp lane is visited).
const STRIDE: usize = 7;

/// A random logical weight: uniform codes, BF16 scales in `[0.002, 0.03)`, zero points when `zp`.
pub fn random_logical(r: &mut Lcg, n: usize, k: usize, zp: bool) -> LogicalInt4 {
    let groups = k / 128;
    let q = (0..k * n).map(|_| r.code()).collect();
    let scales = (0..groups * n)
        .map(|_| spark_model::port::marlin::reference::f32_to_bf16(0.002 + 0.028 * r.unit()))
        .collect();
    let zeros = zp.then(|| (0..groups * n).map(|_| r.code()).collect());
    LogicalInt4 {
        n,
        k,
        group: 128,
        q,
        scales,
        zeros,
    }
}

fn btype_of(zp: bool) -> MarlinType {
    if zp { MarlinType::U4 } else { MarlinType::U4B8 }
}

/// The entry name of the first launch of `rows` under `policy`.
pub fn plan_name(lin: &MarlinLinear, rows: usize, policy: DecodePolicy) -> Result<String> {
    let w = lin.weight();
    let c = lin.plan(rows, policy)?;
    let names: Vec<String> = c
        .iter()
        .map(|c| {
            entry_name(&KernelKey::of(
                w.btype,
                w.problem().group_blocks(),
                c.thread_m_blocks,
                c.m_block_size_8,
                &c.cfg,
            ))
        })
        .collect();
    Ok(names.join("+"))
}

/// Fill `rows * n` outputs with BF16 NaN so an output the kernel never writes fails the oracle.
fn poison(gpu: &dyn GpuBackend, out: DevicePtr, rows: usize, n: usize) -> Result<()> {
    gpu.memset_async(out, 0xFF, rows * n * 2, 0)
}

fn locks_zero(gpu: &dyn GpuBackend, lin: &MarlinLinear) -> Result<bool> {
    let w = lin.weight();
    let b = download(
        gpu,
        w.locks,
        spark_model::port::marlin::config::locks_bytes(w.sms),
    )?;
    Ok(b.iter().all(|&v| v == 0))
}

/// Every exported entry point once, on `[2560 x 2560]` (a shape every configuration divides),
/// `rows` = the entry's full row tile.
pub fn entries(gpu: &dyn GpuBackend, r: &mut Lcg, s: &W4Scratch, t: &mut Tally) -> Result<()> {
    let (n, k) = (2560usize, 2560usize);
    let mut launched = 0usize;
    for zp in [false, true] {
        let l = random_logical(r, n, k, zp);
        let w_nk = l.dequant_nk();
        let lin = build(gpu, &l, 0)?;
        let x_host = r.bf16s(64 * k, 1.0);
        let x = upload_u16(gpu, &x_host)?;
        let out = gpu.alloc(64 * n * 2)?;
        let w = lin.weight().clone();
        for key in kernels::instances()
            .into_iter()
            .filter(|key| key.btype == btype_of(zp))
        {
            let rows = key.m_rows();
            let cfg = ThreadConfig {
                thread_k: key.thread_k_blocks * 16,
                thread_n: key.thread_n_blocks * 16,
                num_threads: key.threads,
            };
            let chunk = Chunk {
                row_offset: 0,
                rows,
                thread_m_blocks: key.thread_m_blocks,
                m_block_size_8: key.m_block_size_8,
                cfg,
                blocks: w.sms,
                smem: w.max_shared_mem,
            };
            poison(gpu, out, rows, n)?;
            lin.run_plan(gpu, x, out, &[chunk], s, 0)?;
            let got = download_u16(gpu, out, rows * n)?;
            let (worst, bad) = oracle(&got, &x_host, rows, &w_nk, n, k, STRIDE);
            t.gate(
                bad == 0,
                &format!(
                    "{} m={rows} n={n} k={k} worst={worst:.3} bad={bad}",
                    entry_name(&key)
                ),
            );
            launched += 1;
        }
        t.gate(
            locks_zero(gpu, &lin)?,
            &format!("locks zero after every {} launch", btype_of(zp).label()),
        );
        lin.release_unboxed(gpu)?;
        gpu.free(x)?;
        gpu.free(out)?;
    }
    t.gate(
        launched == kernels::ENTRY_COUNT,
        &format!("entry coverage {launched}/{}", kernels::ENTRY_COUNT),
    );
    Ok(())
}

/// The decode path on one shape and type: numerics at the tested M values (with a planted control),
/// row invariance, its vLLM contrast, and a five-run byte repeat.
pub fn decode(
    gpu: &dyn GpuBackend,
    r: &mut Lcg,
    s: &W4Scratch,
    t: &mut Tally,
    n: usize,
    k: usize,
    zp: bool,
) -> Result<()> {
    let l = random_logical(r, n, k, zp);
    let w_nk = l.dequant_nk();
    let lin = build(gpu, &l, 0)?;
    let x_host = r.bf16s(64 * k, 1.0);
    let x = upload_u16(gpu, &x_host)?;
    let out = gpu.alloc(64 * n * 2)?;
    let label = btype_of(zp).label();

    for m in [1usize, 8, 16, 17, 32, 64] {
        poison(gpu, out, m, n)?;
        lin.decode(gpu, x, m, out, s, 0)?;
        let got = download_u16(gpu, out, m * n)?;
        let (worst, bad) = oracle(&got, &x_host, m, &w_nk, n, k, STRIDE);
        let name = plan_name(&lin, m, DecodePolicy::RowInvariant)?;
        t.gate(
            bad == 0,
            &format!("decode {name} m={m} n={n} k={k} worst={worst:.3} bad={bad}"),
        );
        if m == 1 {
            // Planted control: the reference without the last group of inputs must miss the bound.
            let mut cut = w_nk.clone();
            for c in 0..n {
                cut[c * k + k - 128..(c + 1) * k].fill(0);
            }
            let (_, cbad) = oracle(&got, &x_host, 1, &cut, n, k, STRIDE);
            t.control(
                cbad > 0,
                &format!("decode {label} [{n}x{k}] against a weight missing its last group"),
            );
        }
    }

    // Row invariance: row r at every M equals row r at M = 64 and at M = 1.
    lin.decode(gpu, x, 64, out, s, 0)?;
    let full = download_u16(gpu, out, 64 * n)?;
    for m in [1usize, 2, 3, 8, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64] {
        lin.decode(gpu, x, m, out, s, 0)?;
        let got = download_u16(gpu, out, m * n)?;
        let diff = (0..m)
            .filter(|&row| got[row * n..(row + 1) * n] != full[row * n..(row + 1) * n])
            .count();
        let name = plan_name(&lin, m, DecodePolicy::RowInvariant)?;
        t.gate(
            diff == 0,
            &format!("row-invariant {name} m={m} n={n} k={k} rows_differing={diff}"),
        );
    }
    for row in [0usize, 5, 31, 63] {
        lin.decode(gpu, x.offset(row * k * 2), 1, out, s, 0)?;
        let got = download_u16(gpu, out, n)?;
        t.gate(
            got[..] == full[row * n..(row + 1) * n],
            &format!("row-invariant {label} [{n}x{k}] row {row} alone equals row {row} of M=64"),
        );
    }
    // The contrast that motivates the decode policy: vLLM's selector at M = 1 and M = 17.
    lin.run_rows(gpu, x, 1, out, s, DecodePolicy::Vllm, 0)?;
    let a = download_u16(gpu, out, n)?;
    lin.run_rows(gpu, x, 17, out, s, DecodePolicy::Vllm, 0)?;
    let b = download_u16(gpu, out, n)?;
    let cols = a.iter().zip(&b).filter(|(p, q)| p != q).count();
    println!(
        "info vllm-selector {label} [{n}x{k}] row 0 at M=1 ({}) vs M=17 ({}): {cols} of {n} outputs differ",
        plan_name(&lin, 1, DecodePolicy::Vllm)?,
        plan_name(&lin, 17, DecodePolicy::Vllm)?
    );

    // Five-run byte repeat at M = 8.
    let mut first: Option<Vec<u16>> = None;
    let mut same = true;
    for _ in 0..5 {
        lin.decode(gpu, x, 8, out, s, 0)?;
        let got = download_u16(gpu, out, 8 * n)?;
        match &first {
            None => first = Some(got),
            Some(f) => same &= *f == got,
        }
    }
    t.gate(
        same,
        &format!("repeat {label} [{n}x{k}] m=8 five runs byte-equal"),
    );
    t.gate(
        locks_zero(gpu, &lin)?,
        &format!("locks zero after decode {label} [{n}x{k}]"),
    );
    lin.release_unboxed(gpu)?;
    gpu.free(x)?;
    gpu.free(out)?;
    Ok(())
}

/// The prompt path: `prefill` at M past the decode limit, vLLM's split, against the oracle.
pub fn prefill(
    gpu: &dyn GpuBackend,
    r: &mut Lcg,
    s: &W4Scratch,
    t: &mut Tally,
    n: usize,
    k: usize,
    zp: bool,
) -> Result<()> {
    let l = random_logical(r, n, k, zp);
    let w_nk = l.dequant_nk();
    let lin = build(gpu, &l, 0)?;
    for m in [65usize, 200, 513] {
        let x_host = r.bf16s(m * k, 1.0);
        let x = upload_u16(gpu, &x_host)?;
        let out = gpu.alloc(m * n * 2)?;
        poison(gpu, out, m, n)?;
        let took = lin.prefill(gpu, x, m, out, s, 0)?;
        ensure!(took, "marlin prefill declined at m={m} with a full scratch");
        let got = download_u16(gpu, out, m * n)?;
        let (worst, bad) = oracle(&got, &x_host, m, &w_nk, n, k, 31);
        let name = plan_name(&lin, m, DecodePolicy::Vllm)?;
        t.gate(
            bad == 0,
            &format!("prefill {name} m={m} n={n} k={k} worst={worst:.3} bad={bad}"),
        );
        gpu.free(x)?;
        gpu.free(out)?;
    }
    t.gate(
        locks_zero(gpu, &lin)?,
        &format!("locks zero after prefill [{n}x{k}]"),
    );
    lin.release_unboxed(gpu)?;
    Ok(())
}
