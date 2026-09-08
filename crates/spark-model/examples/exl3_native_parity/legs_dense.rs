// SPDX-License-Identifier: AGPL-3.0-only
//! Dense-linear leg: the PRODUCTION `ops::exl3_dense_linear` dispatch
//! (bf16 ingress -> gemv/gemm over the shared `Exl3DenseStage` under an
//! `Exl3LaunchState` section -> bf16 egress) at qwen4_exp's GDN/attention
//! shapes, synthetic K=4 MUL1 weights, vs a reconstruct -> f64 reference
//! (`decode_what_f64` + `truth_matmul`, the legs_moe pattern).
//!
//! Coverage per shape ([2560->10240] qkv, [2560->6144] z, [6144->2560]
//! out/o, [2560->12288] q, [2560->512] k/v): m in {1, 4, 8} (GEMV tier,
//! f32 C), 64 (GEMM tier, one batch, in-place f16 egress), 700 (GEMM tier,
//! 3 row batches at the deliberately small 256-row test stage — checked at
//! the batch boundaries); a negative control (wrong codebook must BLOW the
//! gate) and launch timing at m=1 / m=64. Then the shared-A pair helper
//! writing qkv + z as two STRIDED column blocks of one wider arena row
//! (ld 16384 + 128 sentinel pad, exact-untouched check).
//!
//! Gates: the GEMV-leg pair (rel 8e-3 / z 4e-2) — the bf16 egress rounding
//! (~1.1e-3 rel_rms) is the dominant term over the f16/f32 C kernels.

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use half::{bf16, f16};
use spark_model::layers::ops::{
    Exl3DenseOut, Exl3DenseStage, Exl3DenseWeight, Exl3LaunchState, exl3_dense_linear,
    exl3_dense_linear_shared_a,
};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::truth::{cb_enum, decode_what_f64, truth_matmul};
use crate::util::{Ctx, DevWeight, GEMV_MAX_Z, GEMV_REL_RMS, Lcg, as_bytes, gate_leg, metrics, up};

const K_BITS: u32 = 4;
const CB: u32 = 2;
/// Small on purpose: m=700 must batch (3 launches: 256 + 256 + 188).
const STAGE_ROWS: usize = 256;
const M_MAX: usize = 700;
const SENTINEL: u16 = 0x449A; // bf16 1232.0 — never a plausible output

struct DenseW {
    suh: Vec<u16>,
    svh: Vec<u16>,
    what: Vec<f64>,
    dev: DevWeight,
    w: Exl3DenseWeight,
    k: usize,
    n: usize,
}

impl DenseW {
    fn generate(ctx: &Ctx, rng: &mut Lcg, k: usize, n: usize) -> Result<Self> {
        let trellis: Vec<u16> = (0..(k / 16) * (n / 16) * 16 * K_BITS as usize)
            .map(|_| rng.u16())
            .collect();
        let suh: Vec<u16> = (0..k).map(|_| rng.scale_f16()).collect();
        let svh: Vec<u16> = (0..n).map(|_| rng.scale_f16()).collect();
        let what = decode_what_f64(&trellis, k, n, K_BITS, cb_enum(CB));
        let dev = DevWeight::upload(ctx.g, &trellis, &suh, &svh)?;
        let w = Exl3DenseWeight {
            trellis: dev.trellis,
            suh: dev.suh,
            svh: dev.svh,
            in_dim: k,
            out_dim: n,
            k_bits: K_BITS,
            cb: CB,
        };
        Ok(Self {
            suh,
            svh,
            what,
            dev,
            w,
            k,
            n,
        })
    }

    /// f64 truth for the given rows of `a_f16` (f16 bits `[M_MAX, k]`).
    fn truth_rows(&self, a_f16: &[u16], rows: &[usize]) -> Vec<f64> {
        let mut out = Vec::with_capacity(rows.len() * self.n);
        for &r in rows {
            out.extend(truth_matmul(
                &a_f16[r * self.k..(r + 1) * self.k],
                &self.suh,
                &self.svh,
                &self.what,
                1,
                self.k,
                self.n,
                1.0,
            ));
        }
        out
    }
}

/// Rows worth checking for `m`: all of them up to 64, else the batch
/// boundaries of the 256-row stage + first/last.
fn check_rows(m: usize) -> Vec<usize> {
    if m <= 64 {
        (0..m).collect()
    } else {
        [0, 1, 255, 256, 511, 512, m - 1]
            .into_iter()
            .filter(|&r| r < m)
            .collect()
    }
}

fn down_bf16_bits(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

/// Gather `rows` of a `[m, ld]` bf16 matrix, columns `c0..c0+n`, as f64.
fn gather(bits: &[u16], rows: &[usize], ld: usize, c0: usize, n: usize) -> Vec<f64> {
    let mut y = Vec::with_capacity(rows.len() * n);
    for &r in rows {
        y.extend(
            bits[r * ld + c0..r * ld + c0 + n]
                .iter()
                .map(|&b| bf16::from_bits(b).to_f64()),
        );
    }
    y
}

fn time_launch(g: &dyn GpuBackend, stream: u64, launch: &dyn Fn() -> Result<()>) -> Result<f64> {
    for _ in 0..20 {
        launch()?;
    }
    g.synchronize(stream)?;
    let t0 = Instant::now();
    for _ in 0..100 {
        launch()?;
    }
    g.synchronize(stream)?;
    Ok(t0.elapsed().as_secs_f64() * 1e6 / 100.0)
}

pub fn run(ctx: &Ctx, rng: &mut Lcg) -> Result<bool> {
    let g = ctx.g;
    let stream = g.default_stream();
    let mut ok = true;

    let launch = Arc::new(Exl3LaunchState::new(g)?);
    let stage = Exl3DenseStage::new(g, launch.clone(), STAGE_ROWS, 6144, 12288)?;
    if stage.rows_cap != STAGE_ROWS {
        println!(
            "dense: stage rows overridden to {} (ATLAS_EXL3_DENSE_STAGE_ROWS) — batching \
             boundaries differ from the leg's row picks",
            stage.rows_cap
        );
    }

    // Activations once at the widest k (6144); the k=2560 shapes read the
    // first 2560 columns of each row via a contiguous [M_MAX, k] copy.
    let act_bf16: Vec<u16> = (0..M_MAX * 6144)
        .map(|_| bf16::from_f32(rng.gauss()).to_bits())
        .collect();
    let a_for = |k: usize| -> (Vec<u16>, Vec<u16>) {
        let mut a = Vec::with_capacity(M_MAX * k);
        for r in 0..M_MAX {
            a.extend_from_slice(&act_bf16[r * 6144..r * 6144 + k]);
        }
        let a16 = a
            .iter()
            .map(|&b| f16::from_f32(bf16::from_bits(b).to_f32()).to_bits())
            .collect();
        (a, a16)
    };

    let shapes: [(usize, usize, &str); 5] = [
        (2560, 10240, "gdn in_proj_qkv"),
        (2560, 6144, "gdn in_proj_z"),
        (6144, 2560, "gdn out_proj / attn o_proj"),
        (2560, 12288, "attn q_proj"),
        (2560, 512, "attn k/v_proj"),
    ];
    let dst = g.alloc(M_MAX * 12288 * 2)?;
    let sentinel_row: Vec<u8> = as_bytes(&vec![SENTINEL; M_MAX * 12288]);

    for (k, n, label) in shapes {
        let w = DenseW::generate(ctx, rng, k, n)?;
        let (a_bf16, a_f16) = a_for(k);
        let a_d = up(g, &as_bytes(&a_bf16))?;

        for m in [1usize, 4, 8, 64, M_MAX] {
            g.copy_h2d(&sentinel_row[..m * n * 2], dst)?;
            exl3_dense_linear(
                g,
                &w.w,
                a_d,
                Exl3DenseOut::contiguous(dst),
                m,
                &stage,
                stream,
            )?;
            g.synchronize(stream)?;
            let bits = down_bf16_bits(g, dst, m * n)?;
            let rows = check_rows(m);
            let y = gather(&bits, &rows, n, 0, n);
            let y64 = w.truth_rows(&a_f16, &rows);
            ok &= gate_leg(
                &format!(
                    "dense [{k}->{n}] {label} m={m} ({} rows checked{})",
                    rows.len(),
                    if m > stage.rows_cap {
                        ", row-batched"
                    } else {
                        ""
                    }
                ),
                &y,
                &y64,
                GEMV_REL_RMS,
                GEMV_MAX_Z,
            );
            if m == 1 {
                // Negative control: the wrong codebook must blow the gate.
                let mut bad = w.w;
                bad.cb = 1;
                exl3_dense_linear(
                    g,
                    &bad,
                    a_d,
                    Exl3DenseOut::contiguous(dst),
                    1,
                    &stage,
                    stream,
                )?;
                g.synchronize(stream)?;
                let yb = gather(&down_bf16_bits(g, dst, n)?, &[0], n, 0, n);
                let (rel, _) = metrics(&yb, &y64);
                let blew = rel > GEMV_REL_RMS;
                println!(
                    "dense [{k}->{n}] control (cb=MCG on a MUL1 trellis) rel_rms={rel:.3e} \
                     blows gate={blew}"
                );
                ok &= blew;
            }
        }
        for m in [1usize, 64] {
            let us = time_launch(g, stream, &|| {
                exl3_dense_linear(
                    g,
                    &w.w,
                    a_d,
                    Exl3DenseOut::contiguous(dst),
                    m,
                    &stage,
                    stream,
                )
            })?;
            println!(
                "dense timing [{k}->{n}] m={m}: {us:.1} us per exl3_dense_linear (ingress + \
                 {} + egress, host-timed incl. launch overhead)",
                if m <= 8 { "gemv f32-C" } else { "gemm f16-C" }
            );
        }
        g.free(a_d).ok();
        w.dev.free(g);
    }
    g.free(dst).ok();

    // Shared-A pair into a strided arena: [Q|K|V (10240) | Z (6144) | pad 128].
    let qkv = DenseW::generate(ctx, rng, 2560, 10240)?;
    let z = DenseW::generate(ctx, rng, 2560, 6144)?;
    let (a_bf16, a_f16) = a_for(2560);
    let a_d = up(g, &as_bytes(&a_bf16))?;
    let ld = 10240 + 6144 + 128;
    let arena = g.alloc(M_MAX * ld * 2)?;
    let sentinel_arena: Vec<u8> = as_bytes(&vec![SENTINEL; M_MAX * ld]);
    for m in [1usize, 8, 64, M_MAX] {
        g.copy_h2d(&sentinel_arena[..m * ld * 2], arena)?;
        exl3_dense_linear_shared_a(
            g,
            &[
                (qkv.w, Exl3DenseOut::strided(arena, ld)),
                (z.w, Exl3DenseOut::strided(arena.offset(10240 * 2), ld)),
            ],
            a_d,
            m,
            &stage,
            stream,
        )?;
        g.synchronize(stream)?;
        let bits = down_bf16_bits(g, arena, m * ld)?;
        let rows = check_rows(m);
        ok &= gate_leg(
            &format!("dense shared-A pair, qkv block (strided ld={ld}) m={m}"),
            &gather(&bits, &rows, ld, 0, 10240),
            &qkv.truth_rows(&a_f16, &rows),
            GEMV_REL_RMS,
            GEMV_MAX_Z,
        );
        ok &= gate_leg(
            &format!("dense shared-A pair, z block (strided ld={ld} @+10240) m={m}"),
            &gather(&bits, &rows, ld, 10240, 6144),
            &z.truth_rows(&a_f16, &rows),
            GEMV_REL_RMS,
            GEMV_MAX_Z,
        );
        let pad_ok = (0..m).all(|r| {
            bits[r * ld + 16384..(r + 1) * ld]
                .iter()
                .all(|&b| b == SENTINEL)
        });
        println!("dense shared-A pair m={m}: pad columns untouched = {pad_ok}");
        ok &= pad_ok;
    }
    let us = time_launch(g, stream, &|| {
        exl3_dense_linear_shared_a(
            g,
            &[
                (qkv.w, Exl3DenseOut::strided(arena, ld)),
                (z.w, Exl3DenseOut::strided(arena.offset(10240 * 2), ld)),
            ],
            a_d,
            1,
            &stage,
            stream,
        )
    })?;
    println!("dense timing shared-A pair qkv+z m=1 strided: {us:.1} us per call");

    for p in [a_d, arena] {
        g.free(p).ok();
    }
    qkv.dev.free(g);
    z.dev.free(g);
    stage.release(g)?;
    launch.release(g)?;
    Ok(ok)
}
