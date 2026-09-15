// SPDX-License-Identifier: AGPL-3.0-only
//! Native GDN layer-arm legs: the two `Exl3GdnWeights` funnels every
//! `Qwen3SsmLayer` GDN site (M=1 decode, batched decode, multi-seq decode,
//! prefill) calls, at qwen4_exp's shapes, synthetic K=4 MUL1 weights, over a
//! production-sized stage (the default 2048-row slabs, sized exactly as the
//! loader sizes them: in 6144, out 10240), vs a reconstruct -> f64 reference.
//!
//! Leg I.1 — `out_proj_linear` [6144 -> 2560]: m in {1, 3, 8} exercise the
//! small-row tier (at this shape the GEMV heuristic declines for K=4, so it
//! is the f32-C split-K GEMM through the GEMV fallthrough — exactly what
//! decode runs), 64 / 300 the f16-C GEMM tier with in-place egress. The
//! destination is written exactly `m` rows (row m keeps its sentinel) and a
//! second call over a DIFFERENT stream is admitted (the cross-stream fence
//! path of the shared launch state) with identical output.
//!
//! Leg I.2 — `in_proj_linear`: in_proj_qkv [2560 -> 10240] + in_proj_z
//! [2560 -> 6144] as the shared-A pair with STRIDED egress into the fused
//! `[m, 16384]` `[Q|K|V|Z]` arena (the step-2 arena decision), m in {1, 2, 4,
//! 8, 64, 300}. Both column blocks are gated against the f64 truth, row m
//! (past the end) must keep its poison, and — per block in ISOLATION through
//! the same strided destinations — the columns outside the block must stay
//! poison while the block itself is bit-identical to the pair's output.

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use half::{bf16, f16};
use spark_model::layers::Exl3GdnWeights;
use spark_model::layers::ops::{
    EXL3_DENSE_STAGE_ROWS_DEFAULT, Exl3DenseOut, Exl3DenseStage, Exl3DenseWeight, Exl3LaunchState,
    exl3_dense_linear,
};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::truth::{cb_enum, decode_what_f64, truth_matmul};
use crate::util::{Ctx, DevWeight, GEMV_MAX_Z, GEMV_REL_RMS, Lcg, as_bytes, gate_leg, up};

const K_BITS: u32 = 4;
const CB: u32 = 2;
/// qwen4_exp GDN geometry: hidden 2560, conv (Q|K|V) 10240, value (Z) 6144.
const H: usize = 2560;
const CONV: usize = 10240;
const VAL: usize = 6144;
const QKVZ: usize = CONV + VAL;
const M_MAX: usize = 300;
const SENTINEL: u16 = 0x449A; // bf16 1232.0 — never a plausible output

/// One synthetic packed weight + its f64 reconstruction.
struct W {
    suh: Vec<u16>,
    svh: Vec<u16>,
    what: Vec<f64>,
    dev: DevWeight,
    w: Exl3DenseWeight,
    k: usize,
    n: usize,
}

impl W {
    fn generate(g: &dyn GpuBackend, rng: &mut Lcg, k: usize, n: usize) -> Result<Self> {
        let trellis: Vec<u16> = (0..(k / 16) * (n / 16) * 16 * K_BITS as usize)
            .map(|_| rng.u16())
            .collect();
        let suh: Vec<u16> = (0..k).map(|_| rng.scale_f16()).collect();
        let svh: Vec<u16> = (0..n).map(|_| rng.scale_f16()).collect();
        let what = decode_what_f64(&trellis, k, n, K_BITS, cb_enum(CB));
        let dev = DevWeight::upload(g, &trellis, &suh, &svh)?;
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

    /// f64 truth for the given rows of `a_f16` (f16 bits `[_, k]`).
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

/// Rows worth checking for `m`: all of them up to 64, else first/last plus
/// a few interior rows (the 2048-row production stage never batches here).
fn check_rows(m: usize) -> Vec<usize> {
    if m <= 64 {
        (0..m).collect()
    } else {
        [0, 1, 127, 128, 255, 256, m - 1]
            .into_iter()
            .filter(|&r| r < m)
            .collect()
    }
}

fn down_bits(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

/// Gather `rows` of a `[_, ld]` bf16 matrix, columns `c0..c0+n`, as f64.
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

/// True iff columns `c0..c1` of rows `0..m` (row stride `ld`) are all poison.
fn block_is_poison(bits: &[u16], m: usize, ld: usize, c0: usize, c1: usize) -> bool {
    (0..m).all(|r| {
        bits[r * ld + c0..r * ld + c1]
            .iter()
            .all(|&b| b == SENTINEL)
    })
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

/// BF16 gaussian activations `[M_MAX, k]` plus their exact f16 image (the
/// ingress conversion is exact) for the reference.
fn activations(rng: &mut Lcg, k: usize) -> (Vec<u16>, Vec<u16>) {
    let a_bf16: Vec<u16> = (0..M_MAX * k)
        .map(|_| bf16::from_f32(rng.gauss()).to_bits())
        .collect();
    let a_f16 = a_bf16
        .iter()
        .map(|&b| f16::from_f32(bf16::from_bits(b).to_f32()).to_bits())
        .collect();
    (a_bf16, a_f16)
}

pub fn run(ctx: &Ctx, rng: &mut Lcg) -> Result<bool> {
    let g = ctx.g;
    let stream = g.default_stream();
    let mut ok = true;

    // The loader's stage: in = max(hidden, value_dim), out = max(hidden,
    // conv_dim, value_dim) at the default row count.
    let launch = Arc::new(Exl3LaunchState::new(g)?);
    let stage = Arc::new(Exl3DenseStage::new_with_fp32(
        g,
        launch.clone(),
        EXL3_DENSE_STAGE_ROWS_DEFAULT,
        VAL.max(H),
        CONV.max(H).max(VAL),
        H, // out_proj is fp32-C on the GEMM tier, exactly like serving
    )?);

    let out = W::generate(g, rng, VAL, H)?;
    let qkv = W::generate(g, rng, H, CONV)?;
    let z = W::generate(g, rng, H, VAL)?;
    let gdn = Exl3GdnWeights {
        in_proj_qkv: qkv.w,
        in_proj_z: z.w,
        out_proj: out.w,
        stage,
    };
    anyhow::ensure!(gdn.qkvz_row_elems() == QKVZ);

    ok &= leg_out_proj(g, stream, &gdn, &out, rng)?;
    ok &= leg_in_proj(g, stream, &gdn, &qkv, &z, rng)?;

    for w in [&out, &qkv, &z] {
        w.dev.free(g);
    }
    gdn.stage.release(g)?;
    launch.release(g)?;
    Ok(ok)
}

/// Leg I.1 (see the module docs).
fn leg_out_proj(
    g: &dyn GpuBackend,
    stream: u64,
    gdn: &Exl3GdnWeights,
    out: &W,
    rng: &mut Lcg,
) -> Result<bool> {
    let mut ok = true;
    let (a_bf16, a_f16) = activations(rng, VAL);
    let a_d = up(g, &as_bytes(&a_bf16))?;
    // One extra sentinel row past M_MAX: the arm must write exactly m rows.
    let dst = g.alloc((M_MAX + 1) * H * 2)?;
    let sentinel: Vec<u8> = as_bytes(&vec![SENTINEL; (M_MAX + 1) * H]);
    let side_stream = g.create_stream()?;

    for m in [1usize, 3, 8, 64, M_MAX] {
        g.copy_h2d(&sentinel, dst)?;
        gdn.out_proj_linear(g, a_d, dst, m, stream)?;
        g.synchronize(stream)?;
        let bits = down_bits(g, dst, (m + 1) * H)?;
        let rows = check_rows(m);
        let y = gather(&bits, &rows, H, 0, H);
        let y64 = out.truth_rows(&a_f16, &rows);
        ok &= gate_leg(
            &format!(
                "dense-gdn out_proj [{VAL}->{H}] m={m} via Exl3GdnWeights::out_proj_linear ({})",
                if m <= 8 {
                    "small-row tier"
                } else {
                    "gemm f16-C tier"
                }
            ),
            &y,
            &y64,
            GEMV_REL_RMS,
            GEMV_MAX_Z,
        );
        let tail_ok = bits[m * H..].iter().all(|&b| b == SENTINEL);
        println!("dense-gdn out_proj m={m}: row {m} (past the end) untouched = {tail_ok}");
        ok &= tail_ok;

        // Same call on a second stream: the launch state's fence path
        // (stream change -> stream_wait_event) must admit it and the
        // output must be bit-identical to the first stream's.
        let side = g.alloc(m * H * 2)?;
        gdn.out_proj_linear(g, a_d, side, m, side_stream)?;
        g.synchronize(side_stream)?;
        let same = down_bits(g, side, m * H)? == bits[..m * H];
        println!("dense-gdn out_proj m={m}: second stream (fenced) bit-identical = {same}");
        ok &= same;
        g.free(side).ok();
        // And back on the default stream (fence the other way).
        gdn.out_proj_linear(g, a_d, dst, m, stream)?;
        g.synchronize(stream)?;
    }

    // (No stream destroy on the trait; the example process exits shortly.)
    for p in [a_d, dst] {
        g.free(p).ok();
    }
    Ok(ok)
}

/// Leg I.2 (see the module docs).
fn leg_in_proj(
    g: &dyn GpuBackend,
    stream: u64,
    gdn: &Exl3GdnWeights,
    qkv: &W,
    z: &W,
    rng: &mut Lcg,
) -> Result<bool> {
    let mut ok = true;
    let (a_bf16, a_f16) = activations(rng, H);
    let a_d = up(g, &as_bytes(&a_bf16))?;
    // [M_MAX + 1, QKVZ] arena, poison-filled before every call.
    let arena = g.alloc((M_MAX + 1) * QKVZ * 2)?;
    let poison: Vec<u8> = as_bytes(&vec![SENTINEL; (M_MAX + 1) * QKVZ]);
    let qkv_out = Exl3DenseOut::strided(arena, QKVZ);
    let z_out = Exl3DenseOut::strided(arena.offset(CONV * 2), QKVZ);

    for m in [1usize, 2, 4, 8, 64, M_MAX] {
        let tier = if m <= 8 {
            "small-row tier"
        } else {
            "gemm f16-C tier"
        };
        g.copy_h2d(&poison, arena)?;
        gdn.in_proj_linear(g, a_d, arena, m, stream)?;
        g.synchronize(stream)?;
        let pair = down_bits(g, arena, (m + 1) * QKVZ)?;
        let rows = check_rows(m);
        ok &= gate_leg(
            &format!(
                "dense-gdn in_proj pair, qkv block [{H}->{CONV}] (ld={QKVZ}) m={m} via \
                 Exl3GdnWeights::in_proj_linear ({tier})"
            ),
            &gather(&pair, &rows, QKVZ, 0, CONV),
            &qkv.truth_rows(&a_f16, &rows),
            GEMV_REL_RMS,
            GEMV_MAX_Z,
        );
        ok &= gate_leg(
            &format!(
                "dense-gdn in_proj pair, z block [{H}->{VAL}] (ld={QKVZ} @+{CONV}) m={m} via \
                 Exl3GdnWeights::in_proj_linear ({tier})"
            ),
            &gather(&pair, &rows, QKVZ, CONV, VAL),
            &z.truth_rows(&a_f16, &rows),
            GEMV_REL_RMS,
            GEMV_MAX_Z,
        );
        let tail_ok = pair[m * QKVZ..].iter().all(|&b| b == SENTINEL);
        println!("dense-gdn in_proj pair m={m}: row {m} (past the end) untouched = {tail_ok}");
        ok &= tail_ok;

        // Each block in isolation through the SAME strided destination the
        // funnel uses: the other block's columns must stay poison, and the
        // block must be bit-identical to what the pair wrote.
        g.copy_h2d(&poison, arena)?;
        exl3_dense_linear(g, &qkv.w, a_d, qkv_out, m, &gdn.stage, stream)?;
        g.synchronize(stream)?;
        let only_qkv = down_bits(g, arena, (m + 1) * QKVZ)?;
        let z_cols_poison = block_is_poison(&only_qkv, m, QKVZ, CONV, QKVZ);
        let qkv_same =
            (0..m).all(|r| only_qkv[r * QKVZ..r * QKVZ + CONV] == pair[r * QKVZ..r * QKVZ + CONV]);
        println!(
            "dense-gdn in_proj qkv ALONE m={m}: z columns untouched = {z_cols_poison}, qkv \
             block bit-identical to the pair = {qkv_same}"
        );
        ok &= z_cols_poison && qkv_same;

        g.copy_h2d(&poison, arena)?;
        exl3_dense_linear(g, &z.w, a_d, z_out, m, &gdn.stage, stream)?;
        g.synchronize(stream)?;
        let only_z = down_bits(g, arena, (m + 1) * QKVZ)?;
        let qkv_cols_poison = block_is_poison(&only_z, m, QKVZ, 0, CONV);
        let z_same = (0..m).all(|r| {
            only_z[r * QKVZ + CONV..(r + 1) * QKVZ] == pair[r * QKVZ + CONV..(r + 1) * QKVZ]
        });
        println!(
            "dense-gdn in_proj z ALONE m={m}: qkv columns untouched = {qkv_cols_poison}, z \
             block bit-identical to the pair = {z_same}"
        );
        ok &= qkv_cols_poison && z_same;
    }

    for m in [1usize, 64] {
        let us = time_launch(g, stream, &|| gdn.in_proj_linear(g, a_d, arena, m, stream))?;
        println!(
            "dense-gdn timing in_proj pair (qkv+z, strided ld={QKVZ}) m={m}: {us:.1} us per \
             Exl3GdnWeights::in_proj_linear (ingress + 2 matmuls + 2 egress, host-timed)"
        );
    }

    for p in [a_d, arena] {
        g.free(p).ok();
    }
    Ok(ok)
}
