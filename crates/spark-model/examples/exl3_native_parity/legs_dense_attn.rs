// SPDX-License-Identifier: AGPL-3.0-only
//! Native attention q/k/v/o layer-arm leg: `Exl3AttnWeights::{proj_linear,
//! kv_linear, qkv_linear, o_proj_linear}` — the EXACT funnels every
//! `Qwen3AttentionLayer` site (decode Q / K+V / O, multi-seq decode Q|K|V
//! rows + O, paged and cache-skip prefill) calls — at qwen4_exp's attention
//! shapes (q [2560 -> 12288] gated, k/v [2560 -> 512], o [6144 -> 2560]),
//! synthetic K=4 MUL1 weights, over a production-sized stage (2048 rows x
//! in 6144 x out 12288, as the loader sizes it), vs a reconstruct -> f64
//! reference.
//!
//! Per m in {1, 4, 8, 64, 300} (small-row tier at <= 8, row-batched f16-C
//! GEMM above):
//!  * Q: raw `[Q|gate]`-interleaved row vs the reference (the checkpoint's
//!    column order), then the CONSUMER CONTRACT: the production
//!    `deinterleave_qg` over that output must equal (a) the reference's Q and
//!    gate halves through the same per-head un-interleave and (b) an EXACT
//!    permutation of the raw GPU row (bitwise).
//!  * K/V: the decode pair funnel into two contiguous blocks.
//!  * Multi-seq: the q/k/v triple into pitched `[m, Q|K|V]` rows (+ a pad
//!    column block that must stay byte-untouched; row m sentinel).
//!  * O: contiguous `[m, 2560]`.
//! Then host-timed decode (m=1) cost per site: Q, K+V, O.

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use half::{bf16, f16};
use spark_model::layers::ops::{
    EXL3_DENSE_STAGE_ROWS_DEFAULT, Exl3DenseOut, Exl3DenseStage, Exl3DenseWeight, Exl3LaunchState,
    deinterleave_qg,
};
use spark_model::layers::{AttnProj, Exl3AttnWeights};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::truth::{cb_enum, decode_what_f64, truth_matmul};
use crate::util::{Ctx, DevWeight, GEMV_MAX_Z, GEMV_REL_RMS, Lcg, as_bytes, gate_leg, up};

const K_BITS: u32 = 4;
const CB: u32 = 2;
const H: usize = 2560;
const NQ: usize = 24;
const HD: usize = 256;
const Q_N: usize = NQ * HD * 2; // gated: [Q|gate] interleaved per head
const KV_N: usize = 512;
const O_IN: usize = NQ * HD;
const M_MAX: usize = 300;
/// Multi-seq row: [Q | K | V] + a 128-element pad block that must stay untouched.
const PAD: usize = 128;
const LD: usize = Q_N + 2 * KV_N + PAD;
const SENTINEL: u16 = 0x449A; // bf16 1232.0

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
    fn truth(&self, a_f16: &[u16], m: usize) -> Vec<f64> {
        truth_matmul(
            &a_f16[..m * self.k],
            &self.suh,
            &self.svh,
            &self.what,
            m,
            self.k,
            self.n,
            1.0,
        )
    }
}

fn down_bits(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

/// Rows `0..m`, columns `c0..c0+n` of a `[_, ld]` bf16 matrix, as f64.
fn gather(bits: &[u16], m: usize, ld: usize, c0: usize, n: usize) -> Vec<f64> {
    let mut y = Vec::with_capacity(m * n);
    for r in 0..m {
        y.extend(
            bits[r * ld + c0..r * ld + c0 + n]
                .iter()
                .map(|&b| bf16::from_bits(b).to_f64()),
        );
    }
    y
}

/// The `deinterleave_qg` index map: output column i of a row reads raw
/// column `perm(i)` (per-head `[q(HD) | gate(HD)]` -> `[Q_all | gate_all]`).
fn deinterleave_src(i: usize) -> usize {
    let q_total = NQ * HD;
    if i < q_total {
        (i / HD) * 2 * HD + i % HD
    } else {
        let gi = i - q_total;
        (gi / HD) * 2 * HD + HD + gi % HD
    }
}

fn bf16_rows(rng: &mut Lcg, m: usize, k: usize) -> (Vec<u16>, Vec<u16>) {
    let a_bf16: Vec<u16> = (0..m * k)
        .map(|_| bf16::from_f32(rng.gauss()).to_bits())
        .collect();
    let a_f16: Vec<u16> = a_bf16
        .iter()
        .map(|&b| f16::from_f32(bf16::from_bits(b).to_f32()).to_bits())
        .collect();
    (a_bf16, a_f16)
}

pub fn run(ctx: &Ctx, rng: &mut Lcg) -> Result<bool> {
    let g = ctx.g;
    let stream = g.default_stream();
    let mut ok = true;

    let launch = Arc::new(Exl3LaunchState::new(g)?);
    let stage = Arc::new(Exl3DenseStage::new_with_fp32(
        g,
        launch.clone(),
        EXL3_DENSE_STAGE_ROWS_DEFAULT,
        O_IN,
        Q_N,
        H, // o_proj is fp32-C on the GEMM tier, exactly like serving
    )?);
    let q = W::generate(g, rng, H, Q_N)?;
    let k = W::generate(g, rng, H, KV_N)?;
    let v = W::generate(g, rng, H, KV_N)?;
    let o = W::generate(g, rng, O_IN, H)?;
    let attn = Exl3AttnWeights {
        q_proj: q.w,
        k_proj: k.w,
        v_proj: v.w,
        o_proj: o.w,
        stage,
    };
    let deint_k = g.kernel("ssm_preprocess", "deinterleave_qg")?;

    // Activations: `normed` [M_MAX, H] for q/k/v, `attn_out` [M_MAX, O_IN] for o.
    let (normed_bf16, normed_f16) = bf16_rows(rng, M_MAX, H);
    let (attn_bf16, attn_f16) = bf16_rows(rng, M_MAX, O_IN);
    let normed_d = up(g, &as_bytes(&normed_bf16))?;
    let attn_d = up(g, &as_bytes(&attn_bf16))?;
    // Destinations with one sentinel row past M_MAX.
    let q_buf = g.alloc((M_MAX + 1) * Q_N * 2)?;
    let k_buf = g.alloc((M_MAX + 1) * KV_N * 2)?;
    let v_buf = g.alloc((M_MAX + 1) * KV_N * 2)?;
    let arena = g.alloc((M_MAX + 1) * LD * 2)?;
    let o_buf = g.alloc((M_MAX + 1) * H * 2)?;
    let fill =
        |p: DevicePtr, n: usize| -> Result<()> { g.copy_h2d(&as_bytes(&vec![SENTINEL; n]), p) };

    for m in [1usize, 4, 8, 64, M_MAX] {
        let tier = if m <= 8 {
            "small-row tier"
        } else {
            "gemm f16-C tier"
        };
        let q64 = q.truth(&normed_f16, m);
        let k64 = k.truth(&normed_f16, m);
        let v64 = v.truth(&normed_f16, m);
        let o64 = o.truth(&attn_f16, m);

        // ── Q (decode / prefill arm): raw interleaved row, then the consumer
        // contract through the production deinterleave.
        fill(q_buf, (M_MAX + 1) * Q_N)?;
        attn.proj_linear(
            g,
            AttnProj::Q,
            normed_d,
            Exl3DenseOut::contiguous(q_buf),
            m,
            stream,
        )?;
        g.synchronize(stream)?;
        let raw = down_bits(g, q_buf, (m + 1) * Q_N)?;
        ok &= gate_leg(
            &format!("dense-attn q_proj [{H}->{Q_N}] m={m} raw [Q|gate] via proj_linear ({tier})"),
            &gather(&raw, m, Q_N, 0, Q_N),
            &q64,
            GEMV_REL_RMS,
            GEMV_MAX_Z,
        );
        let tail_ok = raw[m * Q_N..].iter().all(|&b| b == SENTINEL);
        println!("dense-attn q_proj m={m}: row {m} (past the end) untouched = {tail_ok}");
        ok &= tail_ok;
        deinterleave_qg(
            g, deint_k, q_buf, m as u32, NQ as u32, HD as u32, Q_N as u32, stream,
        )?;
        g.synchronize(stream)?;
        let deint = down_bits(g, q_buf, m * Q_N)?;
        let mut q64_deint = vec![0f64; m * Q_N];
        let mut perm_exact = true;
        for r in 0..m {
            for i in 0..Q_N {
                let src = deinterleave_src(i);
                q64_deint[r * Q_N + i] = q64[r * Q_N + src];
                perm_exact &= deint[r * Q_N + i] == raw[r * Q_N + src];
            }
        }
        ok &= gate_leg(
            &format!(
                "dense-attn q_proj m={m} -> deinterleave_qg [Q_all|gate_all] vs reference halves"
            ),
            &gather(&deint, m, Q_N, 0, Q_N),
            &q64_deint,
            GEMV_REL_RMS,
            GEMV_MAX_Z,
        );
        println!(
            "dense-attn q_proj m={m}: deinterleaved output is an exact bitwise permutation of \
             the raw row = {perm_exact}"
        );
        ok &= perm_exact;

        // ── K+V (decode arm): one ingress, two contiguous blocks.
        fill(k_buf, (M_MAX + 1) * KV_N)?;
        fill(v_buf, (M_MAX + 1) * KV_N)?;
        attn.kv_linear(g, normed_d, k_buf, v_buf, m, stream)?;
        g.synchronize(stream)?;
        let kb = down_bits(g, k_buf, (m + 1) * KV_N)?;
        let vb = down_bits(g, v_buf, (m + 1) * KV_N)?;
        ok &= gate_leg(
            &format!("dense-attn k_proj [{H}->{KV_N}] m={m} via kv_linear ({tier})"),
            &gather(&kb, m, KV_N, 0, KV_N),
            &k64,
            GEMV_REL_RMS,
            GEMV_MAX_Z,
        );
        ok &= gate_leg(
            &format!("dense-attn v_proj [{H}->{KV_N}] m={m} via kv_linear ({tier})"),
            &gather(&vb, m, KV_N, 0, KV_N),
            &v64,
            GEMV_REL_RMS,
            GEMV_MAX_Z,
        );
        let tail_ok = kb[m * KV_N..].iter().all(|&b| b == SENTINEL)
            && vb[m * KV_N..].iter().all(|&b| b == SENTINEL);
        println!("dense-attn k/v_proj m={m}: row {m} (past the end) untouched = {tail_ok}");
        ok &= tail_ok;

        // ── Multi-seq arm: q/k/v into pitched [m, Q|K|V|pad] rows.
        fill(arena, (M_MAX + 1) * LD)?;
        attn.qkv_linear(
            g,
            normed_d,
            Exl3DenseOut::strided(arena, LD),
            Exl3DenseOut::strided(arena.offset(Q_N * 2), LD),
            Exl3DenseOut::strided(arena.offset((Q_N + KV_N) * 2), LD),
            m,
            stream,
        )?;
        g.synchronize(stream)?;
        let ab = down_bits(g, arena, (m + 1) * LD)?;
        for (label, c0, n, truth) in [
            ("q", 0usize, Q_N, &q64),
            ("k", Q_N, KV_N, &k64),
            ("v", Q_N + KV_N, KV_N, &v64),
        ] {
            ok &= gate_leg(
                &format!("dense-attn multi-seq {label}_proj m={m} strided ld={LD} via qkv_linear"),
                &gather(&ab, m, LD, c0, n),
                truth,
                GEMV_REL_RMS,
                GEMV_MAX_Z,
            );
        }
        let pad_ok = (0..m).all(|r| {
            ab[r * LD + LD - PAD..(r + 1) * LD]
                .iter()
                .all(|&b| b == SENTINEL)
        }) && ab[m * LD..].iter().all(|&b| b == SENTINEL);
        println!("dense-attn multi-seq m={m}: pad columns + row {m} byte-untouched = {pad_ok}");
        ok &= pad_ok;

        // ── O (decode / multi-seq / prefill arm): contiguous [m, H].
        fill(o_buf, (M_MAX + 1) * H)?;
        attn.o_proj_linear(g, attn_d, o_buf, m, stream)?;
        g.synchronize(stream)?;
        let ob = down_bits(g, o_buf, (m + 1) * H)?;
        ok &= gate_leg(
            &format!("dense-attn o_proj [{O_IN}->{H}] m={m} via o_proj_linear ({tier})"),
            &gather(&ob, m, H, 0, H),
            &o64,
            GEMV_REL_RMS,
            GEMV_MAX_Z,
        );
        let tail_ok = ob[m * H..].iter().all(|&b| b == SENTINEL);
        println!("dense-attn o_proj m={m}: row {m} (past the end) untouched = {tail_ok}");
        ok &= tail_ok;
    }

    // ── Decode-site timing (m=1): host-timed µs per funnel call, the
    // per-layer per-token attention projection cost the arm adds.
    let time = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
        for _ in 0..20 {
            f()?;
        }
        g.synchronize(stream)?;
        let t0 = Instant::now();
        for _ in 0..100 {
            f()?;
        }
        g.synchronize(stream)?;
        Ok(t0.elapsed().as_secs_f64() * 1e6 / 100.0)
    };
    let tq = time(&|| {
        attn.proj_linear(
            g,
            AttnProj::Q,
            normed_d,
            Exl3DenseOut::contiguous(q_buf),
            1,
            stream,
        )
    })?;
    let tkv = time(&|| attn.kv_linear(g, normed_d, k_buf, v_buf, 1, stream))?;
    let to = time(&|| attn.o_proj_linear(g, attn_d, o_buf, 1, stream))?;
    println!(
        "dense-attn decode m=1 host-timed: q_proj {tq:.1} us, k+v_proj {tkv:.1} us, o_proj \
         {to:.1} us (total {:.1} us / layer / token)",
        tq + tkv + to
    );

    for p in [normed_d, attn_d, q_buf, k_buf, v_buf, arena, o_buf] {
        g.free(p).ok();
    }
    for w in [&q, &k, &v, &o] {
        w.dev.free(g);
    }
    attn.stage.release(g)?;
    launch.release(g)?;
    Ok(ok)
}
