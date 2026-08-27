// SPDX-License-Identifier: AGPL-3.0-only
//! Slice 6 REMAINDER — the integrated GLM-5.3-Flash KDA layer, end to end.
//!
//! Three parts:
//!
//! 1. **Synthetic full-layer oracle** — LCG weights at production geometry, four regimes
//!    (decode T=1 · short prefill · ragged prefill · ragged prefill -> decode), GPU vs a CPU
//!    reference built from `glm5next_kda_ref` primitives, cross-checked against
//!    `glm5next_kda_ref::kda_reference_layer` on the sub-path that function covers.
//! 2. **Real layer-0 checkpoint oracle** — the 16-tensor 262.7 MiB packet extracted from
//!    `LibertAIDAI/GLM-5.3-Flash-NVFP4` @ `9e0d74e3`, shard 1/120, versus a golden produced by
//!    the genuine `transformers` 5.16.1 `Glm5NextTextLinearAttention` on the same bytes. Nothing
//!    in a KDA block is quantised, so this golden IS the production numerics.
//! 3. **Pad-corruption regression (Slice 5)** — a ragged prefill whose padded tail is poisoned,
//!    followed by a decode. Correct prefill outputs are NOT sufficient evidence: the original
//!    bug left every output right and the carried state off by 1.623e13.
//!
//! Floors are reported separately so a residual is never confused with a rounding budget:
//!   * **A** HF/reference math — CPU reference in fp32 vs the fp32 HF golden.
//!   * **B** bf16 activation/input — the bf16 HF golden vs the fp32 HF golden.
//!   * **C** dequantised-real-weight — **N/A for KDA**: nothing in the block is quantised.
//!   * **D** GPU kernel residual — GPU vs a CPU reference fed the SAME bf16-rounded values.
//!   * **E** complete integrated-layer residual — GPU final output vs the bf16 HF golden.
//!
//! 🪤 Atlas's L2 writes **bf16** (fused on decode, `l2_norm_bf16` on prefill); HF normalises in
//! **fp32 inside** the KDA kernel. Atlas therefore carries one extra bf16 rounding on q|k that
//! HF does not, and every downstream stage inherits it. That is a contract difference, not an
//! error — it is why floor D is measured against a CPU reference that reproduces Atlas's exact
//! dtype ladder rather than against HF.
//!
//!   KDA_LAYER0_PACKET=/path/to/layer0.safetensors \
//!   cargo run -p spark-model --release --example kda_layer_microtest \
//!       --features cuda,gpu-examples

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use half::bf16;
use serde_json::Value;
use spark_model::layers::glm5next_kda::{
    KdaLayer, KdaLayerConfig, KdaLayerDims, KdaLayerKernels, KdaLayerWeights, KdaStages,
};
use spark_model::layers::glm5next_kda_ref as kref;
use spark_model::weight_map::DenseWeight;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

#[path = "common/golden.rs"]
mod golden;

static GOLDEN: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| golden::load("crates/spark-model/src/layers/glm5next_kda_ref/kda_layer_golden.json", "gen_kda_layer_golden.py"));

/// Chunk width for the prefill scan. `C = 64` needs 81 920 B of shared memory and the backend
/// has no `cuFuncSetAttribute` opt-in, so 32 is the ceiling at D = 128 (blocker 12). Slice 5
/// verified identical results at C = 2..32, and HF runs C = 64 — agreement across both is part
/// of what this test shows.
const CHUNK: usize = 32;

// ───────────────────────────────────────────────────────────────────── plumbing

fn up_f32(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn up_bf16(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d
        .iter()
        .flat_map(|x| bf16::from_f32(*x).to_bits().to_le_bytes())
        .collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn down_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
fn down_bf16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}

struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 40) as f32) / ((1u32 << 24) as f32)) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.u()).collect()
    }
    fn scaled(&mut self, n: usize, s: f32) -> Vec<f32> {
        (0..n).map(|_| self.u() * s).collect()
    }
}

fn r(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}
/// Rounding that the `pure` (floor-A) pass switches off. Weights are BF16 *on disk* — that is
/// the checkpoint, not a rounding — so only INTERMEDIATE values are affected.
fn rq(x: f32, pure: bool) -> f32 {
    if pure { x } else { r(x) }
}
fn round_bf16(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| r(*x)).collect()
}
fn sample(v: &[f32], stride: usize) -> Vec<f32> {
    v.iter().step_by(stride).copied().collect()
}
fn maxabs(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "len {} vs {}", a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((*x as f64 - *y as f64).abs()))
}
fn checksum(s: &[f32]) -> f64 {
    s.iter()
        .enumerate()
        .map(|(i, v)| *v as f64 * (i as f64 + 1.0))
        .sum()
}

// ─────────────────────────────────────────────────────────── safetensors packet

/// Minimal safetensors reader. The packet holds BF16 and F32 only; both are returned as f32
/// (BF16 -> f32 is exact, so nothing is lost and the same buffer can be re-rounded on upload).
fn read_packet(path: &str) -> Result<BTreeMap<String, (Vec<usize>, Vec<f32>)>> {
    let raw = std::fs::read(path).with_context(|| format!("reading {path}"))?;
    let hn = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
    let hdr: Value = serde_json::from_slice(&raw[8..8 + hn])?;
    let base = 8 + hn;
    let mut out = BTreeMap::new();
    for (k, m) in hdr.as_object().unwrap() {
        if k == "__metadata__" {
            continue;
        }
        let shape: Vec<usize> = m["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();
        let a = m["data_offsets"][0].as_u64().unwrap() as usize + base;
        let b = m["data_offsets"][1].as_u64().unwrap() as usize + base;
        let bytes = &raw[a..b];
        let data: Vec<f32> = match m["dtype"].as_str().unwrap() {
            "BF16" => bytes
                .chunks_exact(2)
                .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
            "F32" => bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            d => bail!("unexpected dtype {d} for {k} — a KDA block must be BF16 / F32 only"),
        };
        out.insert(k.clone(), (shape, data));
    }
    Ok(out)
}

// ────────────────────────────────────────────────── CPU reference, Atlas's ladder

/// Host weights, already bf16-rounded exactly as they sit in the checkpoint.
struct Wts {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    conv: Vec<f32>,
    f_a: Vec<f32>,
    f_b: Vec<f32>,
    dt_bias: Vec<f32>,
    a_log: Vec<f32>,
    b: Vec<f32>,
    g_a: Vec<f32>,
    g_b: Vec<f32>,
    o_norm: Vec<f32>,
    o: Vec<f32>,
}

#[derive(Clone, Copy)]
struct Dims {
    hid: usize,
    h: usize,
    d: usize,
    ks: usize,
}
impl Dims {
    fn qkv(&self) -> usize {
        self.h * self.d
    }
    fn conv_dim(&self) -> usize {
        3 * self.qkv()
    }
    fn qk(&self) -> usize {
        2 * self.qkv()
    }
}

/// `C = A @ B^T`, mirroring `dense_gemm_bf16`: bf16 operands, strictly ascending fp32
/// accumulation, bf16 result.
fn gemm_p(a: &[f32], w: &[f32], m: usize, n: usize, k: usize, pure: bool) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += a[row * k + i] * w[col * k + i];
            }
            out[row * n + col] = rq(acc, pure);
        }
    }
    out
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// L2 over `head_dim` rows of the first `qk` channels, `1/sqrt(sum + eps)`, result bf16 —
/// what BOTH Atlas conv paths produce. V is left alone.
fn l2_qk_bf16(x: &mut [f32], dm: Dims, eps: f32, row_stride: usize, pure: bool) {
    for tok in x.chunks_exact_mut(row_stride) {
        for grp in tok[..dm.qk()].chunks_exact_mut(dm.d) {
            let inv = 1.0 / (grp.iter().map(|a| a * a).sum::<f32>() + eps).sqrt();
            for a in grp.iter_mut() {
                *a = rq(*a * inv, pure);
            }
        }
    }
}

/// Decode conv: shift-left 4-slot state, fp32 accumulate, SiLU, fused L2 on q|k, bf16 out.
fn cpu_conv_decode(
    state4: &mut [f32],
    tok: &[f32],
    w: &[f32],
    dm: Dims,
    eps: f32,
    pure: bool,
) -> Vec<f32> {
    let (dim, ks) = (dm.conv_dim(), dm.ks);
    let mut out = vec![0.0f32; dim];
    for ch in 0..dim {
        let s = &mut state4[ch * ks..(ch + 1) * ks];
        for i in 0..ks - 1 {
            s[i] = s[i + 1];
        }
        s[ks - 1] = tok[ch];
        let mut acc = 0.0f32;
        for k in 0..ks {
            acc += s[k] * w[ch * ks + k];
        }
        out[ch] = silu(acc);
    }
    l2_qk_bf16(&mut out, dm, eps, dim, pure);
    for x in out[dm.qk()..].iter_mut() {
        *x = rq(*x, pure);
    }
    out
}

/// Prefill conv: same window, SiLU only, bf16 out; L2 is a separate pass.
fn cpu_conv_prefill(
    state4: &mut [f32],
    toks: &[f32],
    w: &[f32],
    dm: Dims,
    t: usize,
    pure: bool,
) -> Vec<f32> {
    let (dim, ks) = (dm.conv_dim(), dm.ks);
    let mut out = vec![0.0f32; t * dim];
    for ch in 0..dim {
        let mut s = [0.0f32; 4];
        s[..ks].copy_from_slice(&state4[ch * ks..(ch + 1) * ks]);
        for (tt, o) in out.chunks_exact_mut(dim).enumerate() {
            let nv = toks[tt * dim + ch];
            for i in 0..ks - 1 {
                s[i] = s[i + 1];
            }
            s[ks - 1] = nv;
            let mut acc = 0.0f32;
            for k in 0..ks {
                acc += s[k] * w[ch * ks + k];
            }
            o[ch] = rq(silu(acc), pure);
        }
        state4[ch * ks..(ch + 1) * ks].copy_from_slice(&s[..ks]);
    }
    out
}

/// Every observable stage, on the host, in fp32 values that carry Atlas's exact bf16 ladder.
struct Stages {
    qkv_proj: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    gate: Vec<f32>,
    beta: Vec<f32>,
    core: Vec<f32>,
    state: Vec<f32>,
    out_gate: Vec<f32>,
    o_norm: Vec<f32>,
    final_out: Vec<f32>,
    conv_state: Vec<f32>,
    /// Independent cross-check, populated on the pure-fp32 prefill pass only:
    /// `max_abs(final_out, glm5next_kda_ref::kda_reference_layer(...))`. That function is the
    /// handoff's named full-layer oracle and it drives the core through the **recurrent**
    /// formulation, so agreeing with it also re-proves chunk == recurrent at production
    /// geometry on the real weights.
    ref_layer_delta: Option<f64>,
}

/// CPU reference for one layer. `decode` selects the fused-conv + recurrent path; otherwise the
/// prefill-conv + separate-L2 + chunked path. `pure_f32 = true` drops every intermediate bf16
/// rounding, which is what floor A is measured with.
#[allow(clippy::too_many_arguments)]
fn cpu_layer(
    w: &Wts,
    dm: Dims,
    cfg: KdaLayerConfig,
    hidden: &[f32],
    t: usize,
    conv_state4: &mut Vec<f32>,
    state: &mut Vec<f32>,
    decode: bool,
    chunk: usize,
    pure: bool,
) -> Stages {
    let (hid, qkv, hd) = (dm.hid, dm.qkv(), dm.d);
    let cd = dm.conv_dim();

    let mut qkv_proj = vec![0.0f32; t * cd];
    for (i, ww) in [&w.q, &w.k, &w.v].into_iter().enumerate() {
        let p = gemm_p(hidden, ww, t, qkv, hid, pure);
        for tt in 0..t {
            qkv_proj[tt * cd + i * qkv..tt * cd + (i + 1) * qkv]
                .copy_from_slice(&p[tt * qkv..(tt + 1) * qkv]);
        }
    }

    let mut pre_l2 = Vec::new();
    let conv_out = if decode {
        assert_eq!(t, 1);
        cpu_conv_decode(conv_state4, &qkv_proj, &w.conv, dm, cfg.l2_eps, pure)
    } else {
        let c = cpu_conv_prefill(conv_state4, &qkv_proj, &w.conv, dm, t, pure);
        pre_l2 = c.clone();
        let mut c = c;
        l2_qk_bf16(&mut c, dm, cfg.l2_eps, cd, pure);
        c
    };
    let pick_from = |src: &[f32], off: usize| -> Vec<f32> {
        (0..t)
            .flat_map(|tt| src[tt * cd + off..tt * cd + off + qkv].to_vec())
            .collect()
    };
    let (q, k, v) = (
        pick_from(&conv_out, 0),
        pick_from(&conv_out, qkv),
        pick_from(&conv_out, 2 * qkv),
    );

    let f_a = gemm_p(hidden, &w.f_a, t, hd, hid, pure);
    let g_raw = gemm_p(&f_a, &w.f_b, t, qkv, hd, pure);
    let kd = kref::KdaDims {
        hidden: hid,
        heads: dm.h,
        head_dim: hd,
        tokens: t,
    };
    let gate = kref::bounded_gate(&g_raw, &w.dt_bias, &w.a_log, kd, cfg.lower_bound);
    let beta: Vec<f32> = gemm_p(hidden, &w.b, t, dm.h, hid, pure)
        .iter()
        .map(|x| sigmoid(*x))
        .collect();

    let core = if decode {
        kref::kda_recurrent_prenorm(&q, &k, &v, &gate, &beta, kd, state)
    } else {
        kref::kda_chunked_prenorm(&q, &k, &v, &gate, &beta, kd, chunk, state)
    };

    let g_a = gemm_p(hidden, &w.g_a, t, hd, hid, pure);
    let out_gate = gemm_p(&g_a, &w.g_b, t, qkv, hd, pure);
    let raw_norm = kref::rms_norm_gated(&core, &w.o_norm, &out_gate, hd, cfg.rms_eps);
    let o_norm: Vec<f32> = if pure {
        raw_norm
    } else {
        round_bf16(&raw_norm)
    };
    let final_out = gemm_p(&o_norm, &w.o, t, hid, qkv, pure);

    // Cross-check against the handoff's named oracle. Only on the pure-fp32 prefill pass:
    // `kda_reference_layer` normalises q/k itself, so it needs the PRE-L2 conv output, and on
    // bf16 values a second normalisation would restore rounding (the Slice-4 hazard).
    let ref_layer_delta = if pure && !decode {
        let mut st2 = vec![0.0f32; dm.h * hd * hd];
        let rw = kref::KdaWeights {
            w_f_a: &w.f_a,
            w_f_b: &w.f_b,
            dt_bias: &w.dt_bias,
            a_log: &w.a_log,
            w_b: &w.b,
            w_g_a: &w.g_a,
            w_g_b: &w.g_b,
            o_norm_w: &w.o_norm,
            w_o: &w.o,
        };
        let out = kref::kda_reference_layer(
            hidden,
            &pick_from(&pre_l2, 0),
            &pick_from(&pre_l2, qkv),
            &pick_from(&pre_l2, 2 * qkv),
            &rw,
            kd,
            cfg.lower_bound,
            cfg.rms_eps,
            &mut st2,
        );
        Some(maxabs(&final_out, &out))
    } else {
        None
    };

    Stages {
        qkv_proj,
        q,
        k,
        v,
        gate,
        beta,
        core,
        state: state.clone(),
        out_gate,
        o_norm,
        final_out,
        conv_state: conv_state4.clone(),
        ref_layer_delta,
    }
}

// ─────────────────────────────────────────────────────────────── GPU harness

struct Gpu<'a> {
    g: &'a dyn GpuBackend,
    layer: KdaLayer,
    dm: Dims,
}

struct Bufs {
    s: KdaStages,
    lowrank: DevicePtr,
    beta_bf: DevicePtr,
    gc: DevicePtr,
    u: DevicePtr,
    w: DevicePtr,
}

impl Gpu<'_> {
    fn alloc(&self, t: usize, t_pad: usize) -> Result<Bufs> {
        let g = self.g;
        let (qkv, cd, hd, h) = (self.dm.qkv(), self.dm.conv_dim(), self.dm.d, self.dm.h);
        let n = t_pad * qkv;
        Ok(Bufs {
            s: KdaStages {
                qkv_parts: g.alloc(3 * t * qkv * 2)?,
                qkv_proj: g.alloc(t * cd * 2)?,
                conv_out: g.alloc(t * cd * 2)?,
                q_f32: g.alloc(n * 4)?,
                k_f32: g.alloc(n * 4)?,
                v_f32: g.alloc(n * 4)?,
                gate: g.alloc(t_pad * qkv * 4)?,
                beta: g.alloc(t_pad * h * 4)?,
                core: g.alloc(n * 4)?,
                state: DevicePtr::NULL, // caller supplies
                g_raw: g.alloc(t * qkv * 2)?,
                out_gate: g.alloc(t * qkv * 2)?,
                o_norm_out: g.alloc(t * qkv * 2)?,
                final_out: g.alloc(t * self.dm.hid * 2)?,
                conv_state: DevicePtr::NULL, // caller supplies
                t_pad,
            },
            lowrank: g.alloc(t * hd * 2)?,
            beta_bf: g.alloc(t * h * 2)?,
            gc: g.alloc(n * 4)?,
            u: g.alloc(n * 4)?,
            w: g.alloc(n * 4)?,
        })
    }

    /// `gate` and `beta` are read by `kda_chunk_prepare` at padded positions, so the pad tail of
    /// both is zeroed here — the kernel's own `T` guard covers it, and zeroing makes any guard
    /// failure show up in q/k/v (which the regression deliberately poisons) rather than here.
    fn run(
        &self,
        hidden: &[f32],
        t: usize,
        decode: bool,
        conv_state4: &[f32],
        state: &[f32],
        pad_fill: f32,
    ) -> Result<(Stages, Vec<f32>, Vec<f32>)> {
        let g = self.g;
        let (qkv, cd, h, hid) = (self.dm.qkv(), self.dm.conv_dim(), self.dm.h, self.dm.hid);
        let t_pad = if decode { 1 } else { t.div_ceil(CHUNK) * CHUNK };
        let mut b = self.alloc(t, t_pad)?;
        let dstate = up_f32(g, state)?;
        let dconv = up_f32(g, conv_state4)?;
        b.s.state = dstate;
        b.s.conv_state = dconv;
        g.copy_h2d(&vec![0u8; t_pad * qkv * 4], b.s.gate)?;
        g.copy_h2d(&vec![0u8; t_pad * h * 4], b.s.beta)?;
        let dh = up_bf16(g, hidden)?;

        if decode {
            self.layer.decode(g, dh, &b.s, b.lowrank, b.beta_bf, 0)?;
        } else {
            self.layer.prefill(
                g, dh, t, &b.s, b.lowrank, b.beta_bf, b.gc, b.u, b.w, pad_fill, 0,
            )?;
        }
        g.synchronize(0)?;

        let conv_out = down_bf16(g, b.s.conv_out, t * cd)?;
        let pick = |off: usize| -> Vec<f32> {
            (0..t)
                .flat_map(|tt| conv_out[tt * cd + off..tt * cd + off + qkv].to_vec())
                .collect()
        };
        let core_full = down_f32(g, b.s.core, t_pad * qkv)?;
        let st = Stages {
            qkv_proj: down_bf16(g, b.s.qkv_proj, t * cd)?,
            q: pick(0),
            k: pick(qkv),
            v: pick(2 * qkv),
            gate: down_f32(g, b.s.gate, t_pad * qkv)?[..t * qkv].to_vec(),
            beta: down_f32(g, b.s.beta, t_pad * h)?[..t * h].to_vec(),
            core: core_full[..t * qkv].to_vec(),
            state: down_f32(g, b.s.state, h * self.dm.d * self.dm.d)?,
            out_gate: down_bf16(g, b.s.out_gate, t * qkv)?,
            o_norm: down_bf16(g, b.s.o_norm_out, t * qkv)?,
            final_out: down_bf16(g, b.s.final_out, t * hid)?,
            conv_state: down_f32(g, b.s.conv_state, cd * self.dm.ks)?,
            ref_layer_delta: None,
        };
        let (cs, rs) = (st.conv_state.clone(), st.state.clone());
        Ok((st, cs, rs))
    }
}

// ────────────────────────────────────────────────────────────── comparison table

struct Row {
    name: &'static str,
    /// Largest |value| in the fp32 golden sample — without it a residual is unreadable.
    mag: f64,
    /// **A** — pure-fp32 reference vs the fp32 golden: HF/reference math only.
    floor_a: f64,
    /// **B** — bf16 golden vs fp32 golden: the activation-dtype budget.
    floor_b: f64,
    /// **D** — GPU vs a CPU reference on Atlas's exact bf16 ladder: kernel residual only.
    floor_d: f64,
    gpu_vs_bf16: f64,
    gpu_vs_f32: f64,
    ck_rel: f64,
}

fn ck_rel(a: f64, b: f64) -> f64 {
    let d = a.abs().max(b.abs()).max(1e-30);
    (a - b).abs() / d
}

#[allow(clippy::too_many_arguments)]
fn row(
    name: &'static str,
    gpu: &[f32],
    cpu: &[f32],
    pure: Option<&[f32]>,
    gold: &Value,
    key: &str,
) -> Result<Row> {
    let get = |dt: &str| -> Result<(Vec<f32>, f64, usize, usize)> {
        let e = &gold[dt][key];
        if e.is_null() {
            bail!("golden {dt} is missing stage {key}");
        }
        let data: Vec<f32> = e["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        Ok((
            data,
            e["ck"].as_f64().unwrap(),
            e["n"].as_u64().unwrap() as usize,
            e["stride"].as_u64().unwrap() as usize,
        ))
    };
    let (gbf, ck_bf, n, stride) = get("bf16")?;
    let (gf32, _, n2, s2) = get("f32")?;
    if n != gpu.len() || n2 != gpu.len() {
        bail!("stage {key}: golden n={n} but GPU produced {}", gpu.len());
    }
    if stride != s2 {
        bail!("stage {key}: bf16/f32 goldens disagree on stride ({stride} vs {s2})");
    }
    let sg = sample(gpu, stride);
    Ok(Row {
        name,
        mag: gf32.iter().fold(0.0f64, |m, x| m.max((*x as f64).abs())),
        floor_a: match pure {
            Some(p) => maxabs(&sample(p, stride), &gf32),
            None => f64::NAN,
        },
        floor_b: maxabs(&gbf, &gf32),
        floor_d: maxabs(gpu, cpu),
        gpu_vs_bf16: maxabs(&sg, &gbf),
        gpu_vs_f32: maxabs(&sg, &gf32),
        ck_rel: ck_rel(checksum(gpu), ck_bf),
    })
}

fn print_table(title: &str, rows: &[Row], rows_ref: Option<f64>) {
    println!("\n  {title}");
    println!(
        "    {:<18} {:>10} {:>10} {:>10} {:>10} {:>11} {:>10} {:>9}",
        "stage", "|max|", "A:ref", "B:bf16", "D:kernel", "GPUvsHFbf16", "GPUvsHFf32", "ck_rel"
    );
    for r in rows {
        println!(
            "    {:<18} {:>10.3e} {:>10.3e} {:>10.3e} {:>10.3e} {:>11.3e} {:>10.3e} {:>9.2e}",
            r.name, r.mag, r.floor_a, r.floor_b, r.floor_d, r.gpu_vs_bf16, r.gpu_vs_f32, r.ck_rel
        );
    }
    println!("    (C: dequantised-real-weight — N/A, nothing in a KDA block is quantised)");
    if let Some(d) = rows_ref {
        println!(
            "    kda_reference_layer cross-check (recurrent formulation, fp32): max_abs {d:.3e}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────── main

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let v: Value = serde_json::from_str(GOLDEN)?;
    let f = &v["fixture"];
    let dm = Dims {
        hid: f["hidden"].as_u64().unwrap() as usize,
        h: f["heads"].as_u64().unwrap() as usize,
        d: f["head_dim"].as_u64().unwrap() as usize,
        ks: f["kernel"].as_u64().unwrap() as usize,
    };
    let cfg = KdaLayerConfig {
        lower_bound: f["lower_bound"].as_f64().unwrap() as f32,
        rms_eps: f["rms_eps"].as_f64().unwrap() as f32,
        l2_eps: f["l2_eps"].as_f64().unwrap() as f32,
    };

    println!("GLM-5.3-Flash KDA layer — integrated Atlas layer vs HF 5.16.1");
    println!("  checkpoint {} layer {}", f["checkpoint"], f["layer"]);
    println!(
        "  hidden={} heads={} head_dim={} conv_dim={} kernel={} act={} o_norm_act={}",
        dm.hid,
        dm.h,
        dm.d,
        dm.conv_dim(),
        dm.ks,
        f["hidden_act"],
        f["o_norm_act"]
    );
    println!(
        "  READ from config: gate_lower_bound={} rms_norm_eps={:e} (never defaulted)",
        cfg.lower_bound, cfg.rms_eps
    );
    println!(
        "  Atlas chunk C={CHUNK} (smem ceiling), HF chunk C={}",
        f["hf_chunk"]
    );

    // LCG parity with the generator, or the whole fixture is a different fixture.
    let probe: Vec<f32> = v["lcg_probe"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    if Lcg(0x5EED_1A70)
        .vec(probe.len())
        .iter()
        .zip(&probe)
        .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        bail!("LCG mismatch with the generator");
    }
    println!("  LCG parity with the generator: ok");

    let dims = KdaLayerDims {
        hidden: dm.hid,
        heads: dm.h,
        head_dim: dm.d,
        conv_kernel: dm.ks,
        chunk: CHUNK,
    };
    let kernels = KdaLayerKernels::resolve(gpu)?;
    println!("  all 13 kernel entry points resolved (no fallback path)");

    let mut ok = true;

    // ── PART 1 — synthetic full-layer oracle ─────────────────────────────────
    println!("\n=== PART 1 — synthetic full-layer oracle (LCG weights, production geometry) ===");
    let mut wr = Lcg(0xA11A_5000);
    let wts = Wts {
        q: round_bf16(&wr.scaled(dm.qkv() * dm.hid, 0.02)),
        k: round_bf16(&wr.scaled(dm.qkv() * dm.hid, 0.02)),
        v: round_bf16(&wr.scaled(dm.qkv() * dm.hid, 0.02)),
        conv: round_bf16(&wr.scaled(dm.conv_dim() * dm.ks, 0.5)),
        f_a: round_bf16(&wr.scaled(dm.d * dm.hid, 0.02)),
        f_b: round_bf16(&wr.scaled(dm.qkv() * dm.d, 0.05)),
        dt_bias: wr.scaled(dm.qkv(), 0.5),
        a_log: wr.scaled(dm.h, 0.5),
        b: round_bf16(&wr.scaled(dm.h * dm.hid, 0.02)),
        g_a: round_bf16(&wr.scaled(dm.d * dm.hid, 0.02)),
        g_b: round_bf16(&wr.scaled(dm.qkv() * dm.d, 0.05)),
        o_norm: round_bf16(&wr.scaled(dm.d, 1.0)),
        o: round_bf16(&wr.scaled(dm.hid * dm.qkv(), 0.02)),
    };
    ok &= run_suite(gpu, &backend, dims, cfg, dm, &wts, &kernels, None)?;

    // ── PART 2 — real layer-0 checkpoint oracle ──────────────────────────────
    let path = std::env::var("KDA_LAYER0_PACKET")
        .unwrap_or_else(|_| "/home/msi1/atlas-scratch/kda-layer0/layer0.safetensors".to_string());
    println!("\n=== PART 2 — real layer-0 checkpoint oracle ===");
    println!("  packet {path}");
    let p = read_packet(&path)?;
    let need = |n: &str| -> Result<&(Vec<usize>, Vec<f32>)> {
        p.get(n).with_context(|| format!("packet is missing {n}"))
    };
    let bytes: usize = p.values().map(|(_, d)| d.len() * 2).sum();
    println!(
        "  {} tensors, {} elements",
        p.len(),
        p.values().map(|(_, d)| d.len()).sum::<usize>()
    );
    let _ = bytes;
    // 🪤 checkpoint conv is [dim, 1, ks]; Atlas wants [dim, ks]. squeeze(1) is a SHAPE fix only —
    // the bytes are already contiguous — but a loader asserting rank 2 rejects the tensor.
    for n in ["q_conv1d", "k_conv1d", "v_conv1d"] {
        let (sh, _) = need(&format!("self_attn.{n}.weight"))?;
        if sh.as_slice() != [dm.qkv(), 1, dm.ks] {
            bail!("{n}: expected [{}, 1, {}], got {sh:?}", dm.qkv(), dm.ks);
        }
    }
    println!(
        "  q/k/v_conv1d are [dim,1,{}] -> squeeze(1) -> concat(q,k,v) -> [{},{}]",
        dm.ks,
        dm.conv_dim(),
        dm.ks
    );
    let mut conv = Vec::with_capacity(dm.conv_dim() * dm.ks);
    for n in ["q_conv1d", "k_conv1d", "v_conv1d"] {
        conv.extend_from_slice(&need(&format!("self_attn.{n}.weight"))?.1);
    }
    let real = Wts {
        q: need("self_attn.q_proj.weight")?.1.clone(),
        k: need("self_attn.k_proj.weight")?.1.clone(),
        v: need("self_attn.v_proj.weight")?.1.clone(),
        conv,
        f_a: need("self_attn.f_a_proj.weight")?.1.clone(),
        f_b: need("self_attn.f_b_proj.weight")?.1.clone(),
        dt_bias: need("self_attn.dt_bias")?.1.clone(),
        a_log: need("self_attn.A_log")?.1.clone(),
        b: need("self_attn.b_proj.weight")?.1.clone(),
        g_a: need("self_attn.g_a_proj.weight")?.1.clone(),
        g_b: need("self_attn.g_b_proj.weight")?.1.clone(),
        o_norm: need("self_attn.o_norm.weight")?.1.clone(),
        o: need("self_attn.o_proj.weight")?.1.clone(),
    };
    ok &= run_suite(gpu, &backend, dims, cfg, dm, &real, &kernels, Some(&v))?;

    println!(
        "\n{}",
        if ok {
            "RESULT: PASS — integrated KDA layer agrees with HF 5.16.1 on the real checkpoint"
        } else {
            "RESULT: FAIL"
        }
    );
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}

/// Upload one weight set and drive all four regimes plus the pad regression.
#[allow(clippy::too_many_arguments)]
fn run_suite(
    gpu: &dyn GpuBackend,
    _backend: &AtlasCudaBackend,
    dims: KdaLayerDims,
    cfg: KdaLayerConfig,
    dm: Dims,
    w: &Wts,
    k: &KdaLayerKernels,
    golden: Option<&Value>,
) -> Result<bool> {
    let dw = |v: &[f32]| -> Result<DenseWeight> {
        Ok(DenseWeight {
            weight: up_bf16(gpu, v)?,
        })
    };
    let weights = KdaLayerWeights {
        q_proj: dw(&w.q)?,
        k_proj: dw(&w.k)?,
        v_proj: dw(&w.v)?,
        conv: dw(&w.conv)?,
        f_a: dw(&w.f_a)?,
        f_b: dw(&w.f_b)?,
        dt_bias: up_f32(gpu, &w.dt_bias)?,
        a_log: up_f32(gpu, &w.a_log)?,
        b_proj: dw(&w.b)?,
        g_a: dw(&w.g_a)?,
        g_b: dw(&w.g_b)?,
        o_norm: dw(&w.o_norm)?,
        o_proj: dw(&w.o)?,
    };
    let layer = KdaLayer::new(dims, cfg, weights, *k)?;
    let g = Gpu { g: gpu, layer, dm };

    let cd = dm.conv_dim();
    let sz_state = dm.h * dm.d * dm.d;
    let mut ok = true;

    // Fixture draw order must match the generator exactly.
    // Returns (bf16-rounded hidden, RAW fp32 hidden, conv state, recurrent state). The raw
    // hidden is what floor A must be fed: the generator's fp32 arm never rounds its input, so
    // handing the pure-fp32 reference a bf16 copy would fold floor B back into floor A.
    let draw = |t: usize| -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut rng = Lcg(0x5EED_1A70);
        let hidden = rng.vec(t.max(8) * dm.hid)[..t * dm.hid].to_vec();
        let conv3 = rng.scaled(cd * (dm.ks - 1), 0.5);
        let rec = rng.scaled(sz_state, 0.1);
        (round_bf16(&hidden), hidden, conv3, rec)
    };
    // HF's 3 slots -> Atlas's 4; slot 0 is shifted out before the conv and never participates.
    let widen = |c3: &[f32]| -> Vec<f32> {
        let mut s = vec![0.0f32; cd * dm.ks];
        for ch in 0..cd {
            for i in 0..dm.ks - 1 {
                s[ch * dm.ks + 1 + i] = c3[ch * (dm.ks - 1) + i];
            }
        }
        s
    };
    let tail3 = |s4: &[f32]| -> Vec<f32> {
        (0..cd)
            .flat_map(|ch| (1..dm.ks).map(move |i| (ch, i)))
            .map(|(ch, i)| s4[ch * dm.ks + i])
            .collect()
    };

    for (rname, t, decode) in [
        ("decode1", 1usize, true),
        ("prefill4", 4, false),
        ("prefill7", 7, false),
    ] {
        let (hidden, hidden_f32, c3, rec) = draw(t);
        let cs4 = if decode {
            widen(&c3)
        } else {
            vec![0.0f32; cd * dm.ks]
        };
        let st0 = if decode {
            rec.clone()
        } else {
            vec![0.0f32; sz_state]
        };

        let (gs, _, _) = g.run(&hidden, t, decode, &cs4, &st0, 0.0)?;
        let mut cpu_cs = cs4.clone();
        let mut cpu_st = st0.clone();
        let cs = cpu_layer(
            w,
            dm,
            cfg,
            &hidden,
            t,
            &mut cpu_cs,
            &mut cpu_st,
            decode,
            CHUNK,
            false,
        );
        // Floor A: the same reference with every intermediate bf16 rounding switched off.
        let (mut acs, mut ast) = (cs4.clone(), st0.clone());
        let ca = cpu_layer(
            w,
            dm,
            cfg,
            &hidden_f32,
            t,
            &mut acs,
            &mut ast,
            decode,
            CHUNK,
            true,
        );

        ok &= report(rname, t, dm, &gs, &cs, Some(&ca), golden, rname, &tail3)?;
    }

    // ── ragged prefill -> decode: the regime that carries state across formulations ─────
    {
        let (hidden, hidden_f32, _, _) = draw(7);
        let mut gcs = vec![0.0f32; cd * dm.ks];
        let mut gst = vec![0.0f32; sz_state];
        let (gp, cs_out, st_out) = g.run(&hidden, 7, false, &gcs, &gst, 0.0)?;
        gcs = cs_out;
        gst = st_out;
        let mut ccs = vec![0.0f32; cd * dm.ks];
        let mut cst = vec![0.0f32; sz_state];
        let cp = cpu_layer(
            w, dm, cfg, &hidden, 7, &mut ccs, &mut cst, false, CHUNK, false,
        );
        let (mut acs, mut ast) = (vec![0.0f32; cd * dm.ks], vec![0.0f32; sz_state]);
        let ap = cpu_layer(
            w,
            dm,
            cfg,
            &hidden_f32,
            7,
            &mut acs,
            &mut ast,
            false,
            CHUNK,
            true,
        );

        let mut h2_f32 = Lcg(0xD3C0_DE01).vec(8 * dm.hid);
        h2_f32.truncate(dm.hid);
        let h2 = round_bf16(&h2_f32);
        let (gd, _, _) = g.run(&h2, 1, true, &gcs, &gst, 0.0)?;
        let cd_st = cpu_layer(w, dm, cfg, &h2, 1, &mut ccs, &mut cst, true, CHUNK, false);
        let ad = cpu_layer(
            w, dm, cfg, &h2_f32, 1, &mut acs, &mut ast, true, CHUNK, true,
        );

        ok &= report(
            "prefill7_decode1 (prefill leg)",
            7,
            dm,
            &gp,
            &cp,
            Some(&ap),
            golden,
            "prefill7_decode1",
            &tail3,
        )?;
        // The prefill leg's stages are stored under a `prefill_` prefix in the golden.
        ok &= report_decode_leg(
            "prefill7_decode1 (decode leg)",
            dm,
            &gd,
            &cd_st,
            Some(&ad),
            golden,
            &tail3,
        )?;

        // ── PAD-CORRUPTION REGRESSION (Slice 5) ────────────────────────────────
        // 7 real tokens, chunk 32 -> 25 padded positions. Fill them with 7.5 and demand the
        // CARRIED STATE be bit-identical, not merely the outputs: the original bug produced
        // correct outputs with the state off by 1.623e13.
        let (pg, pcs, pst) = g.run(
            &hidden,
            7,
            false,
            &vec![0.0f32; cd * dm.ks],
            &vec![0.0f32; sz_state],
            7.5,
        )?;
        let d_out = maxabs(&pg.final_out, &gp.final_out);
        let d_state = maxabs(&pst, &gst);
        let d_conv = maxabs(&pcs, &gcs);
        let d_core = maxabs(&pg.core, &gp.core);
        // …and then decode from the poisoned-prefill state, because that is where the original
        // bug actually surfaced.
        let (pd, _, _) = g.run(&h2, 1, true, &pcs, &pst, 0.0)?;
        let d_dec = maxabs(&pd.final_out, &gd.final_out);
        let clean =
            d_out == 0.0 && d_state == 0.0 && d_conv == 0.0 && d_core == 0.0 && d_dec == 0.0;
        println!(
            "\n  PAD-CORRUPTION REGRESSION — T=7 real, {} padded positions filled with 7.5",
            32 - 7
        );
        println!("    prefill out delta   {d_out:.3e}");
        println!("    KDA core delta      {d_core:.3e}");
        println!("    CARRIED STATE delta {d_state:.3e}   <- the one the Slice-5 bug broke");
        println!("    conv state delta    {d_conv:.3e}");
        println!("    NEXT decode delta   {d_dec:.3e}   <- where the bug actually surfaced");
        println!(
            "    [{}]",
            if clean {
                "ok, kernels self-guard past T"
            } else {
                "FAIL"
            }
        );
        ok &= clean;
    }

    Ok(ok)
}

#[allow(clippy::too_many_arguments)]
fn report(
    title: &str,
    t: usize,
    dm: Dims,
    gs: &Stages,
    cs: &Stages,
    ca: Option<&Stages>,
    golden: Option<&Value>,
    regime: &str,
    tail3: &dyn Fn(&[f32]) -> Vec<f32>,
) -> Result<bool> {
    let prefix = if regime == "prefill7_decode1" {
        "prefill_"
    } else {
        ""
    };
    stage_report(title, t, dm, gs, cs, ca, golden, regime, prefix, tail3)
}

#[allow(clippy::too_many_arguments)]
fn report_decode_leg(
    title: &str,
    dm: Dims,
    gs: &Stages,
    cs: &Stages,
    ca: Option<&Stages>,
    golden: Option<&Value>,
    tail3: &dyn Fn(&[f32]) -> Vec<f32>,
) -> Result<bool> {
    stage_report(
        title,
        1,
        dm,
        gs,
        cs,
        ca,
        golden,
        "prefill7_decode1",
        "",
        tail3,
    )
}

#[allow(clippy::too_many_arguments)]
fn stage_report(
    title: &str,
    t: usize,
    dm: Dims,
    gs: &Stages,
    cs: &Stages,
    ca: Option<&Stages>,
    golden: Option<&Value>,
    regime: &str,
    prefix: &str,
    tail3: &dyn Fn(&[f32]) -> Vec<f32>,
) -> Result<bool> {
    let g_tail = tail3(&gs.conv_state);
    let c_tail = tail3(&cs.conv_state);
    let a_tail = ca.map(|a| tail3(&a.conv_state));

    let Some(v) = golden else {
        // Synthetic run: only GPU-vs-CPU (floor D) exists — there is no HF golden for LCG weights.
        let d = [
            ("qkv_proj", maxabs(&gs.qkv_proj, &cs.qkv_proj)),
            ("conv+L2 q", maxabs(&gs.q, &cs.q)),
            ("conv+L2 k", maxabs(&gs.k, &cs.k)),
            ("conv v", maxabs(&gs.v, &cs.v)),
            ("gate", maxabs(&gs.gate, &cs.gate)),
            ("beta", maxabs(&gs.beta, &cs.beta)),
            ("kda core", maxabs(&gs.core, &cs.core)),
            ("recurrent state", maxabs(&gs.state, &cs.state)),
            ("out_gate", maxabs(&gs.out_gate, &cs.out_gate)),
            ("o_norm", maxabs(&gs.o_norm, &cs.o_norm)),
            ("o_proj / final", maxabs(&gs.final_out, &cs.final_out)),
            ("conv state", maxabs(&g_tail, &c_tail)),
        ];
        println!("\n  {title}  (T={t}) — floor D only, no HF golden for synthetic weights");
        let mut worst = 0.0f64;
        for (n, e) in d {
            println!("    {n:<22} D:GPUvsCPU {e:>11.3e}");
            worst = worst.max(e);
        }
        let pass = worst <= 5.0e-2;
        println!(
            "    worst floor-D residual {worst:.3e}  [{}]",
            if pass { "ok" } else { "FAIL" }
        );
        return Ok(pass);
    };

    let gold = &v["regimes"];
    let sel = |dt: &str| -> Value { gold[format!("{dt}__{regime}")].clone() };
    let bundle = serde_json::json!({ "bf16": sel("bf16"), "f32": sel("f32") });
    let key = |n: &str| format!("{prefix}{n}");

    let rows = vec![
        row(
            "qkv_proj",
            &gs.qkv_proj,
            &cs.qkv_proj,
            ca.map(|a| a.qkv_proj.as_slice()),
            &bundle,
            &key("qkv_proj"),
        )?,
        row(
            "conv+L2 q",
            &gs.q,
            &cs.q,
            ca.map(|a| a.q.as_slice()),
            &bundle,
            &key("q_l2"),
        )?,
        row(
            "conv+L2 k",
            &gs.k,
            &cs.k,
            ca.map(|a| a.k.as_slice()),
            &bundle,
            &key("k_l2"),
        )?,
        row(
            "conv v",
            &gs.v,
            &cs.v,
            ca.map(|a| a.v.as_slice()),
            &bundle,
            &key("v_raw"),
        )?,
        row(
            "gate",
            &gs.gate,
            &cs.gate,
            ca.map(|a| a.gate.as_slice()),
            &bundle,
            &key("gate"),
        )?,
        row(
            "beta",
            &gs.beta,
            &cs.beta,
            ca.map(|a| a.beta.as_slice()),
            &bundle,
            &key("beta"),
        )?,
        row(
            "kda core",
            &gs.core,
            &cs.core,
            ca.map(|a| a.core.as_slice()),
            &bundle,
            &key("core"),
        )?,
        row(
            "recurrent state",
            &gs.state,
            &cs.state,
            ca.map(|a| a.state.as_slice()),
            &bundle,
            &key("state"),
        )?,
        row(
            "out_gate",
            &gs.out_gate,
            &cs.out_gate,
            ca.map(|a| a.out_gate.as_slice()),
            &bundle,
            &key("out_gate"),
        )?,
        row(
            "o_norm",
            &gs.o_norm,
            &cs.o_norm,
            ca.map(|a| a.o_norm.as_slice()),
            &bundle,
            &key("o_norm_out"),
        )?,
        row(
            "o_proj / final",
            &gs.final_out,
            &cs.final_out,
            ca.map(|a| a.final_out.as_slice()),
            &bundle,
            &key("final_out"),
        )?,
        row(
            "conv state",
            &g_tail,
            &c_tail,
            a_tail.as_deref(),
            &bundle,
            &key("conv_state"),
        )?,
    ];
    print_table(
        &format!("{title}  (T={t})"),
        &rows,
        ca.and_then(|a| a.ref_layer_delta),
    );

    // E = the integrated-layer residual at the layer output, judged against floor B there.
    let fin = rows.iter().find(|r| r.name == "o_proj / final").unwrap();
    let stt = rows.iter().find(|r| r.name == "recurrent state").unwrap();
    let ratio = fin.gpu_vs_bf16 / fin.floor_b.max(1e-30);
    println!(
        "    E: final_out GPU vs HF-bf16 {:.3e}   floor B {:.3e}   E/B {:.2}",
        fin.gpu_vs_bf16, fin.floor_b, ratio
    );
    let _ = dm;
    // The layer output is bf16, so the smallest representable step near |x|~0.04 is ~2e-4;
    // demand the residual stay within a few multiples of the bf16 floor, and the carried state
    // — which no later stage can correct — within its own.
    let pass = fin.gpu_vs_bf16 <= (fin.floor_b * 8.0).max(1.0e-3)
        && stt.gpu_vs_bf16 <= (stt.floor_b * 8.0).max(1.0e-3);
    if !pass {
        println!("    [FAIL] residual exceeds 8x the bf16 floor");
    }
    Ok(pass)
}
