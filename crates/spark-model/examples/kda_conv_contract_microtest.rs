// SPDX-License-Identifier: AGPL-3.0-only
//! Slice 6 GATE — conv contract at GLM-5.3-Flash production geometry.
//!
//! Slice 2 classed Atlas's `causal_conv1d_update_l2norm` as REUSE for KDA. It has never been
//! driven at KDA geometry, and it carries hardcoded structural assumptions:
//! `BLOCK = 256`, `head_dim = 128` -> exactly 2 L2 groups per block, and
//! `qk_channels % 256 == 0`. This binds it to HF 5.16.1 BEFORE any layer integration, because
//! "REUSE, launch args only" was an inference, not a measurement.
//!
//! Geometry: `conv_dim = 3*64*128 = 24576`, `qk_channels = 2*64*128 = 16384`, `head_dim = 128`,
//! `kernel = 4`, activation `silu` (from config `hidden_act`, not assumed).
//!
//! ## Two paths, because Atlas has two and they are different kernels
//! * **decode** — `causal_conv1d_update_l2norm`: conv + SiLU + L2, fused.
//! * **prefill** — `causal_conv1d_update_prefill`: conv + SiLU only. L2 must then be applied
//!   separately by `l2_norm_bf16`, over the q|k channels ONLY. Getting this wrong in either
//!   direction (skipped, or applied twice) is the Slice-4 hazard: a second normalisation is
//!   nearly invisible in fp32 and very visible in bf16.
//!
//! ## State-width mapping
//! HF keeps `kernel_size - 1 = 3` slots. Atlas keeps 4 and shifts LEFT before convolving, so
//! the oldest slot is shifted out and never participates:
//! `HF_state[0..3] == Atlas_state[1..4]` pre-shift. The Rust side widens HF's 3 into Atlas's 4.
//!
//!   cargo run -p spark-model --release --example kda_conv_contract_microtest \
//!       --features cuda,gpu-examples

use anyhow::{Result, bail};
use half::bf16;
use serde_json::Value;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

#[path = "common/golden.rs"]
mod golden;

static GOLDEN: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    golden::load(
        "crates/spark-model/src/layers/glm5next_kda_ref/kda_conv_golden.json",
        "gen_kda_conv_golden.py",
    )
});

/// bf16 has ~8 mantissa bits; the conv runs in bf16 in production, so the achievable floor is
/// set by input rounding, not by the kernel. Every figure is reported next to a CPU reference
/// fed the SAME bf16-rounded values, so kernel error is separated from that floor.
const MAX_ABS_BF16: f64 = 6.0e-3;

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
fn down_bf16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}
fn down_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
fn arr(v: &Value, name: &str) -> Vec<f32> {
    v["outputs"][name]["data"]
        .as_array()
        .unwrap_or_else(|| panic!("missing {name}"))
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
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
fn sample(s: &[f32], stride: usize) -> Vec<f32> {
    s.iter().step_by(stride).copied().collect()
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
}

/// CPU conv reference fed the same bf16-rounded inputs the GPU sees. Establishes the floor.
fn cpu_conv_silu(
    state4: &[f32],
    tok: &[f32],
    w: &[f32],
    dim: usize,
    ks: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut st = state4.to_vec();
    let mut out = vec![0.0f32; dim];
    for ch in 0..dim {
        let s = &mut st[ch * ks..(ch + 1) * ks];
        for i in 0..ks - 1 {
            s[i] = s[i + 1];
        }
        s[ks - 1] = bf16::from_f32(tok[ch]).to_f32();
        let mut acc = 0.0f32;
        for k in 0..ks {
            acc += s[k] * bf16::from_f32(w[ch * ks + k]).to_f32();
        }
        out[ch] = acc * (1.0 / (1.0 + (-acc).exp()));
    }
    (out, st)
}

fn l2_rows(x: &mut [f32], d: usize, upto: usize) {
    for r in x[..upto].chunks_exact_mut(d) {
        let inv = 1.0 / (r.iter().map(|a| a * a).sum::<f32>() + 1e-6).sqrt();
        for a in r.iter_mut() {
            *a *= inv;
        }
    }
}

fn main() -> Result<()> {
    let g = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &g;
    let v: Value = serde_json::from_str(&GOLDEN)?;
    let f = &v["fixture"];
    let dim = f["conv_dim"].as_u64().unwrap() as usize;
    let qk = f["qk_channels"].as_u64().unwrap() as usize;
    let hd = f["head_dim"].as_u64().unwrap() as usize;
    let ks = f["kernel"].as_u64().unwrap() as usize;
    let tpre = f["t_prefill"].as_u64().unwrap() as usize;
    let stride = f["sample_stride"].as_u64().unwrap() as usize;
    let hf_w = f["hf_state_width"].as_u64().unwrap() as usize;

    println!("conv contract at production geometry");
    println!(
        "  conv_dim={dim} qk_channels={qk} head_dim={hd} kernel={ks} act={}",
        f["activation"]
    );
    println!("  qk_channels % 256 = {}  (kernel requires 0)", qk % 256);
    assert_eq!(qk % 256, 0);
    assert_eq!(
        hd, 128,
        "the fused kernel hardcodes 2 heads per 256-thread block"
    );

    let k_dec = gpu.kernel("causal_conv1d", "causal_conv1d_update_l2norm")?;
    let k_pre = gpu.kernel("causal_conv1d", "causal_conv1d_update_prefill")?;
    let k_l2 = gpu.kernel("norm", "l2_norm_bf16")?;
    println!("  all three conv/L2 entry points resolved (no fallback)\n");

    // LCG must match the generator or the fixture is different.
    let probe: Vec<f32> = v["lcg_probe"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    let mut pr = Lcg(0x5EED_C0F0);
    if pr
        .vec(probe.len())
        .iter()
        .zip(&probe)
        .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        bail!("LCG mismatch with the generator");
    }

    let mut rng = Lcg(0x5EED_C0F0);
    let weight: Vec<f32> = rng.vec(dim * ks).iter().map(|x| x * 0.5).collect();
    let state3: Vec<f32> = rng.vec(dim * hf_w).iter().map(|x| x * 0.5).collect();
    let tok = rng.vec(dim);
    let pre = rng.vec(tpre * dim);

    // HF's 3-wide state -> Atlas's 4-wide: slot 0 is shifted out before the conv, so it is a
    // don't-care. It is filled with a poison value here to PROVE that.
    let widen = |poison: f32| -> Vec<f32> {
        let mut s = vec![0.0f32; dim * ks];
        for ch in 0..dim {
            s[ch * ks] = poison;
            for i in 0..hf_w {
                s[ch * ks + 1 + i] = state3[ch * hf_w + i];
            }
        }
        s
    };

    let mut ok = true;

    // ── DECODE ────────────────────────────────────────────────────────────────
    let dw = up_bf16(gpu, &weight)?;
    let dtok = up_bf16(gpu, &tok)?;
    let run_decode = |st: &[f32]| -> Result<(Vec<f32>, Vec<f32>)> {
        let dstate = up_f32(gpu, st)?;
        let dout = gpu.alloc(dim * 2)?;
        KernelLaunch::new(gpu, k_dec)
            .grid([(dim as u32).div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(dstate)
            .arg_ptr(dtok)
            .arg_ptr(dw)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(dout)
            .arg_u32(1)
            .arg_u32(dim as u32)
            .arg_u32(ks as u32)
            .arg_u32(qk as u32)
            .arg_u32(hd as u32)
            .arg_f32(1e-6)
            .launch(0)?;
        gpu.synchronize(0)?;
        Ok((down_bf16(gpu, dout, dim)?, down_f32(gpu, dstate, dim * ks)?))
    };

    let (d_out, d_state) = run_decode(&widen(0.0))?;
    let want_out = arr(&v, "decode_out");
    // CPU floor on the same bf16-rounded inputs.
    let (mut cpu_out, cpu_state) = cpu_conv_silu(&widen(0.0), &tok, &weight, dim, ks);
    let cpu_state_tail: Vec<f32> = (0..dim)
        .flat_map(|ch| (1..ks).map(move |i| (ch, i)))
        .map(|(ch, i)| cpu_state[ch * ks + i])
        .collect();
    l2_rows(&mut cpu_out, hd, qk);
    let cpu_out_b: Vec<f32> = cpu_out
        .iter()
        .map(|x| bf16::from_f32(*x).to_f32())
        .collect();

    let e_hf = maxabs(&d_out, &want_out);
    let e_floor = maxabs(&cpu_out_b, &want_out);
    let e_kern = maxabs(&d_out, &cpu_out_b);
    println!("DECODE (conv + SiLU + L2 fused)");
    println!("  CPU-ref(bf16) vs HF   max_abs={e_floor:.3e}   <- input-rounding floor");
    println!("  GPU vs HF             max_abs={e_hf:.3e}");
    println!("  GPU vs CPU-ref        max_abs={e_kern:.3e}   <- kernel only");
    // Atlas keeps 4 slots, HF keeps 3: compare the overlapping window Atlas[1..4] vs HF[0..3].
    let d_state_tail: Vec<f32> = (0..dim)
        .flat_map(|ch| (1..ks).map(move |i| (ch, i)))
        .map(|(ch, i)| d_state[ch * ks + i])
        .collect();
    let s_hf = maxabs(
        &sample(&d_state_tail, stride),
        &arr(&v, "decode_state_sample"),
    );
    let s_cpu = maxabs(&d_state_tail, &cpu_state_tail);
    println!("  state Atlas[1..4] vs HF[0..3] max_abs={s_hf:.3e}   vs CPU-ref {s_cpu:.3e}");
    ok &= s_hf <= MAX_ABS_BF16;
    ok &= e_hf <= MAX_ABS_BF16 && e_kern <= MAX_ABS_BF16;

    // Slot 0 must be a genuine don't-care.
    let (p_out, _) = run_decode(&widen(1.0e6))?;
    let poison = maxabs(&p_out, &d_out);
    println!(
        "  poisoned state slot 0 -> output delta {poison:.3e}  [{}]",
        if poison == 0.0 {
            "ok, slot 0 is shifted out"
        } else {
            "FAIL"
        }
    );
    ok &= poison == 0.0;

    // L2 applied exactly once: every q|k head row must be unit norm, and V must NOT be.
    let qk_norms: Vec<f32> = d_out[..qk]
        .chunks_exact(hd)
        .map(|r| r.iter().map(|a| a * a).sum::<f32>().sqrt())
        .collect();
    let v_norms: Vec<f32> = d_out[qk..]
        .chunks_exact(hd)
        .map(|r| r.iter().map(|a| a * a).sum::<f32>().sqrt())
        .collect();
    let (qmin, qmax) = (
        qk_norms.iter().cloned().fold(f32::MAX, f32::min),
        qk_norms.iter().cloned().fold(0.0, f32::max),
    );
    let vmax = v_norms.iter().cloned().fold(0.0, f32::max);
    println!(
        "  q|k row norms in [{qmin:.6}, {qmax:.6}] (must be ~1); V max norm {vmax:.4} (must NOT be 1)"
    );
    ok &= (qmin - 1.0).abs() < 0.02 && (qmax - 1.0).abs() < 0.02 && (vmax - 1.0).abs() > 0.05;

    // ── PREFILL ───────────────────────────────────────────────────────────────
    // conv + SiLU over T tokens from ZERO state, then L2 on q|k only.
    let dpre = up_bf16(gpu, &pre)?;
    let dstate = up_f32(gpu, &vec![0.0f32; dim * ks])?;
    let dpout = gpu.alloc(tpre * dim * 2)?;
    KernelLaunch::new(gpu, k_pre)
        .grid([(dim as u32).div_ceil(256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(dstate)
        .arg_ptr(dpre)
        .arg_ptr(dw)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(dpout)
        .arg_u32(dim as u32)
        .arg_u32(ks as u32)
        .arg_u32(tpre as u32)
        .arg_u32(dim as u32)
        .arg_u32(dim as u32)
        .launch(0)?;
    // L2 over q|k channels only: 128 groups of 128 = 16384 = qk_channels.
    KernelLaunch::new(gpu, k_l2)
        .grid([(qk / hd) as u32, tpre as u32, 1])
        .block([hd as u32, 1, 1])
        .arg_ptr(dpout)
        .arg_u32(hd as u32)
        .arg_f32(1e-6)
        .arg_u32(dim as u32)
        .launch(0)?;
    gpu.synchronize(0)?;
    let p_out = down_bf16(gpu, dpout, tpre * dim)?;
    let p_state = down_f32(gpu, dstate, dim * ks)?;

    println!("\nPREFILL (conv + SiLU, then a SEPARATE L2 on q|k only)");
    let po_hf = maxabs(&sample(&p_out, stride), &arr(&v, "prefill_out_sample"));
    let po_ck = checksum(&p_out);
    let want_ck = v["checksums"]["prefill_out"].as_f64().unwrap();
    println!("  out sample vs HF      max_abs={po_hf:.3e}");
    println!(
        "  full-out fp64 checksum rel={:.3e}",
        (po_ck - want_ck).abs() / want_ck.abs().max(1.0)
    );
    ok &= po_hf <= MAX_ABS_BF16;

    // Atlas's 4-wide final state vs HF's 3-wide: compare the overlapping window.
    let atlas_tail: Vec<f32> = (0..dim)
        .flat_map(|ch| (1..ks).map(move |i| (ch, i)))
        .map(|(ch, i)| p_state[ch * ks + i])
        .collect();
    let ps_hf = maxabs(
        &sample(&atlas_tail, stride),
        &arr(&v, "prefill_state_sample"),
    );
    println!("  final conv state (Atlas[1..4] vs HF[0..3]) max_abs={ps_hf:.3e}");
    ok &= ps_hf <= MAX_ABS_BF16;

    let pqk: Vec<f32> = p_out[..qk]
        .chunks_exact(hd)
        .map(|r| r.iter().map(|a| a * a).sum::<f32>().sqrt())
        .collect();
    let (pmin, pmax) = (
        pqk.iter().cloned().fold(f32::MAX, f32::min),
        pqk.iter().cloned().fold(0.0, f32::max),
    );
    println!(
        "  token-0 q|k row norms in [{pmin:.6}, {pmax:.6}] (must be ~1, i.e. L2 ran exactly once)"
    );
    ok &= (pmin - 1.0).abs() < 0.02 && (pmax - 1.0).abs() < 0.02;

    // Double-L2 must be detectable, or the "exactly once" check proves nothing.
    let mut twice = p_out.clone();
    for t in 0..tpre {
        l2_rows(&mut twice[t * dim..(t + 1) * dim], hd, qk);
    }
    let dbl = maxabs(&twice, &p_out);
    println!(
        "  applying L2 a SECOND time moves the result by {dbl:.3e}  [{}]",
        if dbl > 1e-4 {
            "ok, detectable"
        } else {
            "not detectable at bf16 -- see report"
        }
    );

    if ok {
        println!("\nCONV CONTRACT: PASS");
        Ok(())
    } else {
        bail!("CONV CONTRACT: FAIL")
    }
}
