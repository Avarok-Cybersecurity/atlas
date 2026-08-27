// SPDX-License-Identifier: AGPL-3.0-only

//! GATE — `glm5next_dsa_mla_decode_fp8` vs the `dsa_mla_masked_attn` oracle.
//!
//! Run: `cargo run --release --example glm5next_dsa_decode_gate`
//! Env: `KDA_DSA_PACKET_DIR` (default `/home/msi1/atlas-scratch/dsa-family`).
//!
//! # What is actually being compared
//!
//! Both kernels are made to compute the SAME operation — absorbed NoPE MLA attention in
//! latent space — by two independent code paths:
//!
//! | | oracle `dsa_mla_masked_attn` | subject `glm5next_dsa_mla_decode_fp8` |
//! |---|---|---|
//! | selection | dense `[Q, S]` visibility mask | per-row gather of a `[width]` index row |
//! | cache | flat `[S, H, 512]` BF16 | **paged** FP8, block table |
//! | scores | `[S]` row staged in shared memory | streamed, online softmax |
//! | reduction | one block per (query, head) | 8 warps split the selection row, merged |
//!
//! The oracle takes `qd`/`vd` as arguments, so feeding it `qd = vd = 512` with the latent
//! broadcast across heads makes it compute exactly what the subject computes. That is the
//! point: same algebra, nothing shared but the inputs.
//!
//! # Isolating logic from dtype noise
//!
//! The FP8 scale is a **power of two**, so every dequantised value `fp8 * scale` is exactly
//! representable in BF16 (E4M3 carries 3 mantissa bits, BF16 carries 7). The oracle is then
//! fed the dequantised latent, so both paths see **bit-identical** numbers and any
//! disagreement is logic, not quantisation.
//!
//! # The attributable floor
//!
//! * the subject writes **BF16** output, the oracle writes FP32 → ~2^-8 relative
//! * the two accumulate in a different order (8 warps vs one) → FP32 associativity, ~1e-6
//!
//! So the floor is BF16 output rounding: **rel ≈ 4e-3**. Anything structured or larger is
//! a bug, and the all-`-1` row must be exactly zero, not merely small.

use anyhow::{Context, Result, bail};
use serde_json::Value;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;
use std::collections::BTreeMap;

const LAYER: usize = 3;
const S: usize = 64; // short context, by instruction
const HEADS: usize = 64;
const KVL: usize = 512;
const NOPE: usize = 256;
const VD: usize = 256;
const HID: usize = 4096;
const QL: usize = 1536;
const BLOCK_SIZE: usize = 64;
const WIDTH: usize = 2051; // index_topk 2048 + (kpool - 1)

fn up_bf16(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d
        .iter()
        .flat_map(|x| half::bf16::from_f32(*x).to_le_bytes())
        .collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn up_i32(g: &dyn GpuBackend, d: &[i32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn up_u8(g: &dyn GpuBackend, d: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(d.len().max(1))?;
    g.copy_h2d(d, p)?;
    Ok(p)
}
fn down_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}
fn down_bf16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks(2)
        .map(|c| half::bf16::from_le_bytes(c.try_into().unwrap()).to_f32())
        .collect())
}
fn round_bf16(v: &[f32]) -> Vec<f32> {
    v.iter()
        .map(|x| half::bf16::from_f32(*x).to_f32())
        .collect()
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }
}

struct Packet {
    raw: Vec<u8>,
    base: usize,
    hdr: BTreeMap<String, (String, Vec<usize>, usize, usize)>,
}
impl Packet {
    fn open(path: &str) -> Result<Self> {
        let raw = std::fs::read(path).with_context(|| format!("reading {path}"))?;
        let hn = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
        let j: Value = serde_json::from_slice(&raw[8..8 + hn])?;
        let mut hdr = BTreeMap::new();
        for (k, m) in j.as_object().unwrap() {
            if k == "__metadata__" {
                continue;
            }
            let dt = m["dtype"].as_str().unwrap().to_string();
            let shape = m["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect();
            let a = m["data_offsets"][0].as_u64().unwrap() as usize;
            let b = m["data_offsets"][1].as_u64().unwrap() as usize;
            hdr.insert(k.clone(), (dt, shape, a, b));
        }
        Ok(Self {
            raw,
            base: 8 + hn,
            hdr,
        })
    }
    fn f32s(&self, name: &str) -> Result<Vec<f32>> {
        let (dt, _, a, b) = self
            .hdr
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?;
        let s = &self.raw[self.base + a..self.base + b];
        Ok(match dt.as_str() {
            "BF16" => s
                .chunks(2)
                .map(|c| half::bf16::from_le_bytes(c.try_into().unwrap()).to_f32())
                .collect(),
            "F32" => s
                .chunks(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            o => bail!("{name}: unexpected dtype {o}"),
        })
    }
}

fn gemm(x: &[f32], m: usize, k: usize, w: &[f32], n: usize) -> Vec<f32> {
    let mut o = vec![0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            let mut a = 0f32;
            for i in 0..k {
                a += x[r * k + i] * w[c * k + i];
            }
            o[r * n + c] = a;
        }
    }
    o
}
fn rms_norm(x: &[f32], w: &[f32], d: usize, eps: f32) -> Vec<f32> {
    let mut o = vec![0f32; x.len()];
    for r in 0..x.len() / d {
        let s: f32 = x[r * d..r * d + d].iter().map(|v| v * v).sum::<f32>() / d as f32;
        let inv = 1.0 / (s + eps).sqrt();
        for i in 0..d {
            o[r * d + i] = x[r * d + i] * inv * w[i];
        }
    }
    o
}

/// Every finite OCP E4M3 code, decoded exactly: sign(1) exp(4, bias 7) mantissa(3),
/// `e == 0` subnormal at `2^-6`, `0x7F`/`0xFF` are NaN and excluded. Built by hand rather
/// than pulling a dependency, and cross-checked against the anchors the kernel relies on.
fn e4m3_table() -> Vec<(u8, f32)> {
    let mut t = Vec::with_capacity(254);
    for bits in 0u16..256 {
        let b = bits as u8;
        let (s, e, m) = (b >> 7, (b >> 3) & 0xF, b & 0x7);
        if e == 0xF && m == 0x7 {
            continue; // NaN
        }
        let mag = if e == 0 {
            (m as f32 / 8.0) * (2f32).powi(-6)
        } else {
            (1.0 + m as f32 / 8.0) * (2f32).powi(e as i32 - 7)
        };
        t.push((b, if s == 1 { -mag } else { mag }));
    }
    t
}

/// Nearest E4M3 code to `v`, and the value it decodes back to.
fn fp8_e4m3_round(table: &[(u8, f32)], v: f32) -> (u8, f32) {
    let mut best = table[0];
    let mut bd = f32::INFINITY;
    for &(b, x) in table {
        let d = (x - v).abs();
        if d < bd {
            bd = d;
            best = (b, x);
        }
    }
    best
}

fn main() -> Result<()> {
    let dir = std::env::var("KDA_DSA_PACKET_DIR")
        .unwrap_or_else(|_| "/home/msi1/atlas-scratch/dsa-family".to_string());
    let pkt = Packet::open(&format!("{dir}/dsa_layer{LAYER}.safetensors"))?;
    println!("GATE — glm5next_dsa_mla_decode_fp8 vs dsa_mla_masked_attn");
    println!("  layer {LAYER}, S={S}, heads={HEADS}, latent={KVL}, real weights from {dir}\n");

    // `ptx_modules()` aliases target 0 (deepseek-v4-flash). The subject kernel lives in
    // the glm-5.3-flash target — which is the point: it is GLM-only, so V4 cannot see it.
    let sets = atlas_kernels::all_ptx_sets();
    let glm = sets
        .iter()
        .find(|s| s.target.model == "glm-5.3-flash")
        .context("glm-5.3-flash kernel target not built")?;
    println!(
        "  kernel target {} / {} / {} — {} modules",
        glm.target.arch,
        glm.target.model,
        glm.target.quant,
        glm.modules.len()
    );
    let gpu = AtlasCudaBackend::new(0, &glm.modules)?;
    let k_sub = gpu.kernel("glm5next_dsa_mla_decode", "glm5next_dsa_mla_decode_fp8")?;
    let k_orc = gpu.kernel("dsa_indexer", "dsa_mla_masked_attn")?;
    // The indexer's k_norm is an nn.LayerNorm WITH A BIAS. Prove the bias-bearing kernel
    // actually resolves in the GLM target rather than assuming common/ reached it.
    let _ = gpu
        .kernel("nllb_encoder", "nllb_layernorm_bf16")
        .context("indexer k_norm LayerNorm(bias) kernel missing from the GLM target")?;
    println!("  indexer k_norm LayerNorm(weight, bias) resolves in the GLM target");

    // Every module name the DSA layer resolves, checked against real hardware. Module names
    // follow TWO conventions -- `common/KERNEL.toml` [modules] overrides, else the .cu file
    // stem -- so a wrong one is invisible until construction. This caught
    // `dense_gemm_bf16` (stem) where the map says `gemm`.
    spark_model::layers::glm5next_dsa::Glm5NextDsaKernels::resolve(&gpu)
        .context("Glm5NextDsaKernels")?;
    spark_model::layers::glm5next_dsa::layer::Glm5NextDsaLayerKernels::resolve(&gpu)
        .context("Glm5NextDsaLayerKernels")?;
    spark_model::layers::glm5next_dsa::attend::Glm5NextDsaDecodeKernel::resolve(&gpu)
        .context("Glm5NextDsaDecodeKernel")?;
    spark_model::layers::ops::Glm5NextMhcKernels::resolve(&gpu).context("Glm5NextMhcKernels")?;
    println!("  every DSA-layer + mHC kernel module resolves");

    // ── real weights ────────────────────────────────────────────────────────────────
    let w_qa = round_bf16(&pkt.f32s("self_attn.q_a_proj.weight")?);
    let n_qa = pkt.f32s("self_attn.q_a_layernorm.weight")?;
    let w_qb = round_bf16(&pkt.f32s("self_attn.q_b_proj.weight")?);
    let w_kva = round_bf16(&pkt.f32s("self_attn.kv_a_proj_with_mqa.weight")?);
    let n_kva = pkt.f32s("self_attn.kv_a_layernorm.weight")?;
    let w_kvb = round_bf16(&pkt.f32s("self_attn.kv_b_proj.weight")?);

    let mut rng = Lcg(0x0D5A_C0DE);
    let hidden = round_bf16(&rng.vec(S * HID).iter().map(|x| x * 0.5).collect::<Vec<_>>());

    // KV latent — NoPE, so kv_a_proj emits exactly kv_lora_rank and there is no rope split.
    let kva = gemm(&hidden, S, HID, &w_kva, KVL);
    let latent = round_bf16(&rms_norm(&round_bf16(&kva), &n_kva, KVL, 1e-5));

    // Q, absorbed into latent space: q_abs[h] = W_k[h]^T @ q_nope[h].
    // W_k[h] is rows [h*(NOPE+VD) .. +NOPE] of kv_b_proj, shape [NOPE, KVL].
    let qa = gemm(&hidden, S, HID, &w_qa, QL);
    let q_resid = round_bf16(&rms_norm(&round_bf16(&qa), &n_qa, QL, 1e-5));
    let q_nope = round_bf16(&gemm(&q_resid, S, QL, &w_qb, HEADS * NOPE));
    let last = S - 1; // decode: one query row, the newest token
    let mut q_abs = vec![0f32; HEADS * KVL];
    for h in 0..HEADS {
        for c in 0..KVL {
            let mut a = 0f32;
            for r in 0..NOPE {
                a += q_nope[last * HEADS * NOPE + h * NOPE + r]
                    * w_kvb[(h * (NOPE + VD) + r) * KVL + c];
            }
            q_abs[h * KVL + c] = a;
        }
    }
    let q_abs = round_bf16(&q_abs);

    // ── FP8 cache, power-of-two scale so dequant is BF16-exact ──────────────────────
    let maxabs = latent.iter().fold(0f32, |m, v| m.max(v.abs()));
    let scale = (maxabs / 448.0).max(f32::MIN_POSITIVE).log2().ceil().exp2();
    let table = e4m3_table();
    assert_eq!(table.len(), 254, "E4M3 has 254 finite codes");
    assert!(
        table.iter().any(|(_, v)| *v == 448.0),
        "E4M3 max normal must be 448"
    );
    let mut fp8_bytes = vec![0u8; S * KVL];
    let mut latent_q = vec![0f32; S * KVL];
    for i in 0..S * KVL {
        let (bits, back) = fp8_e4m3_round(&table, latent[i] / scale);
        fp8_bytes[i] = bits;
        latent_q[i] = back * scale;
    }
    let exact = latent_q
        .iter()
        .all(|v| half::bf16::from_f32(*v).to_f32() == *v);
    println!("  FP8 scale 2^{:+} (power of two)", scale.log2() as i32);
    println!("  dequantised latent is BF16-exact: {exact}  <- both paths see identical values");
    if !exact {
        bail!("dequantised latent is not BF16-exact; the gate would measure dtype noise");
    }

    // ── oracle inputs: latent broadcast across heads, qd = vd = KVL ─────────────────
    let mut kv_exp = vec![0f32; S * HEADS * KVL];
    for t in 0..S {
        for h in 0..HEADS {
            kv_exp[(t * HEADS + h) * KVL..(t * HEADS + h) * KVL + KVL]
                .copy_from_slice(&latent_q[t * KVL..t * KVL + KVL]);
        }
    }
    let d_q = up_bf16(&gpu, &q_abs)?;
    let d_kexp = up_bf16(&gpu, &kv_exp)?;
    let inv_sqrt_d = (KVL as f32).powf(-0.5);

    // ── subject inputs: one paged FP8 block ────────────────────────────────────────
    let d_cache = up_u8(&gpu, &fp8_bytes)?;
    let d_bt = up_i32(&gpu, &[0i32])?;
    let d_sl = up_i32(&gpu, &[S as i32])?;

    let cases: [(&str, Vec<i32>); 4] = [
        ("all 64 tokens", (0..S as i32).collect()),
        ("every 4th", (0..S as i32).filter(|t| t % 4 == 0).collect()),
        ("single token 37", vec![37]),
        ("empty (all -1)", vec![]),
    ];

    let mut worst = 0f32;
    let mut failed = false;
    println!(
        "\n  {:<16} {:>8} {:>13} {:>13} {:>10}",
        "selection", "picked", "max abs diff", "max rel diff", "verdict"
    );
    for (name, picked) in cases {
        let mut sel = vec![-1i32; WIDTH];
        sel[..picked.len()].copy_from_slice(&picked);
        let d_sel = up_i32(&gpu, &sel)?;

        // oracle: CPU-built visibility mask, so the selection is not shared with the
        // subject through any kernel.
        let mut mask = vec![0u8; S];
        for t in &picked {
            mask[*t as usize] = 1;
        }
        let d_mask = up_u8(&gpu, &mask)?;
        let d_orc = gpu.alloc(HEADS * KVL * 4)?;
        KernelLaunch::new(&gpu, k_orc)
            .grid([1, HEADS as u32, 1])
            .block([256, 1, 1])
            .shared_mem((S * 4) as u32)
            .arg_ptr(d_q)
            .arg_ptr(d_kexp)
            .arg_ptr(d_kexp)
            .arg_ptr(d_mask)
            .arg_ptr(d_orc)
            .arg_u32(1)
            .arg_u32(S as u32)
            .arg_u32(HEADS as u32)
            .arg_u32(KVL as u32)
            .arg_u32(KVL as u32)
            .arg_f32(inv_sqrt_d)
            .arg_u32(0)
            .launch(0)?;

        let d_out = gpu.alloc(HEADS * KVL * 2)?;
        gpu.memset(d_out, 0xAB, HEADS * KVL * 2)?; // poison: a skipped write must show
        KernelLaunch::new(&gpu, k_sub)
            .grid([HEADS as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(d_q)
            .arg_ptr(d_cache)
            .arg_ptr(d_cache)
            .arg_ptr(d_out)
            .arg_ptr(d_bt)
            .arg_ptr(d_sl)
            .arg_ptr(d_sel)
            .arg_u32(WIDTH as u32)
            .arg_u32(1)
            .arg_u32(HEADS as u32)
            .arg_u32(1)
            .arg_u32(KVL as u32)
            .arg_u32(BLOCK_SIZE as u32)
            .arg_f32(inv_sqrt_d)
            .arg_f32(scale)
            .arg_f32(scale)
            .arg_u64((BLOCK_SIZE * KVL) as u64)
            .launch(0)?;
        gpu.synchronize(0)?;

        let o = down_f32(&gpu, d_orc, HEADS * KVL)?;
        let s = down_bf16(&gpu, d_out, HEADS * KVL)?;

        let (mut mabs, mut mrel) = (0f32, 0f32);
        for i in 0..HEADS * KVL {
            let d = (o[i] - s[i]).abs();
            mabs = mabs.max(d);
            let den = o[i].abs().max(1e-6);
            mrel = mrel.max(d / den);
        }
        // The empty row is not "small", it is exactly zero.
        let ok = if picked.is_empty() {
            let z = s.iter().all(|v| *v == 0.0);
            if !z {
                println!("    empty row is NOT exactly zero");
            }
            z
        } else {
            mrel <= 8e-3
        };
        failed |= !ok;
        worst = worst.max(if picked.is_empty() { 0.0 } else { mrel });
        println!(
            "  {:<16} {:>8} {:>13.3e} {:>13.3e} {:>10}",
            name,
            picked.len(),
            mabs,
            mrel,
            if ok { "PASS" } else { "FAIL" }
        );
    }

    println!("\n  worst relative diff {worst:.3e} vs BF16 output floor 4e-3 (threshold 8e-3)");
    if failed {
        bail!("GATE FAILED — isolate the kernel before touching the layer");
    }
    println!("  GATE PASS");
    Ok(())
}
