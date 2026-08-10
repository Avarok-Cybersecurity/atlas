// SPDX-License-Identifier: AGPL-3.0-only

//! Standalone correctness microtest for the runtime LoRA delta apply
//! (`ops::lora_delta::apply_lora_delta`, m=1 decode path).
//!
//! This is the RUNTIME parity oracle the offline `reference_deltas.py` never
//! provided: it runs the REAL CUDA shrink/expand/fold (`dense_gemv_bf16` +
//! `bf16_scaled_add`) on the production `GpuBackend`, and bisects each stage
//! against a bf16-faithful CPU reference so a divergence points at the exact
//! kernel:
//!   shrink:  xa[j]    = Σ_k x[k]·A[j,k]            (A packed [max_rank, k_in])
//!   expand:  delta[n] = Σ_j xa[j]·B[n,j]           (B packed [n_out, max_rank])
//!   fold:    out[n]  += scale·delta[n]             (out starts zeroed)
//!
//! Usage:
//!   cargo run --release -p spark-model --example lora_apply_microtest \
//!       -- [k_in] [n_out] [r] [max_rank] [m] [seed]
//! Defaults: Holo-3.1-0.8B k_proj shape — k_in=1024 n_out=512 r=8 max_rank=64 m=1.
//! `m=1` exercises the decode `dense_gemv` path; `m>1` the prefill
//! `dense_gemm_tc`/`dense_gemm` path (which produces the FIRST token).
//! Exit 0 = all stages PASS (cosine >= gate), 1 = FAIL.

use anyhow::{Result, bail};
use spark_model::layers::ops::lora_delta::{LoraKernels, LoraPair, apply_lora_delta};
use spark_model::weight_map::DenseWeight;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

const COSINE_GATE: f64 = 0.999;

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        lo + u * (hi - lo)
    }
}

fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}
fn f32_to_bf16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}
fn u16s_to_le(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn le_to_u16s(v: &[u8]) -> Vec<u16> {
    v.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}
fn upload(gpu: &dyn GpuBackend, bits: &[u16]) -> Result<DevicePtr> {
    let bytes = u16s_to_le(bits);
    let p = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(&bytes, p)?;
    Ok(p)
}

/// bf16-faithful compare in f32 space: cosine + max/mean relative error.
fn compare(label: &str, gpu: &[u16], reference: &[f32]) -> bool {
    let (mut dot, mut ng, mut nr, mut maxrel, mut sumrel) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (g_bits, &r) in gpu.iter().zip(reference) {
        let g = bf16_bits_to_f32(*g_bits) as f64;
        let r = r as f64;
        dot += g * r;
        ng += g * g;
        nr += r * r;
        let denom = r.abs().max(1e-3);
        let rel = (g - r).abs() / denom;
        maxrel = maxrel.max(rel);
        sumrel += rel;
    }
    let cos = if ng > 0.0 && nr > 0.0 {
        dot / (ng.sqrt() * nr.sqrt())
    } else {
        0.0
    };
    let pass = cos >= COSINE_GATE;
    println!(
        "  {:6} {label:12} cosine={cos:.6} max_rel={maxrel:.4} mean_rel={:.4}  |gpu|₂={:.4} |ref|₂={:.4}",
        if pass { "PASS" } else { "FAIL ❌" },
        sumrel / gpu.len().max(1) as f64,
        ng.sqrt(),
        nr.sqrt(),
    );
    pass
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let k_in: usize = a.get(1).map_or(1024, |s| s.parse().unwrap());
    let n_out: usize = a.get(2).map_or(512, |s| s.parse().unwrap());
    let r: usize = a.get(3).map_or(8, |s| s.parse().unwrap());
    let max_rank: usize = a.get(4).map_or(64, |s| s.parse().unwrap());
    let m: usize = a.get(5).map_or(1, |s| s.parse().unwrap());
    let seed: u64 = a.get(6).map_or(0x51A7, |s| {
        u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0x51A7)
    });
    let scale = 2.0f32;
    assert!(r <= max_rank);
    let path = if m == 1 {
        "decode dense_gemv"
    } else {
        "prefill dense_gemm"
    };
    println!(
        "=== lora_apply microtest: k_in={k_in} n_out={n_out} r={r} max_rank={max_rank} m={m} ({path}) scale={scale} seed=0x{seed:X} ==="
    );

    // ── inputs (bf16) — small magnitudes like post-norm activations / PEFT weights ──
    let mut rng = Rng(seed);
    let x: Vec<u16> = (0..m * k_in)
        .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
        .collect();
    // real A [r, k_in], real B [n_out, r]
    let a_real: Vec<u16> = (0..r * k_in)
        .map(|_| f32_to_bf16_bits(rng.uniform(-0.05, 0.05)))
        .collect();
    let b_real: Vec<u16> = (0..n_out * r)
        .map(|_| f32_to_bf16_bits(rng.uniform(-0.05, 0.05)))
        .collect();

    // ── pack into pool layout: A [max_rank, k_in] (pad rows 0), B [n_out, max_rank] (pad cols 0) ──
    let mut a_pool = vec![0u16; max_rank * k_in];
    for j in 0..r {
        a_pool[j * k_in..(j + 1) * k_in].copy_from_slice(&a_real[j * k_in..(j + 1) * k_in]);
    }
    let mut b_pool = vec![0u16; n_out * max_rank];
    for n in 0..n_out {
        b_pool[n * max_rank..n * max_rank + r].copy_from_slice(&b_real[n * r..(n + 1) * r]);
    }

    // ── GPU ──
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    let kernels = LoraKernels::new(gpu)?;

    let x_ptr = upload(gpu, &x)?;
    let a_ptr = upload(gpu, &a_pool)?;
    let b_ptr = upload(gpu, &b_pool)?;
    let xa_ptr = gpu.alloc(m * max_rank * 2)?;
    let delta_ptr = gpu.alloc(m * n_out * 2)?;
    let out_ptr = gpu.alloc(m * n_out * 2)?;
    gpu.memset(out_ptr, 0, m * n_out * 2)?;

    let pair = LoraPair {
        a: DenseWeight { weight: a_ptr },
        b: DenseWeight { weight: b_ptr },
        rank: r as u32,
        k_in: k_in as u32,
        n_out: n_out as u32,
        scale,
        max_rank: max_rank as u32,
    };

    apply_lora_delta(
        gpu, &kernels, &pair, x_ptr, out_ptr, m as u32, xa_ptr, delta_ptr, stream,
    )?;
    gpu.synchronize(stream)?;

    let mut xa_raw = vec![0u8; m * max_rank * 2];
    let mut delta_raw = vec![0u8; m * n_out * 2];
    let mut out_raw = vec![0u8; m * n_out * 2];
    gpu.copy_d2h(xa_ptr, &mut xa_raw)?;
    gpu.copy_d2h(delta_ptr, &mut delta_raw)?;
    gpu.copy_d2h(out_ptr, &mut out_raw)?;
    let xa_gpu = le_to_u16s(&xa_raw);
    let delta_gpu = le_to_u16s(&delta_raw);
    let out_gpu = le_to_u16s(&out_raw);

    // ── bf16-faithful CPU reference (fp32 accumulate, narrow to bf16 between stages), per row ──
    let xf: Vec<f32> = x.iter().map(|&b| bf16_bits_to_f32(b)).collect();
    let af: Vec<f32> = a_real.iter().map(|&b| bf16_bits_to_f32(b)).collect();
    let bf: Vec<f32> = b_real.iter().map(|&b| bf16_bits_to_f32(b)).collect();
    let mut xa_ref = vec![0f32; m * max_rank]; // padded rows/cols stay 0
    let mut delta_ref = vec![0f32; m * n_out];
    for row in 0..m {
        for j in 0..r {
            let mut acc = 0f32;
            for k in 0..k_in {
                acc += xf[row * k_in + k] * af[j * k_in + k];
            }
            xa_ref[row * max_rank + j] = bf16_bits_to_f32(f32_to_bf16_bits(acc)); // xa stored bf16
        }
        for n in 0..n_out {
            let mut acc = 0f32;
            for j in 0..r {
                acc += xa_ref[row * max_rank + j] * bf[n * r + j];
            }
            delta_ref[row * n_out + n] = bf16_bits_to_f32(f32_to_bf16_bits(acc));
        }
    }
    let out_ref: Vec<f32> = delta_ref.iter().map(|&d| scale * d).collect();

    println!("stage bisection ({path}):");
    // compare only the valid (non-padded) xa columns across all rows
    let mut xa_gpu_v = Vec::new();
    let mut xa_ref_v = Vec::new();
    for row in 0..m {
        for j in 0..r {
            xa_gpu_v.push(xa_gpu[row * max_rank + j]);
            xa_ref_v.push(xa_ref[row * max_rank + j]);
        }
    }
    let p1 = compare("shrink xa", &xa_gpu_v, &xa_ref_v);
    let p2 = compare("expand delta", &delta_gpu, &delta_ref);
    let p3 = compare("fold out", &out_gpu, &out_ref);

    // flag if padded xa cols aren't zero (a silent-garbage source)
    let mut pad_nonzero = 0usize;
    for row in 0..m {
        for j in r..max_rank {
            if bf16_bits_to_f32(xa_gpu[row * max_rank + j]) != 0.0 {
                pad_nonzero += 1;
            }
        }
    }
    if pad_nonzero > 0 {
        println!("  NOTE: {pad_nonzero} padded xa cols are NONZERO across {m} rows (should be 0)");
    }

    if p1 && p2 && p3 {
        println!("RESULT: PASS ✅ — runtime apply matches reference");
        Ok(())
    } else {
        bail!("RESULT: FAIL — first diverging stage localizes the bug");
    }
}
