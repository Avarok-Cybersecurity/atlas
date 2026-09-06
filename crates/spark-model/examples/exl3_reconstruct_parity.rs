// SPDX-License-Identifier: AGPL-3.0-only
//! BYTE-parity gate for the EXL3 trellis reconstruct kernel
//! (`exl3_reconstruct_had_k{K}_cb{CB}` + `exl3_f16_to_bf16_t`).
//!
//! Two deliberately independent implementations of the EXL3 decode spec —
//! the CUDA port of ExLlamaV3's `reconstruct_had_slice` and the CPU
//! reference in `spark_runtime::weights::exl3::cpu_ref` (written from the
//! format spec, not transcribed from the kernel's thread structure) — must
//! agree BIT-FOR-BIT on random data across every (shape, K, codebook) leg,
//! at BOTH stages: the raw f16 `[in, out]` reconstruction and the
//! transposed BF16 `[out, in]` Atlas layout.
//!
//! Plus one negative control per shape: a single flipped trellis bit MUST
//! change the output (else this harness is vacuous).
//!
//! Optionally, with real checkpoint tensors dumped by
//! `.research/fetch_exl3_tensor.py` into `EXL3_REAL_DIR`, runs the same
//! GPU-vs-CPU comparison on a REAL tensor from
//! turboderp/Qwen3.8-Flash-Next-exl3 and prints value statistics.
//!
//! Exit: 0 all legs byte-identical, 1 any leg differs or a control fails,
//! 2 kernels absent from this target's module set.
//!
//! Run:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.8-flash-next \
//!   ATLAS_TARGET_QUANT=nvfp4 cargo run -p spark-model --release \
//!     --features cuda,gpu-examples --example exl3_reconstruct_parity

use anyhow::Result;
use half::{bf16, f16};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::weights::exl3::{
    Exl3Codebook, cpu_ref, reconstruct_had_bf16, reconstruct_had_f16_device,
};

/// `(label, in_dim, out_dim)` — all multiples of 128. The third is a real
/// qwen4_exp MoE expert shape (gate/up: hidden 2560 -> inter 640).
const SHAPES: [(&str, usize, usize); 3] = [
    ("min [128 x 128]", 128, 128),
    ("rect [256 x 384]", 256, 384),
    ("expert [2560 x 640]", 2560, 640),
];

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn u16(&mut self) -> u16 {
        (self.next() >> 24) as u16
    }
    fn f(&mut self) -> f32 {
        (((self.next() >> 11) as f64) / ((1u64 << 53) as f64)) as f32
    }
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(1))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn down_u16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

fn as_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// CPU-side transpose+convert matching `exl3_f16_to_bf16_t` exactly:
/// f16 -> f32 (exact) -> bf16 (one rounding), `[in, out]` -> `[out, in]`.
fn cpu_transpose_bf16(f16_bits: &[u16], in_dim: usize, out_dim: usize) -> Vec<u16> {
    let mut out = vec![0u16; out_dim * in_dim];
    for r in 0..in_dim {
        for c in 0..out_dim {
            let v = f16::from_bits(f16_bits[r * out_dim + c]).to_f32();
            out[c * in_dim + r] = bf16::from_f32(v).to_bits();
        }
    }
    out
}

fn diff_count(a: &[u16], b: &[u16]) -> usize {
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
}

#[allow(clippy::too_many_arguments)]
fn run_leg(
    g: &dyn GpuBackend,
    label: &str,
    in_dim: usize,
    out_dim: usize,
    k: u32,
    cb: Exl3Codebook,
    trellis: &[u16],
    suh: &[u16],
    svh: &[u16],
) -> Result<bool> {
    // CPU
    let cpu_f16 = cpu_ref::reconstruct_had_f16(trellis, suh, svh, in_dim, out_dim, k, cb);
    let cpu_bf16 = cpu_transpose_bf16(&cpu_f16, in_dim, out_dim);

    // GPU
    let t_d = up(g, &as_bytes(trellis))?;
    let su_d = up(g, &as_bytes(suh))?;
    let sv_d = up(g, &as_bytes(svh))?;
    let f16_d = reconstruct_had_f16_device(g, t_d, su_d, sv_d, in_dim, out_dim, k, cb)?;
    g.synchronize(g.default_stream())?;
    let gpu_f16 = down_u16(g, f16_d, in_dim * out_dim)?;
    g.free(f16_d).ok();
    let bf16_d = reconstruct_had_bf16(g, t_d, su_d, sv_d, in_dim, out_dim, k, cb)?;
    let gpu_bf16 = down_u16(g, bf16_d, out_dim * in_dim)?;
    for p in [t_d, su_d, sv_d, bf16_d] {
        g.free(p).ok();
    }

    let f16_ok = gpu_f16 == cpu_f16;
    let bf16_ok = gpu_bf16 == cpu_bf16;
    println!(
        "{label}  K={k} cb={cb:?}  f16-identical={f16_ok:<5} (diff {})  bf16-identical={bf16_ok:<5} (diff {})",
        diff_count(&gpu_f16, &cpu_f16),
        diff_count(&gpu_bf16, &cpu_bf16),
    );
    Ok(f16_ok && bf16_ok)
}

fn gen_inputs(
    rng: &mut Lcg,
    in_dim: usize,
    out_dim: usize,
    k: u32,
) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let trellis: Vec<u16> = (0..(in_dim / 16) * (out_dim / 16) * 16 * k as usize)
        .map(|_| rng.u16())
        .collect();
    // Sign-ish scale vectors: random magnitude 0.5..1.5, random sign — the
    // real suh/svh are +-1-scaled but a spread catches scale-path bugs.
    let sv = |rng: &mut Lcg, n: usize| -> Vec<u16> {
        (0..n)
            .map(|_| {
                let m = 0.5 + rng.f();
                let s = if rng.next() & 1 == 0 { 1.0 } else { -1.0 };
                f16::from_f32(m * s).to_bits()
            })
            .collect()
    };
    let suh = sv(rng, in_dim);
    let svh = sv(rng, out_dim);
    (trellis, suh, svh)
}

fn real_tensor_leg(g: &dyn GpuBackend) -> Result<Option<bool>> {
    let Ok(dir) = std::env::var("EXL3_REAL_DIR") else {
        return Ok(None);
    };
    let read_u16 = |name: &str| -> Result<Vec<u16>> {
        let b = std::fs::read(format!("{dir}/{name}"))?;
        Ok(b.chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect())
    };
    let meta = std::fs::read_to_string(format!("{dir}/meta.txt"))?;
    let mut in_dim = 0usize;
    let mut out_dim = 0usize;
    let mut k = 0u32;
    let mut cb = Exl3Codebook::Mul1;
    for line in meta.lines() {
        let (key, val) = line.split_once('=').unwrap_or(("", ""));
        match key {
            "in" => in_dim = val.parse()?,
            "out" => out_dim = val.parse()?,
            "k" => k = val.parse()?,
            "cb" => {
                cb = match val {
                    "0" => Exl3Codebook::Inst3,
                    "1" => Exl3Codebook::Mcg,
                    _ => Exl3Codebook::Mul1,
                }
            }
            _ => {}
        }
    }
    let trellis = read_u16("trellis.bin")?;
    let suh = read_u16("suh.bin")?;
    let svh = read_u16("svh.bin")?;
    println!(
        "REAL tensor from {dir}: [{in_dim} x {out_dim}] K={k} cb={cb:?} ({} trellis words)",
        trellis.len()
    );
    let ok = run_leg(g, "REAL", in_dim, out_dim, k, cb, &trellis, &suh, &svh)?;

    // Value statistics of the reconstructed weight (CPU side) — a real
    // checkpoint tensor must look like a weight matrix: zero-ish mean,
    // sane spread, no NaN/Inf.
    let w = cpu_ref::reconstruct_had_f16(&trellis, &suh, &svh, in_dim, out_dim, k, cb);
    let vals: Vec<f32> = w.iter().map(|&b| f16::from_bits(b).to_f32()).collect();
    let n = vals.len() as f64;
    let mean = vals.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = vals.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    let finite = vals.iter().all(|v| v.is_finite());
    let absmax = vals.iter().fold(0f32, |m, &v| m.max(v.abs()));
    println!(
        "REAL stats: mean={mean:.6} std={:.6} absmax={absmax:.4} all-finite={finite}",
        var.sqrt()
    );
    Ok(Some(ok && finite))
}

/// GPU-vs-CPU parity for the `batched_embed_exl3` ngram-row gather kernel:
/// synthetic packed rows in a fake arena (slot = row index), random head
/// bias, K in {4, 6}. The kernel decodes+scales+biases and writes BF16;
/// the CPU reference does the same in f32 and rounds to BF16 last — the
/// kernel's f32 chain matches, so outputs must be bit-identical.
fn ngram_leg(g: &dyn GpuBackend, rng: &mut Lcg) -> Result<bool> {
    let kernel = match g.kernel("embed_from_argmax", "batched_embed_exl3") {
        Ok(k) => k,
        Err(_) => {
            println!("batched_embed_exl3 absent — ngram leg SKIP");
            return Ok(true);
        }
    };
    let mut all_ok = true;
    for k_bits in [4u32, 6u32] {
        let heads = 16usize;
        let rows = 64usize; // 4 tokens x 16 heads
        let words = cpu_ref::ngram_words_per_row(k_bits);
        let dim = cpu_ref::NGRAM_ROW_DIM;

        let mut arena = vec![0u16; rows * words];
        for w in arena.iter_mut() {
            *w = rng.u16();
        }
        // Sane fp16 row scales (word 0): 0.001..0.06-ish magnitudes.
        for r in 0..rows {
            arena[r * words] = f16::from_f32(0.001 + rng.f() * 0.06).to_bits();
        }
        let bias: Vec<u16> = (0..heads * dim)
            .map(|_| f16::from_f32((rng.f() - 0.5) * 0.2).to_bits())
            .collect();
        let slots: Vec<u8> = (0..rows as u32).flat_map(|s| s.to_le_bytes()).collect();

        let arena_d = up(g, &as_bytes(&arena))?;
        let bias_d = up(g, &as_bytes(&bias))?;
        let slots_d = up(g, &slots)?;
        let out_d = g.alloc(rows * dim * 2)?;
        KernelLaunch::new(g, kernel)
            .grid([rows as u32, 1, 1])
            .block([192, 1, 1])
            .arg_ptr(slots_d)
            .arg_ptr(arena_d)
            .arg_ptr(bias_d)
            .arg_ptr(out_d)
            .arg_u32(heads as u32)
            .arg_u32(k_bits)
            .launch(g.default_stream())?;
        g.synchronize(g.default_stream())?;
        let gpu_out = down_u16(g, out_d, rows * dim)?;
        for p in [arena_d, bias_d, slots_d, out_d] {
            g.free(p).ok();
        }

        let mut cpu_out = Vec::with_capacity(rows * dim);
        for r in 0..rows {
            let head = r % heads;
            let vals = cpu_ref::decode_ngram_row(
                &arena[r * words..(r + 1) * words],
                k_bits,
                Some(&bias[head * dim..(head + 1) * dim]),
            );
            cpu_out.extend(vals.iter().map(|&v| bf16::from_f32(v).to_bits()));
        }
        let ok = gpu_out == cpu_out;
        println!(
            "ngram rows K={k_bits}  bf16-identical={ok:<5} (diff {})",
            diff_count(&gpu_out, &cpu_out)
        );
        all_ok &= ok;
    }
    Ok(all_ok)
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;

    // Probe one kernel; absent = this target set doesn't carry the module.
    if g.kernel("exl3_reconstruct", "exl3_reconstruct_had_k4_cb2")
        .is_err()
    {
        println!("exl3_reconstruct kernels absent from this target set — SKIP");
        std::process::exit(2);
    }

    let mut rng = Lcg(0x5EED_CAFE);
    let mut clean = true;
    let mut control_ok = true;

    for (label, in_dim, out_dim) in SHAPES {
        // The checkpoint family's real bitrates (2.05..6.05 bpw bodies) plus
        // both live codebooks. cb0 exists upstream but no published branch
        // uses it; one leg keeps it honest.
        for (k, cb) in [
            (2u32, Exl3Codebook::Mul1),
            (3, Exl3Codebook::Mul1),
            (4, Exl3Codebook::Mul1),
            (5, Exl3Codebook::Mul1),
            (6, Exl3Codebook::Mul1),
            (4, Exl3Codebook::Mcg),
            (6, Exl3Codebook::Mcg),
            (4, Exl3Codebook::Inst3),
        ] {
            let (trellis, suh, svh) = gen_inputs(&mut rng, in_dim, out_dim, k);
            clean &= run_leg(g, label, in_dim, out_dim, k, cb, &trellis, &suh, &svh)?;
        }

        // Negative control: flip ONE trellis bit, outputs must differ.
        let (mut trellis, suh, svh) = gen_inputs(&mut rng, in_dim, out_dim, 4);
        let base = cpu_ref::reconstruct_had_f16(
            &trellis,
            &suh,
            &svh,
            in_dim,
            out_dim,
            4,
            Exl3Codebook::Mul1,
        );
        trellis[7] ^= 1 << 3;
        let pert = cpu_ref::reconstruct_had_f16(
            &trellis,
            &suh,
            &svh,
            in_dim,
            out_dim,
            4,
            Exl3Codebook::Mul1,
        );
        let differs = base != pert;
        control_ok &= differs;
        println!("{label}  CONTROL 1-bit trellis flip detected={differs}");
    }

    clean &= ngram_leg(g, &mut rng)?;

    match real_tensor_leg(g) {
        Ok(Some(ok)) => clean &= ok,
        Ok(None) => println!("(no EXL3_REAL_DIR set — real-tensor leg skipped)"),
        Err(e) => {
            println!("REAL tensor leg FAILED to run: {e:#}");
            clean = false;
        }
    }

    if !control_ok {
        println!("FAIL — negative control did not change the output; harness is VACUOUS.");
        std::process::exit(1);
    }
    if clean {
        println!(
            "PASS — GPU exl3_reconstruct is byte-identical to the independent CPU reference at every leg."
        );
        Ok(())
    } else {
        println!("FAIL — GPU and CPU EXL3 reconstructions differ.");
        std::process::exit(1);
    }
}
