// SPDX-License-Identifier: AGPL-3.0-only

//! Correctness + speed gate for `gated_delta_rule_prefill_regresident` (the
//! register-resident warm-replay GDN recurrence) vs the in-tree scalar
//! reference `gated_delta_rule_prefill` (H in shared memory, token-sequential).
//!
//! Both run the SAME token-sequential FP32 recurrence from the SAME initial
//! H-state; the only difference is the data layout (registers vs smem). They
//! must be token-equal (cosine ~1.0) on BOTH the per-token output and the
//! final H-state — the acceptance class WY4 already operates under. The
//! optional ATLAS_BENCH_ITERS timing loop reports the speedup that motivates
//! the swap.
//!
//! Usage: cargo run --release -p spark-model --example gdn_regresident_microtest \
//!          --features cuda,gpu-examples -- [seq] [seed]

use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

const NK: usize = 16;
const NV: usize = 32;
const KD: usize = 128;
const VD: usize = 128;
const COSINE_GATE: f64 = 0.9999;

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
        lo + (hi - lo) * ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32)
    }
}

fn f32_to_bf16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    ((bits.wrapping_add(0x7FFF + ((bits >> 16) & 1))) >> 16) as u16
}
fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}
fn u16s_to_le(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn f32s_to_le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn upload(gpu: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(b.len())?;
    gpu.copy_h2d(b, p)?;
    Ok(p)
}

fn cos_bf16(a: &[u16], b: &[u16]) -> f64 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for i in 0..a.len() {
        let x = bf16_bits_to_f32(a[i]) as f64;
        let y = bf16_bits_to_f32(b[i]) as f64;
        d += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 { return f64::NAN; }
    d / (na.sqrt() * nb.sqrt())
}
fn cos_f32(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for i in 0..a.len() {
        let (x, y) = (a[i] as f64, b[i] as f64);
        d += x * y; na += x * x; nb += y * y;
    }
    if na == 0.0 || nb == 0.0 { return f64::NAN; }
    d / (na.sqrt() * nb.sqrt())
}

/// Bind the shared GDN-prefill kernel argument list (h_state + output differ
/// per call). Free fn with an explicit lifetime so the returned launch keeps
/// the backend borrow (a closure can't annotate the return lifetime).
#[allow(clippy::too_many_arguments)]
fn bind<'a>(
    kl: KernelLaunch<'a>,
    h: DevicePtr, qp: DevicePtr, kp: DevicePtr, vp: DevicePtr, gp: DevicePtr, bp: DevicePtr, o: DevicePtr,
    seq: u32, qk_stride: u32, v_stride: u32, gb_stride: u32,
) -> KernelLaunch<'a> {
    kl.arg_ptr(h).arg_ptr(qp).arg_ptr(kp).arg_ptr(vp).arg_ptr(gp).arg_ptr(bp).arg_ptr(o)
        .arg_u32(1).arg_u32(seq).arg_u32(NK as u32).arg_u32(NV as u32)
        .arg_u32(KD as u32).arg_u32(VD as u32)
        .arg_u32(qk_stride).arg_u32(v_stride).arg_u32(gb_stride)
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let seq: usize = a.get(1).map_or(256, |s| s.parse().unwrap());
    let seed: u64 = a.get(2).map_or(0x6DEF, |s| {
        u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0x6DEF)
    });
    println!("=== gdn_regresident microtest: seq={seq} nk={NK} nv={NV} kd={KD} vd={VD} seed=0x{seed:X} ===");

    let mut rng = Rng(seed);
    // Bounded inputs so the (un-clamped) prefill recurrence stays numerically
    // stable over the whole suffix — neither the scalar reference nor WY4 clamps
    // the H-norm in prefill, so random large k/v would explode IDENTICALLY in
    // both (a test artifact, not a kernel diff). Realistic GDN inputs are
    // structured/contractive; these small magnitudes mimic that stability.
    let h0: Vec<f32> = (0..NV * KD * VD).map(|_| rng.uniform(-0.02, 0.02)).collect();
    let q: Vec<u16> = (0..seq * NK * KD).map(|_| f32_to_bf16_bits(rng.uniform(-0.25, 0.25))).collect();
    let k: Vec<u16> = (0..seq * NK * KD).map(|_| f32_to_bf16_bits(rng.uniform(-0.25, 0.25))).collect();
    let v: Vec<u16> = (0..seq * NV * VD).map(|_| f32_to_bf16_bits(rng.uniform(-0.25, 0.25))).collect();
    // gate = decay in (0,1); beta in (0, 0.5). [seq, nv] FP32 each.
    let gate: Vec<f32> = (0..seq * NV).map(|_| rng.uniform(0.88, 0.97)).collect();
    let beta: Vec<f32> = (0..seq * NV).map(|_| rng.uniform(0.0, 0.5)).collect();

    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;

    let qp = upload(gpu, &u16s_to_le(&q))?;
    let kp = upload(gpu, &u16s_to_le(&k))?;
    let vp = upload(gpu, &u16s_to_le(&v))?;
    let gp = upload(gpu, &f32s_to_le(&gate))?;
    let bp = upload(gpu, &f32s_to_le(&beta))?;
    // Separate H buffers per kernel (each mutates in place).
    let h_ref = upload(gpu, &f32s_to_le(&h0))?;
    let h_new = upload(gpu, &f32s_to_le(&h0))?;
    let o_ref = gpu.alloc(seq * NV * VD * 2)?;
    let o_new = gpu.alloc(seq * NV * VD * 2)?;

    let qk_stride = (NK * KD) as u32;
    let v_stride = (NV * VD) as u32;
    let gb_stride = NV as u32; // separate gate/beta buffers, one row of NV per token

    // Reference: gated_delta_rule_prefill (H in dynamic smem). Grid (nv, batch, 1).
    let ref_smem = ((KD * VD + 2 * KD) * 4) as u32;
    let ref_h = gpu.kernel("gated_delta_rule", "gated_delta_rule_prefill")?;
    bind(KernelLaunch::new(gpu, ref_h).grid([NV as u32, 1, 1]).block([128, 1, 1]).shared_mem(ref_smem),
         h_ref, qp, kp, vp, gp, bp, o_ref, seq as u32, qk_stride, v_stride, gb_stride)
        .launch(stream)?;
    gpu.synchronize(stream)?;

    // New: gated_delta_rule_prefill_regresident. Grid (nv, batch, vd/4), no smem.
    let new_h = gpu.kernel("gated_delta_rule_regresident", "gated_delta_rule_prefill_regresident")?;
    let gz = (VD / 4) as u32;
    bind(KernelLaunch::new(gpu, new_h).grid([NV as u32, 1, gz]).block([128, 1, 1]),
         h_new, qp, kp, vp, gp, bp, o_new, seq as u32, qk_stride, v_stride, gb_stride)
        .launch(stream)?;
    gpu.synchronize(stream)?;

    // Read back + compare.
    let mut raw_or = vec![0u8; seq * NV * VD * 2];
    let mut raw_on = vec![0u8; seq * NV * VD * 2];
    gpu.copy_d2h(o_ref, &mut raw_or)?;
    gpu.copy_d2h(o_new, &mut raw_on)?;
    let or: Vec<u16> = raw_or.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let on: Vec<u16> = raw_on.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let mut raw_hr = vec![0u8; NV * KD * VD * 4];
    let mut raw_hn = vec![0u8; NV * KD * VD * 4];
    gpu.copy_d2h(h_ref, &mut raw_hr)?;
    gpu.copy_d2h(h_new, &mut raw_hn)?;
    let hr: Vec<f32> = raw_hr.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let hn: Vec<f32> = raw_hn.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let cos_out = cos_bf16(&or, &on);
    let cos_h = cos_f32(&hr, &hn);
    let max_abs_h = hr.iter().zip(&hn).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    println!("cosine(output)={cos_out:.7}  cosine(h_state)={cos_h:.7}  max|dH|={max_abs_h:.3e}");

    // Optional timing A/B (inlined to avoid borrowing KernelLaunch through a closure).
    if let Ok(iters_s) = std::env::var("ATLAS_BENCH_ITERS") {
        let iters: usize = iters_s.parse().unwrap_or(50);
        for _ in 0..10 {
            bind(KernelLaunch::new(gpu, ref_h).grid([NV as u32, 1, 1]).block([128, 1, 1]).shared_mem(ref_smem),
                 h_ref, qp, kp, vp, gp, bp, o_ref, seq as u32, qk_stride, v_stride, gb_stride).launch(stream)?;
        }
        gpu.synchronize(stream)?;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            bind(KernelLaunch::new(gpu, ref_h).grid([NV as u32, 1, 1]).block([128, 1, 1]).shared_mem(ref_smem),
                 h_ref, qp, kp, vp, gp, bp, o_ref, seq as u32, qk_stride, v_stride, gb_stride).launch(stream)?;
        }
        gpu.synchronize(stream)?;
        let ref_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

        for _ in 0..10 {
            bind(KernelLaunch::new(gpu, new_h).grid([NV as u32, 1, gz]).block([128, 1, 1]),
                 h_new, qp, kp, vp, gp, bp, o_new, seq as u32, qk_stride, v_stride, gb_stride).launch(stream)?;
        }
        gpu.synchronize(stream)?;
        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            bind(KernelLaunch::new(gpu, new_h).grid([NV as u32, 1, gz]).block([128, 1, 1]),
                 h_new, qp, kp, vp, gp, bp, o_new, seq as u32, qk_stride, v_stride, gb_stride).launch(stream)?;
        }
        gpu.synchronize(stream)?;
        let new_us = t1.elapsed().as_secs_f64() * 1e6 / iters as f64;

        // WY4 (the production warm-replay kernel we are replacing): H in smem,
        // 4 tokens/iter. smem = H + 8*k/q buffers + warp sums + WY scalars.
        let wy4_us = match gpu.kernel("gated_delta_rule_persistent", "gated_delta_rule_prefill_persistent_wy4") {
            Ok(wy4_h) => {
                let wy4_smem = ((KD * VD * 4 + 8 * KD * 4 + 56) as u32).max(ref_smem);
                for _ in 0..10 {
                    bind(KernelLaunch::new(gpu, wy4_h).grid([NV as u32, 1, 1]).block([128, 1, 1]).shared_mem(wy4_smem),
                         h_ref, qp, kp, vp, gp, bp, o_ref, seq as u32, qk_stride, v_stride, gb_stride).launch(stream)?;
                }
                gpu.synchronize(stream)?;
                let t2 = std::time::Instant::now();
                for _ in 0..iters {
                    bind(KernelLaunch::new(gpu, wy4_h).grid([NV as u32, 1, 1]).block([128, 1, 1]).shared_mem(wy4_smem),
                         h_ref, qp, kp, vp, gp, bp, o_ref, seq as u32, qk_stride, v_stride, gb_stride).launch(stream)?;
                }
                gpu.synchronize(stream)?;
                t2.elapsed().as_secs_f64() * 1e6 / iters as f64
            }
            Err(_) => f64::NAN,
        };
        println!("BENCH ref(smem-H)={ref_us:.2}us  WY4={wy4_us:.2}us  regresident={new_us:.2}us  \
                  vs_ref={:.2}x  vs_WY4={:.2}x", ref_us / new_us, wy4_us / new_us);
    }

    for p in [qp, kp, vp, gp, bp, h_ref, h_new, o_ref, o_new] { gpu.free(p).ok(); }
    if cos_out >= COSINE_GATE && cos_h >= COSINE_GATE {
        println!("RESULT: PASS (out {cos_out:.7} & h {cos_h:.7} >= {COSINE_GATE})");
        Ok(())
    } else {
        println!("RESULT: FAIL (out {cos_out:.7} h {cos_h:.7} < {COSINE_GATE})");
        std::process::exit(1);
    }
}
