// SPDX-License-Identifier: AGPL-3.0-only
//! Oracle for moe_topk_softmax: gate_logits[num_experts] BF16 -> top_k indices
//! (desc, lower-idx tie-break) + softmax-over-all weights (renormalized over
//! top-k when normalize=1). Grid(1,1,1) Block 256. CPU ref uses exact exp.
use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kernel_args::KernelLaunch;
struct Rng(u64);
impl Rng {
    fn n(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn u(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * ((self.n() >> 40) as f32 / (1u64 << 24) as f32)
    }
}
fn f32_to_bf16(f: f32) -> u16 {
    let b = f.to_bits();
    ((b.wrapping_add(0x7FFF + ((b >> 16) & 1))) >> 16) as u16
}
fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let ne: usize = a.get(1).map_or(256, |s| s.parse().unwrap());
    let tk: usize = a.get(2).map_or(8, |s| s.parse().unwrap());
    let seed: u64 = a.get(3).map_or(0x51A7, |s| {
        u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0x51A7)
    });
    println!("=== moe_topk_softmax microtest: num_experts={ne} top_k={tk} seed=0x{seed:X} ===");
    let mut r = Rng(seed);
    // realistic gate logits: mostly negative, a few high (like the dumped -1.9..-6)
    let gl: Vec<u16> = (0..ne).map(|_| f32_to_bf16(r.u(-7.0, -1.5))).collect();
    let be = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &be;
    let st = gpu.create_stream()?;
    let glp = {
        let mut bytes = Vec::new();
        for x in &gl {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        let p = gpu.alloc(bytes.len())?;
        gpu.copy_h2d(&bytes, p)?;
        p
    };
    let idx = gpu.alloc(tk * 4)?;
    let wt = gpu.alloc(tk * 4)?;
    let h = gpu.kernel("moe_topk", "moe_topk_softmax")?;
    KernelLaunch::new(gpu, h)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(glp)
        .arg_ptr(idx)
        .arg_ptr(wt)
        .arg_u32(ne as u32)
        .arg_u32(tk as u32)
        .arg_u32(1)
        .launch(st)?;
    gpu.synchronize(st)?;
    let mut ib = vec![0u8; tk * 4];
    gpu.copy_d2h(idx, &mut ib)?;
    let mut wb = vec![0u8; tk * 4];
    gpu.copy_d2h(wt, &mut wb)?;
    let gi: Vec<u32> = ib
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let gw: Vec<f32> = wb
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    // CPU ref
    let lg: Vec<f32> = gl.iter().map(|&x| bf16_to_f32(x)).collect();
    let mut order: Vec<usize> = (0..ne).collect();
    order.sort_by(|&i, &j| lg[j].partial_cmp(&lg[i]).unwrap().then(i.cmp(&j)));
    let top: Vec<usize> = order[..tk].to_vec();
    let mx = lg.iter().cloned().fold(f32::MIN, f32::max);
    let sum: f32 = lg.iter().map(|&v| (v - mx).exp()).sum();
    let mut w: Vec<f32> = top.iter().map(|&i| (lg[i] - mx).exp() / sum).collect();
    let ws: f32 = w.iter().sum();
    for x in w.iter_mut() {
        *x /= ws;
    }
    let idx_match = gi.iter().map(|x| *x as usize).eq(top.iter().cloned());
    let werr: f32 = gw
        .iter()
        .zip(&w)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    println!("gpu_idx={:?}", gi);
    println!("cpu_idx={:?}", top);
    println!("idx_match={idx_match}  max_weight_abs_err={werr:.5}");
    println!(
        "gpu_w={:?}",
        gw.iter()
            .map(|x| (x * 1000.0).round() / 1000.0)
            .collect::<Vec<_>>()
    );
    println!(
        "cpu_w={:?}",
        w.iter()
            .map(|x| (x * 1000.0).round() / 1000.0)
            .collect::<Vec<_>>()
    );
    for p in [glp, idx, wt] {
        gpu.free(p).ok();
    }
    if idx_match && werr < 0.01 {
        println!("RESULT: PASS");
        Ok(())
    } else {
        println!("RESULT: FAIL");
        std::process::exit(1);
    }
}
