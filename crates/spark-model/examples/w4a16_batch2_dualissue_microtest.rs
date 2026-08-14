// SPDX-License-Identifier: AGPL-3.0-only
//! BYTE-parity + optional cold-DRAM timing for `w4a16_gemv_batch2_dualissue`.
//!
//! Dual-issue must match template `w4a16_gemv_batch2` (and therefore 2×
//! `w4a16_gemv`) bit-for-bit. It is not wired to production. Default-on
//! only after `w4a16_batch2_bw_oracle` with
//! `ATLAS_GEMV_BATCH2_CANDIDATE=w4a16_gemv_batch2_dualissue:w4a16_gemv_batch2_dualissue`
//! beats template batch2 by ≥3% on 12288×2048 and 8192×2048.
//!
//! Run:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.6-35b-a3b \
//!   ATLAS_TARGET_QUANT=nvfp4 cargo run -p spark-model --release \
//!     --features cuda,gpu-examples --example w4a16_batch2_dualissue_microtest

use anyhow::Result;
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const GROUP_SIZE: usize = 16;
const SCALE2: f32 = 0.0123_f32;
const M: usize = 2;

const SHAPES: [(&str, usize, usize); 3] = [
    ("gdn in_proj  [12288 x 2048]", 12288, 2048),
    ("attn Q       [ 8192 x 2048]", 8192, 2048),
    ("gdn out      [ 2048 x 4096]", 2048, 4096),
];

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 11) as f64) / ((1u64 << 53) as f64)) as f32
    }
    fn r(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f()
    }
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(1))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn down(g: &dyn GpuBackend, p: DevicePtr, n_bytes: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n_bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

fn launch(
    g: &dyn GpuBackend,
    kh: KernelHandle,
    a: DevicePtr,
    w: DevicePtr,
    ws: DevicePtr,
    c: DevicePtr,
    n: u32,
    k: u32,
) -> Result<()> {
    KernelLaunch::new(g, kh)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(w)
        .arg_ptr(ws)
        .arg_f32(SCALE2)
        .arg_ptr(c)
        .arg_u32(n)
        .arg_u32(k)
        .launch(0)
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let b2 = g.kernel("w4a16_gemv", "w4a16_gemv_batch2");
    let du = g.kernel("w4a16_gemv_batch2_dualissue", "w4a16_gemv_batch2_dualissue");
    let (Ok(b2), Ok(du)) = (b2, du) else {
        println!("batch2 or dualissue kernel absent — SKIP");
        std::process::exit(2);
    };

    let mut clean = true;
    for seed in [1u64, 99, 12345] {
        let mut rng = Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xB4B4);
        for (label, n, k) in SHAPES {
            let a: Vec<u8> = (0..M * k)
                .flat_map(|_| bf16::from_f32(rng.r(-1.5, 1.5)).to_bits().to_le_bytes())
                .collect();
            let w: Vec<u8> = (0..n * k / 2).map(|_| rng.r(0.0, 256.0) as u8).collect();
            let ws: Vec<u8> = (0..n * k / GROUP_SIZE)
                .map(|_| 0x30u8 + (rng.r(0.0, 24.0) as u8))
                .collect();
            let a_d = up(g, &a)?;
            let w_d = up(g, &w)?;
            let ws_d = up(g, &ws)?;
            let c_b2 = g.alloc(M * n * 2)?;
            let c_du = g.alloc(M * n * 2)?;
            g.memset(c_b2, 0, M * n * 2)?;
            g.memset(c_du, 0, M * n * 2)?;
            launch(g, b2, a_d, w_d, ws_d, c_b2, n as u32, k as u32)?;
            launch(g, du, a_d, w_d, ws_d, c_du, n as u32, k as u32)?;
            g.synchronize(0)?;
            let xb = down(g, c_b2, M * n * 2)?;
            let xd = down(g, c_du, M * n * 2)?;
            let identical = xb == xd;
            clean &= identical;
            println!("seed {seed:>5}  {label}  byte-identical={identical}");
            g.free(a_d)?;
            g.free(w_d)?;
            g.free(ws_d)?;
            g.free(c_b2)?;
            g.free(c_du)?;
        }
    }
    if clean {
        println!("PASS — dualissue is byte-identical to template batch2");
        Ok(())
    } else {
        println!("FAIL — dualissue diverged from template batch2");
        std::process::exit(1);
    }
}
