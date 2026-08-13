// SPDX-License-Identifier: AGPL-3.0-only
//! BYTE-parity oracle for `w4a16_gemv_cpasync` vs current `w4a16_gemv`
//! (and vs `w4a16_gemv_sw`, which shares `w4a16_gemv_partial`).
//!
//! PASS bar is bit_id == 100% on production 35B shapes including hidden=2048.
//! Exit: 0 all legs byte-identical, 1 any mismatch, 2 kernels absent.

use anyhow::Result;
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const GROUP_SIZE: usize = 16;
const SCALE2: f32 = 0.0123_f32;

const SHAPES: [(&str, usize, usize); 6] = [
    ("gdn in_proj  [12288 x 2048]", 12288, 2048),
    ("attn Q       [ 8192 x 2048]", 8192, 2048),
    ("gdn out_proj [ 2048 x 4096]", 2048, 4096),
    ("attn o_proj  [ 2048 x 2048]", 2048, 2048),
    ("N%4!=0 tail  [ 2050 x 2048]", 2050, 2048),
    ("K-tail       [ 2048 x 2032]", 2048, 2032),
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

fn worst_delta(a: &[u8], b: &[u8]) -> (usize, f32) {
    let mut n_diff = 0usize;
    let mut worst = 0f32;
    for (x, y) in a.chunks_exact(2).zip(b.chunks_exact(2)) {
        if x != y {
            n_diff += 1;
            let fx = bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32();
            let fy = bf16::from_bits(u16::from_le_bytes([y[0], y[1]])).to_f32();
            worst = worst.max((fx - fy).abs());
        }
    }
    (n_diff, worst)
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
    grid_n: u32,
) -> Result<()> {
    KernelLaunch::new(g, kh)
        .grid([div_ceil(n, grid_n), 1, 1])
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

fn gen_inputs(seed: u64, n: usize, k: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut rng = Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xB4B4);
    let a = (0..k)
        .flat_map(|_| bf16::from_f32(rng.r(-1.5, 1.5)).to_bits().to_le_bytes())
        .collect();
    let w = (0..n * k / 2).map(|_| rng.r(0.0, 256.0) as u8).collect();
    let ws = (0..n * k / GROUP_SIZE)
        .map(|_| 0x30u8 + (rng.r(0.0, 24.0) as u8))
        .collect();
    (a, w, ws)
}

fn report(tag: &str, seed: u64, label: &str, a: &[u8], b: &[u8]) -> bool {
    let identical = a == b;
    let (n_diff, worst) = worst_delta(a, b);
    let bit_id = if a.is_empty() {
        100.0
    } else {
        100.0 * (1.0 - n_diff as f64 / (a.len() / 2) as f64)
    };
    println!(
        "seed {seed:>5}  {label}  {tag:<18} byte-identical={identical:<5} \
         bit_id={bit_id:>7.3}%  diff_elems={n_diff:<7} max|delta|={worst:.6}"
    );
    identical
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let m1 = g.kernel("w4a16_gemv", "w4a16_gemv");
    let sw = g.kernel("w4a16_gemv", "w4a16_gemv_sw");
    let cp = g.kernel("w4a16_gemv", "w4a16_gemv_cpasync");
    let (Ok(m1), Ok(sw), Ok(cp)) = (m1, sw, cp) else {
        println!("w4a16 GEMV cpasync kernels absent — SKIP");
        std::process::exit(2);
    };

    let mut clean = true;
    let mut control_ok = true;
    for seed in [1u64, 99, 12345] {
        for (label, n, k) in SHAPES {
            let (a, w, ws) = gen_inputs(seed, n, k);
            let a_d = up(g, &a)?;
            let w_d = up(g, &w)?;
            let ws_d = up(g, &ws)?;
            let c_cp = g.alloc(n * 2)?;
            let c_m1 = g.alloc(n * 2)?;
            let c_sw = g.alloc(n * 2)?;
            g.memset(c_cp, 0, n * 2)?;
            g.memset(c_m1, 0, n * 2)?;
            g.memset(c_sw, 0, n * 2)?;

            launch(g, cp, a_d, w_d, ws_d, c_cp, n as u32, k as u32, 4)?;
            launch(g, m1, a_d, w_d, ws_d, c_m1, n as u32, k as u32, 4)?;
            launch(g, sw, a_d, w_d, ws_d, c_sw, n as u32, k as u32, 8)?;
            g.synchronize(0)?;
            let out_cp = down(g, c_cp, n * 2)?;
            let out_m1 = down(g, c_m1, n * 2)?;
            let out_sw = down(g, c_sw, n * 2)?;
            clean &= report("cpasync vs gemv", seed, label, &out_cp, &out_m1);
            clean &= report("cpasync vs sw  ", seed, label, &out_cp, &out_sw);

            let mut pert = a.clone();
            pert[2 * 7] ^= 1;
            let a_pert = up(g, &pert)?;
            g.memset(c_m1, 0, n * 2)?;
            launch(g, m1, a_pert, w_d, ws_d, c_m1, n as u32, k as u32, 4)?;
            g.synchronize(0)?;
            let differs = down(g, c_m1, n * 2)? != out_cp;
            control_ok &= differs;
            println!("seed {seed:>5}  {label}  CONTROL 1-ULP perturbation detected={differs}");
            g.free(a_pert).ok();
            for p in [a_d, w_d, ws_d, c_cp, c_m1, c_sw] {
                g.free(p).ok();
            }
        }
    }

    if !control_ok {
        println!("FAIL — negative control did not mismatch; harness is VACUOUS.");
        std::process::exit(1);
    }
    if clean {
        println!(
            "PASS — w4a16_gemv_cpasync is byte-identical to w4a16_gemv and \
             w4a16_gemv_sw at every 35B production shape."
        );
        Ok(())
    } else {
        println!("FAIL — cpasync is NOT byte-identical (do not ship).");
        std::process::exit(1);
    }
}
