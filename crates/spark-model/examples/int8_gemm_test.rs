// SPDX-License-Identifier: AGPL-3.0-only
//! Correctness + speed for `int8_gemm_t_m128` (W4A8 prefill core).
//! Correctness: small shape vs an exact host reference of the per-block-scaled
//! int8 GEMM (cosine ~1.0 proves the MMA + dequant indexing). Speed: prefill shapes.

use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use std::time::Instant;

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn i8(&mut self) -> i8 {
        (self.next_u64() % 255) as i8 - 127
    }
    fn pos_scale(&mut self) -> f32 {
        0.001 + (self.next_u64() % 1000) as f32 * 0.0005
    }
}

fn up(gpu: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(b.len().max(1))?;
    gpu.copy_h2d(b, p)?;
    Ok(p)
}

fn run(
    gpu: &dyn GpuBackend,
    stream: u64,
    h: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    asc: DevicePtr,
    bsc: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    KernelLaunch::new(gpu, h)
        .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(asc)
        .arg_ptr(bsc)
        .arg_ptr(c)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    let h = gpu.kernel("w4a16", "int8_gemm_t_m128")?;
    let h64 = gpu.kernel("w4a16", "int8_gemm_t_m64")?;
    let hk64 = gpu.kernel("w4a16", "int8_gemm_t_m128_k64")?;
    let h8w = gpu.kernel("w4a16", "int8_gemm_8w")?;
    let h8w3 = gpu.kernel("w4a16", "int8_gemm_8w3")?;
    let h8wl = gpu.kernel("w4a16", "int8_gemm_8w_ldm")?;
    let h8wi = gpu.kernel("w4a16", "int8_gemm_8w_ilp")?;
    let h8wab = gpu.kernel("w4a16", "int8_gemm_8w_ldmab")?;
    let hpipe = gpu.kernel("w4a16", "int8_gemm_8w_pipe")?;
    let hpada = gpu.kernel("w4a16", "int8_gemm_padA")?;
    let hfaith = gpu.kernel("w4a16", "int8_gemm_faith")?;
    let hfaith2 = gpu.kernel("w4a16", "int8_gemm_faith2")?;
    let hfaith3 = gpu.kernel("w4a16", "int8_gemm_faith3")?;
    let hfaith4 = gpu.kernel("w4a16", "int8_gemm_faith4")?;
    let hfaith5 = gpu.kernel("w4a16", "int8_gemm_faith5")?;
    let hfaith6 = gpu.kernel("w4a16", "int8_gemm_faith6")?;
    let hfaith7 = gpu.kernel("w4a16", "int8_gemm_faith7")?;
    let hfaith8 = gpu.kernel("w4a16", "int8_gemm_faith8")?;
    let hfaith9 = gpu.kernel("w4a16", "int8_gemm_faith9")?;
    let hfaith10 = gpu.kernel("w4a16", "int8_gemm_faith10")?;
    let hmmqf = gpu.kernel("w4a16", "int8_gemm_mmqf")?;
    let hmmqf2 = gpu.kernel("w4a16", "int8_gemm_mmqf2")?;
    let hmmqf3 = gpu.kernel("w4a16", "int8_gemm_mmqf3")?;
    let hreqa_il = gpu.kernel("w4a16", "requant_a_bf16_int8_il")?;
    let hmmq = gpu.kernel("w4a16", "int8_gemm_mmq")?;
    let hmmq2 = gpu.kernel("w4a16", "int8_gemm_mmq2")?;
    let hreqw = gpu.kernel("w4a16", "requant_w_nvfp4_int8")?;
    let hreqa = gpu.kernel("w4a16", "requant_a_bf16_int8")?;
    let hsk = gpu.kernel("w4a16", "int8_gemm_splitk")?;
    let hred = gpu.kernel("w4a16", "int8_splitk_reduce")?;

    // ---- correctness: small shape, exact host reference ----
    let (m, n, k) = (128usize, 256usize, 512usize);
    let nb = k / 32;
    let mut rng = Rng(0xC0FFEE);
    let a_i8: Vec<i8> = (0..m * k).map(|_| rng.i8()).collect();
    let b_i8: Vec<i8> = (0..n * k).map(|_| rng.i8()).collect();
    let a_sc: Vec<f32> = (0..m * nb).map(|_| rng.pos_scale()).collect();
    let b_sc: Vec<f32> = (0..n * nb).map(|_| rng.pos_scale()).collect();

    // host ref: C[m,n] = sum_blk ( sum_{k in blk} A[m,k]*B[n,k] ) * As[m,blk]*Bs[n,blk]
    let mut c_ref = vec![0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0f32;
            for blk in 0..nb {
                let mut s = 0i32;
                for kk in 0..32 {
                    let ki = blk * 32 + kk;
                    s += a_i8[mi * k + ki] as i32 * b_i8[ni * k + ki] as i32;
                }
                acc += s as f32 * a_sc[mi * nb + blk] * b_sc[ni * nb + blk];
            }
            c_ref[mi * n + ni] = acc;
        }
    }

    let a_p = up(gpu, bytemuck_i8(&a_i8))?;
    let b_p = up(gpu, bytemuck_i8(&b_i8))?;
    let as_p = up(gpu, bytemuck_f32(&a_sc))?;
    let bs_p = up(gpu, bytemuck_f32(&b_sc))?;
    let c_p = gpu.alloc(m * n * 2)?;
    run(gpu, stream, h, a_p, b_p, as_p, bs_p, c_p, m, n, k)?;
    gpu.synchronize(stream)?;
    let mut raw = vec![0u8; m * n * 2];
    gpu.copy_d2h(c_p, &mut raw)?;
    let c_gpu: Vec<f32> = raw
        .chunks_exact(2)
        .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();

    let (mut dot, mut na, mut nbb) = (0f64, 0f64, 0f64);
    let mut maxrel = 0f64;
    for i in 0..m * n {
        let (x, y) = (c_ref[i] as f64, c_gpu[i] as f64);
        dot += x * y;
        na += x * x;
        nbb += y * y;
        if x.abs() > 1.0 {
            maxrel = maxrel.max(((x - y).abs() / x.abs()).min(9.9));
        }
    }
    let cos = dot / (na.sqrt() * nbb.sqrt());
    println!("int8_gemm correctness {m}x{n}x{k}: cosine={cos:.6}  max_rel={maxrel:.4}");
    println!("  ref[0..4]={:?}  gpu[0..4]={:?}", &c_ref[..4], &c_gpu[..4]);
    let pass = cos > 0.999;
    println!("  RESULT: {}", if pass { "PASS" } else { "FAIL" });

    // ---- 8w_ldm correctness (ldmatrix fragment must match host ref) ----
    {
        let c2 = gpu.alloc(m * n * 2)?;
        KernelLaunch::new(gpu, h8wl)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(a_p)
            .arg_ptr(b_p)
            .arg_ptr(as_p)
            .arg_ptr(bs_p)
            .arg_ptr(c2)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)?;
        gpu.synchronize(stream)?;
        let mut raw2 = vec![0u8; m * n * 2];
        gpu.copy_d2h(c2, &mut raw2)?;
        let cg: Vec<f32> = raw2
            .chunks_exact(2)
            .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let (mut d, mut nr, mut ng) = (0f64, 0f64, 0f64);
        for i in 0..m * n {
            let (x, y) = (c_ref[i] as f64, cg[i] as f64);
            d += x * y;
            nr += x * x;
            ng += y * y;
        }
        let cl = d / (nr.sqrt() * ng.sqrt());
        println!(
            "8w_ldm (ldmatrix.x4) correctness: cosine={cl:.6}  RESULT: {}",
            if cl > 0.999 { "PASS" } else { "FAIL" }
        );
        let _ = gpu.free(c2);
    }
    {
        let c3 = gpu.alloc(m * n * 2)?;
        KernelLaunch::new(gpu, hmmq)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(a_p)
            .arg_ptr(b_p)
            .arg_ptr(as_p)
            .arg_ptr(bs_p)
            .arg_ptr(c3)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)?;
        gpu.synchronize(stream)?;
        let mut r3 = vec![0u8; m * n * 2];
        gpu.copy_d2h(c3, &mut r3)?;
        let cg: Vec<f32> = r3
            .chunks_exact(2)
            .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let (mut d, mut nr, mut ng) = (0f64, 0f64, 0f64);
        for i in 0..m * n {
            let (x, y) = (c_ref[i] as f64, cg[i] as f64);
            d += x * y;
            nr += x * x;
            ng += y * y;
        }
        println!(
            "MMQ-tile correctness: cosine={:.6}  RESULT: {}",
            d / (nr.sqrt() * ng.sqrt()),
            if d / (nr.sqrt() * ng.sqrt()) > 0.999 {
                "PASS"
            } else {
                "FAIL"
            }
        );
        let _ = gpu.free(c3);
    }
    {
        let c4 = gpu.alloc(m * n * 2)?;
        KernelLaunch::new(gpu, h8wab)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(a_p)
            .arg_ptr(b_p)
            .arg_ptr(as_p)
            .arg_ptr(bs_p)
            .arg_ptr(c4)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)?;
        gpu.synchronize(stream)?;
        let mut r4 = vec![0u8; m * n * 2];
        gpu.copy_d2h(c4, &mut r4)?;
        let cg: Vec<f32> = r4
            .chunks_exact(2)
            .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let (mut d, mut nr, mut ng) = (0f64, 0f64, 0f64);
        for i in 0..m * n {
            let (x, y) = (c_ref[i] as f64, cg[i] as f64);
            d += x * y;
            nr += x * x;
            ng += y * y;
        }
        println!(
            "8w_ldmAB (ldmatrix A+B) correctness: cosine={:.6}  RESULT: {}",
            d / (nr.sqrt() * ng.sqrt()),
            if d / (nr.sqrt() * ng.sqrt()) > 0.999 {
                "PASS"
            } else {
                "FAIL"
            }
        );
        let _ = gpu.free(c4);
    }
    {
        let c5 = gpu.alloc(m * n * 2)?;
        KernelLaunch::new(gpu, hpipe)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
            .block([512, 1, 1])
            .arg_ptr(a_p)
            .arg_ptr(b_p)
            .arg_ptr(as_p)
            .arg_ptr(bs_p)
            .arg_ptr(c5)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)?;
        gpu.synchronize(stream)?;
        let mut r5 = vec![0u8; m * n * 2];
        gpu.copy_d2h(c5, &mut r5)?;
        let cg: Vec<f32> = r5
            .chunks_exact(2)
            .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let (mut d, mut nr, mut ng) = (0f64, 0f64, 0f64);
        for i in 0..m * n {
            let (x, y) = (c_ref[i] as f64, cg[i] as f64);
            d += x * y;
            nr += x * x;
            ng += y * y;
        }
        println!(
            "8w_pipe (occ 512) correctness: cosine={:.6}  RESULT: {}",
            d / (nr.sqrt() * ng.sqrt()),
            if d / (nr.sqrt() * ng.sqrt()) > 0.999 {
                "PASS"
            } else {
                "FAIL"
            }
        );
        let _ = gpu.free(c5);
    }
    {
        let c6 = gpu.alloc(m * n * 2)?;
        KernelLaunch::new(gpu, hpada)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(a_p)
            .arg_ptr(b_p)
            .arg_ptr(as_p)
            .arg_ptr(bs_p)
            .arg_ptr(c6)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)?;
        gpu.synchronize(stream)?;
        let mut r6 = vec![0u8; m * n * 2];
        gpu.copy_d2h(c6, &mut r6)?;
        let cg: Vec<f32> = r6
            .chunks_exact(2)
            .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let (mut d, mut nr, mut ng) = (0f64, 0f64, 0f64);
        for i in 0..m * n {
            let (x, y) = (c_ref[i] as f64, cg[i] as f64);
            d += x * y;
            nr += x * x;
            ng += y * y;
        }
        println!(
            "padA (bank-fix ldmatrix) correctness: cosine={:.6}  RESULT: {}",
            d / (nr.sqrt() * ng.sqrt()),
            if d / (nr.sqrt() * ng.sqrt()) > 0.999 {
                "PASS"
            } else {
                "FAIL"
            }
        );
        let _ = gpu.free(c6);
    }
    {
        let c7 = gpu.alloc(m * n * 2)?;
        KernelLaunch::new(gpu, hfaith)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(a_p)
            .arg_ptr(b_p)
            .arg_ptr(as_p)
            .arg_ptr(bs_p)
            .arg_ptr(c7)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)?;
        gpu.synchronize(stream)?;
        let mut r7 = vec![0u8; m * n * 2];
        gpu.copy_d2h(c7, &mut r7)?;
        let cg: Vec<f32> = r7
            .chunks_exact(2)
            .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let (mut d, mut nr, mut ng) = (0f64, 0f64, 0f64);
        for i in 0..m * n {
            let (x, y) = (c_ref[i] as f64, cg[i] as f64);
            d += x * y;
            nr += x * x;
            ng += y * y;
        }
        println!(
            "FAITH (llama-MMQ port) correctness: cosine={:.6}  RESULT: {}",
            d / (nr.sqrt() * ng.sqrt()),
            if d / (nr.sqrt() * ng.sqrt()) > 0.999 {
                "PASS"
            } else {
                "FAIL"
            }
        );
        let _ = gpu.free(c7);
    }
    {
        let c8 = gpu.alloc(m * n * 2)?;
        KernelLaunch::new(gpu, hfaith2)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(a_p)
            .arg_ptr(b_p)
            .arg_ptr(as_p)
            .arg_ptr(bs_p)
            .arg_ptr(c8)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)?;
        gpu.synchronize(stream)?;
        let mut r8 = vec![0u8; m * n * 2];
        gpu.copy_d2h(c8, &mut r8)?;
        let cg: Vec<f32> = r8
            .chunks_exact(2)
            .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let (mut d, mut nr, mut ng) = (0f64, 0f64, 0f64);
        for i in 0..m * n {
            let (x, y) = (c_ref[i] as f64, cg[i] as f64);
            d += x * y;
            nr += x * x;
            ng += y * y;
        }
        println!(
            "FAITH2 (big-K rolling) correctness: cosine={:.6}  RESULT: {}",
            d / (nr.sqrt() * ng.sqrt()),
            if d / (nr.sqrt() * ng.sqrt()) > 0.999 {
                "PASS"
            } else {
                "FAIL"
            }
        );
        let _ = gpu.free(c8);
    }
    {
        let c9 = gpu.alloc(m * n * 2)?;
        KernelLaunch::new(gpu, hfaith3)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(a_p)
            .arg_ptr(b_p)
            .arg_ptr(as_p)
            .arg_ptr(bs_p)
            .arg_ptr(c9)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)?;
        gpu.synchronize(stream)?;
        let mut r9 = vec![0u8; m * n * 2];
        gpu.copy_d2h(c9, &mut r9)?;
        let cg: Vec<f32> = r9
            .chunks_exact(2)
            .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let (mut d, mut nr, mut ng) = (0f64, 0f64, 0f64);
        for i in 0..m * n {
            let (x, y) = (c_ref[i] as f64, cg[i] as f64);
            d += x * y;
            nr += x * x;
            ng += y * y;
        }
        println!(
            "FAITH3 (B-frag ILP) correctness: cosine={:.6}  RESULT: {}",
            d / (nr.sqrt() * ng.sqrt()),
            if d / (nr.sqrt() * ng.sqrt()) > 0.999 {
                "PASS"
            } else {
                "FAIL"
            }
        );
        let _ = gpu.free(c9);
    }
    {
        let c10 = gpu.alloc(m * n * 2)?;
        KernelLaunch::new(gpu, hfaith4)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
            .block([512, 1, 1])
            .arg_ptr(a_p)
            .arg_ptr(b_p)
            .arg_ptr(as_p)
            .arg_ptr(bs_p)
            .arg_ptr(c10)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)?;
        gpu.synchronize(stream)?;
        let mut r10 = vec![0u8; m * n * 2];
        gpu.copy_d2h(c10, &mut r10)?;
        let cg: Vec<f32> = r10
            .chunks_exact(2)
            .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let (mut d, mut nr, mut ng) = (0f64, 0f64, 0f64);
        for i in 0..m * n {
            let (x, y) = (c_ref[i] as f64, cg[i] as f64);
            d += x * y;
            nr += x * x;
            ng += y * y;
        }
        println!(
            "FAITH4 (512-thread occ) correctness: cosine={:.6}  RESULT: {}",
            d / (nr.sqrt() * ng.sqrt()),
            if d / (nr.sqrt() * ng.sqrt()) > 0.999 {
                "PASS"
            } else {
                "FAIL"
            }
        );
        let _ = gpu.free(c10);
    }
    {
        let c11 = gpu.alloc(m * n * 2)?;
        KernelLaunch::new(gpu, hmmq2)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(a_p)
            .arg_ptr(b_p)
            .arg_ptr(as_p)
            .arg_ptr(bs_p)
            .arg_ptr(c11)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)?;
        gpu.synchronize(stream)?;
        let mut r11 = vec![0u8; m * n * 2];
        gpu.copy_d2h(c11, &mut r11)?;
        let cg: Vec<f32> = r11
            .chunks_exact(2)
            .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let (mut d, mut nr, mut ng) = (0f64, 0f64, 0f64);
        for i in 0..m * n {
            let (x, y) = (c_ref[i] as f64, cg[i] as f64);
            d += x * y;
            nr += x * x;
            ng += y * y;
        }
        println!(
            "MMQ2 (faith2 + double-buffer) correctness: cosine={:.6}  RESULT: {}",
            d / (nr.sqrt() * ng.sqrt()),
            if d / (nr.sqrt() * ng.sqrt()) > 0.999 {
                "PASS"
            } else {
                "FAIL"
            }
        );
        let _ = gpu.free(c11);
    }

    // ---- END-TO-END requant precision: NVFP4 weights -> int8 (requant_w) +
    //      bf16 acts -> int8 (requant_a) -> faith2, vs host full-precision
    //      dequant GEMM. Gates the whole int8 integration (per-16 NVFP4 scales
    //      re-blocked to per-32 int8 must stay coherent). ----
    {
        let (m2, n2, k2) = (256usize, 256usize, 512usize);
        let nb2 = k2 / 32;
        let mut rng = Rng(0xBEEF);
        // NVFP4 weights: packed E2M1 nibbles [n2, k2/2], per-16 E4M3 scales [n2, k2/16], scale2.
        let e2m1_lut = [
            0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0,
            -6.0,
        ];
        let e4m3_dec = |b: u8| -> f32 {
            let s = (b >> 7) & 1;
            let e = (b >> 3) & 0xF;
            let mm = b & 0x7;
            let v = if e == 0 {
                (mm as f32) * 0.001953125
            } else {
                (1.0 + (mm as f32) / 8.0) * 2f32.powi(e as i32 - 7)
            };
            if s == 1 { -v } else { v }
        };
        let scale2 = 0.05f32;
        let nibbles: Vec<u8> = (0..n2 * k2).map(|_| (rng.next_u64() % 16) as u8).collect();
        let mut packed = vec![0u8; n2 * k2 / 2];
        for i in 0..n2 * k2 / 2 {
            packed[i] = nibbles[2 * i] | (nibbles[2 * i + 1] << 4);
        }
        // per-16 E4M3 scale bytes, constrained to e in [1,14], m in [0,7], s=0 (clean positive)
        let e4m3_bytes: Vec<u8> = (0..n2 * (k2 / 16))
            .map(|_| {
                let e = 1 + (rng.next_u64() % 14) as u8;
                let mm = (rng.next_u64() % 8) as u8;
                (e << 3) | mm
            })
            .collect();
        // host dequant W[n,k] = LUT[nibble] * e4m3(scale16) * scale2
        let mut w_real = vec![0f32; n2 * k2];
        for ni in 0..n2 {
            for ki in 0..k2 {
                let nib = nibbles[ni * k2 + ki] as usize;
                let s16 = e4m3_dec(e4m3_bytes[ni * (k2 / 16) + ki / 16]) * scale2;
                w_real[ni * k2 + ki] = e2m1_lut[nib] * s16;
            }
        }
        // bf16 activations
        let a_f: Vec<f32> = (0..m2 * k2).map(|_| (rng.i8() as f32) * 0.02).collect();
        let a_bf16_bits: Vec<u16> = a_f.iter().map(|&x| (x.to_bits() >> 16) as u16).collect();
        let a_bf16_round: Vec<f32> = a_bf16_bits.iter().map(|&b| bf16_bits_to_f32(b)).collect();
        // host ref: C[m,n] = sum_k a_round[m,k] * w_real[n,k]
        let mut cref = vec![0f32; m2 * n2];
        for mi in 0..m2 {
            for ni in 0..n2 {
                let mut acc = 0f32;
                for ki in 0..k2 {
                    acc += a_bf16_round[mi * k2 + ki] * w_real[ni * k2 + ki];
                }
                cref[mi * n2 + ni] = acc;
            }
        }
        // GPU: upload, requant_w, requant_a, faith2
        let packed_p = up(gpu, &packed)?;
        let e4m3_p = up(gpu, &e4m3_bytes)?;
        let a_bf16_u8: Vec<u8> = a_bf16_bits.iter().flat_map(|&b| b.to_le_bytes()).collect();
        let abf_p = up(gpu, &a_bf16_u8)?;
        let wi8_p = gpu.alloc(n2 * k2)?;
        let wsc_p = gpu.alloc(n2 * nb2 * 4)?;
        let ai8_p = gpu.alloc(m2 * k2)?;
        let asc_p = gpu.alloc(m2 * nb2 * 4)?;
        let wblocks = (n2 * nb2) as u32;
        KernelLaunch::new(gpu, hreqw)
            .grid([wblocks.div_ceil(128), 1, 1])
            .block([128, 1, 1])
            .arg_ptr(packed_p)
            .arg_ptr(e4m3_p)
            .arg_f32(scale2)
            .arg_ptr(wi8_p)
            .arg_ptr(wsc_p)
            .arg_u32(n2 as u32)
            .arg_u32(k2 as u32)
            .launch(stream)?;
        let ablocks = (m2 * nb2) as u32;
        KernelLaunch::new(gpu, hreqa)
            .grid([ablocks.div_ceil(128), 1, 1])
            .block([128, 1, 1])
            .arg_ptr(abf_p)
            .arg_ptr(ai8_p)
            .arg_ptr(asc_p)
            .arg_u32(m2 as u32)
            .arg_u32(k2 as u32)
            .launch(stream)?;
        let ce = gpu.alloc(m2 * n2 * 2)?;
        KernelLaunch::new(gpu, hfaith2)
            .grid([(n2 as u32).div_ceil(128), (m2 as u32).div_ceil(128), 1])
            .block([256, 1, 1])
            .arg_ptr(ai8_p)
            .arg_ptr(wi8_p)
            .arg_ptr(asc_p)
            .arg_ptr(wsc_p)
            .arg_ptr(ce)
            .arg_u32(m2 as u32)
            .arg_u32(n2 as u32)
            .arg_u32(k2 as u32)
            .launch(stream)?;
        gpu.synchronize(stream)?;
        let mut re = vec![0u8; m2 * n2 * 2];
        gpu.copy_d2h(ce, &mut re)?;
        let cg: Vec<f32> = re
            .chunks_exact(2)
            .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let (mut d, mut nr, mut ng) = (0f64, 0f64, 0f64);
        for i in 0..m2 * n2 {
            let (x, y) = (cref[i] as f64, cg[i] as f64);
            d += x * y;
            nr += x * x;
            ng += y * y;
        }
        let cos = d / (nr.sqrt() * ng.sqrt());
        println!(
            "REQUANT e2e (NVFP4 w->int8 + bf16 a->int8 -> faith2) vs host dequant: cosine={cos:.6}  RESULT: {}",
            if cos > 0.999 { "PASS" } else { "FAIL" }
        );

        // faith5 path: interleaved requant (requant_a_il) + faith5 (int2 B-load)
        // vs the SAME host reference. Must match faith2 (same fragment values).
        let ai8_il_p = gpu.alloc(m2 * k2)?;
        let asc5_p = gpu.alloc(m2 * nb2 * 4)?;
        KernelLaunch::new(gpu, hreqa_il)
            .grid([ablocks.div_ceil(128), 1, 1])
            .block([128, 1, 1])
            .arg_ptr(abf_p)
            .arg_ptr(ai8_il_p)
            .arg_ptr(asc5_p)
            .arg_u32(m2 as u32)
            .arg_u32(k2 as u32)
            .launch(stream)?;
        let ce5 = gpu.alloc(m2 * n2 * 2)?;
        KernelLaunch::new(gpu, hfaith5)
            .grid([(n2 as u32).div_ceil(128), (m2 as u32).div_ceil(128), 1])
            .block([256, 1, 1])
            .arg_ptr(ai8_il_p)
            .arg_ptr(wi8_p)
            .arg_ptr(asc5_p)
            .arg_ptr(wsc_p)
            .arg_ptr(ce5)
            .arg_u32(m2 as u32)
            .arg_u32(n2 as u32)
            .arg_u32(k2 as u32)
            .launch(stream)?;
        gpu.synchronize(stream)?;
        let mut re5 = vec![0u8; m2 * n2 * 2];
        gpu.copy_d2h(ce5, &mut re5)?;
        let cg5: Vec<f32> = re5
            .chunks_exact(2)
            .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let (mut d5, mut nr5, mut ng5) = (0f64, 0f64, 0f64);
        for i in 0..m2 * n2 {
            let (x, y) = (cref[i] as f64, cg5[i] as f64);
            d5 += x * y;
            nr5 += x * x;
            ng5 += y * y;
        }
        let cos5 = d5 / (nr5.sqrt() * ng5.sqrt());
        println!(
            "REQUANT e2e faith5 (interleaved-q8_1 B-load) vs host dequant: cosine={cos5:.6}  RESULT: {}",
            if cos5 > 0.999 { "PASS" } else { "FAIL" }
        );

        // faith6 path: same non-interleaved int8 inputs as faith2 (3-phase WAR-break
        // schedule), MUST be bit-identical to faith2 vs host.
        let ce6 = gpu.alloc(m2 * n2 * 2)?;
        KernelLaunch::new(gpu, hfaith6)
            .grid([(n2 as u32).div_ceil(128), (m2 as u32).div_ceil(128), 1])
            .block([256, 1, 1])
            .arg_ptr(ai8_p)
            .arg_ptr(wi8_p)
            .arg_ptr(asc_p)
            .arg_ptr(wsc_p)
            .arg_ptr(ce6)
            .arg_u32(m2 as u32)
            .arg_u32(n2 as u32)
            .arg_u32(k2 as u32)
            .launch(stream)?;
        gpu.synchronize(stream)?;
        let mut re6 = vec![0u8; m2 * n2 * 2];
        gpu.copy_d2h(ce6, &mut re6)?;
        let cg6: Vec<f32> = re6
            .chunks_exact(2)
            .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let (mut d6, mut nr6, mut ng6) = (0f64, 0f64, 0f64);
        for i in 0..m2 * n2 {
            let (x, y) = (cref[i] as f64, cg6[i] as f64);
            d6 += x * y;
            nr6 += x * x;
            ng6 += y * y;
        }
        let cos6 = d6 / (nr6.sqrt() * ng6.sqrt());
        println!(
            "REQUANT e2e faith6 (WAR-break 3-phase schedule) vs host dequant: cosine={cos6:.6}  RESULT: {}",
            if cos6 > 0.999 { "PASS" } else { "FAIL" }
        );

        // faith7 path: same non-interleaved int8 inputs as faith2 (64Nx32M reuse
        // reshape), MUST be bit-identical to faith2 vs host.
        let ce7 = gpu.alloc(m2 * n2 * 2)?;
        KernelLaunch::new(gpu, hfaith7)
            .grid([(n2 as u32).div_ceil(128), (m2 as u32).div_ceil(128), 1])
            .block([256, 1, 1])
            .arg_ptr(ai8_p)
            .arg_ptr(wi8_p)
            .arg_ptr(asc_p)
            .arg_ptr(wsc_p)
            .arg_ptr(ce7)
            .arg_u32(m2 as u32)
            .arg_u32(n2 as u32)
            .arg_u32(k2 as u32)
            .launch(stream)?;
        gpu.synchronize(stream)?;
        let mut re7 = vec![0u8; m2 * n2 * 2];
        gpu.copy_d2h(ce7, &mut re7)?;
        let cg7: Vec<f32> = re7
            .chunks_exact(2)
            .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let (mut d7, mut nr7, mut ng7) = (0f64, 0f64, 0f64);
        for i in 0..m2 * n2 {
            let (x, y) = (cref[i] as f64, cg7[i] as f64);
            d7 += x * y;
            nr7 += x * x;
            ng7 += y * y;
        }
        let cos7 = d7 / (nr7.sqrt() * ng7.sqrt());
        println!(
            "REQUANT e2e faith7 (64Nx32M 4:1-reuse reshape) vs host dequant: cosine={cos7:.6}  RESULT: {}",
            if cos7 > 0.999 { "PASS" } else { "FAIL" }
        );

        // faith8 path: 2-stage cp.async pipeline, same inputs as faith2, MUST be bit-identical.
        let ce8 = gpu.alloc(m2 * n2 * 2)?;
        KernelLaunch::new(gpu, hfaith8)
            .grid([(n2 as u32).div_ceil(128), (m2 as u32).div_ceil(128), 1])
            .block([256, 1, 1])
            .arg_ptr(ai8_p)
            .arg_ptr(wi8_p)
            .arg_ptr(asc_p)
            .arg_ptr(wsc_p)
            .arg_ptr(ce8)
            .arg_u32(m2 as u32)
            .arg_u32(n2 as u32)
            .arg_u32(k2 as u32)
            .launch(stream)?;
        gpu.synchronize(stream)?;
        let mut re8 = vec![0u8; m2 * n2 * 2];
        gpu.copy_d2h(ce8, &mut re8)?;
        let cg8: Vec<f32> = re8
            .chunks_exact(2)
            .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let (mut d8, mut nr8, mut ng8) = (0f64, 0f64, 0f64);
        for i in 0..m2 * n2 {
            let (x, y) = (cref[i] as f64, cg8[i] as f64);
            d8 += x * y;
            nr8 += x * x;
            ng8 += y * y;
        }
        let cos8 = d8 / (nr8.sqrt() * ng8.sqrt());
        println!(
            "REQUANT e2e faith8 (2-stage cp.async pipeline) vs host dequant: cosine={cos8:.6}  RESULT: {}",
            if cos8 > 0.999 { "PASS" } else { "FAIL" }
        );

        // faith9 path: ITER_K=256, same inputs as faith2, MUST be bit-identical.
        let ce9 = gpu.alloc(m2 * n2 * 2)?;
        KernelLaunch::new(gpu, hfaith9)
            .grid([(n2 as u32).div_ceil(128), (m2 as u32).div_ceil(128), 1])
            .block([256, 1, 1])
            .arg_ptr(ai8_p)
            .arg_ptr(wi8_p)
            .arg_ptr(asc_p)
            .arg_ptr(wsc_p)
            .arg_ptr(ce9)
            .arg_u32(m2 as u32)
            .arg_u32(n2 as u32)
            .arg_u32(k2 as u32)
            .launch(stream)?;
        gpu.synchronize(stream)?;
        let mut re9 = vec![0u8; m2 * n2 * 2];
        gpu.copy_d2h(ce9, &mut re9)?;
        let cg9: Vec<f32> = re9
            .chunks_exact(2)
            .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let (mut d9, mut nr9, mut ng9) = (0f64, 0f64, 0f64);
        for i in 0..m2 * n2 {
            let (x, y) = (cref[i] as f64, cg9[i] as f64);
            d9 += x * y;
            nr9 += x * x;
            ng9 += y * y;
        }
        let cos9 = d9 / (nr9.sqrt() * ng9.sqrt());
        println!(
            "REQUANT e2e faith9 (ITER_K=256) vs host dequant: cosine={cos9:.6}  RESULT: {}",
            if cos9 > 0.999 { "PASS" } else { "FAIL" }
        );

        // faith10 path: 16N x 128M per-warp (128-token weight reuse), same inputs as faith2, MUST be bit-identical.
        let ce10 = gpu.alloc(m2 * n2 * 2)?;
        KernelLaunch::new(gpu, hfaith10)
            .grid([(n2 as u32).div_ceil(128), (m2 as u32).div_ceil(128), 1])
            .block([256, 1, 1])
            .arg_ptr(ai8_p)
            .arg_ptr(wi8_p)
            .arg_ptr(asc_p)
            .arg_ptr(wsc_p)
            .arg_ptr(ce10)
            .arg_u32(m2 as u32)
            .arg_u32(n2 as u32)
            .arg_u32(k2 as u32)
            .launch(stream)?;
        gpu.synchronize(stream)?;
        let mut re10 = vec![0u8; m2 * n2 * 2];
        gpu.copy_d2h(ce10, &mut re10)?;
        let cg10: Vec<f32> = re10
            .chunks_exact(2)
            .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let (mut d10, mut nr10, mut ng10) = (0f64, 0f64, 0f64);
        for i in 0..m2 * n2 {
            let (x, y) = (cref[i] as f64, cg10[i] as f64);
            d10 += x * y;
            nr10 += x * x;
            ng10 += y * y;
        }
        let cos10 = d10 / (nr10.sqrt() * ng10.sqrt());
        println!(
            "REQUANT e2e faith10 (128-token weight reuse) vs host dequant: cosine={cos10:.6}  RESULT: {}",
            if cos10 > 0.999 { "PASS" } else { "FAIL" }
        );

        // mmqf path: faithful llama-MMQ port (ITER_K=256 + resident A reused across full 128-token tile + co-resident scales). MUST be bit-identical.
        let cemf = gpu.alloc(m2 * n2 * 2)?;
        KernelLaunch::new(gpu, hmmqf)
            .grid([(n2 as u32).div_ceil(128), (m2 as u32).div_ceil(128), 1])
            .block([256, 1, 1])
            .arg_ptr(ai8_p)
            .arg_ptr(wi8_p)
            .arg_ptr(asc_p)
            .arg_ptr(wsc_p)
            .arg_ptr(cemf)
            .arg_u32(m2 as u32)
            .arg_u32(n2 as u32)
            .arg_u32(k2 as u32)
            .launch(stream)?;
        gpu.synchronize(stream)?;
        let mut remf = vec![0u8; m2 * n2 * 2];
        gpu.copy_d2h(cemf, &mut remf)?;
        let cgmf: Vec<f32> = remf
            .chunks_exact(2)
            .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let (mut dmf, mut nrmf, mut ngmf) = (0f64, 0f64, 0f64);
        for i in 0..m2 * n2 {
            let (x, y) = (cref[i] as f64, cgmf[i] as f64);
            dmf += x * y;
            nrmf += x * x;
            ngmf += y * y;
        }
        let cosmf = dmf / (nrmf.sqrt() * ngmf.sqrt());
        println!(
            "REQUANT e2e mmqf (faithful llama-MMQ port) vs host dequant: cosine={cosmf:.6}  RESULT: {}",
            if cosmf > 0.999 { "PASS" } else { "FAIL" }
        );

        // mmqf3 path: ILP-max MMA schedule (independent acc banks + pipelined B). MUST be bit-identical.
        let cemf3 = gpu.alloc(m2 * n2 * 2)?;
        KernelLaunch::new(gpu, hmmqf3)
            .grid([(n2 as u32).div_ceil(128), (m2 as u32).div_ceil(128), 1])
            .block([256, 1, 1])
            .arg_ptr(ai8_p)
            .arg_ptr(wi8_p)
            .arg_ptr(asc_p)
            .arg_ptr(wsc_p)
            .arg_ptr(cemf3)
            .arg_u32(m2 as u32)
            .arg_u32(n2 as u32)
            .arg_u32(k2 as u32)
            .launch(stream)?;
        gpu.synchronize(stream)?;
        let mut remf3 = vec![0u8; m2 * n2 * 2];
        gpu.copy_d2h(cemf3, &mut remf3)?;
        let cgmf3: Vec<f32> = remf3
            .chunks_exact(2)
            .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let (mut dmf3, mut nrmf3, mut ngmf3) = (0f64, 0f64, 0f64);
        for i in 0..m2 * n2 {
            let (x, y) = (cref[i] as f64, cgmf3[i] as f64);
            dmf3 += x * y;
            nrmf3 += x * x;
            ngmf3 += y * y;
        }
        let cosmf3 = dmf3 / (nrmf3.sqrt() * ngmf3.sqrt());
        println!(
            "REQUANT e2e mmqf3 (ILP-max MMA schedule) vs host dequant: cosine={cosmf3:.6}  RESULT: {}",
            if cosmf3 > 0.999 { "PASS" } else { "FAIL" }
        );

        for p in [
            packed_p, e4m3_p, abf_p, wi8_p, wsc_p, ai8_p, asc_p, ce, ai8_il_p, asc5_p, ce5, ce6,
            ce7, ce8, ce9, ce10, cemf, cemf3,
        ] {
            let _ = gpu.free(p);
        }
    }

    // ---- speed: prefill shapes ----
    println!("\n=== int8 speed (TFLOP/s) ===");
    for &(label, m, n, k) in &[
        ("gate/up M=4096", 4096usize, 17408usize, 5120usize),
        ("down    M=4096", 4096, 5120, 17408),
    ] {
        let nb = k / 32;
        let mut rng = Rng(7);
        let a_i8: Vec<i8> = (0..m * k).map(|_| rng.i8()).collect();
        let b_i8: Vec<i8> = (0..n * k).map(|_| rng.i8()).collect();
        let a_sc: Vec<f32> = (0..m * nb).map(|_| rng.pos_scale()).collect();
        let b_sc: Vec<f32> = (0..n * nb).map(|_| rng.pos_scale()).collect();
        let a_p = up(gpu, bytemuck_i8(&a_i8))?;
        let b_p = up(gpu, bytemuck_i8(&b_i8))?;
        let as_p = up(gpu, bytemuck_f32(&a_sc))?;
        let bs_p = up(gpu, bytemuck_f32(&b_sc))?;
        let c_p = gpu.alloc(m * n * 2)?;
        for _ in 0..3 {
            run(gpu, stream, h, a_p, b_p, as_p, bs_p, c_p, m, n, k)?;
        }
        gpu.synchronize(stream)?;
        let iters = 30;
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        // M128
        let t0 = Instant::now();
        for _ in 0..iters {
            run(gpu, stream, h, a_p, b_p, as_p, bs_p, c_p, m, n, k)?;
        }
        gpu.synchronize(stream)?;
        let tf128 = flops / (t0.elapsed().as_secs_f64() / iters as f64) / 1e12;
        // M64 (grid m/64)
        let launch64 = || -> Result<()> {
            KernelLaunch::new(gpu, h64)
                .grid([n.div_ceil(128) as u32, m.div_ceil(64) as u32, 1])
                .block([128, 1, 1])
                .arg_ptr(a_p)
                .arg_ptr(b_p)
                .arg_ptr(as_p)
                .arg_ptr(bs_p)
                .arg_ptr(c_p)
                .arg_u32(m as u32)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .launch(stream)
        };
        for _ in 0..3 {
            launch64()?;
        }
        gpu.synchronize(stream)?;
        let t1 = Instant::now();
        for _ in 0..iters {
            launch64()?;
        }
        gpu.synchronize(stream)?;
        let tf64 = flops / (t1.elapsed().as_secs_f64() / iters as f64) / 1e12;
        // K_STEP=64 (grid m/128)
        let launchk64 = || -> Result<()> {
            KernelLaunch::new(gpu, hk64)
                .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
                .block([128, 1, 1])
                .arg_ptr(a_p)
                .arg_ptr(b_p)
                .arg_ptr(as_p)
                .arg_ptr(bs_p)
                .arg_ptr(c_p)
                .arg_u32(m as u32)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .launch(stream)
        };
        for _ in 0..3 {
            launchk64()?;
        }
        gpu.synchronize(stream)?;
        let tk = Instant::now();
        for _ in 0..iters {
            launchk64()?;
        }
        gpu.synchronize(stream)?;
        let tfk64 = flops / (tk.elapsed().as_secs_f64() / iters as f64) / 1e12;
        // 8-warp (block 256, grid m/128 n/128)
        let launch8w = || -> Result<()> {
            KernelLaunch::new(gpu, h8w)
                .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(a_p)
                .arg_ptr(b_p)
                .arg_ptr(as_p)
                .arg_ptr(bs_p)
                .arg_ptr(c_p)
                .arg_u32(m as u32)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .launch(stream)
        };
        for _ in 0..3 {
            launch8w()?;
        }
        gpu.synchronize(stream)?;
        let t8 = Instant::now();
        for _ in 0..iters {
            launch8w()?;
        }
        gpu.synchronize(stream)?;
        let tf8w = flops / (t8.elapsed().as_secs_f64() / iters as f64) / 1e12;
        let launch8w3 = || -> Result<()> {
            KernelLaunch::new(gpu, h8w3)
                .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(a_p)
                .arg_ptr(b_p)
                .arg_ptr(as_p)
                .arg_ptr(bs_p)
                .arg_ptr(c_p)
                .arg_u32(m as u32)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .launch(stream)
        };
        for _ in 0..3 {
            launch8w3()?;
        }
        gpu.synchronize(stream)?;
        let t83 = Instant::now();
        for _ in 0..iters {
            launch8w3()?;
        }
        gpu.synchronize(stream)?;
        let tf8w3 = flops / (t83.elapsed().as_secs_f64() / iters as f64) / 1e12;
        let launch8wl = || -> Result<()> {
            KernelLaunch::new(gpu, h8wl)
                .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(a_p)
                .arg_ptr(b_p)
                .arg_ptr(as_p)
                .arg_ptr(bs_p)
                .arg_ptr(c_p)
                .arg_u32(m as u32)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .launch(stream)
        };
        for _ in 0..3 {
            launch8wl()?;
        }
        gpu.synchronize(stream)?;
        let tl = Instant::now();
        for _ in 0..iters {
            launch8wl()?;
        }
        gpu.synchronize(stream)?;
        let tf8wl = flops / (tl.elapsed().as_secs_f64() / iters as f64) / 1e12;
        let launch8wi = || -> Result<()> {
            KernelLaunch::new(gpu, h8wi)
                .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(a_p)
                .arg_ptr(b_p)
                .arg_ptr(as_p)
                .arg_ptr(bs_p)
                .arg_ptr(c_p)
                .arg_u32(m as u32)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .launch(stream)
        };
        for _ in 0..3 {
            launch8wi()?;
        }
        gpu.synchronize(stream)?;
        let ti = Instant::now();
        for _ in 0..iters {
            launch8wi()?;
        }
        gpu.synchronize(stream)?;
        let tf8wi = flops / (ti.elapsed().as_secs_f64() / iters as f64) / 1e12;
        let launchmmq = || -> Result<()> {
            KernelLaunch::new(gpu, hmmq)
                .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(a_p)
                .arg_ptr(b_p)
                .arg_ptr(as_p)
                .arg_ptr(bs_p)
                .arg_ptr(c_p)
                .arg_u32(m as u32)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .launch(stream)
        };
        for _ in 0..3 {
            launchmmq()?;
        }
        gpu.synchronize(stream)?;
        let tmq = Instant::now();
        for _ in 0..iters {
            launchmmq()?;
        }
        gpu.synchronize(stream)?;
        let tfmmq = flops / (tmq.elapsed().as_secs_f64() / iters as f64) / 1e12;
        let launch8wab = || -> Result<()> {
            KernelLaunch::new(gpu, h8wab)
                .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(a_p)
                .arg_ptr(b_p)
                .arg_ptr(as_p)
                .arg_ptr(bs_p)
                .arg_ptr(c_p)
                .arg_u32(m as u32)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .launch(stream)
        };
        for _ in 0..3 {
            launch8wab()?;
        }
        gpu.synchronize(stream)?;
        let tab = Instant::now();
        for _ in 0..iters {
            launch8wab()?;
        }
        gpu.synchronize(stream)?;
        let tf8wab = flops / (tab.elapsed().as_secs_f64() / iters as f64) / 1e12;
        let launchpipe = || -> Result<()> {
            KernelLaunch::new(gpu, hpipe)
                .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
                .block([512, 1, 1])
                .arg_ptr(a_p)
                .arg_ptr(b_p)
                .arg_ptr(as_p)
                .arg_ptr(bs_p)
                .arg_ptr(c_p)
                .arg_u32(m as u32)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .launch(stream)
        };
        for _ in 0..3 {
            launchpipe()?;
        }
        gpu.synchronize(stream)?;
        let tp = Instant::now();
        for _ in 0..iters {
            launchpipe()?;
        }
        gpu.synchronize(stream)?;
        let tfpipe = flops / (tp.elapsed().as_secs_f64() / iters as f64) / 1e12;
        let launchpada = || -> Result<()> {
            KernelLaunch::new(gpu, hpada)
                .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(a_p)
                .arg_ptr(b_p)
                .arg_ptr(as_p)
                .arg_ptr(bs_p)
                .arg_ptr(c_p)
                .arg_u32(m as u32)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .launch(stream)
        };
        for _ in 0..3 {
            launchpada()?;
        }
        gpu.synchronize(stream)?;
        let tpa = Instant::now();
        for _ in 0..iters {
            launchpada()?;
        }
        gpu.synchronize(stream)?;
        let tfpada = flops / (tpa.elapsed().as_secs_f64() / iters as f64) / 1e12;
        let launchfaith = || -> Result<()> {
            KernelLaunch::new(gpu, hfaith)
                .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(a_p)
                .arg_ptr(b_p)
                .arg_ptr(as_p)
                .arg_ptr(bs_p)
                .arg_ptr(c_p)
                .arg_u32(m as u32)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .launch(stream)
        };
        for _ in 0..3 {
            launchfaith()?;
        }
        gpu.synchronize(stream)?;
        let tfa = Instant::now();
        for _ in 0..iters {
            launchfaith()?;
        }
        gpu.synchronize(stream)?;
        let tffaith = flops / (tfa.elapsed().as_secs_f64() / iters as f64) / 1e12;
        let launchfaith2 = || -> Result<()> {
            KernelLaunch::new(gpu, hfaith2)
                .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(a_p)
                .arg_ptr(b_p)
                .arg_ptr(as_p)
                .arg_ptr(bs_p)
                .arg_ptr(c_p)
                .arg_u32(m as u32)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .launch(stream)
        };
        for _ in 0..3 {
            launchfaith2()?;
        }
        gpu.synchronize(stream)?;
        let tf2 = Instant::now();
        for _ in 0..iters {
            launchfaith2()?;
        }
        gpu.synchronize(stream)?;
        let tffaith2 = flops / (tf2.elapsed().as_secs_f64() / iters as f64) / 1e12;
        let launchfaith3 = || -> Result<()> {
            KernelLaunch::new(gpu, hfaith3)
                .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(a_p)
                .arg_ptr(b_p)
                .arg_ptr(as_p)
                .arg_ptr(bs_p)
                .arg_ptr(c_p)
                .arg_u32(m as u32)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .launch(stream)
        };
        for _ in 0..3 {
            launchfaith3()?;
        }
        gpu.synchronize(stream)?;
        let tf3 = Instant::now();
        for _ in 0..iters {
            launchfaith3()?;
        }
        gpu.synchronize(stream)?;
        let tffaith3 = flops / (tf3.elapsed().as_secs_f64() / iters as f64) / 1e12;
        let launchfaith4 = || -> Result<()> {
            KernelLaunch::new(gpu, hfaith4)
                .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
                .block([512, 1, 1])
                .arg_ptr(a_p)
                .arg_ptr(b_p)
                .arg_ptr(as_p)
                .arg_ptr(bs_p)
                .arg_ptr(c_p)
                .arg_u32(m as u32)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .launch(stream)
        };
        for _ in 0..3 {
            launchfaith4()?;
        }
        gpu.synchronize(stream)?;
        let tf4 = Instant::now();
        for _ in 0..iters {
            launchfaith4()?;
        }
        gpu.synchronize(stream)?;
        let tffaith4 = flops / (tf4.elapsed().as_secs_f64() / iters as f64) / 1e12;
        let launchfaith5 = || -> Result<()> {
            KernelLaunch::new(gpu, hfaith5)
                .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(a_p)
                .arg_ptr(b_p)
                .arg_ptr(as_p)
                .arg_ptr(bs_p)
                .arg_ptr(c_p)
                .arg_u32(m as u32)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .launch(stream)
        };
        for _ in 0..3 {
            launchfaith5()?;
        }
        gpu.synchronize(stream)?;
        let tf5 = Instant::now();
        for _ in 0..iters {
            launchfaith5()?;
        }
        gpu.synchronize(stream)?;
        let tffaith5 = flops / (tf5.elapsed().as_secs_f64() / iters as f64) / 1e12;
        let launchfaith6 = || -> Result<()> {
            KernelLaunch::new(gpu, hfaith6)
                .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(a_p)
                .arg_ptr(b_p)
                .arg_ptr(as_p)
                .arg_ptr(bs_p)
                .arg_ptr(c_p)
                .arg_u32(m as u32)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .launch(stream)
        };
        for _ in 0..3 {
            launchfaith6()?;
        }
        gpu.synchronize(stream)?;
        let tf6 = Instant::now();
        for _ in 0..iters {
            launchfaith6()?;
        }
        gpu.synchronize(stream)?;
        let tffaith6 = flops / (tf6.elapsed().as_secs_f64() / iters as f64) / 1e12;
        let launchfaith7 = || -> Result<()> {
            KernelLaunch::new(gpu, hfaith7)
                .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(a_p)
                .arg_ptr(b_p)
                .arg_ptr(as_p)
                .arg_ptr(bs_p)
                .arg_ptr(c_p)
                .arg_u32(m as u32)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .launch(stream)
        };
        for _ in 0..3 {
            launchfaith7()?;
        }
        gpu.synchronize(stream)?;
        let tf7 = Instant::now();
        for _ in 0..iters {
            launchfaith7()?;
        }
        gpu.synchronize(stream)?;
        let tffaith7 = flops / (tf7.elapsed().as_secs_f64() / iters as f64) / 1e12;
        let launchfaith8 = || -> Result<()> {
            KernelLaunch::new(gpu, hfaith8)
                .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(a_p)
                .arg_ptr(b_p)
                .arg_ptr(as_p)
                .arg_ptr(bs_p)
                .arg_ptr(c_p)
                .arg_u32(m as u32)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .launch(stream)
        };
        for _ in 0..3 {
            launchfaith8()?;
        }
        gpu.synchronize(stream)?;
        let tf8 = Instant::now();
        for _ in 0..iters {
            launchfaith8()?;
        }
        gpu.synchronize(stream)?;
        let tffaith8 = flops / (tf8.elapsed().as_secs_f64() / iters as f64) / 1e12;
        let launchfaith9 = || -> Result<()> {
            KernelLaunch::new(gpu, hfaith9)
                .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(a_p)
                .arg_ptr(b_p)
                .arg_ptr(as_p)
                .arg_ptr(bs_p)
                .arg_ptr(c_p)
                .arg_u32(m as u32)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .launch(stream)
        };
        for _ in 0..3 {
            launchfaith9()?;
        }
        gpu.synchronize(stream)?;
        let tf9 = Instant::now();
        for _ in 0..iters {
            launchfaith9()?;
        }
        gpu.synchronize(stream)?;
        let tffaith9 = flops / (tf9.elapsed().as_secs_f64() / iters as f64) / 1e12;
        let launchfaith10 = || -> Result<()> {
            KernelLaunch::new(gpu, hfaith10)
                .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(a_p)
                .arg_ptr(b_p)
                .arg_ptr(as_p)
                .arg_ptr(bs_p)
                .arg_ptr(c_p)
                .arg_u32(m as u32)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .launch(stream)
        };
        for _ in 0..3 {
            launchfaith10()?;
        }
        gpu.synchronize(stream)?;
        let tf10 = Instant::now();
        for _ in 0..iters {
            launchfaith10()?;
        }
        gpu.synchronize(stream)?;
        let tffaith10 = flops / (tf10.elapsed().as_secs_f64() / iters as f64) / 1e12;
        let launchmmqf = || -> Result<()> {
            KernelLaunch::new(gpu, hmmqf)
                .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(a_p)
                .arg_ptr(b_p)
                .arg_ptr(as_p)
                .arg_ptr(bs_p)
                .arg_ptr(c_p)
                .arg_u32(m as u32)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .launch(stream)
        };
        for _ in 0..3 {
            launchmmqf()?;
        }
        gpu.synchronize(stream)?;
        let tfmf = Instant::now();
        for _ in 0..iters {
            launchmmqf()?;
        }
        gpu.synchronize(stream)?;
        let tfmmqf = flops / (tfmf.elapsed().as_secs_f64() / iters as f64) / 1e12;
        let launchmmqf2 = || -> Result<()> {
            KernelLaunch::new(gpu, hmmqf2)
                .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(a_p)
                .arg_ptr(b_p)
                .arg_ptr(as_p)
                .arg_ptr(bs_p)
                .arg_ptr(c_p)
                .arg_u32(m as u32)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .launch(stream)
        };
        for _ in 0..3 {
            launchmmqf2()?;
        }
        gpu.synchronize(stream)?;
        let tfmf2 = Instant::now();
        for _ in 0..iters {
            launchmmqf2()?;
        }
        gpu.synchronize(stream)?;
        let tfmmqf2 = flops / (tfmf2.elapsed().as_secs_f64() / iters as f64) / 1e12;
        let launchmmqf3 = || -> Result<()> {
            KernelLaunch::new(gpu, hmmqf3)
                .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(a_p)
                .arg_ptr(b_p)
                .arg_ptr(as_p)
                .arg_ptr(bs_p)
                .arg_ptr(c_p)
                .arg_u32(m as u32)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .launch(stream)
        };
        for _ in 0..3 {
            launchmmqf3()?;
        }
        gpu.synchronize(stream)?;
        let tfmf3 = Instant::now();
        for _ in 0..iters {
            launchmmqf3()?;
        }
        gpu.synchronize(stream)?;
        let tfmmqf3 = flops / (tfmf3.elapsed().as_secs_f64() / iters as f64) / 1e12;
        let launchmmq2 = || -> Result<()> {
            KernelLaunch::new(gpu, hmmq2)
                .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(a_p)
                .arg_ptr(b_p)
                .arg_ptr(as_p)
                .arg_ptr(bs_p)
                .arg_ptr(c_p)
                .arg_u32(m as u32)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .launch(stream)
        };
        for _ in 0..3 {
            launchmmq2()?;
        }
        gpu.synchronize(stream)?;
        let tm2 = Instant::now();
        for _ in 0..iters {
            launchmmq2()?;
        }
        gpu.synchronize(stream)?;
        let tfmmq2 = flops / (tm2.elapsed().as_secs_f64() / iters as f64) / 1e12;
        let _ = (
            tf64, tfk64, tf8w3, tf8w, tf8wl, tf8wi, tfpipe, tfpada, tf8wab,
        );
        print!(
            "{label}: M128 {tf128:.2} | padA {tfpada:.2} | FAITH {tffaith:.2} | FAITH2 {tffaith2:.2} | FAITH3 {tffaith3:.2} | FAITH4 {tffaith4:.2} | FAITH5 {tffaith5:.2} | FAITH6 {tffaith6:.2} | FAITH7 {tffaith7:.2} | FAITH8 {tffaith8:.2} | FAITH9 {tffaith9:.2} | FAITH10 {tffaith10:.2} | MMQF {tfmmqf:.2} | MMQF2 {tfmmqf2:.2} | MMQF3 {tfmmqf3:.2} | MMQ {tfmmq:.2} | MMQ2 {tfmmq2:.2}  (bf16=30, llama Q4K=65/Q6K=41)"
        );
        // split-K sweep (partial + reduce)
        for &ks in &[2u32, 4, 8, 16] {
            let cp = gpu.alloc(ks as usize * m * n * 4)?;
            let mn = (m * n) as u32;
            let go = || -> Result<()> {
                KernelLaunch::new(gpu, hsk)
                    .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, ks])
                    .block([128, 1, 1])
                    .arg_ptr(a_p)
                    .arg_ptr(b_p)
                    .arg_ptr(as_p)
                    .arg_ptr(bs_p)
                    .arg_ptr(cp)
                    .arg_u32(m as u32)
                    .arg_u32(n as u32)
                    .arg_u32(k as u32)
                    .arg_u32(ks)
                    .launch(stream)?;
                KernelLaunch::new(gpu, hred)
                    .grid([mn.div_ceil(256), 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(cp)
                    .arg_ptr(c_p)
                    .arg_u32(m as u32)
                    .arg_u32(n as u32)
                    .arg_u32(ks)
                    .launch(stream)
            };
            for _ in 0..3 {
                go()?;
            }
            gpu.synchronize(stream)?;
            let t = Instant::now();
            for _ in 0..iters {
                go()?;
            }
            gpu.synchronize(stream)?;
            let tf = flops / (t.elapsed().as_secs_f64() / iters as f64) / 1e12;
            print!(" | sk{ks}={tf:.1}");
            let _ = gpu.free(cp);
        }
        println!("   (v2 bf16=30)");
        for p in [a_p, b_p, as_p, bs_p, c_p] {
            let _ = gpu.free(p);
        }
    }
    Ok(())
}

fn bytemuck_i8(v: &[i8]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) }
}
fn bytemuck_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}
