// SPDX-License-Identifier: AGPL-3.0-only
//! TEMPORARY (delete after use): bandwidth/B-scaling bench for
//! `gated_delta_rule_decode_f32_strided_norm` at production decode shapes
//! (NV=48, KD=VD=128, per prod nsys: 616us/launch at B=16).
//!
//! Sweeps B in {1,2,4,8,16}. Reports us/launch, GB/s counting state 1R+1W
//! (the DRAM floor), and the ratio over that floor. If the ratio jumps as
//! the working set (B*48*64KB) exceeds L2, the second H pass is missing L2
//! and a single-pass register-resident kernel recovers it.

use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kernel_args::KernelLaunch;
use std::time::Instant;

const NK: usize = 16;
const NV: usize = 48;
const KD: usize = 128;
const VD: usize = 128;
const WARMUP: usize = 20;
const ITERS: usize = 300;

fn main() -> Result<()> {
    let g0 = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &g0;
    let kh = g.kernel("gated_delta_rule", "gated_delta_rule_decode_f32_strided_norm")?;

    let bw: f64 = std::env::var("ATLAS_PEAK_GBPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(215.0);

    let qk_stride = (NK * KD) as u32; // q/k laid out [B, NK*KD] f32
    let v_stride = (NV * VD) as u32;
    let gb_stride = NV as u32;
    let z_stride = (NV * VD) as u32;
    let out_stride = (NV * VD) as u32;

    // Rotate over NLAYERS distinct state buffers, like the 48 per-layer
    // launches of a production step: the state read by launch i was written
    // one full step ago, not by the immediately preceding launch.
    const NLAYERS: usize = 8;
    for &b in &[16usize] {
        let state_bytes = b * NV * KD * VD * 4;
        let hs: Vec<_> = (0..NLAYERS)
            .map(|_| g.alloc(state_bytes))
            .collect::<Result<_>>()?;
        let h = hs[0];
        let q = g.alloc(b * NK * KD * 4)?;
        let k = g.alloc(b * NK * KD * 4)?;
        let v = g.alloc(b * NV * VD * 4)?;
        let gate = g.alloc(b * NV * 4)?;
        let beta = g.alloc(b * NV * 4)?;
        let z = g.alloc(b * NV * VD * 2)?;
        let w = g.alloc(NV * VD * 2)?;
        let out = g.alloc(b * NV * VD * 2)?;
        for (p, n) in [
            (h, state_bytes),
            (hs[1], state_bytes),
            (hs[2], state_bytes),
            (hs[3], state_bytes),
            (hs[4], state_bytes),
            (hs[5], state_bytes),
            (hs[6], state_bytes),
            (hs[7], state_bytes),
            (q, b * NK * KD * 4),
            (k, b * NK * KD * 4),
            (v, b * NV * VD * 4),
            (gate, b * NV * 4),
            (beta, b * NV * 4),
            (z, b * NV * VD * 2),
            (w, NV * VD * 2),
        ] {
            g.memset(p, 0, n)?; // zeros: g clamps to 1e-6, state stays finite
        }

        let launch = |g: &dyn GpuBackend, h: spark_runtime::gpu::DevicePtr| -> Result<()> {
            KernelLaunch::new(g, kh)
                .grid([NV as u32, b as u32, 1])
                .block([128, 1, 1])
                .arg_ptr(h)
                .arg_ptr(q)
                .arg_ptr(k)
                .arg_ptr(v)
                .arg_ptr(gate)
                .arg_ptr(beta)
                .arg_ptr(z)
                .arg_ptr(w)
                .arg_ptr(out)
                .arg_u32(b as u32)
                .arg_u32(NK as u32)
                .arg_u32(NV as u32)
                .arg_u32(KD as u32)
                .arg_u32(VD as u32)
                .arg_u32(qk_stride)
                .arg_u32(v_stride)
                .arg_u32(gb_stride)
                .arg_u32(z_stride)
                .arg_u32(out_stride)
                .arg_f32(1e-6)
                .launch(0)
        };

        for i in 0..WARMUP {
            launch(g, hs[i % NLAYERS])?;
        }
        g.synchronize(0)?;
        let t0 = Instant::now();
        for i in 0..ITERS {
            launch(g, hs[i % NLAYERS])?;
        }
        g.synchronize(0)?;
        let us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

        let rw_bytes = (state_bytes * 2) as f64; // 1R + 1W floor
        let floor_us = rw_bytes / (bw * 1e9) * 1e6;
        let gbps = rw_bytes / (us * 1e-6) / 1e9;
        eprintln!(
            "B={b:>2}  state {:>6.1} MB (ws incl 2nd read {:>6.1} MB)  {us:>8.1} us  \
             eff {gbps:>6.1} GB/s (1R+1W)  {:>4.2}x floor({floor_us:.0} us)",
            state_bytes as f64 / 1e6,
            state_bytes as f64 / 1e6,
            us / floor_us,
        );

        let _ = h;
        for p in hs {
            let _ = g.free(p);
        }
        for p in [q, k, v, gate, beta, z, w, out] {
            let _ = g.free(p);
        }
    }
    Ok(())
}
