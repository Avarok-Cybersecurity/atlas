// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: GPU qualification of the Ada tensor-core chunked GDN
//! prefill (`kernels/ada/common/gdn_chunk_tc_ada.cu`) at the Qwen3.5-4B head
//! geometry (16 key heads, 32 value heads, 128 x 128), in Atlas's packed
//! prefill layout (q/k/v inside one `conv_dim` row, gate and beta interleaved).
//!
//! Per prompt length T (default 64 150 1024 3921 4096):
//!   * the TC route's output and final state against the FP64 recurrence,
//!     under the suite criteria `oracle::{OUTPUT, STATE}` (normwise bound with
//!     a gross pointwise cap; any non-finite value fails);
//!   * the scalar token-sequential spine (`gated_delta_rule_prefill`) against
//!     the same oracle, and the TC output's agreement with it within OUTPUT;
//!   * the production FLA route (recompute_wu -> chunk_delta_h_vfused ->
//!     chunk_fwd_o) against the oracle, for the record;
//!   * a sentinel tail after the output, which a partial last chunk must not
//!     write.
//! Then T=150 at batch 2 through a per-request state POINTER TABLE (the
//! co-dispatch form), each request against its own oracle.
//! `AVAROK_BENCH_ITERS=N` adds per-route timings (TC vs FLA vs scalar).
//!
//! The launch sequence mirrors `layers/qwen3_ssm/gdn_chunk_tc_ada.rs::launch`
//! (crate-private, so it is restated here); the host unit tests pin that
//! function's ABI and scratch carving.
//!
//! Usage: cargo run --release -p spark-model --example gdn_chunk_tc_ada_microtest \
//!          --features cuda,gpu-examples -- [T ...]

#[allow(dead_code)]
#[path = "../src/layers/qwen3_ssm/gdn_chunk_tc_ada/oracle.rs"]
mod oracle;

#[path = "common/gdn_tc_ada_routes.rs"]
mod routes;

use anyhow::Result;
use routes::{
    C, CONV_DIM, D, GB, KEY_DIM, Kernels, NK, NV, regions, run_fla, run_scalar, scratch_bytes,
};
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

const SENTINEL: u16 = 0x7F7F;

fn up<T: Copy>(gpu: &dyn GpuBackend, v: &[T]) -> Result<DevicePtr> {
    let bytes = std::mem::size_of_val(v);
    let p = gpu.alloc(bytes)?;
    // SAFETY: a plain-old-data slice viewed as bytes for the upload.
    let raw = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, bytes) };
    gpu.copy_h2d(raw, p)?;
    Ok(p)
}

fn down_u16(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
    let mut b = vec![0u8; n * 2];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

/// One request's FP32 state `[NV][D][D]`, widened for the comparison.
fn down_state(gpu: &dyn GpuBackend, p: DevicePtr) -> Result<Vec<f64>> {
    let mut b = vec![0u8; NV * D * D * 4];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
        .collect())
}

fn bf16_to_f64(bits: &[u16]) -> Vec<f64> {
    bits.iter()
        .map(|b| f32::from_bits((*b as u32) << 16) as f64)
        .collect()
}

/// Requests packed back to back: qkv `[b * t][CONV_DIM]` BF16 bits, gate/beta
/// `[b * t][2 * NV]`, states `[b][NV][D][D]`.
fn pack(ps: &[oracle::Problem]) -> (Vec<u16>, Vec<f32>, Vec<f32>) {
    let mut qkv = Vec::new();
    let mut gb = Vec::new();
    let mut h = Vec::new();
    for p in ps {
        for t in 0..p.t {
            let mut row = vec![0u16; CONV_DIM];
            for i in 0..KEY_DIM {
                row[i] = oracle::bf16_bits(p.q[t * KEY_DIM + i]);
                row[KEY_DIM + i] = oracle::bf16_bits(p.k[t * KEY_DIM + i]);
            }
            for i in 0..NV * D {
                row[2 * KEY_DIM + i] = oracle::bf16_bits(p.v[t * NV * D + i]);
            }
            qkv.extend(row);
            gb.extend_from_slice(&p.gate[t * NV..(t + 1) * NV]);
            gb.extend_from_slice(&p.beta[t * NV..(t + 1) * NV]);
        }
        h.extend_from_slice(&p.h0);
    }
    (qkv, gb, h)
}

/// The system under test: the three launches of `gdn_chunk_tc_ada.rs::launch`.
#[allow(clippy::too_many_arguments)]
fn run_tc(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    h: DevicePtr,
    table: bool,
    qkv: DevicePtr,
    gb: DevicePtr,
    out: DevicePtr,
    scratch: DevicePtr,
    batch: u32,
    t: u32,
    stream: u64,
) -> Result<()> {
    let nc = t.div_ceil(C);
    let [w, u, s, uc, gc] = regions(scratch, batch, nc);
    let (q, kk, v) = (qkv, qkv.offset(KEY_DIM * 2), qkv.offset(2 * KEY_DIM * 2));
    let (gate, beta) = (gb, gb.offset(NV * 4));
    let (cd, nk, nv) = (CONV_DIM as u32, NK as u32, NV as u32);
    KernelLaunch::new(gpu, k.tc[0])
        .grid([nc, nv, batch])
        .block([128, 1, 1])
        .shared_mem(24_320)
        .arg_ptr(kk)
        .arg_ptr(v)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(w)
        .arg_ptr(u)
        .arg_ptr(gc)
        .arg_u32(t)
        .arg_u32(nc)
        .arg_u32(nk)
        .arg_u32(nv)
        .arg_u32(cd)
        .arg_u32(cd)
        .arg_u32(GB as u32)
        .launch(stream)?;
    KernelLaunch::new(gpu, k.tc[1])
        .grid([nv * 8, batch, 1])
        .block([256, 1, 1])
        .shared_mem(45_312)
        .arg_ptr(h)
        .arg_ptr(w)
        .arg_ptr(u)
        .arg_ptr(kk)
        .arg_ptr(gc)
        .arg_ptr(uc)
        .arg_ptr(s)
        .arg_u32(t)
        .arg_u32(nc)
        .arg_u32(nk)
        .arg_u32(nv)
        .arg_u32(cd)
        .arg_u32(table as u32)
        .launch(stream)?;
    let jobs = nc as u64 * nv as u64 * batch as u64;
    let per = jobs.div_ceil(k.sm as u64 * 4).max(1);
    let gx = (nc as u64).div_ceil(per).max(1) as u32;
    KernelLaunch::new(gpu, k.tc[2])
        .grid([gx, nv, batch])
        .block([128, 1, 1])
        .shared_mem(24_576)
        .arg_ptr(q)
        .arg_ptr(kk)
        .arg_ptr(uc)
        .arg_ptr(gc)
        .arg_ptr(s)
        .arg_ptr(out)
        .arg_u32(t)
        .arg_u32(nc)
        .arg_u32(nk)
        .arg_u32(nv)
        .arg_u32(cd)
        .launch(stream)
}

fn line(name: &str, o: &oracle::Errors, s: Option<&oracle::Errors>) -> String {
    let st = s.map_or(String::new(), |s| {
        format!("  state normwise {:.3e} max {:.3e}", s.normwise, s.max_abs)
    });
    format!(
        "{name:<8} out normwise {:.3e} max {:.3e} (ref max {:.3e}) finite={}{st}",
        o.normwise, o.max_abs, o.ref_max, o.finite
    )
}

/// One length at batch 1; returns whether every gate passed.
fn check_len(gpu: &dyn GpuBackend, k: &Kernels, t: usize, stream: u64) -> Result<bool> {
    let p = oracle::generate(0x6DC0_0000 + t as u64, t, NK, NV, D, D);
    let r = oracle::recurrent(&p);
    let (qkv, gb, h0) = pack(std::slice::from_ref(&p));
    let (dq, dg) = (up(gpu, &qkv)?, up(gpu, &gb)?);
    let n_out = t * NV * D;
    let mut out_init = vec![SENTINEL; n_out + 64 * NV * D];
    let out = up(gpu, &out_init)?;
    let tu = t as u32;
    let scratch = gpu.alloc(scratch_bytes(1, tu.div_ceil(C)))?;
    // TC route.
    let h = up(gpu, &h0)?;
    run_tc(gpu, k, h, false, dq, dg, out, scratch, 1, tu, stream)?;
    gpu.synchronize(stream)?;
    let tc_all = down_u16(gpu, out, out_init.len())?;
    let tail_ok = tc_all[n_out..].iter().all(|x| *x == SENTINEL);
    let tc_out = bf16_to_f64(&tc_all[..n_out]);
    let tc_h = down_state(gpu, h)?;
    // Scalar spine.
    out_init.truncate(n_out);
    let out_s = up(gpu, &out_init)?;
    let h_s = up(gpu, &h0)?;
    run_scalar(gpu, k, h_s, dq, dg, out_s, tu, stream)?;
    gpu.synchronize(stream)?;
    let sc_out = bf16_to_f64(&down_u16(gpu, out_s, n_out)?);
    let sc_h = down_state(gpu, h_s)?;
    // Production FLA route.
    let out_f = up(gpu, &out_init)?;
    let h_f = up(gpu, &h0)?;
    run_fla(gpu, k, h_f, dq, dg, out_f, scratch, tu, stream)?;
    gpu.synchronize(stream)?;
    let fl_out = bf16_to_f64(&down_u16(gpu, out_f, n_out)?);
    let fl_h = down_state(gpu, h_f)?;

    let (to, ts) = (
        oracle::compare(&tc_out, &r.out),
        oracle::compare(&tc_h, &r.state),
    );
    let (so, ss) = (
        oracle::compare(&sc_out, &r.out),
        oracle::compare(&sc_h, &r.state),
    );
    let (fo, fs) = (
        oracle::compare(&fl_out, &r.out),
        oracle::compare(&fl_h, &r.state),
    );
    let agree = oracle::compare(&tc_out, &sc_out);
    println!(
        "--- T={t} (chunks {}, last {} rows)",
        tu.div_ceil(C),
        t - (t - 1) / 64 * 64
    );
    println!("  {}", line("tc", &to, Some(&ts)));
    println!("  {}", line("scalar", &so, Some(&ss)));
    println!("  {}", line("fla", &fo, Some(&fs)));
    println!(
        "  {}  sentinel tail intact={tail_ok}",
        line("tc~scal", &agree, None)
    );
    let pass = oracle::OUTPUT.accepts(&to)
        && oracle::STATE.accepts(&ts)
        && oracle::OUTPUT.accepts(&agree)
        && tail_ok;

    let iters = std::env::var("AVAROK_BENCH_ITERS").ok();
    if let Some(iters) = iters.and_then(|s| s.parse::<usize>().ok()) {
        let time = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
            for _ in 0..3 {
                f()?;
            }
            gpu.synchronize(stream)?;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                f()?;
            }
            gpu.synchronize(stream)?;
            Ok(t0.elapsed().as_secs_f64() * 1e6 / iters.max(1) as f64)
        };
        let tc_us = time(&|| run_tc(gpu, k, h, false, dq, dg, out, scratch, 1, tu, stream))?;
        let fl_us = time(&|| run_fla(gpu, k, h_f, dq, dg, out_f, scratch, tu, stream))?;
        let sc_us = time(&|| run_scalar(gpu, k, h_s, dq, dg, out_s, tu, stream))?;
        let ratio = fl_us / tc_us;
        println!(
            "  BENCH T={t}: tc {tc_us:.1} us  fla {fl_us:.1} us  scalar {sc_us:.1} us  \
             (fla/tc {ratio:.2}x)"
        );
    }
    for x in [dq, dg, out, scratch, h, out_s, h_s, out_f, h_f] {
        gpu.free(x).ok();
    }
    Ok(pass)
}

/// T=150 at batch 2 through a per-request state pointer table.
fn check_table(gpu: &dyn GpuBackend, k: &Kernels, stream: u64) -> Result<bool> {
    let t = 150usize;
    let ps = [
        oracle::generate(0x7AB1_0001, t, NK, NV, D, D),
        oracle::generate(0x7AB1_0002, t, NK, NV, D, D),
    ];
    let (qkv, gb, _) = pack(&ps);
    let (dq, dg) = (up(gpu, &qkv)?, up(gpu, &gb)?);
    let out = up(gpu, &vec![SENTINEL; 2 * t * NV * D])?;
    let hs = [up(gpu, &ps[0].h0)?, up(gpu, &ps[1].h0)?];
    let table = up(gpu, &[hs[0].0, hs[1].0])?;
    let scratch = gpu.alloc(scratch_bytes(2, (t as u32).div_ceil(C)))?;
    run_tc(
        gpu, k, table, true, dq, dg, out, scratch, 2, t as u32, stream,
    )?;
    gpu.synchronize(stream)?;
    let all = bf16_to_f64(&down_u16(gpu, out, 2 * t * NV * D)?);
    let mut pass = true;
    for (b, p) in ps.iter().enumerate() {
        let r = oracle::recurrent(p);
        let o = oracle::compare(&all[b * t * NV * D..(b + 1) * t * NV * D], &r.out);
        let s = oracle::compare(&down_state(gpu, hs[b])?, &r.state);
        println!(
            "--- batch 2 table, request {b}: {}",
            line("tc", &o, Some(&s))
        );
        pass &= oracle::OUTPUT.accepts(&o) && oracle::STATE.accepts(&s);
    }
    for x in [dq, dg, out, hs[0], hs[1], table, scratch] {
        gpu.free(x).ok();
    }
    Ok(pass)
}

fn main() -> Result<()> {
    let mut lens: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    if lens.is_empty() {
        lens = vec![64, 150, 1024, 3921, 4096];
    }
    println!(
        "=== gdn_chunk_tc_ada microtest: nk={NK} nv={NV} d={D}; criteria out {:?} state {:?} ===",
        oracle::OUTPUT,
        oracle::STATE
    );
    let backend = AvarokCudaBackend::new(0, &avarok_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    if !gpu.has_module("gdn_chunk_tc_ada") {
        println!(
            "RESULT: FAIL (module gdn_chunk_tc_ada is not in this image; \
             build with AVAROK_TARGET_HW=ada)"
        );
        std::process::exit(2);
    }
    let k = Kernels {
        tc: [
            gpu.kernel("gdn_chunk_tc_ada", "gdn_chunk_tc_ada_prepare")?,
            gpu.kernel("gdn_chunk_tc_ada", "gdn_chunk_tc_ada_state")?,
            gpu.kernel("gdn_chunk_tc_ada", "gdn_chunk_tc_ada_output")?,
        ],
        fla: [
            gpu.kernel("gated_delta_rule_fla", "gated_delta_rule_recompute_wu")?,
            gpu.kernel(
                "gated_delta_rule_fla",
                "gated_delta_rule_chunk_delta_h_vfused",
            )?,
            gpu.kernel("gated_delta_rule_fla", "gated_delta_rule_chunk_fwd_o")?,
        ],
        scalar: gpu.kernel("gated_delta_rule", "gated_delta_rule_prefill")?,
        sm: gpu.sm_count().unwrap_or(34).max(1),
    };
    let stream = gpu.create_stream()?;
    let mut pass = true;
    for &t in &lens {
        pass &= check_len(gpu, &k, t.max(1), stream)?;
    }
    pass &= check_table(gpu, &k, stream)?;
    if pass {
        println!("RESULT: PASS");
        Ok(())
    } else {
        println!("RESULT: FAIL");
        std::process::exit(1);
    }
}
