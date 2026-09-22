// SPDX-License-Identifier: AGPL-3.0-only
//! Qwen3.6-35B-A3B-FP8 MoE shapes (hidden 2048, moe_intermediate 512, top-8,
//! shared expert 512): the production per-token FP8 MoE loop (single-token
//! gate_up / silu_down / blend kernels, once per row) versus ONE cross-row
//! grouped dispatch (`moe_sort_by_expert` + the `_fp8_grouped` kernels), G9.
//!
//! Both legs are fed the SAME routing (indices + weights), so the bar is
//! exact: identical BF16 output bytes for every row, rows past M untouched,
//! guard bands intact. The expert count is reduced to 32 so rows collide on
//! experts (the grouped kernels' whole point), including experts with more
//! rows than one register pass (GROUP_ROWS=8) at M=32.
//!
//! Also prints per-leg wall time so the launch is a microbench of the lever.
//!
//! Run (GB10):
//!   cargo run --release -p spark-model --features cuda,gpu-examples \
//!     --example fp8_moe_grouped_decode_microtest
use anyhow::{Result, ensure};
use half::bf16;
use spark_model::layers::ops;
use spark_model::weight_map::{Fp8Weight, WeightQuantFormat};
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

const H: usize = 2048;
const INTER: usize = 512;
const E: usize = 32;
const TOP_K: usize = 8;
const MAX_M: usize = 32;
const GUARD: usize = 64;
const SENTINEL: u8 = 0x5a;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len().max(16))?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

fn fp8_bytes(rng: &mut Rng, n: usize) -> Vec<u8> {
    (0..n)
        .map(|_| {
            let x = rng.next();
            ((x % 120) as u8) | (((x >> 7) & 1) as u8 * 128)
        })
        .collect()
}

fn scale_bytes(rng: &mut Rng, n: usize, k: usize) -> Vec<u8> {
    (0..n.div_ceil(128) * k.div_ceil(128))
        .flat_map(|_| (((rng.next() % 16 + 1) as f32) / 512.0).to_le_bytes())
        .collect()
}

fn bf16_bytes(rng: &mut Rng, n: usize, scale: f32) -> Vec<u8> {
    (0..n)
        .flat_map(|_| {
            bf16::from_f32(((rng.next() % 2049) as f32 - 1024.0) / 1024.0 * scale)
                .to_bits()
                .to_le_bytes()
        })
        .collect()
}

fn fp8w(weight: DevicePtr, row_scale: DevicePtr, n: usize, k: usize) -> Fp8Weight {
    Fp8Weight {
        weight,
        row_scale,
        n: n as u32,
        k: k as u32,
        scale_format: WeightQuantFormat::Fp8BlockScaled,
    }
}

struct Handles {
    gate_up: KernelHandle,
    silu_down: KernelHandle,
    blend: KernelHandle,
    sort: KernelHandle,
    g_gate_up: KernelHandle,
    g_silu_down: KernelHandle,
    g_blend: KernelHandle,
}

struct Experts {
    gate_w: DevicePtr,
    gate_s: DevicePtr,
    up_w: DevicePtr,
    up_s: DevicePtr,
    down_w: DevicePtr,
    down_s: DevicePtr,
    sh_gate: Fp8Weight,
    sh_up: Fp8Weight,
    sh_down: Fp8Weight,
    sh_gate_vec: DevicePtr,
}

/// Per-leg scratch, sized for MAX_M rows; sentinel-filled before every run.
struct Scratch {
    gate_out: DevicePtr,
    up_out: DevicePtr,
    down_out: DevicePtr,
    sh_gate_out: DevicePtr,
    sh_up_out: DevicePtr,
    sh_down_out: DevicePtr,
    sort: DevicePtr,
}

#[allow(clippy::too_many_arguments)]
fn run_loop(
    gpu: &dyn GpuBackend,
    h: &Handles,
    x: &Experts,
    s: &Scratch,
    input: DevicePtr,
    indices: DevicePtr,
    weights: DevicePtr,
    out: DevicePtr,
    m: usize,
) -> Result<()> {
    for t in 0..m {
        let in_t = input.offset(t * H * 2);
        let idx_t = indices.offset(t * TOP_K * 4);
        ops::moe_expert_gate_up_shared_fp8(
            gpu,
            h.gate_up,
            in_t,
            x.gate_w,
            x.gate_s,
            s.gate_out,
            x.up_w,
            x.up_s,
            s.up_out,
            idx_t,
            &x.sh_gate,
            s.sh_gate_out,
            &x.sh_up,
            s.sh_up_out,
            INTER as u32,
            H as u32,
            TOP_K as u32,
            0,
        )?;
        ops::moe_expert_silu_down_shared_fp8(
            gpu,
            h.silu_down,
            s.gate_out,
            s.up_out,
            x.down_w,
            x.down_s,
            s.down_out,
            idx_t,
            s.sh_gate_out,
            s.sh_up_out,
            &x.sh_down,
            s.sh_down_out,
            H as u32,
            INTER as u32,
            TOP_K as u32,
            0,
        )?;
        ops::moe_weighted_sum_blend(
            gpu,
            h.blend,
            out.offset(t * H * 2),
            s.down_out,
            weights.offset(t * TOP_K * 4),
            s.sh_down_out,
            in_t,
            x.sh_gate_vec,
            H as u32,
            TOP_K as u32,
            H as u32,
            0,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_grouped(
    gpu: &dyn GpuBackend,
    h: &Handles,
    x: &Experts,
    s: &Scratch,
    input: DevicePtr,
    indices: DevicePtr,
    weights: DevicePtr,
    out: DevicePtr,
    m: usize,
) -> Result<()> {
    let te = m * TOP_K;
    let sorted_token_ids = s.sort;
    let sorted_expert_ids = s.sort.offset(te * 4);
    let expert_offsets = s.sort.offset(te * 8);
    let token_to_perm = s.sort.offset(te * 8 + (E + 1) * 4);
    ops::moe_sort_by_expert(
        gpu,
        h.sort,
        indices,
        sorted_token_ids,
        sorted_expert_ids,
        expert_offsets,
        token_to_perm,
        te as u32,
        E as u32,
        TOP_K as u32,
        0,
    )?;
    ops::moe_expert_gate_up_shared_fp8_grouped(
        gpu,
        h.g_gate_up,
        input,
        x.gate_w,
        x.gate_s,
        s.gate_out,
        x.up_w,
        x.up_s,
        s.up_out,
        expert_offsets,
        sorted_token_ids,
        &x.sh_gate,
        s.sh_gate_out,
        &x.sh_up,
        s.sh_up_out,
        INTER as u32,
        H as u32,
        E as u32,
        m as u32,
        0,
    )?;
    ops::moe_expert_silu_down_shared_fp8_grouped(
        gpu,
        h.g_silu_down,
        s.gate_out,
        s.up_out,
        x.down_w,
        x.down_s,
        s.down_out,
        expert_offsets,
        s.sh_gate_out,
        s.sh_up_out,
        &x.sh_down,
        s.sh_down_out,
        H as u32,
        INTER as u32,
        E as u32,
        m as u32,
        0,
    )?;
    ops::moe_weighted_sum_blend_fp8_grouped(
        gpu,
        h.g_blend,
        out,
        s.down_out,
        weights,
        token_to_perm,
        s.sh_down_out,
        input,
        x.sh_gate_vec,
        H as u32,
        TOP_K as u32,
        H as u32,
        m as u32,
        0,
    )
}

/// The oracle: live rows byte-equal and finite, everything else still sentinel.
fn check(observed: &[u8], baseline: &[u8], sentinel: &[u8], m: usize) -> Result<()> {
    ensure!(observed.len() == sentinel.len() && baseline.len() == sentinel.len());
    let live = GUARD..GUARD + m * H * 2;
    for i in 0..sentinel.len() {
        if !live.contains(&i) {
            ensure!(
                observed[i] == sentinel[i],
                "grouped wrote outside its rows at byte {i}"
            );
            ensure!(
                baseline[i] == sentinel[i],
                "loop wrote outside its rows at byte {i}"
            );
        }
    }
    let (a, b) = (&observed[live.clone()], &baseline[live]);
    for (i, (x, y)) in a.chunks_exact(2).zip(b.chunks_exact(2)).enumerate() {
        let (fx, fy) = (
            bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32(),
            bf16::from_bits(u16::from_le_bytes([y[0], y[1]])).to_f32(),
        );
        ensure!(
            fx.is_finite() && fy.is_finite(),
            "nonfinite output at element {i}"
        );
        ensure!(
            x == y,
            "row {} col {}: grouped {fx} != loop {fy} (bf16 bits {x:?} vs {y:?})",
            i / H,
            i % H
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let gpu = AvarokCudaBackend::new(0, &avarok_kernels::ptx_modules())?;
    let h = Handles {
        gate_up: gpu.kernel(
            "moe_shared_expert_fused_fp8",
            "moe_expert_gate_up_shared_fp8",
        )?,
        silu_down: gpu.kernel(
            "moe_shared_expert_fused_fp8",
            "moe_expert_silu_down_shared_fp8",
        )?,
        blend: gpu.kernel("moe_expert_gemv", "moe_weighted_sum_blend")?,
        sort: gpu.kernel("moe", "moe_sort_by_expert")?,
        g_gate_up: gpu.kernel(
            "moe_shared_expert_fused_fp8_grouped",
            "moe_expert_gate_up_shared_fp8_grouped",
        )?,
        g_silu_down: gpu.kernel(
            "moe_shared_expert_fused_fp8_grouped",
            "moe_expert_silu_down_shared_fp8_grouped",
        )?,
        g_blend: gpu.kernel(
            "moe_fp8_grouped_blend",
            "moe_weighted_sum_blend_fp8_grouped",
        )?,
    };
    let mut rng = Rng(0x6d6f_6520_6739_2026);

    // Routed experts: per-expert [INTER, H] gate/up and [H, INTER] down, FP8
    // block-scaled, exposed through device pointer tables like the layer does.
    let table = |n: usize, k: usize, rng: &mut Rng| -> Result<(DevicePtr, DevicePtr)> {
        let mut wp = Vec::with_capacity(E * 8);
        let mut sp = Vec::with_capacity(E * 8);
        for _ in 0..E {
            wp.extend_from_slice(&(upload(&gpu, &fp8_bytes(rng, n * k))?.0).to_le_bytes());
            sp.extend_from_slice(&(upload(&gpu, &scale_bytes(rng, n, k))?.0).to_le_bytes());
        }
        Ok((upload(&gpu, &wp)?, upload(&gpu, &sp)?))
    };
    let (gate_w, gate_s) = table(INTER, H, &mut rng)?;
    let (up_w, up_s) = table(INTER, H, &mut rng)?;
    let (down_w, down_s) = table(H, INTER, &mut rng)?;
    let shared = |n: usize, k: usize, rng: &mut Rng| -> Result<Fp8Weight> {
        Ok(fp8w(
            upload(&gpu, &fp8_bytes(rng, n * k))?,
            upload(&gpu, &scale_bytes(rng, n, k))?,
            n,
            k,
        ))
    };
    let x = Experts {
        gate_w,
        gate_s,
        up_w,
        up_s,
        down_w,
        down_s,
        sh_gate: shared(INTER, H, &mut rng)?,
        sh_up: shared(INTER, H, &mut rng)?,
        sh_down: shared(H, INTER, &mut rng)?,
        sh_gate_vec: upload(&gpu, &bf16_bytes(&mut rng, H, 0.05))?,
    };
    let input = upload(&gpu, &bf16_bytes(&mut rng, MAX_M * H, 1.0))?;

    // Routing: distinct experts within a row, heavy collisions across rows.
    let mut idx = Vec::with_capacity(MAX_M * TOP_K);
    for _ in 0..MAX_M {
        let mut row: Vec<u32> = Vec::new();
        while row.len() < TOP_K {
            let e = rng.next() % E as u32;
            if !row.contains(&e) {
                row.push(e);
            }
        }
        idx.extend(row);
    }
    let idx_bytes: Vec<u8> = idx.iter().flat_map(|e| e.to_le_bytes()).collect();
    let w_bytes: Vec<u8> = (0..MAX_M * TOP_K)
        .flat_map(|_| ((rng.next() % 1000) as f32 / 1000.0).to_le_bytes())
        .collect();
    let indices = upload(&gpu, &idx_bytes)?;
    let weights = upload(&gpu, &w_bytes)?;

    let te = MAX_M * TOP_K;
    let scratch = Scratch {
        gate_out: upload(&gpu, &vec![SENTINEL; te * INTER * 2])?,
        up_out: upload(&gpu, &vec![SENTINEL; te * INTER * 2])?,
        down_out: upload(&gpu, &vec![SENTINEL; te * H * 2])?,
        sh_gate_out: upload(&gpu, &vec![SENTINEL; MAX_M * INTER * 2])?,
        sh_up_out: upload(&gpu, &vec![SENTINEL; MAX_M * INTER * 2])?,
        sh_down_out: upload(&gpu, &vec![SENTINEL; MAX_M * H * 2])?,
        sort: upload(&gpu, &vec![0u8; te * 12 + (E + 1) * 4])?,
    };
    let sentinel = vec![SENTINEL; MAX_M * H * 2 + 2 * GUARD];
    let loop_base = upload(&gpu, &sentinel)?;
    let grouped_base = upload(&gpu, &sentinel)?;
    let (loop_out, grouped_out) = (loop_base.offset(GUARD), grouped_base.offset(GUARD));

    let mut failures = 0usize;
    let mut first = true;
    for m in [2usize, 3, 4, 8, 16, 32] {
        gpu.copy_h2d(&sentinel, loop_base)?;
        gpu.copy_h2d(&sentinel, grouped_base)?;
        run_loop(&gpu, &h, &x, &scratch, input, indices, weights, loop_out, m)?;
        run_grouped(
            &gpu,
            &h,
            &x,
            &scratch,
            input,
            indices,
            weights,
            grouped_out,
            m,
        )?;
        gpu.synchronize(0)?;
        let mut baseline = vec![0u8; sentinel.len()];
        let mut observed = vec![0u8; sentinel.len()];
        gpu.copy_d2h(loop_base, &mut baseline)?;
        gpu.copy_d2h(grouped_base, &mut observed)?;

        if first {
            for mutation in ["output-bit", "past-m", "guard", "nonfinite"] {
                let mut bad = baseline.clone();
                match mutation {
                    "output-bit" => bad[GUARD + 7] ^= 1,
                    "past-m" => bad[GUARD + m * H * 2] ^= 1,
                    "guard" => bad[0] ^= 1,
                    _ => bad[GUARD..GUARD + 2].copy_from_slice(&0x7fc0_u16.to_le_bytes()),
                }
                let err = check(&bad, &baseline, &sentinel, m)
                    .expect_err("known-bad output was admitted by the oracle");
                println!("KNOWN_BAD {mutation}: refused: {err}");
            }
            first = false;
        }

        // Timing: 20 iterations per leg after the correctness run warmed both.
        let time = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
            gpu.synchronize(0)?;
            let t = std::time::Instant::now();
            for _ in 0..20 {
                f()?;
            }
            gpu.synchronize(0)?;
            Ok(t.elapsed().as_secs_f64() * 1e6 / 20.0)
        };
        let us_loop =
            time(&|| run_loop(&gpu, &h, &x, &scratch, input, indices, weights, loop_out, m))?;
        let us_grouped = time(&|| {
            run_grouped(
                &gpu,
                &h,
                &x,
                &scratch,
                input,
                indices,
                weights,
                grouped_out,
                m,
            )
        })?;
        let distinct = {
            let mut seen = [false; E];
            idx[..m * TOP_K]
                .iter()
                .for_each(|&e| seen[e as usize] = true);
            seen.iter().filter(|&&s| s).count()
        };
        match check(&observed, &baseline, &sentinel, m) {
            Ok(()) => println!(
                "M={m:2} distinct_experts={distinct:2}/{} loop={us_loop:8.1}us grouped={us_grouped:8.1}us \
                 speedup={:.2}x  BIT-IDENTICAL",
                m * TOP_K,
                us_loop / us_grouped
            ),
            Err(e) => {
                println!("FAIL M={m}: {e}");
                failures += 1;
            }
        }
    }
    ensure!(failures == 0, "{failures} case(s) failed");
    println!("ALL PASS: grouped FP8 MoE decode == per-token loop, bit for bit, M=2..32");
    Ok(())
}
