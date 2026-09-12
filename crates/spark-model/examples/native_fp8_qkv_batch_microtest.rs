// SPDX-License-Identifier: AGPL-3.0-only
//! Actual Qwen3.8-27B Q/K/V projection shapes: the production per-sequence
//! scalar `w8a16_gemv` loop versus ONE strided batched launch per projection
//! (`w8a16_gemv_batch{4,16}_strided`, issue #927 / O13).
//!
//! No new kernel math is claimed, so the bar is exact: identical BF16 output
//! bytes, and every byte the batched path must NOT touch still holding its
//! sentinel (the rows past M, the unused K/V region of a Q-only run, and the
//! guard bands either side of every buffer).
//!
//! Shapes come from `kernels/gb10/qwen3.8-27b/MODEL.toml`: hidden_dim 5120,
//! head_dim 256, q_heads 24, kv_heads 4, attn_output_gate = true. So
//! K = 5120, q_dim = 6144, q_proj_dim = 2*q_dim = 12288 (interleaved [Q|gate]),
//! kv_dim = 1024, and one sequence's [Q|K|V] block is 14336 BF16 elements —
//! the `per_seq_qkv` row stride the batched kernels have to honour.
//!
//! Run (H100):
//!   cargo run --release -p spark-model --features cuda,gpu-examples \
//!     --example native_fp8_qkv_batch_microtest
use anyhow::{Result, ensure};
use half::bf16;
use spark_model::layers::ops;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

const K: usize = 5120; // hidden_dim
const Q_PROJ_DIM: usize = 12288; // 2 * q_heads * head_dim (gated)
const KV_DIM: usize = 1024; // kv_heads * head_dim
const PER_SEQ_QKV: usize = Q_PROJ_DIM + 2 * KV_DIM; // 14336 BF16 elements
const MAX_M: usize = 8;
const GUARD: usize = 64; // bytes of sentinel either side of every buffer
const SENTINEL: u8 = 0x5a;

/// One projection's slot inside a sequence's [Q|K|V] block.
struct Proj {
    name: &'static str,
    weight: DevicePtr,
    scale: DevicePtr,
    /// Element offset of this projection inside the row.
    offset: usize,
    /// Output width (N).
    n: usize,
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

fn values(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(2)
        .map(|x| bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32() as f64)
        .collect()
}

/// The real comparison oracle, exercised against deliberately-corrupted input
/// below so a green run cannot be a vacuous one.
fn check(observed: &[u8], baseline: &[u8], sentinel: &[u8], live: &[(usize, usize)]) -> Result<()> {
    ensure!(
        observed.len() == sentinel.len() && baseline.len() == sentinel.len(),
        "output extent mismatch"
    );
    // Every byte outside the live extents must still be sentinel, on BOTH runs.
    let mut mask = vec![false; sentinel.len()];
    for &(start, len) in live {
        mask[GUARD + start..GUARD + start + len].fill(true);
    }
    for i in 0..sentinel.len() {
        if !mask[i] {
            ensure!(
                observed[i] == sentinel[i],
                "batched projection wrote outside its extent at byte {i}"
            );
            ensure!(
                baseline[i] == sentinel[i],
                "scalar projection wrote outside its extent at byte {i}"
            );
        }
    }
    for &(start, len) in live {
        let a = &observed[GUARD + start..GUARD + start + len];
        let b = &baseline[GUARD + start..GUARD + start + len];
        ensure!(
            values(a)
                .iter()
                .chain(values(b).iter())
                .all(|x| x.is_finite()),
            "nonfinite projection output"
        );
        ensure!(
            a == b,
            "batch output differs from production scalar BF16 bits"
        );
    }
    Ok(())
}

/// The shared shape of the two strided wrappers, so picking MAX_M is one
/// branch rather than two duplicated 12-argument calls.
type StridedBatchGemv = fn(
    &dyn GpuBackend,
    KernelHandle,
    DevicePtr,
    DevicePtr,
    DevicePtr,
    DevicePtr,
    u32,
    u32,
    u32,
    u32,
    u32,
    u64,
) -> Result<()>;

fn run_batched(
    gpu: &dyn GpuBackend,
    batch4: KernelHandle,
    batch16: KernelHandle,
    input: DevicePtr,
    out: DevicePtr,
    p: &Proj,
    m: usize,
) -> Result<()> {
    let (launch, kernel): (StridedBatchGemv, KernelHandle) = if m <= 4 {
        (ops::w8a16_gemv_batch4_strided, batch4)
    } else {
        (ops::w8a16_gemv_batch16_strided, batch16)
    };
    launch(
        gpu,
        kernel,
        input,
        p.weight,
        p.scale,
        out.offset(p.offset * 2),
        m as u32,
        p.n as u32,
        K as u32,
        K as u32,
        PER_SEQ_QKV as u32,
        0,
    )
}

fn run_scalar(
    gpu: &dyn GpuBackend,
    scalar: KernelHandle,
    input: DevicePtr,
    out: DevicePtr,
    p: &Proj,
    m: usize,
) -> Result<()> {
    for row in 0..m {
        ops::w8a16_gemv(
            gpu,
            scalar,
            input.offset(row * K * 2),
            p.weight,
            p.scale,
            out.offset((row * PER_SEQ_QKV + p.offset) * 2),
            p.n as u32,
            K as u32,
            0,
        )?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let scalar = gpu.kernel("w8a16_gemv", "w8a16_gemv")?;
    let batch4 = gpu.kernel("w8a16_gemv_batch4", "w8a16_gemv_batch4_strided")?;
    let batch16 = gpu.kernel("w8a16_gemv_batch4", "w8a16_gemv_batch16_strided")?;

    let mut state = 0x0132_8a16_2026_u64;
    let mut random = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (state >> 32) as u32
    };
    let mut fp8_weight = |n: usize| -> Vec<u8> {
        (0..n * K)
            .map(|_| {
                let x = random();
                ((x % 127) as u8) | (((x >> 7) & 1) as u8 * 128)
            })
            .collect()
    };
    let q_bytes = fp8_weight(Q_PROJ_DIM);
    let k_bytes = fp8_weight(KV_DIM);
    let v_bytes = fp8_weight(KV_DIM);
    let mut scales = |n: usize| -> Vec<u8> {
        (0..(n / 128) * (K / 128))
            .flat_map(|_| (((random() % 16 + 1) as f32) / 1024.0).to_le_bytes())
            .collect()
    };
    let q_scale = scales(Q_PROJ_DIM);
    let k_scale = scales(KV_DIM);
    let v_scale = scales(KV_DIM);
    // Activations: contiguous [MAX_M, K], the `normed` layout decode hands in.
    let acts: Vec<u8> = (0..MAX_M * K)
        .flat_map(|_| {
            bf16::from_f32(((random() % 2049) as f32 - 1024.0) / 1024.0)
                .to_bits()
                .to_le_bytes()
        })
        .collect();

    let input_base = upload(
        &gpu,
        &[vec![SENTINEL; GUARD], acts, vec![SENTINEL; GUARD]].concat(),
    )?;
    let input = input_base.offset(GUARD);
    let projections = [
        Proj {
            name: "q_proj",
            weight: upload(&gpu, &q_bytes)?,
            scale: upload(&gpu, &q_scale)?,
            offset: 0,
            n: Q_PROJ_DIM,
        },
        Proj {
            name: "k_proj",
            weight: upload(&gpu, &k_bytes)?,
            scale: upload(&gpu, &k_scale)?,
            offset: Q_PROJ_DIM,
            n: KV_DIM,
        },
        Proj {
            name: "v_proj",
            weight: upload(&gpu, &v_bytes)?,
            scale: upload(&gpu, &v_scale)?,
            offset: Q_PROJ_DIM + KV_DIM,
            n: KV_DIM,
        },
    ];

    let output_bytes = MAX_M * PER_SEQ_QKV * 2;
    let sentinel = vec![SENTINEL; output_bytes + 2 * GUARD];
    let scalar_base = upload(&gpu, &sentinel)?;
    let batch_base = upload(&gpu, &sentinel)?;
    let scalar_out = scalar_base.offset(GUARD);
    let batch_out = batch_base.offset(GUARD);
    let mut first_oracle = true;
    let mut failures = 0usize;

    for m in [2_usize, 3, 4, 5, 8] {
        // ── Pass 1: each projection ALONE. Everything else in the row — the
        // other two projections' slots — is a gap that must stay sentinel, so
        // this is the direct test of the c_row_stride arithmetic.
        for p in &projections {
            gpu.copy_h2d(&sentinel, scalar_base)?;
            gpu.copy_h2d(&sentinel, batch_base)?;
            run_scalar(&gpu, scalar, input, scalar_out, p, m)?;
            run_batched(&gpu, batch4, batch16, input, batch_out, p, m)?;
            gpu.synchronize(0)?;
            let mut baseline = vec![0_u8; sentinel.len()];
            let mut observed = vec![0_u8; sentinel.len()];
            gpu.copy_d2h(scalar_base, &mut baseline)?;
            gpu.copy_d2h(batch_base, &mut observed)?;
            let live: Vec<(usize, usize)> = (0..m)
                .map(|r| ((r * PER_SEQ_QKV + p.offset) * 2, p.n * 2))
                .collect();

            if first_oracle {
                for mutation in ["output-bit", "gap", "guard", "nonfinite"] {
                    let mut bad = baseline.clone();
                    match mutation {
                        "output-bit" => bad[GUARD] ^= 1,
                        "gap" => bad[GUARD + (Q_PROJ_DIM + 1) * 2] ^= 1,
                        "guard" => bad[0] ^= 1,
                        _ => bad[GUARD..GUARD + 2].copy_from_slice(&0x7fc0_u16.to_le_bytes()),
                    }
                    let err = check(&bad, &baseline, &sentinel, &live)
                        .expect_err("known-bad output was admitted by the real oracle");
                    println!("KNOWN_BAD {mutation}: refused: {err}");
                }
                first_oracle = false;
            }

            let (mismatches, max_abs) = compare(&observed, &baseline, &live);
            let kernel = if m <= 4 { "batch4" } else { "batch16" };
            println!(
                "{} M={m} N={} K={K} stride={PER_SEQ_QKV} kernel={kernel} \
                 unequal_bf16={mismatches} max_abs={max_abs:.9}",
                p.name, p.n
            );
            if let Err(e) = check(&observed, &baseline, &sentinel, &live) {
                println!("FAIL {} M={m}: {e}", p.name);
                failures += 1;
            }
        }

        // ── Pass 2: the real thing — all three projections into one strided
        // buffer, exactly as `ms_qkv_batchm_fp8` issues them.
        gpu.copy_h2d(&sentinel, scalar_base)?;
        gpu.copy_h2d(&sentinel, batch_base)?;
        for p in &projections {
            run_scalar(&gpu, scalar, input, scalar_out, p, m)?;
            run_batched(&gpu, batch4, batch16, input, batch_out, p, m)?;
        }
        gpu.synchronize(0)?;
        let mut baseline = vec![0_u8; sentinel.len()];
        let mut observed = vec![0_u8; sentinel.len()];
        gpu.copy_d2h(scalar_base, &mut baseline)?;
        gpu.copy_d2h(batch_base, &mut observed)?;
        // Rows m..MAX_M are the gap here: a clamped or over-wide M would fill
        // them, and that is the silent-corruption mode this pass exists for.
        let live: Vec<(usize, usize)> = (0..m)
            .map(|r| (r * PER_SEQ_QKV * 2, PER_SEQ_QKV * 2))
            .collect();
        let (mismatches, max_abs) = compare(&observed, &baseline, &live);
        println!(
            "full-qkv M={m} row_elems={PER_SEQ_QKV} K={K} \
             unequal_bf16={mismatches} max_abs={max_abs:.9}"
        );
        if let Err(e) = check(&observed, &baseline, &sentinel, &live) {
            println!("FAIL full-qkv M={m}: {e}");
            failures += 1;
        }
    }

    ensure!(failures == 0, "{failures} case(s) failed");
    println!(
        "ALL PASS: Qwen3.8-27B q/k/v shapes, strided batch4 M2/M3/M4 and batch16 M5/M8, \
         exact scalar equivalence, gaps and guards intact"
    );
    Ok(())
}

/// Unequal BF16 elements and max absolute delta over the live extents only.
fn compare(observed: &[u8], baseline: &[u8], live: &[(usize, usize)]) -> (usize, f64) {
    let mut mismatches = 0;
    let mut max_abs = 0.0_f64;
    for &(start, len) in live {
        let a = &observed[GUARD + start..GUARD + start + len];
        let b = &baseline[GUARD + start..GUARD + start + len];
        mismatches += a
            .chunks_exact(2)
            .zip(b.chunks_exact(2))
            .filter(|(x, y)| x != y)
            .count();
        max_abs = values(a)
            .iter()
            .zip(values(b).iter())
            .map(|(x, y)| (x - y).abs())
            .fold(max_abs, f64::max);
    }
    (mismatches, max_abs)
}
