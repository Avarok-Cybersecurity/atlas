// SPDX-License-Identifier: AGPL-3.0-only
//! Real-GB10 validation: every GPU dequant kernel must reproduce the CPU
//! reference dequant (the numeric oracle) byte-for-byte in BF16.
//!
//! Ignored by default (needs a CUDA device + the real PTX). Run on a GB10:
//!   cargo test -p spark-runtime --features cuda -- --ignored gguf_gpu_validate
//!
//! Builds deterministic multi-block buffers (pseudo-random quant codes, sane
//! per-block fp16 scales), dequants each on CPU and on the device, and asserts
//! the two BF16 outputs are identical.

use super::{dequant_cpu, dequant_gpu};
use crate::cuda_backend::AtlasCudaBackend;
use crate::gpu::GpuBackend;

/// Safe, varied fp16 scales (finite, no NaN/Inf): 1.0, 0.5, 2.0, 0.25.
const SCALES_F16: [u16; 4] = [0x3C00, 0x3800, 0x4000, 0x3400];

/// Build `n_blocks` raw quant blocks of `block_bytes` each. Every byte is a
/// deterministic LCG pattern (varied quant codes), then the fp16 scale slot(s)
/// listed in `f16_offsets` are overwritten per-block with a rotating safe scale.
fn build_blocks(n_blocks: usize, block_bytes: usize, f16_offsets: &[usize]) -> Vec<u8> {
    let mut raw = vec![0u8; n_blocks * block_bytes];
    // Deterministic LCG (no rand dep); index-seeded so it varies across bytes.
    let mut state: u32 = 0x1234_5678;
    for byte in raw.iter_mut() {
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        *byte = (state >> 16) as u8;
    }
    for b in 0..n_blocks {
        let base = b * block_bytes;
        for (k, &off) in f16_offsets.iter().enumerate() {
            let s = SCALES_F16[(b + k) % SCALES_F16.len()];
            raw[base + off] = (s & 0xFF) as u8;
            raw[base + off + 1] = (s >> 8) as u8;
        }
    }
    raw
}

fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

/// Run one type through CPU + GPU and assert BF16 outputs match exactly.
#[allow(clippy::too_many_arguments)]
fn check_type(
    gpu: &AtlasCudaBackend,
    label: &str,
    id: u32,
    qk: usize,
    block_bytes: usize,
    f16_offsets: &[usize],
    q2_group: usize,
    n_blocks: usize,
) {
    let numel = n_blocks * qk;
    let raw = build_blocks(n_blocks, block_bytes, f16_offsets);

    // CPU oracle.
    let cpu = dequant_cpu::to_bf16_bytes(id, q2_group, &raw, numel)
        .unwrap_or_else(|e| panic!("{label}: CPU dequant failed: {e}"));
    assert_eq!(cpu.len(), numel * 2, "{label}: CPU output length");

    // GPU under test: upload raw blocks, launch kernel, read BF16 back.
    let q_ptr = gpu.alloc(raw.len()).unwrap();
    gpu.copy_h2d(&raw, q_ptr).unwrap();
    let bf16_ptr = dequant_gpu::to_bf16(gpu, id, q_ptr, numel, q2_group)
        .unwrap_or_else(|e| panic!("{label}: GPU dequant failed: {e}"));
    let mut got = vec![0u8; numel * 2];
    gpu.copy_d2h(bf16_ptr, &mut got).unwrap();
    gpu.free(q_ptr).unwrap();
    gpu.free(bf16_ptr).unwrap();

    if got != cpu {
        let a = bf16_bytes_to_f32(&cpu);
        let b = bf16_bytes_to_f32(&got);
        let mut shown = 0;
        let mut mismatches = 0;
        for i in 0..numel {
            if cpu[i * 2..i * 2 + 2] != got[i * 2..i * 2 + 2] {
                mismatches += 1;
                if shown < 12 {
                    eprintln!("  {label}[{i}]: cpu={} gpu={}", a[i], b[i]);
                    shown += 1;
                }
            }
        }
        panic!("{label}: {mismatches}/{numel} BF16 elements differ CPU vs GPU");
    }
    eprintln!("  {label}: {numel} elements match CPU oracle exactly ✓");
}

#[test]
#[ignore = "requires a real CUDA GB10 device + compiled PTX (no ATLAS_SKIP_BUILD)"]
fn gguf_gpu_validate_matches_cpu_oracle() {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())
        .expect("construct AtlasCudaBackend (needs a CUDA device + real PTX)");

    let n = 37; // odd, exercises many grid blocks + the 256-thread stride loop
    // Q8_0: QK=32, 34 B, one fp16 scale at offset 0.
    check_type(&gpu, "Q8_0", 8, 32, 34, &[0], 128, n);
    // Q4_K: QK=256, 144 B, fp16 d@0 + fp16 dmin@2.
    check_type(&gpu, "Q4_K", 12, 256, 144, &[0, 2], 128, n);
    // Q6_K: QK=256, 210 B, fp16 d@208.
    check_type(&gpu, "Q6_K", 14, 256, 210, &[208], 128, n);
    // Q2_0 id42 group-128: QK=128, 34 B, fp16 scale@0.
    check_type(&gpu, "Q2_0_g128", 42, 128, 34, &[0], 128, n);
    // Q2_0 id42 group-64 (fork-master variant): QK=64, 18 B, fp16 scale@0.
    check_type(&gpu, "Q2_0_g64", 42, 64, 18, &[0], 64, n);
}
