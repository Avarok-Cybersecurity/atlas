// SPDX-License-Identifier: AGPL-3.0-only
//! Fixture and byte oracle for `fp8_moe_grouped_decode_microtest`, split out
//! of it (500-LoC cap) by exact piecewise copy.

use anyhow::{Result, ensure};
use half::bf16;
use spark_model::weight_map::{Fp8Weight, WeightQuantFormat};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{GUARD, H};

pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
}

pub fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len().max(16))?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

pub fn fp8_bytes(rng: &mut Rng, n: usize) -> Vec<u8> {
    (0..n)
        .map(|_| {
            let x = rng.next();
            ((x % 120) as u8) | (((x >> 7) & 1) as u8 * 128)
        })
        .collect()
}

pub fn scale_bytes(rng: &mut Rng, n: usize, k: usize) -> Vec<u8> {
    (0..n.div_ceil(128) * k.div_ceil(128))
        .flat_map(|_| (((rng.next() % 16 + 1) as f32) / 512.0).to_le_bytes())
        .collect()
}

pub fn bf16_bytes(rng: &mut Rng, n: usize, scale: f32) -> Vec<u8> {
    (0..n)
        .flat_map(|_| {
            bf16::from_f32(((rng.next() % 2049) as f32 - 1024.0) / 1024.0 * scale)
                .to_bits()
                .to_le_bytes()
        })
        .collect()
}

pub fn fp8w(weight: DevicePtr, row_scale: DevicePtr, n: usize, k: usize) -> Fp8Weight {
    Fp8Weight {
        weight,
        row_scale,
        n: n as u32,
        k: k as u32,
        scale_format: WeightQuantFormat::Fp8BlockScaled,
    }
}

/// The oracle: live rows byte-equal and finite, everything else still sentinel.
pub fn check(observed: &[u8], baseline: &[u8], sentinel: &[u8], m: usize) -> Result<()> {
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
