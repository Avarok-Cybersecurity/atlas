// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: deterministic synthetic weights for the L07 timing examples,
//! so a GEMV or GEMM reads realistic, finite data instead of uninitialized
//! device memory.
//!
//! Timing-only inputs still need care. An FP8 byte of 0x7F or 0xFF is NaN, a
//! zero NVFP4 block scale blanks the output, and constant data lets a kernel
//! look faster than it is on real checkpoints. The generators below avoid all
//! three. One 4 MiB pattern is generated per buffer and repeated across it, so
//! filling a 1.3 GB lm_head costs a few hundred H2D copies, not 1.3 GB of RNG.
#![allow(dead_code)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// What a buffer holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fill {
    /// BF16 values uniform in [-1, 1).
    Bf16,
    /// FP8 E4M3 bytes, never the NaN encodings 0x7F and 0xFF.
    Fp8E4m3,
    /// FP32 block scales in [0.005, 0.15).
    ScaleF32,
    /// NVFP4 packed nibbles: any byte is a valid pair of E2M1 codes.
    Nvfp4Packed,
    /// NVFP4 E4M3 block scales, bytes 0x30..=0x47 (0.5 to 3.75), finite and
    /// positive. 0x7F and 0xFF are NaN and a zero scale would blank the output.
    Nvfp4Scale,
    /// Arbitrary bytes.
    Bytes,
}

/// splitmix64, seeded explicitly so a failing run is reproducible.
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

/// Size of the repeated pattern.
pub const CHUNK: usize = 4 << 20;

/// `len` bytes of `kind`. `len` should be a multiple of the element size
/// (2 for BF16, 4 for FP32); a ragged tail is zero-padded.
pub fn pattern(kind: Fill, len: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng(seed ^ 0xA7_1A5_0000_0001);
    let mut out = Vec::with_capacity(len + 4);
    match kind {
        Fill::Bf16 => {
            while out.len() < len {
                let v = rng.unit() * 2.0 - 1.0;
                out.extend(half::bf16::from_f32(v).to_bits().to_le_bytes());
            }
        }
        Fill::ScaleF32 => {
            while out.len() < len {
                out.extend((0.005 + 0.145 * rng.unit()).to_le_bytes());
            }
        }
        Fill::Fp8E4m3 => {
            while out.len() < len {
                let b = (rng.next_u64() >> 24) as u8;
                // 0x7F and 0xFF are NaN: step to the neighbouring finite code.
                out.push(if b & 0x7F == 0x7F { b ^ 1 } else { b });
            }
        }
        Fill::Nvfp4Scale => {
            while out.len() < len {
                out.push(0x30 + ((rng.next_u64() >> 24) % 24) as u8);
            }
        }
        Fill::Nvfp4Packed | Fill::Bytes => {
            while out.len() < len {
                out.push((rng.next_u64() >> 24) as u8);
            }
        }
    }
    out.truncate(len);
    out
}

/// Fill `bytes` of device memory at `ptr` with `kind`, repeating one pattern.
pub fn fill_device(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    bytes: usize,
    kind: Fill,
    seed: u64,
) -> Result<()> {
    let chunk = pattern(kind, bytes.min(CHUNK), seed);
    let mut off = 0;
    while off < bytes {
        let n = (bytes - off).min(chunk.len());
        gpu.copy_h2d(&chunk[..n], ptr.offset(off))?;
        off += n;
    }
    Ok(())
}

/// Allocate and fill one buffer.
pub fn alloc_filled(
    gpu: &dyn GpuBackend,
    bytes: usize,
    kind: Fill,
    seed: u64,
) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes)?;
    fill_device(gpu, p, bytes, kind, seed)?;
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lengths_are_exact_and_deterministic() {
        for kind in [
            Fill::Bf16,
            Fill::Fp8E4m3,
            Fill::ScaleF32,
            Fill::Nvfp4Packed,
            Fill::Nvfp4Scale,
            Fill::Bytes,
        ] {
            let a = pattern(kind, 1000, 7);
            assert_eq!(a.len(), 1000, "{kind:?}");
            assert_eq!(a, pattern(kind, 1000, 7), "{kind:?}");
            assert_ne!(a, pattern(kind, 1000, 8), "{kind:?}");
        }
    }

    #[test]
    fn bf16_is_finite_and_in_range() {
        let p = pattern(Fill::Bf16, 8192, 1);
        for c in p.chunks_exact(2) {
            let v = half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32();
            // BF16 rounding can land exactly on 1.0, so the range is closed.
            assert!(v.is_finite() && (-1.0..=1.0).contains(&v), "{v}");
        }
    }

    #[test]
    fn fp8_never_nan() {
        let p = pattern(Fill::Fp8E4m3, 1 << 16, 3);
        assert!(p.iter().all(|b| b & 0x7F != 0x7F));
        // Not degenerate: many distinct codes.
        let mut seen = [false; 256];
        for b in &p {
            seen[*b as usize] = true;
        }
        assert!(seen.iter().filter(|s| **s).count() > 200);
    }

    #[test]
    fn nvfp4_scales_are_finite_positive_e4m3() {
        let p = pattern(Fill::Nvfp4Scale, 4096, 5);
        assert!(p.iter().all(|b| (0x30..=0x47).contains(b)));
    }

    #[test]
    fn f32_scales_in_range() {
        let p = pattern(Fill::ScaleF32, 4096, 9);
        for c in p.chunks_exact(4) {
            let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            assert!((0.005..0.15).contains(&v), "{v}");
        }
    }

    #[test]
    fn unit_is_half_open() {
        let mut r = Rng(42);
        for _ in 0..10_000 {
            let u = r.unit();
            assert!((0.0..1.0).contains(&u), "{u}");
        }
    }
}
