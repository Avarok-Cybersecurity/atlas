// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-side reference math for the batchm bench correctness gate: the
//! deterministic input generator plus bit-faithful BF16 / E2M1 / E4M3
//! converters the f64 dequant reference is built from. Split from
//! `main.rs` to stay under the 500-LoC cap.

pub(crate) struct XorShift(pub(crate) u64);
impl XorShift {
    pub(crate) fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    pub(crate) fn byte(&mut self) -> u8 {
        (self.next() >> 32) as u8
    }
    /// Uniform in [-1, 1).
    pub(crate) fn unit_f32(&mut self) -> f32 {
        ((self.next() >> 40) as f32) / ((1u64 << 23) as f32) * 2.0 - 1.0
    }
}

pub(crate) fn f32_to_bf16_bits(v: f32) -> u16 {
    // Round-to-nearest-even, matching __float2bfloat16.
    let bits = v.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

pub(crate) fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

pub(crate) const E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Standard E4M3 (1-4-3, bias 7) decode — matches (float)__nv_fp8_e4m3.
pub(crate) fn e4m3_to_f32(b: u8) -> f32 {
    let s = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = (b >> 3) & 0xF;
    let m = b & 0x7;
    if e == 0 {
        s * (m as f32) * 0.001953125 // subnormal: m * 2^-9
    } else if e == 15 && m == 7 {
        f32::NAN
    } else {
        s * (2.0f32).powi(e as i32 - 7) * (1.0 + m as f32 / 8.0)
    }
}
