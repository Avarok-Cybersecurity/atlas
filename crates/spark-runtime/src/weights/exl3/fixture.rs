// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: test fixtures for the EXL3 reader, written from the documented format (TensorFold
//! `docs/recipes/exl3.md`) rather than from the reader: a seeded PRNG, a packer that lays stream bits
//! into a tile's int16 words, the shift-register reading of a tile's states, and a safetensors writer
//! for synthetic checkpoints.

use std::path::Path;

use super::format::Width;

/// splitmix64: a small seeded generator (no dependency, stable across platforms).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn bits(&mut self, n: usize) -> Vec<bool> {
        (0..n).map(|_| self.next_u64() & 1 == 1).collect()
    }

    pub fn i16s(&mut self, n: usize) -> Vec<i16> {
        (0..n).map(|_| self.next_u64() as i16).collect()
    }

    /// A uniform value in `[lo, hi)`.
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64)
    }

    /// fp16 bit patterns of uniform values in `[lo, hi)`, rounded once.
    pub fn fp16s(&mut self, n: usize, lo: f64, hi: f64) -> Vec<u16> {
        (0..n)
            .map(|_| super::fp16::from_f64(self.uniform(lo, hi)))
            .collect()
    }
}

/// Lays stream bits (index 0 first) into int16 words as the format describes: the words, read in
/// little-endian pairs as 32-bit words, are read from the most significant bit.
pub fn pack_stream(bits: &[bool]) -> Vec<i16> {
    assert!(bits.len().is_multiple_of(32));
    let mut words = Vec::with_capacity(bits.len() / 16);
    for chunk in bits.chunks_exact(32) {
        let mut word = 0u32;
        for &b in chunk {
            word = (word << 1) | u32::from(b);
        }
        words.push((word & 0xFFFF) as u16 as i16);
        words.push((word >> 16) as u16 as i16);
    }
    words
}

/// New stream bits value `p` adds, from the format description: `bits` for a whole width; for
/// `bits = KA + 1/2`, KA at even positions and KA + 1 at odd ones.
pub fn new_bits(w: Width, p: usize) -> u32 {
    let halves = u32::from(w.halves());
    let ka = halves / 2;
    if !halves.is_multiple_of(2) && p % 2 == 1 {
        ka + 1
    } else {
        ka
    }
}

/// A tile's 256 states read as a 16-bit shift register over the circular stream: the register starts
/// as the last 16 bits of the ring, and value `p` shifts in its new bits.
pub fn shift_register_states(w: Width, bits: &[bool]) -> [u16; 256] {
    let ring = bits.len();
    let mut reg = 0u32;
    for &b in &bits[ring - 16..] {
        reg = (reg << 1) | u32::from(b);
    }
    let mut at = 0usize;
    let mut out = [0u16; 256];
    for (p, state) in out.iter_mut().enumerate() {
        for _ in 0..new_bits(w, p) {
            reg = ((reg << 1) | u32::from(bits[at])) & 0xFFFF;
            at += 1;
        }
        *state = reg as u16;
    }
    assert_eq!(at, ring, "the new bits of 256 values fill the ring");
    out
}

/// One tensor to write.
pub struct StTensor {
    pub name: String,
    pub dtype: &'static str,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl StTensor {
    pub fn new(name: &str, dtype: &'static str, shape: &[usize], data: Vec<u8>) -> Self {
        Self {
            name: name.to_string(),
            dtype,
            shape: shape.to_vec(),
            data,
        }
    }
}

pub fn i16_bytes(v: &[i16]) -> Vec<u8> {
    v.iter().flat_map(|w| w.to_le_bytes()).collect()
}

pub fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// The bytes of a safetensors file holding `tensors` back to back in the given order.
pub fn safetensors_bytes(tensors: &[StTensor]) -> Vec<u8> {
    let mut header = String::from("{");
    let mut at = 0usize;
    for (i, t) in tensors.iter().enumerate() {
        if i != 0 {
            header.push(',');
        }
        let shape: Vec<String> = t.shape.iter().map(|d| d.to_string()).collect();
        header.push_str(&format!(
            "\"{}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            t.name,
            t.dtype,
            shape.join(","),
            at,
            at + t.data.len()
        ));
        at += t.data.len();
    }
    header.push('}');
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    for t in tensors {
        out.extend_from_slice(&t.data);
    }
    out
}

pub fn write_safetensors(path: &Path, tensors: &[StTensor]) {
    std::fs::write(path, safetensors_bytes(tensors)).expect("write fixture shard");
}

/// The parts of one synthetic group, and what a load must return for them.
pub struct SynthGroup {
    pub trellis: Vec<i16>,
    pub suh: Vec<u16>,
    pub svh: Vec<u16>,
    pub su: Vec<i16>,
    pub sv: Vec<i16>,
    pub bias: Vec<u16>,
}

/// Random trellis words, fp16 scales in `[0.5, 1.5)`, sign words and a bias for a `[k, n]` group.
pub fn synth_group(w: Width, k: usize, n: usize, seed: u64) -> SynthGroup {
    let mut rng = Rng::new(seed);
    SynthGroup {
        trellis: rng.i16s((k / 16) * (n / 16) * w.tile_words()),
        suh: rng.fp16s(k, 0.5, 1.5),
        svh: rng.fp16s(n, 0.5, 1.5),
        su: rng.i16s(k / 16),
        sv: rng.i16s(n / 16),
        bias: rng.fp16s(n, -0.25, 0.25),
    }
}

/// The tensors of a group: the trellis, fp16 or packed scales, an optional bias, an optional marker.
#[allow(clippy::too_many_arguments)]
pub fn group_tensors(
    prefix: &str,
    w: Width,
    k: usize,
    n: usize,
    g: &SynthGroup,
    signs: bool,
    bias: bool,
    marker: Option<u32>,
) -> Vec<StTensor> {
    let name = |s: &str| format!("{prefix}.{s}");
    let mut out = vec![StTensor::new(
        &name("trellis"),
        "I16",
        &[k / 16, n / 16, w.tile_words()],
        i16_bytes(&g.trellis),
    )];
    if signs {
        out.push(StTensor::new(
            &name("su"),
            "I16",
            &[k / 16],
            i16_bytes(&g.su),
        ));
        out.push(StTensor::new(
            &name("sv"),
            "I16",
            &[n / 16],
            i16_bytes(&g.sv),
        ));
    } else {
        out.push(StTensor::new(&name("suh"), "F16", &[k], u16_bytes(&g.suh)));
        out.push(StTensor::new(&name("svh"), "F16", &[n], u16_bytes(&g.svh)));
    }
    if bias {
        out.push(StTensor::new(
            &name("bias"),
            "F16",
            &[n],
            u16_bytes(&g.bias),
        ));
    }
    if let Some(value) = marker {
        let suffix = match value {
            super::format::MCG_MUL => "mcg",
            _ => "mul1",
        };
        out.push(StTensor::new(
            &name(suffix),
            "I32",
            &[],
            value.to_le_bytes().to_vec(),
        ));
    }
    out
}
