// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce,
// zig/src/core/exl3_load.zig:15-23, 70-90, 120-165 (a group's admission: the trellis shape, the codebook
// marker, odd widths only with mul1, fp16 scales or packed sign words) and zig/src/core/exl3_format.zig:
// 17-28, 125-134 (Width, unpackSigns), itself after ExLlamaV3 (turboderp-org/exllamav3, MIT, Copyright
// (c) 2025 Turboderp). Apache-2.0; see EXL3-ATTRIBUTION.md.
// Changes for Atlas: translated from Zig to Rust over borrowed host byte slices; the admission rules
// become one `Exl3Parts::check` with messages; a group with a bias is refused (the W4Linear contract has
// none); the sign words are expanded to fp16 +1 / -1 at upload.

//! Responsibility: the host-side description of one EXL3 matrix ([`Exl3Parts`]: shape, width, codebook,
//! the raw trellis and scale bytes), the admission checks that turn a mis-read tensor into a startup
//! error instead of a wrong answer, and the byte conversions the upload needs.
//!
//! Layout facts (the kernels and the CPU reference rely on all of them):
//! * `y = x @ W`, `W [k, n]`: `k` inputs, `n` outputs, both multiples of 128 (the Hadamard blocks).
//! * `trellis`: int16 `[k/16, n/16, 8 * halves]`, little-endian, tile `(tk, tn)` at element
//!   `(tk * n/16 + tn) * 8 * halves`; read by the kernels as `u32` words (int16 pair `2w, 2w+1`).
//! * `halves` = 2 x bits a value (the kernels' K2), 2..=16; odd values (3..=13) only with [`Exl3Codebook::Mul1`].
//! * `su` / `sv`: fp16 `[k]` / `[n]` scales, or packed sign words int16 `[k/16]` / `[n/16]` (bit `b` of
//!   word `w` set: element `16 w + b` is -1, else +1).

use anyhow::{Result, bail, ensure};

/// The three EXL3 codebooks, numbered as the kernels' `CB` template argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Exl3Codebook {
    /// The 3-instruction default (no marker tensor).
    Inst3 = 0,
    /// The `.mcg` marker (value 0xCBAC1FED).
    Mcg = 1,
    /// The `.mul1` marker (value 0x83DCD12D); also the only codebook with half-bit widths.
    Mul1 = 2,
}

impl Exl3Codebook {
    /// The kernels' `CB` number (`exl3_ada_linear_<K2>_<CB>_<WK>`).
    pub fn id(self) -> u32 {
        self as u32
    }

    /// The codebook a marker tensor's value names (`exl3_layout.zig` `fromMarker`).
    pub fn from_marker(v: u32) -> Result<Self> {
        match v {
            MARKER_MCG => Ok(Self::Mcg),
            MARKER_MUL1 => Ok(Self::Mul1),
            other => bail!("exl3: unknown codebook marker {other:#010x}"),
        }
    }
}

/// The `.mcg` marker's value.
pub const MARKER_MCG: u32 = 0xCBAC_1FED;
/// The `.mul1` marker's value.
pub const MARKER_MUL1: u32 = 0x83DC_D12D;
/// The Hadamard block (and the alignment of `k` and `n`).
pub const HADAMARD_DIM: usize = 128;
/// fp16 bits of +1.0 and -1.0 (the expanded sign words).
pub const F16_ONE: u16 = 0x3C00;
pub const F16_MINUS_ONE: u16 = 0xBC00;

/// True when `halves` (2 x bits) with `codebook` is a width the kernels are instantiated for: every even
/// value 2..=16 on every codebook, and the mul1 half steps 3, 5, ..., 13 (7.5 bits is not a width).
pub fn width_supported(halves: u16, codebook: Exl3Codebook) -> bool {
    match halves {
        2..=16 if halves.is_multiple_of(2) => true,
        3..=13 => codebook == Exl3Codebook::Mul1,
        _ => false,
    }
}

/// Every `(halves, codebook)` the kernels serve, in a fixed order (tests and the microtest walk it).
pub fn all_widths() -> Vec<(u16, Exl3Codebook)> {
    let mut out = Vec::new();
    for cb in [Exl3Codebook::Inst3, Exl3Codebook::Mcg, Exl3Codebook::Mul1] {
        for halves in 2..=16u16 {
            if width_supported(halves, cb) {
                out.push((halves, cb));
            }
        }
    }
    out
}

/// One scale vector as stored: fp16 per element, or the packed sign words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exl3Scales<'a> {
    /// fp16 little-endian bytes, one per element (`suh` / `svh`).
    F16(&'a [u8]),
    /// int16 little-endian sign words, one per 16 elements (`su` / `sv`).
    Signs(&'a [u8]),
}

impl Exl3Scales<'_> {
    fn check(&self, len: usize, what: &str) -> Result<()> {
        match *self {
            Self::F16(b) => ensure!(
                b.len() == 2 * len,
                "exl3: {what} holds {} bytes, an fp16 [{len}] needs {}",
                b.len(),
                2 * len
            ),
            Self::Signs(b) => ensure!(
                b.len() == 2 * (len / 16),
                "exl3: {what} sign words hold {} bytes, int16 [{}] needs {}",
                b.len(),
                len / 16,
                2 * (len / 16)
            ),
        }
        Ok(())
    }

    /// The vector as fp16 bits, `len` elements (sign words expanded to +1 / -1).
    pub fn to_f16_bits(&self, len: usize) -> Vec<u16> {
        match *self {
            Self::F16(b) => b
                .chunks_exact(2)
                .take(len)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect(),
            Self::Signs(b) => unpack_signs(b, len),
        }
    }
}

/// Packed sign words to fp16 bits: bit `b` of word `w` set means element `16 w + b` is -1.
pub fn unpack_signs(words: &[u8], len: usize) -> Vec<u16> {
    let mut out = Vec::with_capacity(len);
    for c in words.chunks_exact(2) {
        let word = u16::from_le_bytes([c[0], c[1]]);
        for b in 0..16 {
            out.push(if (word >> b) & 1 == 1 {
                F16_MINUS_ONE
            } else {
                F16_ONE
            });
        }
    }
    out.truncate(len);
    out
}

/// One EXL3 matrix as host bytes, borrowed from the reader (`spark_runtime::weights::exl3`) or a test.
#[derive(Clone, Copy, Debug)]
pub struct Exl3Parts<'a> {
    /// Input features (`W`'s rows, the trellis's first dimension x 16).
    pub k: usize,
    /// Output features.
    pub n: usize,
    /// 2 x bits a value (the kernels' K2).
    pub halves: u16,
    pub codebook: Exl3Codebook,
    /// int16 `[k/16, n/16, 8 * halves]` little-endian bytes.
    pub trellis: &'a [u8],
    /// Input scales, `[k]`.
    pub su: Exl3Scales<'a>,
    /// Output scales, `[n]`.
    pub sv: Exl3Scales<'a>,
    /// fp16 `[n]` bias bytes, if the group has one (refused by [`Exl3Parts::check`]).
    pub bias: Option<&'a [u8]>,
}

/// int16 elements of one tile at `halves`.
pub fn tile_elems(halves: u16) -> usize {
    8 * halves as usize
}

/// Bytes of the trellis of a `[k, n]` matrix at `halves`.
pub fn trellis_bytes(k: usize, n: usize, halves: u16) -> usize {
    (k / 16) * (n / 16) * tile_elems(halves) * 2
}

impl Exl3Parts<'_> {
    /// The admission rules; every refusal names the rule.
    pub fn check(&self) -> Result<()> {
        let (k, n) = (self.k, self.n);
        ensure!(
            k > 0 && n > 0 && k.is_multiple_of(HADAMARD_DIM) && n.is_multiple_of(HADAMARD_DIM),
            "exl3: [{k} -> {n}] is not a whole number of 128-value Hadamard blocks"
        );
        ensure!(
            width_supported(self.halves, self.codebook),
            "exl3: {} half-bits with the {:?} codebook is not a served width (even 2..=16 on every \
             codebook, odd 3..=13 with mul1 only)",
            self.halves,
            self.codebook
        );
        let want = trellis_bytes(k, n, self.halves);
        ensure!(
            self.trellis.len() == want,
            "exl3: the trellis of [{k} -> {n}] at {} half-bits holds {} bytes, int16 [{}, {}, {}] needs {want}",
            self.halves,
            self.trellis.len(),
            k / 16,
            n / 16,
            tile_elems(self.halves)
        );
        self.su.check(k, "su")?;
        self.sv.check(n, "sv")?;
        ensure!(
            self.bias.is_none(),
            "exl3: a group with a bias is not served (the W4Linear contract has no bias term)"
        );
        Ok(())
    }

    /// Bits a weight, times 100 (`W4Format::Exl3`): the code rate, 50 per half-bit.
    pub fn bpw_x100(&self) -> u16 {
        self.halves * 50
    }

    /// The trellis as int16 elements (little-endian).
    pub fn trellis_u16(&self) -> Vec<u16> {
        self.trellis
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    }

    /// `su` as fp16 bits `[k]`.
    pub fn suh_bits(&self) -> Vec<u16> {
        self.su.to_f16_bits(self.k)
    }

    /// `sv` as fp16 bits `[n]`.
    pub fn svh_bits(&self) -> Vec<u16> {
        self.sv.to_f16_bits(self.n)
    }
}

/// Little-endian bytes of `u16` values.
pub fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
