// SPDX-License-Identifier: AGPL-3.0-only

//! The fixture half of `gqa_pack_gpu_tests` — the shape constants, the context
//! and magnitude tables, the deterministic input generator, and the two
//! post-conditions every arm's output must satisfy.
//!
//! Split out only because the pair exceeds the repository's 500-line file cap
//! together. The test file next door states what is being proved; nothing here
//! launches a kernel.

// ── The shape the packed kernels are COMPILED for ──────────────────────────
//
// Asserted against `gqa_pack_shape_ok` below rather than merely written here:
// `nq / nkv == 6` and `head_dim == 256` are the Qwen3.8-27B decode shape and
// the kernels' `PD_GQA` / `PD_HDIM` `#define`s, and a test built on any other
// numbers would be testing a route production refuses.
pub(super) const NQ: u32 = 24;
pub(super) const NKV: u32 = 4;
pub(super) const HD: u32 = 256;
pub(super) const BLOCK_SIZE: u32 = 16;
/// Non-unity on purpose: FP8 hoists `k_scale` into the per-position score
/// multiply and `v_scale` into the single smem write, so a 1.0/1.0 run would
/// not exercise either hoist.
pub(super) const K_SCALE: f32 = 0.7;
pub(super) const V_SCALE: f32 = 1.3;
/// The sentinel both output buffers start at. Any BF16 pair that survives as
/// `0xA5A5` is a slice no kernel touched.
pub(super) const UNWRITTEN: u8 = 0xA5;

/// A context-length case.
pub(super) struct Case {
    pub(super) label: &'static str,
    pub(super) seq_lens: &'static [u32],
    pub(super) sliding: u32,
}

/// Context lengths, chosen against the kernel's own block structure.
///
/// Each warp takes `ceil(attended / 8)` positions and walks them in `BC = 4`
/// batches clipped to the physical block, so the interesting lengths are the
/// ones whose per-warp chunk is NOT a multiple of 4 (remainder loop) and the
/// ones that span many physical blocks (block-table indirection).
pub(super) const CASES: &[Case] = &[
    // chunk = 5: four batched + one remainder position per warp.
    Case {
        label: "len-37-unaligned",
        seq_lens: &[37],
        sliding: 0,
    },
    // chunk = 1: the whole attended range is remainder.
    Case {
        label: "len-1-single-token",
        seq_lens: &[1],
        sliding: 0,
    },
    // chunk = 8: fully aligned, two physical blocks per warp.
    Case {
        label: "len-64-block-exact",
        seq_lens: &[64],
        sliding: 0,
    },
    // Three sequences of different lengths in one launch.
    Case {
        label: "len-129-256-1000-mixed",
        seq_lens: &[129, 256, 1000],
        sliding: 0,
    },
    // 128 physical blocks; chunk = 256, well past L2 residency of one head.
    Case {
        label: "len-2048-long",
        seq_lens: &[2048],
        sliding: 0,
    },
    // Prime length: every warp's chunk has a remainder and the last warp clips.
    Case {
        label: "len-4093-prime",
        seq_lens: &[4093],
        sliding: 0,
    },
    // window_start > 0 moves the partition base off zero for both kernels.
    Case {
        label: "len-600-601-sliding-128",
        seq_lens: &[600, 601],
        sliding: 128,
    },
];

/// The K/V magnitude bands.
///
/// FP8 E4M3's exponent field (bias 7) is drawn from `lo..=hi`, so the band is
/// exact rather than the result of a host-side quantization that could itself
/// be the bug. `bf16_mag` is the matching BF16 amplitude. The top band matters
/// most: E4M3 saturates at 448, and a defect that only shows at the top of the
/// range is the kind a unit-magnitude-only test would miss.
pub(super) struct Band {
    pub(super) label: &'static str,
    pub(super) fp8_exp: (u32, u32),
    pub(super) bf16_mag: f32,
}

pub(super) const BANDS: &[Band] = &[
    Band {
        label: "small",
        fp8_exp: (1, 4),
        bf16_mag: 0.031_25,
    },
    Band {
        label: "unit",
        fp8_exp: (6, 8),
        bf16_mag: 1.0,
    },
    Band {
        label: "near-fp8-max",
        fp8_exp: (14, 15),
        bf16_mag: 448.0,
    },
];

/// xorshift64. A fixed stream so a failure is reproducible from its label.
fn xs(s: &mut u64) -> u32 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    (*s >> 32) as u32
}

/// BF16 by truncation — every generated value is exactly representable, so the
/// fixture contributes no rounding of its own.
pub(super) fn bf16_bits(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

pub(super) fn uniform(s: &mut u64, mag: f32) -> f32 {
    let u = xs(s) as f32 / u32::MAX as f32;
    (u * 2.0 - 1.0) * mag
}

/// One E4M3 byte with the exponent field inside `lo..=hi`.
///
/// `S.1111.111` is NaN in E4M3; the mantissa is stepped down rather than the
/// draw rejected, so the stream stays aligned across bands.
pub(super) fn fp8_byte(s: &mut u64, lo: u32, hi: u32) -> u8 {
    let r = xs(s);
    let sign = (r & 1) as u8;
    let exp = lo + (r >> 1) % (hi - lo + 1);
    let mut mant = ((r >> 8) & 7) as u8;
    if exp == 15 && mant == 7 {
        mant = 6;
    }
    (sign << 7) | ((exp as u8) << 3) | mant
}

/// Q amplitude, fixed across bands.
///
/// Deliberately NOT scaled with the K band: at `mag = 448` on both sides the
/// pre-softmax scores would run to 1e6 and both kernels would agree on a
/// buffer of infinities, which any kernel passes. 0.25 keeps the reference
/// output finite (asserted) while the KV side still spans the full FP8 range.
const Q_MAG: f32 = 0.25;

/// The per-case fixture: Q, the KV pool bytes, the block table and seq lens.
pub(super) struct Fixture {
    pub(super) q: Vec<u8>,
    pub(super) k: Vec<u8>,
    pub(super) v: Vec<u8>,
    pub(super) block_table: Vec<u8>,
    pub(super) seq_lens: Vec<u8>,
    pub(super) max_blocks_per_seq: u32,
    pub(super) num_seqs: u32,
    pub(super) out_bytes: usize,
}

/// `elem_bytes` is 1 for the FP8 pools and 2 for BF16; `fill` produces one
/// element's bytes in cache order.
pub(super) fn fixture(
    case: &Case,
    seed: u64,
    elem_bytes: usize,
    mut fill: impl FnMut(&mut u64) -> Vec<u8>,
) -> Fixture {
    let num_seqs = case.seq_lens.len() as u32;
    let max_len = case.seq_lens.iter().copied().max().unwrap();
    let mbps = max_len.div_ceil(BLOCK_SIZE);
    let mut total_blocks = num_seqs * mbps + 3;
    // The scramble below is a bijection only when gcd(7, total_blocks) == 1.
    if total_blocks.is_multiple_of(7) {
        total_blocks += 1;
    }

    let mut s = seed | 1;
    let mut q = Vec::with_capacity((num_seqs * NQ * HD) as usize * 2);
    for _ in 0..(num_seqs * NQ * HD) {
        q.extend_from_slice(&bf16_bits(uniform(&mut s, Q_MAG)).to_le_bytes());
    }

    let pool_elems = (total_blocks * BLOCK_SIZE * NKV * HD) as usize;
    let mut k = Vec::with_capacity(pool_elems * elem_bytes);
    let mut v = Vec::with_capacity(pool_elems * elem_bytes);
    for _ in 0..pool_elems {
        k.extend_from_slice(&fill(&mut s));
        v.extend_from_slice(&fill(&mut s));
    }

    // Logical block j of sequence i lands on a scrambled physical page, so a
    // kernel that ignored the table (or read it with the wrong stride) cannot
    // agree with one that honours it.
    let mut block_table = Vec::with_capacity((num_seqs * mbps) as usize * 4);
    for i in 0..num_seqs {
        for j in 0..mbps {
            let phys = ((i * mbps + j) * 7 + 3) % total_blocks;
            block_table.extend_from_slice(&(phys as i32).to_le_bytes());
        }
    }
    let seq_lens = case
        .seq_lens
        .iter()
        .flat_map(|l| (*l as i32).to_le_bytes())
        .collect();

    Fixture {
        q,
        k,
        v,
        block_table,
        seq_lens,
        max_blocks_per_seq: mbps,
        num_seqs,
        out_bytes: (num_seqs * NQ * HD) as usize * 2,
    }
}

/// Every one of the `num_seqs * NQ` head slices must have left the sentinel.
///
/// This is the positive proof that the arm RAN, and for the packed arm it is
/// also the proof it was the packed kernel: under `grid = (NKV, num_seqs)` an
/// unpacked kernel writes 4 of 24 heads and this fails on head 4.
pub(super) fn heads_written(buf: &[u8], num_seqs: u32) -> Result<(), String> {
    let head_bytes = HD as usize * 2;
    for seq in 0..num_seqs as usize {
        for h in 0..NQ as usize {
            let off = (seq * NQ as usize + h) * head_bytes;
            if buf[off..off + head_bytes].iter().all(|b| *b == UNWRITTEN) {
                return Err(format!("seq {seq} head {h} still all-0x{UNWRITTEN:02X}"));
            }
        }
    }
    Ok(())
}

/// The reference output must be finite — a buffer of NaNs would compare equal
/// to itself and prove nothing about the arithmetic.
pub(super) fn all_finite(buf: &[u8]) -> Result<(), String> {
    for (i, w) in buf.chunks_exact(2).enumerate() {
        let f = f32::from_bits((u16::from_le_bytes([w[0], w[1]]) as u32) << 16);
        if !f.is_finite() {
            return Err(format!("reference element {i} is {f}"));
        }
    }
    Ok(())
}
