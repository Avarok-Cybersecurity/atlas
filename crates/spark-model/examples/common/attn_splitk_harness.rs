// SPDX-License-Identifier: AGPL-3.0-only
//! Fixtures and graders for `native_attn_decode_splitk_hopper_microtest`.
//!
//! Split out of the example for the repository's 500-LoC cap. Nothing here
//! touches a kernel: it builds the paged-KV fixture, scores two BF16 outputs,
//! and owns the KNOWN_BAD control — so the oracle can be read without reading
//! the launch sequence.
#![allow(dead_code)]

use anyhow::{Result, ensure};
use half::bf16;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// Qwen3.8-27B full attention (`kernels/hopper/qwen3.8-27b/MODEL.toml`).
pub const NQ: usize = 24;
pub const NKV: usize = 4;
pub const HD: usize = 256;
/// Paged KV block size. 16 is the allocator's page; it also makes every
/// `PD_BC=4` batch land inside one physical block, which is the loop shape the
/// twins are tuned for.
pub const BLOCK: usize = 16;
/// Sentinel bytes either side of every device buffer.
pub const GUARD: usize = 256;
pub const SENTINEL: u8 = 0x5a;

/// Context lengths under test.
///
/// 1335 and 4847 are the two the round-13 nsys captures actually decoded at
/// (`ATTN-DECODE-SPLITK-ATTRIBUTION.md` §C.1 and §C.5); 16384 is two thirds of
/// the campaign's `--max-seq-len 24576` and the shape where the split has the
/// most to win. 1335 is also the arm where `PD_MIN_KV_PER_SPLIT` starts
/// clamping: 1335/11 = 121 < 256, so six splits carry work and five are empty.
pub const LENGTHS: [usize; 3] = [1335, 4847, 16384];
/// Co-batched row counts. 1 is the C=1 shape the lever is for; 16 is the
/// serve's `--max-batch-size`.
pub const ROWS: [usize; 3] = [1, 4, 16];
/// Split counts graded per arm. 1 is the control — the geometry an H100 ran
/// before #928.
pub const SPLITS: [u32; 4] = [1, 2, 4, 6];

pub const MAX_L: usize = 16384;
pub const MAX_N: usize = 16;
pub const BLOCKS_PER_SEQ: usize = MAX_L / BLOCK;
pub const TOTAL_BLOCKS: usize = MAX_N * BLOCKS_PER_SEQ;

/// A cheap deterministic PRNG — reproducible across hosts, unlike `rand`'s
/// default seeding, so a failure is re-runnable from the printed arm alone.
pub struct Lcg(pub u64);

impl Lcg {
    pub fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }
}

/// Upload with `GUARD` sentinel bytes either side; returns the pointer to the
/// DATA, so an overrun in either direction lands in bytes this harness checks.
pub fn upload_guarded(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let mut framed = vec![SENTINEL; bytes.len() + 2 * GUARD];
    framed[GUARD..GUARD + bytes.len()].copy_from_slice(bytes);
    let base = gpu.alloc(framed.len())?;
    gpu.copy_h2d(&framed, base)?;
    Ok(DevicePtr(base.0 + GUARD as u64))
}

/// Both guard bands around a buffer uploaded by [`upload_guarded`].
pub fn guards_intact(gpu: &dyn GpuBackend, data: DevicePtr, len: usize) -> Result<()> {
    let mut head = vec![0u8; GUARD];
    let mut tail = vec![0u8; GUARD];
    gpu.copy_d2h(DevicePtr(data.0 - GUARD as u64), &mut head)?;
    gpu.copy_d2h(DevicePtr(data.0 + len as u64), &mut tail)?;
    ensure!(
        head.iter().all(|&b| b == SENTINEL) && tail.iter().all(|&b| b == SENTINEL),
        "a kernel wrote outside its extent"
    );
    Ok(())
}

pub fn to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|x| bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32())
        .collect()
}

/// How two BF16 outputs compare: relative RMS, worst element, and cosine.
#[derive(Debug, Clone, Copy)]
pub struct Score {
    pub rel_rms: f64,
    pub max_abs: f64,
    pub cosine: f64,
    pub bit_equal: bool,
}

/// Grade `observed` against `reference`.
///
/// ⚠️ NOT an equality, and the reason is arithmetic, not sloppiness. Splitting
/// the KV range re-brackets the online-softmax merge — `(a⊕b)⊕c` becomes
/// `a⊕(b⊕c)` — and that merge is not associative, so two split counts CANNOT
/// agree bit for bit and an equality here would be a test that can only fail.
/// What must be bit-exact is a different statement, and the example asserts it
/// separately: the same sequence at the same `num_splits` decoded alone and
/// co-batched. That one has no reassociation in it at all, and it is the
/// property the determinism pin exists for.
pub fn score(observed: &[u8], reference: &[u8]) -> Score {
    let a = to_f32(observed);
    let b = to_f32(reference);
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut max_abs = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let (x, y) = (*x as f64, *y as f64);
        num += (x - y) * (x - y);
        den += y * y;
        dot += x * y;
        na += x * x;
        nb += y * y;
        max_abs = max_abs.max((x - y).abs());
    }
    Score {
        rel_rms: if den > 0.0 {
            (num / den).sqrt()
        } else {
            num.sqrt()
        },
        max_abs,
        cosine: if na > 0.0 && nb > 0.0 {
            dot / (na.sqrt() * nb.sqrt())
        } else {
            0.0
        },
        bit_equal: observed == reference,
    }
}

/// The band a re-bracketed softmax merge may land in.
///
/// 2e-3 relative RMS. Sourced, not chosen: the output is BF16, whose 8-bit
/// significand gives a representation error of up to 2^-9 = 2.0e-3 per
/// element on its own, so anything at or under this is indistinguishable from
/// having rounded the same real number twice. The cosine floor is the shape
/// check the GDN microtests use for the same reason — a merge that lost a
/// split entirely would keep a small RMS on a well-conditioned row but move
/// the direction.
pub const REL_RMS_TOL: f64 = 2e-3;
pub const COSINE_FLOOR: f64 = 0.999_99;

pub fn passes(s: &Score) -> bool {
    s.rel_rms <= REL_RMS_TOL && s.cosine >= COSINE_FLOOR
}

/// ★ THE CONTROL. A tolerance nothing can fail is not a gate, and this one is
/// loose enough to deserve the question. Corrupt the reference the way a
/// genuinely wrong split would — drop the contribution of one split's worth of
/// KV by zeroing one head's output — and require [`passes`] to REFUSE it.
///
/// Zeroing a whole head rather than jittering bits because that is the failure
/// mode the code under test actually has: an off-by-one in `pd_split_bounds`
/// or a reduce that skips a non-empty split loses a contiguous range, it does
/// not add noise.
pub fn known_bad(reference: &[u8]) -> Vec<u8> {
    let mut bad = reference.to_vec();
    let head_bytes = HD * 2;
    for b in bad.iter_mut().take(head_bytes) {
        *b = 0;
    }
    bad
}

/// Bytes of KV a launch reads: `n * L * num_kv_heads * head_dim * elem`, K and
/// V both. The denominator of every GB/s figure this microtest prints, and the
/// same model `ATTN-DECODE-SPLITK-ATTRIBUTION.md` §C.3 uses, so the two are
/// comparable without a conversion.
pub fn kv_bytes(n: usize, l: usize, elem: usize) -> f64 {
    (2 * n * l * NKV * HD * elem) as f64
}

pub fn gbs(bytes: f64, seconds: f64) -> f64 {
    if seconds > 0.0 {
        bytes / seconds / 1e9
    } else {
        0.0
    }
}

/// `block_tables[seq][logical] = physical`, with every sequence on its own
/// pages.
///
/// Distinct pages DELIBERATELY: sharing one sequence's pages across the batch
/// would let the L2 serve fifteen of sixteen rows and inflate every GB/s figure
/// at n=16, which is the arm the C=16 prediction rests on.
pub fn block_table(n: usize) -> Vec<i32> {
    let mut table = vec![0i32; n * BLOCKS_PER_SEQ];
    for (seq, chunk) in table.chunks_mut(BLOCKS_PER_SEQ).enumerate() {
        for (logical, slot) in chunk.iter_mut().enumerate() {
            *slot = (seq * BLOCKS_PER_SEQ + logical) as i32;
        }
    }
    table
}

/// One BF16 query row per sequence: `[n, num_q_heads * head_dim]`.
pub fn queries(n: usize, seed: u64) -> Vec<u8> {
    let mut rng = Lcg(seed);
    let mut out = Vec::with_capacity(n * NQ * HD * 2);
    for _ in 0..(n * NQ * HD) {
        out.extend_from_slice(&bf16::from_f32(rng.next_f32()).to_bits().to_le_bytes());
    }
    out
}

/// An FP8 E4M3 KV pool, byte-valued so the harness needs no converter: E4M3
/// bytes 0x38..0x48 span roughly ±1 with an exponent spread, which is the
/// range a calibrated FP8 KV cache actually holds.
pub fn fp8_pool(seed: u64) -> Vec<u8> {
    let mut rng = Lcg(seed);
    (0..TOTAL_BLOCKS * BLOCK * NKV * HD)
        .map(|_| {
            let v = rng.next_f32();
            // E4M3: sign | 4-bit exponent | 3-bit mantissa. Bias the exponent
            // around 0 so values land in [-2, 2).
            let sign = if v < 0.0 { 0x80u8 } else { 0 };
            let mag = ((v.abs() * 56.0) as u8).min(0x3f);
            sign | 0x20 | (mag >> 1)
        })
        .collect()
}

/// A BF16 KV pool of the same shape.
pub fn bf16_pool(seed: u64) -> Vec<u8> {
    let mut rng = Lcg(seed);
    let mut out = Vec::with_capacity(TOTAL_BLOCKS * BLOCK * NKV * HD * 2);
    for _ in 0..(TOTAL_BLOCKS * BLOCK * NKV * HD) {
        out.extend_from_slice(&bf16::from_f32(rng.next_f32()).to_bits().to_le_bytes());
    }
    out
}
