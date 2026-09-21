// SPDX-License-Identifier: AGPL-3.0-only

//! Bit-identity guard for the fused FP8 KV decode write.
//!
//! `fused_k_norm_rope_cache_write_fp8_kv` collapses `norm::rms_norm` ->
//! `rope::rope_forward` -> `reshape_and_cache::reshape_and_cache_flash_fp8`
//! into one launch. The tree relies on bit-identity
//! (`kernels/gb10/common/KERNEL.toml` sets `--fmad=false`), so it must
//! reproduce that chain's rounding schedule and reduction ORDER exactly — a
//! fused kernel that kept intermediates in FP32 would be *more accurate* and
//! would silently change every committed FP8-KV record.
//!
//! PROVES: the index algebra and the rounding schedule agree. The un-fused
//! model is written in ELEMENT space, the way the three `.cu` files are
//! written (whole-row norm; rotate pairs `(d, d+half_rot)`; writer walks
//! global BF16 pairs); the fused model is written in the fused kernel's
//! THREAD space (one CTA per `(token, kv_head)`, thread `t` owns BF16 pair
//! `t`). The derivations are independent, so an off-by-one in
//! `pair = (kv_head*head_dim)/2 + t`, a mis-mapped `e1 = 2t+1` near the
//! `half_rot` boundary, a dropped non-rotary passthrough, a dropped BF16
//! round or a changed reduction tree all show up as a mismatch — each one
//! negative-controlled below by `Mutation`.
//!
//! DOES NOT PROVE: that the compiled CUDA matches this model. That tie is
//! `gpu::gpu_parity_fused_fp8_kv_write`, `#[ignore]`d per repo convention.
//!
//! Transcendentals (`rsqrtf`, `cosf`, `sinf`, `pow`) use Rust's host
//! implementations, which do NOT match libdevice bit for bit. Deliberate and
//! harmless: both models call the same host function on inputs this test
//! asserts are bit-equal. The claim tested is "same inputs reach the same
//! operations in the same order", not "the host reproduces the device".

/// Round-to-nearest-even `f32` -> BF16 bits — `__float2bfloat16`.
fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let round = ((bits >> 16) & 1) + 0x7fff;
    (bits.wrapping_add(round) >> 16) as u16
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// `__shfl_xor_sync(0xFFFFFFFF, v, offset)` butterfly over a full warp, as
/// `warp_reduce_sum` in `rms_norm.cu` performs it. All 32 lanes update
/// simultaneously, so the source array must be read before it is written.
fn warp_reduce_sum(lanes: &mut [f32; 32]) {
    for offset in [16usize, 8, 4, 2, 1] {
        let src = *lanes;
        for (l, slot) in lanes.iter_mut().enumerate() {
            *slot = src[l] + src[l ^ offset];
        }
    }
}

/// `rms_norm.cu::rms_norm` steps 1 and 2 verbatim: each thread accumulates the
/// packed BF16 PAIRS it strides over, then a warp butterfly, then a cross-warp
/// butterfly over `(blockDim.x + 31) / 32` entries.
///
/// `block_dim` is the launch geometry, not a tuning knob — `ops::rms_norm`
/// launches `block = min(hidden_size, 1024)`, and the fused kernel's contract
/// is `block = head_dim`. Passing anything else changes the answer.
fn sum_sq_block(row: &[u16], block_dim: usize) -> f32 {
    assert!(block_dim.is_multiple_of(32), "full warps only");
    let half = row.len() / 2;
    let mut per_thread = vec![0.0f32; block_dim];
    for (t, slot) in per_thread.iter_mut().enumerate() {
        let mut s = 0.0f32;
        let mut i = t;
        while i < half {
            let v0 = bf16_to_f32(row[2 * i]);
            let v1 = bf16_to_f32(row[2 * i + 1]);
            s += v0 * v0 + v1 * v1;
            i += block_dim;
        }
        *slot = s;
    }
    let num_warps = block_dim / 32;
    let mut warp_sums = [0.0f32; 32];
    for (w, slot) in warp_sums.iter_mut().enumerate().take(num_warps) {
        let mut lanes = [0.0f32; 32];
        lanes.copy_from_slice(&per_thread[w * 32..w * 32 + 32]);
        warp_reduce_sum(&mut lanes);
        *slot = lanes[0];
    }
    let mut lanes = [0.0f32; 32];
    for (l, slot) in lanes.iter_mut().enumerate() {
        *slot = if l < num_warps { warp_sums[l] } else { 0.0 };
    }
    warp_reduce_sum(&mut lanes);
    lanes[0]
}

fn rope_freq(pair_idx: usize, rotary_dim: usize, theta: f32) -> f32 {
    (1.0f64 / (theta as f64).powf((2 * pair_idx) as f64 / rotary_dim as f64)) as f32
}

/// One quantized cache element: where it lands and the exact `f32` handed to
/// the FP8 cast. Comparing the PRE-cast bits is strictly stronger than
/// comparing the FP8 byte — E4M3 is lossy and would hide a small divergence —
/// and identical (index, pre-cast bits) implies identical cache bytes for any
/// deterministic conversion.
#[derive(Debug, PartialEq, Eq, Clone)]
struct CacheWrite {
    index: usize,
    scaled_bits: u32,
}

struct Shape {
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    block_size: usize,
    slot: usize,
    cache_stride: usize,
    eps: f32,
    theta: f32,
    k_scale: f32,
    v_scale: f32,
    pos: u32,
}

impl Shape {
    fn dst_base(&self) -> usize {
        let n_elems = self.num_kv_heads * self.head_dim;
        (self.slot / self.block_size) * self.cache_stride + (self.slot % self.block_size) * n_elems
    }
}

/// The three-kernel chain, in ELEMENT space.
fn unfused(
    k_raw: &[u16],
    v_raw: &[u16],
    w: &[u16],
    sh: &Shape,
) -> (Vec<f32>, Vec<CacheWrite>, Vec<CacheWrite>) {
    let (hd, nkv) = (sh.head_dim, sh.num_kv_heads);
    let mut ssq = Vec::with_capacity(nkv);
    // ── rms_norm.cu::rms_norm, grid (nkv,1,1) block (hd) ──
    let mut normed = vec![0u16; nkv * hd];
    for head in 0..nkv {
        let row = &k_raw[head * hd..(head + 1) * hd];
        let s = sum_sq_block(row, hd);
        ssq.push(s);
        let rms = 1.0f32 / (s / hd as f32 + sh.eps).sqrt();
        for i in 0..hd / 2 {
            let x0 = bf16_to_f32(row[2 * i]);
            let x1 = bf16_to_f32(row[2 * i + 1]);
            let w0 = bf16_to_f32(w[2 * i]);
            let w1 = bf16_to_f32(w[2 * i + 1]);
            normed[head * hd + 2 * i] = f32_to_bf16(x0 * rms * (1.0 + w0));
            normed[head * hd + 2 * i + 1] = f32_to_bf16(x1 * rms * (1.0 + w1));
        }
    }
    // ── rope.cu::rope_forward — rotate (d, d+half_rot), d < rotary_dim/2 ──
    let mut roped = normed.clone();
    let half_rot = sh.rotary_dim / 2;
    for head in 0..nkv {
        for d in 0..half_rot {
            let freq = rope_freq(d, sh.rotary_dim, sh.theta);
            let angle = sh.pos as f32 * freq;
            let (c, s) = (angle.cos(), angle.sin());
            let x0 = bf16_to_f32(normed[head * hd + d]);
            let x1 = bf16_to_f32(normed[head * hd + d + half_rot]);
            roped[head * hd + d] = f32_to_bf16(x0 * c - x1 * s);
            roped[head * hd + d + half_rot] = f32_to_bf16(x1 * c + x0 * s);
        }
    }
    // ── reshape_and_cache.cu::reshape_and_cache_flash_fp8 ──
    let inv_k = 1.0f32 / sh.k_scale;
    let inv_v = 1.0f32 / sh.v_scale;
    let base = sh.dst_base();
    let mut kw = Vec::new();
    let mut vw = Vec::new();
    for e in 0..nkv * hd {
        kw.push(CacheWrite {
            index: base + e,
            scaled_bits: (bf16_to_f32(roped[e]) * inv_k).to_bits(),
        });
        vw.push(CacheWrite {
            index: base + e,
            scaled_bits: (bf16_to_f32(v_raw[e]) * inv_v).to_bits(),
        });
    }
    (ssq, kw, vw)
}

/// Ways the fusion can silently go wrong. Each is a negative control: the
/// parity assertion MUST fail with it applied, and `mutations_are_detected`
/// below proves it does. A control that fails by not compiling is not a
/// control, so these are runtime switches over the same model.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mutation {
    None,
    /// Keep the normalized value in FP32 instead of rounding to BF16 where
    /// `rms_norm`'s store rounds it — the "more accurate fused kernel" trap.
    SkipNormRound,
    /// Keep the rotated value in FP32 instead of rounding where
    /// `rope_forward`'s store rounds it.
    SkipRopeRound,
    /// One element per thread in the reduction instead of one packed PAIR —
    /// a different summation tree, hence a different `rms`.
    PerElementReduction,
    /// Forget that a head slice starts at BF16 pair `(kv_head*head_dim)/2`.
    HeadPairOffsetBug,
    /// Rotate the ADJACENT pair `(2t, 2t+1)` — the GPT-J interleaved
    /// convention — instead of `rope_forward`'s rotate-half `(d, d+half_rot)`.
    /// The natural slip when a kernel is rewritten from element space into a
    /// thread space where `t` already owns an adjacent pair.
    InterleavedRopePairs,
}

impl Mutation {
    fn name(self) -> &'static str {
        match self {
            Mutation::None => "None",
            Mutation::SkipNormRound => "SkipNormRound",
            Mutation::SkipRopeRound => "SkipRopeRound",
            Mutation::PerElementReduction => "PerElementReduction",
            Mutation::HeadPairOffsetBug => "HeadPairOffsetBug",
            Mutation::InterleavedRopePairs => "InterleavedRopePairs",
        }
    }
}

/// The fused kernel, in THREAD space: one CTA per `(token, kv_head)`,
/// `blockDim.x == head_dim`, thread `t < head_dim/2` owns BF16 pair `t`.
fn fused(
    k_raw: &[u16],
    v_raw: &[u16],
    w: &[u16],
    sh: &Shape,
    mutation: Mutation,
) -> (Vec<f32>, Vec<CacheWrite>, Vec<CacheWrite>) {
    let (hd, nkv) = (sh.head_dim, sh.num_kv_heads);
    let half_rot = sh.rotary_dim / 2;
    let inv_k = 1.0f32 / sh.k_scale;
    let inv_v = 1.0f32 / sh.v_scale;
    let base = sh.dst_base();
    let mut ssq = Vec::with_capacity(nkv);
    let mut kw = Vec::new();
    let mut vw = Vec::new();

    for kv_head in 0..nkv {
        let row = &k_raw[kv_head * hd..(kv_head + 1) * hd];
        // Stage 1 — `rms_norm`'s reduction.
        let s = if mutation == Mutation::PerElementReduction {
            let mut per_thread = vec![0.0f32; hd];
            for (t, slot) in per_thread.iter_mut().enumerate() {
                let v = bf16_to_f32(row[t]);
                *slot = v * v;
            }
            let mut acc = [0.0f32; 32];
            let mut warp_sums = [0.0f32; 32];
            for wi in 0..hd / 32 {
                acc.copy_from_slice(&per_thread[wi * 32..wi * 32 + 32]);
                warp_reduce_sum(&mut acc);
                warp_sums[wi] = acc[0];
            }
            let mut lanes = [0.0f32; 32];
            for (l, slot) in lanes.iter_mut().enumerate() {
                *slot = if l < hd / 32 { warp_sums[l] } else { 0.0 };
            }
            warp_reduce_sum(&mut lanes);
            lanes[0]
        } else {
            sum_sq_block(row, hd)
        };
        ssq.push(s);
        let rms = 1.0f32 / (s / hd as f32 + sh.eps).sqrt();

        // Stage 2 — normalize, round where `rms_norm`'s store rounds, park in
        // shared memory as BF16 bits.
        let mut s_normed = vec![0.0f32; hd];
        for i in 0..hd / 2 {
            let x0 = bf16_to_f32(row[2 * i]);
            let x1 = bf16_to_f32(row[2 * i + 1]);
            let w0 = bf16_to_f32(w[2 * i]);
            let w1 = bf16_to_f32(w[2 * i + 1]);
            let (n0, n1) = (x0 * rms * (1.0 + w0), x1 * rms * (1.0 + w1));
            if mutation == Mutation::SkipNormRound {
                s_normed[2 * i] = n0;
                s_normed[2 * i + 1] = n1;
            } else {
                s_normed[2 * i] = bf16_to_f32(f32_to_bf16(n0));
                s_normed[2 * i + 1] = bf16_to_f32(f32_to_bf16(n1));
            }
        }

        // Stage 3 — `rope_forward` on the parked row, then its BF16 store.
        let rope_elem = |e: usize| -> f32 {
            if e >= sh.rotary_dim {
                return s_normed[e];
            }
            let is_d0 = if mutation == Mutation::InterleavedRopePairs {
                e.is_multiple_of(2)
            } else {
                e < half_rot
            };
            let pair_idx = match (mutation, is_d0) {
                (Mutation::InterleavedRopePairs, _) => e / 2,
                (_, true) => e,
                (_, false) => e - half_rot,
            };
            let freq = rope_freq(pair_idx, sh.rotary_dim, sh.theta);
            let angle = sh.pos as f32 * freq;
            let (c, sn) = (angle.cos(), angle.sin());
            let (x0, x1) = if mutation == Mutation::InterleavedRopePairs {
                (s_normed[2 * pair_idx], s_normed[2 * pair_idx + 1])
            } else {
                (s_normed[pair_idx], s_normed[pair_idx + half_rot])
            };
            if is_d0 {
                x0 * c - x1 * sn
            } else {
                x1 * c + x0 * sn
            }
        };

        // Stage 4 — `reshape_and_cache_flash_fp8`'s pairing and scales.
        for t in 0..hd / 2 {
            let y0 = rope_elem(2 * t);
            let y1 = rope_elem(2 * t + 1);
            let (q0, q1) = if mutation == Mutation::SkipRopeRound {
                (y0, y1)
            } else {
                (bf16_to_f32(f32_to_bf16(y0)), bf16_to_f32(f32_to_bf16(y1)))
            };
            let pair = if mutation == Mutation::HeadPairOffsetBug {
                t
            } else {
                (kv_head * hd) / 2 + t
            };
            kw.push(CacheWrite {
                index: base + 2 * pair,
                scaled_bits: (q0 * inv_k).to_bits(),
            });
            kw.push(CacheWrite {
                index: base + 2 * pair + 1,
                scaled_bits: (q1 * inv_k).to_bits(),
            });
            let v0 = bf16_to_f32(v_raw[kv_head * hd + 2 * t]);
            let v1 = bf16_to_f32(v_raw[kv_head * hd + 2 * t + 1]);
            vw.push(CacheWrite {
                index: base + 2 * pair,
                scaled_bits: (v0 * inv_v).to_bits(),
            });
            vw.push(CacheWrite {
                index: base + 2 * pair + 1,
                scaled_bits: (v1 * inv_v).to_bits(),
            });
        }
    }
    kw.sort_by_key(|c| c.index);
    vw.sort_by_key(|c| c.index);
    (ssq, kw, vw)
}

/// xorshift — a fixture, not a default: the test needs a reproducible spread
/// of BF16 patterns, including the large-magnitude tail where the extra
/// rounding stages actually bite.
struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) as u32
    }
    fn bf16(&mut self, scale: f32) -> u16 {
        let u = (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0;
        f32_to_bf16(u * scale)
    }
}

fn fixture(sh: &Shape, seed: u64, k_scale_mag: f32) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let mut rng = Rng(seed);
    let n = sh.num_kv_heads * sh.head_dim;
    let k: Vec<u16> = (0..n).map(|_| rng.bf16(k_scale_mag)).collect();
    let v: Vec<u16> = (0..n).map(|_| rng.bf16(k_scale_mag)).collect();
    // k_norm weights are offset-from-1 (`1 + w`), so they cluster near 0.
    let w: Vec<u16> = (0..sh.head_dim).map(|_| rng.bf16(0.25)).collect();
    (k, v, w)
}

fn shapes() -> Vec<Shape> {
    vec![
        // Qwen3.6/3.8 dense flagship: hd=128, rotary_dim=128 (full rotation).
        Shape {
            num_kv_heads: 8,
            head_dim: 128,
            rotary_dim: 128,
            block_size: 16,
            slot: 16 * 37 + 5,
            cache_stride: 16 * 8 * 128,
            eps: 1e-6,
            theta: 1_000_000.0,
            k_scale: 0.011_718_75,
            v_scale: 0.009_765_625,
            pos: 30_011,
        },
        // Partial rotary: hd=256, rotary_dim=64 — exercises the passthrough
        // arm for e >= rotary_dim, which is most of the head.
        Shape {
            num_kv_heads: 4,
            head_dim: 256,
            rotary_dim: 64,
            block_size: 16,
            slot: 16 * 2 + 15,
            cache_stride: 16 * 4 * 256,
            eps: 1e-6,
            theta: 10_000_000.0,
            k_scale: 0.031_25,
            v_scale: 0.031_25,
            pos: 8_191,
        },
        // Odd head count and a head_dim that is one warp shy of the smem row.
        Shape {
            num_kv_heads: 3,
            head_dim: 224,
            rotary_dim: 112,
            block_size: 32,
            slot: 32 * 11,
            cache_stride: 32 * 3 * 224,
            eps: 1e-5,
            theta: 500_000.0,
            k_scale: 0.25,
            v_scale: 0.125,
            pos: 0,
        },
    ]
}

#[test]
fn fused_fp8_kv_write_is_bit_identical_to_the_unfused_chain() {
    for (si, sh) in shapes().iter().enumerate() {
        for (fi, mag) in [0.5f32, 8.0, 448.0].iter().enumerate() {
            let (k, v, w) = fixture(
                sh,
                0x9E37_79B9_7F4A_7C15 ^ (si as u64) << 8 ^ fi as u64,
                *mag,
            );
            let (ssq_u, kw_u, vw_u) = unfused(&k, &v, &w, sh);
            let (ssq_f, kw_f, vw_f) = fused(&k, &v, &w, sh, Mutation::None);
            // The reduction feeds `rsqrtf`; if its input differs by one ULP
            // every element of the head differs downstream.
            for (h, (a, b)) in ssq_u.iter().zip(ssq_f.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "shape {si} mag {mag}: sum-of-squares differs at kv_head {h}"
                );
            }
            assert_eq!(kw_u.len(), kw_f.len());
            assert_eq!(kw_u, kw_f, "shape {si} mag {mag}: K cache writes differ");
            assert_eq!(vw_u, vw_f, "shape {si} mag {mag}: V cache writes differ");
        }
    }
}

#[test]
fn mutations_are_detected() {
    let all = [
        Mutation::SkipNormRound,
        Mutation::SkipRopeRound,
        Mutation::PerElementReduction,
        Mutation::HeadPairOffsetBug,
        Mutation::InterleavedRopePairs,
    ];
    for m in all {
        let mut caught = false;
        for sh in shapes().iter() {
            for mag in [0.5f32, 8.0, 448.0] {
                let (k, v, w) = fixture(sh, 0xDEAD_BEEF_CAFE_0001, mag);
                let (ssq_u, kw_u, vw_u) = unfused(&k, &v, &w, sh);
                let (ssq_f, kw_f, vw_f) = fused(&k, &v, &w, sh, m);
                let ssq_differs = ssq_u
                    .iter()
                    .zip(ssq_f.iter())
                    .any(|(a, b)| a.to_bits() != b.to_bits());
                if ssq_differs || kw_u != kw_f || vw_u != vw_f {
                    caught = true;
                }
            }
        }
        assert!(
            caught,
            "negative control {} did not fire: the parity assertion accepts a \
             kernel with this defect, so it would not catch the real one",
            m.name()
        );
    }
}

// The on-GPU half of the same claim. Its own file for the 500-line budget; a
// CHILD module so it still reads `shapes` and `fixture` from here — the two
// halves must be held to the same fixtures or they stop being one claim.
#[path = "write_kv_cache_fp8_gpu_tests.rs"]
mod gpu;
