// SPDX-License-Identifier: AGPL-3.0-only
//
//! Shared support for the ARM-2 Phase-K Leg-2 native-MXFP4 (E8M0) numeric gate.
//!
//! Included via `#[path]` by both `arm2_leg2_decode.rs` and
//! `arm2_leg2_prefill.rs`; each test binary uses a subset of these helpers, so
//! `dead_code` is allowed module-wide.
//!
//! Method (Mike-blessed 2026-07-09, spec §SESSION-3): the two kernel families
//! use DIFFERENT arithmetic — decode (Family A) is bf16->f32 GEMV (weights stay
//! f32); prefill (Family B) casts BOTH operands to FP8-E4M3 then MMAs. So:
//!   - Family A decode  -> host f32 GEMV reference (bf16-tol, full-range E8M0).
//!   - Family B prefill -> BIT-EXACT kernel-vs-kernel: the `_e8m0` wrapper vs the
//!     proven NVFP4 wrapper fed the SAME packed nibbles + a power-of-2-equivalent
//!     scale encoding (E4M3 encodes 2^e exactly for e in [-6,8]; both per-16
//!     subgroups of each 32-group set equal; global scale2 = 1.0). Identical
//!     dequant -> identical FP8 recast -> identical MMA -> bit-identical output.

#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use anyhow::Result;
use spark_model::layers::ops;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// Init the CUDA backend + a stream. `#[ignore]`d tests only; needs a GB10 GPU.
pub fn setup() -> Result<(AtlasCudaBackend, u64)> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let st = backend.create_stream()?;
    Ok((backend, st))
}

// ───────────────────────── deterministic PRNG ─────────────────────────
pub struct Rng(pub u64);
impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
    pub fn nibble(&mut self) -> u8 {
        (self.next_u64() & 0xF) as u8
    }
}

// ───────────────────────── bf16 / e2m1 / e8m0 / e4m3 ─────────────────────────
// E2M1 table — identical to E2M1_LUT_T (decode) and E2M1_LUT_MOE (prefill).
pub const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

// RNE f32 -> bf16 bits (matches __float2bfloat16 / Rust f32_to_bf16).
pub fn f32_to_bf16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}
pub fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

// E8M0 byte -> f32 = 2^(sb-127); sb in {0,255} -> 0.  Byte-exact to Rust
// `fp8_e8m0_to_f32` (from_bits(exp<<23)) and CUDA `mx_block_scale<true>`.
pub fn e8m0_to_f32(sb: u8) -> f32 {
    if sb == 0 || sb == 255 {
        0.0
    } else {
        f32::from_bits((sb as u32) << 23)
    }
}
// E4M3 byte encoding of exactly 2^e (mant=0 normal), e in [-6,8].
pub fn e4m3_pow2_byte(e: i32) -> u8 {
    assert!((-6..=8).contains(&e), "e {e} outside E4M3 exact-pow range");
    (((e + 7) as u8) & 0x0F) << 3
}

// ───────────────────────── device upload / download ─────────────────────────
pub fn up_u8(g: &dyn GpuBackend, v: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(v.len().max(1))?;
    if !v.is_empty() {
        g.copy_h2d(v, p)?;
    }
    Ok(p)
}
pub fn up_u16(g: &dyn GpuBackend, v: &[u16]) -> Result<DevicePtr> {
    up_u8(
        g,
        &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
    )
}
pub fn up_u32(g: &dyn GpuBackend, v: &[u32]) -> Result<DevicePtr> {
    up_u8(
        g,
        &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
    )
}
pub fn up_i32(g: &dyn GpuBackend, v: &[i32]) -> Result<DevicePtr> {
    up_u8(
        g,
        &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
    )
}
pub fn up_f32(g: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    up_u8(
        g,
        &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
    )
}
pub fn up_u64(g: &dyn GpuBackend, v: &[u64]) -> Result<DevicePtr> {
    up_u8(
        g,
        &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
    )
}
pub fn rd_u16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
    let mut raw = vec![0u8; n * 2];
    g.copy_d2h(p, &mut raw)?;
    Ok(raw
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

// bit-identical compare of two bf16 buffers. Returns (all_equal, n_diff, first_idx).
pub fn cmp_bits(a: &[u16], b: &[u16]) -> (bool, usize, isize) {
    let mut n = 0usize;
    let mut first = -1isize;
    for i in 0..a.len() {
        if a[i] != b[i] {
            n += 1;
            if first < 0 {
                first = i as isize;
            }
        }
    }
    (n == 0, n, first)
}
// host-ref tolerance compare: PASS if every element within <=1 bf16 ULP.
// Returns (pass, exact_matches, max_ulp, worst_idx).
pub fn cmp_tol(kern: &[u16], href: &[u16]) -> (bool, usize, u32, usize) {
    let mut exact = 0usize;
    let mut max_ulp = 0u32;
    let mut worst = 0usize;
    for i in 0..kern.len() {
        if kern[i] == href[i] {
            exact += 1;
            continue;
        }
        // ULP distance in bf16 (both finite, same-ish magnitude expected).
        let ulp = (kern[i] as i32 - href[i] as i32).unsigned_abs();
        if ulp > max_ulp {
            max_ulp = ulp;
            worst = i;
        }
    }
    (max_ulp <= 1, exact, max_ulp, worst)
}

// ───────────────────────── weight generators ─────────────────────────
// Transposed weight [K/2, N] packed + scale. `t = true` for `_t` entries
// (scale [K/GS, N]); `t = false` for the non-transposed entry #1 (packed
// [N, K/2], scale [N, K/GS]). Returns (packed, scale_e8m0[K/32 groups],
// scale_nvfp4[K/16 groups], nibbles[k*N+n]).
pub struct Wt {
    pub packed: Vec<u8>,
    pub s_e8m0: Vec<u8>,  // GS=32 groups
    pub s_nvfp4: Vec<u8>, // GS=16 groups, power-of-2-paired to s_e8m0
    pub nib: Vec<u8>,     // logical [k*N + n]
}

// Generate for the bit-exact family-B check: sb restricted to [121,135]
// (e in [-6,8], E4M3-encodable), UNIQUE per group where groups <= 15.
pub fn gen_wt_bitexact(rng: &mut Rng, k: usize, n: usize, t: bool) -> Wt {
    let g32 = k / 32;
    let g16 = k / 16;
    let mut nib = vec![0u8; k * n];
    for x in nib.iter_mut() {
        *x = rng.nibble();
    }
    // Pack.
    let mut packed = vec![0u8; k / 2 * n];
    for kh in 0..k / 2 {
        for col in 0..n {
            let lo = nib[(2 * kh) * n + col];
            let hi = nib[(2 * kh + 1) * n + col];
            let byte = (lo & 0xF) | ((hi & 0xF) << 4);
            let idx = if t {
                kh * n + col // [K/2, N]
            } else {
                col * (k / 2) + kh // [N, K/2]
            };
            packed[idx] = byte;
        }
    }
    // E8M0 scale bytes: unique power-of-2 per (group,col). Vary by group first
    // (rider-1: a group-index slip must move the scale). Window sb in [121,135].
    let sb_of = |g: usize, col: usize| -> u8 {
        // UNIQUE per group at fixed col (rider-1: a K/16->K/32 group-index slip
        // MUST move the scale). (g+col)%15 -> consecutive groups always differ
        // (15 = full E4M3-exact-pow window e in [-6,8]); also varies by col.
        // Tiles sized so g32 <= 15 stay fully unique (K=64/128/448 -> 2/4/14).
        let e = -6 + (((g + col) % 15) as i32);
        (127 + e) as u8
    };
    let mut s_e8m0 = vec![0u8; g32 * n];
    for g in 0..g32 {
        for col in 0..n {
            let idx = if t { g * n + col } else { col * g32 + g };
            s_e8m0[idx] = sb_of(g, col);
        }
    }
    // NVFP4 per-16 scale: E4M3 encoding of the SAME 2^e as the covering
    // e8m0 per-32 group (g16 -> g32 = g16/2). scale2 = 1.0 at launch.
    let mut s_nvfp4 = vec![0u8; g16 * n];
    for g in 0..g16 {
        for col in 0..n {
            let sb = sb_of(g / 2, col);
            let e = sb as i32 - 127;
            let idx = if t { g * n + col } else { col * g16 + g };
            s_nvfp4[idx] = e4m3_pow2_byte(e);
        }
    }
    Wt {
        packed,
        s_e8m0,
        s_nvfp4,
        nib,
    }
}

// Generate for the decode host-ref check: FULL-RANGE E8M0 (e in [-14,14],
// sb in [113,141]) — exercises mx_block_scale beyond E4M3's range. Transposed
// [K/2,N] / [K/32,N] (decode layout). s_nvfp4 unused.
pub fn gen_wt_fullrange(rng: &mut Rng, k: usize, n: usize) -> Wt {
    let g32 = k / 32;
    let mut nib = vec![0u8; k * n];
    for x in nib.iter_mut() {
        *x = rng.nibble();
    }
    let mut packed = vec![0u8; k / 2 * n];
    for kh in 0..k / 2 {
        for col in 0..n {
            let lo = nib[(2 * kh) * n + col];
            let hi = nib[(2 * kh + 1) * n + col];
            packed[kh * n + col] = (lo & 0xF) | ((hi & 0xF) << 4);
        }
    }
    let mut s_e8m0 = vec![0u8; g32 * n];
    for g in 0..g32 {
        for col in 0..n {
            let e = -14 + (((g * 5 + col * 3) % 29) as i32); // e in [-14,14]
            s_e8m0[g * n + col] = (127 + e) as u8;
        }
    }
    Wt {
        packed,
        s_e8m0,
        s_nvfp4: vec![],
        nib,
    }
}

// ───────────────────── decode kernel launch (hand-rolled) ─────────────────────
pub fn launch_decode_gate_up(
    g: &dyn GpuBackend,
    kern: KernelHandle,
    a: DevicePtr,
    gate_packed_ptrs: DevicePtr,
    gate_scale_ptrs: DevicePtr,
    gate_s2: DevicePtr,
    gate_out: DevicePtr,
    up_packed_ptrs: DevicePtr,
    up_scale_ptrs: DevicePtr,
    up_s2: DevicePtr,
    up_out: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate_p: DevicePtr,
    sh_gate_s: DevicePtr,
    sh_gate_s2: f32,
    sh_gate_out: DevicePtr,
    sh_up_p: DevicePtr,
    sh_up_s: DevicePtr,
    sh_up_s2: f32,
    sh_up_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    let bx = n.div_ceil(32);
    KernelLaunch::new(g, kern)
        .grid([bx, top_k + 1, 2])
        .block([32, 1, 1])
        .arg_ptr(a)
        .arg_ptr(gate_packed_ptrs)
        .arg_ptr(gate_scale_ptrs)
        .arg_ptr(gate_s2)
        .arg_ptr(gate_out)
        .arg_ptr(up_packed_ptrs)
        .arg_ptr(up_scale_ptrs)
        .arg_ptr(up_s2)
        .arg_ptr(up_out)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate_p)
        .arg_ptr(sh_gate_s)
        .arg_f32(sh_gate_s2)
        .arg_ptr(sh_gate_out)
        .arg_ptr(sh_up_p)
        .arg_ptr(sh_up_s)
        .arg_f32(sh_up_s2)
        .arg_ptr(sh_up_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}

// host f32 GEMV reference for a routed transposed weight [K/2,N] / scale [K/32,N].
pub fn host_gemv(a_bf16: &[u16], w: &Wt, k: usize, n: usize) -> Vec<u16> {
    let g32 = k / 32;
    let mut out = vec![0u16; n];
    for col in 0..n {
        let mut acc = 0.0f32;
        for kk in 0..k {
            let a = bf16_bits_to_f32(a_bf16[kk]);
            let nib = w.nib[kk * n + col] as usize;
            let sb = w.s_e8m0[(kk / 32).min(g32 - 1) * n + col];
            acc += a * E2M1[nib] * e8m0_to_f32(sb);
        }
        out[col] = f32_to_bf16_bits(acc);
    }
    out
}

// ───────────── Family B launchers (production ops, handle-param) ─────────────
#[derive(Clone, Copy)]
pub enum GOp {
    Ptr64,
    PtrN128,
    PtrK64N128,
}
#[derive(Clone, Copy)]
pub enum FOp {
    FusedN128,
    FusedK64N128,
}

pub fn launch_grouped(
    g: &dyn GpuBackend,
    op: GOp,
    kern: KernelHandle,
    a: DevicePtr,
    bp: DevicePtr,
    bs: DevicePtr,
    s2: DevicePtr,
    c: DevicePtr,
    off: DevicePtr,
    sti: DevicePtr,
    ne: u32,
    n: u32,
    k: u32,
    mt: u32,
    st: u64,
) -> Result<()> {
    match op {
        GOp::Ptr64 => ops::moe_w4a16_grouped_gemm_ptrtable(
            g, kern, a, bp, bs, s2, c, off, sti, ne, n, k, mt, st,
        ),
        GOp::PtrN128 => ops::moe_w4a16_grouped_gemm_ptrtable_n128(
            g, kern, a, bp, bs, s2, c, off, sti, ne, n, k, mt, st,
        ),
        GOp::PtrK64N128 => ops::moe_w4a16_grouped_gemm_ptrtable_k64_n128(
            g, kern, a, bp, bs, s2, c, off, sti, ne, n, k, mt, st,
        ),
    }
}
pub fn launch_fused(
    g: &dyn GpuBackend,
    op: FOp,
    kern: KernelHandle,
    a: DevicePtr,
    gp: DevicePtr,
    gs: DevicePtr,
    gs2: DevicePtr,
    upp: DevicePtr,
    ups: DevicePtr,
    ups2: DevicePtr,
    cg: DevicePtr,
    cu: DevicePtr,
    off: DevicePtr,
    sti: DevicePtr,
    ne: u32,
    n: u32,
    k: u32,
    mt: u32,
    st: u64,
) -> Result<()> {
    match op {
        FOp::FusedN128 => ops::moe_w4a16_fused_gate_up_n128(
            g, kern, a, gp, gs, gs2, upp, ups, ups2, cg, cu, off, sti, ne, n, k, mt, st,
        ),
        FOp::FusedK64N128 => ops::moe_w4a16_fused_gate_up_k64_n128(
            g, kern, a, gp, gs, gs2, upp, ups, ups2, cg, cu, off, sti, ne, n, k, mt, st,
        ),
    }
}
