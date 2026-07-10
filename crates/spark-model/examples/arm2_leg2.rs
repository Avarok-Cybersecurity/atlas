// SPDX-License-Identifier: AGPL-3.0-only
//
//! ARM-2 Phase-K Leg-2 — native-MXFP4 (E8M0) numeric gate.
//!
//! Method: Mike-blessed 2026-07-09 (spec §SESSION-3). The two families use
//! DIFFERENT arithmetic — decode (Family A) is bf16->f32 GEMV (weights stay
//! f32); prefill (Family B) casts BOTH operands to FP8-E4M3 then MMAs. So:
//!   - Family A decode  -> host f32 GEMV reference (bf16-tol, full-range E8M0).
//!   - Family B prefill -> BIT-EXACT kernel-vs-kernel: the `_e8m0` wrapper vs the
//!     proven NVFP4 wrapper fed the SAME packed nibbles + a power-of-2-equivalent
//!     scale encoding (E4M3 encodes 2^e exactly for e in [-6,8]; both per-16
//!     subgroups of each 32-group set equal; global scale2 = 1.0). Identical
//!     dequant -> identical FP8 recast -> identical MMA -> bit-identical output.
//!   - RIDER A3 : NVFP4 shared branch, baseline vs `_e8m0` wrapper -> bit-identical.
//!   - RIDER A4 : mixed launch (routed-E8M0 + shared-NVFP4) -> no cross-branch
//!     contamination (routed == check-1 routed, shared == check-2 shared).
//!   - RIDER 2  : two independent anchors (decode host f32 GEMV + shipping-NVFP4
//!     kernel). Rider-1: unique power-of-2 per group. Rider-3: multi-K + realistic.
//!
//! Build (deepseek-v4-flash target — carries BOTH families):
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=deepseek-v4-flash ATLAS_TARGET_QUANT=nvfp4 \
//!     cargo build --release -p spark-model --example arm2_leg2 \
//!       --no-default-features --features "cuda gpu-examples"
//! Run (n3/n4 GB10): target/release/examples/arm2_leg2

use anyhow::{Result, bail};
use spark_model::layers::ops;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

// ───────────────────────── deterministic PRNG ─────────────────────────
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
    fn nibble(&mut self) -> u8 {
        (self.next_u64() & 0xF) as u8
    }
}

// ───────────────────────── bf16 / e2m1 / e8m0 / e4m3 ─────────────────────────
// E2M1 table — identical to E2M1_LUT_T (decode) and E2M1_LUT_MOE (prefill).
const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

// RNE f32 -> bf16 bits (matches __float2bfloat16 / Rust f32_to_bf16).
fn f32_to_bf16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}
fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

// E8M0 byte -> f32 = 2^(sb-127); sb in {0,255} -> 0.  Byte-exact to Rust
// `fp8_e8m0_to_f32` (from_bits(exp<<23)) and CUDA `mx_block_scale<true>`.
fn e8m0_to_f32(sb: u8) -> f32 {
    if sb == 0 || sb == 255 {
        0.0
    } else {
        f32::from_bits((sb as u32) << 23)
    }
}
// E4M3 byte encoding of exactly 2^e (mant=0 normal), e in [-6,8].
fn e4m3_pow2_byte(e: i32) -> u8 {
    assert!((-6..=8).contains(&e), "e {e} outside E4M3 exact-pow range");
    (((e + 7) as u8) & 0x0F) << 3
}

// ───────────────────────── device upload / download ─────────────────────────
fn up_u8(g: &dyn GpuBackend, v: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(v.len().max(1))?;
    if !v.is_empty() {
        g.copy_h2d(v, p)?;
    }
    Ok(p)
}
fn up_u16(g: &dyn GpuBackend, v: &[u16]) -> Result<DevicePtr> {
    up_u8(g, &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
}
fn up_u32(g: &dyn GpuBackend, v: &[u32]) -> Result<DevicePtr> {
    up_u8(g, &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
}
fn up_i32(g: &dyn GpuBackend, v: &[i32]) -> Result<DevicePtr> {
    up_u8(g, &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
}
fn up_f32(g: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    up_u8(g, &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
}
fn up_u64(g: &dyn GpuBackend, v: &[u64]) -> Result<DevicePtr> {
    up_u8(g, &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
}
fn rd_u16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
    let mut raw = vec![0u8; n * 2];
    g.copy_d2h(p, &mut raw)?;
    Ok(raw
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

// bit-identical compare of two bf16 buffers. Returns (all_equal, n_diff, first_idx).
fn cmp_bits(a: &[u16], b: &[u16]) -> (bool, usize, isize) {
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
fn cmp_tol(kern: &[u16], href: &[u16]) -> (bool, usize, u32, usize) {
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
struct Wt {
    packed: Vec<u8>,
    s_e8m0: Vec<u8>,  // GS=32 groups
    s_nvfp4: Vec<u8>, // GS=16 groups, power-of-2-paired to s_e8m0
    nib: Vec<u8>,     // logical [k*N + n]
}

// Generate for the bit-exact family-B check: sb restricted to [121,135]
// (e in [-6,8], E4M3-encodable), UNIQUE per group where groups <= 15.
fn gen_wt_bitexact(rng: &mut Rng, k: usize, n: usize, t: bool) -> Wt {
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
fn gen_wt_fullrange(rng: &mut Rng, k: usize, n: usize) -> Wt {
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

// ───────────────────────── decode kernel launch (hand-rolled) ─────────────────────────
#[allow(clippy::too_many_arguments)]
fn launch_decode_gate_up(
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
fn host_gemv(a_bf16: &[u16], w: &Wt, k: usize, n: usize) -> Vec<u16> {
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

// ───────────────────────── Family B launchers (production ops, handle-param) ─────────────────────────
#[derive(Clone, Copy)]
enum GOp {
    Ptr64,
    PtrN128,
    PtrK64N128,
}
#[derive(Clone, Copy)]
enum FOp {
    FusedN128,
    FusedK64N128,
}

#[allow(clippy::too_many_arguments)]
fn launch_grouped(
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
        GOp::Ptr64 => ops::moe_w4a16_grouped_gemm_ptrtable(g, kern, a, bp, bs, s2, c, off, sti, ne, n, k, mt, st),
        GOp::PtrN128 => ops::moe_w4a16_grouped_gemm_ptrtable_n128(g, kern, a, bp, bs, s2, c, off, sti, ne, n, k, mt, st),
        GOp::PtrK64N128 => ops::moe_w4a16_grouped_gemm_ptrtable_k64_n128(g, kern, a, bp, bs, s2, c, off, sti, ne, n, k, mt, st),
    }
}
#[allow(clippy::too_many_arguments)]
fn launch_fused(
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
        FOp::FusedN128 => ops::moe_w4a16_fused_gate_up_n128(g, kern, a, gp, gs, gs2, upp, ups, ups2, cg, cu, off, sti, ne, n, k, mt, st),
        FOp::FusedK64N128 => ops::moe_w4a16_fused_gate_up_k64_n128(g, kern, a, gp, gs, gs2, upp, ups, ups2, cg, cu, off, sti, ne, n, k, mt, st),
    }
}

fn main() -> Result<()> {
    println!("=== ARM-2 Phase-K Leg-2 — native MXFP4 (E8M0) numeric gate ===");
    println!("method: spec §SESSION-3 (Mike-blessed). decode=host-ref f32 GEMV; prefill=bit-exact kernel-vs-NVFP4.\n");
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let st = gpu.create_stream()?;
    let mut rng = Rng(0x_ADA2_1E62_5EED_0002u64);
    let null = DevicePtr(0);
    let mut all_pass = true;

    // ══════════ resolve handles ══════════
    let dmod = "moe_shared_expert_fused_t";
    let k_dec_base = gpu.kernel(dmod, "moe_expert_gate_up_shared_t")?;
    let k_dec_e8m0 = gpu.kernel(dmod, "moe_expert_gate_up_shared_t_e8m0")?;
    let bmod = "moe_w4a16";

    // ══════════ CHECK 1 — Family A decode, e8m0 routed vs host f32 GEMV (full-range E8M0) ══════════
    {
        let (k, n, top_k) = (512usize, 256usize, 1u32);
        let a: Vec<u16> = (0..k).map(|_| f32_to_bf16_bits(rng.unit() * 2.0 - 1.0)).collect();
        let gw = gen_wt_fullrange(&mut rng, k, n);
        let uw = gen_wt_fullrange(&mut rng, k, n);
        let a_p = up_u16(gpu, &a)?;
        let gwp = up_u8(gpu, &gw.packed)?;
        let gws = up_u8(gpu, &gw.s_e8m0)?;
        let uwp = up_u8(gpu, &uw.packed)?;
        let uws = up_u8(gpu, &uw.s_e8m0)?;
        let gpt = up_u64(gpu, &[gwp.0])?;
        let gst = up_u64(gpu, &[gws.0])?;
        let upt = up_u64(gpu, &[uwp.0])?;
        let ust = up_u64(gpu, &[uws.0])?;
        let s2 = up_f32(gpu, &[1.0])?;
        let eidx = up_u32(gpu, &[0u32])?;
        let gate_out = gpu.alloc(n * 2)?;
        let up_out = gpu.alloc(n * 2)?;
        let sh_g_out = gpu.alloc(n * 2)?;
        let sh_u_out = gpu.alloc(n * 2)?;
        launch_decode_gate_up(
            gpu, k_dec_e8m0, a_p, gpt, gst, s2, gate_out, upt, ust, s2, up_out, eidx,
            null, null, 0.0, sh_g_out, null, null, 0.0, sh_u_out, n as u32, k as u32, top_k, st,
        )?;
        gpu.synchronize(st)?;
        let kg = rd_u16(gpu, gate_out, n)?;
        let ku = rd_u16(gpu, up_out, n)?;
        let hg = host_gemv(&a, &gw, k, n);
        let hu = host_gemv(&a, &uw, k, n);
        let (pg, eg, ug, _) = cmp_tol(&kg, &hg);
        let (pu, eu, uu, _) = cmp_tol(&ku, &hu);
        let pass = pg && pu;
        all_pass &= pass;
        println!(
            "CHECK 1  Family A decode e8m0 vs host f32 GEMV (K={k} N={n}, full-range E8M0):"
        );
        println!(
            "         gate: exact {eg}/{n} maxULP {ug} | up: exact {eu}/{n} maxULP {uu}  => {}",
            if pass { "PASS (<=1 ULP)" } else { "FAIL" }
        );
        for p in [a_p, gwp, gws, uwp, uws, gpt, gst, upt, ust, s2, eidx, gate_out, up_out, sh_g_out, sh_u_out] {
            gpu.free(p).ok();
        }
    }

    // ══════════ CHECK 2 — RIDER A3: NVFP4 shared branch bit-identical (baseline vs e8m0 wrapper) ══════════
    // Shared expert is NVFP4 in BOTH wrappers (<GROUP_SIZE,false>). Routed ptr
    // table = [0] so the routed slot writes 0 (no deref). Compare sh_*_out.
    {
        let (k, n, top_k) = (512usize, 256usize, 1u32);
        let a: Vec<u16> = (0..k).map(|_| f32_to_bf16_bits(rng.unit() * 2.0 - 1.0)).collect();
        // NVFP4 shared weight: nibbles + random valid E4M3 scale bytes (avoid NaN 0x7F/0xFF).
        let g16 = k / 16;
        let mk_nvfp4 = |rng: &mut Rng| -> (Vec<u8>, Vec<u8>) {
            let mut nib = vec![0u8; k * n];
            for x in nib.iter_mut() {
                *x = rng.nibble();
            }
            let mut packed = vec![0u8; k / 2 * n];
            for kh in 0..k / 2 {
                for col in 0..n {
                    packed[kh * n + col] = (nib[(2 * kh) * n + col] & 0xF) | ((nib[(2 * kh + 1) * n + col] & 0xF) << 4);
                }
            }
            let mut sc = vec![0u8; g16 * n];
            for x in sc.iter_mut() {
                let mut b = (rng.next_u64() & 0xFF) as u8;
                if b == 0x7F || b == 0xFF {
                    b = 0x38; // 1.0
                }
                *x = b;
            }
            (packed, sc)
        };
        let (sgp, sgs) = mk_nvfp4(&mut rng);
        let (sup, sus) = mk_nvfp4(&mut rng);
        let a_p = up_u16(gpu, &a)?;
        let sgp_p = up_u8(gpu, &sgp)?;
        let sgs_p = up_u8(gpu, &sgs)?;
        let sup_p = up_u8(gpu, &sup)?;
        let sus_p = up_u8(gpu, &sus)?;
        // routed ptr tables = [0] (null routed weight -> slot writes 0).
        let z_tbl = up_u64(gpu, &[0u64])?;
        let s2 = up_f32(gpu, &[1.0])?;
        let eidx = up_u32(gpu, &[0u32])?;
        let (sh_g2, sh_u2) = (0.75f32, 1.25f32);
        let run = |kern: KernelHandle| -> Result<(Vec<u16>, Vec<u16>)> {
            let gate_out = gpu.alloc(n * 2)?;
            let up_out = gpu.alloc(n * 2)?;
            let sh_g_out = gpu.alloc(n * 2)?;
            let sh_u_out = gpu.alloc(n * 2)?;
            launch_decode_gate_up(
                gpu, kern, a_p, z_tbl, z_tbl, s2, gate_out, z_tbl, z_tbl, s2, up_out, eidx,
                sgp_p, sgs_p, sh_g2, sh_g_out, sup_p, sus_p, sh_u2, sh_u_out, n as u32, k as u32, top_k, st,
            )?;
            gpu.synchronize(st)?;
            let g = rd_u16(gpu, sh_g_out, n)?;
            let u = rd_u16(gpu, sh_u_out, n)?;
            for p in [gate_out, up_out, sh_g_out, sh_u_out] {
                gpu.free(p).ok();
            }
            Ok((g, u))
        };
        let (bg, bu) = run(k_dec_base)?;
        let (eg, eu) = run(k_dec_e8m0)?;
        let (p1, d1, _) = cmp_bits(&bg, &eg);
        let (p2, d2, _) = cmp_bits(&bu, &eu);
        let pass = p1 && p2;
        all_pass &= pass;
        println!("CHECK 2  RIDER A3 NVFP4 shared branch baseline vs e8m0-wrapper (K={k} N={n}):");
        println!(
            "         sh_gate diffs {d1}/{n} | sh_up diffs {d2}/{n}  => {}",
            if pass { "PASS (bit-identical)" } else { "FAIL" }
        );
        // keep buffers for CHECK 3 (mixed) reuse via regen; free here.
        for p in [a_p, sgp_p, sgs_p, sup_p, sus_p, z_tbl, s2, eidx] {
            gpu.free(p).ok();
        }
    }

    // ══════════ CHECK 3 — RIDER A4: mixed launch (routed-E8M0 + shared-NVFP4) no cross-contamination ══════════
    {
        let (k, n, top_k) = (512usize, 256usize, 1u32);
        let a: Vec<u16> = (0..k).map(|_| f32_to_bf16_bits(rng.unit() * 2.0 - 1.0)).collect();
        let gw = gen_wt_fullrange(&mut rng, k, n); // routed E8M0
        let uw = gen_wt_fullrange(&mut rng, k, n);
        let g16 = k / 16;
        let mut snib = vec![0u8; k * n];
        for x in snib.iter_mut() {
            *x = rng.nibble();
        }
        let mut spacked = vec![0u8; k / 2 * n];
        for kh in 0..k / 2 {
            for col in 0..n {
                spacked[kh * n + col] = (snib[(2 * kh) * n + col] & 0xF) | ((snib[(2 * kh + 1) * n + col] & 0xF) << 4);
            }
        }
        let mut sscale = vec![0u8; g16 * n];
        for x in sscale.iter_mut() {
            let mut b = (rng.next_u64() & 0xFF) as u8;
            if b == 0x7F || b == 0xFF {
                b = 0x38;
            }
            *x = b;
        }
        let a_p = up_u16(gpu, &a)?;
        let gwp = up_u8(gpu, &gw.packed)?;
        let gws = up_u8(gpu, &gw.s_e8m0)?;
        let uwp = up_u8(gpu, &uw.packed)?;
        let uws = up_u8(gpu, &uw.s_e8m0)?;
        let gpt = up_u64(gpu, &[gwp.0])?;
        let gst = up_u64(gpu, &[gws.0])?;
        let upt = up_u64(gpu, &[uwp.0])?;
        let ust = up_u64(gpu, &[uws.0])?;
        let sgp_p = up_u8(gpu, &spacked)?;
        let sgs_p = up_u8(gpu, &sscale)?;
        let s2 = up_f32(gpu, &[1.0])?;
        let eidx = up_u32(gpu, &[0u32])?;
        let sh_g2 = 0.9f32;

        // (a) routed-only reference (shared null).
        let run_routed = |sh_p: DevicePtr, sh_s: DevicePtr| -> Result<(Vec<u16>, Vec<u16>)> {
            let gate_out = gpu.alloc(n * 2)?;
            let up_out = gpu.alloc(n * 2)?;
            let sh_g_out = gpu.alloc(n * 2)?;
            let sh_u_out = gpu.alloc(n * 2)?;
            launch_decode_gate_up(
                gpu, k_dec_e8m0, a_p, gpt, gst, s2, gate_out, upt, ust, s2, up_out, eidx,
                sh_p, sh_s, sh_g2, sh_g_out, sh_p, sh_s, sh_g2, sh_u_out, n as u32, k as u32, top_k, st,
            )?;
            gpu.synchronize(st)?;
            let go = rd_u16(gpu, gate_out, n)?;
            let sgo = rd_u16(gpu, sh_g_out, n)?;
            for p in [gate_out, up_out, sh_g_out, sh_u_out] {
                gpu.free(p).ok();
            }
            Ok((go, sgo))
        };
        // routed-only: shared null -> routed gate output = pure routed.
        let (routed_only_gate, _) = run_routed(null, null)?;
        // shared-only: routed table null would need null routed; reuse baseline
        // NVFP4 shared alone by nulling routed via z table.
        let ztbl = up_u64(gpu, &[0u64])?;
        let run_shared_only = || -> Result<Vec<u16>> {
            let gate_out = gpu.alloc(n * 2)?;
            let up_out = gpu.alloc(n * 2)?;
            let sh_g_out = gpu.alloc(n * 2)?;
            let sh_u_out = gpu.alloc(n * 2)?;
            launch_decode_gate_up(
                gpu, k_dec_e8m0, a_p, ztbl, ztbl, s2, gate_out, ztbl, ztbl, s2, up_out, eidx,
                sgp_p, sgs_p, sh_g2, sh_g_out, sgp_p, sgs_p, sh_g2, sh_u_out, n as u32, k as u32, top_k, st,
            )?;
            gpu.synchronize(st)?;
            let sgo = rd_u16(gpu, sh_g_out, n)?;
            for p in [gate_out, up_out, sh_g_out, sh_u_out] {
                gpu.free(p).ok();
            }
            Ok(sgo)
        };
        let shared_only_gate = run_shared_only()?;
        // (c) mixed: routed-E8M0 + shared-NVFP4 in ONE launch.
        let (mixed_routed, mixed_shared) = run_routed(sgp_p, sgs_p)?;
        let (p1, d1, _) = cmp_bits(&routed_only_gate, &mixed_routed);
        let (p2, d2, _) = cmp_bits(&shared_only_gate, &mixed_shared);
        let pass = p1 && p2;
        all_pass &= pass;
        println!("CHECK 3  RIDER A4 mixed-fusion (routed-E8M0 + shared-NVFP4) no cross-contamination (K={k} N={n}):");
        println!(
            "         routed(mixed vs alone) diffs {d1}/{n} | shared(mixed vs alone) diffs {d2}/{n}  => {}",
            if pass { "PASS (bit-identical)" } else { "FAIL" }
        );
        for p in [a_p, gwp, gws, uwp, uws, gpt, gst, upt, ust, sgp_p, sgs_p, s2, eidx, ztbl] {
            gpu.free(p).ok();
        }
    }

    // ══════════ CHECK 4 — Family B prefill, 5 entries, bit-exact e8m0 vs NVFP4-equivalent ══════════
    println!("CHECK 4  Family B prefill — 5 W4A16 entries, e8m0 vs NVFP4-equivalent (bit-exact):");
    // entry: (label, base, e8m0, transposed, fused, op). K set hits unroll boundaries + realistic.
    let grouped_entries: &[(&str, &str, &str, bool, GOp, &[usize])] = &[
        ("ptrtable(k16,non-t)", "moe_w4a16_grouped_gemm_ptrtable", "moe_w4a16_grouped_gemm_ptrtable_e8m0", false, GOp::Ptr64, &[64, 128, 448]),
        ("ptrtable_t(k32)", "moe_w4a16_grouped_gemm_ptrtable_t", "moe_w4a16_grouped_gemm_ptrtable_t_e8m0", true, GOp::PtrN128, &[64, 128, 448]),
        ("ptrtable_t_k64(down*)", "moe_w4a16_grouped_gemm_ptrtable_t_k64", "moe_w4a16_grouped_gemm_ptrtable_t_k64_e8m0", true, GOp::PtrK64N128, &[64, 128, 448]),
    ];
    let (n_b, m_b) = (256usize, 128usize);
    for (label, base, e8m0, t, op, ks) in grouped_entries.iter().copied() {
        let kbase = gpu.kernel(bmod, base)?;
        let ke8 = gpu.kernel(bmod, e8m0)?;
        let mut ok = true;
        let mut detail = String::new();
        for &k in ks {
            let w = gen_wt_bitexact(&mut rng, k, n_b, t);
            let a: Vec<u16> = (0..m_b * k).map(|_| f32_to_bf16_bits(rng.unit() * 2.0 - 1.0)).collect();
            let a_p = up_u16(gpu, &a)?;
            let wp = up_u8(gpu, &w.packed)?;
            let ws_e = up_u8(gpu, &w.s_e8m0)?;
            let ws_n = up_u8(gpu, &w.s_nvfp4)?;
            let bpt = up_u64(gpu, &[wp.0])?;
            let bst_e = up_u64(gpu, &[ws_e.0])?;
            let bst_n = up_u64(gpu, &[ws_n.0])?;
            let s2 = up_f32(gpu, &[1.0])?;
            let off = up_i32(gpu, &[0, m_b as i32])?;
            let sti: Vec<i32> = (0..m_b as i32).collect();
            let sti_p = up_i32(gpu, &sti)?;
            let mt = (m_b as u32).div_ceil(64);
            let run = |kern: KernelHandle, bst: DevicePtr| -> Result<Vec<u16>> {
                let c = gpu.alloc(m_b * n_b * 2)?;
                launch_grouped(gpu, op, kern, a_p, bpt, bst, s2, c, off, sti_p, 1, n_b as u32, k as u32, mt, st)?;
                gpu.synchronize(st)?;
                let v = rd_u16(gpu, c, m_b * n_b)?;
                gpu.free(c).ok();
                Ok(v)
            };
            let ce = run(ke8, bst_e)?;
            let cn = run(kbase, bst_n)?;
            let (p, d, _) = cmp_bits(&cn, &ce);
            ok &= p;
            detail.push_str(&format!(" K{k}:{}", if p { "ok".into() } else { format!("DIFF{d}") }));
            for pp in [a_p, wp, ws_e, ws_n, bpt, bst_e, bst_n, s2, off, sti_p] {
                gpu.free(pp).ok();
            }
        }
        all_pass &= ok;
        println!("   [{}] {} =>{}", if ok { "PASS" } else { "FAIL" }, label, detail);
    }
    // fused entries (gate+up, 2 outputs).
    let fused_entries: &[(&str, &str, &str, FOp, &[usize])] = &[
        ("fused_gate_up_t(k32)", "moe_w4a16_fused_gate_up_t", "moe_w4a16_fused_gate_up_t_e8m0", FOp::FusedN128, &[64, 128, 448]),
        ("fused_gate_up_t_k64(gate/up*)", "moe_w4a16_fused_gate_up_t_k64", "moe_w4a16_fused_gate_up_t_k64_e8m0", FOp::FusedK64N128, &[64, 128, 448]),
    ];
    for (label, base, e8m0, op, ks) in fused_entries.iter().copied() {
        let kbase = gpu.kernel(bmod, base)?;
        let ke8 = gpu.kernel(bmod, e8m0)?;
        let mut ok = true;
        let mut detail = String::new();
        for &k in ks {
            let gw = gen_wt_bitexact(&mut rng, k, n_b, true);
            let uw = gen_wt_bitexact(&mut rng, k, n_b, true);
            let a: Vec<u16> = (0..m_b * k).map(|_| f32_to_bf16_bits(rng.unit() * 2.0 - 1.0)).collect();
            let a_p = up_u16(gpu, &a)?;
            let gwp = up_u8(gpu, &gw.packed)?;
            let gse = up_u8(gpu, &gw.s_e8m0)?;
            let gsn = up_u8(gpu, &gw.s_nvfp4)?;
            let uwp = up_u8(gpu, &uw.packed)?;
            let use_ = up_u8(gpu, &uw.s_e8m0)?;
            let usn = up_u8(gpu, &uw.s_nvfp4)?;
            let gpt = up_u64(gpu, &[gwp.0])?;
            let gse_t = up_u64(gpu, &[gse.0])?;
            let gsn_t = up_u64(gpu, &[gsn.0])?;
            let upt = up_u64(gpu, &[uwp.0])?;
            let use_t = up_u64(gpu, &[use_.0])?;
            let usn_t = up_u64(gpu, &[usn.0])?;
            let s2 = up_f32(gpu, &[1.0])?;
            let off = up_i32(gpu, &[0, m_b as i32])?;
            let sti: Vec<i32> = (0..m_b as i32).collect();
            let sti_p = up_i32(gpu, &sti)?;
            let mt = (m_b as u32).div_ceil(64);
            let run = |kern: KernelHandle, gst: DevicePtr, ust: DevicePtr| -> Result<(Vec<u16>, Vec<u16>)> {
                let cg = gpu.alloc(m_b * n_b * 2)?;
                let cu = gpu.alloc(m_b * n_b * 2)?;
                launch_fused(gpu, op, kern, a_p, gpt, gst, s2, upt, ust, s2, cg, cu, off, sti_p, 1, n_b as u32, k as u32, mt, st)?;
                gpu.synchronize(st)?;
                let g = rd_u16(gpu, cg, m_b * n_b)?;
                let u = rd_u16(gpu, cu, m_b * n_b)?;
                gpu.free(cg).ok();
                gpu.free(cu).ok();
                Ok((g, u))
            };
            let (ge, ue) = run(ke8, gse_t, use_t)?;
            let (gn, un) = run(kbase, gsn_t, usn_t)?;
            let (pg, dg, _) = cmp_bits(&gn, &ge);
            let (pu, du, _) = cmp_bits(&un, &ue);
            ok &= pg && pu;
            detail.push_str(&format!(" K{k}:{}", if pg && pu { "ok".into() } else { format!("gDIFF{dg}/uDIFF{du}") }));
            for pp in [a_p, gwp, gse, gsn, uwp, use_, usn, gpt, gse_t, gsn_t, upt, use_t, usn_t, s2, off, sti_p] {
                gpu.free(pp).ok();
            }
        }
        all_pass &= ok;
        println!("   [{}] {} =>{}", if ok { "PASS" } else { "FAIL" }, label, detail);
    }

    println!(
        "\n(*) = V4-serve-path entry (fused_gate_up_t_k64 gate/up + ptrtable_t_k64 down). Others off-path, tested for RIDER-2 completeness."
    );
    println!("RIDER 2 (cross-family): two independent anchors — decode host f32 GEMV (CHECK 1) + shipping-NVFP4 kernel (CHECK 4).");
    if all_pass {
        println!("\n===== LEG-2 RESULT: PASS (all checks clean, bit-exact / <=1 ULP) =====");
        Ok(())
    } else {
        bail!("LEG-2 RESULT: FAIL — first divergence STOPs the line (see above).");
    }
}
