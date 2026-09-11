// SPDX-License-Identifier: AGPL-3.0-only
//! GPU oracle + receipt for the native-FP8 M=1 decode DOWN projection (#928).
//!
//! WHY. nsys, 1xH100, Qwen/Qwen3.8-27B-FP8, 2026-09-11 round 7, C=1
//! steady-state decode step 21.891 ms (GPU busy 96%): the fused
//! `w8a16_gemv_silu_input` (down, N=5120 K=17408, grid 1280) is 64 launches x
//! 103.9 us = 6.65 ms/step = 30.4% of the step at **858 GB/s**, while
//! `w8a16_gemv_dual` (gate+up, N=17408x2 K=5120, grid 4352) moves the SAME
//! 89.1 MB of FP8 weights per layer at **1,979 GB/s** and `w8a16_gemv` on
//! N=16384 K=5120 (grid 4096) at 1,852. The attention k/v projections
//! (N=1024 K=5120, grid 256) sit at 861 GB/s. Diagnosis, both causes and the
//! split plan: `layers::dense_ffn::fp8_down` and `layers::ops::w8a16_decode_gemv`.
//!
//! WHAT THIS PINS.
//!  * **Bit-identity, as a hard `unequal=0`**: `w8a16_gemv_splitk` launched
//!    with `splits=1` must produce the SAME BF16 BYTES as `w8a16_gemv`. The
//!    split kernel's per-lane chains are byte-identical runs of the scalar
//!    kernel's operands and the BF16 round still happens once, in the combine
//!    — so at one split there is nothing left to differ, and anything but 0 is
//!    a bug in the bounds, the partial layout or the reduce.
//!  * **The documented deltas** at `splits>1` (FP32 reassociation of at most
//!    `SPLITK_MAX` addends) and for the split-SiLU default vs the fused kernel
//!    (a BF16 round of the activation plus reciprocal-vs-divide), reported as
//!    max abs and max BF16 ULP rather than asserted to zero.
//!  * **The timings** the two changes exist for: old vs new, at the real
//!    shapes, sync'd `Instant` over 20 reps, us and GB/s of FP8 weight bytes.
//!
//! TIMING METHOD: `synchronize` + host `Instant` over `REPS`, the house
//! pattern (`examples/native_fp8_ffn_batch16_microtest.rs`). `GpuBackend`
//! exposes event record and synchronize but no elapsed-time query, so a CUDA
//! event delta is not available through the abstraction.
//!
//! Run on the H100:
//!     cargo run --release --example native_fp8_ffn_down_gemv_microtest \
//!       --features cuda,gpu-examples
use anyhow::{Result, ensure};
use half::bf16;
use spark_model::layers::ops;
use spark_model::layers::ops::w8a16_decode_gemv::{GEMV_K_PER_CHUNK, GEMV_LANES_PER_OUT};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use std::time::Instant;

/// Qwen3.8-27B: hidden 5120, intermediate 17408, 4 kv heads x head_dim 256.
const H: u32 = 5120;
const INTER: u32 = 17408;
const KV_N: u32 = 1024;
const GUARD: usize = 64;
const REPS: u32 = 20;
const WARMUP: u32 = 3;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 32) as u32
    }
    /// BF16 in roughly [-1, 1) — the range post-norm decode activations live in.
    fn act(&mut self) -> [u8; 2] {
        bf16::from_f32(((self.next() % 2049) as f32 - 1024.0) / 1024.0)
            .to_bits()
            .to_le_bytes()
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

/// Sync'd wall clock over `REPS`, minus a warmup. Returns microseconds per rep.
fn time_us(gpu: &dyn GpuBackend, mut run: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..WARMUP {
        run()?;
    }
    gpu.synchronize(0)?;
    let t0 = Instant::now();
    for _ in 0..REPS {
        run()?;
    }
    gpu.synchronize(0)?;
    Ok(t0.elapsed().as_secs_f64() * 1e6 / f64::from(REPS))
}

/// FP8 weight bytes per pass / seconds — the only budget that matters at M=1.
fn gbs(n: u32, k: u32, us: f64) -> f64 {
    (u64::from(n) * u64::from(k)) as f64 / (us * 1e-6) / 1e9
}

/// Max absolute difference and max BF16 ULP distance between two BF16 buffers.
fn compare(expected: &[u8], actual: &[u8]) -> (usize, f32, u32) {
    let mut unequal = 0;
    let mut max_abs = 0.0_f32;
    let mut max_ulp = 0_u32;
    for (e, a) in expected.chunks_exact(2).zip(actual.chunks_exact(2)) {
        let eb = u16::from_le_bytes([e[0], e[1]]);
        let ab = u16::from_le_bytes([a[0], a[1]]);
        if eb == ab {
            continue;
        }
        unequal += 1;
        let (ef, af) = (bf16::from_bits(eb).to_f32(), bf16::from_bits(ab).to_f32());
        max_abs = max_abs.max((ef - af).abs());
        // Monotone-ordinal ULP: map the sign-magnitude bits onto a signed line.
        let ord = |b: u16| -> i32 {
            if b & 0x8000 != 0 {
                -((b & 0x7FFF) as i32)
            } else {
                b as i32
            }
        };
        max_ulp = max_ulp.max((ord(eb) - ord(ab)).unsigned_abs());
    }
    (unequal, max_abs, max_ulp)
}

/// The split-K plan that reproduces the scalar kernel exactly: ONE split
/// covering every chunk iteration. This is the oracle configuration.
fn single_split(k: u32) -> ops::SplitKPlan {
    ops::SplitKPlan {
        splits: 1,
        iters_per_split: (k / GEMV_K_PER_CHUNK).div_ceil(GEMV_LANES_PER_OUT),
    }
}

struct Kernels {
    gemv: KernelHandle,
    silu_input: KernelHandle,
    silu_mul: KernelHandle,
    splitk: KernelHandle,
    reduce: KernelHandle,
}

/// One projection's device-side inputs, allocated once and reused by every route.
struct Case {
    name: &'static str,
    n: u32,
    k: u32,
    weight: DevicePtr,
    scale: DevicePtr,
    /// `[K]` BF16 — the gate vector for the FFN case, the plain activation for k/v.
    gate: DevicePtr,
    /// `[K]` BF16 up vector. Built for every case; the k/v routes ignore it
    /// (there is no SwiGLU on a projection input).
    up: DevicePtr,
    /// `[K]` BF16 staging buffer for `silu(gate)*up`.
    act: DevicePtr,
    /// `[SPLITK_MAX, N]` FP32 split-K partials.
    partials: DevicePtr,
}

impl Case {
    fn build(
        gpu: &dyn GpuBackend,
        rng: &mut Rng,
        name: &'static str,
        n: u32,
        k: u32,
    ) -> Result<Self> {
        let (nn, kk) = (n as usize, k as usize);
        // E4M3 byte draws skip 0x7F/0xFF (NaN), as the batch4 oracle does.
        let weights: Vec<u8> = (0..nn * kk)
            .map(|_| {
                let x = rng.next();
                ((x % 127) as u8) | (((x >> 7) & 1) as u8 * 128)
            })
            .collect();
        let scales: Vec<u8> = (0..(nn / 128) * (kk / 128))
            .flat_map(|_| (((rng.next() % 16 + 1) as f32) / 1024.0).to_le_bytes())
            .collect();
        let gate: Vec<u8> = (0..kk).flat_map(|_| rng.act()).collect();
        let up: Vec<u8> = (0..kk).flat_map(|_| rng.act()).collect();
        Ok(Self {
            name,
            n,
            k,
            weight: upload(gpu, &weights)?,
            scale: upload(gpu, &scales)?,
            gate: upload(gpu, &gate)?,
            up: upload(gpu, &up)?,
            act: upload(gpu, &vec![0_u8; kk * 2])?,
            partials: upload(gpu, &vec![0_u8; ops::splitk_partial_bytes(n)])?,
        })
    }
}

/// A guarded `[1, N]` BF16 output: sentinel bytes either side, re-stamped
/// before every route so a short or long write is visible.
struct Out {
    base: DevicePtr,
    sentinel: Vec<u8>,
    bytes: usize,
}

impl Out {
    fn new(gpu: &dyn GpuBackend, n: u32) -> Result<Self> {
        let bytes = n as usize * 2;
        let sentinel = vec![0x5a_u8; bytes + 2 * GUARD];
        Ok(Self {
            base: upload(gpu, &sentinel)?,
            sentinel,
            bytes,
        })
    }
    fn ptr(&self) -> DevicePtr {
        self.base.offset(GUARD)
    }
    fn reset(&self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.copy_h2d(&self.sentinel, self.base)
    }
    /// Read back the payload, refusing anything that trampled a guard.
    fn read(&self, gpu: &dyn GpuBackend) -> Result<Vec<u8>> {
        let mut host = vec![0_u8; self.sentinel.len()];
        gpu.copy_d2h(self.base, &mut host)?;
        for (i, b) in host.iter().enumerate() {
            let in_payload = (GUARD..GUARD + self.bytes).contains(&i);
            ensure!(
                in_payload || *b == 0x5a,
                "route wrote outside its extent at byte {i}"
            );
        }
        Ok(host[GUARD..GUARD + self.bytes].to_vec())
    }
}

/// `silu_mul` + the plain scalar GEMV — the #928 default arm.
fn split_silu_route(gpu: &dyn GpuBackend, kern: &Kernels, c: &Case, out: DevicePtr) -> Result<()> {
    ops::silu_mul(gpu, kern.silu_mul, c.gate, c.up, c.act, c.k, 0)?;
    ops::w8a16_gemv(gpu, kern.gemv, c.act, c.weight, c.scale, out, c.n, c.k, 0)
}

/// `silu_mul` + the split-K GEMV pair — the `ATLAS_FFN_DOWN_SPLITK` arm.
fn split_silu_splitk_route(
    gpu: &dyn GpuBackend,
    kern: &Kernels,
    c: &Case,
    out: DevicePtr,
    plan: ops::SplitKPlan,
) -> Result<()> {
    ops::silu_mul(gpu, kern.silu_mul, c.gate, c.up, c.act, c.k, 0)?;
    splitk_route(gpu, kern, c, c.act, out, plan)
}

/// The split-K GEMV pair over an already-staged activation.
fn splitk_route(
    gpu: &dyn GpuBackend,
    kern: &Kernels,
    c: &Case,
    input: DevicePtr,
    out: DevicePtr,
    plan: ops::SplitKPlan,
) -> Result<()> {
    ops::w8a16_gemv_splitk(
        gpu,
        kern.splitk,
        input,
        c.weight,
        c.scale,
        c.partials,
        c.n,
        c.k,
        plan,
        0,
    )?;
    ops::w8a16_gemv_splitk_reduce(gpu, kern.reduce, c.partials, out, c.n, plan.splits, 0)
}

/// splits=1 must reproduce `w8a16_gemv`'s bytes exactly. Returns the failure count.
fn bit_identity_gate(gpu: &dyn GpuBackend, kern: &Kernels, c: &Case) -> Result<usize> {
    let (base, split) = (Out::new(gpu, c.n)?, Out::new(gpu, c.n)?);
    base.reset(gpu)?;
    split.reset(gpu)?;
    ops::w8a16_gemv(
        gpu,
        kern.gemv,
        c.gate,
        c.weight,
        c.scale,
        base.ptr(),
        c.n,
        c.k,
        0,
    )?;
    splitk_route(gpu, kern, c, c.gate, split.ptr(), single_split(c.k))?;
    gpu.synchronize(0)?;
    let (unequal, max_abs, _) = compare(&base.read(gpu)?, &split.read(gpu)?);
    let verdict = if unequal == 0 { "PASS" } else { "FAIL" };
    println!(
        "  [{verdict}] {:<8} splits=1 vs w8a16_gemv: unequal={unequal} max_abs={max_abs:.9}",
        c.name
    );
    Ok(usize::from(unequal != 0))
}

fn main() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let kern = Kernels {
        gemv: gpu.kernel("w8a16_gemv", "w8a16_gemv")?,
        silu_input: gpu.kernel("w8a16_gemv_fused", "w8a16_gemv_silu_input")?,
        silu_mul: gpu.kernel("moe_silu_mul", "moe_silu_mul")?,
        splitk: gpu.kernel("w8a16_gemv_splitk", "w8a16_gemv_splitk")?,
        reduce: gpu.kernel("w8a16_gemv_splitk", "w8a16_gemv_splitk_reduce")?,
    };
    let mut rng = Rng(0x0928_2026_5a5a_0001);
    let mut failures = 0_usize;

    let down = Case::build(&gpu, &mut rng, "down", H, INTER)?;
    let kv = Case::build(&gpu, &mut rng, "k/v", KV_N, 5120)?;

    // ── 1. Bit-identity: splits=1 IS the scalar kernel ──
    println!("== split-K structural oracle (splits=1, hard unequal=0) ==");
    for c in [&down, &kv] {
        failures += bit_identity_gate(&gpu, &kern, c)?;
    }

    // ── 2. Down projection: the three arms, numerics then time ──
    println!(
        "\n== down N={H} K={INTER} (grid {}, 89.1 MB FP8/layer) ==",
        H.div_ceil(4)
    );
    let plan = ops::splitk_plan(H, INTER);
    println!(
        "   plan: splits={} iters_per_split={} -> grid.z {} x grid.x {} = {} CTAs",
        plan.splits,
        plan.iters_per_split,
        plan.splits,
        H.div_ceil(4),
        plan.splits * H.div_ceil(4)
    );

    let fused = Out::new(&gpu, H)?;
    let staged = Out::new(&gpu, H)?;
    let split = Out::new(&gpu, H)?;
    for o in [&fused, &staged, &split] {
        o.reset(&gpu)?;
    }
    ops::w8a16_gemv_silu_input(
        &gpu,
        kern.silu_input,
        down.gate,
        down.up,
        down.weight,
        down.scale,
        fused.ptr(),
        H,
        INTER,
        0,
    )?;
    split_silu_route(&gpu, &kern, &down, staged.ptr())?;
    split_silu_splitk_route(&gpu, &kern, &down, split.ptr(), plan)?;
    gpu.synchronize(0)?;
    let (fused_b, staged_b, split_b) = (fused.read(&gpu)?, staged.read(&gpu)?, split.read(&gpu)?);

    // Documented, NOT asserted to zero: `moe_silu_mul` rounds the activation
    // to BF16 and uses g*(1/(1+e^-g))*u where the fused kernel keeps
    // (g/(1+e^-g))*u in FP32 straight into the dot product.
    let (u1, a1, ulp1) = compare(&fused_b, &staged_b);
    println!("   split-SiLU vs fused silu_input: unequal={u1} max_abs={a1:.6} max_ulp={ulp1}");
    // Reassociation of `splits` FP32 partials, and nothing else.
    let (u2, a2, ulp2) = compare(&staged_b, &split_b);
    println!("   split-K   vs split-SiLU scalar: unequal={u2} max_abs={a2:.6} max_ulp={ulp2}");

    let t_fused = time_us(&gpu, || {
        ops::w8a16_gemv_silu_input(
            &gpu,
            kern.silu_input,
            down.gate,
            down.up,
            down.weight,
            down.scale,
            fused.ptr(),
            H,
            INTER,
            0,
        )
    })?;
    let t_staged = time_us(&gpu, || split_silu_route(&gpu, &kern, &down, staged.ptr()))?;
    let t_split = time_us(&gpu, || {
        split_silu_splitk_route(&gpu, &kern, &down, split.ptr(), plan)
    })?;
    println!(
        "   OLD fused silu_input         {t_fused:8.1} us  {:7.0} GB/s  (nsys: 103.9 us / 858 GB/s)",
        gbs(H, INTER, t_fused)
    );
    println!(
        "   NEW silu_mul + w8a16_gemv    {t_staged:8.1} us  {:7.0} GB/s  {:.2}x",
        gbs(H, INTER, t_staged),
        t_fused / t_staged
    );
    println!(
        "   NEW + ATLAS_FFN_DOWN_SPLITK  {t_split:8.1} us  {:7.0} GB/s  {:.2}x",
        gbs(H, INTER, t_split),
        t_fused / t_split
    );
    println!(
        "   target >= 1,700 GB/s (~52 us): staged {} splitk {}",
        if gbs(H, INTER, t_staged) >= 1700.0 {
            "MET"
        } else {
            "miss"
        },
        if gbs(H, INTER, t_split) >= 1700.0 {
            "MET"
        } else {
            "miss"
        }
    );

    // ── 3. Attention k/v: grid 256 is the other starved shape ──
    println!("\n== k/v N={KV_N} K=5120 (grid {}) ==", KV_N.div_ceil(4));
    let kv_plan = ops::splitk_plan(KV_N, 5120);
    println!(
        "   plan: splits={} iters_per_split={} -> {} CTAs",
        kv_plan.splits,
        kv_plan.iters_per_split,
        kv_plan.splits * KV_N.div_ceil(4)
    );
    let kv_base = Out::new(&gpu, KV_N)?;
    let kv_split = Out::new(&gpu, KV_N)?;
    kv_base.reset(&gpu)?;
    kv_split.reset(&gpu)?;
    ops::w8a16_gemv(
        &gpu,
        kern.gemv,
        kv.gate,
        kv.weight,
        kv.scale,
        kv_base.ptr(),
        KV_N,
        5120,
        0,
    )?;
    splitk_route(&gpu, &kern, &kv, kv.gate, kv_split.ptr(), kv_plan)?;
    gpu.synchronize(0)?;
    let (u3, a3, ulp3) = compare(&kv_base.read(&gpu)?, &kv_split.read(&gpu)?);
    println!("   split-K vs w8a16_gemv: unequal={u3} max_abs={a3:.6} max_ulp={ulp3}");
    let t_kv = time_us(&gpu, || {
        ops::w8a16_gemv(
            &gpu,
            kern.gemv,
            kv.gate,
            kv.weight,
            kv.scale,
            kv_base.ptr(),
            KV_N,
            5120,
            0,
        )
    })?;
    let t_kv_split = time_us(&gpu, || {
        splitk_route(&gpu, &kern, &kv, kv.gate, kv_split.ptr(), kv_plan)
    })?;
    println!(
        "   OLD w8a16_gemv               {t_kv:8.1} us  {:7.0} GB/s  (nsys: 861 GB/s)",
        gbs(KV_N, 5120, t_kv)
    );
    println!(
        "   NEW split-K                  {t_kv_split:8.1} us  {:7.0} GB/s  {:.2}x  (target >= 1,500 GB/s)",
        gbs(KV_N, 5120, t_kv_split),
        t_kv / t_kv_split
    );

    ensure!(
        failures == 0,
        "{failures} bit-identity gate(s) failed: split-K at splits=1 is not the scalar kernel"
    );
    println!("\nALL PASS: w8a16_gemv_splitk at splits=1 is byte-identical to w8a16_gemv");
    Ok(())
}
