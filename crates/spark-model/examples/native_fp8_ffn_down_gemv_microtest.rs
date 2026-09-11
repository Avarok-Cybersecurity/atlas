// SPDX-License-Identifier: AGPL-3.0-only
//! GPU oracle + receipt for the native-FP8 M=1 decode DOWN projection (#928).
//!
//! WHY. nsys, 1xH100, Qwen/Qwen3.8-27B-FP8, 2026-09-11 round 7, C=1
//! steady-state decode step 21.891 ms (GPU busy 96%): the fused
//! `w8a16_gemv_silu_input` (down, N=5120 K=17408, grid 1280) is 64 launches x
//! 103.9 us = 6.65 ms/step = 30.4% of the step at **858 GB/s**, while
//! `w8a16_gemv_dual` (gate+up, N=17408x2 K=5120, grid 4352) moves the SAME
//! 89.1 MB of FP8 weights per layer at **1,979 GB/s** and `w8a16_gemv` on
//! N=16384 K=5120 (grid 4096) at 1,852. Diagnosis and the arm rule:
//! `layers::dense_ffn::fp8_down`.
//!
//! WHAT THIS PINS.
//!  * **Bit-identity, as a hard `unequal=0`**: production stages the SwiGLU
//!    IN PLACE — `ops::silu_mul(gate_out, up_out, gate_out)` — so `gate` and
//!    `output` are the same buffer on a kernel whose parameters are both
//!    `__restrict__`. `moe_silu_mul` is one thread per element, reading its
//!    own index before writing it, so that aliasing must produce the same
//!    BYTES as staging into a separate buffer, and the down GEMV that
//!    consumes it must too. Anything but 0 means the default decode arm is
//!    reading a value it already overwrote.
//!  * **The documented delta** for the split-SiLU default vs the fused kernel
//!    (a BF16 round of the activation plus reciprocal-vs-divide), reported as
//!    max abs and max BF16 ULP rather than asserted to zero.
//!  * **The timings** the change exists for: old vs new, at the real shape,
//!    sync'd `Instant` over 20 reps, us and GB/s of FP8 weight bytes.
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
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use std::time::Instant;

/// Qwen3.8-27B: hidden 5120, intermediate 17408.
const H: u32 = 5120;
const INTER: u32 = 17408;
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

struct Kernels {
    gemv: KernelHandle,
    silu_input: KernelHandle,
    silu_mul: KernelHandle,
}

/// One projection's device-side inputs, allocated once and reused by every route.
struct Case {
    name: &'static str,
    n: u32,
    k: u32,
    weight: DevicePtr,
    scale: DevicePtr,
    /// `[K]` BF16 gate vector, and the host bytes it was uploaded from — the
    /// in-place gate needs restoring before every rep that overwrites it.
    gate: DevicePtr,
    gate_host: Vec<u8>,
    /// `[K]` BF16 up vector.
    up: DevicePtr,
    /// `[K]` BF16 staging buffer for `silu(gate)*up`, written out-of-place.
    act: DevicePtr,
    /// `[K]` BF16 scratch that plays the production `gate_out`: the SwiGLU is
    /// staged over it IN PLACE, exactly as `DenseFfnLayer::forward` does.
    inplace: DevicePtr,
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
            gate_host: gate,
            up: upload(gpu, &up)?,
            act: upload(gpu, &vec![0_u8; kk * 2])?,
            inplace: upload(gpu, &vec![0_u8; kk * 2])?,
        })
    }

    /// Restore the production-shaped `gate_out` scratch, which the in-place
    /// route destroys.
    fn reload_inplace(&self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.copy_h2d(&self.gate_host, self.inplace)
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

/// `silu_mul` into a separate buffer, then the plain scalar GEMV.
fn staged_route(gpu: &dyn GpuBackend, kern: &Kernels, c: &Case, out: DevicePtr) -> Result<()> {
    ops::silu_mul(gpu, kern.silu_mul, c.gate, c.up, c.act, c.k, 0)?;
    ops::w8a16_gemv(gpu, kern.gemv, c.act, c.weight, c.scale, out, c.n, c.k, 0)
}

/// The PRODUCTION arm: `silu_mul` staged IN PLACE over `gate_out`, then the
/// plain scalar GEMV over that same buffer.
fn inplace_route(gpu: &dyn GpuBackend, kern: &Kernels, c: &Case, out: DevicePtr) -> Result<()> {
    c.reload_inplace(gpu)?;
    ops::silu_mul(gpu, kern.silu_mul, c.inplace, c.up, c.inplace, c.k, 0)?;
    ops::w8a16_gemv(
        gpu, kern.gemv, c.inplace, c.weight, c.scale, out, c.n, c.k, 0,
    )
}

/// The in-place SwiGLU staging production runs must be byte-identical to the
/// out-of-place one. Returns the failure count.
fn bit_identity_gate(gpu: &dyn GpuBackend, kern: &Kernels, c: &Case) -> Result<usize> {
    let (staged, inplace) = (Out::new(gpu, c.n)?, Out::new(gpu, c.n)?);
    staged.reset(gpu)?;
    inplace.reset(gpu)?;
    staged_route(gpu, kern, c, staged.ptr())?;
    inplace_route(gpu, kern, c, inplace.ptr())?;
    gpu.synchronize(0)?;
    // The staged activation itself, then the projection that consumed it.
    let mut act_host = vec![0_u8; c.k as usize * 2];
    let mut inplace_host = vec![0_u8; c.k as usize * 2];
    gpu.copy_d2h(c.act, &mut act_host)?;
    gpu.copy_d2h(c.inplace, &mut inplace_host)?;
    let (act_unequal, ..) = compare(&act_host, &inplace_host);
    let (unequal, max_abs, _) = compare(&staged.read(gpu)?, &inplace.read(gpu)?);
    let failed = act_unequal != 0 || unequal != 0;
    let verdict = if failed { "FAIL" } else { "PASS" };
    println!(
        "  [{verdict}] {:<8} in-place vs out-of-place SwiGLU staging: \
         activation unequal={act_unequal}, down unequal={unequal} max_abs={max_abs:.9}",
        c.name
    );
    Ok(usize::from(failed))
}

fn main() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let kern = Kernels {
        gemv: gpu.kernel("w8a16_gemv", "w8a16_gemv")?,
        silu_input: gpu.kernel("w8a16_gemv_fused", "w8a16_gemv_silu_input")?,
        silu_mul: gpu.kernel("moe_silu_mul", "moe_silu_mul")?,
    };
    let mut rng = Rng(0x0928_2026_5a5a_0001);
    let mut failures = 0_usize;

    let down = Case::build(&gpu, &mut rng, "down", H, INTER)?;

    // ── 1. Bit-identity: the production in-place staging IS the staged route ──
    println!("== split-SiLU structural oracle (in-place staging, hard unequal=0) ==");
    failures += bit_identity_gate(&gpu, &kern, &down)?;

    // ── 2. Down projection: the two arms, numerics then time ──
    println!(
        "\n== down N={H} K={INTER} (grid {}, 89.1 MB FP8/layer) ==",
        H.div_ceil(4)
    );

    let fused = Out::new(&gpu, H)?;
    let staged = Out::new(&gpu, H)?;
    for o in [&fused, &staged] {
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
    staged_route(&gpu, &kern, &down, staged.ptr())?;
    gpu.synchronize(0)?;
    let (fused_b, staged_b) = (fused.read(&gpu)?, staged.read(&gpu)?);

    // Documented, NOT asserted to zero: `moe_silu_mul` rounds the activation
    // to BF16 and uses g*(1/(1+e^-g))*u where the fused kernel keeps
    // (g/(1+e^-g))*u in FP32 straight into the dot product.
    let (u1, a1, ulp1) = compare(&fused_b, &staged_b);
    println!("   split-SiLU vs fused silu_input: unequal={u1} max_abs={a1:.6} max_ulp={ulp1}");

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
    let t_staged = time_us(&gpu, || staged_route(&gpu, &kern, &down, staged.ptr()))?;
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
        "   target >= 1,700 GB/s (~52 us): staged {}",
        if gbs(H, INTER, t_staged) >= 1700.0 {
            "MET"
        } else {
            "miss"
        }
    );

    ensure!(
        failures == 0,
        "{failures} bit-identity gate(s) failed: the in-place SwiGLU staging the \
         decode arm runs is not the out-of-place one"
    );
    println!("\nALL PASS: the in-place SwiGLU staging is byte-identical to the staged route");
    Ok(())
}
