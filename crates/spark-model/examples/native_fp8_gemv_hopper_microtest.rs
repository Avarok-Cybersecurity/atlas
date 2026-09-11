// SPDX-License-Identifier: AGPL-3.0-only
//! GPU oracle + receipt for the HOPPER-TUNED W8A16 M=1 decode GEMV (#928).
//!
//! WHY. nsys, 1xH100 SXM5, Qwen/Qwen3.8-27B-FP8 (native FP8), 2026-09-11
//! round 10, C=1 steady-state decode step 18.5 ms, GPU idle 5%:
//! `w8a16_gemv` is 224 launches = 7.39 ms (40% of the step) and
//! `w8a16_gemv_dual` is 64 launches = 5.79 ms (31%). Together they move
//! 24.3 GB of FP8 weights per token in 13.2 ms = **1.84 TB/s** against HBM3's
//! 3.35 TB/s peak. `kernels/hopper/common/w8a16_gemv_hopper.cuh` diagnoses the
//! two reasons that is not the roofline — a shared-memory E4M3 LUT gather that
//! saturates the SM load/store unit, and one outstanding weight load per warp
//! at the small-N shapes the `ceil(N/4)` host grid cannot fill — and replaces
//! both. Target: >= 2.6 TB/s aggregate on these shapes.
//!
//! WHAT THIS PINS.
//!  * **Bit-identity as a hard `unequal=0`.** The arm under test is whatever
//!    `w8a16_gemv` the TARGET resolves — the Hopper override on an H100, the
//!    gb10 kernel everywhere else. The reference arm is `w8a16_gemv_splitk`
//!    launched with `splits=1`, which is the gb10 LUT kernel's per-lane chain
//!    and is already pinned bit-identical to the gb10 `w8a16_gemv` by
//!    `native_fp8_ffn_down_gemv_microtest`. So `unequal=0` here says the
//!    override changed the instruction selection and NOTHING about the
//!    arithmetic or its order. The dual is checked the same way, one
//!    projection at a time.
//!  * **The extent.** Every output buffer carries 64 sentinel bytes either
//!    side, re-stamped before each route, so a short or long write is a
//!    failure rather than a silent pass.
//!  * **The timings** the override exists for, at all six real decode shapes
//!    plus the dual: sync'd `Instant` over 20 reps, us and GB/s of FP8 weight
//!    bytes. The reference arm writes `[1, N]` FP32 partials and runs a second
//!    tiny reduce kernel; at these shapes that is <= 0.08% of the bytes, but
//!    it is why the reference column is a floor on the old kernel's time and
//!    not the old kernel's time exactly.
//!
//! TIMING METHOD: `synchronize` + host `Instant` over `REPS`, the house
//! pattern (`examples/native_fp8_ffn_down_gemv_microtest.rs`). `GpuBackend`
//! exposes event record and synchronize but no elapsed-time query.
//!
//! Run on the H100 (no env changes — the override is selected by target):
//!     cargo run --release --example native_fp8_gemv_hopper_microtest \
//!       --features cuda,gpu-examples
use anyhow::{Result, ensure};
use spark_model::layers::ops;
use spark_model::layers::ops::w8a16_decode_gemv::{GEMV_K_PER_CHUNK, GEMV_LANES_PER_OUT};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use std::time::Instant;

/// The six shapes `w8a16_gemv` runs per decode step on Qwen3.8-27B, with the
/// launches per step the round-10 nsys table counted.
const SHAPES: &[(&str, u32, u32, u32)] = &[
    ("ffn down", 5120, 17408, 64),
    ("ssm in_proj_qkvz", 16384, 5120, 48),
    ("ssm out_proj", 5120, 6144, 48),
    ("attn q", 12288, 5120, 16),
    ("attn k/v", 1024, 5120, 32),
    ("attn o", 5120, 6144, 16),
];
/// gate+up, fused into one launch by `w8a16_gemv_dual` (64 launches/step).
const DUAL: (u32, u32) = (17408, 5120);
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
        half::bf16::from_f32(((self.next() % 2049) as f32 - 1024.0) / 1024.0)
            .to_bits()
            .to_le_bytes()
    }
    /// An E4M3 byte from the 0x00..0x7E / 0x80..0xFE alphabet — the one a
    /// block-scaled FP8 checkpoint can contain. 0x7F/0xFF are the format's only
    /// NaNs and are excluded, as the batch4 oracle excludes them: they are the
    /// single value where the LUT (+-0) and `cvt` (NaN) paths differ.
    fn e4m3(&mut self) -> u8 {
        let x = self.next();
        ((x % 127) as u8) | (((x >> 7) & 1) as u8) << 7
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

/// The split-K plan that reproduces the scalar kernel exactly: ONE split over
/// every chunk iteration. This is the reference configuration.
fn single_split(k: u32) -> ops::SplitKPlan {
    ops::SplitKPlan {
        splits: 1,
        iters_per_split: (k / GEMV_K_PER_CHUNK).div_ceil(GEMV_LANES_PER_OUT),
    }
}

struct Kernels {
    gemv: KernelHandle,
    dual: KernelHandle,
    splitk: KernelHandle,
    reduce: KernelHandle,
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

/// One projection's device-side inputs, allocated once and reused by every route.
struct Case {
    n: u32,
    k: u32,
    weight: DevicePtr,
    scale: DevicePtr,
    input: DevicePtr,
    partials: DevicePtr,
}

impl Case {
    fn build(gpu: &dyn GpuBackend, rng: &mut Rng, n: u32, k: u32) -> Result<Self> {
        let (nn, kk) = (n as usize, k as usize);
        let weights: Vec<u8> = (0..nn * kk).map(|_| rng.e4m3()).collect();
        let scales: Vec<u8> = (0..nn.div_ceil(128) * kk.div_ceil(128))
            .flat_map(|_| (((rng.next() % 16 + 1) as f32) / 1024.0).to_le_bytes())
            .collect();
        let input: Vec<u8> = (0..kk).flat_map(|_| rng.act()).collect();
        Ok(Self {
            n,
            k,
            weight: upload(gpu, &weights)?,
            scale: upload(gpu, &scales)?,
            input: upload(gpu, &input)?,
            partials: upload(gpu, &vec![0_u8; ops::splitk_partial_bytes(n)])?,
        })
    }
}

/// The reference arm: the gb10 per-lane chain, via `splits=1`. `input` is
/// passed rather than read off the case so the dual's two projections can be
/// referenced against the ONE activation they share.
fn reference(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    c: &Case,
    input: DevicePtr,
    out: DevicePtr,
) -> Result<()> {
    let plan = single_split(c.k);
    ops::w8a16_gemv_splitk(
        gpu, k.splitk, input, c.weight, c.scale, c.partials, c.n, c.k, plan, 0,
    )?;
    ops::w8a16_gemv_splitk_reduce(gpu, k.reduce, c.partials, out, c.n, plan.splits, 0)
}

/// One shape: bit-identity against the reference, then both timings.
fn shape(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    rng: &mut Rng,
    row: (&str, u32, u32, u32),
) -> Result<usize> {
    let (name, n, kk, launches) = row;
    let c = Case::build(gpu, rng, n, kk)?;
    let (r, t) = (Out::new(gpu, n)?, Out::new(gpu, n)?);
    r.reset(gpu)?;
    t.reset(gpu)?;
    reference(gpu, k, &c, c.input, r.ptr())?;
    ops::w8a16_gemv(gpu, k.gemv, c.input, c.weight, c.scale, t.ptr(), n, kk, 0)?;
    gpu.synchronize(0)?;
    let unequal = r
        .read(gpu)?
        .chunks_exact(2)
        .zip(t.read(gpu)?.chunks_exact(2))
        .filter(|(a, b)| a != b)
        .count();

    let ref_us = time_us(gpu, || reference(gpu, k, &c, c.input, r.ptr()))?;
    let new_us = time_us(gpu, || {
        ops::w8a16_gemv(gpu, k.gemv, c.input, c.weight, c.scale, t.ptr(), n, kk, 0)
    })?;
    println!(
        "  [{}] {name:<16} N={n:<6} K={kk:<6} grid={:<5} unequal={unequal}  \
         ref {ref_us:8.1} us {:7.0} GB/s -> new {new_us:8.1} us {:7.0} GB/s  \
         ({launches}x/step: {:.2} -> {:.2} ms)",
        if unequal == 0 { "PASS" } else { "FAIL" },
        n.div_ceil(4),
        gbs(n, kk, ref_us),
        gbs(n, kk, new_us),
        ref_us * f64::from(launches) / 1e3,
        new_us * f64::from(launches) / 1e3,
    );
    Ok(usize::from(unequal != 0))
}

/// The dual: each projection must equal a reference GEMV over its own weights.
fn dual(gpu: &dyn GpuBackend, k: &Kernels, rng: &mut Rng) -> Result<usize> {
    let (n, kk) = DUAL;
    let (gate, up) = (Case::build(gpu, rng, n, kk)?, Case::build(gpu, rng, n, kk)?);
    let outs: Vec<Out> = (0..4).map(|_| Out::new(gpu, n).unwrap()).collect();
    for o in &outs {
        o.reset(gpu)?;
    }
    // The dual shares ONE activation; give the second case the first's input.
    let a = gate.input;
    reference(gpu, k, &gate, a, outs[0].ptr())?;
    reference(gpu, k, &up, a, outs[1].ptr())?;
    ops::w8a16_gemv_dual(
        gpu,
        k.dual,
        a,
        gate.weight,
        gate.scale,
        outs[2].ptr(),
        up.weight,
        up.scale,
        outs[3].ptr(),
        n,
        kk,
        0,
    )?;
    gpu.synchronize(0)?;
    let mut unequal = 0;
    for (i, j) in [(0, 2), (1, 3)] {
        unequal += outs[i]
            .read(gpu)?
            .chunks_exact(2)
            .zip(outs[j].read(gpu)?.chunks_exact(2))
            .filter(|(x, y)| x != y)
            .count();
    }

    let ref_us = time_us(gpu, || {
        reference(gpu, k, &gate, a, outs[0].ptr())?;
        reference(gpu, k, &up, a, outs[1].ptr())
    })?;
    let new_us = time_us(gpu, || {
        ops::w8a16_gemv_dual(
            gpu,
            k.dual,
            a,
            gate.weight,
            gate.scale,
            outs[2].ptr(),
            up.weight,
            up.scale,
            outs[3].ptr(),
            n,
            kk,
            0,
        )
    })?;
    println!(
        "  [{}] {:<16} N=2x{n:<4} K={kk:<6} grid={:<5} unequal={unequal}  \
         ref {ref_us:8.1} us {:7.0} GB/s -> new {new_us:8.1} us {:7.0} GB/s  \
         (64x/step: {:.2} -> {:.2} ms)",
        if unequal == 0 { "PASS" } else { "FAIL" },
        "gate+up dual",
        n.div_ceil(4),
        gbs(2 * n, kk, ref_us),
        gbs(2 * n, kk, new_us),
        ref_us * 64.0 / 1e3,
        new_us * 64.0 / 1e3,
    );
    Ok(usize::from(unequal != 0))
}

fn main() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let kern = Kernels {
        gemv: gpu.kernel("w8a16_gemv", "w8a16_gemv")?,
        dual: gpu.kernel("w8a16_gemv_fused", "w8a16_gemv_dual")?,
        splitk: gpu.kernel("w8a16_gemv_splitk", "w8a16_gemv_splitk")?,
        reduce: gpu.kernel("w8a16_gemv_splitk", "w8a16_gemv_splitk_reduce")?,
    };
    let mut rng = Rng(0x0928_2026_5a5a_0010);
    let mut failures = 0_usize;

    println!(
        "== W8A16 M=1 decode GEMV: target kernel vs gb10 chain (splits=1) ==\n\
         == reference = w8a16_gemv_splitk(splits=1) + reduce; hard unequal=0 =="
    );
    for row in SHAPES {
        failures += shape(&gpu, &kern, &mut rng, *row)?;
    }
    failures += dual(&gpu, &kern, &mut rng)?;

    println!(
        "\n{}",
        if failures == 0 {
            "ALL SHAPES BIT-IDENTICAL"
        } else {
            "FAILURES — the override changed the arithmetic, not just the instructions"
        }
    );
    ensure!(
        failures == 0,
        "{failures} shape(s) diverged from the gb10 chain"
    );
    Ok(())
}
