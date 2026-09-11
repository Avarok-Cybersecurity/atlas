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
//!    gb10 kernel everywhere else. The reference is [`host_gemv`], an EXACT
//!    host model of `kernels/gb10/common/w8a16_gemv.cu`'s reduction: the same
//!    64 lanes walking the same chunks in the same order, the same separate
//!    FMUL/FADD per element (`--fmad=false` is tree-wide, and Rust never
//!    contracts), the same five-step `shfl.down` butterfly, the same two-warp
//!    add, the same single `__float2bfloat16`. So `unequal=0` says the
//!    override changed the instruction selection and NOTHING about the
//!    arithmetic or its order. The dual is checked the same way, one
//!    projection at a time.
//!
//!    A HOST model rather than a second kernel, because on an H100 there is no
//!    second kernel to ask: `kernels/hopper/common/w8a16_gemv.cu` overrides
//!    the gb10 source, so a build for this target does not contain the chain
//!    it is being compared against. This arm used to borrow
//!    `w8a16_gemv_splitk` at `splits=1` as a surviving copy of that chain;
//!    that kernel was removed (#993) after the H100 microtest measured it a
//!    null (down 61.8 us split-K vs 58.9 us staged scalar), and borrowing a
//!    kernel kept alive for an unrelated lever was never the reason the oracle
//!    was sound.
//!  * **The extent.** Every output buffer carries 64 sentinel bytes either
//!    side, re-stamped before each route, so a short or long write is a
//!    failure rather than a silent pass.
//!  * **The timings** the override exists for, at all six real decode shapes
//!    plus the dual: sync'd `Instant` over 20 reps, us and GB/s of FP8 weight
//!    bytes, and the ms/step those launch counts buy. There is deliberately NO
//!    "before" column: the kernel this replaces is not in an H100 build, and a
//!    stand-in's time is not that kernel's time. The GB10 number to compare
//!    against is this same example run on a GB10 box, where `w8a16_gemv`
//!    resolves to the gb10 source.
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
    /// BF16 bits, in roughly [-1, 1) — the range post-norm decode activations
    /// live in. BITS rather than bytes so the host oracle and the uploaded
    /// buffer are the same values and not two decodings of them.
    fn act(&mut self) -> u16 {
        half::bf16::from_f32(((self.next() % 2049) as f32 - 1024.0) / 1024.0).to_bits()
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

/// K values one lane consumes per chunk — one `uint4` of FP8 bytes. SSOT:
/// `K_PER_CHUNK` in `kernels/hopper/common/w8a16_gemv_hopper.cuh`, which is
/// the gb10 kernel's `k16 * 16` stride spelled as a name.
const K_PER_CHUNK: u32 = 16;
/// Lanes cooperating on one output — `BLOCK_SIZE / N_PER_BLOCK` = 256 / 4.
const LANES_PER_OUT: u32 = 64;
/// One FP8 block-scale tile, in both N and K.
const FP8_BLOCK: u32 = 128;

/// E4M3 byte -> the FP32 value `E4M3_LUT` holds for it.
///
/// COMPUTED from the format rather than pasted as 256 literals, so the oracle
/// cannot drift from the table by a transcription slip: sign / 4 exponent bits
/// (bias 7) / 3 mantissa bits, subnormal when the exponent field is 0. Every
/// finite E4M3 is exactly representable in FP32, so this reproduces the LUT's
/// bits exactly. 0x7F/0xFF — the format's only NaNs, which the LUT maps to +-0
/// and `cvt` maps to NaN — are excluded from the drawn alphabet ([`Rng::e4m3`])
/// and so never reach here.
fn e4m3_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0_f32 } else { 1.0 };
    let exp = i32::from((byte >> 3) & 0x0F);
    let mant = f32::from(byte & 0x07);
    if exp == 0 {
        // Subnormal: mant * 2^-9 (no implicit leading 1, bias 7, 3 mantissa bits).
        sign * mant * 2.0_f32.powi(-9)
    } else {
        sign * (1.0 + mant / 8.0) * 2.0_f32.powi(exp - 7)
    }
}

/// THE ORACLE: `kernels/gb10/common/w8a16_gemv.cu`, evaluated on the host with
/// its reduction order preserved element for element.
///
/// Every step is the kernel's, in the kernel's order, because a reference that
/// merely computes the same DOT PRODUCT would pass an override that
/// reassociated the sum — which is the one thing this microtest exists to
/// refuse:
///
///  * lane `l` of 64 walks chunks `l, l+64, l+128, …`;
///  * inside a chunk, elements 0..15 in index order, each `w = LUT[b] * scale`
///    then `acc += a * w` as a separate multiply and add (`--fmad=false` is a
///    tree-wide nvcc flag; Rust does not contract either);
///  * a 5-step `__shfl_down_sync` butterfly per 32-lane warp. Out-of-range
///    source lanes return the caller's OWN value in CUDA, which is reproduced
///    here rather than skipped — it does not reach lane 0, and a model that
///    quietly disagrees about a lane is not a model;
///  * the two warps' lane-0 values added in smem order, then ONE
///    `__float2bfloat16` (round-to-nearest-even, which `bf16::from_f32` is).
///
/// Returns `[1, N]` BF16 little-endian bytes — the kernel's own output layout,
/// so the comparison is on bytes and never on a re-decoded float.
fn host_gemv(n: u32, k: u32, weight: &[u8], scale: &[f32], act: &[u16]) -> Vec<u8> {
    let chunks = k / K_PER_CHUNK;
    let k_blocks = k.div_ceil(FP8_BLOCK);
    let mut out = Vec::with_capacity(n as usize * 2);
    for row in 0..n {
        let n_block = row / FP8_BLOCK;
        let base = row as usize * k as usize;
        let mut part = [0.0_f32; 64];
        for (l, acc) in part.iter_mut().enumerate() {
            let mut chunk = l as u32;
            while chunk < chunks {
                let first = chunk * K_PER_CHUNK;
                let s = scale[(n_block * k_blocks + first / FP8_BLOCK) as usize];
                for j in 0..K_PER_CHUNK {
                    let idx = (first + j) as usize;
                    let w = e4m3_to_f32(weight[base + idx]) * s;
                    *acc += half::bf16::from_bits(act[idx]).to_f32() * w;
                }
                chunk += LANES_PER_OUT;
            }
        }
        let mut warp_sum = [0.0_f32; 2];
        for (w, sum) in warp_sum.iter_mut().enumerate() {
            let mut v = [0.0_f32; 32];
            v.copy_from_slice(&part[w * 32..w * 32 + 32]);
            for offset in [16_usize, 8, 4, 2, 1] {
                let snap = v;
                for (i, cell) in v.iter_mut().enumerate() {
                    // `__shfl_down_sync` past the warp returns the caller's own
                    // value; that is `snap[i] + snap[i]`, not a skipped add.
                    *cell = snap[i] + snap[if i + offset < 32 { i + offset } else { i }];
                }
            }
            *sum = v[0];
        }
        out.extend_from_slice(&half::bf16::from_f32(warp_sum[0] + warp_sum[1]).to_bits().to_le_bytes());
    }
    out
}

struct Kernels {
    gemv: KernelHandle,
    dual: KernelHandle,
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

/// One projection's inputs, on the device for the kernel and RETAINED on the
/// host for [`host_gemv`] — the oracle reads the same bytes the kernel does,
/// rather than a second draw that could differ.
struct Case {
    n: u32,
    k: u32,
    weight: DevicePtr,
    scale: DevicePtr,
    input: DevicePtr,
    weights: Vec<u8>,
    scales: Vec<f32>,
    act: Vec<u16>,
}

impl Case {
    fn build(gpu: &dyn GpuBackend, rng: &mut Rng, n: u32, k: u32) -> Result<Self> {
        let (nn, kk) = (n as usize, k as usize);
        let weights: Vec<u8> = (0..nn * kk).map(|_| rng.e4m3()).collect();
        let scales: Vec<f32> = (0..nn.div_ceil(128) * kk.div_ceil(128))
            .map(|_| ((rng.next() % 16 + 1) as f32) / 1024.0)
            .collect();
        let act: Vec<u16> = (0..kk).map(|_| rng.act()).collect();
        let scale_bytes: Vec<u8> = scales.iter().flat_map(|s| s.to_le_bytes()).collect();
        let input: Vec<u8> = act.iter().flat_map(|a| a.to_le_bytes()).collect();
        Ok(Self {
            n,
            k,
            weight: upload(gpu, &weights)?,
            scale: upload(gpu, &scale_bytes)?,
            input: upload(gpu, &input)?,
            weights,
            scales,
            act,
        })
    }
    /// The oracle's answer for this case, over `act` (the dual's two
    /// projections share ONE activation, so it is passed rather than read).
    fn expected(&self, act: &[u16]) -> Vec<u8> {
        host_gemv(self.n, self.k, &self.weights, &self.scales, act)
    }
}

/// One shape: bit-identity against the host oracle, then the timing.
fn shape(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    rng: &mut Rng,
    row: (&str, u32, u32, u32),
) -> Result<usize> {
    let (name, n, kk, launches) = row;
    let c = Case::build(gpu, rng, n, kk)?;
    let t = Out::new(gpu, n)?;
    t.reset(gpu)?;
    ops::w8a16_gemv(gpu, k.gemv, c.input, c.weight, c.scale, t.ptr(), n, kk, 0)?;
    gpu.synchronize(0)?;
    let unequal = c
        .expected(&c.act)
        .chunks_exact(2)
        .zip(t.read(gpu)?.chunks_exact(2))
        .filter(|(a, b)| a != b)
        .count();

    let new_us = time_us(gpu, || {
        ops::w8a16_gemv(gpu, k.gemv, c.input, c.weight, c.scale, t.ptr(), n, kk, 0)
    })?;
    println!(
        "  [{}] {name:<16} N={n:<6} K={kk:<6} grid={:<5} unequal={unequal}  \
         {new_us:8.1} us {:7.0} GB/s  ({launches}x/step: {:.2} ms)",
        if unequal == 0 { "PASS" } else { "FAIL" },
        n.div_ceil(4),
        gbs(n, kk, new_us),
        new_us * f64::from(launches) / 1e3,
    );
    Ok(usize::from(unequal != 0))
}

/// The dual: each projection must equal the oracle over its own weights and
/// the ONE activation the two share.
fn dual(gpu: &dyn GpuBackend, k: &Kernels, rng: &mut Rng) -> Result<usize> {
    let (n, kk) = DUAL;
    let (gate, up) = (Case::build(gpu, rng, n, kk)?, Case::build(gpu, rng, n, kk)?);
    let outs: Vec<Out> = (0..2).map(|_| Out::new(gpu, n).unwrap()).collect();
    for o in &outs {
        o.reset(gpu)?;
    }
    // The dual shares ONE activation; give the second case the first's.
    let (a, act) = (gate.input, &gate.act);
    let launch = || {
        ops::w8a16_gemv_dual(
            gpu,
            k.dual,
            a,
            gate.weight,
            gate.scale,
            outs[0].ptr(),
            up.weight,
            up.scale,
            outs[1].ptr(),
            n,
            kk,
            0,
        )
    };
    launch()?;
    gpu.synchronize(0)?;
    let mut unequal = 0;
    for (case, out) in [(&gate, &outs[0]), (&up, &outs[1])] {
        unequal += case
            .expected(act)
            .chunks_exact(2)
            .zip(out.read(gpu)?.chunks_exact(2))
            .filter(|(x, y)| x != y)
            .count();
    }

    let new_us = time_us(gpu, launch)?;
    println!(
        "  [{}] {:<16} N=2x{n:<4} K={kk:<6} grid={:<5} unequal={unequal}  \
         {new_us:8.1} us {:7.0} GB/s  (64x/step: {:.2} ms)",
        if unequal == 0 { "PASS" } else { "FAIL" },
        "gate+up dual",
        n.div_ceil(4),
        gbs(2 * n, kk, new_us),
        new_us * 64.0 / 1e3,
    );
    Ok(usize::from(unequal != 0))
}

fn main() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let kern = Kernels {
        gemv: gpu.kernel("w8a16_gemv", "w8a16_gemv")?,
        dual: gpu.kernel("w8a16_gemv_fused", "w8a16_gemv_dual")?,
    };
    let mut rng = Rng(0x0928_2026_5a5a_0010);
    let mut failures = 0_usize;

    println!(
        "== W8A16 M=1 decode GEMV: the target's kernel vs the gb10 chain ==\n\
         == reference = host_gemv, an exact model of that chain's reduction \
         order; hard unequal=0 =="
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
