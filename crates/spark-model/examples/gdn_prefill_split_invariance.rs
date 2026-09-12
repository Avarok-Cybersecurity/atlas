// SPDX-License-Identifier: AGPL-3.0-only

//! The numerical premise behind `spark_model::model::gdn_replay`: which GDN
//! prefill recurrence kernels are invariant to WHERE a prompt is cut.
//!
//! A warm Marconi restore replays `[snap_tok, total)` from a snapshot; the cold
//! prefill that produced the cached KV ran the same tokens inside differently
//! cut chunks. The two agree only if the recurrence gives the same answer for
//! `run([0, N))` and `run([0, S)) ; run([S, N))` with H chained — i.e. only if
//! the kernel is split-invariant.
//!
//! This example runs exactly that A/B, twice:
//!
//! * `gated_delta_rule_prefill_regresident` — the token-sequential ladder the
//!   fix routes ALL prefill through. Must be BIT-IDENTICAL. Asserted.
//! * `gdn_prefill_fla` — the 64-token chunked kernel. Its grid is anchored at
//!   the start of the pass, so a cut at `S % 64 != 0` regroups the recurrence.
//!   Reported, and asserted NON-identical: if this ever became bit-identical
//!   the fix would be unnecessary, and that is worth failing loudly for.
//!
//! Usage: cargo run --release -p spark-model --features cuda,gpu-examples \
//!          --example gdn_prefill_split_invariance -- [seq] [split] [seed]
//!
//! Defaults `seq=512 split=464` mirror the measured repro: a 490-token prompt
//! whose tail checkpoint lands at 464, which is 16 mod 64 — so the FLA chunk
//! grids of the two decompositions do not line up.

use anyhow::Result;
use spark_model::layers::ops;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

const NK: usize = 16;
const NV: usize = 32;
const KD: usize = 128;
const VD: usize = 128;
const FLA_CHUNK: usize = 64;

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32)
    }
}

fn f32_to_bf16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    ((bits.wrapping_add(0x7FFF + ((bits >> 16) & 1))) >> 16) as u16
}
fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}
fn u16s_to_le(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn f32s_to_le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn upload(gpu: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(b.len())?;
    gpu.copy_h2d(b, p)?;
    Ok(p)
}

/// Relative L2 over bf16 output rows, plus the count of differing bf16 words.
fn compare_bf16(a: &[u8], b: &[u8]) -> (f64, usize) {
    let mut num = 0f64;
    let mut den = 0f64;
    let mut ndiff = 0usize;
    for (ca, cb) in a.chunks_exact(2).zip(b.chunks_exact(2)) {
        let wa = u16::from_le_bytes([ca[0], ca[1]]);
        let wb = u16::from_le_bytes([cb[0], cb[1]]);
        if wa != wb {
            ndiff += 1;
        }
        let (x, y) = (bf16_bits_to_f32(wa) as f64, bf16_bits_to_f32(wb) as f64);
        num += (x - y) * (x - y);
        den += x * x;
    }
    (if den > 0.0 { (num / den).sqrt() } else { 0.0 }, ndiff)
}

struct Inputs {
    q: DevicePtr,
    k: DevicePtr,
    v: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    h0: Vec<f32>,
}

impl Inputs {
    /// Byte offsets of token `t` in each stream.
    fn at(&self, t: usize) -> (DevicePtr, DevicePtr, DevicePtr, DevicePtr, DevicePtr) {
        (
            self.q.offset(t * NK * KD * 2),
            self.k.offset(t * NK * KD * 2),
            self.v.offset(t * NV * VD * 2),
            self.gate.offset(t * NV * 4),
            self.beta.offset(t * NV * 4),
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn run_regresident(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    inp: &Inputs,
    h: DevicePtr,
    out: DevicePtr,
    start: usize,
    len: usize,
    stream: u64,
) -> Result<()> {
    let (q, k, v, g, b) = inp.at(start);
    ops::gdn_prefill_regresident(
        gpu,
        kernel,
        h,
        q,
        k,
        v,
        g,
        b,
        out.offset(start * NV * VD * 2),
        1,
        len as u32,
        NK as u32,
        NV as u32,
        KD as u32,
        VD as u32,
        (NK * KD) as u32,
        (NV * VD) as u32,
        NV as u32,
        stream,
    )
}

struct FlaKernels {
    recompute_wu: KernelHandle,
    chunk_delta_h: KernelHandle,
    /// The production state spine (`..._vfused`), when the PTX set has it —
    /// `gdn_prefill_fla` prefers it over the ksplit spine, so the example must
    /// hand it over or it would be measuring a path serving does not take.
    chunk_delta_h_fused: KernelHandle,
    chunk_fwd_o: KernelHandle,
}

#[allow(clippy::too_many_arguments)]
fn run_fla(
    gpu: &dyn GpuBackend,
    kern: &FlaKernels,
    scratch: DevicePtr,
    nt_max: usize,
    inp: &Inputs,
    h: DevicePtr,
    out: DevicePtr,
    start: usize,
    len: usize,
    stream: u64,
) -> Result<()> {
    let (q, k, v, g, b) = inp.at(start);
    let num_chunks = len.div_ceil(FLA_CHUNK);
    // Sub-divide exactly as `trait_prefill_recur.rs` does, but at the MAX
    // chunk count so the two decompositions share one allocation.
    let w_out = scratch;
    let u_out = w_out.offset(nt_max * NV * FLA_CHUNK * KD * 2);
    let s_out = u_out.offset(nt_max * NV * FLA_CHUNK * VD * 2);
    let uc_out = s_out.offset(nt_max * NV * KD * VD * 2);
    let gc_out = uc_out.offset(nt_max * NV * FLA_CHUNK * VD * 2);
    ops::gdn_prefill_fla(
        gpu,
        kern.recompute_wu,
        kern.chunk_delta_h,
        KernelHandle(0),
        kern.chunk_delta_h_fused,
        KernelHandle(0),
        kern.chunk_fwd_o,
        h,
        q,
        k,
        v,
        g,
        b,
        out.offset(start * NV * VD * 2),
        w_out,
        u_out,
        s_out,
        uc_out,
        gc_out,
        1,
        len as u32,
        num_chunks as u32,
        NK as u32,
        NV as u32,
        KD as u32,
        VD as u32,
        (NK * KD) as u32,
        (NV * VD) as u32,
        NV as u32,
        false,
        DevicePtr::NULL,
        DevicePtr::NULL,
        false,
        false,
        stream,
    )
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let seq: usize = a.get(1).map_or(512, |s| s.parse().unwrap());
    let split: usize = a.get(2).map_or(464, |s| s.parse().unwrap());
    let seed: u64 = a.get(3).map_or(0x6DEF, |s| {
        u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0x6DEF)
    });
    assert!(split > 0 && split < seq, "need 0 < split < seq");
    println!(
        "=== GDN prefill split invariance: seq={seq} split={split} (split%64={}) \
         nk={NK} nv={NV} kd={KD} vd={VD} seed=0x{seed:X} ===",
        split % FLA_CHUNK
    );

    // Bounded inputs: the prefill recurrence is unclamped, so large random k/v
    // would blow up identically in every kernel and hide the comparison.
    let mut rng = Rng(seed);
    let h0: Vec<f32> = (0..NV * KD * VD)
        .map(|_| rng.uniform(-0.02, 0.02))
        .collect();
    let q: Vec<u16> = (0..seq * NK * KD)
        .map(|_| f32_to_bf16_bits(rng.uniform(-0.25, 0.25)))
        .collect();
    let k: Vec<u16> = (0..seq * NK * KD)
        .map(|_| f32_to_bf16_bits(rng.uniform(-0.25, 0.25)))
        .collect();
    let v: Vec<u16> = (0..seq * NV * VD)
        .map(|_| f32_to_bf16_bits(rng.uniform(-0.25, 0.25)))
        .collect();
    let gate: Vec<f32> = (0..seq * NV).map(|_| rng.uniform(0.88, 0.97)).collect();
    let beta: Vec<f32> = (0..seq * NV).map(|_| rng.uniform(0.0, 0.5)).collect();

    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;

    let inp = Inputs {
        q: upload(gpu, &u16s_to_le(&q))?,
        k: upload(gpu, &u16s_to_le(&k))?,
        v: upload(gpu, &u16s_to_le(&v))?,
        gate: upload(gpu, &f32s_to_le(&gate))?,
        beta: upload(gpu, &f32s_to_le(&beta))?,
        h0,
    };

    let out_bytes = seq * NV * VD * 2;
    let h_full = upload(gpu, &f32s_to_le(&inp.h0))?;
    let h_split = upload(gpu, &f32s_to_le(&inp.h0))?;
    let o_full = gpu.alloc(out_bytes)?;
    let o_split = gpu.alloc(out_bytes)?;

    let mut whole = vec![0u8; out_bytes];
    let mut cut = vec![0u8; out_bytes];
    let mut failures: Vec<String> = Vec::new();

    // ── 1. token-sequential (register-resident): must be BIT-IDENTICAL ──
    let regres = gpu.kernel(
        "gated_delta_rule_regresident",
        "gated_delta_rule_prefill_regresident",
    )?;
    run_regresident(gpu, regres, &inp, h_full, o_full, 0, seq, stream)?;
    run_regresident(gpu, regres, &inp, h_split, o_split, 0, split, stream)?;
    run_regresident(
        gpu,
        regres,
        &inp,
        h_split,
        o_split,
        split,
        seq - split,
        stream,
    )?;
    gpu.synchronize(stream)?;
    gpu.copy_d2h(o_full, &mut whole)?;
    gpu.copy_d2h(o_split, &mut cut)?;
    let (rel_seq, ndiff_seq) = compare_bf16(&whole, &cut);
    println!(
        "token-sequential (regresident): differing bf16 words = {ndiff_seq} / {}, relL2 = {rel_seq:.3e}",
        out_bytes / 2
    );
    if ndiff_seq != 0 {
        failures.push(format!(
            "token-sequential recurrence is NOT split-invariant ({ndiff_seq} words differ, \
             relL2 {rel_seq:.3e}) — the warm/cold fix in model::gdn_replay rests on it being \
             bit-identical, so a warm Marconi replay can no longer reproduce a cold prefill"
        ));
    }

    // ── 2. FLA chunked: expected to DIFFER (this is why the fix exists) ──
    let fla = match (
        gpu.kernel("gated_delta_rule_fla", "gated_delta_rule_recompute_wu"),
        gpu.kernel(
            "gated_delta_rule_fla",
            "gated_delta_rule_chunk_delta_h_ksplit",
        ),
        gpu.kernel("gated_delta_rule_fla", "gated_delta_rule_chunk_fwd_o"),
    ) {
        (Ok(recompute_wu), Ok(chunk_delta_h), Ok(chunk_fwd_o)) => Some(FlaKernels {
            recompute_wu,
            chunk_delta_h,
            chunk_delta_h_fused: gpu
                .kernel(
                    "gated_delta_rule_fla",
                    "gated_delta_rule_chunk_delta_h_vfused",
                )
                .unwrap_or(KernelHandle(0)),
            chunk_fwd_o,
        }),
        _ => None,
    };
    match fla {
        None => println!(
            "FLA chunked: kernels absent in this PTX set — skipped (the invariance \
             assertion above is the load-bearing one)"
        ),
        Some(kern) => {
            let nt_max = seq.div_ceil(FLA_CHUNK) + 2;
            let scratch_bytes = nt_max * NV * FLA_CHUNK * KD * 2
                + nt_max * NV * FLA_CHUNK * VD * 2
                + nt_max * NV * KD * VD * 2
                + nt_max * NV * FLA_CHUNK * VD * 2
                + nt_max * NV * FLA_CHUNK * 4;
            let scratch = gpu.alloc(scratch_bytes)?;
            let hf = upload(gpu, &f32s_to_le(&inp.h0))?;
            let hs = upload(gpu, &f32s_to_le(&inp.h0))?;
            run_fla(
                gpu, &kern, scratch, nt_max, &inp, hf, o_full, 0, seq, stream,
            )?;
            run_fla(
                gpu, &kern, scratch, nt_max, &inp, hs, o_split, 0, split, stream,
            )?;
            run_fla(
                gpu,
                &kern,
                scratch,
                nt_max,
                &inp,
                hs,
                o_split,
                split,
                seq - split,
                stream,
            )?;
            gpu.synchronize(stream)?;
            gpu.copy_d2h(o_full, &mut whole)?;
            gpu.copy_d2h(o_split, &mut cut)?;
            let (rel_fla, ndiff_fla) = compare_bf16(&whole, &cut);
            println!(
                "FLA chunked:                    differing bf16 words = {ndiff_fla} / {}, relL2 = {rel_fla:.3e}",
                out_bytes / 2
            );
            if ndiff_fla == 0 {
                failures.push(
                    "FLA chunked recurrence is now split-invariant too — the premise of \
                     model::gdn_replay no longer holds and forcing the token-sequential \
                     ladder under prefix caching is costing throughput for nothing"
                        .to_string(),
                );
            }
            gpu.free(scratch).ok();
            gpu.free(hf).ok();
            gpu.free(hs).ok();
        }
    }

    for p in [
        inp.q, inp.k, inp.v, inp.gate, inp.beta, h_full, h_split, o_full, o_split,
    ] {
        gpu.free(p).ok();
    }

    if failures.is_empty() {
        println!("RESULT: PASS");
        Ok(())
    } else {
        for f in &failures {
            println!("RESULT: FAIL — {f}");
        }
        std::process::exit(1);
    }
}
