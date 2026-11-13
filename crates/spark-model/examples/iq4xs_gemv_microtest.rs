// SPDX-License-Identifier: AGPL-3.0-only

//! GPU oracle and timings for the keep-packed GGUF 4-bit route on Ada (port lane L19, row
//! `gguf_native_iq4`): the IQ4 / Q5_K dequant (`kernels/ada/common/dequant_iq4xs_q5k_bf16_ada.cu`), the
//! q8_1 activation quantizers (`kquant_q8_1_ada.cu`), the decode GEMVs (`kquant_mmvq_q4_ada.cu`) and the
//! MMQ prefill GEMMs (`iq4_xs_mmq_ada.cu`), through `port::gguf_q4` and its `W4Linear` face.
//!
//! WHAT THIS PINS (each gate prints PASS/FAIL, each planted control CONTROL-OK; any FAIL, or any control
//! that passes its gate, exits non-zero):
//!  * dequant: IQ4_XS, IQ4_NL and Q5_K on the GPU BIT FOR BIT equal to spark-runtime's CPU decoder (the
//!    ggml arithmetic) on random blocks; control: one block's codes flipped on the GPU side only. Q4_K and
//!    Q6_K through gb10's linked `dequant_gguf_bf16` are reported (INFO);
//!  * the decode rows quantizer's codes and f16 scales equal to the host mirror, its partial sums within
//!    two f16 units; the MMQ quantizer's codes and scales in the D4 (IQ4_XS) and DS4 (Q4_K) layouts;
//!    control: the mirror of the rows in reverse order;
//!  * per type x decode shape: the block-shape GEMV within the float64 bound
//!    (`port::gguf_q4::reference`) at M = 1, 2, 3, 4, 5, 8 (control: the last super-block of every row
//!    zeroed); row r at every M bit-equal to row r computed alone (control: row 1 against row 0); a
//!    five-run byte repeat; `decode_pair` byte-equal to two `decode` calls; the warp shape within the bound;
//!  * per type x prefill shape: the MMQ GEMM within the prefill bound at M = 9, 77, 129, 260, with the
//!    Q4_K partial-sum term mirrored (control: the zeroed last super-block).
//!
//! TIMINGS (`timing.rs`): GB/s of packed weight bytes at M = 1 and 8, block against warp shape, with the
//! R2 bar (M = 1 at 85% of 288 GB/s) and the R3 bar (M = 8 within 25% of M = 1) printed as BAR PASS /
//! BAR MISS (a miss does not fail the run).
//!
//! Run on the Ada card, in a build of the ada target:
//!     AVAROK_TARGET_HW=ada CUDARC_CUDA_VERSION=13000 AVAROK_NO_RDMA=1 cargo run --release -p spark-model \
//!       --example iq4xs_gemv_microtest --features cuda,gpu-examples
//! `--gates-only` skips the timing rows; `--quick` runs two decode shapes and one prefill shape.

#[path = "iq4xs_gemv_microtest/q8_parse.rs"]
mod q8_parse;
#[path = "iq4xs_gemv_microtest/timing.rs"]
mod timing;
#[path = "iq4xs_gemv_microtest/util.rs"]
mod util;

use anyhow::{Context, Result, ensure};
use spark_model::port::gguf_q4::launch::{self, MmvqShape};
use spark_model::port::gguf_q4::reference::{
    Q8Block, bound, dequant_weight, ggml_type, mmq_layout, prefill_bound, q4_k_mins,
    q4_k_sum_correction, q8_1_values, quantize_q8_1_mmq_row, quantize_q8_1_row,
};
use spark_model::port::gguf_q4::{Q4Type, from_gguf_tensor, gguf_of, release};
use spark_model::port::w4::W4Scratch;
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::dequant_cpu::{GgmlType, dequant_to_bf16};
use spark_runtime::weights::keep_packed_iq4;

use q8_parse::{compare_q8, parse_q8_mmq, parse_q8_rows};
use util::{
    Rng, Tally, download, download_u16, gemv_par, grade_slack, u16_bytes, unequal, upload, widen,
};

const TYPES: [Q4Type; 3] = [Q4Type::Iq4Xs, Q4Type::Q4K, Q4Type::Q6K];
/// Decode shapes N x K: the 4B FFN, the 27B FFN, and a ragged row count.
const DECODE: [(usize, usize); 5] = [
    (9216, 2560),
    (2560, 9216),
    (17408, 5120),
    (5120, 17408),
    (129, 4096),
];
const DECODE_QUICK: [(usize, usize); 2] = [(2560, 9216), (129, 4096)];
const DECODE_MS: [usize; 6] = [1, 2, 3, 4, 5, 8];
/// Prefill shapes: a tile-aligned N (the `_nc` entry) and a ragged one (`_wc`).
const PREFILL: [(usize, usize); 2] = [(2048, 2560), (129, 4096)];
const PREFILL_MS: [usize; 4] = [9, 77, 129, 260];

fn scratch(gpu: &dyn GpuBackend, bytes: usize) -> Result<W4Scratch> {
    Ok(W4Scratch {
        ws: gpu.alloc(bytes)?,
        ws_bytes: bytes,
        tickets: DevicePtr::NULL,
        xq: DevicePtr::NULL,
        xs: DevicePtr::NULL,
    })
}

/// GPU dequant against the CPU decoder, bit for bit.
fn dequant_gates(gpu: &dyn GpuBackend, rng: &mut Rng, t: &mut Tally) -> Result<()> {
    for (name, id, qk, bb, f16s) in [
        (
            "IQ4_XS",
            keep_packed_iq4::ID_IQ4_XS,
            256usize,
            136usize,
            1usize,
        ),
        ("IQ4_NL", keep_packed_iq4::ID_IQ4_NL, 32, 18, 1),
        ("Q5_K", keep_packed_iq4::ID_Q5_K, 256, 176, 2),
    ] {
        let n_blocks = 4096;
        let numel = n_blocks * qk;
        let raw = util::random_blocks(rng, n_blocks, bb, f16s);
        let mut want = vec![0u16; numel];
        dequant_to_bf16(GgmlType::from_id(id, 128)?, &raw, numel, &mut want)?;
        let ptr = keep_packed_iq4::dequant_to_bf16_gpu(gpu, id, &raw, numel)?;
        let got = download_u16(gpu, ptr, numel)?;
        gpu.free(ptr)?;
        t.gate(
            &format!("dequant {name} bitwise"),
            unequal(&got, &want),
            &format!("{n_blocks} blocks"),
        );

        let mut planted = raw.clone();
        let codes_from = if id == keep_packed_iq4::ID_Q5_K {
            48
        } else if bb == 136 {
            8
        } else {
            2
        };
        for b in &mut planted[codes_from..bb] {
            *b ^= 0xFF;
        }
        let ptr = keep_packed_iq4::dequant_to_bf16_gpu(gpu, id, &planted, numel)?;
        let got = download_u16(gpu, ptr, numel)?;
        gpu.free(ptr)?;
        t.control(
            &format!("dequant {name}, block 0 codes flipped"),
            unequal(&got, &want),
            "",
        );
    }
    for ty in [Q4Type::Q4K, Q4Type::Q6K] {
        let (n, k) = (64, 4096);
        let raw = util::random_weight(rng, ty, n, k);
        let w = from_gguf_tensor(gpu, ggml_type(ty), &raw, n, k, 0)?;
        let out = gpu.alloc(n * k * 2)?;
        w.dequant_bf16(gpu, out, 0)?;
        let got = download_u16(gpu, out, n * k)?;
        let mut want = vec![0u16; n * k];
        dequant_to_bf16(ggml_type(ty), &raw, n * k, &mut want)?;
        t.info(
            &format!("dequant {ty:?} (gb10 dequant_gguf_bf16)"),
            &format!(
                "{} of {} values differ from the CPU decoder",
                unequal(&got, &want),
                n * k
            ),
        );
        gpu.free(out)?;
        release(gpu, &*w)?;
    }
    Ok(())
}

fn quantizer_gates(gpu: &dyn GpuBackend, rng: &mut Rng, t: &mut Tally) -> Result<()> {
    let k = 2560;
    let xb = rng.bf16_bits(9 * k);
    let x = upload(gpu, &u16_bytes(&xb))?;
    let mirror = |m: usize| -> Vec<Vec<Q8Block>> {
        (0..m)
            .map(|r| quantize_q8_1_row(&xb[r * k..(r + 1) * k]))
            .collect()
    };

    // Decode rows: an IQ4_XS weight's decode leaves the q8_1 rows in `ws`.
    let raw = util::random_weight(rng, Q4Type::Iq4Xs, 16, k);
    let w = from_gguf_tensor(gpu, GgmlType::Iq4Xs, &raw, 16, k, 0)?;
    let s = scratch(gpu, w.scratch_bytes(260))?;
    let out = gpu.alloc(260 * 16 * 2)?;
    w.decode(gpu, x, 8, out, &s, 0)?;
    let got = parse_q8_rows(&download(gpu, s.ws, launch::q8_rows_bytes(8, k))?, 8, k);
    let want = mirror(8);
    let (bad, bad_s) = compare_q8(&got, &want, true);
    t.gate("q8_1 rows: codes and f16 scales", bad, "8 rows x 80 blocks");
    t.gate("q8_1 rows: partial sums within two f16 units", bad_s, "");
    let reversed: Vec<Vec<Q8Block>> = want.iter().rev().cloned().collect();
    t.control(
        "q8_1 rows against the rows in reverse order",
        compare_q8(&got, &reversed, false).0,
        "",
    );

    // MMQ layouts: a prefill leaves the activations in `ws`.
    for (ty, ds4) in [(Q4Type::Iq4Xs, false), (Q4Type::Q4K, true)] {
        let raw = util::random_weight(rng, ty, 16, k);
        let w2 = from_gguf_tensor(gpu, ggml_type(ty), &raw, 16, k, 0)?;
        ensure!(
            w2.prefill(gpu, x, 9, out, &s, 0)?,
            "the prefill declined with a full workspace"
        );
        let got = parse_q8_mmq(
            &download(gpu, s.ws, launch::mmq_act_bytes(9, k))?,
            9,
            k,
            ds4,
        );
        let want: Vec<Vec<Q8Block>> = (0..9)
            .map(|r| quantize_q8_1_mmq_row(&xb[r * k..(r + 1) * k], mmq_layout(ty)))
            .collect();
        let (bad, bad_s) = compare_q8(&got, &want, ds4);
        t.gate(
            &format!("q8_1 MMQ {:?} layout: codes and scales", mmq_layout(ty)),
            bad,
            "9 rows",
        );
        if ds4 {
            t.gate("q8_1 MMQ DS4 partial sums within two f16 units", bad_s, "");
        }
        release(gpu, &*w2)?;
    }
    release(gpu, &*w)?;
    for p in [x, out, s.ws] {
        gpu.free(p)?;
    }
    Ok(())
}

/// Every decode gate for one type and shape.
fn decode_shape(
    gpu: &dyn GpuBackend,
    rng: &mut Rng,
    ty: Q4Type,
    n: usize,
    k: usize,
    threads: usize,
    t: &mut Tally,
) -> Result<()> {
    let tag = format!("{ty:?} N={n} K={k}");
    let raw = util::random_weight(rng, ty, n, k);
    let xb = rng.bf16_bits(8 * k);
    let xq: Vec<f32> = xb
        .chunks_exact(k)
        .flat_map(|r| q8_1_values(&quantize_q8_1_row(r)))
        .collect();
    let wf = dequant_weight(ty, &raw, n, k)?;
    let (want, mag) = gemv_par(&wf, n, k, &xq, 8, threads);
    drop(wf);

    let w = from_gguf_tensor(gpu, ggml_type(ty), &raw, n, k, 0)?;
    let planted = from_gguf_tensor(
        gpu,
        ggml_type(ty),
        &util::drop_last_block(&raw, ty, n, k),
        n,
        k,
        0,
    )?;
    let s = scratch(gpu, w.scratch_bytes(8))?;
    let x = upload(gpu, &u16_bytes(&xb))?;
    let (out, out2) = (gpu.alloc(8 * n * 2)?, gpu.alloc(8 * n * 2)?);

    // Each row alone: the reference bits of the row-invariance gate.
    let mut alone = Vec::with_capacity(8 * n);
    for r in 0..8 {
        w.decode(gpu, x.offset(r * k * 2), 1, out, &s, 0)?;
        alone.extend(download_u16(gpu, out, n)?);
    }
    let (mut over, mut worst, mut variant) = (0usize, 0f64, 0usize);
    for m in DECODE_MS {
        w.decode(gpu, x, m, out, &s, 0)?;
        let got = download_u16(gpu, out, m * n)?;
        let (o, wst) = grade_slack(&widen(&got), &want[..m * n], &mag[..m * n], None, k, bound);
        over += o;
        worst = worst.max(wst);
        variant += unequal(&got, &alone[..m * n]);
    }
    t.gate(
        &format!("{tag} GEMV within the bound, M = 1..8"),
        over,
        &format!("worst {worst:.3} of the bound"),
    );
    t.gate(
        &format!("{tag} row r at every M == row r alone"),
        variant,
        "",
    );
    t.control(
        &format!("{tag} row 1 against row 0"),
        unequal(&alone[n..2 * n], &alone[..n]),
        "",
    );

    w.decode(gpu, x, 8, out, &s, 0)?;
    let first = download_u16(gpu, out, 8 * n)?;
    let mut repeat = 0;
    for _ in 0..4 {
        w.decode(gpu, x, 8, out, &s, 0)?;
        repeat += unequal(&download_u16(gpu, out, 8 * n)?, &first);
    }
    t.gate(&format!("{tag} five-run byte repeat at M = 8"), repeat, "");

    planted.decode(gpu, x, 8, out, &s, 0)?;
    let got = download_u16(gpu, out, 8 * n)?;
    let (o, _) = grade_slack(&widen(&got), &want, &mag, None, k, bound);
    t.control(&format!("{tag} last super-block zeroed"), o, "");
    let planted_alone = got;

    w.decode_pair(&*planted, gpu, x, 8, out, out2, &s, 0)?;
    let pair = unequal(&download_u16(gpu, out, 8 * n)?, &first)
        + unequal(&download_u16(gpu, out2, 8 * n)?, &planted_alone);
    t.gate(&format!("{tag} decode_pair == two decodes"), pair, "");

    let d = gguf_of(&*w).context("a live GGUF weight")?;
    launch::q8_rows(gpu, &d.kernels, x, s.ws, 8, k, 0)?;
    launch::mmvq(
        gpu,
        &d.kernels,
        MmvqShape::Warp,
        d.blocks,
        s.ws,
        out,
        n,
        k,
        8,
        0,
    )?;
    let (o, wst) = grade_slack(
        &widen(&download_u16(gpu, out, 8 * n)?),
        &want,
        &mag,
        None,
        k,
        bound,
    );
    t.gate(
        &format!("{tag} warp shape within the bound"),
        o,
        &format!("worst {wst:.3}"),
    );

    release(gpu, &*w)?;
    release(gpu, &*planted)?;
    for p in [s.ws, x, out, out2] {
        gpu.free(p)?;
    }
    Ok(())
}

/// The MMQ gates for one type and shape.
fn prefill_shape(
    gpu: &dyn GpuBackend,
    rng: &mut Rng,
    ty: Q4Type,
    n: usize,
    k: usize,
    threads: usize,
    t: &mut Tally,
) -> Result<()> {
    let tag = format!("{ty:?} N={n} K={k}");
    let mmax = PREFILL_MS[PREFILL_MS.len() - 1];
    let raw = util::random_weight(rng, ty, n, k);
    let w = from_gguf_tensor(gpu, ggml_type(ty), &raw, n, k, 0)?;
    let planted = from_gguf_tensor(
        gpu,
        ggml_type(ty),
        &util::drop_last_block(&raw, ty, n, k),
        n,
        k,
        0,
    )?;
    let wf = dequant_weight(ty, &raw, n, k)?;
    let mins = (ty == Q4Type::Q4K).then(|| q4_k_mins(&raw, n, k));
    let s = scratch(gpu, w.scratch_bytes(mmax))?;
    let out = gpu.alloc(mmax * n * 2)?;
    for m in PREFILL_MS {
        let xb = rng.bf16_bits(m * k);
        let rows: Vec<Vec<Q8Block>> = xb
            .chunks_exact(k)
            .map(|r| quantize_q8_1_mmq_row(r, mmq_layout(ty)))
            .collect();
        let xq: Vec<f32> = rows.iter().flat_map(|r| q8_1_values(r)).collect();
        let (mut want, mag) = gemv_par(&wf, n, k, &xq, m, threads);
        let slack = match &mins {
            Some(mins) => {
                let (corr, slack) = q4_k_sum_correction(mins, n, k, &rows);
                for (v, c) in want.iter_mut().zip(&corr) {
                    *v += c;
                }
                Some(slack)
            }
            None => None,
        };
        let x = upload(gpu, &u16_bytes(&xb))?;
        ensure!(
            w.prefill(gpu, x, m, out, &s, 0)?,
            "{tag}: the prefill declined with a full workspace"
        );
        let got = widen(&download_u16(gpu, out, m * n)?);
        let (o, wst) = grade_slack(&got, &want, &mag, slack.as_deref(), k, prefill_bound);
        t.gate(
            &format!("{tag} MMQ within the prefill bound, M = {m}"),
            o,
            &format!("worst {wst:.3}"),
        );
        if m == PREFILL_MS[0] {
            ensure!(
                planted.prefill(gpu, x, m, out, &s, 0)?,
                "{tag}: the planted prefill declined"
            );
            let got = widen(&download_u16(gpu, out, m * n)?);
            let (o, _) = grade_slack(&got, &want, &mag, slack.as_deref(), k, prefill_bound);
            t.control(&format!("{tag} MMQ, last super-block zeroed"), o, "");
        }
        gpu.free(x)?;
    }
    release(gpu, &*w)?;
    release(gpu, &*planted)?;
    gpu.free(s.ws)?;
    gpu.free(out)?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let gates_only = args.iter().any(|a| a == "--gates-only");
    let quick = args.iter().any(|a| a == "--quick");
    let gpu = AvarokCudaBackend::new(0, &avarok_kernels::ptx_modules())?;
    for ty in TYPES {
        ensure!(
            launch::available(&gpu, ty),
            "the {ty:?} keep-packed modules are absent: build with AVAROK_TARGET_HW=ada"
        );
    }
    ensure!(
        gpu.has_module(keep_packed_iq4::DEQUANT_MODULE),
        "{} is absent",
        keep_packed_iq4::DEQUANT_MODULE
    );
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let mut rng = Rng(0x0019_4c4b_2026_1112);
    let mut t = Tally::default();

    println!("dequant:");
    dequant_gates(&gpu, &mut rng, &mut t)?;
    println!("activation quantizers:");
    quantizer_gates(&gpu, &mut rng, &mut t)?;
    println!("decode GEMV:");
    let decode: &[(usize, usize)] = if quick { &DECODE_QUICK } else { &DECODE };
    for ty in TYPES {
        for &(n, k) in decode {
            decode_shape(&gpu, &mut rng, ty, n, k, threads, &mut t)?;
        }
    }
    println!("prefill MMQ:");
    let prefill: &[(usize, usize)] = if quick { &PREFILL[..1] } else { &PREFILL };
    for ty in TYPES {
        for &(n, k) in prefill {
            prefill_shape(&gpu, &mut rng, ty, n, k, threads, &mut t)?;
        }
    }
    let bars = if gates_only {
        None
    } else {
        Some(timing::rows(&gpu, &mut rng)?)
    };
    let verdict = |b: bool| if b { "PASS" } else { "MISS" };
    println!(
        "summary: {} gates and controls, {} failed; bars: {}",
        t.gates,
        t.fails,
        match bars {
            None => "not run".to_string(),
            Some((r2, r3)) => format!("R2 {}, R3 {}", verdict(r2), verdict(r3)),
        }
    );
    ensure!(t.fails == 0, "{} gate(s) or control(s) failed", t.fails);
    println!("all gates PASS");
    Ok(())
}
