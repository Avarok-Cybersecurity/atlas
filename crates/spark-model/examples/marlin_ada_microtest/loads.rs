// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the load-side gates and the timing rows of the Marlin microtest: the opt-in RTN
//! route from a device BF16 weight (bit-equal to the CPU quantizer, decode within the bound, the
//! quantization error against the BF16 product reported), synthetic GPTQ (symmetric and v1
//! asymmetric) and AWQ tensor sets through `from_gptq` / `from_awq`, and decode GB/s on the 4B
//! shapes.

use anyhow::{Context, Result};
use spark_model::port::marlin::checkpoint::{self, GptqZeroFormat, ScaleDtype};
use spark_model::port::marlin::quantize::{RtnOpts, rtn_int4_sym};
use spark_model::port::marlin::{
    AwqWeights, GptqWeights, build, from_awq, from_gptq, release, route,
};
use spark_model::port::w4::{W4Box, W4Linear, W4Scratch};
use spark_runtime::gpu::GpuBackend;

use crate::gates::{plan_name, random_logical};
use crate::util::{Lcg, Tally, download_u16, oracle, time_us, u16_bytes, upload, upload_u16};
use spark_model::port::marlin::w4linear::DecodePolicy;

fn dequant_of(gpu: &dyn GpuBackend, w: &dyn W4Linear) -> Result<Vec<u16>> {
    let (n, k) = (w.n(), w.k());
    let out = gpu.alloc(n * k * 2)?;
    w.dequant_bf16(gpu, out, 0)?;
    let v = download_u16(gpu, out, n * k)?;
    gpu.free(out)?;
    Ok(v)
}

/// Decode `rows` rows of a fresh activation through `w` and check them against `w_nk`.
#[allow(clippy::too_many_arguments)]
fn decode_check(
    gpu: &dyn GpuBackend,
    r: &mut Lcg,
    s: &W4Scratch,
    t: &mut Tally,
    w: &dyn W4Linear,
    w_nk: &[u16],
    rows: usize,
    what: &str,
) -> Result<()> {
    let (n, k) = (w.n(), w.k());
    let x_host = r.bf16s(rows * k, 1.0);
    let x = upload_u16(gpu, &x_host)?;
    let out = gpu.alloc(rows * n * 2)?;
    gpu.memset_async(out, 0xFF, rows * n * 2, 0)?;
    w.decode(gpu, x, rows, out, s, 0)?;
    let got = download_u16(gpu, out, rows * n)?;
    let (worst, bad) = oracle(&got, &x_host, rows, w_nk, n, k, 7);
    t.gate(
        bad == 0,
        &format!("{what} m={rows} n={n} k={k} worst={worst:.3} bad={bad}"),
    );
    gpu.free(x)?;
    gpu.free(out)?;
    Ok(())
}

/// The RTN route: device BF16 in, a g128 kU4B8 weight out, with and without the clip search.
pub fn rtn(gpu: &dyn GpuBackend, r: &mut Lcg, s: &W4Scratch, t: &mut Tally) -> Result<()> {
    let (n, k) = (2560usize, 2560usize);
    let w = r.bf16s(n * k, 0.02);
    let src = upload_u16(gpu, &w)?;
    for clip in [0u8, 4] {
        let b = route::quantize_projection_with(gpu, true, src, n, k, clip, 0, "microtest")?
            .context("the RTN route declined an eligible shape")?;
        let want = rtn_int4_sym(
            &w,
            n,
            k,
            &RtnOpts {
                group: 128,
                clip_search: clip,
            },
        )?
        .dequant_nk();
        let got = dequant_of(gpu, &*b)?;
        t.gate(
            got == want,
            &format!("rtn clip={clip} [{n}x{k}] device image equals the CPU quantizer"),
        );
        for rows in [1usize, 4, 17] {
            decode_check(
                gpu,
                r,
                s,
                t,
                &*b,
                &want,
                rows,
                &format!("rtn clip={clip} decode"),
            )?;
        }
        // The quantization error itself (informational): relative RMS of W_q - W.
        let (mut e2, mut w2) = (0f64, 0f64);
        for (&a, &q) in w.iter().zip(&want) {
            let (a, q) = (
                f64::from(spark_model::port::marlin::reference::bf16_to_f32(a)),
                f64::from(spark_model::port::marlin::reference::bf16_to_f32(q)),
            );
            e2 += (q - a) * (q - a);
            w2 += a * a;
        }
        println!(
            "info rtn clip={clip} [{n}x{k}] relative weight RMS error {:.5}",
            (e2 / w2).sqrt()
        );
        release(gpu, &*b)?;
    }
    // Row off: no weight.
    let off = route::quantize_projection_with(gpu, false, src, n, k, 0, 0, "microtest")?;
    t.gate(off.is_none(), "rtn row off returns no weight");
    gpu.free(src)?;
    Ok(())
}

fn le16(v: &[u16]) -> Vec<u8> {
    u16_bytes(v)
}

/// Synthetic GPTQ and AWQ checkpoints: packed like the exporters, parsed, repacked, decoded.
pub fn checkpoints(gpu: &dyn GpuBackend, r: &mut Lcg, s: &W4Scratch, t: &mut Tally) -> Result<()> {
    let (n, k, group) = (2560usize, 2560usize, 128usize);
    let groups = k / group;

    // GPTQ symmetric (no qzeros, a trivial g_idx).
    let l = random_logical(r, n, k, false);
    let qweight = checkpoint::le_bytes(&checkpoint::gptq_pack(&l.q, k, n));
    let scales = le16(&l.scales);
    let g_idx: Vec<u32> = (0..k as u32).map(|i| i / group as u32).collect();
    let g_idx = checkpoint::le_bytes(&g_idx);
    let gw = GptqWeights {
        qweight: &qweight,
        scales: &scales,
        scale_dtype: ScaleDtype::Bf16,
        qzeros: None,
        g_idx: Some(&g_idx),
        n,
        k,
        group,
        zero_format: GptqZeroFormat::V1,
    };
    let b: W4Box = from_gptq(gpu, &gw, 0)?;
    t.gate(
        dequant_of(gpu, &*b)? == l.dequant_nk(),
        "gptq-sym device image equals the CPU dequant",
    );
    for rows in [1usize, 16] {
        decode_check(gpu, r, s, t, &*b, &l.dequant_nk(), rows, "gptq-sym decode")?;
    }
    release(gpu, &*b)?;

    // GPTQ asymmetric, AutoGPTQ v1 zero offsets.
    let l = random_logical(r, n, k, true);
    let qweight = checkpoint::le_bytes(&checkpoint::gptq_pack(&l.q, k, n));
    let scales = le16(&l.scales);
    let zeros = l.zeros.clone().context("zero points")?;
    let qzeros = checkpoint::le_bytes(&checkpoint::gptq_pack_zeros(
        &zeros,
        groups,
        n,
        GptqZeroFormat::V1,
    ));
    let gw = GptqWeights {
        qweight: &qweight,
        scales: &scales,
        scale_dtype: ScaleDtype::Bf16,
        qzeros: Some(&qzeros),
        g_idx: None,
        n,
        k,
        group,
        zero_format: GptqZeroFormat::V1,
    };
    let b = from_gptq(gpu, &gw, 0)?;
    t.gate(
        dequant_of(gpu, &*b)? == l.dequant_nk(),
        "gptq-v1-asym device image equals the CPU dequant",
    );
    for rows in [1usize, 16] {
        decode_check(
            gpu,
            r,
            s,
            t,
            &*b,
            &l.dequant_nk(),
            rows,
            "gptq-v1-asym decode",
        )?;
    }
    release(gpu, &*b)?;

    // AWQ (interleaved codes and zero points).
    let l = random_logical(r, n, k, true);
    let qweight = checkpoint::le_bytes(&checkpoint::awq_pack(&l.q, k, n));
    let zeros = l.zeros.clone().context("zero points")?;
    let qzeros = checkpoint::le_bytes(&checkpoint::awq_pack(&zeros, groups, n));
    let scales = le16(&l.scales);
    let aw = AwqWeights {
        qweight: &qweight,
        qzeros: &qzeros,
        scales: &scales,
        scale_dtype: ScaleDtype::Bf16,
        n,
        k,
        group,
    };
    let b = from_awq(gpu, &aw, 0)?;
    t.gate(
        dequant_of(gpu, &*b)? == l.dequant_nk(),
        "awq device image equals the CPU dequant",
    );
    for rows in [1usize, 16] {
        decode_check(gpu, r, s, t, &*b, &l.dequant_nk(), rows, "awq decode")?;
    }
    release(gpu, &*b)?;
    Ok(())
}

/// Decode GB/s on the 4B projection shapes (weight bytes read per call over the call time). The
/// BF16 column is the same shape's BF16 bytes over the same time, the bandwidth a BF16 GEMV would
/// need to match it.
pub fn timing(gpu: &dyn GpuBackend, r: &mut Lcg, s: &W4Scratch) -> Result<()> {
    for (n, k, what) in [
        (9216usize, 2560usize, "ffn gate/up"),
        (2560, 9216, "ffn down"),
        (4096, 2560, "attn q"),
    ] {
        for zp in [false, true] {
            let l = random_logical(r, n, k, zp);
            let lin = build(gpu, &l, 0)?;
            let x = upload(gpu, &u16_bytes(&r.bf16s(64 * k, 1.0)))?;
            let out = gpu.alloc(64 * n * 2)?;
            for m in [1usize, 4, 8, 16, 32, 64] {
                let us = time_us(gpu, 20, 200, || lin.decode(gpu, x, m, out, s, 0))?;
                let gbs = lin.device_bytes() as f64 / us / 1e3;
                let bf16 = (n * k * 2) as f64 / us / 1e3;
                println!(
                    "info timing {what} {} [{n}x{k}] m={m} {} us={us:.2} GB/s={gbs:.1} bf16_equiv_GB/s={bf16:.1}",
                    if zp { "u4" } else { "u4b8" },
                    plan_name(&lin, m, DecodePolicy::RowInvariant)?
                );
            }
            lin.release_unboxed(gpu)?;
            gpu.free(x)?;
            gpu.free(out)?;
        }
    }
    Ok(())
}
