// SPDX-License-Identifier: AGPL-3.0-only

//! Exact production-kernel regression for speculative EXL3 expert batches.
//! Fixed synthetic checkpoint geometry, real packed weights, no mocks. The
//! reference is repeated single-token production dispatch. Memory stays below
//! 100 MB: this leg needs no dequantized/f64 weight oracle.

use anyhow::{Result, ensure};
use spark_model::layers::ops::{Exl3MoeProj, exl3_moe_decode_routed};

use crate::legs_moe::{ProjSet, Slabs, alloc_slabs};
use crate::util::{Ctx, DevWeight, Lcg, as_bytes, down_u16};

const H: usize = 2560;
const I: usize = 640;
const EXPERTS: usize = 10;
const MAX_ROWS: usize = 4;

fn packed(ctx: &Ctx, rng: &mut Lcg, k: usize, n: usize, bits: u32) -> Result<ProjSet> {
    let mut weights = Vec::new();
    let mut dev = Vec::new();
    for _ in 0..EXPERTS {
        let trellis: Vec<_> = (0..(k / 16) * (n / 16) * 16 * bits as usize)
            .map(|_| rng.u16())
            .collect();
        let suh: Vec<_> = (0..k).map(|_| rng.scale_f16()).collect();
        let svh: Vec<_> = (0..n).map(|_| rng.scale_f16()).collect();
        dev.push(DevWeight::upload(ctx.g, &trellis, &suh, &svh)?);
        weights.push((trellis, suh, svh));
    }
    Ok(ProjSet {
        weights,
        whats: Vec::new(),
        dev,
        k_bits: bits,
    })
}

#[allow(clippy::too_many_arguments)]
fn launch(
    ctx: &Ctx,
    slab: &Slabs,
    tables: &[Exl3MoeProj; 3],
    input: &[u16],
    ids: &[u32],
    probs: &[f32],
    top_k: usize,
    stable: bool,
) -> Result<Vec<u16>> {
    let rows = input.len() / H;
    ensure!(input.len() == rows * H && ids.len() == rows * top_k);
    ensure!(probs.len() == ids.len());
    ctx.g.copy_h2d(&as_bytes(input), slab.input)?;
    ctx.g.copy_h2d(
        &ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
        slab.indices,
    )?;
    ctx.g.copy_h2d(
        &probs
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
        slab.probs,
    )?;
    let stream = ctx.g.default_stream();
    exl3_moe_decode_routed(
        ctx.g,
        slab.input,
        slab.indices,
        slab.probs,
        slab.out,
        tables,
        &slab.scratch,
        ctx.locks,
        rows,
        top_k,
        H,
        I,
        0,
        EXPERTS,
        0.0,
        stable,
        ctx.sms,
        stream,
    )?;
    ctx.g.synchronize(stream)?;
    down_u16(ctx.g, slab.out, rows * H)
}

pub fn run(ctx: &Ctx) -> Result<bool> {
    let mut rng = Lcg(0x005E_ED38_F1A5);
    let mut clean = true;
    let mut control_diffs = 0;
    let slab = alloc_slabs(ctx, MAX_ROWS * EXPERTS, MAX_ROWS)?;
    for bits in [4, 5, 6] {
        let gate = packed(ctx, &mut rng, H, I, bits)?;
        let up = packed(ctx, &mut rng, H, I, bits)?;
        let down = packed(ctx, &mut rng, I, H, bits)?;
        let (gt, go) = gate.table(ctx, 0)?;
        let (ut, uo) = up.table(ctx, 0)?;
        let (dt, d_o) = down.table(ctx, 0)?;
        let tables = [gt, ut, dt];
        for top_k in [3, 10] {
            for rows in [2, 3, 4] {
                let input: Vec<_> = (0..rows * H)
                    .map(|_| half::bf16::from_f32(rng.gauss()).to_bits())
                    .collect();
                // Distinct experts per token, different order per row. Adjacent
                // rows overlap weights but must never mix routing or outputs.
                let ids: Vec<_> = (0..rows)
                    .flat_map(|row| (0..top_k).map(move |slot| ((row * 3 + slot) % EXPERTS) as u32))
                    .collect();
                let mut probs: Vec<_> = (0..ids.len()).map(|_| 0.05 + rng.f()).collect();
                for p in probs.chunks_mut(top_k) {
                    let total: f32 = p.iter().sum();
                    for v in p {
                        *v /= total;
                    }
                }
                let mut serial = Vec::new();
                for row in 0..rows {
                    serial.extend(launch(
                        ctx,
                        &slab,
                        &tables,
                        &input[row * H..(row + 1) * H],
                        &ids[row * top_k..(row + 1) * top_k],
                        &probs[row * top_k..(row + 1) * top_k],
                        top_k,
                        false,
                    )?);
                }
                let stable = launch(ctx, &slab, &tables, &input, &ids, &probs, top_k, true)?;
                let old = launch(ctx, &slab, &tables, &input, &ids, &probs, top_k, false)?;
                let diffs = stable.iter().zip(&serial).filter(|(a, b)| a != b).count();
                let negative = old.iter().zip(&serial).filter(|(a, b)| a != b).count();
                // Reject vacuous all-zero or nonfinite fixtures.
                let finite = serial.iter().all(|&v| half::bf16::from_bits(v).is_finite());
                let nonzero = serial.iter().any(|&v| v & 0x7fff != 0);
                println!(
                    "verify-grid bits={bits} top_k={top_k} rows={rows}: stable_diff={diffs}/{} old_diff={negative} finite={finite} nonzero={nonzero}",
                    serial.len()
                );
                clean &= diffs == 0 && finite && nonzero;
                control_diffs += negative;
            }
        }
        for ptr in go.into_iter().chain(uo).chain(d_o) {
            ctx.g.free(ptr)?;
        }
        gate.free(ctx.g);
        up.free(ctx.g);
        down.free(ctx.g);
    }
    for ptr in slab.owned {
        ctx.g.free(ptr)?;
    }
    println!("verify-grid negative-control differences={control_diffs}");
    Ok(clean && control_diffs > 0)
}
