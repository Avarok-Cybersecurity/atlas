// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the affine microtest's load-side gates, each with a planted control: the GPU
//! quantizer against its host mirror on random and edge groups (groups 32, 64, 128; clip search off and
//! on), the non-finite flag, the tiled pack and the scale transpose, the BF16 dequantizer and the group
//! sums, all byte for byte.

use anyhow::Result;
use spark_model::port::affine::reference::{
    Mlx, OFFS, bf16_of, dequant_bf16, f32_of, group_params, group_sums, npad_of, pack_tiles,
    quantize_mlx, transpose_pad16, unpack_tiles,
};
use spark_model::port::affine::{AffineKernels, kernels, quantize_weight};
use spark_model::port::w4::W4Opts;
use spark_runtime::gpu::GpuBackend;

use crate::util::{
    Lcg, Tally, bits, download_f32, download_u16, download_u32, u16_bytes, u32_bytes, unequal,
    upload,
};

/// Six rows of edge groups, `kg` groups a row: random, flat, one outlier, all negative, a range under
/// the 1e-7 step floor, and ties (`[0, 15]` with halves, where round-half-away and round-half-even
/// differ).
fn edge_rows(r: &mut Lcg, group: usize, kg: usize) -> Vec<f32> {
    let k = group * kg;
    let mut v = r.bf16s(k, 0.02);
    for g in 0..kg {
        v.extend(std::iter::repeat_n(
            f32_of(bf16_of(0.0137 * (g as f32 + 1.0))),
            group,
        ));
    }
    for _ in 0..kg {
        let mut grp = r.bf16s(group, 0.001);
        grp[group / 3] = 0.75;
        v.extend(grp);
    }
    v.extend(
        r.bf16s(k, 0.01)
            .into_iter()
            .map(|x| f32_of(bf16_of(-x.abs() - 0.003))),
    );
    v.extend((0..k).map(|i| f32_of(bf16_of(1e-38 * (i % 3) as f32))));
    for _ in 0..kg {
        let mut grp: Vec<f32> = (0..group).map(|i| (i % 16) as f32).collect();
        for (i, x) in grp.iter_mut().enumerate().skip(1).step_by(4) {
            *x = (i % 15) as f32 + 0.5;
        }
        grp[0] = 0.0;
        grp[group - 1] = 15.0;
        v.extend(grp);
    }
    v
}

/// Run the GPU quantizer alone into MLX-layout buffers; returns (words, scales, biases, flag).
fn gpu_quantize(
    gpu: &dyn GpuBackend,
    k: &AffineKernels,
    values: &[f32],
    n: usize,
    kdim: usize,
    group: u16,
    clip: u8,
) -> Result<(Vec<u32>, Vec<u16>, Vec<u16>, u32)> {
    let kg = kdim / group as usize;
    let src = upload(gpu, &u16_bytes(&bits(values)))?;
    let (w, s, b, f) = (
        gpu.alloc(n * kdim / 2)?,
        gpu.alloc(n * kg * 2)?,
        gpu.alloc(n * kg * 2)?,
        gpu.alloc(4)?,
    );
    gpu.memset(f, 0xFF, 4)?;
    kernels::quantize(gpu, k, src, n, kdim, group, clip, w, s, b, f, 0)?;
    let out = (
        download_u32(gpu, w, n * kdim / 8)?,
        download_u16(gpu, s, n * kg)?,
        download_u16(gpu, b, n * kg)?,
        download_u32(gpu, f, 1)?[0],
    );
    for p in [src, w, s, b, f] {
        gpu.free(p)?;
    }
    Ok(out)
}

fn mlx_unequal(q: &Mlx, w: &[u32], s: &[u16], b: &[u16]) -> usize {
    unequal(&q.words, w) + unequal(&q.scales, s) + unequal(&q.biases, b)
}

pub fn quantizer(
    gpu: &dyn GpuBackend,
    k: &AffineKernels,
    r: &mut Lcg,
    t: &mut Tally,
) -> Result<()> {
    for group in [32u16, 64, 128] {
        let (g, kg) = (group as usize, 8);
        let v = edge_rows(r, g, kg);
        let n = v.len() / (g * kg);
        for clip in [0u8, 10] {
            let (w, s, b, flag) = gpu_quantize(gpu, k, &v, n, g * kg, group, clip)?;
            let host = quantize_mlx(&v, n, g * kg, g, clip).expect("edge rows are finite");
            let bad = mlx_unequal(&host, &w, &s, &b) + usize::from(flag != u32::MAX);
            t.gate(
                &format!("quantizer g{group} clip {clip}"),
                bad,
                &format!("({n} rows of edge groups, words+scales+biases vs host, flag {flag:#x})"),
            );
            if clip == 0 {
                // PLANTED CONTROL: round half to even on the tie row must differ from the kernel.
                let row = n - 1;
                let mut even = vec![0u32; g * kg / 8];
                for gi in 0..kg {
                    let base = row * g * kg + gi * g;
                    let grp = &v[base..base + g];
                    let (sc, bi) = group_params(grp, 0).expect("tie group");
                    for (j, &x) in grp.iter().enumerate() {
                        let q = ((x - bi) / sc).round_ties_even().clamp(0.0, 15.0) as u32;
                        even[(gi * g + j) / 8] |= q << (4 * ((gi * g + j) % 8));
                    }
                }
                let got = &w[row * g * kg / 8..(row + 1) * g * kg / 8];
                t.control(
                    &format!("half-even quantizer g{group}"),
                    unequal(&even, got),
                    "(tie row)",
                );
            }
        }
    }
    // The flag names the FIRST bad group, as the host quantizer does.
    let mut v = r.bf16s(4 * 256, 0.02);
    v[5 * 64 + 9] = f32::INFINITY;
    v[3 * 64 + 2] = f32::NAN;
    let (_, _, _, flag) = gpu_quantize(gpu, k, &v, 4, 256, 64, 0)?;
    let want = quantize_mlx(&v, 4, 256, 64, 0).err();
    t.gate(
        "non-finite flag",
        usize::from(want != Some(flag as usize)),
        &format!("(gpu {flag}, host {want:?})"),
    );
    Ok(())
}

/// The pack, transpose, dequant and group-sum kernels at every group, on shapes with a ragged n.
pub fn layout(gpu: &dyn GpuBackend, k: &AffineKernels, r: &mut Lcg, t: &mut Tally) -> Result<()> {
    for group in [32u16, 64, 128] {
        let g = group as usize;
        let (n, kd) = (200usize, g * 12);
        let (npad, kg) = (npad_of(n), kd / g);
        // Pack and transpose of host-made MLX data.
        let words: Vec<u32> = (0..n * kd / 8)
            .map(|_| (r.unit() * 4_294_967_295.0) as u32)
            .collect();
        let scales: Vec<u16> = (0..n * kg).map(|_| bf16_of(r.unit())).collect();
        let (wd, sd) = (
            upload(gpu, &u32_bytes(&words))?,
            upload(gpu, &u16_bytes(&scales))?,
        );
        let (td, sts) = (gpu.alloc(npad * kd / 2)?, gpu.alloc(kg * npad * 2)?);
        kernels::pack_tiles(gpu, k, wd, n, kd, group, npad, td, 0)?;
        kernels::transpose_pad16(gpu, k, sd, n, kg, npad, sts, 0)?;
        let tiles = download_u32(gpu, td, npad * kd / 8)?;
        let st = download_u16(gpu, sts, kg * npad)?;
        let bad = unequal(&tiles, &pack_tiles(&words, n, kd, g))
            + unequal(&unpack_tiles(&tiles, n, kd, g), &words)
            + unequal(&st, &transpose_pad16(&scales, n, kg, npad));
        t.gate(
            &format!("pack + transpose g{group} [{n}x{kd}]"),
            bad,
            "(vs host, round trip)",
        );
        // PLANTED CONTROL: a host pack with two nibble slots swapped must not match the kernel.
        let wrong: Vec<u32> = pack_tiles(&words, n, kd, g)
            .into_iter()
            .map(|w| {
                let (a, b) = ((w >> 4) & 0xF, (w >> 16) & 0xF);
                (w & !0x000F_00F0) | (b << 4) | (a << 16)
            })
            .collect();
        t.control(
            &format!("swapped-nibble pack g{group}"),
            unequal(&wrong, &tiles),
            &format!("(slots {} and {})", OFFS[1], OFFS[4]),
        );
        for p in [wd, sd, td, sts] {
            gpu.free(p)?;
        }

        // Dequant of a weight quantized on the device.
        let v = r.bf16s(n * kd, 0.02);
        let src = upload(gpu, &u16_bytes(&bits(&v)))?;
        let opts = W4Opts {
            bits: 4,
            group,
            clip_search: 0,
            int8_prefill: false,
        };
        let w = quantize_weight(gpu, src, n, kd, &opts, 0)?;
        let out = gpu.alloc(n * kd * 2)?;
        kernels::dequant_bf16(gpu, &w, out, 0)?;
        let got = download_u16(gpu, out, n * kd)?;
        let host = quantize_mlx(&v, n, kd, g, 0).expect("finite");
        t.gate(
            &format!("dequant g{group}"),
            unequal(&got, &dequant_bf16(&host)),
            "(bf16(s*q+b) vs host)",
        );
        w.release(gpu)?;
        for p in [src, out] {
            gpu.free(p)?;
        }

        // Group sums of 16 rows, in order.
        let x = bits(&r.bf16s(16 * kd, 2.0));
        let xd = upload(gpu, &u16_bytes(&x))?;
        let xs = gpu.alloc(16 * kg * 4)?;
        kernels::group_sums(gpu, k, xd, 16, kd, group, xs, 0)?;
        let got = download_f32(gpu, xs, 16 * kg)?;
        let want = group_sums(&x, 16, kd, g);
        let bad = got
            .iter()
            .zip(&want)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        t.gate(
            &format!("group sums g{group}"),
            bad,
            "(in-order fp32 vs host)",
        );
        for p in [xd, xs] {
            gpu.free(p)?;
        }
    }
    Ok(())
}
