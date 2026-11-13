// SPDX-License-Identifier: AGPL-3.0-only
// Method after TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce,
// docs/recipes/qwen3.8-27b.md:84-86 (time a GEMV over distinct weight copies, since a matrix called again
// sits in the cache and misleads) and zig/src/cuda/qmmf.md "Device checks and timing" (rotate the ways'
// order). Apache-2.0; see TENSORFOLD-ATTRIBUTION.md. Changes: no code copied; Atlas's sync'd `Instant`
// loop, A B B A rotation against `fp8_gemv_ada`.

//! Responsibility: the affine microtest's DRAM-bound timing rows: GB/s of weight bytes at M = 1 and
//! M = 4 for the 4B decode shapes, each call reading a distinct weight copy (more than 64 MB in rotation,
//! twice the RTX 4060 Ti's 32 MB L2), rotated A B B A against `fp8_gemv_ada` on the same shapes.
//!
//! The acceptance bar (master plan L17a): the gate+up M = 1 row at or above 85% of `fp8_gemv_ada`'s GB/s
//! in the same run. A miss is reported as `BAR MISS`; it does not fail the correctness gates.

use anyhow::{Context, Result};
use spark_model::layers::ops;
use spark_model::port::affine::{affine_of, quantize_from_bf16, release};
use spark_model::port::w4::{W4Box, W4Opts, W4Scratch};
use spark_model::weight_map::{DenseWeight, quantize_to_fp8_blockscaled};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::util::{Lcg, Out, bits, time_us, u16_bytes, upload};

/// Bytes the rotation must exceed (2x the 4060 Ti's L2).
const ROTATION_BYTES: usize = 64 << 20;
/// The acceptance bar: affine GB/s over FP8 GB/s on the gate+up M = 1 row.
pub const BAR: f64 = 0.85;

pub struct Fp8Kernels {
    pub quant: KernelHandle,
    pub gemv: KernelHandle,
    pub dual: KernelHandle,
}

impl Fp8Kernels {
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        let k = |m: &str, f: &str| {
            gpu.kernel(m, f)
                .with_context(|| format!("{m}::{f} absent: build with AVAROK_TARGET_HW=ada"))
        };
        Ok(Self {
            quant: k(
                "quantize_bf16_to_fp8_blockscaled",
                "quantize_bf16_to_fp8_blockscaled",
            )?,
            gemv: k("fp8_gemv_ada", "fp8_gemv_ada")?,
            dual: k("fp8_gemv_ada", "fp8_gemv_ada_dual")?,
        })
    }
}

/// One timing row: `members` projections of one input (1, or 2 for gate+up), each `n x k`.
struct Row {
    name: &'static str,
    n: usize,
    k: usize,
    members: usize,
}

const ROWS: [Row; 4] = [
    Row {
        name: "gate+up",
        n: 9216,
        k: 2560,
        members: 2,
    },
    Row {
        name: "down",
        n: 2560,
        k: 9216,
        members: 1,
    },
    Row {
        name: "in_proj_qkvz",
        n: 12288,
        k: 2560,
        members: 1,
    },
    Row {
        name: "out_proj",
        n: 2560,
        k: 4096,
        members: 1,
    },
];

fn copies_for(bytes_per_call: usize) -> usize {
    (ROTATION_BYTES.div_ceil(bytes_per_call.max(1)) + 1).max(2)
}

fn gbs(bytes: usize, us: f64) -> f64 {
    bytes as f64 / us / 1e3
}

/// Affine copies: `copies` sets of `members` boxes from one BF16 source.
fn affine_copies(
    gpu: &dyn GpuBackend,
    src: DevicePtr,
    row: &Row,
    copies: usize,
) -> Result<Vec<Vec<W4Box>>> {
    let opts = W4Opts {
        bits: 4,
        group: 64,
        clip_search: 0,
        int8_prefill: false,
    };
    (0..copies)
        .map(|_| {
            (0..row.members)
                .map(|_| quantize_from_bf16(gpu, src, row.n, row.k, &opts, 0))
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()
}

/// Run every row; returns whether the gate+up M = 1 bar held.
pub fn rows(gpu: &dyn GpuBackend, fk: &Fp8Kernels, r: &mut Lcg) -> Result<bool> {
    println!(
        "timing (GB/s of weight bytes, distinct copies over {} MB, A B B A):",
        ROTATION_BYTES >> 20
    );
    let mut bar = true;
    for row in &ROWS {
        let (n, k) = (row.n, row.k);
        let src = upload(gpu, &u16_bytes(&bits(&r.bf16s(n * k, 0.02))))?;
        let x = upload(gpu, &u16_bytes(&bits(&r.bf16s(16 * k, 1.0))))?;
        let one = affine_copies(gpu, src, row, 1)?;
        let abytes = row.members * affine_of(&*one[0][0]).expect("affine").layout().total();
        one_release(gpu, one)?;
        let acopies = affine_copies(gpu, src, row, copies_for(abytes))?;
        let fbytes = row.members * (n * k + n.div_ceil(128) * k.div_ceil(128) * 4);
        let fcopies: Vec<Vec<_>> = (0..copies_for(fbytes))
            .map(|_| {
                (0..row.members)
                    .map(|_| {
                        quantize_to_fp8_blockscaled(
                            &DenseWeight { weight: src },
                            n,
                            k,
                            gpu,
                            fk.quant,
                            0,
                        )
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<_>>()?;
        let scratch = acopies[0][0].scratch_bytes(16);
        let s = W4Scratch {
            ws: gpu.alloc(scratch)?,
            ws_bytes: scratch,
            tickets: DevicePtr::NULL,
            xq: DevicePtr::NULL,
            xs: DevicePtr::NULL,
        };
        let outs = [Out::new(gpu, 16 * n)?, Out::new(gpu, 16 * n)?];
        for m in [1usize, 4] {
            let affine = |i: usize| -> Result<()> {
                let c = &acopies[i % acopies.len()];
                if row.members == 2 {
                    c[0].decode_pair(&*c[1], gpu, x, m, outs[0].ptr(), outs[1].ptr(), &s, 0)
                } else {
                    c[0].decode(gpu, x, m, outs[0].ptr(), &s, 0)
                }
            };
            let fp8 = |i: usize| -> Result<()> {
                let c = &fcopies[i % fcopies.len()];
                if row.members == 2 {
                    ops::w8a16_gemv_dual(
                        gpu,
                        fk.dual,
                        x,
                        c[0].weight,
                        c[0].row_scale,
                        outs[0].ptr(),
                        c[1].weight,
                        c[1].row_scale,
                        outs[1].ptr(),
                        n as u32,
                        k as u32,
                        0,
                    )
                } else {
                    ops::w8a16_gemv(
                        gpu,
                        fk.gemv,
                        x,
                        c[0].weight,
                        c[0].row_scale,
                        outs[0].ptr(),
                        n as u32,
                        k as u32,
                        0,
                    )
                }
            };
            if m == 1 {
                let a1 = time_us(gpu, affine)?;
                let f1 = time_us(gpu, fp8)?;
                let f2 = time_us(gpu, fp8)?;
                let a2 = time_us(gpu, affine)?;
                let (au, fu) = ((a1 + a2) / 2.0, (f1 + f2) / 2.0);
                let ratio = gbs(abytes, au) / gbs(fbytes, fu);
                let held = ratio >= BAR;
                if row.name == "gate+up" {
                    bar = held;
                }
                println!(
                    "  [{}] {:<13} M=1 N={n} K={k}: affine {au:.1} us {:.0} GB/s | fp8_gemv_ada {fu:.1} us {:.0} GB/s | ratio {ratio:.2}{}",
                    if row.name == "gate+up" {
                        if held { "BAR PASS" } else { "BAR MISS" }
                    } else {
                        "INFO"
                    },
                    row.name,
                    gbs(abytes, au),
                    gbs(fbytes, fu),
                    if row.name == "gate+up" {
                        format!(" (bar {BAR})")
                    } else {
                        String::new()
                    }
                );
            } else {
                let au = time_us(gpu, affine)?;
                println!(
                    "  [INFO] {:<13} M={m} N={n} K={k}: affine {au:.1} us {:.0} GB/s (fp8_gemv_ada is M=1 only)",
                    row.name,
                    gbs(abytes, au)
                );
            }
        }
        for o in outs {
            o.free(gpu)?;
        }
        for c in fcopies.into_iter().flatten() {
            gpu.free(c.weight)?;
            gpu.free(c.row_scale)?;
        }
        one_release(gpu, acopies)?;
        for p in [src, x, s.ws] {
            gpu.free(p)?;
        }
    }
    Ok(bar)
}

fn one_release(gpu: &dyn GpuBackend, sets: Vec<Vec<W4Box>>) -> Result<()> {
    for b in sets.into_iter().flatten() {
        release(gpu, &*b)?;
    }
    Ok(())
}
