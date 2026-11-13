// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the GPU half of `linear_t_sweep`: resolve the linear-layer
//! kernels, allocate rotated weight rings and activation buffers, and build the
//! per-route launch closures. One route is one kernel (or one fused sequence)
//! that computes `out[T, N] = act[T, K] x weight[N, K]^T`.
//!
//! Routes covered (the module and entry names are read from the kernel sources
//! under `kernels/gb10/common` and `kernels/ada/common`):
//!  * BF16: `dense_gemv_bf16`, `dense_gemv_bf16_batchm`, cuBLASLt;
//!  * FP8 (E4M3 weights, 128 x 128 FP32 block scales): `fp8_gemv_ada`, the gb10
//!    `w8a16_gemv`, `w8a16_gemv_batch4` and `_batch16`, and the W8A8 prefill
//!    GEMM (activation quantizer plus `fp8_gemm_ada_w8a8`);
//!  * NVFP4 (packed E2M1, E4M3 group-16 scales): `w4a16_gemv`, `w4a16_gemv_sw`,
//!    the fixed `batch2` and `batch3`, the tiers `batch4` to `batch8`, the rt2
//!    twin of `batch8`, `batch16` and `batch32`.
//! A later lane's kernel with one of the six ABIs of `sweep_extra.rs` joins by
//! name through `--route-kernel`; an absent kernel is skipped.
//!
//! An example that includes this file also declares the `cold_l2`, `profiler`,
//! `weight_fill`, `sweep_report`, `sweep_extra` and `sweep_alloc` modules.
#![allow(dead_code)]

use crate::cold_l2::Arm;
use crate::sweep_alloc::{Io, Weights};
use crate::sweep_extra::{Abi, ExtraSpec};
use crate::sweep_report::Family;
use anyhow::{Result, anyhow, bail};
use spark_model::layers::ops;
use spark_runtime::gpu::{GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

/// The NVFP4 second-level tensor scale used by every timing call.
pub const SCALE2: f32 = 0.0123;

/// Resolved kernel handles. `None` means the target does not carry the kernel.
#[derive(Clone, Copy, Debug, Default)]
pub struct Handles {
    pub bf16_gemv: Option<KernelHandle>,
    pub bf16_batchm: Option<KernelHandle>,
    pub fp8_ada: Option<KernelHandle>,
    pub fp8_gb10: Option<KernelHandle>,
    pub fp8_b4: Option<KernelHandle>,
    pub fp8_b16: Option<KernelHandle>,
    pub act_quant: Option<ops::Fp8ActQuant>,
    pub w8a8_gemm: Option<KernelHandle>,
    pub w4_gemv: Option<KernelHandle>,
    pub w4_sw: Option<KernelHandle>,
    /// `batch2`, `batch3` (no M argument).
    pub w4_fixed: [Option<KernelHandle>; 2],
    /// `batch4` to `batch8`.
    pub w4_batch: [Option<KernelHandle>; 5],
    pub w4_b8_rt2: Option<KernelHandle>,
    pub w4_b16: Option<KernelHandle>,
    pub w4_b32: Option<KernelHandle>,
}

impl Handles {
    pub fn resolve(gpu: &dyn GpuBackend) -> Self {
        let k = |m: &str, e: &str| gpu.kernel(m, e).ok();
        let w4 = |e: &str| k("w4a16_gemv", e);
        let act = k("per_token_group_quant_fp8", "per_token_group_quant_fp8")
            .map(ops::Fp8ActQuant::shared_only);
        Self {
            bf16_gemv: k("gemv", "dense_gemv_bf16"),
            bf16_batchm: k("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm"),
            fp8_ada: k("fp8_gemv_ada", "fp8_gemv_ada"),
            fp8_gb10: k("w8a16_gemv", "w8a16_gemv"),
            fp8_b4: k("w8a16_gemv_batch4", "w8a16_gemv_batch4"),
            fp8_b16: k("w8a16_gemv_batch4", "w8a16_gemv_batch16"),
            act_quant: act,
            w8a8_gemm: k("fp8_gemm_ada", "fp8_gemm_ada_w8a8")
                .or_else(|| k("fp8_gemm_t_blockscaled", "fp8_gemm_t_blockscaled")),
            w4_gemv: w4("w4a16_gemv"),
            w4_sw: w4("w4a16_gemv_sw"),
            w4_fixed: [w4("w4a16_gemv_batch2"), w4("w4a16_gemv_batch3")],
            w4_batch: [4, 5, 6, 7, 8].map(|w| w4(&format!("w4a16_gemv_batch{w}"))),
            w4_b8_rt2: w4("w4a16_gemv_batch8_rt2"),
            w4_b16: w4("w4a16_gemv_batch16"),
            w4_b32: w4("w4a16_gemv_batch32"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Kind {
    Bf16Gemv,
    Bf16BatchM,
    Bf16CublasLt,
    Fp8Gemv(KernelHandle),
    Fp8Batch4,
    Fp8Batch16,
    Fp8W8a8,
    Nvfp4Gemv,
    Nvfp4Sw,
    /// `batch2` or `batch3`: the row count is the kernel's, with no M argument.
    Nvfp4Fixed(u32),
    /// `batch4` to `batch8`: the tier width.
    Nvfp4Batch(u32),
    Nvfp4Batch8Rt2,
    Nvfp4Batch16,
    Nvfp4Batch32,
    /// A kernel named on the command line: its ABI, grid divisor and handle.
    Extra(Abi, u32, KernelHandle),
}

/// A route and the rows it can serve.
#[derive(Clone, Debug)]
pub struct Route {
    pub id: String,
    pub family: Family,
    pub kind: Kind,
    pub t_min: u32,
    pub t_max: u32,
}

impl Route {
    pub fn valid_at(&self, t: u32) -> bool {
        (self.t_min..=self.t_max).contains(&t)
    }
}

/// Every route whose kernels this target carries, plus the named extras.
pub fn catalog(h: &Handles, extras: &[Route]) -> Vec<Route> {
    let mut v = Vec::new();
    let mut add = |id: &str, family, kind, t_min, t_max| {
        v.push(Route {
            id: id.to_string(),
            family,
            kind,
            t_min,
            t_max,
        });
    };
    if h.bf16_gemv.is_some() {
        add("bf16_gemv", Family::Bf16, Kind::Bf16Gemv, 1, 1);
    }
    if h.bf16_batchm.is_some() {
        add("bf16_batchm", Family::Bf16, Kind::Bf16BatchM, 1, 16);
    }
    add("bf16_cublaslt", Family::Bf16, Kind::Bf16CublasLt, 1, 128);
    if let Some(k) = h.fp8_ada {
        add("fp8_gemv_ada", Family::Fp8, Kind::Fp8Gemv(k), 1, 1);
    }
    if let Some(k) = h.fp8_gb10 {
        add("fp8_gemv_gb10", Family::Fp8, Kind::Fp8Gemv(k), 1, 1);
    }
    if h.fp8_b4.is_some() {
        add("fp8_batch4", Family::Fp8, Kind::Fp8Batch4, 1, 4);
    }
    if h.fp8_b16.is_some() {
        add("fp8_batch16", Family::Fp8, Kind::Fp8Batch16, 1, 16);
    }
    if h.act_quant.is_some() && h.w8a8_gemm.is_some() {
        add("fp8_w8a8", Family::Fp8, Kind::Fp8W8a8, 1, 128);
    }
    if h.w4_gemv.is_some() {
        add("nvfp4_gemv", Family::Nvfp4, Kind::Nvfp4Gemv, 1, 1);
    }
    if h.w4_sw.is_some() {
        add("nvfp4_sw", Family::Nvfp4, Kind::Nvfp4Sw, 1, 1);
    }
    for (i, m) in [2u32, 3].into_iter().enumerate() {
        if h.w4_fixed[i].is_some() {
            add(
                &format!("nvfp4_batch{m}"),
                Family::Nvfp4,
                Kind::Nvfp4Fixed(m),
                m,
                m,
            );
        }
    }
    for (i, w) in [4u32, 5, 6, 7, 8].into_iter().enumerate() {
        if h.w4_batch[i].is_some() {
            add(
                &format!("nvfp4_batch{w}"),
                Family::Nvfp4,
                Kind::Nvfp4Batch(w),
                1,
                w,
            );
        }
    }
    if h.w4_b8_rt2.is_some() {
        add(
            "nvfp4_batch8_rt2",
            Family::Nvfp4,
            Kind::Nvfp4Batch8Rt2,
            1,
            8,
        );
    }
    if h.w4_b16.is_some() {
        add("nvfp4_batch16", Family::Nvfp4, Kind::Nvfp4Batch16, 1, 16);
    }
    if h.w4_b32.is_some() {
        add("nvfp4_batch32", Family::Nvfp4, Kind::Nvfp4Batch32, 1, 32);
    }
    v.extend(extras.iter().cloned());
    v
}

/// Resolve a parsed extra to a route, or `None` when the kernel is absent.
pub fn resolve_extra(gpu: &dyn GpuBackend, e: &ExtraSpec) -> Option<Route> {
    let h = gpu.kernel(&e.module, &e.entry).ok()?;
    Some(Route {
        id: format!("x:{}", e.label),
        family: e.abi.family(),
        kind: Kind::Extra(e.abi, e.grid_div, h),
        t_min: 1,
        t_max: e.t_max,
    })
}

/// Everything a launch closure borrows.
pub struct Ctx<'a> {
    pub gpu: &'a dyn GpuBackend,
    pub h: &'a Handles,
    pub w: &'a Weights,
    pub io: &'a Io,
    pub n: u32,
    pub k: u32,
    pub stream: u64,
}

fn need<T>(o: Option<T>, what: &str) -> Result<T> {
    o.ok_or_else(|| anyhow!("{what} is absent from this target"))
}

/// A launch closure for `route` at `t` rows. Each call reads the next weight
/// copy of the route's family ring.
pub fn make_arm<'a>(c: &'a Ctx<'a>, r: &Route, t: u32) -> Result<Arm<'a>> {
    let name = r.id.clone();
    let (gpu, io, n, k, s) = (c.gpu, c.io, c.n, c.k, c.stream);
    let g4 = [div_ceil(n, 4), 1, 1];
    let g8 = [div_ceil(n, 8), 1, 1];
    match r.family {
        Family::Bf16 => {
            let ring = need(c.w.bf16.as_ref(), "the bf16 weights")?;
            let launch = |h: KernelHandle, grid: [u32; 3], batch: bool| {
                move || {
                    let l = KernelLaunch::new(gpu, h)
                        .grid(grid)
                        .block([256, 1, 1])
                        .arg_ptr(io.a)
                        .arg_ptr(ring.pick())
                        .arg_ptr(io.c);
                    let l = if batch { l.arg_u32(t) } else { l };
                    let l = l.arg_u32(n).arg_u32(k);
                    (if batch { l.arg_u32(n) } else { l }).launch(s)
                }
            };
            Ok(match r.kind {
                Kind::Bf16Gemv => Arm::new(
                    name,
                    launch(need(c.h.bf16_gemv, "dense_gemv_bf16")?, g4, false),
                ),
                Kind::Bf16BatchM => Arm::new(
                    name,
                    launch(need(c.h.bf16_batchm, "dense_gemv_bf16_batchm")?, g4, true),
                ),
                Kind::Bf16CublasLt => Arm::new(name, move || {
                    spark_runtime::cublaslt::bf16_gemm_act_weight_t(
                        io.a.0,
                        ring.pick().0,
                        io.c.0,
                        t,
                        n,
                        k,
                        s,
                    )
                }),
                Kind::Extra(Abi::Bf16Gemv, gd, h) => {
                    Arm::new(name, launch(h, [div_ceil(n, gd), 1, 1], false))
                }
                Kind::Extra(Abi::Bf16Batch, gd, h) => {
                    Arm::new(name, launch(h, [div_ceil(n, gd), 1, 1], true))
                }
                _ => bail!("route {} is not a bf16 route", r.id),
            })
        }
        Family::Fp8 => {
            let ring = need(c.w.fp8.as_ref(), "the fp8 weights")?;
            // (A, B, scale, C, [M,] N, K)
            let launch = |h: KernelHandle, grid: [u32; 3], batch: bool| {
                move || {
                    let (w, sc) = ring.pick();
                    let l = KernelLaunch::new(gpu, h)
                        .grid(grid)
                        .block([256, 1, 1])
                        .arg_ptr(io.a)
                        .arg_ptr(w)
                        .arg_ptr(sc)
                        .arg_ptr(io.c);
                    let l = if batch { l.arg_u32(t) } else { l };
                    l.arg_u32(n).arg_u32(k).launch(s)
                }
            };
            Ok(match r.kind {
                Kind::Fp8Gemv(h) => Arm::new(name, launch(h, g4, false)),
                Kind::Fp8Batch4 => Arm::new(
                    name,
                    launch(need(c.h.fp8_b4, "w8a16_gemv_batch4")?, g4, true),
                ),
                Kind::Fp8Batch16 => Arm::new(
                    name,
                    launch(need(c.h.fp8_b16, "w8a16_gemv_batch16")?, g4, true),
                ),
                Kind::Fp8W8a8 => {
                    let aq = need(c.h.act_quant, "per_token_group_quant_fp8")?;
                    let gemm = need(c.h.w8a8_gemm, "the W8A8 GEMM")?;
                    Arm::new(name, move || {
                        ops::per_token_group_quant_fp8(
                            gpu, aq, io.a, io.a_fp8, io.a_scale, t, k, s,
                        )?;
                        let (w, sc) = ring.pick();
                        ops::fp8_gemm_t_blockscaled(
                            gpu, gemm, io.a_fp8, io.a_scale, w, sc, io.c, t, n, k, s,
                        )
                    })
                }
                Kind::Extra(Abi::Fp8Gemv, gd, h) => {
                    Arm::new(name, launch(h, [div_ceil(n, gd), 1, 1], false))
                }
                Kind::Extra(Abi::Fp8Batch, gd, h) => {
                    Arm::new(name, launch(h, [div_ceil(n, gd), 1, 1], true))
                }
                _ => bail!("route {} is not an fp8 route", r.id),
            })
        }
        Family::Nvfp4 => {
            let ring = need(c.w.fp4.as_ref(), "the nvfp4 weights")?;
            // (A, packed, scale, scale2, C, [M,] N, K)
            let launch = |h: KernelHandle, grid: [u32; 3], batch: bool| {
                move || {
                    let (w, sc) = ring.pick();
                    let l = KernelLaunch::new(gpu, h)
                        .grid(grid)
                        .block([256, 1, 1])
                        .arg_ptr(io.a)
                        .arg_ptr(w)
                        .arg_ptr(sc)
                        .arg_f32(SCALE2)
                        .arg_ptr(io.c);
                    let l = if batch { l.arg_u32(t) } else { l };
                    l.arg_u32(n).arg_u32(k).launch(s)
                }
            };
            Ok(match r.kind {
                Kind::Nvfp4Gemv => {
                    Arm::new(name, launch(need(c.h.w4_gemv, "w4a16_gemv")?, g4, false))
                }
                Kind::Nvfp4Sw => {
                    Arm::new(name, launch(need(c.h.w4_sw, "w4a16_gemv_sw")?, g8, false))
                }
                Kind::Nvfp4Fixed(m) => {
                    let h = need(c.h.w4_fixed[m as usize - 2], "w4a16_gemv_batch2/3")?;
                    Arm::new(name, launch(h, g4, false))
                }
                Kind::Nvfp4Batch(w) => {
                    let h = need(c.h.w4_batch[w as usize - 4], "w4a16_gemv_batch4..8")?;
                    Arm::new(name, launch(h, g4, true))
                }
                Kind::Nvfp4Batch8Rt2 => Arm::new(
                    name,
                    launch(need(c.h.w4_b8_rt2, "w4a16_gemv_batch8_rt2")?, g8, true),
                ),
                Kind::Nvfp4Batch16 => Arm::new(
                    name,
                    launch(need(c.h.w4_b16, "w4a16_gemv_batch16")?, g4, true),
                ),
                Kind::Nvfp4Batch32 => Arm::new(
                    name,
                    launch(need(c.h.w4_b32, "w4a16_gemv_batch32")?, g4, true),
                ),
                Kind::Extra(Abi::W4Gemv, gd, h) => {
                    Arm::new(name, launch(h, [div_ceil(n, gd), 1, 1], false))
                }
                Kind::Extra(Abi::W4Batch, gd, h) => {
                    Arm::new(name, launch(h, [div_ceil(n, gd), 1, 1], true))
                }
                _ => bail!("route {} is not an nvfp4 route", r.id),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_lists_only_present_kernels() {
        let none = Handles::default();
        let ids: Vec<String> = catalog(&none, &[]).into_iter().map(|r| r.id).collect();
        // cuBLASLt needs no module, so it is always offered.
        assert_eq!(ids, vec!["bf16_cublaslt"]);
        let k = KernelHandle(7);
        let h = Handles {
            w4_gemv: Some(k),
            w4_batch: [Some(k), None, None, None, Some(k)],
            w4_b32: Some(k),
            ..Handles::default()
        };
        let ids: Vec<String> = catalog(&h, &[]).into_iter().map(|r| r.id).collect();
        assert_eq!(
            ids,
            vec![
                "bf16_cublaslt",
                "nvfp4_gemv",
                "nvfp4_batch4",
                "nvfp4_batch8",
                "nvfp4_batch32"
            ]
        );
    }

    #[test]
    fn route_ranges() {
        let k = KernelHandle(1);
        let h = Handles {
            w4_fixed: [Some(k), Some(k)],
            w4_batch: [Some(k); 5],
            ..Handles::default()
        };
        let routes = catalog(&h, &[]);
        let get = |id: &str| routes.iter().find(|r| r.id == id).unwrap();
        assert!(get("nvfp4_batch2").valid_at(2) && !get("nvfp4_batch2").valid_at(1));
        assert!(
            get("nvfp4_batch6").valid_at(1)
                && get("nvfp4_batch6").valid_at(6)
                && !get("nvfp4_batch6").valid_at(7)
        );
        assert!(get("bf16_cublaslt").valid_at(128) && !get("bf16_cublaslt").valid_at(129));
    }
}
