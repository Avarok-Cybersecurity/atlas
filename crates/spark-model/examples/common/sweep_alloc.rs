// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: device buffers of `linear_t_sweep` and `w4a16_ada_microtest`:
//! one rotated weight ring per format, and the activation and output buffers
//! sized for the largest row count. Weights are synthetic but finite
//! (`weight_fill.rs`); a ring is planned by `cold_l2::plan_ring`, so a cold run
//! cannot be built with weights that fit in the L2.
//!
//! An example that includes this file also declares the `cold_l2`, `profiler`,
//! `weight_fill` and `sweep_report` modules.
#![allow(dead_code)]

use crate::cold_l2::{Ring, RingPlan, build_ring, plan_hot, plan_ring};
use crate::sweep_report::{Family, weight_bytes};
use crate::weight_fill::{Fill, alloc_filled};
use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// Weight rings, one per family in use.
pub struct Weights {
    pub bf16: Option<Ring<DevicePtr>>,
    pub fp8: Option<Ring<(DevicePtr, DevicePtr)>>,
    pub fp4: Option<Ring<(DevicePtr, DevicePtr)>>,
    pub plans: Vec<(Family, RingPlan)>,
}

#[derive(Clone, Copy, Debug)]
pub struct AllocCfg {
    pub l2_bytes: usize,
    pub cap_bytes: usize,
    /// One copy per family: the L2-resident contrast.
    pub hot: bool,
}

pub fn alloc_weights(
    gpu: &dyn GpuBackend,
    families: &[Family],
    n: u32,
    k: u32,
    cfg: &AllocCfg,
) -> Result<Weights> {
    let (nn, kk) = (n as usize, k as usize);
    let mut w = Weights {
        bf16: None,
        fp8: None,
        fp4: None,
        plans: Vec::new(),
    };
    for &f in families {
        let per = weight_bytes(f, n, k);
        let plan = if cfg.hot {
            plan_hot(per, cfg.l2_bytes)
        } else {
            plan_ring(per, cfg.l2_bytes, cfg.cap_bytes)?
        };
        let seed = |c: usize| 0x5EED + 31 * c as u64;
        match f {
            Family::Bf16 => {
                w.bf16 = Some(build_ring(&plan, |c| {
                    alloc_filled(gpu, nn * kk * 2, Fill::Bf16, seed(c))
                })?);
            }
            Family::Fp8 => {
                let sbytes = nn.div_ceil(128) * kk.div_ceil(128) * 4;
                w.fp8 = Some(build_ring(&plan, |c| {
                    Ok((
                        alloc_filled(gpu, nn * kk, Fill::Fp8E4m3, seed(c))?,
                        alloc_filled(gpu, sbytes, Fill::ScaleF32, seed(c) + 1)?,
                    ))
                })?);
            }
            Family::Nvfp4 => {
                w.fp4 = Some(build_ring(&plan, |c| {
                    Ok((
                        alloc_filled(gpu, nn * kk / 2, Fill::Nvfp4Packed, seed(c))?,
                        alloc_filled(gpu, nn * kk / 16, Fill::Nvfp4Scale, seed(c) + 1)?,
                    ))
                })?);
            }
        }
        w.plans.push((f, plan));
    }
    Ok(w)
}

pub fn free_weights(gpu: &dyn GpuBackend, w: &Weights) -> Result<()> {
    if let Some(r) = &w.bf16 {
        for p in r.all() {
            gpu.free(*p)?;
        }
    }
    for r in [&w.fp8, &w.fp4].into_iter().flatten() {
        for (a, b) in r.all() {
            gpu.free(*a)?;
            gpu.free(*b)?;
        }
    }
    Ok(())
}

/// Activation and output buffers sized for `max_t` rows.
#[derive(Clone, Copy, Debug)]
pub struct Io {
    pub a: DevicePtr,
    pub c: DevicePtr,
    pub a_fp8: DevicePtr,
    pub a_scale: DevicePtr,
}

pub fn alloc_io(gpu: &dyn GpuBackend, n: u32, k: u32, max_t: u32) -> Result<Io> {
    let (nn, kk, mt) = (n as usize, k as usize, max_t as usize);
    let io = Io {
        a: alloc_filled(gpu, mt * kk * 2, Fill::Bf16, 0xAC7)?,
        c: gpu.alloc(mt * nn * 2)?,
        a_fp8: gpu.alloc(mt * kk)?,
        a_scale: gpu.alloc(mt * kk.div_ceil(128) * 4)?,
    };
    gpu.memset(io.c, 0, mt * nn * 2)?;
    Ok(io)
}

pub fn free_io(gpu: &dyn GpuBackend, io: &Io) -> Result<()> {
    for p in [io.a, io.c, io.a_fp8, io.a_scale] {
        gpu.free(p)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_hot_config_is_one_copy_per_family() {
        // The planner is exercised in cold_l2's tests; here only the glue that
        // picks it: a hot config never asks for a cold ring.
        let cfg = AllocCfg {
            l2_bytes: 32 << 20,
            cap_bytes: 1 << 30,
            hot: true,
        };
        let per = weight_bytes(Family::Fp8, 9216, 2560);
        let plan = if cfg.hot {
            plan_hot(per, cfg.l2_bytes)
        } else {
            plan_ring(per, cfg.l2_bytes, cfg.cap_bytes).unwrap()
        };
        assert_eq!(plan.copies, 1);
        assert!(!plan.is_cold());
    }
}
