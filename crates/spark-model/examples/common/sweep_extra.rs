// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the `--route-kernel` option of `linear_t_sweep`, which adds a
//! kernel to the sweep by module and entry name when a later lane's kernel
//! (affine 4-bit, Marlin, IQ4) has one of the six launch conventions below. A
//! kernel that is absent from the target is skipped with one line, never an
//! error, so one command line works before and after those lanes land.
//!
//! Syntax: `--route-kernel label=module:entry:abi[:grid=D][:tmax=T]`, repeatable.
//! `grid=D` is the output-column divisor of the launch grid (default 4, the
//! `ceil(N/4)` of every dense GEMV in the tree); `tmax=T` is the largest row
//! count the kernel serves (default 1 for a GEMV ABI, 16 for a batch ABI).
#![allow(dead_code)]

use crate::sweep_report::Family;
use anyhow::{Result, anyhow, bail};

/// The launch conventions a `--route-kernel` can name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Abi {
    /// `(A, B, C, N, K)`.
    Bf16Gemv,
    /// `(A, B, C, M, N, K, out_stride)`.
    Bf16Batch,
    /// `(A, B, scale, C, N, K)`.
    Fp8Gemv,
    /// `(A, B, scale, C, M, N, K)`.
    Fp8Batch,
    /// `(A, B_packed, B_scale, scale2, C, N, K)`.
    W4Gemv,
    /// `(A, B_packed, B_scale, scale2, C, M, N, K)`.
    W4Batch,
}

impl Abi {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "bf16-gemv" => Self::Bf16Gemv,
            "bf16-batch" => Self::Bf16Batch,
            "fp8-gemv" => Self::Fp8Gemv,
            "fp8-batch" => Self::Fp8Batch,
            "w4-gemv" => Self::W4Gemv,
            "w4-batch" => Self::W4Batch,
            _ => return None,
        })
    }

    pub fn family(self) -> Family {
        match self {
            Self::Bf16Gemv | Self::Bf16Batch => Family::Bf16,
            Self::Fp8Gemv | Self::Fp8Batch => Family::Fp8,
            Self::W4Gemv | Self::W4Batch => Family::Nvfp4,
        }
    }

    pub fn is_batch(self) -> bool {
        matches!(self, Self::Bf16Batch | Self::Fp8Batch | Self::W4Batch)
    }
}

/// A parsed `--route-kernel`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExtraSpec {
    pub label: String,
    pub module: String,
    pub entry: String,
    pub abi: Abi,
    pub grid_div: u32,
    pub t_max: u32,
}

/// Parse a `--route-kernel` value.
pub fn parse_extra(spec: &str) -> Result<ExtraSpec> {
    let (label, rest) = spec
        .split_once('=')
        .ok_or_else(|| anyhow!("--route-kernel {spec:?}: expected label=module:entry:abi"))?;
    let parts: Vec<&str> = rest.split(':').collect();
    if parts.len() < 3 {
        bail!("--route-kernel {spec:?}: expected module:entry:abi after the label");
    }
    let abi = Abi::parse(parts[2]).ok_or_else(|| {
        anyhow!(
            "--route-kernel {spec:?}: unknown abi {:?} (bf16-gemv, bf16-batch, fp8-gemv, \
             fp8-batch, w4-gemv, w4-batch)",
            parts[2]
        )
    })?;
    let mut e = ExtraSpec {
        label: label.to_string(),
        module: parts[0].to_string(),
        entry: parts[1].to_string(),
        abi,
        grid_div: 4,
        t_max: if abi.is_batch() { 16 } else { 1 },
    };
    for opt in &parts[3..] {
        let (key, val) = opt
            .split_once('=')
            .ok_or_else(|| anyhow!("--route-kernel {spec:?}: option {opt:?} needs key=value"))?;
        let v: u32 = val
            .parse()
            .map_err(|_| anyhow!("--route-kernel {spec:?}: bad number in {opt:?}"))?;
        match key {
            "grid" if v > 0 => e.grid_div = v,
            "tmax" if v > 0 => e.t_max = v,
            _ => bail!("--route-kernel {spec:?}: unknown or zero option {opt:?}"),
        }
    }
    Ok(e)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extra_specs_parse() {
        let e = parse_extra("aff4=affine_gemv:affine4_gemv_m1:w4-gemv").unwrap();
        assert_eq!(
            (e.label.as_str(), e.module.as_str(), e.entry.as_str()),
            ("aff4", "affine_gemv", "affine4_gemv_m1")
        );
        assert_eq!((e.abi, e.grid_div, e.t_max), (Abi::W4Gemv, 4, 1));
        let e = parse_extra("m=mod:ent:fp8-batch:grid=8:tmax=32").unwrap();
        assert_eq!((e.abi, e.grid_div, e.t_max), (Abi::Fp8Batch, 8, 32));
        assert_eq!(Abi::Fp8Batch.family(), Family::Fp8);
        assert_eq!(parse_extra("a=b:c:bf16-batch").unwrap().t_max, 16);
    }

    #[test]
    fn bad_extra_specs_are_errors() {
        for bad in [
            "nolabel",
            "a=b:c",
            "a=b:c:nope",
            "a=b:c:w4-gemv:grid",
            "a=b:c:w4-gemv:grid=x",
            "a=b:c:w4-gemv:wat=1",
            "a=b:c:w4-gemv:grid=0",
        ] {
            assert!(parse_extra(bad).is_err(), "{bad}");
        }
    }
}
