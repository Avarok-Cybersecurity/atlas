// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the pure (CPU-testable) half of `linear_t_sweep`: the shapes,
//! the T lists and priority marks, the byte accounting, the `T1_lin_x` metric,
//! the route ladder Atlas dispatches today, and the table and JSON rows.
//!
//! Method credit: the per-T linear sweep and its `T1_lin_x` scaling metric come
//! from Cinference (`bench/README.md`, "Linear Op benchmark", and
//! `docs/maintainer/linear-tuning.md` sections 1 to 4). The shapes are the
//! Qwen3.5-4B projections, N x K as listed in `SHAPES`.
#![allow(dead_code)]

use serde_json::{Value, json};

/// Weight format of a route family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Family {
    Bf16,
    Fp8,
    Nvfp4,
}

impl Family {
    pub fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Fp8 => "fp8",
            Self::Nvfp4 => "nvfp4",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "bf16" => Some(Self::Bf16),
            "fp8" => Some(Self::Fp8),
            "nvfp4" => Some(Self::Nvfp4),
            _ => None,
        }
    }
}

/// A projection shape, `[N, K]` weight.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub id: &'static str,
    pub n: u32,
    pub k: u32,
}

/// The 4B shapes (hidden 2560): FFN gate or up, FFN down, the GDN input
/// projection, the 4096-wide output projection, and the LM head. The role names
/// are labels only; only N and K matter to the kernels.
pub const SHAPES: [Shape; 5] = [
    Shape {
        id: "ffn_up",
        n: 9216,
        k: 2560,
    },
    Shape {
        id: "ffn_down",
        n: 2560,
        k: 9216,
    },
    Shape {
        id: "gdn_in",
        n: 12288,
        k: 2560,
    },
    Shape {
        id: "out_proj",
        n: 2560,
        k: 4096,
    },
    Shape {
        id: "lm_head",
        n: 248_320,
        k: 2560,
    },
];

/// Largest T the sweep covers.
pub const MAX_T: u32 = 128;

/// Parse a T list: `all`, `priority`, `a..b`, `a,b,c` or a mix separated by
/// commas (`1..8,16,32`). The result is sorted, unique and within 1..=MAX_T.
pub fn parse_t_list(spec: &str) -> Result<Vec<u32>, String> {
    let mut out: Vec<u32> = Vec::new();
    for part in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        match part {
            "all" => out.extend(1..=MAX_T),
            "priority" => out.extend((1..=8).chain((12..=32).step_by(4))),
            _ => {
                let num = |s: &str| {
                    s.trim()
                        .parse::<u32>()
                        .map_err(|e| format!("bad T {s:?}: {e}"))
                };
                if let Some((a, b)) = part.split_once("..") {
                    let (a, b) = (num(a)?, num(b)?);
                    if a > b {
                        return Err(format!("empty range {part}"));
                    }
                    out.extend(a..=b);
                } else {
                    out.push(num(part)?);
                }
            }
        }
    }
    if out.is_empty() {
        return Err("empty T list".to_string());
    }
    if let Some(bad) = out.iter().find(|t| **t == 0 || **t > MAX_T) {
        return Err(format!("T={bad} is outside 1..={MAX_T}"));
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

/// Priority marks: `D` for decode widths 1 to 8, `M` for the MTP3 widths
/// 4, 8, ..., 32 (C = 1 to 8 streams of 4 rows each).
pub fn marks(t: u32) -> String {
    let mut s = String::new();
    if (1..=8).contains(&t) {
        s.push('D');
    }
    if t.is_multiple_of(4) && t <= 32 {
        s.push('M');
    }
    s
}

/// `T * median(T=1) / median(T)`: how many single-row calls' worth of work one
/// call at `t` rows does for the price of its time. `t` means the weight is
/// read once for `t` rows; a bandwidth-bound route holds this near `t`.
pub fn t1_lin_x(t: u32, med1_us: f64, med_t_us: f64) -> f64 {
    f64::from(t) * med1_us / med_t_us
}

/// Bytes of weights one launch reads once: the logical traffic every GB/s
/// figure in the sweep divides by.
pub fn weight_bytes(f: Family, n: u32, k: u32) -> usize {
    let (n, k) = (n as usize, k as usize);
    match f {
        Family::Bf16 => n * k * 2,
        // E4M3 bytes plus the FP32 128 x 128 block-scale grid.
        Family::Fp8 => n * k + n.div_ceil(128) * k.div_ceil(128) * 4,
        // Two E2M1 nibbles per byte plus one E4M3 scale per 16 values.
        Family::Nvfp4 => n * k / 2 + n * (k / 16),
    }
}

/// The route Atlas dispatches today at `t` rows (informational: the sweep
/// reports how far it sits from the best measured route).
///
/// BF16: `dense_gemv_bf16` at 1, `dense_gemv_bf16_batchm` to its MAX_M of 16
/// (`kernels/gb10/common/dense_gemv_bf16_batchm.cu`), cuBLASLt above
/// (`ffn_bf16_cublas_prefill`, `kernels/ada/HARDWARE.toml`).
/// FP8: `fp8_gemv_ada` at 1, `w8a16_gemv_batch4` to 4, `w8a16_gemv_batch16` to
/// 16, the W8A8 prefill GEMM above. NVFP4: `w4a16_gemv`, the fixed batch2 and
/// batch3 kernels, the exact-M tiers 4 to 8 (width 8 resolves to the rt2 twin,
/// `layers/gemv_tier.rs`), then batch16 and batch32.
pub fn ladder_route(f: Family, t: u32) -> Option<String> {
    let id = match f {
        Family::Bf16 => match t {
            1 => "bf16_gemv",
            2..=16 => "bf16_batchm",
            _ => "bf16_cublaslt",
        },
        Family::Fp8 => match t {
            1 => "fp8_gemv_ada",
            2..=4 => "fp8_batch4",
            5..=16 => "fp8_batch16",
            _ => "fp8_w8a8",
        },
        Family::Nvfp4 => match t {
            1 => "nvfp4_gemv",
            2..=7 => return Some(format!("nvfp4_batch{t}")),
            8 => "nvfp4_batch8_rt2",
            9..=16 => "nvfp4_batch16",
            17..=32 => "nvfp4_batch32",
            _ => return None,
        },
    };
    Some(id.to_string())
}

/// One measured (shape, T, route) cell.
#[derive(Clone, Debug)]
pub struct Row {
    pub shape: String,
    pub n: u32,
    pub k: u32,
    pub t: u32,
    pub route: String,
    pub family: Family,
    pub median_us: f64,
    pub spread_pct: f64,
    pub gbs: f64,
    pub pct_peak: f64,
    pub t1_lin_x: f64,
    pub temp: String,
}

pub fn table_header() -> String {
    format!(
        "{:<9} {:>3} {:<3} {:<18} {:>10} {:>6} {:>8} {:>6} {:>8}  {}",
        "shape", "T", "mk", "route", "median_us", "spr%", "GB/s", "%peak", "T1_lin_x", "temp"
    )
}

pub fn table_row(r: &Row) -> String {
    format!(
        "{:<9} {:>3} {:<3} {:<18} {:>10.1} {:>6.1} {:>8.1} {:>6.1} {:>8.2}  {}",
        r.shape,
        r.t,
        marks(r.t),
        r.route,
        r.median_us,
        r.spread_pct,
        r.gbs,
        r.pct_peak,
        r.t1_lin_x,
        r.temp
    )
}

pub fn row_json(r: &Row) -> Value {
    json!({
        "shape": r.shape, "n": r.n, "k": r.k, "t": r.t, "route": r.route,
        "family": r.family.name(), "median_us": r.median_us, "spread_pct": r.spread_pct,
        "logical_gbs": r.gbs, "pct_of_measured_peak": r.pct_peak,
        "t1_lin_x": r.t1_lin_x, "marks": marks(r.t), "temperature": r.temp,
    })
}

/// Best route among one (shape, T)'s rows, and how the ladder compares.
#[derive(Clone, Debug, PartialEq)]
pub struct Summary {
    pub best: String,
    pub best_us: f64,
    pub ladder: Option<String>,
    pub ladder_us: Option<f64>,
}

/// Summarize the rows of one (shape, T): the fastest route and, for each
/// family present, the ladder route's time. `family` picks the ladder.
pub fn summarize(rows: &[Row], family: Family) -> Option<Summary> {
    let best = rows
        .iter()
        .min_by(|a, b| a.median_us.total_cmp(&b.median_us))?;
    let t = rows.first()?.t;
    let ladder = ladder_route(family, t);
    let ladder_us = ladder
        .as_ref()
        .and_then(|l| rows.iter().find(|r| &r.route == l))
        .map(|r| r.median_us);
    Some(Summary {
        best: best.route.clone(),
        best_us: best.median_us,
        ladder,
        ladder_us,
    })
}

/// How much slower the ladder route is than the best, in percent.
pub fn ladder_gap_pct(ladder_us: f64, best_us: f64) -> f64 {
    (ladder_us / best_us - 1.0) * 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_lists() {
        assert_eq!(parse_t_list("all").unwrap().len(), 128);
        assert_eq!(parse_t_list("1..4,8").unwrap(), vec![1, 2, 3, 4, 8]);
        assert_eq!(parse_t_list("8, 2,8").unwrap(), vec![2, 8]);
        assert_eq!(
            parse_t_list("priority").unwrap(),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 28, 32]
        );
        assert!(parse_t_list("0").is_err());
        assert!(parse_t_list("129").is_err());
        assert!(parse_t_list("5..2").is_err());
        assert!(parse_t_list("").is_err());
        assert!(parse_t_list("x").is_err());
    }

    #[test]
    fn priority_marks() {
        assert_eq!(marks(1), "D");
        assert_eq!(marks(4), "DM");
        assert_eq!(marks(8), "DM");
        assert_eq!(marks(12), "M");
        assert_eq!(marks(32), "M");
        assert_eq!(marks(33), "");
        assert_eq!(marks(64), "");
    }

    #[test]
    fn t1_lin_x_is_t_when_cost_is_flat() {
        assert!((t1_lin_x(8, 100.0, 100.0) - 8.0).abs() < 1e-12);
        // Cost growing linearly with T (nothing shared) gives exactly 1.
        assert!((t1_lin_x(8, 100.0, 800.0) - 1.0).abs() < 1e-12);
        assert!((t1_lin_x(1, 100.0, 100.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn weight_bytes_of_the_ffn_up_shape() {
        assert_eq!(weight_bytes(Family::Bf16, 9216, 2560), 47_185_920);
        assert_eq!(
            weight_bytes(Family::Fp8, 9216, 2560),
            23_592_960 + 72 * 20 * 4
        );
        assert_eq!(
            weight_bytes(Family::Nvfp4, 9216, 2560),
            11_796_480 + 9216 * 160
        );
    }

    #[test]
    fn the_ladder_follows_the_dispatch_rules() {
        let l = |f, t| ladder_route(f, t).unwrap();
        assert_eq!(l(Family::Bf16, 1), "bf16_gemv");
        assert_eq!(l(Family::Bf16, 16), "bf16_batchm");
        assert_eq!(l(Family::Bf16, 17), "bf16_cublaslt");
        assert_eq!(l(Family::Fp8, 4), "fp8_batch4");
        assert_eq!(l(Family::Fp8, 5), "fp8_batch16");
        assert_eq!(l(Family::Fp8, 64), "fp8_w8a8");
        assert_eq!(l(Family::Nvfp4, 3), "nvfp4_batch3");
        assert_eq!(l(Family::Nvfp4, 6), "nvfp4_batch6");
        assert_eq!(l(Family::Nvfp4, 8), "nvfp4_batch8_rt2");
        assert_eq!(l(Family::Nvfp4, 20), "nvfp4_batch32");
        assert_eq!(ladder_route(Family::Nvfp4, 33), None);
    }

    fn row(route: &str, t: u32, us: f64) -> Row {
        Row {
            shape: "ffn_up".into(),
            n: 9216,
            k: 2560,
            t,
            route: route.into(),
            family: Family::Fp8,
            median_us: us,
            spread_pct: 1.0,
            gbs: 100.0,
            pct_peak: 40.0,
            t1_lin_x: 1.0,
            temp: "cold".into(),
        }
    }

    #[test]
    fn summary_finds_the_best_and_the_ladder() {
        let rows = vec![row("fp8_batch4", 4, 120.0), row("fp8_batch16", 4, 100.0)];
        let s = summarize(&rows, Family::Fp8).unwrap();
        assert_eq!(s.best, "fp8_batch16");
        assert_eq!(s.ladder.as_deref(), Some("fp8_batch4"));
        assert_eq!(s.ladder_us, Some(120.0));
        assert!((ladder_gap_pct(120.0, 100.0) - 20.0).abs() < 1e-9);
        assert!(summarize(&[], Family::Fp8).is_none());
    }

    #[test]
    fn rows_render_with_the_marks() {
        let r = row("fp8_batch4", 4, 120.0);
        assert!(table_row(&r).contains("DM"));
        assert_eq!(row_json(&r)["route"], "fp8_batch4");
        assert!(table_header().contains("T1_lin_x"));
    }
}
