// SPDX-License-Identifier: AGPL-3.0-only
// Portions derived from rayline-ai/vllm-arc (vllm_arc/arc_heads.py, commit 4ee6a9f), Copyright 2026 Atlas Futures,
// Inc., licensed under the Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0). Translated from Python
// (torch, then NumPy float64) to Rust f64; each function is one function of arc_heads.py, applied to one row at a time
// instead of over padded batches. By Atlas contributors. See RAYLINE-ROUTING-ATTRIBUTION.md.
//! Responsibility: the building blocks of the Stage-1 usage orderer and the Platt map (searchsorted, interpolation,
//! linear quantile, order ratio, Platt), each tested on its own against NumPy's definitions.

use crate::route_gate::error::{GResult, value_err};
use crate::route_gate::mat::{logaddexp, np_clip, np_max};

/// The usage orderer's thinnest usable pool and its token floor (arc_heads.py:14-16).
pub const MIN_POOL: usize = 20;
pub const FLOOR_TOKENS: f64 = 1.0;
/// A Platt map's clip of p away from 0 and 1 (arc_heads.py:20).
pub const PLATT_EPS: f64 = 1e-6;
/// The clamp on the Stage-1 logit before the sigmoid.
pub const ETA_CLAMP: f64 = 40.0;

/// Which side `searchsorted` returns for ties.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
}

/// NumPy's sort order: NaN is greater than every number.
fn lt(a: f64, b: f64) -> bool {
    a < b || (b.is_nan() && !a.is_nan())
}

/// `numpy.searchsorted(row, v, side)` for one value (the whole padded row is searched, padding is not trimmed).
pub fn search_sorted(row: &[f64], v: f64, side: Side) -> usize {
    match side {
        Side::Left => row.partition_point(|&x| lt(x, v)),
        Side::Right => row.partition_point(|&x| !lt(v, x)),
    }
}

/// `torch.searchsorted` over matching leading dimensions: `rows` is `lead_rows x p` (each row sorted), `values` is
/// `lead_vals x k`; the result is `lead_vals x k`. The leading counts must agree.
pub fn batched_searchsorted(
    rows: &[f64],
    lead_rows: usize,
    p: usize,
    values: &[f64],
    lead_vals: usize,
    k: usize,
    side: Side,
) -> GResult<Vec<usize>> {
    if lead_rows != lead_vals {
        return value_err(format!(
            "searchsorted leading dims differ: {lead_rows} rows vs {lead_vals} value rows"
        ));
    }
    let mut out = Vec::with_capacity(values.len());
    for i in 0..lead_rows {
        let row = &rows[i * p..(i + 1) * p];
        for &v in &values[i * k..(i + 1) * k] {
            out.push(search_sorted(row, v, side));
        }
    }
    Ok(out)
}

/// `numpy.interp(x, xp[:n], fp[:n])` with constant tails (arc_heads._interp); `xp` and `fp` may be longer than `n`.
pub fn interp_row(x: f64, xp: &[f64], fp: &[f64], n: usize) -> f64 {
    if n <= 1 {
        return fp[0];
    }
    let last = n - 1;
    let right = search_sorted(xp, x, Side::Right);
    let j = right.saturating_sub(1).min(last.saturating_sub(1));
    let j1 = (j + 1).min(last);
    let (x0, x1, f0, f1) = (xp[j], xp[j1], fp[j], fp[j1]);
    let span = if x1 > x0 { x1 - x0 } else { 1.0 };
    let inner = (f1 - f0) / span * (x - x0) + f0;
    if x < xp[0] {
        fp[0]
    } else if x >= xp[last] {
        fp[last]
    } else {
        inner
    }
}

/// NumPy's `quantile(method="linear")` of `pool[..n]` at `q`: virtual index `(n - 1) q`, its floor and the next index
/// clipped into the pool, then NumPy's two-sided lerp (arc_heads._linear_quantile).
pub fn linear_quantile_row(pool: &[f64], n: usize, q: f64) -> f64 {
    let last = n.saturating_sub(1);
    let virtual_index = (n as f64 - 1.0) * q;
    let below = virtual_index.floor();
    let previous = (np_max(below, 0.0) as usize).min(last);
    let following = (previous + 1).min(last);
    let (a, b) = (pool[previous], pool[following]);
    let gamma = virtual_index - below;
    let diff = b - a;
    if gamma >= 0.5 {
        b - diff * (1.0 - gamma)
    } else {
        a + diff * gamma
    }
}

/// The usage orderer's ratio for one `(glm, net)` pair against its candidate-leg pools: 1 where either value is
/// non-finite or the pool is shorter than [`MIN_POOL`] (arc_heads.order_ratio).
pub fn order_ratio_one(
    glm: f64,
    net: f64,
    pool_glm: &[f64],
    pool_net: &[f64],
    pool_n: usize,
    alpha: f64,
) -> f64 {
    let finite = glm.is_finite() && net.is_finite();
    let count = pool_n.max(1) as f64;
    let u = search_sorted(pool_net, if finite { net } else { 0.0 }, Side::Right) as f64 / count;
    let v = search_sorted(pool_glm, if finite { glm } else { 0.0 }, Side::Right) as f64 / count;
    let top = np_max(
        linear_quantile_row(pool_glm, pool_n, np_clip(u, 0.0, 1.0)),
        FLOOR_TOKENS,
    );
    let bottom = np_max(
        linear_quantile_row(pool_glm, pool_n, np_clip(v, 0.0, 1.0)),
        FLOOR_TOKENS,
    );
    let thick = pool_n >= MIN_POOL;
    if finite && thick {
        (top / bottom).powf(alpha)
    } else {
        1.0
    }
}

/// `sigmoid(a logit(clip(p)) + b)` where `on` is set and `p` is finite, else `p` unchanged (arc_heads._platt).
pub fn platt(p: f64, on: bool, a: f64, b: f64) -> f64 {
    if !on || !p.is_finite() {
        return p;
    }
    let q = np_clip(p, PLATT_EPS, 1.0 - PLATT_EPS);
    let z = a * (q / (1.0 - q)).ln() + b;
    (-logaddexp(0.0, -z)).exp()
}
