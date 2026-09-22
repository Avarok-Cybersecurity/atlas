// SPDX-License-Identifier: AGPL-3.0-only

//! Rank-local shape contract for Kimi-K3 TP=8 EP=1.
//!
//! `axis_for` is what the GGUF loader stores. `op_expected_numel` is what the
//! host matvec / kernel multiplies. They are written separately so a substring
//! slice (the `down_proj` bug) fails the comparison.
//!
//! 7168/8 and the expert count are both 896. Axes are named so that integer
//! cannot be used as either one.

#[cfg(test)]
use anyhow::{Result, ensure};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    Replicated,
    /// Divide dim 0 by tp. Head stacks and column-parallel N.
    SplitRows {
        why: &'static str,
    },
    /// Divide the last dim by tp. Row-parallel K, then all-reduce.
    SplitCols {
        why: &'static str,
    },
}

/// Loader rule. Checked against [`op_expected_numel`], not used by it.
pub fn axis_for(hf_name: &str) -> Axis {
    if replicated_name(hf_name) {
        return Axis::Replicated;
    }
    if splits_rows_name(hf_name) {
        return Axis::SplitRows {
            why: row_why(hf_name),
        };
    }
    if splits_cols_name(hf_name) {
        return Axis::SplitCols {
            why: col_why(hf_name),
        };
    }
    Axis::Replicated
}

fn replicated_name(name: &str) -> bool {
    name.contains("routed_expert_norm")
        || name.contains("q_a_proj")
        || name.contains("q_a_layernorm")
        || name.contains("kv_a_proj")
        || name.contains("kv_a_layernorm")
        || name.contains("f_a_proj")
        || name.ends_with("embed_tokens.weight")
        || name.ends_with("lm_head.weight")
        || name.ends_with(".norm.weight")
        || name.contains("layernorm.weight")
        || name.contains("gate.weight")
        || name.contains("e_score_correction")
        || name.contains("res_proj")
        || name.contains("res_norm")
        || name.contains("o_norm")
}

fn splits_rows_name(name: &str) -> bool {
    name.contains("k_b_proj")
        || name.contains("v_b_proj")
        || name.contains("q_b_proj")
        || name.contains("q_proj")
        || name.contains("k_proj")
        || name.contains("v_proj")
        || name.contains("g_proj")
        || name.contains("q_conv1d")
        || name.contains("k_conv1d")
        || name.contains("v_conv1d")
        || name.ends_with(".A_log")
        || name.contains("dt_bias")
        || name.contains("f_b_proj")
        || name.contains("b_proj")
        || name.contains("gate_proj")
        || name.contains("up_proj")
        || name.contains(".w1.weight")
        || name.contains(".w3.weight")
}

fn splits_cols_name(name: &str) -> bool {
    name.contains("o_proj") || name.contains("down_proj") || name.contains(".w2.weight")
}

fn row_why(name: &str) -> &'static str {
    if name.contains("routed_expert_up") {
        "hidden_over_tp"
    } else if name.contains("k_b_proj") || name.contains("v_b_proj") || name.contains("q_b_proj") {
        "heads"
    } else if name.ends_with(".A_log") || name.contains("dt_bias") || name.contains("conv1d") {
        "heads"
    } else if name.contains("b_proj") || name.contains("f_b_proj") {
        "heads"
    } else if name.contains("gate_proj")
        || name.contains("up_proj")
        || name.contains(".w1.")
        || name.contains(".w3.")
    {
        "out_features"
    } else {
        "head_concat"
    }
}

fn col_why(name: &str) -> &'static str {
    if name.contains("routed_expert_down") {
        "hidden_over_tp"
    } else if name.contains("o_proj") {
        "head_concat_then_allreduce"
    } else if name.contains(".w2.") {
        "expert_intermediate"
    } else {
        "in_features_then_allreduce"
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg(test)]
pub struct Contract {
    pub after_tp: Vec<usize>,
    pub numel_on_rank: usize,
    pub op_expected_numel: usize,
    pub tp_axis: String,
    pub replicated: bool,
}

#[cfg(test)]
impl Contract {
    pub fn matches(&self) -> bool {
        self.numel_on_rank == self.op_expected_numel && self.numel_on_rank > 0
    }
}

#[cfg(test)]
pub fn contract(hf_name: &str, hf_shape: &[usize], tp: usize) -> Result<Contract> {
    ensure!(tp > 0, "tp");
    let axis = axis_for(hf_name);
    let after = local_shape(hf_shape, axis, tp)?;
    let numel_on_rank = product(&after)?;
    let op_expected_numel = op_expected_numel(hf_name, hf_shape, tp)?;
    let (tp_axis, replicated) = match axis {
        Axis::Replicated => ("replicated".to_string(), true),
        Axis::SplitRows { why } => (format!("rows:{why}"), false),
        Axis::SplitCols { why } => (format!("cols:{why}"), false),
    };
    Ok(Contract {
        after_tp: after,
        numel_on_rank,
        op_expected_numel,
        tp_axis,
        replicated,
    })
}

/// Host/kernel numel. Does not call [`axis_for`].
#[cfg(test)]
pub fn op_expected_numel(hf_name: &str, hf_shape: &[usize], tp: usize) -> Result<usize> {
    ensure!(tp > 0 && !hf_shape.is_empty(), "{hf_name}: empty shape");
    if op_replicated(hf_name) {
        return product(hf_shape);
    }
    if op_split_cols(hf_name) {
        ensure!(
            hf_shape.len() == 2,
            "{hf_name}: col split wants 2D {hf_shape:?}"
        );
        let k = hf_shape[1];
        ensure!(
            k.is_multiple_of(tp),
            "{hf_name}: K {k} not divisible by tp {tp}"
        );
        return Ok(hf_shape[0] * (k / tp));
    }
    if op_split_rows(hf_name) {
        let n = hf_shape[0];
        ensure!(
            n.is_multiple_of(tp),
            "{hf_name}: N {n} not divisible by tp {tp}"
        );
        let rest = product(&hf_shape[1..])?;
        return Ok((n / tp) * rest);
    }
    product(hf_shape)
}

#[cfg(test)]
fn op_replicated(name: &str) -> bool {
    name.contains("routed_expert_norm")
        || name.contains("q_a_proj")
        || name.contains("q_a_layernorm")
        || name.contains("kv_a_proj")
        || name.contains("kv_a_layernorm")
        || name.contains("f_a_proj")
        || name.ends_with("embed_tokens.weight")
        || name.ends_with("lm_head.weight")
        || name.ends_with(".norm.weight")
        || name.contains("layernorm.weight")
        || name.contains("gate.weight")
        || name.contains("e_score_correction")
        || name.contains("res_proj")
        || name.contains("res_norm")
        || name.contains("o_norm")
}

#[cfg(test)]
fn op_split_rows(name: &str) -> bool {
    name.contains("k_b_proj")
        || name.contains("v_b_proj")
        || name.contains("q_b_proj")
        || name.contains("q_proj")
        || name.contains("k_proj")
        || name.contains("v_proj")
        || name.contains("g_proj")
        || name.contains("q_conv1d")
        || name.contains("k_conv1d")
        || name.contains("v_conv1d")
        || name.ends_with(".A_log")
        || name.contains("dt_bias")
        || name.contains("f_b_proj")
        || name.contains("b_proj")
        || name.contains("gate_proj")
        || name.contains("up_proj")
        || name.contains(".w1.weight")
        || name.contains(".w3.weight")
}

#[cfg(test)]
fn op_split_cols(name: &str) -> bool {
    name.contains("o_proj") || name.contains("down_proj") || name.contains(".w2.weight")
}

#[cfg(test)]
pub fn local_shape(shape: &[usize], axis: Axis, tp: usize) -> Result<Vec<usize>> {
    match axis {
        Axis::Replicated => Ok(shape.to_vec()),
        Axis::SplitRows { .. } => {
            ensure!(!shape.is_empty(), "row split");
            ensure!(
                shape[0].is_multiple_of(tp),
                "row split {} not divisible by tp {tp}",
                shape[0]
            );
            let mut out = shape.to_vec();
            out[0] /= tp;
            Ok(out)
        }
        Axis::SplitCols { .. } => {
            ensure!(shape.len() == 2, "col split wants 2D, got {shape:?}");
            let k = shape[1];
            ensure!(
                k.is_multiple_of(tp),
                "col split {k} not divisible by tp {tp}"
            );
            Ok(vec![shape[0], k / tp])
        }
    }
}

#[cfg(test)]
fn product(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |n, &d| {
        n.checked_mul(d)
            .ok_or_else(|| anyhow::anyhow!("numel overflow"))
    })
}

/// Routed expert stack. `count` is the expert axis and is never divided.
/// w1/w3 split `out` (3072). w2 splits `inn` (3072). Latent 3584 stays whole.
#[cfg(test)]
pub fn contract_expert(
    proj: &str,
    count: usize,
    out: usize,
    inn: usize,
    tp: usize,
) -> Result<Contract> {
    ensure!(tp > 0 && count > 0, "expert stack");
    let (local_out, local_in, tp_axis) = match proj {
        "w1" | "w3" | "gate" | "up" => {
            ensure!(out.is_multiple_of(tp), "w1/w3 out {out} / tp {tp}");
            (out / tp, inn, "rows:expert_intermediate")
        }
        "w2" | "down" => {
            ensure!(inn.is_multiple_of(tp), "w2 inn {inn} / tp {tp}");
            (out, inn / tp, "cols:expert_intermediate")
        }
        other => anyhow::bail!("unknown expert proj {other}"),
    };
    let numel = count
        .checked_mul(local_out)
        .and_then(|n| n.checked_mul(local_in))
        .ok_or_else(|| anyhow::anyhow!("expert numel overflow"))?;
    let op = expert_op_numel(proj, count, out, inn, tp)?;
    Ok(Contract {
        after_tp: vec![count, local_out, local_in],
        numel_on_rank: numel,
        op_expected_numel: op,
        tp_axis: tp_axis.to_string(),
        replicated: false,
    })
}

#[cfg(test)]
fn expert_op_numel(proj: &str, count: usize, out: usize, inn: usize, tp: usize) -> Result<usize> {
    let (lo, li) = match proj {
        "w1" | "w3" | "gate" | "up" => (out / tp, inn),
        "w2" | "down" => (out, inn / tp),
        other => anyhow::bail!("unknown expert proj {other}"),
    };
    Ok(count * lo * li)
}

#[cfg(test)]
pub fn format_row(name: &str, ggml_shape: &[usize], row: &Contract) -> String {
    format!(
        "{name} | {ggml_shape:?} | {:?} | {} | {} | {} | {}",
        row.after_tp,
        row.numel_on_rank,
        row.op_expected_numel,
        row.tp_axis,
        if row.replicated { "yes" } else { "no" }
    )
}

#[cfg(test)]
#[path = "kimi_tp_contract_tests.rs"]
mod tests;
