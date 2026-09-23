// SPDX-License-Identifier: AGPL-3.0-only
//! Resident BF16 GEMV via `k3_dense_down_f32io`. Activations FP32.
//! Q8→BF16 weights stay on the card. No OnceLock D2H for these.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
use spark_runtime::weights::WeightDtype;

use super::bound::K3BoundLayer;

const MODULE: &str = "dense_f32io";
const ENTRY: &str = "k3_dense_down_f32io";

fn dtype_tag(dtype: WeightDtype) -> Result<u32> {
    match dtype {
        WeightDtype::FP32 => Ok(0),
        WeightDtype::BF16 => Ok(1),
        other => anyhow::bail!("K3 gpu_gemv: unsupported {other:?}"),
    }
}

/// Tiny tensors that stay on the host OnceLock. Everything else BF16 stays GPU.
pub fn host_keep_bf16(name: &str) -> bool {
    name.contains("conv1d")
        || name.ends_with(".A_log")
        || name.contains("dt_bias")
        || name.contains("layernorm")
        || name.ends_with(".norm.weight")
        || name.contains("res_proj")
        || name.contains("res_norm")
        || name.contains("o_norm")
        || name.contains("e_score_correction")
        || name.contains("routed_expert_norm")
        || name.contains("k_b_proj")
        || name.contains("v_b_proj")
}

fn suffix_for(op: &str) -> Result<&'static str> {
    Ok(match op {
        "q_proj" => "self_attn.q_proj.weight",
        "k_proj" => "self_attn.k_proj.weight",
        "v_proj" => "self_attn.v_proj.weight",
        "o_proj" => "self_attn.o_proj.weight",
        "g_proj" => "self_attn.g_proj.weight",
        "f_a_proj" => "self_attn.f_a_proj.weight",
        "f_b_proj" => "self_attn.f_b_proj.weight",
        "b_proj" => "self_attn.b_proj.weight",
        "q_a_proj" => "self_attn.q_a_proj.weight",
        "q_b_proj" => "self_attn.q_b_proj.weight",
        "kv_a_proj" => "self_attn.kv_a_proj_with_mqa.weight",
        "routed_down" => "block_sparse_moe.routed_expert_down_proj.weight",
        "routed_up" => "block_sparse_moe.routed_expert_up_proj.weight",
        "router" => "block_sparse_moe.gate.weight",
        other => anyhow::bail!("K3 gpu_gemv: unknown op {other}"),
    })
}

fn find(layer: &K3BoundLayer, suffix: &str) -> Result<(DevicePtr, WeightDtype, usize, String)> {
    layer
        .weights
        .iter()
        .zip(&layer.weight_meta)
        .find(|(_, m)| m.name.ends_with(suffix))
        .map(|(w, m)| (w.weight, m.dtype, m.numel, m.name.clone()))
        .with_context(|| format!("K3 gpu_gemv: missing {suffix}"))
}

/// `y[n] = W[n, k] @ x[k]` on resident BF16/FP32.
pub fn launch(
    layer: &K3BoundLayer,
    gpu: &dyn GpuBackend,
    op: &str,
    x: &[f32],
    n: usize,
    k: usize,
    stream: u64,
) -> Result<Vec<f32>> {
    ensure!(
        n > 0 && k > 0 && x.len() == k,
        "K3 gpu_gemv {op}: x {} vs k {k}",
        x.len()
    );
    tracing::debug!(op, n, k, "K3 gpu gemv");
    let suffix = suffix_for(op)?;
    let (ptr, dtype, numel, name) = find(layer, suffix)?;
    ensure!(
        numel == n.checked_mul(k).context("gemv numel")?,
        "K3 gpu_gemv {name}: on-rank {numel} vs op {n}×{k}"
    );
    reject_replicated_routed_down(op, n, k)?;
    gemv_raw(gpu, ptr, dtype, x, n, k, stream)
}

/// Oracle: the H200 OOM shape. Known-bad is full-hidden down after Q8→BF16.
pub(crate) fn reject_replicated_routed_down(op: &str, n: usize, k: usize) -> Result<()> {
    if op == "routed_down" {
        ensure!(
            !(n == 3584 && k == 7168),
            "K3 gpu_gemv routed_down must not be 3584×7168 on-rank (hidden/tp=896, not full hidden)"
        );
    }
    Ok(())
}

pub fn gemv_raw(
    gpu: &dyn GpuBackend,
    w: DevicePtr,
    dtype: WeightDtype,
    x: &[f32],
    n: usize,
    k: usize,
    stream: u64,
) -> Result<Vec<f32>> {
    let kernel = gpu.kernel(MODULE, ENTRY)?;
    let tag = dtype_tag(dtype)?;
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let xin = gpu.alloc(x_bytes.len())?;
    let y_out = gpu.alloc(n * 4)?;
    let result = (|| -> Result<Vec<f32>> {
        gpu.copy_h2d(&x_bytes, xin)?;
        KernelLaunch::new(gpu, kernel)
            .grid([div_ceil(n as u32, 4), 1, 1])
            .block([128, 1, 1])
            .arg_ptr(xin)
            .arg_ptr(w)
            .arg_ptr(y_out)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .arg_u32(tag)
            .launch(stream)?;
        gpu.synchronize(stream)?;
        let mut raw = vec![0u8; n * 4];
        gpu.copy_d2h(y_out, &mut raw)?;
        Ok(raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    })();
    let _ = gpu.free(y_out);
    let _ = gpu.free(xin);
    let out = result?;
    ensure!(out.iter().all(|v| v.is_finite()), "K3 gpu_gemv nonfinite");
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_keep_is_the_replicated_tiny_set() {
        assert!(host_keep_bf16("model.layers.1.self_attn.q_conv1d.weight"));
        assert!(host_keep_bf16("model.layers.1.self_attn.A_log"));
        assert!(host_keep_bf16("model.layers.1.self_attn.k_b_proj.weight"));
        assert!(!host_keep_bf16(
            "model.layers.1.block_sparse_moe.routed_expert_down_proj.weight"
        ));
        assert!(!host_keep_bf16("model.layers.1.self_attn.q_proj.weight"));
        assert!(!host_keep_bf16(
            "model.layers.1.block_sparse_moe.shared_experts.down_proj.weight"
        ));
    }

    #[test]
    fn routed_down_suffix_is_not_full_hidden() {
        assert_eq!(
            suffix_for("routed_down").unwrap(),
            "block_sparse_moe.routed_expert_down_proj.weight"
        );
    }

    #[test]
    fn replicated_routed_down_3584x7168_is_rejected() {
        let err = reject_replicated_routed_down("routed_down", 3584, 7168).unwrap_err();
        assert!(format!("{err}").contains("3584"));
        reject_replicated_routed_down("routed_down", 3584, 896).unwrap();
        reject_replicated_routed_down("q_proj", 3584, 7168).unwrap();
    }
}
