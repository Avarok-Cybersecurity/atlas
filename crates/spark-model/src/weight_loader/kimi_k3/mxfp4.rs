// SPDX-License-Identifier: AGPL-3.0-only

//! Thin wrapper: K3 `weight_packed` + `weight_scale` → DSV4 MXFP4 lander.
//!
//! GPU GEMM is DSV4 `moe_w4a16_grouped_gemm_ptrtable_e8m0` via KERNEL.toml
//! a same-hardware source alias. Default refuse stays unless `K3_ALLOW_MXFP4=1`.

use anyhow::{Result, ensure};
use avarok_core::config::ModelConfig;
use avarok_core::kimi_k3::{
    MixerKind,
    tp::{plan_tensor_bytes, tensor_plan},
};
use avarok_core::mxfp4_e8m0::GROUP_SIZE;
use spark_runtime::weights::WeightDtype;
use spark_runtime::weights::WeightStore;

use crate::weight_map::{QuantizedWeight, quantized_mxfp4_e8m0_pair};

/// Land official K3 expert keys on the DSV4 transcode-free E8M0 path.
pub(super) fn quantized_k3_mxfp4_e8m0(
    store: &WeightStore,
    prefix: &str,
) -> Result<QuantizedWeight> {
    quantized_mxfp4_e8m0_pair(
        store,
        &format!("{prefix}.weight_packed"),
        &format!("{prefix}.weight_scale"),
    )
}

/// Check the packed/scales pair against the rank-local production geometry.
pub(super) fn validate_packed_partition(
    store: &WeightStore,
    prefix: &str,
    config: &ModelConfig,
) -> Result<()> {
    let marked = super::tp::is_prepartitioned(store, config)?;
    ensure!(
        config.tp_world_size <= 1 || marked,
        "K3 TP does not slice packed MXFP4 after upload; use the rank-aware checkpoint loader"
    );
    let weight_key = format!("{prefix}.weight_packed");
    let scale_key = format!("{prefix}.weight_scale");
    let (_, n, k) = tensor_plan(&weight_key, MixerKind::Kda, config);
    let wp = plan_tensor_bytes(&weight_key, &[n, k / 2], 1, MixerKind::Kda, config)?;
    let sp = plan_tensor_bytes(&scale_key, &[n, k / GROUP_SIZE], 1, MixerKind::Kda, config)?;
    let weight = store.get(&weight_key)?;
    let scale = store.get(&scale_key)?;
    ensure!(
        weight.dtype == WeightDtype::UInt8,
        "{weight_key}: expected packed U8, got {:?}",
        weight.dtype
    );
    ensure!(
        matches!(scale.dtype, WeightDtype::UInt8 | WeightDtype::FP8E8M0),
        "{scale_key}: expected E8M0 byte storage, got {:?}",
        scale.dtype
    );
    ensure!(
        weight.shape == wp.local_shape && scale.shape == sp.local_shape,
        "{prefix}: packed/scales shape {:?}/{:?} does not match TP-local {:?}/{:?}",
        weight.shape,
        scale.shape,
        wp.local_shape,
        sp.local_shape
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::GpuBackend;
    use spark_runtime::gpu::mock::MockGpuBackend;
    use spark_runtime::weights::{WeightDtype, WeightTensor};
    use std::collections::HashMap;

    #[test]
    fn k3_packed_names_land_on_dsv4_e8m0_pair() {
        let gpu = MockGpuBackend::new();
        let packed = gpu.alloc(16).unwrap();
        let scale = gpu.alloc(1).unwrap();
        let store = WeightStore::from_map(HashMap::from([
            (
                "experts.0.w1.weight_packed".to_string(),
                WeightTensor {
                    ptr: packed,
                    shape: vec![1, 16],
                    dtype: WeightDtype::UInt8,
                },
            ),
            (
                "experts.0.w1.weight_scale".to_string(),
                WeightTensor {
                    ptr: scale,
                    shape: vec![1],
                    dtype: WeightDtype::UInt8,
                },
            ),
        ]));
        let qw = quantized_k3_mxfp4_e8m0(&store, "experts.0.w1").unwrap();
        assert_eq!(qw.weight, packed);
        assert_eq!(qw.weight_scale, scale);
        assert_eq!(qw.weight_scale_2, 1.0);
    }
}

#[cfg(test)]
#[path = "mxfp4_tests.rs"]
mod partition_tests;
