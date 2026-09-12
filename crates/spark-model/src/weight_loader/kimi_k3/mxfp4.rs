// SPDX-License-Identifier: AGPL-3.0-only

//! Thin wrapper: K3 `weight_packed` + `weight_scale` → DSV4 MXFP4 lander.
//!
//! GPU GEMM is not this slice. `refuse_mxfp4` still gates `load_*`.
//! TODO(S5 GPU): dispatch `moe_w4a16_grouped_gemm_ptrtable_e8m0`.

use anyhow::Result;
use spark_runtime::weights::WeightStore;

use crate::weight_map::{QuantizedWeight, quantized_mxfp4_e8m0_pair};

/// Land official K3 expert keys on the DSV4 transcode-free E8M0 path.
#[allow(dead_code)]
pub(crate) fn quantized_k3_mxfp4_e8m0(
    store: &WeightStore,
    prefix: &str,
) -> Result<QuantizedWeight> {
    quantized_mxfp4_e8m0_pair(
        store,
        &format!("{prefix}.weight_packed"),
        &format!("{prefix}.weight_scale"),
    )
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
