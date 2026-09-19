// SPDX-License-Identifier: AGPL-3.0-only

//! Thin wrapper: K3 `weight_packed` + `weight_scale` → DSV4 MXFP4 lander.
//!
//! GPU GEMM is DSV4 `moe_w4a16_grouped_gemm_ptrtable_e8m0` via KERNEL.toml
//! `extra_cu`. Default refuse stays unless `K3_ALLOW_MXFP4=1`.

use anyhow::Result;
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

    #[test]
    fn k3_extra_cu_gemm_declares_e8m0_ptrtable() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let dsv4 = root.join("kernels/gb10/deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu");
        let src =
            std::fs::read_to_string(&dsv4).unwrap_or_else(|e| panic!("{}: {e}", dsv4.display()));
        assert!(
            src.contains("extern \"C\" __global__ void moe_w4a16_grouped_gemm_ptrtable_e8m0("),
            "DSV4 extra_cu source must declare the E8M0 ptrtable GEMM"
        );
        for quant in ["mxfp4", "nvfp4"] {
            let toml = root.join(format!("kernels/gb10/kimi-k3/{quant}/KERNEL.toml"));
            let text = std::fs::read_to_string(&toml)
                .unwrap_or_else(|e| panic!("{}: {e}", toml.display()));
            assert!(
                text.contains("extra_cu")
                    && text.contains("deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu"),
                "{quant} KERNEL.toml must extra_cu the DSV4 E8M0 grouped GEMM"
            );
            assert!(
                text.contains("moe_w4a16_grouped_gemm = \"moe_w4a16\""),
                "{quant} must map the source stem to its runtime module; entrypoints come from the CUDA source"
            );
            let extra = toml
                .parent()
                .unwrap()
                .join("../../deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu");
            assert!(
                extra.is_file(),
                "{quant} extra_cu path missing: {}",
                extra.display()
            );
        }
        assert!(
            root.join("kernels/gb10/kimi-k3/bf16/kda_decode.cu")
                .is_file(),
            "unique KDA stem must stay"
        );
        assert!(
            root.join("kernels/gb10/kimi-k3/nvfp4/kda_decode.cu")
                .exists(),
            "nvfp4 serve bundle must keep kda_decode"
        );
        assert!(
            !root
                .join("kernels/gb10/kimi-k3/mxfp4/kda_decode.cu")
                .exists()
                || root
                    .join("kernels/gb10/kimi-k3/mxfp4/kda_decode.cu")
                    .is_symlink(),
            "mxfp4 must not grow a second KDA copy"
        );
    }
}
