// SPDX-License-Identifier: AGPL-3.0-only

//! Missing optional MMQ kernels must leave the W4A16 fallback usable at load.

use super::{DenseFfnLayer, DenseFfnWeights};
use crate::weight_map::QuantizedWeight;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{GpuBackend, KernelHandle};

#[test]
fn unavailable_mmq_preserves_transposed_weights_without_repack_or_free() {
    let gpu = MockGpuBackend::new();
    let mut weight = QuantizedWeight::null();
    weight.weight = gpu.alloc(64).unwrap();
    weight.weight_scale = gpu.alloc(8).unwrap();
    let weights = DenseFfnWeights {
        gate_proj: weight,
        up_proj: weight,
        down_proj: weight,
        gate_proj_t: Some(weight),
        up_proj_t: Some(weight),
        down_proj_t: Some(weight),
    };
    let mut layer = DenseFfnLayer::new(weights, &gpu).unwrap();
    // try_kernel returns this handle when the capability-guarded entry point
    // is absent. Other handles remain present: one missing required operation
    // must disable the entire MMQ load path, including freeing fallback twins.
    layer.nvfp4_mmq_nc_k = KernelHandle(0);
    let allocations = gpu.alloc_count();
    let launches = gpu.launch_count();
    layer.finalize_nvfp4_mmq_load(&gpu, 64, 64, 0).unwrap();
    assert_eq!(gpu.alloc_count(), allocations);
    assert_eq!(gpu.launch_count(), launches);
    assert_eq!(gpu.sync_count(), 0);
    for transpose in [
        layer.weights.gate_proj_t,
        layer.weights.up_proj_t,
        layer.weights.down_proj_t,
    ] {
        assert_eq!(transpose.unwrap().weight, weight.weight);
    }
    assert!(layer.fp4mmq_gate.get().is_none());
    assert!(layer.fp4mmq_up.get().is_none());
    assert!(layer.fp4mmq_down.get().is_none());
}
