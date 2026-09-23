// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::HashSet;

use super::WeightStore;
use crate::gpu::GpuBackend;

/// Release every weight tensor.
///
/// Loaders normally allocate once per map entry. Kimi GGUF AttnRes/MLPRes
/// aliases insert the same device pointer under `*_res_proj` and `*_res_norm`;
/// free each unique pointer once.
impl avarok_core::scope::ModelResource<dyn GpuBackend> for WeightStore {
    fn label(&self) -> &'static str {
        "weight store"
    }

    fn release(&mut self, gpu: &dyn GpuBackend) -> anyhow::Result<()> {
        let mut first_error = self.derived.release(gpu).err();
        let mut freed = HashSet::new();
        for (name, tensor) in self.weights.drain() {
            if !freed.insert(tensor.ptr.0) {
                continue;
            }
            if let Err(e) = gpu.free(tensor.ptr)
                && first_error.is_none()
            {
                first_error = Some(e.context(format!("freeing weight {name}")));
            }
        }
        match first_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}
