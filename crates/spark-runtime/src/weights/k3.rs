// SPDX-License-Identifier: AGPL-3.0-only
//! Rank-aware K3 checkpoint I/O. Plans every tensor before GPU allocation;
//! uploads only rank-local bytes. Other model loaders are unchanged.
use super::{WeightLoader, WeightStore, WeightTensor, evict_page_cache};
use crate::gpu::GpuBackend;
use anyhow::{Context, Result, ensure};
use avarok_core::config::ModelConfig;
use avarok_core::kimi_k3::tp::TpAxis;
use std::path::Path;

mod index;
mod inventory;
#[cfg(test)]
mod tests;

pub struct K3SafetensorsLoader {
    config: ModelConfig,
}
impl K3SafetensorsLoader {
    pub fn new(config: ModelConfig) -> Result<Self> {
        ensure!(
            matches!(
                config.model_type.as_str(),
                "kimi_k3" | "kimi_linear" | "Kimi-K3"
            ),
            "K3 rank loader requires a K3 configuration"
        );
        ensure!(
            config.tp_world_size > 0 && config.tp_rank < config.tp_world_size,
            "invalid K3 TP topology"
        );
        Ok(Self { config })
    }
}
impl WeightStore {
    /// Set only after the rank-aware loader has validated and sliced all tensors.
    pub fn prepartitioned_tp(&self) -> Option<(usize, usize)> {
        self.prepartitioned_tp
    }
}

impl WeightLoader for K3SafetensorsLoader {
    fn load(&self, model_dir: &Path, gpu: &dyn GpuBackend, reserve: usize) -> Result<WeightStore> {
        let inventory = inventory::scan(model_dir, &self.config)?;
        let mut local_bytes = 0usize;
        let mut source_bytes = 0usize;
        let mut replicated_bytes = 0usize;
        let mut staging_peak = 0usize;
        let mut binding_extra_bytes = 0usize;
        for file in &inventory {
            for tensor in &file.tensors {
                local_bytes = local_bytes
                    .checked_add(tensor.plan.local_bytes())
                    .context("K3 local byte overflow")?;
                source_bytes = source_bytes
                    .checked_add(tensor.plan.source_bytes)
                    .context("K3 source byte overflow")?;
                if tensor.plan.axis == TpAxis::Replicated {
                    replicated_bytes = replicated_bytes
                        .checked_add(tensor.plan.local_bytes())
                        .context("K3 replicated byte overflow")?;
                }
                staging_peak = staging_peak.max(tensor.plan.local_bytes());
                binding_extra_bytes = binding_extra_bytes
                    .checked_add(avarok_core::kimi_k3::binding_memory::extra_gpu_bytes(
                        &tensor.name,
                        tensor.dtype == super::WeightDtype::FP32,
                        tensor.plan.local_bytes() / tensor.dtype.byte_size(),
                    )?)
                    .context("K3 binding byte overflow")?;
            }
        }
        // Marked layer binding aliases the uploaded pointers; only engine-facing
        // FP32 embed/head/norm create BF16 device copies. Keep staging headroom
        // and the explicit engine/KV/runtime reserve separate from this bound.
        let admitted = local_bytes
            .checked_add(binding_extra_bytes)
            .and_then(|n| n.checked_add(staging_peak))
            .and_then(|n| n.checked_add(reserve))
            .context("K3 admission byte overflow")?;
        let initial_free = gpu.free_memory()?;
        tracing::info!(
            event = "k3_weight_plan",
            rank = self.config.tp_rank,
            world = self.config.tp_world_size,
            source_bytes,
            local_bytes,
            replicated_bytes,
            staging_peak,
            binding_extra_bytes,
            reserve,
            admitted,
            free_bytes = initial_free,
            "K3 pre-upload memory plan; engine workspace/KV/runtime require explicit reserve"
        );
        ensure!(
            admitted <= initial_free,
            "K3 OOM preflight: rank {} needs {} bytes (local {}, replicated {}, staging {}, binding copies {}, reserve {}), free {}",
            self.config.tp_rank,
            admitted,
            local_bytes,
            replicated_bytes,
            staging_peak,
            binding_extra_bytes,
            reserve,
            initial_free
        );
        let mut store = WeightStore::empty();
        let mut uploaded = 0usize;
        let result: Result<()> = (|| {
            for item in inventory {
                let file = std::fs::File::open(&item.path)?;
                let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
                let tensors = safetensors::SafeTensors::deserialize(&mmap)?;
                for spec in item.tensors {
                    let tensor = tensors.tensor(&spec.name)?;
                    ensure!(
                        tensor.shape() == spec.source_shape && tensor.dtype() == spec.source_dtype,
                        "{}: tensor metadata changed after preflight",
                        spec.name
                    );
                    let local = spec.plan.copy_shard(tensor.data())?;
                    // No UVM fallback: admission or allocation failures are failures.
                    let ptr = gpu.alloc(local.len()).with_context(|| {
                        format!(
                            "K3 rank {} allocating {} local bytes for {}",
                            self.config.tp_rank,
                            local.len(),
                            spec.name
                        )
                    })?;
                    if let Err(e) = gpu.copy_h2d(&local, ptr) {
                        let _ = gpu.free(ptr);
                        return Err(e);
                    }
                    uploaded += local.len();
                    store.weights.insert(
                        spec.name,
                        WeightTensor {
                            ptr,
                            shape: spec.plan.local_shape,
                            dtype: spec.dtype,
                        },
                    );
                }
                drop(tensors);
                drop(mmap);
                evict_page_cache(&file);
                let free = gpu.free_memory()?;
                tracing::info!(
                    event = "k3_weight_shard_loaded",
                    rank = self.config.tp_rank,
                    uploaded_bytes = uploaded,
                    free_bytes = free,
                    sampled_device_delta_bytes = initial_free.saturating_sub(free),
                    "K3 rank-local shard uploaded"
                );
                ensure!(
                    free >= reserve,
                    "K3 reserve exhausted during rank-local upload"
                );
            }
            Ok(())
        })();
        if let Err(err) = result {
            for (name, tensor) in store.weights.drain() {
                if let Err(cleanup) = gpu.free(tensor.ptr) {
                    tracing::error!(%cleanup,%name,"K3 allocation cleanup failed");
                }
            }
            return Err(err);
        }
        ensure!(uploaded == local_bytes, "K3 upload accounting mismatch");
        store.prepartitioned_tp = Some((self.config.tp_rank, self.config.tp_world_size));
        Ok(store)
    }
}
