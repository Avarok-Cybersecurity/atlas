// SPDX-License-Identifier: AGPL-3.0-only

//! Per-sequence K3 state: resident CUDA KDA recurrence or host MLA KV.
//!
//! Prefix-cache restore reuses C3 CPU [`LayerCache`] bytes. Not `EmptyLayerState`.

use std::any::Any;

use anyhow::Result;
use avarok_core::kimi_k3::LayerCache;
use spark_runtime::gpu::GpuBackend;

use super::kda_cuda::KdaDeviceState;

use crate::layer::LayerState;

/// Mixer state owned by one sequence; released through the layer release hook.
pub struct K3CpuFallbackState {
    pub cache: LayerCache,
    /// Authoritative recurrence after the first CUDA token; host cache is a seed.
    pub device_kda: Option<KdaDeviceState>,
}

impl K3CpuFallbackState {
    pub fn new(cache: LayerCache) -> Self {
        Self {
            cache,
            device_kda: None,
        }
    }

    pub fn ensure_device_kda(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        if self.device_kda.is_none()
            && let LayerCache::Kda(host) = &self.cache
        {
            self.device_kda = Some(KdaDeviceState::alloc_and_upload(gpu, host)?);
        }
        Ok(())
    }

    pub fn snapshot(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<Vec<u8>> {
        let mut cache = self.cache.clone();
        if let (Some(device), LayerCache::Kda(host)) = (&self.device_kda, &mut cache) {
            gpu.synchronize(stream)?;
            device.download(gpu, host)?;
        }
        Ok(cache.to_bytes())
    }

    pub fn restore(&mut self, gpu: &dyn GpuBackend, bytes: &[u8], stream: u64) -> Result<()> {
        let cache = LayerCache::from_bytes(bytes)?;
        gpu.synchronize(stream)?;
        self.release(gpu)?;
        self.cache = cache;
        Ok(())
    }

    pub fn release(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        if let Some(device) = self.device_kda.take() {
            device.free(gpu)?;
        }
        Ok(())
    }
}

impl LayerState for K3CpuFallbackState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use avarok_core::kimi_k3::{KdaConfig, KdaState};
    use spark_runtime::gpu::{GpuBackend, mock::MockGpuBackend};

    #[test]
    fn resident_snapshot_restore_and_release_preserve_authoritative_state() {
        let gpu = MockGpuBackend::new();
        let mut state =
            K3CpuFallbackState::new(LayerCache::Kda(KdaState::new(&KdaConfig::twin_0_40b())));
        state.ensure_device_kda(&gpu).unwrap();
        state.ensure_device_kda(&gpu).unwrap();
        assert_eq!(gpu.alloc_count(), 2, "reuse resident state between tokens");
        assert!(state.restore(&gpu, b"invalid", 0).is_err());
        assert_eq!(
            gpu.alloc_count(),
            2,
            "invalid snapshot must leave state intact"
        );
        let device = state.device_kda.as_ref().unwrap();
        gpu.copy_h2d(&7.25f32.to_le_bytes(), device.recurrent)
            .unwrap();
        let bytes = state.snapshot(&gpu, 0).unwrap();
        let LayerCache::Kda(saved) = LayerCache::from_bytes(&bytes).unwrap() else {
            panic!("expected KDA");
        };
        assert_eq!(saved.recurrent[0], 7.25);
        state.restore(&gpu, &bytes, 0).unwrap();
        assert!(state.device_kda.is_none());
        assert_eq!(gpu.alloc_count(), 0);
        state.ensure_device_kda(&gpu).unwrap();
        assert_eq!(state.snapshot(&gpu, 0).unwrap(), bytes);
        state.release(&gpu).unwrap();
        state.release(&gpu).unwrap();
        assert_eq!(gpu.alloc_count(), 0);
    }

    #[test]
    fn failed_resident_allocation_releases_partial_state() {
        let gpu = MockGpuBackend::new();
        let cfg = KdaConfig::twin_0_40b();
        gpu.set_max_allocation_bytes(cfg.conv_elems() * 4);
        let mut state = K3CpuFallbackState::new(LayerCache::Kda(KdaState::new(&cfg)));
        assert!(state.ensure_device_kda(&gpu).is_err());
        assert!(state.device_kda.is_none());
        assert_eq!(gpu.alloc_count(), 0);
    }
}
