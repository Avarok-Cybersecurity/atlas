// SPDX-License-Identifier: AGPL-3.0-only

//! Per-sequence K3 host state: KDA conv/recurrent or MLA KV.
//!
//! Prefix-cache restore reuses C3 CPU [`LayerCache`] bytes. Not `EmptyLayerState`.

use std::any::Any;

use atlas_core::kimi_k3::LayerCache;

use crate::layer::LayerState;

/// Host mixer state for one K3 decoder layer (CPU fallback GPU wrapper).
pub struct K3CpuFallbackState {
    pub cache: LayerCache,
}

impl LayerState for K3CpuFallbackState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
