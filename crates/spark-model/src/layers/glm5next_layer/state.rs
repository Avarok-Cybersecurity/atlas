// SPDX-License-Identifier: AGPL-3.0-only

//! One GLM-5.3 decoder layer's per-sequence state.
//!
//! A GLM layer is one of two mixers, and the two need *different kinds* of state: KDA carries a
//! recurrent hidden state plus a causal-conv window and touches no KV cache at all, while DSA
//! carries an indexer key cache alongside paged KV blocks. `alloc_state` returns one boxed
//! `LayerState` per layer, so this enum is how a single composite layer type answers for both.
//!
//! 🪤 The two are NOT interchangeable and admission needs both kinds satisfied — a KDA slot is
//! not a KV block. That is [`crate::layers::glm5next_skeleton::StateKind`], made real.

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layer::LayerState;
use crate::layers::glm5next_dsa::state::Glm5NextDsaState;
use crate::layers::glm5next_kda::{Glm5NextKdaConfig, KdaSeqState};

/// A KDA layer's recurrent state, owned rather than borrowed.
///
/// 🪤 **FP32 is not negotiable.** HF casts the recurrent state to float32 and vLLM hardcodes
/// `kda_state_dtype`, so a bf16 state is a deviation from the reference, not a memory setting.
/// Sizing it at 2 bytes halves the number and is wrong.
pub struct OwnedKdaState {
    pub inner: KdaSeqState,
    recurrent_bytes: usize,
    conv_bytes: usize,
}

impl OwnedKdaState {
    /// Allocate and **zero** both buffers. A fresh sequence starts from a zero recurrent state
    /// and an empty conv window; inheriting the previous sequence's residue is a wrong answer
    /// that decays over a few tokens instead of crashing.
    pub fn alloc(gpu: &dyn GpuBackend, cfg: &Glm5NextKdaConfig) -> Result<Self> {
        let recurrent_bytes = cfg.recurrent_state_elems() * 4;
        let conv_bytes = cfg.conv_state_elems() * 4;
        let recurrent = gpu.alloc(recurrent_bytes)?;
        let conv = gpu.alloc(conv_bytes)?;
        gpu.memset_async(recurrent, 0, recurrent_bytes, 0)?;
        gpu.memset_async(conv, 0, conv_bytes, 0)?;
        gpu.synchronize(0)?;
        Ok(Self {
            inner: KdaSeqState { conv, recurrent },
            recurrent_bytes,
            conv_bytes,
        })
    }

    pub fn bytes(&self) -> usize {
        self.recurrent_bytes + self.conv_bytes
    }
}

/// Per-sequence state for one composite GLM layer.
pub enum Glm5NextLayerState {
    Kda(OwnedKdaState),
    Dsa(Glm5NextDsaState),
}

impl Glm5NextLayerState {
    /// The DSA indexer cache, or an error naming the mismatch.
    ///
    /// 🪤 A mixer/state mismatch means the scheduler handed this layer another layer's slot.
    /// Refuse loudly: silently allocating a fresh state here would decode with an empty
    /// indexer cache and select over nothing.
    pub fn dsa(&mut self) -> Result<&mut Glm5NextDsaState> {
        match self {
            Self::Dsa(s) => Ok(s),
            Self::Kda(_) => bail!("GLM layer: a DSA mixer was handed KDA recurrent state"),
        }
    }

    pub fn kda(&mut self) -> Result<&mut OwnedKdaState> {
        match self {
            Self::Kda(s) => Ok(s),
            Self::Dsa(_) => bail!("GLM layer: a KDA mixer was handed a DSA indexer cache"),
        }
    }
}

impl LayerState for Glm5NextLayerState {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Free helper for a state's device allocations. Not a `Drop` impl: `DevicePtr` carries no
/// backend handle, so freeing needs the GPU passed in.
pub fn free_kda_state(gpu: &dyn GpuBackend, s: &OwnedKdaState) -> Result<()> {
    gpu.free(s.inner.conv)?;
    gpu.free(s.inner.recurrent)
}

/// The device pointers a KDA layer's decode takes.
pub fn kda_ptrs(s: &OwnedKdaState) -> (DevicePtr, DevicePtr) {
    (s.inner.conv, s.inner.recurrent)
}
