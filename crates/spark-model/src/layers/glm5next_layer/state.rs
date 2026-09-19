// SPDX-License-Identifier: AGPL-3.0-only

//! One GLM-5.3 decoder layer's per-sequence state.
//!
//! A GLM layer is one of two mixers and the two need *different kinds* of state: KDA carries a
//! recurrent hidden state plus a causal-conv window and touches no KV cache at all, while DSA
//! carries an indexer key cache alongside paged KV blocks. Each mixer therefore returns its own
//! concrete `LayerState` and [`crate::layers::glm5next_layer::Glm5NextLayer`] downcasts to the
//! one its mixer expects.
//!
//! 🔴 **KDA's state is the pool's [`SsmLayerState`], not a GLM-private type.**
//! `rollback_ssm_states_dispatch` walks every `LayerType::LinearAttention` layer and downcasts
//! to exactly that type to rewind a rejected speculative draft. GLM's KDA blocks *are*
//! `linear_attention` in `layer_types`, so a GLM-private state means the first rejected draft
//! is a hard error — and the shapes line up byte-for-byte anyway:
//!
//! | | pool (`config` fields, TP-local) | GLM (`Glm5NextKdaConfig`) |
//! |---|---|---|
//! | h    | `nv · vd · kd · 4`               | `heads · head_dim² · 4` |
//! | conv | `(nk · kd · 2 + nv · vd) · d_conv · 4` | `3 · heads · head_dim · conv_kernel · 4` |
//!
//! The parser fills `linear_num_{key,value}_heads` / `linear_{key,value}_head_dim` /
//! `linear_conv_kernel_dim` from `linear_attn_config`, already divided by TP, so
//! `ModelConfig::ssm_h_state_bytes()` and `ssm_conv_state_bytes()` return GLM's own numbers.
//!
//! 🪤 The two state kinds are NOT interchangeable and admission needs both kinds satisfied — a
//! KDA slot is not a KV block. That is [`crate::layers::glm5next_skeleton::StateKind`], made
//! real.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use crate::layer::SsmLayerState;
use crate::layers::glm5next_kda::Glm5NextKdaConfig;

/// Allocate and **zero** a KDA layer's recurrent + conv state, pool-free.
///
/// Used only where a state is built outside the SSM pool; the serving path takes pool slots
/// (`Glm5NextLayer::uses_ssm_pool()`), which is what carries the checkpoints and per-token
/// intermediates a speculative rollback needs.
///
/// 🪤 **FP32 is not negotiable.** HF casts the recurrent state to float32 and vLLM hardcodes
/// `kda_state_dtype`, so a BF16/FP16 state is a deviation from the reference, not a memory
/// setting. `h_is_f16: false` here, and `Glm5NextLayer::kda_state` refuses a narrowed slot.
pub fn alloc_kda_ssm_state(gpu: &dyn GpuBackend, cfg: &Glm5NextKdaConfig) -> Result<SsmLayerState> {
    let h_bytes = cfg.recurrent_state_elems() * 4;
    let conv_bytes = cfg.conv_state_elems() * 4;
    let h_state = gpu.alloc(h_bytes)?;
    let conv_state = gpu.alloc(conv_bytes)?;
    gpu.memset_async(h_state, 0, h_bytes, 0)?;
    gpu.memset_async(conv_state, 0, conv_bytes, 0)?;
    gpu.synchronize(0)?;
    Ok(SsmLayerState {
        h_state,
        conv_state,
        h_state_checkpoint: None,
        conv_state_checkpoint: None,
        h_state_intermediates: Vec::new(),
        conv_state_intermediates: Vec::new(),
        h_is_f16: false,
        h_prefill_stage: None,
        // GLM-5.3 hosts no `PleLayer` (its linear-attention block is KDA), so
        // there is no PLE per-sequence carry to hold. Upstream #753 item B
        // added this field; `None` is the correct answer, not a placeholder.
        ple: None,
    })
}
