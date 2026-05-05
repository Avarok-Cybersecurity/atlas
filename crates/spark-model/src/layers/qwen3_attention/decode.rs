// SPDX-License-Identifier: AGPL-3.0-only

//! Decode path for [`super::Qwen3AttentionLayer`]. The bulk of this file
//! lived as a single ~1750 LoC monolith pre-refactor; it has now been
//! split into method-cluster sub-modules under `decode/`. This file
//! retains only [`Qwen3AttentionLayer::effective_fp8_scales`].

use super::Qwen3AttentionLayer;

mod attention_forward;
mod attention_forward_kv;
mod attention_forward_mla;
mod attention_forward_oproj;
mod high_speed_swap;
mod run_paged_decode;
mod write_kv_cache;

impl Qwen3AttentionLayer {
    pub(super) fn effective_fp8_scales(&self) -> (f32, f32) {
        if let Some(ref cal) = self.fp8_calibration {
            cal.scales()
        } else {
            (self.attn.k_scale, self.attn.v_scale)
        }
    }
}
