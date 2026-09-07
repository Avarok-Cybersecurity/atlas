// SPDX-License-Identifier: AGPL-3.0-only

//! Which tensors the fast loader does NOT upload.
//!
//! Four independent rules, worth reading together because each one withholds
//! bytes a downstream loader might expect:
//!
//!   1. **EP sharding** — remote experts belong to another rank.
//!   2. **`skip_activation_scales`** — W4A4 `*.input_scale`, opt-in.
//!   3. **`skip_mtp`** — `mtp.*` for a loader that builds no MTP head, opt-in.
//!   4. **BEL** — `--expert-category`: experts this category never routes to.
//!
//! Rules 2 and 3 default OFF and are allow-listed per model, because
//! withholding a tensor a loader DOES read is invisible until the output is
//! subtly wrong. Rule 1 is structural and always active under EP. Rule 4 is
//! paired with a router mask built from the SAME plan — an expert skipped
//! here is one the top-k cannot select, and the two must never disagree.

use super::FastSafetensorsLoader;
use crate::weights::{parse_expert_index, parse_layer_expert};

impl FastSafetensorsLoader {
    /// Whether this tensor is withheld. `pub` so the model-side load loops
    /// can be tested against the SAME decision they have to agree with;
    /// disagreement is a failed load or a silently nulled expert.
    pub fn should_skip_tensor(&self, name: &str) -> bool {
        // MTP head weights for a model whose loader does not build one.
        if self.skip_mtp && name.starts_with("mtp.") {
            return true;
        }
        // W4A4 activation scales: never read on the w4a16 path (the NVFP4
        // loader falls back to `DevicePtr::NULL`), and 4-byte allocations are
        // almost pure granule padding at expert scale.
        if self.skip_activation_scales && name.ends_with(".input_scale") {
            return true;
        }
        // BEL: an expert this category does not route to. Checked before
        // the EP early-return because BEL applies at ep_world_size == 1,
        // which is every single-GPU serve.
        if let Some(plan) = self.bel.as_ref()
            && let Some((layer, expert)) = parse_layer_expert(name)
            && !plan.is_loaded(layer, expert)
        {
            return true;
        }
        if self.ep_world_size <= 1 {
            return false;
        }
        if name.starts_with("mtp.") {
            return false;
        }
        if let Some(idx) = parse_expert_index(name) {
            let per_rank = self.num_experts / self.ep_world_size;
            let local_start = self.ep_rank * per_rank;
            let local_end = if self.ep_rank == self.ep_world_size - 1 {
                self.num_experts
            } else {
                local_start + per_rank
            };
            idx < local_start || idx >= local_end
        } else {
            false
        }
    }
}

#[cfg(test)]
#[path = "skip_tests.rs"]
mod tests;
