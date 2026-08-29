// SPDX-License-Identifier: AGPL-3.0-only

//! Which tensors the fast loader does NOT upload.
//!
//! Three independent rules, worth reading together because each one withholds
//! bytes a downstream loader might expect:
//!
//!   1. **EP sharding** — remote experts belong to another rank.
//!   2. **`skip_activation_scales`** — W4A4 `*.input_scale`, opt-in.
//!   3. **`skip_mtp`** — `mtp.*` for a loader that builds no MTP head, opt-in.
//!
//! Rules 2 and 3 default OFF and are allow-listed per model, because
//! withholding a tensor a loader DOES read is invisible until the output is
//! subtly wrong. Rule 1 is structural and always active under EP.

use super::FastSafetensorsLoader;
use crate::weights::parse_expert_index;

impl FastSafetensorsLoader {
    pub(super) fn should_skip_tensor(&self, name: &str) -> bool {
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
