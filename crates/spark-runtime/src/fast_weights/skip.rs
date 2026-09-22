// SPDX-License-Identifier: AGPL-3.0-only

//! Which tensors the fast loader does NOT upload.
//!
//! Three independent rules, worth reading together because each one withholds
//! bytes a downstream loader might expect:
//!
//!   1. **EP sharding** — remote experts belong to another rank.
//!   2. **`skip_activation_scales`** — W4A4 `*.input_scale`, opt-in.
//!   3. **`skip_mtp`** — `mtp.*` for a loader that builds no MTP head, opt-in.
//!   4. **`skip_vision`** — the vision tower, for a text-only port that binds
//!      no vision encoder. Derived from the loader, not opt-in per model.
//!
//! Rules 2 and 3 default OFF and are allow-listed per model, because
//! withholding a tensor a loader DOES read is invisible until the output is
//! subtly wrong. Rule 1 is structural and always active under EP.
//!
//! **DEFER is a fifth rule and a different kind.** A skipped tensor is gone; a
//! deferred one is recorded with its on-disk location because the model's own
//! loader will read it from the host. [`FastSafetensorsLoader::is_deferred`]
//! lives here so the two questions are answered side by side, but it is asked
//! AFTER `should_skip_tensor` — see [`crate::weights::deferred`].

use super::FastSafetensorsLoader;
use crate::weights::{WeightDtype, parse_expert_index};

impl FastSafetensorsLoader {
    /// Does the model's loader claim this tensor? `false` when no hook is set,
    /// which is every model but the ones that opt in.
    ///
    /// 🪤 Keyed on the STORE dtype — the width the tensor would have had in the
    /// store, which for an F16 export is BF16. A predicate that says "BF16
    /// routed expert" therefore also catches the F16 spelling of the same
    /// export, which is the honest answer.
    pub fn is_deferred(&self, name: &str, dtype: WeightDtype) -> bool {
        self.defer.as_ref().is_some_and(|f| f(name, dtype))
    }
}

impl FastSafetensorsLoader {
    pub(super) fn should_skip_tensor(&self, name: &str) -> bool {
        // Checked before the EP short-circuit: a text-only port skips the
        // vision tower at tp/ep 1 too.
        if self.skip_vision && super::is_vision_tensor(name) {
            return true;
        }
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

#[cfg(test)]
mod activation_scale_tests {
    use super::FastSafetensorsLoader;

    fn loader() -> FastSafetensorsLoader {
        let mut l = FastSafetensorsLoader::new();
        l.skip_activation_scales = true;
        l
    }

    /// 🔴 The BLAST RADIUS of the opt-in, pinned. The rule is a `.input_scale`
    /// suffix and nothing else; every model added to the server's allow-list
    /// depends on it never reaching `weight_scale` / `weight_scale_2`, which
    /// ARE the w4a16 operands. A predicate widened to `contains("scale")` or
    /// to `.starts_with` would pass every other test in this crate and quietly
    /// unbind every quantised projection in the checkpoint.
    #[test]
    fn the_rule_reaches_only_activation_scales() {
        let l = loader();
        assert!(l.should_skip_tensor("model.layers.0.mlp.gate_proj.input_scale"));
        assert!(l.should_skip_tensor("model.layers.0.mlp.experts.7.up_proj.input_scale"));

        for keep in [
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.gate_proj.weight_scale",
            "model.layers.0.mlp.gate_proj.weight_scale_2",
            "model.layers.0.self_attn.k_proj.k_scale",
            "model.layers.0.mlp.gate_proj.input_scale_inv",
            "input_scale",
        ] {
            assert!(!l.should_skip_tensor(keep), "{keep} must still load");
        }
    }

    /// OPT-IN: off by default, so a model that never asks keeps every scalar.
    #[test]
    fn nothing_is_skipped_unless_the_model_opted_in() {
        let l = FastSafetensorsLoader::new();
        assert!(!l.skip_activation_scales);
        assert!(!l.should_skip_tensor("model.layers.0.mlp.gate_proj.input_scale"));
    }
}

#[cfg(test)]
#[path = "defer_tests.rs"]
mod defer_tests;
