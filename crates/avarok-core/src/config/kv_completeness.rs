// SPDX-License-Identifier: AGPL-3.0-only

//! The two KV-completeness capability gates.
//!
//! Split out of `methods.rs` (the 500-LoC cap) when the GLM validation switch
//! pushed that file over, and placed beside `kv_completeness_tests.rs` so the
//! predicates and the tests that pin them sit together: a new model type must
//! be taught to both gates, and forgetting one is the defect this file exists
//! to make visible.

use super::ModelConfig;

impl ModelConfig {
    /// Whether every byte of a sequence's per-layer state is represented by
    /// its KV blocks.
    ///
    /// False for models whose PREFILL builds per-sequence state that KV pages
    /// do not carry: GLM-5.3's DSA indexer rows (`Glm5NextDsaState`) and
    /// compressed DeepSeek V4's compressor pool/ring. Every KV-only mechanism
    /// — radix prefix reuse and the `--swap-space-gb` spill image alike — is
    /// unsafe for those models, and this is the single fact both gates below
    /// are asking about.
    ///
    /// 🔒 `glm5_next` stays `false` even though the machinery to make it true now
    /// exists. `Glm5NextLayer` implements the Marconi aux hooks
    /// (`has_aux_state`/`snapshot_aux`/`restore_aux`, `layers/glm5next_dsa/aux_state.rs`),
    /// so a snapshot CAN now carry the DSA indexer rows the KV pages do not — and
    /// KDA never needed carrying, because `uses_ssm_pool()` is true for it and
    /// `SsmSnapshotPool` already captures that region device-to-device. What is
    /// missing is not code, it is PROOF: the round trip is verified on CPU against
    /// `MockGpuBackend` only, and nothing has yet shown that a warm-cache GLM-5.3
    /// serve produces the same tokens as a cold one on real hardware. Flipping this
    /// arm before that measurement exists would trade a correct-but-slow serve for
    /// an unproven one. Flip it in the same change that lands the GPU evidence.
    ///
    /// 🧪 The measurement itself needs the arm OPEN, so
    /// `kv_only_prefix_cache_is_safe` below carries an env-only validation switch
    /// (`AVAROK_GLM53_PREFIX_CACHE_UNPROVEN`) that opens the prefix-cache arm and
    /// nothing else. This `match` — the compiled default and the swap-out answer —
    /// is untouched by it. `scripts/glm53-pc/` in the spark-bench repo is the
    /// harness that turns the switch into the evidence.
    fn per_sequence_state_is_kv_complete(&self) -> bool {
        match self.model_type.as_str() {
            "glm5_next" | "glm5_next_text" => false,
            "deepseek_v4" => self.compress_ratios.iter().all(|&ratio| ratio == 0),
            _ => true,
        }
    }

    /// Whether the radix prefix cache captures every state needed to resume
    /// this model exactly. Preflight SSOT for `build_prefix_cache`.
    ///
    /// 🧪 `AVAROK_GLM53_PREFIX_CACHE_UNPROVEN` (PRESENCE, house convention — `=0`
    /// is NOT "off") opens this arm for `glm5_next` so the GPU evidence the gate
    /// above asks for can actually be MEASURED. It is a validation switch, not a
    /// feature flag: the compiled default is unchanged, and unsetting the
    /// variable is the whole rollback.
    ///
    /// 🔴 Scoped to the PREFIX-CACHE arm on purpose. `kv_only_swap_out_is_safe()`
    /// deliberately does not consult it, because the `--swap-space-gb` image is
    /// structurally incapable of carrying aux blobs — `save_sequence_state_dispatch`
    /// writes KV blocks plus `SsmLayerState` and calls no aux hook — so a shared
    /// override would unlock a path the DSA codec never reaches.
    ///
    /// Both call sites (`preflight_reserve` and `build_prefix_cache`) reach the
    /// predicate through this method, so the reservation and the allocation cannot
    /// disagree about whether the Marconi slots are live (ANOMALIES A68's class).
    pub fn kv_only_prefix_cache_is_safe(&self) -> bool {
        self.kv_only_prefix_cache_is_safe_with(Self::glm53_prefix_cache_validation_env())
    }

    /// Pure core of [`Self::kv_only_prefix_cache_is_safe`] (env-free, unit-testable
    /// — the `marconi_snapshot_slots_with` pattern, so no test has to mutate a
    /// process-global variable that a sibling test is reading).
    ///
    /// The override is narrow by construction: it can only ever open `glm5_next`,
    /// so a stray export cannot silently re-enable a compressed DeepSeek-V4 serve.
    pub(crate) fn kv_only_prefix_cache_is_safe_with(&self, validation_override: bool) -> bool {
        self.per_sequence_state_is_kv_complete()
            || (validation_override
                && matches!(self.model_type.as_str(), "glm5_next" | "glm5_next_text"))
    }

    /// The `AVAROK_GLM53_PREFIX_CACHE_UNPROVEN` validation switch (PRESENCE).
    fn glm53_prefix_cache_validation_env() -> bool {
        std::env::var_os("AVAROK_GLM53_PREFIX_CACHE_UNPROVEN").is_some()
    }

    /// Whether a sequence may be swapped out to the `--swap-space-gb` pool and
    /// restored from it. Preflight SSOT for `resolve_swap_space_gb`.
    ///
    /// `save_sequence_state_dispatch` writes KV blocks plus the `SsmLayerState`
    /// of each `LayerType::LinearAttention` layer, and nothing else; the
    /// swap-out then calls `free_sequence`, which hands every remaining
    /// per-layer state to #821's `release_state`. A model that is not
    /// KV-complete therefore resumes with a freshly ZEROED pool behind a KV
    /// image that assumes a populated one — a silently wrong answer, not a
    /// crash. Distinct from the prefix-cache predicate because they are
    /// distinct guarantees; they happen to have the same answer today.
    pub fn kv_only_swap_out_is_safe(&self) -> bool {
        self.per_sequence_state_is_kv_complete()
    }
}
