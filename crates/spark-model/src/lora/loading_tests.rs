// SPDX-License-Identifier: AGPL-3.0-only

//! Family allow-list tests for the LoRA v0 loader gate. GPU-free: `check_family`
//! is a pure predicate over `ModelConfig`.

use super::check_family;
use crate::lora::test_support::cfg;

/// Every admitted family passes; `laguna` (Laguna-S-2.1) is one of them because
/// its `LagunaWeightLoader` builds the same `Qwen3AttentionLayer` + `MoeLayer`
/// trunk the install walk downcasts to. This is the fails-without assertion for
/// the laguna admit: drop the `|| cfg.model_type == "laguna"` clause and the
/// `laguna` leg below bails with REJECT[unvalidated-family].
#[test]
fn admitted_families_pass() {
    let mut c = cfg();
    for mt in ["holo3_1_moe", "qwen3_6_moe", "laguna"] {
        c.model_type = mt.to_string();
        assert!(
            check_family(&c).is_ok(),
            "model_type={mt} must be admitted by LoRA v0"
        );
    }
    // qwen3_5 is admitted only in its DENSE form (num_experts == 0).
    c.model_type = "qwen3_5".to_string();
    c.num_experts = 0;
    assert!(check_family(&c).is_ok());
}

/// Unvalidated families still bail, and the message names the admitted set —
/// including laguna — so the reject stays actionable.
#[test]
fn unvalidated_family_rejected_with_named_allow_list() {
    let mut c = cfg();
    c.model_type = "nemotron_h".to_string();
    let err = check_family(&c).unwrap_err().to_string();
    assert!(err.contains("REJECT[unvalidated-family]"), "{err}");
    assert!(err.contains("nemotron_h"), "{err}");
    assert!(err.contains("laguna"), "{err}");
    // The factory config is a MoE qwen3_next: not dense qwen3_5, so rejected.
    let c = cfg();
    assert!(check_family(&c).is_err());
    // qwen3_5 WITH experts is not the validated dense trunk.
    let mut c = cfg();
    c.model_type = "qwen3_5".to_string();
    assert!(check_family(&c).is_err());
}
