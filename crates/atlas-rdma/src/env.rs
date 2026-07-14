// SPDX-License-Identifier: AGPL-3.0-only
//
// Rail-env resolution helpers. The five RDMA clients share the SHAPE of this
// logic but not the semantics: two distinct empty-string behaviors exist in
// the deployed configs and BOTH must be preserved exactly —
//
//   * `first_set`      — an exported-but-EMPTY var counts as set (the
//     `Result::or_else` / `unwrap_or_else` chains: KV / expert / snapshot
//     single-key reads and the LoRA DEV chain).
//   * `first_nonempty` — an empty var is SKIPPED (weight tier's `env_str`,
//     which lets `ATLAS_WEIGHT_RDMA_DEV=""` fall through to the EXPERT name).
//
// Each client passes its EXACT key list; the fallback chains are per-tier
// deployment surface (e.g. LoRA's GID reads ONLY `ATLAS_LORA_RDMA_GID` — no
// WEIGHT/EXPERT fallback), so no chain is hardcoded here.

/// First key whose var is present in the environment — even if empty.
pub fn first_set(keys: &[&str], default: &str) -> String {
    for k in keys {
        if let Ok(v) = std::env::var(k) {
            return v;
        }
    }
    default.to_string()
}

/// First key whose var is present AND non-empty (weight-tier semantics).
pub fn first_nonempty(keys: &[&str], default: &str) -> String {
    for k in keys {
        if let Ok(v) = std::env::var(k)
            && !v.is_empty()
        {
            return v;
        }
    }
    default.to_string()
}

/// First key whose var is present AND parses as `u32`; otherwise `default`.
/// (A set-but-unparseable var falls through — the behavior of every existing
/// per-client `env_u32`, single- and multi-key alike.)
pub fn first_set_u32(keys: &[&str], default: u32) -> u32 {
    for k in keys {
        if let Some(v) = std::env::var(k).ok().and_then(|s| s.parse().ok()) {
            return v;
        }
    }
    default
}
