// SPDX-License-Identifier: AGPL-3.0-only

//! Env-guarded selector default-path tests (moved from the pre-split
//! ssm_tier.rs).

use super::*;

fn fp() -> ModelFingerprint {
    let cfg = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
    ModelFingerprint::derive_with_id(&cfg, 4, "").unwrap()
}

#[test]
fn decode_tier_defaults_to_host_ram_non_dropping() {
    assert!(
        std::env::var_os("ATLAS_SSM_DECODE_TIER").is_none(),
        "unset ATLAS_SSM_DECODE_TIER: this check owns the default branch"
    );
    let s = build_decode_tier_store(fp(), 4, /*min_slots*/ 8).unwrap();
    for k in 0..2000u64 {
        assert!(s.put(k, &[0; 4]).unwrap(), "non-dropping: nothing refused");
    }
    assert_eq!(s.len(), 2000);
}

#[test]
fn disk_cap_defaults_to_unbounded() {
    // Default-OFF guard for ATLAS_SSM_TIER_DISK_GB: unset ⇒ 0 ⇒ the Marconi
    // arms construct exactly the pre-cap unbounded store.
    for var in [
        "ATLAS_SSM_TIER_DISK_GB",
        "ATLAS_SSM_RDMA_TIER",
        "ATLAS_SSM_TIER_UNIFIED",
        "ATLAS_SSM_SWAP",
    ] {
        assert!(
            std::env::var_os(var).is_none(),
            "unset {var}: this check owns the legacy host-RAM default branch"
        );
    }
    assert_eq!(
        ssm_tier_disk_slots(4).unwrap(),
        0,
        "unset budget must resolve to the unbounded sentinel"
    );
    let s = build_tier_store(fp(), 4).unwrap();
    assert!(s.put(u64::MAX, &[1, 2, 3, 4]).unwrap());
    for k in 0..1000u64 {
        assert!(s.put(k, &[0; 4]).unwrap(), "unbounded: nothing dropped");
    }
    assert_eq!(s.len(), 1001);
    let mut out = [0u8; 4];
    assert!(s.get(u64::MAX, &mut out).unwrap());
    assert_eq!(out, [1, 2, 3, 4], "default store preserves payloads");
}
