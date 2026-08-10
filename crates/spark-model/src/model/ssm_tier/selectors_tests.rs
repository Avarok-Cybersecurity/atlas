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
    // With ATLAS_SSM_DECODE_TIER unset the decode store is unbounded host-RAM
    // and never drops (the correctness floor). Guard on the var being unset.
    if std::env::var_os("ATLAS_SSM_DECODE_TIER").is_none() {
        let s = build_decode_tier_store(fp(), 4, /*min_slots*/ 8).unwrap();
        for k in 0..2000u64 {
            assert!(s.put(k, &[0; 4]).unwrap(), "non-dropping: nothing refused");
        }
        assert_eq!(s.len(), 2000);
    }
}

#[test]
fn disk_cap_defaults_to_unbounded() {
    // Default-OFF guard for ATLAS_SSM_TIER_DISK_GB: unset ⇒ 0 ⇒ the Marconi
    // arms construct exactly the pre-cap unbounded store. Guarded on the var
    // being unset (same idiom as the other selector tests).
    if std::env::var_os("ATLAS_SSM_TIER_DISK_GB").is_none()
        && std::env::var_os("ATLAS_SSM_RDMA_TIER").is_none()
    {
        assert_eq!(
            ssm_tier_disk_slots(4).unwrap(),
            0,
            "unset budget must resolve to the unbounded sentinel"
        );
        let s = build_tier_store(fp(), 4).unwrap();
        for k in 0..1000u64 {
            assert!(s.put(k, &[0; 4]).unwrap());
        }
        assert_eq!(s.len(), 1000, "nothing dropped without an explicit budget");
    }
}

#[test]
fn build_tier_store_defaults_to_host_ram_unbounded() {
    // With ATLAS_SSM_RDMA_TIER absent (the byte-identical default), the
    // selector yields the unbounded host-RAM store. Guarded on the var being
    // unset so a concurrent env-setting test can't flake this.
    if std::env::var_os("ATLAS_SSM_RDMA_TIER").is_none() {
        let s = build_tier_store(fp(), 4).unwrap();
        assert!(s.put(1, &[1, 2, 3, 4]).unwrap());
        let mut o = [0u8; 4];
        assert!(s.get(1, &mut o).unwrap());
        assert_eq!(o, [1, 2, 3, 4]);
        for k in 0..1000u64 {
            assert!(s.put(k, &[0; 4]).unwrap(), "unbounded: nothing dropped");
        }
        assert_eq!(s.len(), 1000);
    }
}
