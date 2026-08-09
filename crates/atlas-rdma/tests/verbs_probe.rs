// SPDX-License-Identifier: AGPL-3.0-only

// Public-API tests for atlas-rdma that need NO RDMA hardware and NO network:
// the cfg witness, and `Verbs::create` failing cleanly for a device name that
// cannot exist (`ibv_get_device_list` lookup miss — purely local).

/// The always-compiled witness must agree with the cfg as seen by this test
/// target (build-script cfgs apply to a crate's tests too). Runs in BOTH cfg
/// states: verbs hosts assert `true`, ATLAS_SKIP_BUILD/macOS assert `false`.
#[test]
fn verbs_enabled_matches_cfg() {
    assert_eq!(atlas_rdma::verbs_enabled(), cfg!(atlas_rdma_verbs));
}

#[cfg(atlas_rdma_verbs)]
mod with_shim {
    use atlas_rdma::{Gid, MrKeys, Verbs};

    /// A nonexistent device must fail `rs_create` (NULL) and surface the
    /// device name + gid index in the error. Exercises the real C shim and
    /// libibverbs linkage without touching any NIC state.
    #[test]
    fn create_unknown_device_errors() {
        let err = match Verbs::create("atlas-rdma-no-such-dev", 3, 0x0012_3456) {
            Ok(_) => panic!("create() succeeded for a device that cannot exist"),
            Err(e) => e,
        };
        let msg = format!("{err:#}");
        assert!(msg.contains("rs_create failed"), "unexpected error: {msg}");
        assert!(
            msg.contains("atlas-rdma-no-such-dev"),
            "device name missing: {msg}"
        );
    }

    /// `MrKeys` stays `Copy` (clients stash lkey/rkey by value everywhere) and
    /// `Gid` stays exactly 16 bytes — it is exchanged verbatim on the wire.
    #[test]
    fn mr_keys_copy_and_gid_layout() {
        let k = MrKeys { lkey: 1, rkey: 2 };
        let k2 = k; // Copy, not move
        assert_eq!((k.lkey, k.rkey), (k2.lkey, k2.rkey));
        assert_eq!(std::mem::size_of::<Gid>(), 16);
    }
}
