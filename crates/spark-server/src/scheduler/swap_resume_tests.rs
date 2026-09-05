// SPDX-License-Identifier: AGPL-3.0-only

//! A failed swap-IN must reach the client, exactly as a failed swap-OUT
//! already does.
//!
//! `swap_out_sequence` was fixed once for this: "on error the victim is
//! surfaced to its client and freed here rather than silently dropped
//! (the old path leaked the GPU blocks AND the client saw only
//! 'Inference cancelled')". The RESUME half kept the old shape — bare
//! `?` on `alloc_sequence` / `open_file` / `restore_sequence_state`,
//! consuming the `SwappedSeq` and with it the only handle to the waiting
//! request. The scheduler logged `Swap-in failed: …` and moved on; the
//! client's oneshot closed, which the API layer reports as the generic
//! "Inference cancelled", and a streaming client got a stream that
//! simply stopped.

use super::lifecycle::{resume_swapped_seq, swap_out_sequence};
use super::lifecycle_tests::StubModel;
use super::test_support::test_seq;
use spark_runtime::kv_spill::KvSpillManager;

/// Per-test spill root under the process's temp dir; `KvSpillManager::new`
/// creates it and wipes stale `swap_*` files.
fn spill(tag: &str) -> KvSpillManager {
    let dir = std::env::temp_dir().join(format!("atlas-swap-resume-{tag}-{}", std::process::id()));
    KvSpillManager::new(dir, 8 * 1024 * 1024).expect("spill manager")
}

#[test]
fn a_swap_in_that_cannot_read_its_image_tells_the_client() {
    let model = StubModel::default();
    let mut spill = spill("lost-image");
    let (a, mut rx) = test_seq(vec![7, 8, 9], 5, None, 3);
    let mut active = vec![a];
    let s = swap_out_sequence(&model, &mut active, 0, &mut spill).expect("swap-out writes");

    // The image disappears between swap-out and resume: an operator
    // clearing the spill dir, a tmpfs eviction, a disk-full reclaim.
    // `open_file` then fails and the sequence can never come back.
    spill.remove_file(s.swap_id).expect("remove the image");
    let freed_before = model.freed.lock().expect("freed list").len();

    let server_side = match resume_swapped_seq(None, None, &model, s, &mut spill) {
        Ok(_) => panic!("a swap-in with no image on disk must fail"),
        Err(e) => e,
    };

    match rx.try_recv() {
        Ok(Err(e)) => {
            let msg = format!("{e:#}");
            // Naming the phase is the point: "Inference cancelled" is
            // what the client used to get, and it is indistinguishable
            // from a client-side abort.
            assert!(
                msg.contains("swap-in"),
                "the error must name what failed, got {msg:?}"
            );
        }
        Ok(Ok(_)) => panic!("a failed swap-in must not report success"),
        Err(e) => panic!(
            "the client's sink was dropped instead of told ({e}); \
             the server-side error was {server_side:#}"
        ),
    }

    // The slot claimed by `alloc_sequence` before the failure must go
    // back, or every failed resume permanently narrows the pool.
    let freed_after = model.freed.lock().expect("freed list").len();
    assert_eq!(
        freed_after,
        freed_before + 1,
        "the sequence allocated for the resume must be freed on the error path"
    );
}
