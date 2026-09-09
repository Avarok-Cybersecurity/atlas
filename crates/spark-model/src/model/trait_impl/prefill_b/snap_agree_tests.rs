// SPDX-License-Identifier: AGPL-3.0-only

//! A100 two-rank pins: the Marconi restore decision is rank-agreed and
//! all-or-nothing. Each test models rank 0 and rank 1 as two `LocalGates`
//! (their rank-local post-F83 state), votes, agrees, and asserts that BOTH
//! ranks land on the same decision and the same replay length. The last test
//! drives the real rooted-broadcast gather over two threads with a mock
//! communicator and two `MockGpuBackend`s — no GPU, no container.

use std::sync::{Arc, Condvar, Mutex};

use spark_comm::CommBackend;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::gpu::mock::MockGpuBackend;

use super::snap_agree::{LocalGates, agree, gather_u32_via_broadcast, local_proposal, skip_point};

const MATCHED: usize = 16352;
const TOTAL: usize = 16356;
const T: usize = 16336;

/// A rank that holds a restorable intermediate checkpoint at `snap_tok`.
fn holder(snap_tok: usize) -> LocalGates {
    LocalGates {
        snap_tok,
        matched: MATCHED,
        total: TOTAL,
        min_tokens: 256,
        has_hidden: false,
        exact_enabled: false,
        is_tail: false,
        session_ok: true,
        needs_aux: true,
        has_aux: true,
    }
}

/// A rank whose pool evicted the checkpoint (the S4 05:14:52Z rank 0).
fn none() -> LocalGates {
    holder(0)
}

/// Run the whole decision for two ranks: proposals → agreement → skip point.
/// Returns `(agreed, [skip_point_r0, skip_point_r1])`.
fn decide(r0: &LocalGates, r1: &LocalGates) -> (Option<u32>, [usize; 2]) {
    let votes = [local_proposal(r0), local_proposal(r1)];
    let agreed = agree(&votes);
    let point = |g: &LocalGates| {
        let skip = agreed.is_some();
        skip_point(
            skip,
            agreed.map_or(0, |t| t as usize),
            g.matched,
            g.total,
            true,
        )
    };
    (agreed, [point(r0), point(r1)])
}

// 1. both ranks hold the same snapshot T ⇒ RESTORE T on both.
#[test]
fn both_ranks_same_snapshot_restore_it() {
    let (agreed, pts) = decide(&holder(T), &holder(T));
    assert_eq!(agreed, Some(T as u32));
    assert_eq!(pts, [T, T]);
}

// 2. rank 0 has T, rank 1 has none ⇒ RECOMPUTE on both (the A100 wedge shape).
#[test]
fn one_rank_without_snapshot_forces_recompute_everywhere() {
    let (agreed, pts) = decide(&holder(T), &none());
    assert_eq!(agreed, None);
    assert_eq!(pts, [0, 0]);
    // Symmetric.
    let (agreed, pts) = decide(&none(), &holder(T));
    assert_eq!(agreed, None);
    assert_eq!(pts, [0, 0]);
}

// 3. different tokens ⇒ never a token only one rank owns; both recompute.
#[test]
fn different_tokens_never_pick_a_token_one_rank_lacks() {
    let (agreed, pts) = decide(&holder(T), &holder(T - 256));
    assert_eq!(agreed, None, "a naive min would have chosen {}", T - 256);
    assert_eq!(pts, [0, 0]);
}

// 4. same candidate, rank 1 fails the aux-restorable gate ⇒ both recompute.
#[test]
fn aux_gate_failure_on_one_rank_declines_everywhere() {
    let mut r1 = holder(T);
    r1.has_aux = false;
    let (agreed, pts) = decide(&holder(T), &r1);
    assert_eq!(agreed, None);
    assert_eq!(pts, [0, 0]);
}

// 5. a later local decline on one rank (session / min-tokens / hidden /
//    exact-bypass) cannot leave the other rank restoring.
#[test]
fn any_local_decline_on_one_rank_declines_everywhere() {
    let declines: Vec<(&str, LocalGates)> = vec![
        ("tail snapshot from another session", {
            let mut g = holder(T);
            g.is_tail = true;
            g.session_ok = false;
            g
        }),
        ("below marconi_min_tokens", {
            let mut g = holder(T);
            g.min_tokens = T + 1;
            g
        }),
        ("exact full-prompt hit without hidden", {
            let mut g = holder(TOTAL);
            g.matched = TOTAL;
            g.exact_enabled = true;
            g.has_hidden = false;
            g
        }),
        ("exact full-prompt shortcut bypassed by default", {
            let mut g = holder(TOTAL);
            g.matched = TOTAL;
            g.has_hidden = true;
            g
        }),
    ];
    for (why, r1) in declines {
        assert_eq!(
            local_proposal(&r1),
            0,
            "{why}: rank 1 must propose RECOMPUTE"
        );
        let (agreed, pts) = decide(&holder(r1.snap_tok), &r1);
        assert_eq!(agreed, None, "{why}");
        assert_eq!(pts, [0, 0], "{why}");
    }
}

// 6. no snapshot anywhere ⇒ RECOMPUTE.
#[test]
fn no_snapshot_anywhere_recomputes() {
    let (agreed, pts) = decide(&none(), &none());
    assert_eq!(agreed, None);
    assert_eq!(pts, [0, 0]);
    assert_eq!(agree(&[]), None);
}

// 7. replay length identical across ranks after the decision, for every
//    agreement outcome and every skip shape.
#[test]
fn replay_length_identical_across_ranks() {
    for (r0, r1) in [
        (holder(T), holder(T)),
        (holder(T), none()),
        (holder(T), holder(T - 256)),
        (none(), none()),
    ] {
        let (_, [p0, p1]) = decide(&r0, &r1);
        assert_eq!(TOTAL - p0, TOTAL - p1, "replay differs: {r0:?} vs {r1:?}");
    }
    // Non-SSM (F82) skip and the exact-leaf shape are pure functions of the
    // agreed inputs too.
    assert_eq!(skip_point(true, 0, MATCHED, TOTAL, false), MATCHED);
    assert_eq!(skip_point(true, TOTAL, TOTAL, TOTAL, true), TOTAL);
    assert_eq!(skip_point(true, T, MATCHED, TOTAL, true), T);
    assert_eq!(skip_point(false, T, MATCHED, TOTAL, true), 0);
}

/// Two-rank rendezvous broadcast: the root hands its device bytes to the
/// other rank, which lands them at its own (rank-local) pointer.
struct Link {
    /// `(root, bytes)` — tagged with the root so a root waiting for its own
    /// bytes to be taken cannot mistake the NEXT broadcast's post (already
    /// issued by the faster peer) for its own still-unread one.
    slot: Mutex<Option<(usize, Vec<u8>)>>,
    cv: Condvar,
}

struct PairComm {
    rank: usize,
    gpu: Arc<MockGpuBackend>,
    link: Arc<Link>,
}

impl CommBackend for PairComm {
    fn all_reduce(&self, _: u64, _: usize) -> anyhow::Result<()> {
        unreachable!("gather uses broadcast only")
    }
    fn all_gather(&self, _: u64, _: u64, _: usize) -> anyhow::Result<()> {
        unreachable!()
    }
    fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> anyhow::Result<()> {
        unreachable!()
    }
    fn broadcast(&self, ptr: u64, bytes: usize, root: usize) -> anyhow::Result<()> {
        let dev = spark_runtime::gpu::DevicePtr(ptr);
        let mut slot = self.link.slot.lock().unwrap();
        if self.rank == root {
            let mut out = vec![0u8; bytes];
            self.gpu.copy_d2h(dev, &mut out)?;
            *slot = Some((root, out));
            self.link.cv.notify_all();
            // Wait until the peer has taken the bytes: a broadcast completes
            // on every rank together.
            while slot.as_ref().is_some_and(|(r, _)| *r == root) {
                slot = self.link.cv.wait(slot).unwrap();
            }
        } else {
            while slot.as_ref().is_none_or(|(r, _)| *r != root) {
                slot = self.link.cv.wait(slot).unwrap();
            }
            let (_, bytes) = slot.take().unwrap();
            self.gpu.copy_h2d(&bytes, dev)?;
            self.link.cv.notify_all();
        }
        Ok(())
    }
    fn barrier(&self) -> anyhow::Result<()> {
        Ok(())
    }
    fn send_to(&self, _: u64, _: usize, _: usize, _: u64) -> anyhow::Result<()> {
        unreachable!()
    }
    fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> anyhow::Result<()> {
        unreachable!()
    }
    fn rank(&self) -> usize {
        self.rank
    }
    fn world_size(&self) -> usize {
        2
    }
}

/// Run the real gather on two threads; returns what each rank observed.
fn gather_two(vals: [u32; 2]) -> [Vec<u32>; 2] {
    let link = Arc::new(Link {
        slot: Mutex::new(None),
        cv: Condvar::new(),
    });
    let handles: Vec<_> = (0..2)
        .map(|rank| {
            let link = Arc::clone(&link);
            std::thread::spawn(move || {
                let gpu = Arc::new(MockGpuBackend::new());
                let buf = gpu.alloc(4).unwrap();
                let comm = PairComm {
                    rank,
                    gpu: Arc::clone(&gpu),
                    link,
                };
                gather_u32_via_broadcast(gpu.as_ref(), &comm, buf, 2, vals[rank]).unwrap()
            })
        })
        .collect();
    let mut out = handles.into_iter().map(|h| h.join().unwrap());
    [out.next().unwrap(), out.next().unwrap()]
}

// 8. the collective helper itself: every rank sees the same vector, in rank
//    order, so `agree` (and F83's `min`) is deterministic on every rank —
//    for agreement AND for disagreement.
#[test]
fn gather_gives_every_rank_the_same_votes() {
    let t = T as u32;
    for (vals, want) in [
        ([t, t], Some(t)),
        ([t, 0], None),
        ([0, t], None),
        ([t, t - 256], None),
        ([0, 0], None),
    ] {
        let [r0, r1] = gather_two(vals);
        assert_eq!(r0, vals.to_vec(), "rank 0 view of {vals:?}");
        assert_eq!(r1, vals.to_vec(), "rank 1 view of {vals:?}");
        assert_eq!(agree(&r0), want);
        assert_eq!(agree(&r1), want);
        // F83's min-reduce is the same schedule; keep it honest.
        assert_eq!(r0.iter().min(), r1.iter().min());
    }
}
