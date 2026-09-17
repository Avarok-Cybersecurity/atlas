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

// ── A109: worker-symmetric decode checkpoints ────────────────────────────
//
// The A100 vote above is only as useful as the pools it votes over. Under
// TP=2/EP=2 the head saved decode-time Marconi checkpoints from a bare local
// call and the EP protocol had no counterpart, so rank 1 never held them:
// every vote on such a checkpoint was `[T, 0]` → `agreed=None` → recompute
// (HANDOFF-30 §6/§8). These tests pin the fix's invariant — **every rank
// saves the same (slot, token, session) checkpoint, or none does** — on the
// same two-rank CPU harness, with the real wire codec in the loop.

use super::super::decode_checkpoint::{
    CkptInputs, CkptPlan, decode_ckpt_payload, decode_ckpt_plan, encode_ckpt_payload,
};

/// One rank's snapshot pool, reduced to the identity the vote cares about.
type Pool = Vec<(usize, usize, u64)>; // (slot, token, session)

const SLOT: usize = 1;
const SESSION: u64 = 0xdead_beef_0bad_f00d;
const ADAPTER_HEAD: u64 = 0x0000_0001_0000_0002;
const BS: usize = 16;
const INTERVAL: usize = 4; // ATLAS_DECODE_CKPT_BLOCKS default → 64 tokens
const CKPT_TOKENS: usize = 16_384; // 1024 complete blocks, a multiple of 4

fn inputs(enabled: bool, tokens_len: usize, last_ckpt_block: usize) -> CkptInputs {
    CkptInputs {
        enabled,
        num_ssm_layers: 12,
        hss_window_start: 0,
        slot_idx: SLOT,
        tokens_len,
        block_size: BS,
        block_table_len: tokens_len / BS,
        last_ckpt_block,
        interval: INTERVAL,
    }
}

/// Rank 0: decide, save locally, and return the bytes it puts on the wire.
/// `None` ⇒ nothing saved and **no EP command emitted**.
fn head_checkpoint(inputs: &CkptInputs, pool: &mut Pool) -> Option<[u32; 6]> {
    let plan = decode_ckpt_plan(inputs)?;
    pool.push((inputs.slot_idx, plan.snap_tokens, SESSION));
    Some(encode_ckpt_payload(plan, SESSION, ADAPTER_HEAD))
}

/// A worker rank handling `EP_CMD_DECODE_CKPT`: it re-derives nothing — it
/// saves what rank 0 says, at rank 0's `(slot, token, session)`.
fn worker_checkpoint(slot: usize, words: &[u32], pool: &mut Pool) {
    let (plan, session, _adapter) = decode_ckpt_payload(words).expect("payload decodes");
    pool.push((slot, plan.snap_tokens, session));
}

/// What this rank would propose on a later warm turn that matches `matched`
/// tokens of a `total`-token prompt: the deepest checkpoint it holds at or
/// below `matched`, run through the real A100 gate chain.
fn proposal_from(pool: &Pool, matched: usize, total: usize) -> u32 {
    let snap_tok = pool
        .iter()
        .filter(|&&(_, tok, _)| tok <= matched)
        .map(|&(_, tok, _)| tok)
        .max()
        .unwrap_or(0);
    local_proposal(&LocalGates {
        snap_tok,
        matched,
        total,
        min_tokens: 256,
        has_hidden: false,
        exact_enabled: false,
        is_tail: false,
        session_ok: true,
        needs_aux: true,
        has_aux: snap_tok != 0,
    })
}

// 9. after a checkpoint at T on rank 0, BOTH pools hold the same
//    (slot, token, session) — the A109 invariant itself.
#[test]
fn worker_saves_the_same_slot_token_session_as_rank0() {
    let (mut r0, mut r1): (Pool, Pool) = (vec![], vec![]);
    let words = head_checkpoint(&inputs(true, CKPT_TOKENS, 0), &mut r0).expect("rank 0 fires");
    worker_checkpoint(SLOT, &words, &mut r1);
    assert_eq!(r0, vec![(SLOT, CKPT_TOKENS, SESSION)]);
    assert_eq!(r1, r0, "worker pool must hold exactly rank 0's checkpoint");
}

// 10. and a later lookup at T then agrees on BOTH ranks — through the real
//     rooted-broadcast gather, not a local shortcut.
#[test]
fn symmetric_checkpoint_makes_the_later_lookup_agree() {
    let (mut r0, mut r1): (Pool, Pool) = (vec![], vec![]);
    let words = head_checkpoint(&inputs(true, CKPT_TOKENS, 0), &mut r0).expect("rank 0 fires");
    worker_checkpoint(SLOT, &words, &mut r1);

    let (matched, total) = (CKPT_TOKENS + 16, CKPT_TOKENS + 20);
    let votes = [
        proposal_from(&r0, matched, total),
        proposal_from(&r1, matched, total),
    ];
    let [seen0, seen1] = gather_two(votes);
    assert_eq!(agree(&seen0), Some(CKPT_TOKENS as u32));
    assert_eq!(agree(&seen1), Some(CKPT_TOKENS as u32));
    assert_eq!(
        skip_point(true, CKPT_TOKENS, matched, total, true),
        CKPT_TOKENS,
    );
}

// 11. the A109 shape this replaces: head-only save ⇒ votes [T, 0] ⇒ the A100
//     vote refuses (correct, and the reason hit-rate was poor).
#[test]
fn head_only_checkpoint_is_refused_by_the_vote() {
    let (mut r0, r1): (Pool, Pool) = (vec![], vec![]);
    head_checkpoint(&inputs(true, CKPT_TOKENS, 0), &mut r0).expect("rank 0 fires");
    let (matched, total) = (CKPT_TOKENS + 16, CKPT_TOKENS + 20);
    let votes = [
        proposal_from(&r0, matched, total),
        proposal_from(&r1, matched, total),
    ];
    assert_eq!(votes, [CKPT_TOKENS as u32, 0]);
    let [seen0, seen1] = gather_two(votes);
    assert_eq!(agree(&seen0), None);
    assert_eq!(agree(&seen1), None);
}

// 12. prefix cache disabled ⇒ no save, and the opcode never goes on the wire.
//     Default-OFF behaviour stays byte-identical.
#[test]
fn disabled_prefix_cache_emits_no_ep_command() {
    let (mut r0, r1): (Pool, Pool) = (vec![], vec![]);
    assert_eq!(decode_ckpt_plan(&inputs(false, CKPT_TOKENS, 0)), None);
    assert!(
        head_checkpoint(&inputs(false, CKPT_TOKENS, 0), &mut r0).is_none(),
        "no payload ⇒ no EP_CMD_DECODE_CKPT broadcast",
    );
    assert!(r0.is_empty() && r1.is_empty());
}

// 13. every other skip reason is equally silent on the wire — the command is
//     emitted iff rank 0 actually saves.
#[test]
fn every_skip_reason_emits_no_ep_command() {
    let cases: Vec<(&str, CkptInputs)> = vec![
        ("no SSM layers", {
            let mut i = inputs(true, CKPT_TOKENS, 0);
            i.num_ssm_layers = 0;
            i
        }),
        ("sliding SSM window", {
            let mut i = inputs(true, CKPT_TOKENS, 0);
            i.hss_window_start = 64;
            i
        }),
        ("no pool slot", {
            let mut i = inputs(true, CKPT_TOKENS, 0);
            i.slot_idx = usize::MAX;
            i
        }),
        (
            "not on an interval boundary",
            inputs(true, CKPT_TOKENS + BS, 0),
        ),
        ("already checkpointed this block", {
            inputs(true, CKPT_TOKENS, CKPT_TOKENS / BS)
        }),
        ("no complete block yet", inputs(true, BS - 1, 0)),
        ("block table shorter than the coverage", {
            let mut i = inputs(true, CKPT_TOKENS, 0);
            i.block_table_len = CKPT_TOKENS / BS - 1;
            i
        }),
    ];
    for (why, i) in cases {
        let mut pool: Pool = vec![];
        assert!(head_checkpoint(&i, &mut pool).is_none(), "{why}");
        assert!(pool.is_empty(), "{why}");
    }
}

// 14. the wire codec itself: a mis-ordered lo/hi word would silently retag
//     every worker checkpoint with a different session.
#[test]
fn ckpt_payload_round_trips() {
    let plan = CkptPlan {
        snap_tokens: 16_356,
        end_block: 1_022,
    };
    let words = encode_ckpt_payload(plan, SESSION, ADAPTER_HEAD);
    assert_eq!(words.len(), 6);
    let (back, session, adapter) = decode_ckpt_payload(&words).unwrap();
    assert_eq!(back, plan);
    assert_eq!(session, SESSION);
    assert_eq!(adapter, ADAPTER_HEAD);
    // A short or long payload is a protocol error, not a silent mis-parse.
    assert!(decode_ckpt_payload(&words[..5]).is_err());
    assert!(decode_ckpt_payload(&[0u32; 7]).is_err());
}
