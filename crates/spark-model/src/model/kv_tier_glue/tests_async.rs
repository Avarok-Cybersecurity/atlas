// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: CPU oracles of the async restore path (lane L27's parked requests) and of the
//! per-model slot, on the mock rig of `tests.rs`: an async restore stays out of the radix tree
//! while its copy runs, enters it on `Ready` with the bytes the synchronous path restores, and
//! the leased in-flight cap declines a further restore; settling a running restore waits and
//! inserts; teardown drops every held reference and never rebuilds; a refused build is final.

use spark_runtime::kv_host_tier::{RestoreState, RestoreTicket, TierOptions};
use spark_runtime::prefix_cache::PrefixCache;

use super::plan::PrefetchOutcome;
use super::restore::Env;
use super::state::{Live, TierState};
use super::tests::{BS, Rig, prompt};

impl Rig {
    fn begin(&mut self, tokens: &[u32]) -> RestoreTicket {
        let mut env = Env {
            gpu: &self.gpu,
            kv: &mut self.kv,
            cache: &self.cache,
            has_ssm: false,
        };
        self.live
            .restore_begin(&mut env, tokens, 0, 0)
            .unwrap()
            .expect("a restore starts")
    }

    fn poll(&mut self, t: &RestoreTicket) -> RestoreState {
        let mut env = Env {
            gpu: &self.gpu,
            kv: &mut self.kv,
            cache: &self.cache,
            has_ssm: false,
        };
        self.live.restore_poll(&mut env, t).unwrap()
    }

    fn settle(&mut self, t: RestoreTicket) {
        let mut env = Env {
            gpu: &self.gpu,
            kv: &mut self.kv,
            cache: &self.cache,
            has_ssm: false,
        };
        self.live.settle(&mut env, t);
    }
}

#[test]
fn the_async_path_restores_what_the_sync_path_restores_and_respects_the_cap() {
    // Budget 24 blocks: the in-flight cap is 12, two restores of 6 lease all of it.
    let mut rig = Rig::new(64, 24, TierOptions::default());
    let a = prompt(6, 10_000);
    let b = prompt(6, 20_000);
    let (_, bytes_a) = rig.prefill(&a, true);
    let (_, bytes_b) = rig.prefill(&b, true);
    rig.reap();
    rig.evict_all();
    rig.scribble_free();
    rig.gpu.set_unique_events(true);
    rig.gpu.hold_events(true);
    let ta = rig.begin(&a);
    let tb = rig.begin(&b);
    // Not in the radix tree while the copy runs: a parked request's blocks stay invisible.
    assert_eq!(rig.cache.peek_matched_tokens(&a, BS, 0), 0);
    assert_eq!(rig.poll(&ta), RestoreState::Pending);
    // The whole in-flight cap is leased: another restore is declined, not queued.
    assert_eq!(rig.prefetch(&a, false).outcome, PrefetchOutcome::Declined);
    rig.gpu.complete_all_events();
    assert_eq!(rig.poll(&ta), RestoreState::Ready);
    assert_eq!(
        rig.cache.peek_matched_tokens(&a, BS, 0),
        6 * BS,
        "inserted on Ready"
    );
    rig.settle(ta);
    rig.settle(tb);
    assert!(rig.live.books.lock().pending.is_empty());
    assert_eq!(
        rig.gpu_hit(&a),
        bytes_a,
        "same bytes as the synchronous path"
    );
    assert_eq!(rig.gpu_hit(&b), bytes_b);
}

#[test]
fn settling_a_running_restore_waits_and_inserts() {
    let mut rig = Rig::eager();
    let tokens = prompt(3, 11_000);
    let (_, bytes) = rig.prefill(&tokens, true);
    rig.reap();
    rig.evict_all();
    rig.gpu.set_unique_events(true);
    rig.gpu.hold_events(true);
    let ticket = rig.begin(&tokens);
    // The request was cancelled while parked: settle without a Ready poll.
    rig.settle(ticket);
    assert!(rig.live.books.lock().pending.is_empty());
    assert_eq!(rig.gpu_hit(&tokens), bytes);
}

#[test]
fn teardown_drops_every_held_reference() {
    let mut rig = Rig::eager();
    rig.gpu.set_unique_events(true);
    rig.gpu.hold_events(true);
    let tokens = prompt(3, 12_000);
    let (table, _) = rig.prefill(&tokens, true);
    assert_eq!(rig.held(), 3);
    let state = TierState::default();
    let spare = Live::new(rig.live.tier.clone(), 0, BS, None);
    let live = std::mem::replace(&mut rig.live, spare);
    assert!(state.get_or_build(|| Ok(Some(live))).is_some());
    state.teardown(&rig.gpu, &mut rig.kv);
    for &b in &table {
        assert_eq!(rig.kv.ref_count(b), 1, "only the cache's reference is left");
    }
    assert!(state.live().is_none());
    assert!(state.refused(), "a torn-down tier never rebuilds");
    assert!(state.get_or_build(|| panic!("must not rebuild")).is_none());
    state.teardown(&rig.gpu, &mut rig.kv);
}

#[test]
fn a_refused_build_is_not_retried() {
    let state = TierState::default();
    assert!(state.get_or_build(|| Ok(None)).is_none());
    assert!(state.refused());
    assert!(state.get_or_build(|| panic!("must not retry")).is_none());
    let failed = TierState::default();
    assert!(failed.get_or_build(|| anyhow::bail!("no pool")).is_none());
    assert!(failed.refused());
}
