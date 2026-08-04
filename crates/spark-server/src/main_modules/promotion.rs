// SPDX-License-Identifier: AGPL-3.0-only

//! Task #27: demand-driven RDMA adapter promotion — the HTTP-side pieces.
//!
//! A request naming a STAGEABLE (promotable-but-not-resident) adapter triggers
//! an on-miss RDMA promotion of that adapter from the `$ATLAS_LORA_PEER` weight
//! peer into a cache pool slot, then routes to it (instead of a 404). The RDMA
//! stage + victim selection run on the scheduler thread at a quiescent point
//! (see [`crate::scheduler::LoraCommand::Promote`]); everything in THIS module
//! lives above that boundary in [`crate::AppState`]:
//!
//!   * the STAGEABLE REGISTRY (`name -> {peer_stage_id, peft}`),
//!   * the promoted-name -> cache-slot overlay,
//!   * and the LOAD-COALESCING single-flight so N concurrent misses for the
//!     SAME cold adapter collapse to ONE promote.

use std::collections::HashMap;
use std::sync::Mutex;

use tokio::sync::oneshot;

/// A stageable (not-yet-resident) adapter: its id on the weight peer plus the
/// peft config the peer manifest does not carry (r/alpha/scaling, parsed from
/// the CLI `CONFIG_DIR/adapter_config.json` once at startup).
#[derive(Clone, Debug)]
pub struct StageableAdapter {
    pub peer_stage_id: String,
    pub peft: atlas_core::config::PeftAdapterConfig,
}

/// Why a demand-promotion did not yield a slot. Cloned to every coalesced
/// waiter, so it is `Clone`. Mapped to an HTTP status by the caller:
/// `PoolFull` → 503 (retryable), `Peer` → 502 (upstream/RDMA error).
#[derive(Clone, Debug)]
pub enum PromoteReject {
    /// Every cache slot is busy, or the promote timed out waiting for scheduler
    /// quiescence — retry once in-flight work drains.
    PoolFull(String),
    /// The peer/RDMA stage or the control channel failed.
    Peer(String),
}

impl std::fmt::Display for PromoteReject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PromoteReject::PoolFull(m) | PromoteReject::Peer(m) => f.write_str(m),
        }
    }
}

/// Single-flight coordinator for demand promotions. One entry per adapter name
/// currently being promoted; the FIRST caller for a name becomes the LEADER and
/// runs the (single) promote, later callers become FOLLOWERS that await the
/// leader's result over a per-follower oneshot.
///
/// The inner `Mutex` is held ONLY for the map insert/remove — NEVER across the
/// leader's `.await` (the scheduler round-trip). The scheduler thread never
/// touches this lock, so there is no lock cycle and no deadlock (constraint c).
/// Followers parked on one in-flight promotion, each awaiting the leader's
/// result over its own oneshot.
type PromoteWaiters = Vec<oneshot::Sender<Result<i32, PromoteReject>>>;

#[derive(Default)]
pub struct PromotionManager {
    inflight: Mutex<HashMap<String, PromoteWaiters>>,
}

enum Role {
    Leader,
    Follower(oneshot::Receiver<Result<i32, PromoteReject>>),
}

/// RAII leadership guard. The leader claims the map entry, then runs its promote
/// across an `.await`. If that future is CANCELLED (the axum handler task is
/// dropped on client disconnect) or PANICS, this guard's `Drop` removes the map
/// entry so the adapter name is never permanently wedged; dropping the waiter
/// `Sender`s wakes any followers with a `RecvError` (recoverable — the next miss
/// simply re-leads). On the happy path the leader calls `disarm()` after it has
/// already removed the entry + broadcast, so `Drop` is a no-op.
struct LeaderGuard<'a> {
    mgr: &'a PromotionManager,
    name: String,
    armed: bool,
}

impl LeaderGuard<'_> {
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for LeaderGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            // Cancelled/panicked before broadcasting: clear the entry (dropping
            // the waiter Senders → followers get RecvError → recoverable).
            let _ = self.mgr.inflight.lock().unwrap().remove(&self.name);
        }
    }
}

impl PromotionManager {
    /// Coalesced promote: if a promote for `name` is already in flight, await its
    /// result; otherwise become the leader, run `leader` exactly once, then
    /// broadcast the (cloned) result to every waiter that joined meanwhile and
    /// remove the entry (on BOTH success and failure — no leak, no poison).
    ///
    /// `leader` is the actual promote round-trip; it is invoked at most once per
    /// concurrent burst for a given name. Distinct names run independent
    /// leaders.
    pub async fn coalesce<F, Fut>(&self, name: &str, leader: F) -> Result<i32, PromoteReject>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<i32, PromoteReject>>,
    {
        let role = {
            let mut map = self.inflight.lock().unwrap();
            match map.get_mut(name) {
                Some(waiters) => {
                    let (tx, rx) = oneshot::channel();
                    waiters.push(tx);
                    Role::Follower(rx)
                }
                None => {
                    // Insert the (empty) waiter list to claim leadership; drop the
                    // lock before running the promote so followers can register.
                    map.insert(name.to_string(), Vec::new());
                    Role::Leader
                }
            }
        };

        match role {
            Role::Follower(rx) => rx.await.unwrap_or_else(|_| {
                Err(PromoteReject::Peer(
                    "promotion leader dropped without a result".to_string(),
                ))
            }),
            Role::Leader => {
                // Arm the cancel/panic-safe guard BEFORE the await so a dropped
                // handler task cannot leave the entry wedged.
                let mut guard = LeaderGuard {
                    mgr: self,
                    name: name.to_string(),
                    armed: true,
                };
                let result = leader().await;
                // Remove the entry and notify everyone who joined while we ran,
                // then disarm so the guard's Drop is a no-op.
                let waiters = self
                    .inflight
                    .lock()
                    .unwrap()
                    .remove(name)
                    .unwrap_or_default();
                guard.disarm();
                for w in waiters {
                    let _ = w.send(result.clone());
                }
                result
            }
        }
    }

    /// Test-only: number of in-flight entries (should be 0 at rest).
    #[cfg(test)]
    fn inflight_len(&self) -> usize {
        self.inflight.lock().unwrap().len()
    }
}

/// Pure decision for constraint (a): a miss should only ATTEMPT a promote when
/// promotion is enabled AND the name is registered stageable; otherwise the
/// handler falls through to the byte-identical resident-only 400.
pub fn should_attempt_promote(promotion_enabled: bool, is_stageable: bool) -> bool {
    promotion_enabled && is_stageable
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn attempt_gate() {
        assert!(should_attempt_promote(true, true));
        assert!(!should_attempt_promote(false, true)); // no peer/registry
        assert!(!should_attempt_promote(true, false)); // unknown name
        assert!(!should_attempt_promote(false, false));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn single_flight_same_name_one_promote() {
        let mgr = Arc::new(PromotionManager::default());
        let calls = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..100 {
            let mgr = Arc::clone(&mgr);
            let calls = Arc::clone(&calls);
            handles.push(tokio::spawn(async move {
                mgr.coalesce("sparky", || async {
                    // The leader holds the entry open long enough for all
                    // followers to register (they coalesce onto this one call).
                    calls.fetch_add(1, Ordering::SeqCst);
                    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                    Ok(5)
                })
                .await
            }));
        }
        let results: Vec<_> = futures::future::join_all(handles).await;
        // Exactly ONE promote for 100 concurrent same-name misses.
        assert_eq!(calls.load(Ordering::SeqCst), 1, "coalesced to one promote");
        // All callers resolve to the SAME slot.
        for r in results {
            assert_eq!(r.unwrap().unwrap(), 5);
        }
        // No leaked/poisoned entry.
        assert_eq!(mgr.inflight_len(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn single_flight_failure_broadcasts_and_clears() {
        let mgr = Arc::new(PromotionManager::default());
        let calls = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..50 {
            let mgr = Arc::clone(&mgr);
            let calls = Arc::clone(&calls);
            handles.push(tokio::spawn(async move {
                mgr.coalesce("cold", || async {
                    calls.fetch_add(1, Ordering::SeqCst);
                    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                    Err::<i32, _>(PromoteReject::PoolFull("all busy".to_string()))
                })
                .await
            }));
        }
        let results: Vec<_> = futures::future::join_all(handles).await;
        assert_eq!(calls.load(Ordering::SeqCst), 1, "one promote attempt");
        for r in results {
            match r.unwrap() {
                Err(PromoteReject::PoolFull(m)) => assert_eq!(m, "all busy"),
                other => panic!("expected same PoolFull, got {other:?}"),
            }
        }
        // Entry removed on failure too (no poison) — a later miss re-leads.
        assert_eq!(mgr.inflight_len(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn distinct_names_promote_independently() {
        let mgr = Arc::new(PromotionManager::default());
        let calls = Arc::new(AtomicUsize::new(0));
        let names = ["a", "b", "c"];
        let mut handles = Vec::new();
        for n in names {
            for _ in 0..10 {
                let mgr = Arc::clone(&mgr);
                let calls = Arc::clone(&calls);
                handles.push(tokio::spawn(async move {
                    mgr.coalesce(n, || async {
                        calls.fetch_add(1, Ordering::SeqCst);
                        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
                        Ok(7)
                    })
                    .await
                }));
            }
        }
        let _ = futures::future::join_all(handles).await;
        // One promote PER distinct name.
        assert_eq!(calls.load(Ordering::SeqCst), names.len());
        assert_eq!(mgr.inflight_len(), 0);
    }
}
