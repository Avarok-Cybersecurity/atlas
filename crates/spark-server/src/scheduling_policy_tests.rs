// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for [`super`] (`scheduling_policy`). A sibling file via `#[path]`
//! — the `mtp_dcut.rs`/`mtp_dcut_tests.rs` idiom — so `scheduling_policy.rs`
//! stays under the 500-line cap. Module position (child of
//! `scheduling_policy`) is unchanged, so `super::*` paths are untouched.
use super::*;

#[test]
fn fifo_always_prefills() {
    let policy = FifoPolicy;
    let timings = vec![ActiveSeqTiming {
        last_token_time: Instant::now(),
    }];
    assert!(policy.should_prefill(&timings));
    assert!(policy.should_prefill(&[]));
}

#[test]
fn fifo_selects_first_n() {
    let policy = FifoPolicy;
    let requests = vec![
        PendingRequestInfo {
            prompt_len: 100,
            index: 0,
        },
        PendingRequestInfo {
            prompt_len: 10,
            index: 1,
        },
        PendingRequestInfo {
            prompt_len: 50,
            index: 2,
        },
        PendingRequestInfo {
            prompt_len: 200,
            index: 3,
        },
    ];
    assert_eq!(policy.select_prefills(&requests, 2), vec![0, 1]);
    assert_eq!(policy.select_prefills(&requests, 10), vec![0, 1, 2, 3]);
}

#[test]
fn slai_prefills_when_no_active() {
    let policy = SlaiPolicy::new(100);
    assert!(policy.should_prefill(&[]));
}

#[test]
fn slai_prefills_when_fresh() {
    let policy = SlaiPolicy::new(100);
    let timings = vec![ActiveSeqTiming {
        last_token_time: Instant::now(),
    }];
    assert!(policy.should_prefill(&timings));
}

#[test]
fn slai_skips_prefill_near_deadline() {
    let policy = SlaiPolicy::new(100); // 80ms margin
    let old_time = Instant::now() - Duration::from_millis(85);
    let timings = vec![ActiveSeqTiming {
        last_token_time: old_time,
    }];
    assert!(!policy.should_prefill(&timings));
}

#[test]
fn slai_prefills_within_margin() {
    let policy = SlaiPolicy::new(100); // 80ms margin
    let recent = Instant::now() - Duration::from_millis(50);
    let timings = vec![ActiveSeqTiming {
        last_token_time: recent,
    }];
    assert!(policy.should_prefill(&timings));
}

#[test]
fn slai_one_urgent_blocks_prefill() {
    let policy = SlaiPolicy::new(100);
    let now = Instant::now();
    let timings = vec![
        ActiveSeqTiming {
            last_token_time: now,
        },
        ActiveSeqTiming {
            last_token_time: now - Duration::from_millis(90),
        },
    ];
    assert!(!policy.should_prefill(&timings));
}

#[test]
fn slai_selects_shortest_from_all() {
    let policy = SlaiPolicy::new(100);
    let requests = vec![
        PendingRequestInfo {
            prompt_len: 500,
            index: 0,
        },
        PendingRequestInfo {
            prompt_len: 10,
            index: 1,
        },
        PendingRequestInfo {
            prompt_len: 200,
            index: 2,
        },
        PendingRequestInfo {
            prompt_len: 50,
            index: 3,
        },
        PendingRequestInfo {
            prompt_len: 300,
            index: 4,
        },
    ];
    // Capacity 3: seat 0 is RESERVED for the queue head (index 0, the
    // 500-token prompt — see `select_prefills`), then the shortest two
    // of the rest: 1(10) and 3(50). Before the head reservation this
    // read [1, 3, 2], which is the selection that starves index 0
    // forever under a steady arrival of shorter prompts.
    assert_eq!(policy.select_prefills(&requests, 3), vec![0, 1, 3]);
    // The SJF preference is intact on the unreserved seats: widen the
    // capacity by one and 2(200) joins, not 4(300).
    assert_eq!(policy.select_prefills(&requests, 4), vec![0, 1, 3, 2]);
    // One seat: the head takes it.
    assert_eq!(policy.select_prefills(&requests, 1), vec![0]);
    // No seats: nothing, and no panic on the `capacity - 1` arithmetic.
    assert!(policy.select_prefills(&requests, 0).is_empty());
}

#[test]
fn slai_selects_all_when_capacity_exceeds() {
    let policy = SlaiPolicy::new(100);
    let requests = vec![
        PendingRequestInfo {
            prompt_len: 100,
            index: 0,
        },
        PendingRequestInfo {
            prompt_len: 10,
            index: 1,
        },
    ];
    // Capacity 10 > 2 requests: BOTH are selected either way, so the
    // head reservation changes only the order — oldest first, then
    // shortest-first over the rest. (Was [1, 0] under pure SJF.)
    assert_eq!(policy.select_prefills(&requests, 10), vec![0, 1]);
}

#[test]
fn slai_stable_order_for_equal_lengths() {
    let policy = SlaiPolicy::new(100);
    let requests = vec![
        PendingRequestInfo {
            prompt_len: 50,
            index: 0,
        },
        PendingRequestInfo {
            prompt_len: 50,
            index: 1,
        },
        PendingRequestInfo {
            prompt_len: 50,
            index: 2,
        },
    ];
    assert_eq!(policy.select_prefills(&requests, 3), vec![0, 1, 2]);
}

/// Drive a policy the way the scheduler does: each tick it selects up
/// to `capacity`, the selected requests LEAVE the queue, and a new
/// short request arrives. Returns the tick on which `watch` (tracked by
/// its `index` field, which is stable across the run) was selected, or
/// `None` if it never was.
///
/// This is the arrival pattern a busy serve actually sees: a long
/// prompt queued behind a steady stream of short ones.
fn ticks_until_selected(
    policy: &dyn SchedulingPolicy,
    watch: usize,
    capacity: usize,
    ticks: usize,
) -> Option<usize> {
    // index 0 = the long prompt under test; the queue is in ARRIVAL
    // order, which is what the scheduler hands the policy.
    let mut queue: Vec<usize> = vec![5000];
    let mut ids: Vec<usize> = vec![watch];
    for tick in 0..ticks {
        // Arrivals land at the BACK of the queue BEFORE this tick's
        // selection — the ordering that makes the starvation reachable
        // at all. `capacity` of them, so the queue never drains and the
        // policy always has a full menu of shorter jobs to prefer.
        for a in 0..capacity.max(1) {
            queue.push(10);
            ids.push(watch + 1 + tick * capacity.max(1) + a);
        }
        let infos: Vec<PendingRequestInfo> = queue
            .iter()
            .enumerate()
            .map(|(i, &prompt_len)| PendingRequestInfo {
                prompt_len,
                index: i,
            })
            .collect();
        let sel = policy.select_prefills(&infos, capacity);
        if sel.iter().any(|&i| ids[i] == watch) {
            return Some(tick);
        }
        let mut rm = sel.clone();
        rm.sort_unstable_by(|a, b| b.cmp(a));
        for i in rm {
            queue.remove(i);
            ids.remove(i);
        }
    }
    None
}

// A queued long prompt must eventually be prefilled. Pure
// shortest-job-first re-sorts it to the tail on EVERY tick, so under a
// steady arrival of shorter requests it is never selected — an
// unbounded wait, not a slow one. The bound is the queue position, so
// the head of the queue must be selected on the very first tick.
#[test]
fn slai_does_not_starve_a_long_prompt_behind_short_arrivals() {
    let policy = SlaiPolicy::new(100);
    assert_eq!(
        ticks_until_selected(&policy, 0, 1, 500),
        Some(0),
        "the oldest pending request must be selected immediately"
    );
}

// Same guarantee at wider capacity: reserving a seat for the queue head
// must not depend on the batch being narrow.
#[test]
fn slai_head_reservation_holds_at_every_capacity() {
    let policy = SlaiPolicy::new(100);
    for capacity in 1..=8usize {
        assert_eq!(
            ticks_until_selected(&policy, 0, capacity, 200),
            Some(0),
            "starved at capacity {capacity}"
        );
    }
}

// FIFO is bounded by construction — pinned so the harness above is
// shown to detect selection, not merely to return Some for anything.
#[test]
fn fifo_never_starves_the_head() {
    assert_eq!(ticks_until_selected(&FifoPolicy, 0, 1, 10), Some(0));
}

#[test]
fn select_prefills_empty() {
    assert!(FifoPolicy.select_prefills(&[], 5).is_empty());
    assert!(SlaiPolicy::new(100).select_prefills(&[], 5).is_empty());
}

#[test]
fn fifo_slice_budget_is_full_chunk() {
    // Default trait impl: FIFO always injects the full chunk.
    let policy = FifoPolicy;
    assert_eq!(policy.prefill_slice_budget(&[], 4080), 4080);
    let timings = vec![ActiveSeqTiming {
        last_token_time: Instant::now(),
    }];
    assert_eq!(policy.prefill_slice_budget(&timings, 4080), 4080);
}

#[test]
fn slai_slice_budget_full_when_no_active() {
    let policy = SlaiPolicy::new(100);
    assert_eq!(policy.prefill_slice_budget(&[], 4080), 4080);
}

#[test]
fn slai_slice_budget_zero_past_deadline() {
    // worst >= tbt_deadline → hard suppress (0), decode-only this tick.
    let policy = SlaiPolicy::new(100);
    let timings = vec![ActiveSeqTiming {
        last_token_time: Instant::now() - Duration::from_millis(120),
    }];
    assert_eq!(policy.prefill_slice_budget(&timings, 4080), 0);
}

#[test]
fn slai_slice_budget_bounded_and_wy4_aligned() {
    // Fresh decode, under deadline → a positive, WY4-aligned slice in
    // [min, full_chunk], never 0.
    let policy = SlaiPolicy::new(100);
    let timings = vec![ActiveSeqTiming {
        last_token_time: Instant::now(),
    }];
    let b = policy.prefill_slice_budget(&timings, 4080);
    assert!(b > 0, "non-deadline budget must be > 0");
    assert!(b <= 4080, "must never exceed full_chunk");
    assert_eq!(b % 4, 0, "must be WY4-aligned");
}

#[test]
fn slai_slice_budget_never_exceeds_small_full_chunk() {
    // Small chunk cap must clamp the slice (and stay WY4-aligned).
    let policy = SlaiPolicy::new(100);
    let timings = vec![ActiveSeqTiming {
        last_token_time: Instant::now(),
    }];
    let b = policy.prefill_slice_budget(&timings, 64);
    assert!(b <= 64);
    assert_eq!(b % 4, 0);
}
