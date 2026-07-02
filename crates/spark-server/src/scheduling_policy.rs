// SPDX-License-Identifier: AGPL-3.0-only

//! Scheduling policy trait (SDD: FIFO vs SLAI).
//!
//! Controls two decisions in the scheduler loop:
//! 1. Whether to accept new prefills or prioritize decode (TBT deadline).
//! 2. Which pending requests to prefill and in what order.
//!
//! Implementations:
//! - [`FifoPolicy`]: always prefill, take first N from queue (current behavior).
//! - [`SlaiPolicy`]: skip prefills when active sequences approach TBT deadline,
//!   select shortest prompts first from ALL pending (SLAI — arXiv:2407.08353).

use std::time::{Duration, Instant};

/// Metadata about a pending request for selection decisions.
pub struct PendingRequestInfo {
    /// Number of prompt tokens (determines prefill cost).
    pub prompt_len: usize,
    /// Index into the full pending requests vec.
    pub index: usize,
}

/// Per-sequence timing for decode urgency decisions.
pub struct ActiveSeqTiming {
    /// When the last token was emitted for this sequence.
    pub last_token_time: Instant,
}

/// Scheduling policy controlling prefill admission and ordering.
pub trait SchedulingPolicy: Send {
    /// Whether to accept new prefills this iteration.
    ///
    /// Returns `false` to skip prefill and proceed directly to decode
    /// (e.g., when active sequences approach their TBT deadline).
    fn should_prefill(&self, active_timings: &[ActiveSeqTiming]) -> bool;

    /// Select up to `capacity` requests from ALL pending, in prefill order.
    ///
    /// Returns indices into `requests` for the selected items, ordered
    /// by desired prefill execution order. FIFO takes the first N;
    /// SLAI picks the N shortest prompts.
    fn select_prefills(&self, requests: &[PendingRequestInfo], capacity: usize) -> Vec<usize>;

    /// Policy name for logging.
    fn name(&self) -> &str;
}

/// FIFO scheduling: always prefill, take first N from queue.
pub struct FifoPolicy;

impl SchedulingPolicy for FifoPolicy {
    fn should_prefill(&self, _active_timings: &[ActiveSeqTiming]) -> bool {
        true
    }

    fn select_prefills(&self, requests: &[PendingRequestInfo], capacity: usize) -> Vec<usize> {
        // First N in queue order (FIFO).
        (0..requests.len().min(capacity)).collect()
    }

    fn name(&self) -> &str {
        "fifo"
    }
}

/// SLO-aware scheduling (SLAI-inspired).
///
/// - Skips prefills when any active sequence waited > 80% of `tbt_deadline`
///   since its last token emission (decode-first priority).
/// - Selects the N shortest prompts from ALL pending (reduces median TTFT).
pub struct SlaiPolicy {
    tbt_deadline: Duration,
}

impl SlaiPolicy {
    pub fn new(tbt_deadline_ms: u64) -> Self {
        Self {
            tbt_deadline: Duration::from_millis(tbt_deadline_ms),
        }
    }
}

impl SchedulingPolicy for SlaiPolicy {
    fn should_prefill(&self, active_timings: &[ActiveSeqTiming]) -> bool {
        if active_timings.is_empty() {
            return true;
        }
        let now = Instant::now();
        let margin = self.tbt_deadline.mul_f64(0.8);
        for timing in active_timings {
            if now.duration_since(timing.last_token_time) >= margin {
                return false;
            }
        }
        true
    }

    fn select_prefills(&self, requests: &[PendingRequestInfo], capacity: usize) -> Vec<usize> {
        // Sort ALL pending by prompt_len, pick shortest N.
        let mut indices: Vec<usize> = (0..requests.len()).collect();
        indices.sort_by_key(|&i| requests[i].prompt_len);
        indices.truncate(capacity);
        indices
    }

    fn name(&self) -> &str {
        "slai"
    }
}

#[cfg(test)]
mod tests {
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
        // Capacity 3: picks shortest 3 → indices 1(10), 3(50), 2(200)
        assert_eq!(policy.select_prefills(&requests, 3), vec![1, 3, 2]);
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
        // Capacity 10 > 2 requests: returns all sorted
        assert_eq!(policy.select_prefills(&requests, 10), vec![1, 0]);
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

    #[test]
    fn select_prefills_empty() {
        assert!(FifoPolicy.select_prefills(&[], 5).is_empty());
        assert!(SlaiPolicy::new(100).select_prefills(&[], 5).is_empty());
    }
}
