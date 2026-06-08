// SPDX-License-Identifier: AGPL-3.0-only
//
// ProcessQueue — the Earley parser's per-advance processing queue plus
// the visited-set deduplicator. Port of `RepeatDetector` and the
// `tmp_process_state_queue_` / `tmp_states_visited_in_queue_` machinery
// from `cpp/earley_parser.{h,cc}`.

use std::collections::VecDeque;

use ahash::AHashSet;

use super::state::ParserState;

/// Below this many distinct states the C++ `RepeatDetector` scans a
/// small vector; above it, it switches to a hash set. We always use a
/// hash set — `AHashSet` makes the small-N case cheap enough that the
/// dual representation is not worth the complexity. Behavior (which
/// states are reported visited) is identical.
///
/// FIFO queue of states awaiting predict/complete, with an attached
/// visited set so a state is enqueued at most once per advance round.
#[derive(Debug, Default)]
pub struct ProcessQueue {
    queue: VecDeque<ParserState>,
    visited: AHashSet<ParserState>,
}

impl ProcessQueue {
    /// An empty queue.
    pub fn new() -> Self {
        Self::default()
    }

    /// Clear the visited set (called at the start of each advance).
    /// The queue itself is asserted empty by the caller.
    pub fn clear_visited(&mut self) {
        self.visited.clear();
    }

    /// True if `state` has already been seen this round.
    pub fn is_visited(&self, state: &ParserState) -> bool {
        self.visited.contains(state)
    }

    /// Enqueue `state` for predict/complete unless already visited.
    /// Returns true if it was newly enqueued.
    pub fn enqueue(&mut self, state: ParserState) -> bool {
        if self.visited.insert(state) {
            self.queue.push_back(state);
            true
        } else {
            false
        }
    }

    /// Mark `state` visited without enqueuing it for processing.
    /// Returns true if it was newly marked. Mirrors the C++
    /// `EnqueueWithoutProcessing` visited-set side effect; the caller
    /// is responsible for adding the state to the scanable list.
    pub fn mark_visited(&mut self, state: ParserState) -> bool {
        self.visited.insert(state)
    }

    /// Pop the next state to process, or `None` when drained.
    pub fn pop(&mut self) -> Option<ParserState> {
        self.queue.pop_front()
    }

    /// True if no states are pending.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st(id: i32) -> ParserState {
        ParserState::new(id, 0, 0, 0, 0)
    }

    #[test]
    fn enqueue_dedups() {
        let mut q = ProcessQueue::new();
        assert!(q.enqueue(st(1)));
        assert!(!q.enqueue(st(1)));
        assert!(q.enqueue(st(2)));
        assert_eq!(q.pop(), Some(st(1)));
        assert_eq!(q.pop(), Some(st(2)));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn fifo_order() {
        let mut q = ProcessQueue::new();
        for i in 0..5 {
            q.enqueue(st(i));
        }
        for i in 0..5 {
            assert_eq!(q.pop(), Some(st(i)));
        }
    }

    #[test]
    fn clear_visited_allows_reenqueue() {
        let mut q = ProcessQueue::new();
        q.enqueue(st(1));
        q.pop();
        assert!(!q.enqueue(st(1)));
        q.clear_visited();
        assert!(q.enqueue(st(1)));
    }

    #[test]
    fn mark_visited_blocks_enqueue() {
        let mut q = ProcessQueue::new();
        assert!(q.mark_visited(st(7)));
        assert!(!q.mark_visited(st(7)));
        assert!(!q.enqueue(st(7)));
        assert!(q.is_visited(&st(7)));
    }
}
