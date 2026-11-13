// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the host tier's notion of time (lease deadlines, lazy-store deferral, host
//! fallback timing), behind a trait so tests drive it by hand.

use std::time::{Duration, Instant};

/// A monotonic clock: time since an arbitrary origin.
pub trait Clock: Send + Sync {
    fn now(&self) -> Duration;
}

/// The real clock: time since the tier was created.
pub struct MonotonicClock {
    origin: Instant,
}

impl MonotonicClock {
    pub fn new() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl Default for MonotonicClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for MonotonicClock {
    fn now(&self) -> Duration {
        self.origin.elapsed()
    }
}
