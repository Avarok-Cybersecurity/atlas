// SPDX-License-Identifier: AGPL-3.0-only

//! One way to do work off the render thread.
//!
//! The dashboard's rule is that **the render thread never polls a future; its
//! only contact with work happening elsewhere is `try_recv` on a channel**
//! (`.github/workflows/tui-threading.yml` enforces it). This is the helper
//! that makes following the rule shorter than breaking it.
//!
//! Before this existed, four callers — the recipe index fetch, the recipe date
//! lookup, the library scan and the freshness check — each hand-rolled the
//! same twelve lines: build a channel, name a thread, clone the sender, and
//! answer anyway if the spawn failed. The last part is the one worth sharing:
//! a caller that forgets it leaves the UI polling a receiver that will never
//! produce anything, which renders as a spinner that spins forever.

use std::sync::mpsc::{Receiver, channel};

/// Run `work` on a named thread; the result arrives on the returned channel.
///
/// `on_spawn_failure` supplies the answer when the thread cannot be created,
/// so **the receiver always resolves** and the caller needs no second code
/// path for "the OS said no". Callers poll with `try_recv` and treat
/// `Disconnected` as "the producer is gone" — the existing idiom everywhere in
/// this module tree.
pub fn spawn<T, W, F>(name: &str, work: W, on_spawn_failure: F) -> Receiver<T>
where
    T: Send + 'static,
    W: FnOnce() -> T + Send + 'static,
    F: FnOnce(std::io::Error) -> T,
{
    let (tx, rx) = channel();
    let spawned = std::thread::Builder::new().name(name.to_string()).spawn({
        let tx = tx.clone();
        move || {
            // A disconnected receiver means the UI moved on, not an error.
            let _ = tx.send(work());
        }
    });
    if let Err(e) = spawned {
        tracing::warn!("could not spawn {name}: {e}");
        let _ = tx.send(on_spawn_failure(e));
    }
    rx
}

#[cfg(test)]
#[path = "worker_tests.rs"]
mod tests;
