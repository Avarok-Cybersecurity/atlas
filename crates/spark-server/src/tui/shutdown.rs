// SPDX-License-Identifier: AGPL-3.0-only

//! One shutdown path for every trigger.
//!
//! Sources that request shutdown:
//!  * plain mode: `SIGINT` (Ctrl+C) / `SIGTERM` via the tokio signal listener
//!    installed by [`install_signal_listeners`] — nothing handled these before
//!    this module; the process just died mid-write.
//!  * TUI mode: raw mode swallows `SIGINT`, so Ctrl+C arrives as a KEY event —
//!    the event loop calls [`request`]. (`SIGTERM`, e.g. `docker stop`, still
//!    arrives as a signal and is caught by the same listener.)
//!  * `/quit` in the Terminal tab.
//!
//! Effect: the router's accept loop (which `select!`s on [`wait`]) stops
//! accepting, waits for in-flight requests to drain (bounded grace), and
//! returns — unwinding `serve()` normally so Drop impls (TerminalGuard, tee
//! flush) run. That IS the clean shutdown: no `exit()` shortcuts.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use tokio::sync::{Notify, oneshot};

static REQUESTED: AtomicBool = AtomicBool::new(false);

fn notify() -> &'static Notify {
    static N: OnceLock<Notify> = OnceLock::new();
    N.get_or_init(Notify::new)
}

/// Startup escape hatch: the sending half of `main`'s one-shot, held so that
/// whichever trigger fires FIRST takes it and the rest are no-ops.
///
/// `serve()` is `async` but never awaits between the banner and the accept loop
/// — startup is one blocking sequence (weight load, KV alloc, kernel audit). So
/// the accept loop's `select!` on [`wait`] cannot be reached while a model is
/// loading, and Ctrl+C looked ignored for the whole load. `main` now races the
/// spawned `serve()` task against this channel, which needs no cooperation from
/// the blocking code at all.
///
/// It is DISARMED by [`disarm_startup_escape`] the moment the accept loop takes
/// over. After that a shutdown must go through the graceful path below, or a
/// Ctrl+C on a live server would exit while requests were still in flight.
static STARTUP_ESCAPE: OnceLock<std::sync::Mutex<Option<oneshot::Sender<&'static str>>>> =
    OnceLock::new();

fn escape() -> &'static std::sync::Mutex<Option<oneshot::Sender<&'static str>>> {
    STARTUP_ESCAPE.get_or_init(|| std::sync::Mutex::new(None))
}

/// Whether a shutdown request should still take the startup escape. Cleared when
/// the accept loop takes over; the SENDER itself is deliberately kept alive (see
/// [`disarm_startup_escape`]).
static IN_STARTUP: AtomicBool = AtomicBool::new(true);

/// Arm the startup escape with `main`'s sender. Called once, before `serve()`.
pub fn arm_startup_escape(tx: oneshot::Sender<&'static str>) {
    *escape().lock().expect("shutdown escape poisoned") = Some(tx);
}

/// Close the startup escape once the server is accepting: from here on, shutdown
/// means "stop accepting and drain", not "return from main".
///
/// This only flips a flag — it must NOT drop the sender. Dropping it closes the
/// channel, which resolves `main`'s receiver with `Err(RecvError)`; that is
/// indistinguishable from a shutdown unless the receiving side special-cases it,
/// and taking it at face value exits a healthy server the instant it comes up.
/// Parking the sender here for the life of the process means the channel simply
/// never closes and there is no such edge to get wrong.
pub fn disarm_startup_escape() {
    IN_STARTUP.store(false, Ordering::SeqCst);
}

/// Request a clean shutdown. Idempotent; safe from any thread.
pub fn request(reason: &'static str) {
    if !REQUESTED.swap(true, Ordering::SeqCst) {
        tracing::info!("Shutdown requested ({reason}) — draining in-flight requests");
    }
    // Still in startup? Hand the reason to main's select! and let it unwind
    // there. Taking the sender is the one legitimate way it is consumed — only
    // the first trigger sends, and the process is on its way out. Once the accept
    // loop owns shutdown the sender stays parked and untouched, so the channel
    // never closes and the notification below is what does the work.
    if IN_STARTUP.load(Ordering::SeqCst)
        && let Ok(mut slot) = escape().lock()
        && let Some(tx) = slot.take()
    {
        let _ = tx.send(reason);
    }
    notify().notify_waiters();
}

/// Has shutdown been requested?
pub fn requested() -> bool {
    REQUESTED.load(Ordering::SeqCst)
}

/// Resolve when shutdown is requested (immediately if it already was).
pub async fn wait() {
    if requested() {
        return;
    }
    // Register interest BEFORE re-checking to close the race with `request`.
    let notified = notify().notified();
    if requested() {
        return;
    }
    notified.await;
}

/// Install `SIGINT` + `SIGTERM` listeners on the tokio runtime. Called once
/// from `serve()`; both modes need it (`SIGTERM` is how `docker stop` speaks
/// even when the TUI owns the keyboard).
pub fn install_signal_listeners() {
    tokio::spawn(async {
        let ctrl_c = tokio::signal::ctrl_c();
        #[cfg(unix)]
        {
            let mut term =
                match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::warn!("SIGTERM listener unavailable: {e}");
                        if ctrl_c.await.is_ok() {
                            request("SIGINT");
                        }
                        return;
                    }
                };
            tokio::select! {
                r = ctrl_c => { if r.is_ok() { request("SIGINT"); } }
                _ = term.recv() => { request("SIGTERM"); }
            }
        }
        #[cfg(not(unix))]
        {
            if ctrl_c.await.is_ok() {
                request("SIGINT");
            }
        }
    });
}

/// Post-accept-loop drain: wait until no requests are active or the grace
/// window expires. Uses the existing `REQUESTS_ACTIVE` gauge — no new
/// scheduler plumbing, and honest about what "drained" means.
pub async fn drain_in_flight(grace: Duration) {
    let start = std::time::Instant::now();
    loop {
        let active = crate::metrics::REQUESTS_ACTIVE.get();
        if active <= 0 {
            tracing::info!("Drain complete — no active requests");
            return;
        }
        if start.elapsed() >= grace {
            tracing::warn!(
                "Drain grace ({}s) expired with {active} request(s) still active — exiting",
                grace.as_secs()
            );
            return;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}
