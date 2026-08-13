// SPDX-License-Identifier: AGPL-3.0-only

//! Bounded stream-send helpers — the scheduler-thread → client-channel
//! backpressure SSOT. Split from `mod_helpers.rs` (500-LoC cap).

/// Deadline for a scheduler-thread send into a FULL stream channel.
///
/// Default 5000 ms; override `ATLAS_STREAM_SEND_DEADLINE_MS` (strict integer).
/// Rationale for a default at all (PCND): the pre-existing behaviour was an
/// UNBOUNDED `blocking_send`, i.e. an implicit deadline of infinity — the most
/// dangerous possible default. 5 s is >5000x the per-token cadence and only
/// triggers after the 1024-deep channel is ALSO full, so a consumer that hits
/// it has been unresponsive for thousands of events.
fn stream_send_deadline() -> std::time::Duration {
    // STATIC, DELIBERATELY — transport configuration. This bounds how long a
    // send to a CLIENT's stream channel may block; it is a property of the
    // HTTP transport, not of the model behind it, and the two send helpers
    // that read it are called from every emit site with no scheduler context
    // in hand. Reaching one would mean threading a carrier through the whole
    // emit path to configure a socket timeout.
    static MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    std::time::Duration::from_millis(*MS.get_or_init(|| {
        std::env::var("ATLAS_STREAM_SEND_DEADLINE_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5000)
    }))
}

/// Bounded send from the scheduler/GPU thread into a per-request stream
/// channel — the SSOT replacing every direct `blocking_send` in the scheduler.
///
/// Why this exists: `emit_token` and its siblings run ON the scheduler thread,
/// which also drives every GPU step. The old backpressure arm fell back to an
/// unbounded `tx.blocking_send`, so ONE stalled SSE consumer (channel full =
/// 1024 undelivered events) froze the entire engine — all concurrent
/// sequences, prefill and decode alike — for as long as that client stayed
/// wedged. At C=1 that stalls one user's own request; at C=16 it is a
/// denial-of-service pivot. There were TWELVE direct `blocking_send` call
/// sites scattered across the scheduler, so the bound lives here once.
///
/// Semantics: identical to the old path until the deadline — try, then poll
/// the full channel at 1 ms — after which the send is abandoned and `false` is
/// returned, which every caller already treats exactly like the
/// receiver-dropped case (finish/cancel that sequence). A consumer that is
/// both 1024 events behind AND unresponsive for the whole deadline is
/// indistinguishable from a dead one; delivering its backlog later cannot
/// matter.
pub(in crate::scheduler) fn bounded_stream_send(
    tx: &tokio::sync::mpsc::Sender<crate::api::inference_types::StreamEvent>,
    event: crate::api::inference_types::StreamEvent,
    what: &str,
) -> bool {
    use tokio::sync::mpsc::error::TrySendError;
    let mut event = match tx.try_send(event) {
        Ok(()) => return true,
        Err(TrySendError::Closed(_)) => {
            // INFO, not debug. This is the client hanging up mid-stream, and it
            // is the single most common reason a generation "just stops" — but
            // at debug it is invisible at the default level, so the server log
            // shows a `Request:` line, then nothing: no completion, no error.
            // That is indistinguishable from a hung server, and on 2026-08-07
            // it cost a real debugging session chasing a stall that had not
            // happened. One line per abandoned request is the right price for
            // being able to tell "they left" from "we wedged".
            tracing::info!("stream receiver dropped — client went away ({what})");
            return false;
        }
        Err(TrySendError::Full(ev)) => ev,
    };
    let deadline = std::time::Instant::now() + stream_send_deadline();
    loop {
        std::thread::sleep(std::time::Duration::from_millis(1));
        match tx.try_send(event) {
            Ok(()) => return true,
            Err(TrySendError::Closed(_)) => {
                // Same event, later: they hung up while we were waiting on a
                // full channel. Same reasoning, same level.
                tracing::info!(
                    "stream receiver dropped during backpressure — client went away ({what})"
                );
                return false;
            }
            Err(TrySendError::Full(ev)) => {
                if std::time::Instant::now() >= deadline {
                    tracing::warn!(
                        "stream consumer stalled past {:?} with a full channel — abandoning ({what})",
                        stream_send_deadline()
                    );
                    return false;
                }
                event = ev;
            }
        }
    }
}

/// Tokio runtime handle captured at serve startup (async context), used by
/// [`spawn_terminal_send`] from the scheduler OS thread — where
/// `Handle::try_current()` would fail because the thread is not a runtime
/// worker. Unset (tests, exotic embeddings) falls back to the bounded
/// synchronous send.
static RUNTIME_HANDLE: std::sync::OnceLock<tokio::runtime::Handle> = std::sync::OnceLock::new();

/// Capture the current tokio runtime handle. Call from async context BEFORE
/// spawning the scheduler thread. Idempotent.
pub fn capture_runtime_handle() {
    let _ = RUNTIME_HANDLE.set(tokio::runtime::Handle::current());
}

/// Fire-and-forget send for a TERMINAL stream event (Done / Error) — the last
/// event a sequence's channel will ever carry.
///
/// Terminal frames are the only events that may be detached from the scheduler
/// thread without an ordering hazard: every earlier token was already QUEUED
/// (its send returned true before this is called) and nothing follows, so the
/// FIFO channel delivers the spawned send after the full backlog regardless of
/// when the task runs. Mid-stream events must NOT go through here — a spawned
/// first-token send still waiting for capacity would race the next step's
/// `try_send` of token #2 and deliver out of order.
///
/// The spawned task still applies [`stream_send_deadline`] via
/// `tokio::time::timeout` so a wedged consumer cannot leak the task forever;
/// on timeout the frame is dropped, which for a consumer that is 1024 events
/// behind and unresponsive is indistinguishable from the receiver-drop case.
/// Without a captured runtime handle this degrades to the synchronous bounded
/// send.
pub(in crate::scheduler) fn spawn_terminal_send(
    tx: &tokio::sync::mpsc::Sender<crate::api::inference_types::StreamEvent>,
    event: crate::api::inference_types::StreamEvent,
    what: &'static str,
) {
    if let Some(h) = RUNTIME_HANDLE.get() {
        let tx = tx.clone();
        let deadline = stream_send_deadline();
        h.spawn(async move {
            match tokio::time::timeout(deadline, tx.send(event)).await {
                Ok(Ok(())) => {}
                Ok(Err(_)) => tracing::debug!("terminal send: receiver dropped ({what})"),
                Err(_) => tracing::warn!(
                    "terminal send: consumer stalled past {deadline:?}, frame dropped ({what})"
                ),
            }
        });
    } else if !bounded_stream_send(tx, event, what) {
        tracing::debug!("terminal send failed synchronously ({what})");
    }
}
