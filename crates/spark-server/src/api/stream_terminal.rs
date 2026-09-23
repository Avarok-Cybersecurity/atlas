// SPDX-License-Identifier: AGPL-3.0-only

//! A streaming response never ends in silence.
//!
//! The scheduler owns each streaming request's sender and ends the stream
//! with exactly one terminal event, `Done` or `Error`. If the sender is
//! dropped without one — a failure path that lost the sink, a panic on the
//! scheduler thread, a shutdown — the SSE body simply stopped: HTTP 200, no
//! error frame, no finish reason. A client cannot tell that from a model that
//! produced nothing, and the bench harness counted exactly that as a
//! successful 0-token completion (G25: every request after a poisoned verify
//! capture read 200 / 0 tokens / 0 errors).
//!
//! [`terminated`] wraps the receiver so a channel that closes without a
//! terminal event yields one synthetic `Error` instead. Each surface renders
//! it through the `Error` arm it already has, so the frame a client sees is
//! the same one a scheduler-reported failure produces.

use futures::{Stream, StreamExt};

use super::inference_types::StreamEvent;

/// The message the synthetic terminal event carries.
pub(crate) const ENDED_WITHOUT_RESULT: &str = "inference ended without a result: the server dropped this request's stream \
     before it finished or reported an error";

/// `events`, plus one `StreamEvent::Error` if it ends without `Done`/`Error`.
pub(crate) fn terminated<S>(events: S) -> impl Stream<Item = StreamEvent>
where
    S: Stream<Item = StreamEvent> + Unpin,
{
    // State: (events, a terminal event has passed, the synthetic one was sent).
    // Events after a terminal one are forwarded untouched: this adds a frame
    // only where the stream would otherwise have ended in silence.
    futures::stream::unfold(
        (events, false, false),
        |(mut events, seen_terminal, synthesized)| async move {
            if synthesized {
                return None;
            }
            match events.next().await {
                Some(event) => {
                    let terminal =
                        matches!(event, StreamEvent::Done { .. } | StreamEvent::Error(_));
                    Some((event, (events, seen_terminal || terminal, false)))
                }
                None if seen_terminal => None,
                None => Some((
                    StreamEvent::Error(ENDED_WITHOUT_RESULT.to_string()),
                    (events, true, true),
                )),
            }
        },
    )
}

#[cfg(test)]
#[path = "stream_terminal_tests.rs"]
mod tests;
