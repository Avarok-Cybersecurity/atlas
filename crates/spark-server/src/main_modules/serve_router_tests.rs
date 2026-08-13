// SPDX-License-Identifier: AGPL-3.0-only

//! The readiness line: emitted only once it is true.
//!
//! The defect class under test is a promise printed early. The scheduler's
//! last init line lands minutes before the socket exists, and the old
//! "Listening on" was once printed BEFORE the bind — both taught a user to
//! curl a server that refused the connection. So the success test asserts the
//! line appears with the real address, and the failure test asserts a bind
//! that never happened is never announced.

use std::sync::mpsc::{Sender, channel};

use tracing::field::{Field, Visit};
use tracing_subscriber::layer::SubscriberExt as _;

use super::{bind_and_announce, ready_line};
use crate::main_modules::model_host::ModelHost;

/// Collect every event's `message` text. Local to this file rather than
/// reusing `log_ring`, because the ring is process-global — a test reading it
/// would also read every other test's log lines, and pass or fail on ordering
/// luck.
struct MessageSink(Sender<String>);

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for MessageSink {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        struct V(Option<String>);
        impl Visit for V {
            fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
                if field.name() == "message" {
                    self.0 = Some(format!("{value:?}"));
                }
            }
        }
        let mut v = V(None);
        event.record(&mut v);
        if let Some(msg) = v.0 {
            let _ = self.0.send(msg);
        }
    }
}

/// Run `f` on a CURRENT-THREAD runtime with the sink installed. Single-thread
/// on purpose: `with_default` is thread-scoped, so events from a worker thread
/// would silently miss the sink and the assertion would pass on an empty log.
fn logged<T>(f: impl FnOnce() -> T) -> (T, Vec<String>) {
    let (tx, rx) = channel();
    let sub = tracing_subscriber::registry().with(MessageSink(tx));
    let out = tracing::subscriber::with_default(sub, f);
    (out, rx.try_iter().collect())
}

fn current_thread_rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime")
}

/// A loopback port that is free at the moment of asking.
fn free_port() -> u16 {
    let sock = std::net::TcpListener::bind("127.0.0.1:0").expect("ephemeral bind");
    sock.local_addr().expect("addr").port()
}

#[test]
fn the_ready_line_is_emitted_after_a_successful_bind_with_the_real_address() {
    // `bind_and_announce`, not `build_and_serve`: the accept loop behind it
    // never returns and disarms the process-wide startup escape, which cannot
    // be re-armed — running it here made an unrelated shutdown test fail on
    // test ORDER. The emission point under test is the bind step itself.
    let port = free_port();
    let (result, lines) = logged(|| {
        let rt = current_thread_rt();
        rt.block_on(bind_and_announce(&ModelHost::empty(), "127.0.0.1", port))
    });
    assert!(result.is_ok(), "the bind succeeds: {result:?}");
    // No model is loaded in this boot, so the line must be the live-but-503
    // variant — asserting "ready … running" here would demand a lie.
    let expected = format!("Server live at 127.0.0.1:{port}");
    assert!(
        lines.iter().any(|l| l.contains(&expected)),
        "the line and its real address are in the log: {lines:#?}"
    );
    assert!(
        !lines.iter().any(|l| l.contains("live and ready")),
        "a modelless boot must not claim a model: {lines:#?}"
    );
}

#[test]
fn a_failed_bind_never_announces_readiness() {
    // Occupy the port first: the most common startup failure, per the comment
    // at the bind site.
    let taken = std::net::TcpListener::bind("127.0.0.1:0").expect("occupy");
    let port = taken.local_addr().expect("addr").port();
    let (result, lines) = logged(|| {
        let rt = current_thread_rt();
        rt.block_on(bind_and_announce(&ModelHost::empty(), "127.0.0.1", port))
    });
    assert!(result.is_err(), "the bind must fail: port is held");
    assert!(
        !lines.iter().any(|l| l.contains("Server live")),
        "no readiness claim on the failure path: {lines:#?}"
    );
}

#[test]
fn the_ready_line_names_the_model_the_way_the_dashboard_does() {
    // `live_model()` is `AppState::model_name` — the same string the Main tab
    // header and /v1/models serve — so the three never disagree about what is
    // running. `AppState` needs a loaded model, so the with-model spelling is
    // pinned here and the emission point is covered by the two tests above.
    assert_eq!(
        ready_line("127.0.0.1", 8888, Some("Qwen/Qwen3.6-35B-A3B-FP8")),
        "Server live and ready at 127.0.0.1:8888 running Qwen/Qwen3.6-35B-A3B-FP8"
    );
}

#[test]
fn a_wildcard_bind_is_rendered_as_an_address_a_user_can_paste() {
    // 0.0.0.0 accepts on every interface but is not a destination; loopback is
    // the one address guaranteed to reach the process from this machine.
    assert_eq!(
        ready_line("0.0.0.0", 8888, Some("m")),
        "Server live and ready at 127.0.0.1:8888 running m"
    );
    assert_eq!(
        ready_line("::", 8000, Some("m")),
        "Server live and ready at 127.0.0.1:8000 running m"
    );
    // An explicit bind is already pasteable and is shown as given.
    assert_eq!(
        ready_line("10.10.10.1", 8000, Some("m")),
        "Server live and ready at 10.10.10.1:8000 running m"
    );
    // An IPv6 literal needs brackets or the port fuses into the address.
    assert_eq!(
        ready_line("::1", 8000, Some("m")),
        "Server live and ready at [::1]:8000 running m"
    );
}
