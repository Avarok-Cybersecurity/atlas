// SPDX-License-Identifier: AGPL-3.0-only

//! The switchable log writer, without installing a subscriber.
//!
//! `install_tty_subscriber` is a once-per-process side effect — it sets the
//! global tracing dispatcher and fills a `OnceLock` — so calling it from a test
//! would decide the answer for every other test in the binary. What IS testable
//! is the part that runs on every log line afterwards: the writer must never
//! fail, never panic when no tee file was ever opened, and must fall silent for
//! stdout exactly while the TUI owns the screen. A writer that returns an error
//! or panics here takes the process down from inside `tracing`.

use super::*;

/// The flag is process-global, so a test that flips it must put it back —
/// otherwise it decides what the rest of the binary's logging does.
struct RestoreActive(bool);

impl Drop for RestoreActive {
    fn drop(&mut self) {
        TUI_ACTIVE.store(self.0, Ordering::Relaxed);
    }
}

/// One test rather than several, because `TUI_ACTIVE` is process-global: two
/// tests toggling it in parallel would decide each other's answers. Content is
/// only ever written while the TUI is "active", which is exactly the state in
/// which the writer must NOT reach stdout — so this cannot pollute the harness
/// output either.
#[test]
fn the_writer_reports_every_byte_as_written_and_never_fails() {
    // `fmt` treats a short write as a failure and gives up on the line. There
    // is no partial-write path here — the tee is buffered and stdout is
    // best-effort — so the whole buffer is always accounted for.
    let restore = RestoreActive(TUI_ACTIVE.load(Ordering::Relaxed));
    TUI_ACTIVE.store(true, Ordering::Relaxed);

    let line = b"INFO atlas: a log line\n";
    assert_eq!(
        SwitchableIo.write(line).expect("the writer cannot fail"),
        line.len()
    );
    SwitchableIo.flush().expect("flush cannot fail");

    // `fmt` asks `make_writer` for a writer per event; each one must behave the
    // same, because there is no shared state to get out of step.
    use tracing_subscriber::fmt::MakeWriter as _;
    let mut a = SwitchableWriter.make_writer();
    let mut b = SwitchableWriter.make_writer();
    assert_eq!(a.write(line).expect("write"), line.len());
    assert_eq!(b.write(line).expect("write"), line.len());

    // Detached: the same accounting, with stdout back in the picture. Empty, so
    // the assertion does not depend on where the bytes went.
    TUI_ACTIVE.store(false, Ordering::Relaxed);
    assert_eq!(SwitchableIo.write(b"").expect("empty is fine"), 0);
    SwitchableIo.flush().expect("flush cannot fail");

    // Folded into this test rather than given its own, for the reason above:
    // a second test toggling the same global in parallel would decide this
    // one's answer.
    //
    // The claim's whole job is the release, and the release used to be a
    // statement at the bottom of the event loop that two `return`s jumped over
    // — leaving the flag set, and stdout logging dead, for the rest of the
    // process. Dropping is the part that must hold even when nobody calls it.
    {
        let _claim = ActiveClaim::claim();
        assert!(
            TUI_ACTIVE.load(Ordering::SeqCst),
            "a claim means the TUI owns the screen and logs stay off stdout"
        );
    }
    assert!(
        !TUI_ACTIVE.load(Ordering::SeqCst),
        "dropping the claim hands stdout back — the two bail-out paths in the \
         event loop depend on this happening without being asked"
    );
    drop(restore);
}

#[test]
fn with_no_tee_installed_there_is_nothing_to_name_and_no_fd_to_redirect() {
    // Plain mode never installs one, and the terminal guard asks for the fd on
    // a path that runs before the subscriber exists — so both accessors have to
    // answer "none" rather than assume.
    if TEE.get().is_none() {
        assert!(tee_file_path().is_none(), "nothing to name");
        assert!(tee_raw_fd().is_none(), "and no fd to redirect stderr onto");
    }
    flush_tee();
}

#[test]
fn the_tee_path_follows_its_environment_override_when_one_is_set() {
    // `$ATLAS_TUI_LOG_FILE` is how a benchmark driver puts the log where it can
    // collect it; without it the file lands under the cache dir, named by pid
    // so two runs cannot overwrite each other.
    match std::env::var("ATLAS_TUI_LOG_FILE") {
        Ok(explicit) => assert_eq!(tee_path(), std::path::PathBuf::from(explicit)),
        Err(_) => {
            let p = tee_path();
            let name = p.file_name().expect("a file name").to_string_lossy();
            assert!(name.starts_with("spark-serve-"), "{name}");
            assert!(
                name.contains(&std::process::id().to_string()),
                "named by pid so concurrent runs do not collide: {name}"
            );
            assert!(name.ends_with(".log"), "{name}");
            assert!(
                p.parent().expect("a parent").ends_with(".cache/atlas/logs"),
                "{}",
                p.display()
            );
        }
    }
}

#[test]
fn two_tee_paths_taken_in_the_same_second_are_the_same_file() {
    // The name carries a pid and a timestamp and nothing else random: a second
    // call must not invent a second log file for the same process.
    if std::env::var("ATLAS_TUI_LOG_FILE").is_err() {
        let a = tee_path();
        let b = tee_path();
        assert_eq!(
            a.parent(),
            b.parent(),
            "the directory is fixed even if the second ticks over"
        );
    }
}

#[test]
fn the_default_filter_is_info_when_the_environment_says_nothing() {
    // The pane and stdout share this spec, so a change here changes both — and
    // a silently different default would make the log pane disagree with the
    // file the user is told to read.
    if std::env::var("RUST_LOG").is_err() {
        let spec = env_filter().to_string();
        assert!(spec.contains("info"), "{spec}");
    }
}
