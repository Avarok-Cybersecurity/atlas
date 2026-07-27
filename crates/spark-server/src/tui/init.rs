// SPDX-License-Identifier: AGPL-3.0-only

//! Subscriber installation + the switchable log writer.
//!
//! Plain mode (`tui::plain_mode()` true) never reaches this file — `main.rs`
//! keeps the pre-TUI 5-line `fmt().init()` byte-for-byte, which is the format
//! every benchmark driver greps.
//!
//! TUI mode installs a `Registry` with three layers:
//!  1. fmt layer (`ansi=false`) → [`SwitchableWriter`]: every formatted line
//!     goes to the tee FILE always (the alternate screen eats stdout, so the
//!     file is the post-mortem record) and to raw stdout whenever the TUI is
//!     not active (before attach / after detach / after a panic restore).
//!  2. [`super::log_ring::LogRingLayer`] — structured lines for the log pane.
//!  3. [`super::capture_layer::ProgressCaptureLayer`] — typed progress events,
//!     always-on for its dedicated target.
//!
//! Layers 1 and 2 share `RUST_LOG` semantics via two `EnvFilter` instances
//! built from the same spec, so the pane shows exactly what stdout would.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;

use parking_lot::Mutex;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::Layer as _;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use super::capture_layer::{ProgressCaptureLayer, ProgressEvent};
use super::log_ring::LogRingLayer;

/// True while the TUI owns the screen. The writer routes stdout-vs-silent on
/// this; the event loop flips it on attach/detach.
pub static TUI_ACTIVE: AtomicBool = AtomicBool::new(false);

static TEE: OnceLock<Mutex<BufWriter<File>>> = OnceLock::new();
static TEE_PATH: OnceLock<String> = OnceLock::new();
/// Raw fd of the tee file, for the guard's stderr redirection (-1 = none).
static TEE_FD: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(-1);

/// The tee file's raw fd, if one is open.
pub fn tee_raw_fd() -> Option<i32> {
    match TEE_FD.load(Ordering::Relaxed) {
        -1 => None,
        fd => Some(fd),
    }
}

/// Where the tee file lives: `$ATLAS_TUI_LOG_FILE` or
/// `~/.cache/atlas/logs/spark-serve-<pid>-<ts>.log`.
fn tee_path() -> PathBuf {
    if let Ok(p) = std::env::var("ATLAS_TUI_LOG_FILE") {
        return PathBuf::from(p);
    }
    let base = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    PathBuf::from(base)
        .join(".cache/atlas/logs")
        .join(format!("spark-serve-{}-{ts}.log", std::process::id()))
}

/// The tee file's path, once installed (for the panic hook + header display).
pub fn tee_file_path() -> Option<&'static str> {
    TEE_PATH.get().map(String::as_str)
}

/// Flush the tee file (shutdown path).
pub fn flush_tee() {
    if let Some(t) = TEE.get() {
        let _ = t.lock().flush();
    }
}

/// `MakeWriter` whose writers tee to the log file and, when the TUI is not
/// active, to stdout.
#[derive(Clone, Copy)]
pub struct SwitchableWriter;

pub struct SwitchableIo;

impl Write for SwitchableIo {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if let Some(t) = TEE.get() {
            let _ = t.lock().write_all(buf);
        }
        if !TUI_ACTIVE.load(Ordering::Relaxed) {
            let _ = std::io::stdout().write_all(buf);
        }
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        flush_tee();
        if !TUI_ACTIVE.load(Ordering::Relaxed) {
            std::io::stdout().flush()?;
        }
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for SwitchableWriter {
    type Writer = SwitchableIo;
    fn make_writer(&'a self) -> Self::Writer {
        SwitchableIo
    }
}

fn env_filter() -> EnvFilter {
    EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into())
}

/// Install the TTY-mode subscriber stack. Returns the progress-event receiver
/// hand-off is done by the caller wiring `progress_tx` into the layer here.
pub fn install_tty_subscriber(progress_tx: Sender<ProgressEvent>) {
    let path = tee_path();
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(f) = File::create(&path) {
        // fd juggling is a unix concept; on Windows the tee still captures the
        // tracing stream, only the stderr redirection in `terminal_guard` is
        // unavailable (see `tee_raw_fd`).
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            TEE_FD.store(f.as_raw_fd(), Ordering::Relaxed);
        }
        let _ = TEE.set(Mutex::new(BufWriter::new(f)));
        let _ = TEE_PATH.set(path.display().to_string());
    }
    // Filters: one instance per layer, same spec — EnvFilter is not Clone.
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_ansi(false)
        .with_writer(SwitchableWriter)
        .with_filter(env_filter());
    let ring_layer = LogRingLayer.with_filter(env_filter());
    // Progress layer: its own always-on target filter, independent of RUST_LOG.
    let progress_layer =
        ProgressCaptureLayer::new(progress_tx).with_filter(tracing_subscriber::filter::filter_fn(
            |meta| meta.target() == spark_runtime::progress::TARGET,
        ));
    tracing_subscriber::registry()
        .with(fmt_layer)
        .with(ring_layer)
        .with(progress_layer)
        .init();
}
