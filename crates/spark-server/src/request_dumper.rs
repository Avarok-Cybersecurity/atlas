// SPDX-License-Identifier: AGPL-3.0-only

//! Request / response dumper for the `--dump` CLI flag.
//!
//! When `--dump` is enabled, the server appends one JSONL entry per
//! incoming request and one per outgoing response to a file. The
//! primary use case is extracting the exact system prompt and tool
//! schema a client (opencode, Claude Code, Codex, Anthropic SDK) is
//! sending, so failure cases can be replayed as fixtures without
//! guesswork.
//!
//! Entries are correlated with a monotonic `seq` counter: a request
//! and its response share the same `seq`. This lets consumers group
//! pairs with a trivial `sort -t'"' -k14 -n` or a jq pipeline.
//!
//! The writer is fail-open: if the file can't be written, the error
//! is logged once and serving continues. Dump failures MUST NOT kill
//! live requests.

use parking_lot::Mutex;
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Shared handle installed in `AppState::dump_writer` when `--dump` is
/// set. Cloning is cheap (Arc); all clones write to the same file
/// under a single mutex.
#[derive(Clone)]
pub struct DumpHandle {
    inner: Arc<DumpInner>,
}

struct DumpInner {
    writer: Mutex<std::io::BufWriter<std::fs::File>>,
    path: std::path::PathBuf,
    seq: AtomicU64,
    /// Set once the first I/O error has been logged. Prevents the
    /// server log from flooding if the dump target disappears.
    io_error_logged: std::sync::atomic::AtomicBool,
}

impl DumpHandle {
    /// Open `path` in append mode, creating it if missing. Errors are
    /// surfaced to the caller (startup) so the user knows immediately
    /// if the path is bad.
    pub fn open(path: std::path::PathBuf) -> std::io::Result<Self> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        Ok(Self {
            inner: Arc::new(DumpInner {
                writer: Mutex::new(std::io::BufWriter::new(file)),
                path,
                seq: AtomicU64::new(0),
                io_error_logged: std::sync::atomic::AtomicBool::new(false),
            }),
        })
    }

    /// Absolute path of the dump file (for startup logging).
    pub fn path(&self) -> &std::path::Path {
        &self.inner.path
    }

    /// Reserve a sequence number. Caller later references the same
    /// `seq` when emitting the matching response entry.
    pub fn next_seq(&self) -> u64 {
        self.inner.seq.fetch_add(1, Ordering::Relaxed)
    }

    /// Write a `{"kind":"request",...}` entry for an incoming request
    /// body. `body` is any serde-serialisable value — typically the
    /// `ChatCompletionRequest` / `ResponsesRequest` / Anthropic
    /// `MessagesRequest` struct already deserialised by the handler.
    pub fn dump_request<T: serde::Serialize>(&self, endpoint: &str, seq: u64, body: &T) {
        self.write_entry("request", endpoint, seq, body, None);
    }

    /// Write a `{"kind":"response",...}` entry. `is_stream` is true
    /// when the `body` is the aggregated SSE chunk list rather than a
    /// final non-streaming response object.
    pub fn dump_response<T: serde::Serialize>(
        &self,
        endpoint: &str,
        seq: u64,
        body: &T,
        is_stream: bool,
    ) {
        self.write_entry("response", endpoint, seq, body, Some(is_stream));
    }

    fn write_entry<T: serde::Serialize>(
        &self,
        kind: &str,
        endpoint: &str,
        seq: u64,
        body: &T,
        is_stream: Option<bool>,
    ) {
        // Build the JSON object *outside* the lock to keep the
        // critical section tight.
        let mut obj = serde_json::Map::with_capacity(6);
        obj.insert("ts".into(), serde_json::Value::String(iso8601_now()));
        obj.insert("kind".into(), serde_json::Value::String(kind.into()));
        obj.insert(
            "endpoint".into(),
            serde_json::Value::String(endpoint.into()),
        );
        obj.insert("seq".into(), serde_json::Value::Number(seq.into()));
        if let Some(s) = is_stream {
            obj.insert("stream".into(), serde_json::Value::Bool(s));
        }
        match serde_json::to_value(body) {
            Ok(v) => {
                obj.insert("body".into(), v);
            }
            Err(e) => {
                // Body failed to serialise — record the error string
                // in place so the entry still lands in the file.
                obj.insert(
                    "body".into(),
                    serde_json::Value::String(format!("<serialization error: {e}>")),
                );
            }
        }
        let mut line = match serde_json::to_string(&serde_json::Value::Object(obj)) {
            Ok(s) => s,
            Err(_) => return, // Unreachable — the map above always serialises.
        };
        line.push('\n');

        // parking_lot::Mutex::lock() returns the guard directly and never
        // poisons, so the prior "poisoned mutex" recovery branch is gone.
        let mut w = self.inner.writer.lock();
        if let Err(e) = w.write_all(line.as_bytes()).and_then(|_| w.flush()) {
            self.log_io_error_once(&format!("dump write failed: {e}"));
        }
    }

    fn log_io_error_once(&self, msg: &str) {
        if !self.inner.io_error_logged.swap(true, Ordering::Relaxed) {
            tracing::warn!(path = %self.inner.path.display(), "{msg}");
        }
    }
}

/// ISO-8601 UTC timestamp with milliseconds, suitable for human and
/// machine consumption.
fn iso8601_now() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs() as i64;
    let millis = now.subsec_millis();
    let days = secs.div_euclid(86400);
    let time_secs = secs.rem_euclid(86400) as u32;

    // Civil-from-days algorithm (Howard Hinnant). Unix epoch is day 0
    // = 1970-01-01.
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y };

    let hh = time_secs / 3600;
    let mm = (time_secs % 3600) / 60;
    let ss = time_secs % 60;

    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}.{millis:03}Z")
}

/// Resolve the `--dump` argument to a final file path. `"<auto>"`
/// (from clap's `default_missing_value`) maps to a timestamped file
/// under `$TMPDIR`; anything else is treated as an explicit path.
pub fn resolve_path(arg: &str) -> std::path::PathBuf {
    if arg == "<auto>" {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        std::env::temp_dir().join(format!("atlas-dump-{ts}.jsonl"))
    } else {
        std::path::PathBuf::from(arg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_auto_goes_to_tmp() {
        let p = resolve_path("<auto>");
        assert!(p.starts_with(std::env::temp_dir()));
        assert!(
            p.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("atlas-dump-")
        );
    }

    #[test]
    fn resolve_explicit_is_verbatim() {
        let p = resolve_path("/tmp/my-dump.jsonl");
        assert_eq!(p, std::path::PathBuf::from("/tmp/my-dump.jsonl"));
    }

    #[test]
    fn dump_writes_pair_with_shared_seq() {
        let tmp =
            std::env::temp_dir().join(format!("atlas-dump-test-{}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&tmp);
        let h = DumpHandle::open(tmp.clone()).expect("open");

        let seq = h.next_seq();
        #[derive(serde::Serialize)]
        struct Req {
            model: &'static str,
        }
        #[derive(serde::Serialize)]
        struct Resp {
            ok: bool,
        }
        h.dump_request("/v1/chat/completions", seq, &Req { model: "test" });
        h.dump_response("/v1/chat/completions", seq, &Resp { ok: true }, false);

        drop(h); // flush via BufWriter drop
        let contents = std::fs::read_to_string(&tmp).unwrap();
        let lines: Vec<_> = contents.lines().collect();
        assert_eq!(lines.len(), 2, "two JSONL lines expected");
        let a: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        let b: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(a["kind"], "request");
        assert_eq!(b["kind"], "response");
        assert_eq!(a["seq"], b["seq"], "request and response share seq");
        assert_eq!(a["body"]["model"], "test");
        assert_eq!(b["body"]["ok"], true);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn iso8601_has_expected_shape() {
        let s = iso8601_now();
        // YYYY-MM-DDTHH:MM:SS.sssZ  = 24 chars
        assert_eq!(s.len(), 24, "{s}");
        assert!(s.ends_with('Z'));
        assert_eq!(s.as_bytes()[10], b'T');
    }
}
