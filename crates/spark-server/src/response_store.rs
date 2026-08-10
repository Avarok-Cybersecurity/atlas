// SPDX-License-Identifier: AGPL-3.0-only

//! LRU + TTL store for stateful Responses API resume
//! (`previous_response_id`) and opt-in Chat-Completions storage (`store:
//! true`). Pluggable persistence backend — defaults to in-memory only;
//! set `ATLAS_STORE_DIR` to persist entries to disk.
//!
//! Design notes
//! ------------
//! - **Kind-typed.** Every entry declares `Response` or `ChatCompletion`
//!   so a cross-kind lookup (chatcmpl-id passed to previous_response_id)
//!   returns None instead of leaking.
//! - **Two eviction pressures.** TTL (`ATLAS_STORE_TTL_SECONDS`, default
//!   24 h) reclaims idle entries lazily on get/insert; capacity
//!   (`ATLAS_STORE_MAX_ENTRIES`, default 10 000) reclaims the coldest
//!   LRU entry when the map would exceed its bound.
//! - **Persistence (optional).** When `ATLAS_STORE_DIR=/path/to/dir` is
//!   set, each `insert` writes a `<id>.json` file and each eviction
//!   (capacity or TTL) deletes it. On startup, the directory is
//!   replayed into memory, skipping files whose `persisted_at_unix`
//!   plus TTL is in the past. Writes are fire-and-forget; failures are
//!   logged but never propagate (we'd rather serve a correct in-memory
//!   response than fail the request because the FS is full).
//! - **Single mutex.** Contention is low (one lock-roundtrip per
//!   `/v1/*` request that touches the store); parking_lot's Mutex is
//!   cheap enough that a sharded map is premature.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::openai::IncomingMessage;

/// What kind of object is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StoredKind {
    Response,
    ChatCompletion,
}

impl StoredKind {
    pub fn id_prefix(self) -> &'static str {
        match self {
            StoredKind::Response => "resp_",
            StoredKind::ChatCompletion => "chatcmpl-",
        }
    }
}

/// One stored entry. The in-memory copy carries `last_access` (an
/// `Instant` — non-persistable). The on-disk copy carries
/// `persisted_at_unix` instead so TTL survives a restart.
pub struct StoredEntry {
    pub id: String,
    pub kind: StoredKind,
    pub model: String,
    pub created_at: u64,
    pub messages: Vec<IncomingMessage>,
    pub body: serde_json::Value,
    pub last_access: Instant,
}

/// Disk layout. Mirrors `StoredEntry` minus the `Instant`, which is
/// replaced by a wall-clock timestamp so TTL decisions are correct
/// across restarts. `messages` is stored as the parsed JSON shape that
/// would lower back into `IncomingMessage` on replay.
#[derive(Serialize, Deserialize)]
struct DiskEntry {
    id: String,
    kind: StoredKind,
    model: String,
    created_at: u64,
    messages: serde_json::Value,
    body: serde_json::Value,
    persisted_at_unix: u64,
}

/// Persistence backend. Implementations run inside the store's critical
/// section, so they should be **fast** and non-blocking; the filesystem
/// backend uses synchronous `std::fs` calls because the cost is
/// dominated by the actual syscall which tokio can't help with either.
pub trait StoreBackend: Send + Sync {
    fn persist(&self, entry: &StoredEntry);
    fn forget(&self, id: &str);
    /// Called once at startup; returns all entries that were on disk
    /// and whose TTL has not elapsed.
    fn replay(&self, ttl: Duration) -> Vec<StoredEntry>;
}

/// No-op backend used when persistence is disabled.
struct NoopBackend;
impl StoreBackend for NoopBackend {
    fn persist(&self, _entry: &StoredEntry) {}
    fn forget(&self, _id: &str) {}
    fn replay(&self, _ttl: Duration) -> Vec<StoredEntry> {
        Vec::new()
    }
}

/// Filesystem-per-entry backend. Each entry is one JSON file at
/// `{dir}/{urlencoded_id}.json`. File names are URL-encoded in case an
/// id ever contains a path separator (shouldn't happen — our ids are
/// `resp_<uuid>` / `chatcmpl-<uuid>` — but defense in depth).
/// Queue depth for pending disk operations. Bounded so a stalled disk cannot
/// grow it without limit; see [`FilesystemBackend::persist`] for what happens
/// when it fills (it does NOT drop entries — this is functional state, not
/// diagnostics).
const DISK_QUEUE_DEPTH: usize = 1024;

enum DiskOp {
    Persist(Box<DiskEntry>),
    Forget(String),
}

/// Persists entries to disk from a dedicated thread.
///
/// `persist` and `forget` are reached from ASYNC request handlers
/// (`finalize_responses_stream`, `translate_chat_response_to_responses`) through
/// the sync `ResponseStore::insert`. Writing inline blocked a tokio worker on a
/// `write + fsync-ish rename` for every completed Responses request. The work now
/// goes to this thread; ordering is preserved because one queue carries both
/// writes and deletes, so a persist can never be reordered past the forget that
/// supersedes it.
pub struct FilesystemBackend {
    dir: PathBuf,
    /// `None` only while `Drop` is closing the queue.
    tx: Mutex<Option<std::sync::mpsc::SyncSender<DiskOp>>>,
    worker: Mutex<Option<std::thread::JoinHandle<()>>>,
    /// Latches the "disk queue full" warning so it is logged once per backend
    /// rather than once per process — the queue it describes is this one's.
    queue_full_warned: std::sync::atomic::AtomicBool,
}

fn write_to_disk(dir: &std::path::Path, disk: &DiskEntry) {
    let path = dir.join(format!("{}.json", sanitize_id(&disk.id)));
    let tmp = path.with_extension("json.tmp");
    let bytes = match serde_json::to_vec(disk) {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!("response_store: serialize failed for {}: {e}", disk.id);
            return;
        }
    };
    // Write-then-rename for crash-atomicity.
    if let Err(e) = std::fs::write(&tmp, &bytes) {
        tracing::warn!("response_store: write {}: {e}", tmp.display());
        return;
    }
    if let Err(e) = std::fs::rename(&tmp, &path) {
        tracing::warn!("response_store: rename {}: {e}", path.display());
    }
}

fn remove_from_disk(dir: &std::path::Path, id: &str) {
    let path = dir.join(format!("{}.json", sanitize_id(id)));
    if let Err(e) = std::fs::remove_file(&path)
        && e.kind() != std::io::ErrorKind::NotFound
    {
        tracing::warn!("response_store: remove {}: {e}", path.display());
    }
}

impl FilesystemBackend {
    pub fn new(dir: impl Into<PathBuf>) -> std::io::Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        let (tx, rx) = std::sync::mpsc::sync_channel::<DiskOp>(DISK_QUEUE_DEPTH);
        let worker_dir = dir.clone();
        let worker = std::thread::Builder::new()
            .name("atlas-respstore".into())
            .spawn(move || {
                while let Ok(op) = rx.recv() {
                    match op {
                        DiskOp::Persist(d) => write_to_disk(&worker_dir, &d),
                        DiskOp::Forget(id) => remove_from_disk(&worker_dir, &id),
                    }
                }
            })?;
        Ok(Self {
            dir,
            tx: Mutex::new(Some(tx)),
            worker: Mutex::new(Some(worker)),
            queue_full_warned: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Enqueue a disk op, or run it inline if the queue is full.
    ///
    /// Unlike the request dumper, dropping is NOT acceptable here: a lost write
    /// means a resumable response silently disappears after a restart. Under
    /// sustained overload we take the blocking write rather than the data loss,
    /// and say so once.
    fn submit(&self, op: DiskOp) {
        let guard = self.tx.lock();
        let Some(tx) = guard.as_ref() else {
            return;
        };
        if let Err(std::sync::mpsc::TrySendError::Full(op)) = tx.try_send(op) {
            // Latched on the BACKEND, not in a static: the warning is about
            // this backend's disk queue, and `&self` is right here.
            if !self
                .queue_full_warned
                .swap(true, std::sync::atomic::Ordering::Relaxed)
            {
                tracing::warn!(
                    "response_store: disk queue full — falling back to inline writes                      (persistence is keeping up poorly; requests will see the write latency)"
                );
            }
            match op {
                DiskOp::Persist(d) => write_to_disk(&self.dir, &d),
                DiskOp::Forget(id) => remove_from_disk(&self.dir, &id),
            }
        }
    }

    fn path_for(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{}.json", sanitize_id(id)))
    }
}

/// Strip any path separator or control chars. Our ids only contain
/// `[a-zA-Z0-9_-]`, so this is a belt-and-braces check.
/// Maximum stem length. Bounds the filename regardless of what a client sends,
/// which also keeps us inside every filesystem's per-component limit.
const MAX_STEM: usize = 96;

/// Map a client-supplied response id to a safe filename stem.
///
/// Response ids reach this from request bodies, so the result must not be able
/// to leave the store directory (CWE-22). The allowlist is `[A-Za-z0-9_-]` and
/// everything else — including `.`, `/`, `\\`, NUL and every Unicode separator —
/// becomes `_`. Excluding `.` is deliberate: with no dots, `..` cannot be
/// expressed at all, so traversal is impossible by construction rather than by
/// argument about what the join does. The length cap bounds the rest.
///
/// Dropping `.` changes the on-disk name for ids that contain one. Nothing reads
/// ids back out of filenames — `replay` takes them from the JSON body — so the
/// only effect is that a pre-existing file for such an id is no longer matched
/// by `forget`; TTL replay removes it on the next start.
fn sanitize_id(id: &str) -> String {
    id.chars()
        .take(MAX_STEM)
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Convert `Vec<IncomingMessage>` to a JSON array that can be
/// round-tripped back via the same `IncomingMessage::deserialize` path
/// that serves inbound requests. Avoids deriving `Serialize` on the
/// request-side types (which would couple our on-disk shape to the
/// parse-time representation).
fn messages_to_disk_json(msgs: &[IncomingMessage]) -> serde_json::Value {
    serde_json::Value::Array(
        msgs.iter()
            .map(|m| {
                let mut obj = serde_json::Map::new();
                obj.insert("role".into(), serde_json::Value::String(m.role.clone()));
                if m.content.images.is_empty() {
                    obj.insert(
                        "content".into(),
                        serde_json::Value::String(m.content.text.clone()),
                    );
                } else {
                    // Multi-part content: text + images as data-uri
                    // image_url parts. Mirrors the OpenAI chat content
                    // array shape so replay deserializes cleanly.
                    let mut parts: Vec<serde_json::Value> = Vec::new();
                    if !m.content.text.is_empty() {
                        parts.push(serde_json::json!({
                            "type": "text",
                            "text": m.content.text,
                        }));
                    }
                    for img in &m.content.images {
                        parts.push(serde_json::json!({
                            "type": "image_url",
                            "image_url": { "url": img },
                        }));
                    }
                    obj.insert("content".into(), serde_json::Value::Array(parts));
                }
                if let Some(tc) = &m.tool_calls
                    && let Ok(v) = serde_json::to_value(tc)
                {
                    obj.insert("tool_calls".into(), v);
                }
                if let Some(id) = &m.tool_call_id {
                    obj.insert("tool_call_id".into(), serde_json::Value::String(id.clone()));
                }
                if let Some(n) = &m.name {
                    obj.insert("name".into(), serde_json::Value::String(n.clone()));
                }
                serde_json::Value::Object(obj)
            })
            .collect(),
    )
}

impl Drop for FilesystemBackend {
    /// Close the queue and wait for the writer, so a dropped store has finished
    /// its disk work — a fresh store replaying the same directory sees every
    /// entry the old one accepted.
    fn drop(&mut self) {
        self.tx.lock().take();
        if let Some(h) = self.worker.lock().take() {
            let _ = h.join();
        }
    }
}

impl StoreBackend for FilesystemBackend {
    fn persist(&self, entry: &StoredEntry) {
        let disk = DiskEntry {
            id: entry.id.clone(),
            kind: entry.kind,
            model: entry.model.clone(),
            created_at: entry.created_at,
            messages: messages_to_disk_json(&entry.messages),
            body: entry.body.clone(),
            persisted_at_unix: now_unix(),
        };
        self.submit(DiskOp::Persist(Box::new(disk)));
    }

    fn forget(&self, id: &str) {
        self.submit(DiskOp::Forget(id.to_string()));
    }

    fn replay(&self, ttl: Duration) -> Vec<StoredEntry> {
        let mut out = Vec::new();
        let rd = match std::fs::read_dir(&self.dir) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("response_store: read_dir {}: {e}", self.dir.display());
                return out;
            }
        };
        let now = now_unix();
        let ttl_s = ttl.as_secs();
        for ent in rd.flatten() {
            let p = ent.path();
            if p.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let bytes = match std::fs::read(&p) {
                Ok(b) => b,
                Err(e) => {
                    tracing::warn!("response_store: read {}: {e}", p.display());
                    continue;
                }
            };
            let disk: DiskEntry = match serde_json::from_slice(&bytes) {
                Ok(d) => d,
                Err(e) => {
                    tracing::warn!("response_store: parse {}: {e}", p.display());
                    // Leave the file — operator can inspect.
                    continue;
                }
            };
            if now.saturating_sub(disk.persisted_at_unix) > ttl_s {
                // Expired on disk; remove and skip.
                let _ = std::fs::remove_file(&p);
                continue;
            }
            let messages: Vec<IncomingMessage> = match serde_json::from_value(disk.messages) {
                Ok(m) => m,
                Err(e) => {
                    tracing::warn!(
                        "response_store: messages shape drifted for {}: {e}",
                        disk.id
                    );
                    continue;
                }
            };
            out.push(StoredEntry {
                id: disk.id,
                kind: disk.kind,
                model: disk.model,
                created_at: disk.created_at,
                messages,
                body: disk.body,
                last_access: Instant::now(),
            });
        }
        out
    }
}

pub struct ResponseStore {
    inner: Mutex<Inner>,
    ttl: Duration,
    max_entries: usize,
    backend: Box<dyn StoreBackend>,
    /// True when `backend` is anything other than `NoopBackend`. Public
    /// so startup logging can mention persistence mode.
    persistent: bool,
    persist_dir: Option<PathBuf>,
}

struct Inner {
    map: HashMap<String, StoredEntry>,
    order: std::collections::VecDeque<String>,
}

pub struct GetResult {
    pub model: String,
    pub created_at: u64,
    pub messages: Vec<IncomingMessage>,
    pub body: serde_json::Value,
}

mod store_impl;

#[cfg(test)]
mod tests;
