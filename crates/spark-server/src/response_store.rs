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
pub struct FilesystemBackend {
    dir: PathBuf,
}

impl FilesystemBackend {
    pub fn new(dir: impl Into<PathBuf>) -> std::io::Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    fn path_for(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{}.json", sanitize_id(id)))
    }
}

/// Strip any path separator or control chars. Our ids only contain
/// `[a-zA-Z0-9_-]`, so this is a belt-and-braces check.
fn sanitize_id(id: &str) -> String {
    id.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
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
        let path = self.path_for(&entry.id);
        let tmp = path.with_extension("json.tmp");
        let bytes = match serde_json::to_vec(&disk) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!("response_store: serialize failed for {}: {e}", entry.id);
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

    fn forget(&self, id: &str) {
        let path = self.path_for(id);
        if let Err(e) = std::fs::remove_file(&path)
            && e.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!("response_store: remove {}: {e}", path.display());
        }
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
