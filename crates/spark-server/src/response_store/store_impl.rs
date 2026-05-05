// SPDX-License-Identifier: AGPL-3.0-only

//! `impl ResponseStore` — Store API operations.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use super::{
    FilesystemBackend, GetResult, Inner, NoopBackend, ResponseStore, StoreBackend, StoredEntry,
    StoredKind,
};

impl ResponseStore {
    /// Build from env.
    /// - `ATLAS_STORE_MAX_ENTRIES` (default 10 000)
    /// - `ATLAS_STORE_TTL_SECONDS` (default 86 400)
    /// - `ATLAS_STORE_DIR` — when set, enable filesystem persistence
    ///   and replay any non-expired entries on startup.
    pub fn from_env() -> Arc<Self> {
        let max_entries = std::env::var("ATLAS_STORE_MAX_ENTRIES")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n: &usize| n > 0)
            .unwrap_or(10_000);
        let ttl_secs = std::env::var("ATLAS_STORE_TTL_SECONDS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(86_400_u64);
        let ttl = Duration::from_secs(ttl_secs);

        let (backend, persistent, persist_dir): (Box<dyn StoreBackend>, bool, Option<PathBuf>) =
            match std::env::var("ATLAS_STORE_DIR")
                .ok()
                .filter(|s| !s.is_empty())
            {
                Some(dir) => {
                    let p = PathBuf::from(&dir);
                    match FilesystemBackend::new(&p) {
                        Ok(fb) => (Box::new(fb), true, Some(p)),
                        Err(e) => {
                            tracing::warn!(
                                "response_store: falling back to in-memory (ATLAS_STORE_DIR={dir} init failed: {e})"
                            );
                            (Box::new(NoopBackend), false, None)
                        }
                    }
                }
                None => (Box::new(NoopBackend), false, None),
            };

        // Replay before building the Arc — we need to populate the map.
        let replayed = backend.replay(ttl);
        let mut map = HashMap::with_capacity(max_entries.min(1024).max(replayed.len()));
        let mut order = std::collections::VecDeque::with_capacity(map.capacity());
        for e in replayed {
            order.push_back(e.id.clone());
            map.insert(e.id.clone(), e);
        }

        Arc::new(Self {
            inner: Mutex::new(Inner { map, order }),
            ttl,
            max_entries,
            backend,
            persistent,
            persist_dir,
        })
    }

    #[cfg(test)]
    pub fn with_config(max_entries: usize, ttl: Duration) -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(Inner {
                map: HashMap::new(),
                order: std::collections::VecDeque::new(),
            }),
            ttl,
            max_entries,
            backend: Box::new(NoopBackend),
            persistent: false,
            persist_dir: None,
        })
    }

    #[cfg(test)]
    pub fn with_filesystem(
        max_entries: usize,
        ttl: Duration,
        dir: &Path,
    ) -> std::io::Result<Arc<Self>> {
        let fb = FilesystemBackend::new(dir)?;
        let replayed = fb.replay(ttl);
        let mut map = HashMap::new();
        let mut order = std::collections::VecDeque::new();
        for e in replayed {
            order.push_back(e.id.clone());
            map.insert(e.id.clone(), e);
        }
        Ok(Arc::new(Self {
            inner: Mutex::new(Inner { map, order }),
            ttl,
            max_entries,
            backend: Box::new(fb),
            persistent: true,
            persist_dir: Some(dir.to_path_buf()),
        }))
    }

    /// Insert or replace. Bumps to the back of the LRU and persists to
    /// the backend (no-op when persistence is disabled). If a capacity
    /// eviction triggers, the evicted entry is also removed from the
    /// backend.
    pub fn insert(&self, entry: StoredEntry) {
        self.backend.persist(&entry);
        let mut inner = self.inner.lock();
        if inner.map.contains_key(&entry.id) {
            inner.order.retain(|k| k != &entry.id);
        }
        let id = entry.id.clone();
        inner.map.insert(id.clone(), entry);
        inner.order.push_back(id);
        while inner.map.len() > self.max_entries {
            if let Some(oldest) = inner.order.pop_front() {
                inner.map.remove(&oldest);
                self.backend.forget(&oldest);
            } else {
                break;
            }
        }
    }

    /// Fetch a clone of the stored body + metadata. TTL-expired entries
    /// are evicted (and removed from the backend) and treated as
    /// missing. Kind mismatch returns None without eviction.
    pub fn get(&self, id: &str, kind: StoredKind) -> Option<GetResult> {
        let mut inner = self.inner.lock();
        let expired = match inner.map.get(id) {
            Some(e) if e.kind != kind => return None,
            Some(e) => e.last_access.elapsed() > self.ttl,
            None => return None,
        };
        if expired {
            inner.map.remove(id);
            inner.order.retain(|k| k != id);
            self.backend.forget(id);
            return None;
        }
        inner.order.retain(|k| k != id);
        inner.order.push_back(id.to_string());
        let entry = inner.map.get_mut(id).expect("entry present");
        entry.last_access = Instant::now();
        Some(GetResult {
            model: entry.model.clone(),
            created_at: entry.created_at,
            messages: entry.messages.clone(),
            body: entry.body.clone(),
        })
    }

    /// Remove an entry. Returns `true` when the id existed and matched
    /// `kind`, `false` otherwise. Also removes the entry from the
    /// persistent backend when one is attached.
    pub fn delete(&self, id: &str, kind: StoredKind) -> bool {
        let mut inner = self.inner.lock();
        let matches = inner.map.get(id).map(|e| e.kind == kind).unwrap_or(false);
        if !matches {
            return false;
        }
        inner.map.remove(id);
        inner.order.retain(|k| k != id);
        self.backend.forget(id);
        true
    }

    pub fn len(&self) -> usize {
        self.inner.lock().map.len()
    }

    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    pub fn ttl(&self) -> Duration {
        self.ttl
    }

    pub fn is_persistent(&self) -> bool {
        self.persistent
    }

    pub fn persist_dir(&self) -> Option<&Path> {
        self.persist_dir.as_deref()
    }
}
