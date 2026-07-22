// SPDX-License-Identifier: AGPL-3.0-only

//! Conversation store — OpenAI 2026 Conversations API backend.
//!
//! OpenAI's Conversations API replaces the deprecated Assistants
//! `threads` primitive. A Conversation is a durable bag of **items** —
//! most commonly `message` items, plus `function_call` /
//! `function_call_output` items from tool turns. The Responses API
//! gained a `conversation: <conv_id>` field in 2026 that:
//!
//!   1. Prepends the conversation's items onto the new turn's `input`.
//!   2. Appends the new turn's items (user input + assistant output)
//!      back to the conversation after completion.
//!
//! Atlas stores conversations in-memory with an LRU+TTL bound, mirroring
//! the [`crate::response_store`] design but keyed on `conv_<uuid>`.
//! Persistence (filesystem) is a natural follow-up; the public API
//! already hides the backend so that's a drop-in swap.
//!
//! Items are stored as raw `serde_json::Value` so we can round-trip the
//! OpenAI wire format (input_items, function_call, function_call_output,
//! reasoning, …) without coupling to Atlas's internal message schema.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

/// One conversation — a durable, opaque bag of items with per-
/// conversation metadata. `items` preserves insertion order; each item
/// carries an opaque `id` assigned at insert time.
pub struct Conversation {
    pub id: String,
    pub created_at: u64,
    pub metadata: HashMap<String, String>,
    /// Items in wire-format JSON. Each has a top-level `"id"` field.
    pub items: Vec<serde_json::Value>,
    /// For LRU — bumped on every read + write.
    last_access: Instant,
}

pub struct ConversationStore {
    inner: Mutex<Inner>,
    ttl: Duration,
    max_entries: usize,
}

struct Inner {
    map: HashMap<String, Conversation>,
    order: VecDeque<String>,
}

/// Maximum items per conversation insert (mirrors OpenAI's spec).
pub const MAX_ITEMS_PER_INSERT: usize = 20;

impl ConversationStore {
    pub fn from_env() -> Arc<Self> {
        let max_entries = std::env::var("ATLAS_CONVERSATION_MAX_ENTRIES")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n: &usize| n > 0)
            .unwrap_or(10_000);
        let ttl_secs = std::env::var("ATLAS_CONVERSATION_TTL_SECONDS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(86_400_u64);
        Arc::new(Self {
            inner: Mutex::new(Inner {
                map: HashMap::new(),
                order: VecDeque::new(),
            }),
            ttl: Duration::from_secs(ttl_secs),
            max_entries,
        })
    }

    #[cfg(test)]
    pub fn with_config(max_entries: usize, ttl: Duration) -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(Inner {
                map: HashMap::new(),
                order: VecDeque::new(),
            }),
            ttl,
            max_entries,
        })
    }

    /// Create a new conversation with optional initial items and
    /// metadata. Returns the generated id.
    pub fn create(
        &self,
        initial_items: Vec<serde_json::Value>,
        metadata: HashMap<String, String>,
    ) -> String {
        let id = format!("conv_{}", crate::ids::uuid_v4());
        let now_unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let items = initial_items
            .into_iter()
            .enumerate()
            .map(|(i, v)| stamp_id(v, &id, i))
            .collect();
        let entry = Conversation {
            id: id.clone(),
            created_at: now_unix,
            metadata,
            items,
            last_access: Instant::now(),
        };
        let mut inner = self.inner.lock();
        inner.map.insert(id.clone(), entry);
        inner.order.push_back(id.clone());
        while inner.map.len() > self.max_entries {
            if let Some(oldest) = inner.order.pop_front() {
                inner.map.remove(&oldest);
            } else {
                break;
            }
        }
        id
    }

    /// Fetch a snapshot of the conversation. TTL-expired entries are
    /// evicted and return None.
    pub fn get(&self, id: &str) -> Option<ConversationSnapshot> {
        let mut inner = self.inner.lock();
        let expired = match inner.map.get(id) {
            Some(c) => c.last_access.elapsed() > self.ttl,
            None => return None,
        };
        if expired {
            inner.map.remove(id);
            inner.order.retain(|k| k != id);
            return None;
        }
        inner.order.retain(|k| k != id);
        inner.order.push_back(id.to_string());
        let c = inner.map.get_mut(id).expect("entry present");
        c.last_access = Instant::now();
        Some(ConversationSnapshot {
            id: c.id.clone(),
            created_at: c.created_at,
            metadata: c.metadata.clone(),
            items: c.items.clone(),
        })
    }

    /// Merge `patch` into the conversation's metadata. Returns the
    /// updated snapshot; None when the id doesn't exist / is expired.
    pub fn update_metadata(
        &self,
        id: &str,
        patch: HashMap<String, String>,
    ) -> Option<ConversationSnapshot> {
        let mut inner = self.inner.lock();
        let expired = match inner.map.get(id) {
            Some(c) => c.last_access.elapsed() > self.ttl,
            None => return None,
        };
        if expired {
            inner.map.remove(id);
            inner.order.retain(|k| k != id);
            return None;
        }
        let c = inner.map.get_mut(id).expect("entry present");
        c.last_access = Instant::now();
        for (k, v) in patch {
            c.metadata.insert(k, v);
        }
        Some(ConversationSnapshot {
            id: c.id.clone(),
            created_at: c.created_at,
            metadata: c.metadata.clone(),
            items: c.items.clone(),
        })
    }

    /// Append items. Enforces the per-call cap and returns the list of
    /// new items with assigned ids.
    pub fn add_items(
        &self,
        id: &str,
        new_items: Vec<serde_json::Value>,
    ) -> Result<Vec<serde_json::Value>, AddItemsError> {
        if new_items.len() > MAX_ITEMS_PER_INSERT {
            return Err(AddItemsError::TooMany(new_items.len()));
        }
        let mut inner = self.inner.lock();
        let expired = match inner.map.get(id) {
            Some(c) => c.last_access.elapsed() > self.ttl,
            None => return Err(AddItemsError::NotFound),
        };
        if expired {
            inner.map.remove(id);
            inner.order.retain(|k| k != id);
            return Err(AddItemsError::NotFound);
        }
        let c = inner.map.get_mut(id).expect("entry present");
        let start = c.items.len();
        let stamped: Vec<serde_json::Value> = new_items
            .into_iter()
            .enumerate()
            .map(|(i, v)| stamp_id(v, id, start + i))
            .collect();
        c.items.extend(stamped.iter().cloned());
        c.last_access = Instant::now();
        Ok(stamped)
    }

    /// Remove a single item by id. Returns true when an item with that
    /// id was present.
    pub fn remove_item(&self, conv_id: &str, item_id: &str) -> bool {
        let mut inner = self.inner.lock();
        let Some(c) = inner.map.get_mut(conv_id) else {
            return false;
        };
        let before = c.items.len();
        c.items
            .retain(|v| v.get("id").and_then(|v| v.as_str()).unwrap_or("") != item_id);
        let removed = c.items.len() < before;
        if removed {
            c.last_access = Instant::now();
        }
        removed
    }

    pub fn delete(&self, id: &str) -> bool {
        let mut inner = self.inner.lock();
        if inner.map.remove(id).is_some() {
            inner.order.retain(|k| k != id);
            true
        } else {
            false
        }
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
}

pub struct ConversationSnapshot {
    pub id: String,
    pub created_at: u64,
    pub metadata: HashMap<String, String>,
    pub items: Vec<serde_json::Value>,
}

#[derive(Debug)]
pub enum AddItemsError {
    NotFound,
    TooMany(usize),
}

/// Stamp an `id` onto an item if it doesn't already have one.
/// Preserves any client-supplied `id`; otherwise mints a deterministic
/// `item_<conv>_<idx>`.
fn stamp_id(mut v: serde_json::Value, conv_id: &str, idx: usize) -> serde_json::Value {
    if let Some(obj) = v.as_object_mut()
        && !obj.contains_key("id")
    {
        obj.insert(
            "id".to_string(),
            serde_json::Value::String(format!(
                "item_{}_{}",
                conv_id.trim_start_matches("conv_"),
                idx
            )),
        );
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn create_and_retrieve() {
        let store = ConversationStore::with_config(16, Duration::from_secs(60));
        let id = store.create(
            vec![json!({"type": "message", "role": "user", "content": "hi"})],
            HashMap::new(),
        );
        let snap = store.get(&id).expect("hit");
        assert_eq!(snap.items.len(), 1);
        assert_eq!(snap.items[0]["role"], "user");
        assert!(snap.items[0]["id"].as_str().unwrap().starts_with("item_"));
    }

    #[test]
    fn add_items_respects_cap() {
        let store = ConversationStore::with_config(16, Duration::from_secs(60));
        let id = store.create(Vec::new(), HashMap::new());
        let twenty_one: Vec<_> = (0..21)
            .map(|i| json!({"type": "message", "role": "user", "content": format!("m{i}")}))
            .collect();
        let err = store.add_items(&id, twenty_one).unwrap_err();
        assert!(matches!(err, AddItemsError::TooMany(21)));
    }

    #[test]
    fn delete_item_by_id() {
        let store = ConversationStore::with_config(16, Duration::from_secs(60));
        let id = store.create(
            vec![
                json!({"type": "message", "role": "user", "content": "a"}),
                json!({"type": "message", "role": "user", "content": "b"}),
            ],
            HashMap::new(),
        );
        let snap = store.get(&id).unwrap();
        let item_id = snap.items[0]["id"].as_str().unwrap().to_string();
        assert!(store.remove_item(&id, &item_id));
        let snap2 = store.get(&id).unwrap();
        assert_eq!(snap2.items.len(), 1);
        assert_eq!(snap2.items[0]["content"], "b");
    }

    #[test]
    fn update_metadata_merges() {
        let store = ConversationStore::with_config(16, Duration::from_secs(60));
        let mut md = HashMap::new();
        md.insert("k1".to_string(), "v1".to_string());
        let id = store.create(Vec::new(), md);
        let mut patch = HashMap::new();
        patch.insert("k2".to_string(), "v2".to_string());
        let snap = store.update_metadata(&id, patch).unwrap();
        assert_eq!(snap.metadata["k1"], "v1");
        assert_eq!(snap.metadata["k2"], "v2");
    }

    #[test]
    fn ttl_evicts() {
        let store = ConversationStore::with_config(16, Duration::from_millis(10));
        let id = store.create(Vec::new(), HashMap::new());
        std::thread::sleep(Duration::from_millis(30));
        assert!(store.get(&id).is_none());
    }

    #[test]
    fn delete_removes_entry() {
        let store = ConversationStore::with_config(16, Duration::from_secs(60));
        let id = store.create(Vec::new(), HashMap::new());
        assert!(store.delete(&id));
        assert!(store.get(&id).is_none());
        assert!(!store.delete(&id));
    }
}
