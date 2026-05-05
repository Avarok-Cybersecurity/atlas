// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn entry(id: &str, kind: StoredKind) -> StoredEntry {
    StoredEntry {
        id: id.to_string(),
        kind,
        model: "test-model".into(),
        created_at: 0,
        messages: vec![],
        body: serde_json::json!({"id": id}),
        last_access: Instant::now(),
    }
}

#[test]
fn insert_and_get_roundtrip() {
    let store = ResponseStore::with_config(16, Duration::from_secs(60));
    store.insert(entry("resp_1", StoredKind::Response));
    let got = store.get("resp_1", StoredKind::Response).expect("hit");
    assert_eq!(got.body["id"], "resp_1");
}

#[test]
fn get_rejects_wrong_kind() {
    let store = ResponseStore::with_config(16, Duration::from_secs(60));
    store.insert(entry("resp_1", StoredKind::Response));
    assert!(store.get("resp_1", StoredKind::ChatCompletion).is_none());
}

#[test]
fn capacity_evicts_lru() {
    let store = ResponseStore::with_config(2, Duration::from_secs(60));
    store.insert(entry("a", StoredKind::Response));
    store.insert(entry("b", StoredKind::Response));
    let _ = store.get("a", StoredKind::Response);
    store.insert(entry("c", StoredKind::Response));
    assert!(store.get("b", StoredKind::Response).is_none());
    assert!(store.get("a", StoredKind::Response).is_some());
    assert!(store.get("c", StoredKind::Response).is_some());
}

#[test]
fn ttl_expires_entry() {
    let store = ResponseStore::with_config(16, Duration::from_millis(10));
    store.insert(entry("resp_1", StoredKind::Response));
    std::thread::sleep(Duration::from_millis(30));
    assert!(store.get("resp_1", StoredKind::Response).is_none());
    assert_eq!(store.len(), 0);
}

#[test]
fn filesystem_roundtrip_survives_restart() {
    let tmp = tempfile::tempdir().expect("tmpdir");
    // Write an entry with one store instance.
    {
        let store = ResponseStore::with_filesystem(16, Duration::from_secs(60), tmp.path())
            .expect("fs store");
        store.insert(entry("resp_persist", StoredKind::Response));
        assert!(store.get("resp_persist", StoredKind::Response).is_some());
    }
    // New store instance — replays from disk.
    let store =
        ResponseStore::with_filesystem(16, Duration::from_secs(60), tmp.path()).expect("fs store");
    let got = store
        .get("resp_persist", StoredKind::Response)
        .expect("replayed entry");
    assert_eq!(got.body["id"], "resp_persist");
}

#[test]
fn filesystem_forgets_on_eviction() {
    let tmp = tempfile::tempdir().expect("tmpdir");
    let store = ResponseStore::with_filesystem(1, Duration::from_secs(60), tmp.path()).expect("fs");
    store.insert(entry("a", StoredKind::Response));
    store.insert(entry("b", StoredKind::Response));
    // `a` evicted → file gone.
    let files: Vec<_> = std::fs::read_dir(tmp.path())
        .unwrap()
        .flatten()
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|e| e.to_str())
                .map(|s| s == "json")
                .unwrap_or(false)
        })
        .collect();
    assert_eq!(files.len(), 1);
}

#[test]
fn filesystem_skips_expired_on_replay() {
    let tmp = tempfile::tempdir().expect("tmpdir");
    {
        let store =
            ResponseStore::with_filesystem(16, Duration::from_secs(60), tmp.path()).unwrap();
        store.insert(entry("old", StoredKind::Response));
    }
    // Rewrite the entry with an old persisted_at so TTL replay skips it.
    let file = tmp.path().join("old.json");
    let mut disk: DiskEntry = serde_json::from_slice(&std::fs::read(&file).unwrap()).unwrap();
    disk.persisted_at_unix = 0; // 1970 — very expired.
    std::fs::write(&file, serde_json::to_vec(&disk).unwrap()).unwrap();

    let store = ResponseStore::with_filesystem(16, Duration::from_secs(1), tmp.path()).unwrap();
    assert!(store.get("old", StoredKind::Response).is_none());
    // Expired file was cleaned up.
    assert!(!file.exists());
}
