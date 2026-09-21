// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn owner(sid: &str) -> LockOwner {
    LockOwner {
        session_id: sid.into(),
        hostname: "box".into(),
        user: "u".into(),
        cwd: "/w".into(),
    }
}

fn campaign(pid: Option<u32>) -> Campaign {
    Campaign {
        pr: Some(1027),
        branch: "pr/x".into(),
        anchor_sha: "abc".into(),
        avarok_home: "/home".into(),
        driver_pid: pid,
        driver_cmdline: None,
        started_at: None,
        current_gate: None,
        gates_done: vec![],
        heartbeat_at: None,
        guard_last_rc: None,
    }
}

fn tmp() -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!(
        "certify-lock-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

#[test]
fn rfc3339_round_trips() {
    for t in [0u64, 1_789_313_858, 4_102_444_799] {
        assert_eq!(parse_rfc3339(&rfc3339(t)), Some(t), "{t}");
    }
    assert_eq!(rfc3339(1_789_313_858), "2026-09-13T15:37:38Z");
}

#[test]
fn claim_writes_the_v1_schema_and_drop_removes_it() {
    let root = tmp();
    let now = 1_789_313_858;
    {
        let g = LockGuard::claim(&root, owner("s1"), campaign(Some(1)), now, &|_| true).unwrap();
        let on_disk: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join(LOCK_NAME)).unwrap()).unwrap();
        assert_eq!(on_disk["schema"], SCHEMA);
        assert_eq!(on_disk["status"], "running_certification");
        assert_eq!(on_disk["campaign"]["pr"], 1027);
        assert_eq!(on_disk["owner"]["session_id"], "s1");
        assert_eq!(g.file().campaign.anchor_sha, "abc");
    }
    assert!(!root.join(LOCK_NAME).exists(), "drop removes the lock");
}

/// NEGATIVE CONTROL: a live lock is refused, and the refusal names the owner.
#[test]
fn a_live_lock_is_refused_by_name() {
    let root = tmp();
    let now = 1_789_313_858;
    let _held = LockGuard::claim(&root, owner("first"), campaign(Some(7)), now, &|_| true).unwrap();
    let err = LockGuard::claim(&root, owner("second"), campaign(Some(8)), now, &|_| true)
        .unwrap_err()
        .to_string();
    assert!(err.contains("session first"), "{err}");
    assert!(err.contains("driver is still running"), "{err}");
}

/// A recent heartbeat keeps a lock live even when its pid is gone.
#[test]
fn a_dead_driver_with_a_fresh_heartbeat_is_still_live() {
    let root = tmp();
    let now = 1_789_313_858;
    {
        let mut g =
            LockGuard::claim(&root, owner("first"), campaign(Some(7)), now, &|_| true).unwrap();
        g.beat("decode-floor", 0, now).unwrap();
        g.release_as("running_certification", now).unwrap();
    }
    let err = LockGuard::claim(&root, owner("second"), campaign(None), now + 60, &|_| false)
        .unwrap_err()
        .to_string();
    assert!(err.contains("heartbeat 60 s ago"), "{err}");
}

/// A finished campaign's lock is reclaimable the moment its driver is gone,
/// however fresh its last heartbeat; a LIVE driver still holds it.
#[test]
fn a_finished_campaign_with_a_dead_driver_is_reclaimed_at_once() {
    for status in TERMINAL_STATUSES {
        let root = tmp();
        let now = 1_789_313_858;
        {
            let mut g =
                LockGuard::claim(&root, owner("first"), campaign(Some(7)), now, &|_| true).unwrap();
            g.beat("kat-equality-gate", 0, now).unwrap();
            g.release_as(status, now).unwrap();
        }
        // NEGATIVE CONTROL: the driver pid still answers — refused, whatever
        // the file says.
        let err = LockGuard::claim(&root, owner("second"), campaign(None), now + 5, &|_| true)
            .unwrap_err()
            .to_string();
        assert!(err.contains("still running"), "{status}: {err}");
        let g = LockGuard::claim(&root, owner("second"), campaign(Some(9)), now + 5, &|_| {
            false
        })
        .unwrap_or_else(|e| panic!("{status}: {e}"));
        assert_eq!(g.file().owner.session_id, "second");
    }
}

#[test]
fn a_dead_driver_with_an_old_heartbeat_is_reclaimed_and_archived() {
    let root = tmp();
    let now = 1_789_313_858;
    {
        let mut g =
            LockGuard::claim(&root, owner("first"), campaign(Some(7)), now, &|_| true).unwrap();
        g.beat("decode-floor", 0, now).unwrap();
        g.release_as("running_certification", now).unwrap();
    }
    let later = now + STALE_AFTER_SECS + 1;
    let g = LockGuard::claim(&root, owner("second"), campaign(Some(9)), later, &|_| false).unwrap();
    assert_eq!(g.file().owner.session_id, "second");
    let archived = std::fs::read_dir(&root)
        .unwrap()
        .filter_map(|e| e.ok())
        .any(|e| e.file_name().to_string_lossy().contains(".stale."));
    assert!(archived, "the stale lock is kept beside the new one");
}

#[test]
fn beat_and_done_update_the_file() {
    let root = tmp();
    let now = 1_789_313_858;
    let mut g = LockGuard::claim(&root, owner("s"), campaign(Some(1)), now, &|_| true).unwrap();
    g.beat("ttft-cold-gate", 0, now + 5).unwrap();
    g.done("ttft-cold-gate").unwrap();
    let on_disk: LockFile =
        serde_json::from_slice(&std::fs::read(root.join(LOCK_NAME)).unwrap()).unwrap();
    assert_eq!(
        on_disk.campaign.current_gate.as_deref(),
        Some("ttft-cold-gate")
    );
    assert_eq!(on_disk.campaign.guard_last_rc, Some(0));
    assert_eq!(
        on_disk.campaign.heartbeat_at.as_deref(),
        Some(rfc3339(now + 5).as_str())
    );
    assert_eq!(on_disk.campaign.gates_done, vec!["ttft-cold-gate"]);
}

/// NEGATIVE CONTROL: a file that is not a lockfile is never silently replaced.
#[test]
fn an_unparseable_lock_is_refused() {
    let root = tmp();
    std::fs::write(root.join(LOCK_NAME), "not json").unwrap();
    let err = LockGuard::claim(&root, owner("s"), campaign(None), 1, &|_| false)
        .unwrap_err()
        .to_string();
    assert!(err.contains("not a v1 lockfile"), "{err}");
}
