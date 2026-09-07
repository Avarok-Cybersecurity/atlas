// SPDX-License-Identifier: AGPL-3.0-only
//! Every doctor line must be able to go RED.
//!
//! Four checks have shipped in this repo that could not fail. A check seen only
//! green is indistinguishable from one that is wired to nothing, so each
//! finding here is asserted in BOTH states.
//!
//! These drive the pure `check_*` functions rather than the binary, and set
//! `ATLAS_HOME` around each. Env is process-global, so they run under one mutex.

use super::{check_home, check_recipes, check_writable};
use std::sync::{Mutex, MutexGuard, OnceLock};

fn env_lock() -> MutexGuard<'static, ()> {
    static L: OnceLock<Mutex<()>> = OnceLock::new();
    L.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

struct Dir(std::path::PathBuf);
impl Dir {
    fn new(tag: &str) -> Self {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default();
        let p = std::env::temp_dir().join(format!("atlas-doctor-{tag}-{n}"));
        std::fs::create_dir_all(&p).expect("scratch");
        Self(p)
    }
}
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn with_home<T>(root: &std::path::Path, f: impl FnOnce() -> T) -> T {
    let _g = env_lock();
    let prev = std::env::var_os("ATLAS_HOME");
    // SAFETY: single-threaded within the env lock held above.
    unsafe { std::env::set_var("ATLAS_HOME", root) };
    let out = f();
    match prev {
        Some(v) => unsafe { std::env::set_var("ATLAS_HOME", v) },
        None => unsafe { std::env::remove_var("ATLAS_HOME") },
    }
    out
}

#[test]
fn home_is_green_when_resolvable_and_names_its_provenance() {
    let d = Dir::new("home");
    let f = with_home(&d.0, check_home);
    assert!(!f.problem, "{}", f.detail);
    assert!(
        f.detail.contains("from ATLAS_HOME"),
        "the provenance is the point: {}",
        f.detail
    );
}

/// RED. An empty `ATLAS_HOME` is a real configuration people produce with
/// `export ATLAS_HOME=$SOMETHING_UNSET`.
#[test]
fn home_goes_red_when_atlas_home_is_empty() {
    let _g = env_lock();
    let prev = std::env::var_os("ATLAS_HOME");
    unsafe { std::env::set_var("ATLAS_HOME", "") };
    let f = check_home();
    match prev {
        Some(v) => unsafe { std::env::set_var("ATLAS_HOME", v) },
        None => unsafe { std::env::remove_var("ATLAS_HOME") },
    }
    assert!(f.problem, "an empty ATLAS_HOME must be a problem");
    assert!(f.detail.contains("empty"), "{}", f.detail);
}

#[test]
fn writable_is_green_on_a_writable_home() {
    let d = Dir::new("w-ok");
    let f = with_home(&d.0, check_writable);
    assert!(!f.problem, "{}", f.detail);
}

/// RED. The exact incident: a home that exists but this process cannot write.
/// Probed by writing, because ownership, ACLs, a read-only mount and a full
/// disk all look different in the metadata and identical to what matters.
#[test]
fn writable_goes_red_on_a_read_only_home() {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let d = Dir::new("w-ro");
        std::fs::set_permissions(&d.0, std::fs::Permissions::from_mode(0o500)).expect("chmod");
        let f = with_home(&d.0, check_writable);
        std::fs::set_permissions(&d.0, std::fs::Permissions::from_mode(0o755)).expect("restore");
        assert!(
            f.problem,
            "a read-only home must be a problem: {}",
            f.detail
        );
        assert!(
            f.detail.contains("not writable"),
            "must say what is wrong: {}",
            f.detail
        );
    }
}

/// RED. A path that exists and is not a directory.
#[test]
fn writable_goes_red_when_the_home_is_a_file() {
    let d = Dir::new("w-file");
    let file = d.0.join("notadir");
    std::fs::write(&file, b"x").expect("write");
    let f = with_home(&file, check_writable);
    assert!(f.problem, "{}", f.detail);
    assert!(f.detail.contains("not a directory"), "{}", f.detail);
}

#[test]
fn recipes_is_green_when_the_index_lists_some() {
    let d = Dir::new("r-ok");
    std::fs::create_dir_all(d.0.join("atlas-recipes")).expect("mkdir");
    std::fs::write(
        d.0.join("atlas-recipes/index.json"),
        br#"{"recipes":[1,2,3]}"#,
    )
    .expect("write");
    let f = with_home(&d.0, check_recipes);
    assert!(!f.problem, "{}", f.detail);
    assert!(f.detail.contains('3'), "{}", f.detail);
}

/// RED, and the remedy must be `sync-recipes`.
#[test]
fn recipes_goes_red_when_never_written() {
    let d = Dir::new("r-none");
    let f = with_home(&d.0, check_recipes);
    assert!(f.problem);
    assert!(f.detail.contains("never been written"), "{}", f.detail);
    assert!(f.remedy.contains("sync-recipes"), "{}", f.remedy);
}

/// THE DISTINCTION THAT COST HOURS. An index that exists but cannot be READ is
/// not an index that was never written, and it must NOT be answered with "run
/// sync-recipes" — that command goes to the network and then dies on the same
/// unreadable path.
#[test]
fn an_unreadable_index_is_not_reported_as_a_missing_one() {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let d = Dir::new("r-perm");
        std::fs::create_dir_all(d.0.join("atlas-recipes")).expect("mkdir");
        let idx = d.0.join("atlas-recipes/index.json");
        std::fs::write(&idx, br#"{"recipes":[1]}"#).expect("write");
        std::fs::set_permissions(&idx, std::fs::Permissions::from_mode(0o000)).expect("chmod");
        let f = with_home(&d.0, check_recipes);
        std::fs::set_permissions(&idx, std::fs::Permissions::from_mode(0o644)).expect("restore");
        assert!(f.problem);
        assert!(
            f.detail.contains("cannot be read"),
            "must distinguish unreadable from absent: {}",
            f.detail
        );
        assert!(
            !f.remedy.contains("run `spark sync-recipes`"),
            "must NOT send the operator to the network for a permission fault: {}",
            f.remedy
        );
    }
}

/// RED. Present, parses, lists nothing — the state a half-finished sync leaves.
#[test]
fn recipes_goes_red_when_the_index_is_empty() {
    let d = Dir::new("r-empty");
    std::fs::create_dir_all(d.0.join("atlas-recipes")).expect("mkdir");
    std::fs::write(d.0.join("atlas-recipes/index.json"), br#"{"recipes":[]}"#).expect("write");
    let f = with_home(&d.0, check_recipes);
    assert!(f.problem);
    assert!(f.detail.contains("lists none"), "{}", f.detail);
}
