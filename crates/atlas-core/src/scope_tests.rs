// SPDX-License-Identifier: AGPL-3.0-only

//! These tests share the process-global generation counter, so they serialise
//! on `SERIAL` rather than depending on `--test-threads=1` — a test that only
//! passes under a flag someone has to remember is a test that will fail in CI.

use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;

/// Held for the body of every test in this module. The generation counter is
/// process-global by design, so concurrent tests would invalidate each other's
/// caches mid-assertion.
static SERIAL: Mutex<()> = Mutex::new(());

/// Take the serial lock, ignoring poisoning: a panicking test has already
/// failed and must not cascade into every other test in the module.
fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

#[test]
fn a_generation_is_never_reused() {
    let _serial = serial();
    let a = advance();
    let b = advance();
    assert!(b > a, "{b:?} must follow {a:?}");
    assert_ne!(a, Generation::NONE);
}

#[test]
fn scoped_rebuilds_exactly_once_per_generation() {
    let _serial = serial();
    let cell: Scoped<u64> = Scoped::new();
    let builds = AtomicUsize::new(0);
    let build = || {
        builds.fetch_add(1, Ordering::Relaxed);
        current().as_u64()
    };

    advance();
    let first = cell.get_or_init(build);
    assert_eq!(cell.get_or_init(build), first, "second read is cached");
    assert_eq!(builds.load(Ordering::Relaxed), 1);

    // The model changed: the cached value must not be served.
    advance();
    let second = cell.get_or_init(build);
    assert_ne!(second, first, "a new generation must not see the old value");
    assert_eq!(builds.load(Ordering::Relaxed), 2);
}

#[test]
fn a_value_from_a_previous_generation_is_never_visible() {
    let _serial = serial();
    let cell: Scoped<&'static str> = Scoped::new();
    advance();
    cell.get_or_init(|| "model-a");
    assert_eq!(cell.get(), Some("model-a"));
    advance();
    assert_eq!(
        cell.get(),
        None,
        "this is the whole point: no stale read, not even once"
    );
}

#[test]
fn a_failed_derivation_is_not_cached() {
    let _serial = serial();
    let cell: Scoped<u32> = Scoped::new();
    advance();
    let attempts = AtomicUsize::new(0);
    let try_build = |ok: bool| {
        cell.get_or_try_init(|| {
            attempts.fetch_add(1, Ordering::Relaxed);
            if ok { Ok(7u32) } else { Err("cuda busy") }
        })
    };
    assert!(try_build(false).is_err());
    assert_eq!(cell.get(), None, "a failure must leave the slot empty");
    assert_eq!(try_build(true).unwrap(), 7);
    assert_eq!(attempts.load(Ordering::Relaxed), 2);
    // Now it caches.
    assert_eq!(try_build(false).unwrap(), 7);
}

#[test]
fn scoped_map_misses_on_a_recycled_key_from_a_previous_generation() {
    let _serial = serial();
    // The hazard this type exists for: the key is a device pointer, the model
    // is released, the next model allocates and the allocator hands back the
    // same address. A generation-compare-on-read would still HIT.
    let map: ScopedMap<u64, &'static str> = ScopedMap::new();
    let recycled_ptr = 0x7f00_1234_0000u64;
    advance();
    assert_eq!(
        map.get_or_insert_with(recycled_ptr, || "model-a-tensor"),
        "model-a-tensor"
    );
    assert_eq!(map.len(), 1);

    advance();
    assert_eq!(
        map.len(),
        0,
        "entries do not survive the model that made them"
    );
    assert_eq!(
        map.get_or_insert_with(recycled_ptr, || "model-b-tensor"),
        "model-b-tensor",
        "the same address must describe the NEW allocation"
    );
}

#[test]
fn clearing_a_scoped_cell_drops_the_value_it_holds() {
    let _serial = serial();
    // The case this exists for: a cell holding an owning handle. Advancing the
    // generation stops it being SERVED, but only `clear` stops it being HELD —
    // and teardown blocks on the handle, not on the read.
    let cell: Scoped<std::sync::Arc<u32>> = Scoped::new();
    advance();
    let held = cell.get_or_init(|| std::sync::Arc::new(1));
    assert_eq!(std::sync::Arc::strong_count(&held), 2, "the cell holds one");
    advance();
    assert!(cell.get().is_none(), "not served across a generation");
    assert_eq!(
        std::sync::Arc::strong_count(&held),
        2,
        "but still HELD — refusing to serve is not releasing"
    );
    cell.clear();
    assert_eq!(std::sync::Arc::strong_count(&held), 1, "now released");
}

#[test]
fn clear_empties_the_map_within_a_generation() {
    let _serial = serial();
    let map: ScopedMap<u64, u32> = ScopedMap::new();
    advance();
    map.get_or_insert_with(1, || 10);
    map.get_or_insert_with(2, || 20);
    assert_eq!(map.len(), 2);
    map.clear();
    assert!(
        map.is_empty(),
        "clear runs before the memory the keys point into is freed"
    );
}

// ── Teardown ────────────────────────────────────────────────────────────────

struct Fake {
    label: &'static str,
    log: std::sync::Arc<Mutex<Vec<&'static str>>>,
    fail: bool,
    released: usize,
}

/// Stands in for the allocator a real resource frees through.
struct Allocator;

impl ModelResource<Allocator> for Fake {
    fn label(&self) -> &'static str {
        self.label
    }
    fn release(&mut self, _cx: &Allocator) -> anyhow::Result<()> {
        self.released += 1;
        self.log.lock().unwrap().push(self.label);
        if self.fail {
            anyhow::bail!("{} refused", self.label);
        }
        Ok(())
    }
}

fn fake(
    label: &'static str,
    log: &std::sync::Arc<Mutex<Vec<&'static str>>>,
    fail: bool,
) -> Box<dyn ModelResource<Allocator>> {
    Box::new(Fake {
        label,
        log: log.clone(),
        fail,
        released: 0,
    })
}

#[test]
fn teardown_releases_in_reverse_construction_order() {
    let _serial = serial();
    let log = std::sync::Arc::new(Mutex::new(Vec::new()));
    let mut t: Teardown<Allocator> = Teardown::new();
    t.push(fake("weights", &log, false));
    t.push(fake("kv-pool", &log, false));
    t.push(fake("modules", &log, false));
    assert_eq!(t.len(), 3);
    t.release_all(&Allocator).unwrap();
    assert_eq!(
        *log.lock().unwrap(),
        vec!["modules", "kv-pool", "weights"],
        "later resources borrow earlier ones, so they go first"
    );
    assert!(t.is_empty());
}

#[test]
fn one_failure_does_not_abandon_the_rest() {
    let _serial = serial();
    let log = std::sync::Arc::new(Mutex::new(Vec::new()));
    let mut t: Teardown<Allocator> = Teardown::new();
    t.push(fake("weights", &log, false));
    t.push(fake("kv-pool", &log, true));
    t.push(fake("modules", &log, false));
    let err = t.release_all(&Allocator).unwrap_err();
    assert_eq!(
        *log.lock().unwrap(),
        vec!["modules", "kv-pool", "weights"],
        "a half-torn-down GPU is worse than a reported error"
    );
    assert!(format!("{err:#}").contains("kv-pool"), "{err:#}");
    assert!(format!("{err:#}").contains("1 resource(s)"), "{err:#}");
}
