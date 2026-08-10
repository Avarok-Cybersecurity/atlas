// SPDX-License-Identifier: AGPL-3.0-only

//! `advance()` is monotonic and shared, so the generation test asserts only
//! about generations it created, never about an absolute value. The teardown
//! tests touch no global state at all.

use std::sync::Mutex;

use super::*;

/// Stands in for the allocator a real resource frees through.
struct Allocator;

struct Fake {
    label: &'static str,
    log: std::sync::Arc<Mutex<Vec<&'static str>>>,
    fail: bool,
}

impl ModelResource<Allocator> for Fake {
    fn label(&self) -> &'static str {
        self.label
    }
    fn release(&mut self, _cx: &Allocator) -> anyhow::Result<()> {
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
    })
}

#[test]
fn teardown_releases_in_reverse_construction_order() {
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
