// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the fake CUDA-graph driver and the cache harness that
//! `graph_family_tests.rs` drives (test-only).
//!
//! The fake records, for every template and executable, its topology class
//! and the key whose parameters it bakes, and panics on a double destroy, so
//! the tests can check leaks, dangling handles and what each key replays.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};

use anyhow::{Result, bail};
use spark_runtime::gpu::{GraphHandle, GraphTemplate};

use super::super::{FamilyOps, Finish, LruGraphCache, finish_capture};
use crate::port::graph_stats::{GraphCounters, GraphCounts};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum UpdateMode {
    /// Apply when the topology matches, refuse otherwise (the driver's rule).
    Apply,
    /// Refuse every update (`CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE`).
    Refuse,
    /// Return an error from every update.
    Fail,
}

/// What a fake graph bakes: its topology class and the key it was captured for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Body {
    pub(super) sig: u64,
    pub(super) content: Vec<u32>,
}

pub(super) struct Fake {
    pub(super) next_id: Cell<u64>,
    next_capture: RefCell<Option<Body>>,
    pub(super) templates: RefCell<HashMap<u64, Body>>,
    pub(super) execs: RefCell<HashMap<u64, Body>>,
    pub(super) update: Cell<UpdateMode>,
    pub(super) topology_fails: Cell<bool>,
    pub(super) instantiate_fails: Cell<bool>,
    pub(super) instantiate_null: Cell<bool>,
    pub(super) null_capture: Cell<bool>,
    calls: RefCell<Vec<&'static str>>,
}

impl Fake {
    pub(super) fn new() -> Self {
        Self {
            next_id: Cell::new(100),
            next_capture: RefCell::new(None),
            templates: RefCell::new(HashMap::new()),
            execs: RefCell::new(HashMap::new()),
            update: Cell::new(UpdateMode::Apply),
            topology_fails: Cell::new(false),
            instantiate_fails: Cell::new(false),
            instantiate_null: Cell::new(false),
            null_capture: Cell::new(false),
            calls: RefCell::new(Vec::new()),
        }
    }

    fn id(&self) -> u64 {
        let v = self.next_id.get();
        self.next_id.set(v + 1);
        v
    }

    /// Record a capture of `key` in topology class `sig` (begin_capture + body).
    pub(super) fn capture(&self, sig: u64, key: &[u32]) {
        *self.next_capture.borrow_mut() = Some(Body {
            sig,
            content: key.to_vec(),
        });
    }

    fn call(&self, name: &'static str) {
        self.calls.borrow_mut().push(name);
    }

    /// The driver calls since the last take, in order.
    pub(super) fn take_calls(&self) -> Vec<&'static str> {
        std::mem::take(&mut *self.calls.borrow_mut())
    }
}

impl FamilyOps for Fake {
    fn close_capture(&self, _stream: u64) -> Result<GraphTemplate> {
        self.call("close");
        let body = self
            .next_capture
            .borrow_mut()
            .take()
            .expect("a capture was begun");
        if self.null_capture.get() {
            return Ok(GraphTemplate(0));
        }
        let id = self.id();
        self.templates.borrow_mut().insert(id, body);
        Ok(GraphTemplate(id))
    }

    fn topology(&self, t: &GraphTemplate) -> Result<u64> {
        self.call("topology");
        if self.topology_fails.get() {
            bail!("fake topology failure");
        }
        Ok(self.templates.borrow()[&t.0].sig)
    }

    fn exec_update(&self, exec: GraphHandle, t: &GraphTemplate) -> Result<bool> {
        self.call("update");
        match self.update.get() {
            UpdateMode::Fail => bail!("fake update error"),
            UpdateMode::Refuse => Ok(false),
            UpdateMode::Apply => {
                let body = self.templates.borrow()[&t.0].clone();
                let mut execs = self.execs.borrow_mut();
                let live = execs.get_mut(&exec.0).expect("update of a live executable");
                if live.sig != body.sig {
                    return Ok(false);
                }
                *live = body;
                Ok(true)
            }
        }
    }

    fn instantiate_template(&self, t: &GraphTemplate) -> Result<GraphHandle> {
        self.call("instantiate");
        if self.instantiate_fails.get() {
            bail!("fake instantiate failure");
        }
        if self.instantiate_null.get() {
            return Ok(GraphHandle(0));
        }
        let body = self.templates.borrow()[&t.0].clone();
        let id = self.id();
        self.execs.borrow_mut().insert(id, body);
        Ok(GraphHandle(id))
    }

    fn drop_template(&self, t: GraphTemplate) -> Result<()> {
        self.call("drop_template");
        let removed = self.templates.borrow_mut().remove(&t.0);
        assert!(removed.is_some(), "template {} destroyed twice", t.0);
        Ok(())
    }

    fn drop_exec(&self, exec: GraphHandle) -> Result<()> {
        self.call("drop_exec");
        let removed = self.execs.borrow_mut().remove(&exec.0);
        assert!(removed.is_some(), "executable {} destroyed twice", exec.0);
        Ok(())
    }
}

/// The cache, its signature table and the counters of one family.
pub(super) struct Family {
    pub(super) cache: LruGraphCache,
    pub(super) sigs: HashMap<u64, u64>,
    pub(super) counters: GraphCounters,
    pub(super) cap: usize,
}

impl Family {
    pub(super) fn new(cap: usize) -> Self {
        Self {
            cache: (HashMap::new(), 0),
            sigs: HashMap::new(),
            counters: GraphCounters::new(),
            cap,
        }
    }

    /// The call sites' hit test: bump the tick, then touch the entry on a hit.
    pub(super) fn lookup(&mut self, key: &[u32]) -> Option<GraphHandle> {
        touch(&mut self.cache, key)
    }

    /// A miss: capture `key` in class `sig`, then finish it.
    pub(super) fn miss(
        &mut self,
        f: &Fake,
        key: &[u32],
        sig: u64,
    ) -> Result<(GraphHandle, Finish)> {
        f.capture(sig, key);
        finish_capture(
            f,
            &mut self.sigs,
            &mut self.cache,
            self.cap,
            key.to_vec(),
            7,
            &self.counters,
            "test",
        )
    }

    pub(super) fn counts(&self) -> GraphCounts {
        self.counters.snapshot()
    }

    pub(super) fn keys(&self) -> HashSet<Vec<u32>> {
        self.cache.0.keys().cloned().collect()
    }

    /// Key to tick, the part of the cache the row must not change.
    pub(super) fn ticks(&self) -> HashMap<Vec<u32>, u64> {
        ticks(&self.cache)
    }
}

/// `decode_a2.rs` / `verify_e.rs`: `g.1 += 1` on every lookup, then the hit's
/// entry takes the new tick.
pub(super) fn touch(cache: &mut LruGraphCache, key: &[u32]) -> Option<GraphHandle> {
    cache.1 += 1;
    let tick = cache.1;
    let e = cache.0.get_mut(key)?;
    e.1 = tick;
    Some(e.0)
}

/// Key to tick of any cache.
pub(super) fn ticks(cache: &LruGraphCache) -> HashMap<Vec<u32>, u64> {
    cache.0.iter().map(|(k, (_, t))| (k.clone(), *t)).collect()
}

/// No leaks, no dangling executables, every key replays its own parameters,
/// and every recorded signature of a cached executable is its true class.
pub(super) fn check_invariants(f: &Fake, fam: &Family) {
    assert!(f.templates.borrow().is_empty(), "a template leaked");
    let execs = f.execs.borrow();
    let cached: HashSet<u64> = fam.cache.0.values().map(|(g, _)| g.0).collect();
    let live: HashSet<u64> = execs.keys().copied().collect();
    assert_eq!(cached, live, "cached executables == live executables");
    assert_eq!(cached.len(), fam.cache.0.len(), "one executable per key");
    for (key, (g, _)) in &fam.cache.0 {
        assert_eq!(&execs[&g.0].content, key, "{key:?} replays its own capture");
        if let Some(sig) = fam.sigs.get(&g.0) {
            assert_eq!(*sig, execs[&g.0].sig, "recorded class of {}", g.0);
        }
    }
    assert!(fam.cache.0.len() <= fam.cap);
}

/// Miss on each key in turn (all instantiate), then forget the driver calls.
pub(super) fn fill(f: &Fake, fam: &mut Family, keys: &[&[u32]], sig: u64) {
    for k in keys {
        assert!(fam.lookup(k).is_none());
        let (_, how) = fam.miss(f, k, sig).unwrap();
        assert_eq!(how, Finish::Instantiated);
    }
    f.take_calls();
}
