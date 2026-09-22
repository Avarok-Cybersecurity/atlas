// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! The reader pool behind [`ExpertLru`](super::expert_lru::ExpertLru): a fixed
//! set of threads serving byte-range reads of expert slices into their slots,
//! in two priorities (the experts a layer is waiting for, then the ones a
//! prediction asked for), with a completion ticket per slot so a waiter can
//! block on exactly the slots it needs.
//!
//! One 12.22 MiB expert reads in 1.8-2.0 ms on the Spark's NVMe however it is
//! split; the disk's 11 GB/s only appears with several experts in flight. So
//! the pool exists to keep predicted experts in flight while the GPU works,
//! not to make one read faster.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};

use anyhow::{Result, bail};

/// One slot's outstanding read: `parts` byte ranges, done when all landed.
pub struct Ticket {
    state: Mutex<(usize, Option<String>)>,
    cv: Condvar,
}

impl Ticket {
    pub fn new(parts: usize) -> Arc<Self> {
        Arc::new(Ticket {
            state: Mutex::new((parts, None)),
            cv: Condvar::new(),
        })
    }

    /// One part landed (or failed).
    pub fn finish(&self, r: Result<()>) {
        let mut g = self.state.lock().unwrap();
        if let Err(e) = r
            && g.1.is_none()
        {
            g.1 = Some(format!("{e:#}"));
        }
        g.0 = g.0.saturating_sub(1);
        if g.0 == 0 {
            self.cv.notify_all();
        }
    }

    pub fn is_done(&self) -> bool {
        self.state.lock().unwrap().0 == 0
    }

    /// Block until every part landed; the first failure, if any.
    pub fn wait(&self) -> Result<()> {
        let mut g = self.state.lock().unwrap();
        while g.0 > 0 {
            g = self.cv.wait(g).unwrap();
        }
        match &g.1 {
            None => Ok(()),
            Some(e) => bail!("expert read failed: {e}"),
        }
    }

    /// `Some(error)` once done and failed, `None` otherwise.
    pub fn error(&self) -> Option<String> {
        let g = self.state.lock().unwrap();
        if g.0 == 0 { g.1.clone() } else { None }
    }
}

pub type Job = Box<dyn FnOnce() + Send>;

struct Queues {
    urgent: VecDeque<Job>,
    background: VecDeque<Job>,
    /// workers busy with a background job right now
    bg_active: usize,
    closed: bool,
}

/// `threads` workers over two FIFO queues; urgent jobs always go first, and
/// at most `threads - reserve` workers may be on background jobs at once, so
/// a miss that is not in flight always finds a reader at once.
pub struct ReaderPool {
    q: Arc<(Mutex<Queues>, Condvar)>,
    handles: Vec<std::thread::JoinHandle<()>>,
    threads: usize,
}

impl ReaderPool {
    pub fn new(threads: usize) -> Self {
        let threads = threads.max(1);
        let reserve = (threads / 4)
            .max(1)
            .min(threads - 1)
            .max(if threads == 1 { 0 } else { 1 });
        let bg_max = threads - reserve;
        let q = Arc::new((
            Mutex::new(Queues {
                urgent: VecDeque::new(),
                background: VecDeque::new(),
                bg_active: 0,
                closed: false,
            }),
            Condvar::new(),
        ));
        let handles = (0..threads)
            .map(|i| {
                let q = q.clone();
                std::thread::Builder::new()
                    .name(format!("expert-reader-{i}"))
                    .spawn(move || {
                        loop {
                            let (job, bg) = {
                                let (m, cv) = &*q;
                                let mut g = m.lock().unwrap();
                                loop {
                                    if let Some(j) = g.urgent.pop_front() {
                                        break (Some(j), false);
                                    }
                                    if g.bg_active < bg_max
                                        && let Some(j) = g.background.pop_front()
                                    {
                                        g.bg_active += 1;
                                        break (Some(j), true);
                                    }
                                    if g.closed {
                                        break (None, false);
                                    }
                                    g = cv.wait(g).unwrap();
                                }
                            };
                            match job {
                                Some(j) => {
                                    j();
                                    if bg {
                                        let (m, cv) = &*q;
                                        m.lock().unwrap().bg_active -= 1;
                                        cv.notify_all();
                                    }
                                }
                                None => return,
                            }
                        }
                    })
                    .expect("spawn expert reader")
            })
            .collect();
        ReaderPool {
            q,
            handles,
            threads,
        }
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    pub fn submit(&self, job: Job, urgent: bool) {
        let (m, cv) = &*self.q;
        let mut g = m.lock().unwrap();
        if urgent {
            g.urgent.push_back(job);
        } else {
            g.background.push_back(job);
        }
        drop(g);
        cv.notify_one();
    }
}

impl Drop for ReaderPool {
    fn drop(&mut self) {
        {
            let (m, cv) = &*self.q;
            m.lock().unwrap().closed = true;
            cv.notify_all();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}
