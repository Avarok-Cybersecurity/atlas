// SPDX-License-Identifier: AGPL-3.0-only
// Algorithm: N. Megiddo and D. S. Modha, "ARC: A Self-Tuning, Low Overhead Replacement Cache", FAST 2003.

//! Responsibility: a deliberately naive reference ARC for the trace tests. Plain vectors (index
//! 0 is the least recent), linear searches, the rules written out as in the paper, adapted the
//! same documented way as `policy_arc.rs` (admission never evicts; replacement skips rejected
//! keys and falls back to the other list). It shares no code with the policy under test.

use crate::kv_host_tier::BlockKey;

pub(super) struct RefArc {
    pub(super) c: usize,
    pub(super) p: usize,
    pub(super) t1: Vec<BlockKey>,
    pub(super) t2: Vec<BlockKey>,
    pub(super) b1: Vec<BlockKey>,
    pub(super) b2: Vec<BlockKey>,
}

fn take_out(v: &mut Vec<BlockKey>, k: &BlockKey) -> bool {
    match v.iter().position(|x| x == k) {
        Some(i) => {
            v.remove(i);
            true
        }
        None => false,
    }
}

impl RefArc {
    pub(super) fn new(c: usize) -> Self {
        Self {
            c: c.max(1),
            p: 0,
            t1: Vec::new(),
            t2: Vec::new(),
            b1: Vec::new(),
            b2: Vec::new(),
        }
    }

    pub(super) fn admit(&mut self, k: BlockKey) {
        if self.t1.contains(&k) || self.t2.contains(&k) {
            self.hit(k);
            return;
        }
        if self.b1.contains(&k) {
            let d = std::cmp::max(1, self.b2.len() / self.b1.len());
            self.p = std::cmp::min(self.c, self.p + d);
            take_out(&mut self.b1, &k);
            self.t2.push(k);
        } else if self.b2.contains(&k) {
            let d = std::cmp::max(1, self.b1.len() / self.b2.len());
            self.p = self.p.saturating_sub(d);
            take_out(&mut self.b2, &k);
            self.t2.push(k);
        } else {
            self.t1.push(k);
        }
        self.trim();
    }

    pub(super) fn hit(&mut self, k: BlockKey) {
        if take_out(&mut self.t1, &k) || take_out(&mut self.t2, &k) {
            self.t2.push(k);
        } else {
            self.admit(k);
        }
    }

    pub(super) fn forget(&mut self, k: &BlockKey) {
        if !take_out(&mut self.t1, k) {
            take_out(&mut self.t2, k);
        }
    }

    pub(super) fn victims(&mut self, want: usize, ok: &dyn Fn(&BlockKey) -> bool) -> Vec<BlockKey> {
        let mut out = Vec::new();
        while out.len() < want {
            let prefer_t1 = !self.t1.is_empty() && (self.t1.len() > self.p || self.t2.is_empty());
            let order = if prefer_t1 { [1, 2] } else { [2, 1] };
            let mut got = None;
            for list in order {
                let (src, ghost) = if list == 1 {
                    (&mut self.t1, &mut self.b1)
                } else {
                    (&mut self.t2, &mut self.b2)
                };
                if let Some(i) = src.iter().position(ok) {
                    let k = src.remove(i);
                    take_out(ghost, &k);
                    ghost.push(k);
                    got = Some(k);
                    break;
                }
            }
            match got {
                Some(k) => out.push(k),
                None => break,
            }
        }
        self.trim();
        out
    }

    fn trim(&mut self) {
        while self.t1.len() + self.b1.len() > self.c && !self.b1.is_empty() {
            self.b1.remove(0);
        }
        while self.t1.len() + self.t2.len() + self.b1.len() + self.b2.len() > 2 * self.c {
            if !self.b2.is_empty() {
                self.b2.remove(0);
            } else if !self.b1.is_empty() {
                self.b1.remove(0);
            } else {
                break;
            }
        }
    }

    pub(super) fn where_is(&self, k: &BlockKey) -> &'static str {
        if self.t1.contains(k) {
            "t1"
        } else if self.t2.contains(k) {
            "t2"
        } else if self.b1.contains(k) {
            "b1"
        } else if self.b2.contains(k) {
            "b2"
        } else {
            "none"
        }
    }
}
