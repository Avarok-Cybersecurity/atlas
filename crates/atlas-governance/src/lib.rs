// SPDX-License-Identifier: AGPL-3.0-only

//! The PR journey ledger: an append-only record of how a change reached main.
//!
//! `.benchmarks/` already answers *"did this commit pass?"*. It cannot answer
//! *"how did this pull request get here?"* — which gates were re-opened and by
//! what, which runs superseded which, what the classifier thought at the time.
//! Those questions are traversals over a history, not lookups in a directory.
//!
//! # Two representations, and which one is authoritative
//!
//! ```text
//!   CANONICAL  ── governance/pr-<n>.jsonl ──  text, append-only, in git
//!        │                                    reviewable in a diff
//!        │ materialize()
//!        ▼
//!   DERIVED    ── lattice-db graph        ──  binary, local, disposable
//!                                            traversals and similarity
//! ```
//!
//! The JSONL is the system of record and the only thing committed. The graph is
//! rebuilt from it on demand and never stored, because a binary database file
//! is unmergeable and undiffable in git — which would recreate exactly the
//! conflict problem a per-PR ledger exists to avoid.
//!
//! # Why the canonical form is a grow-only set
//!
//! Records are keyed `(head_sha, run_id, attempt)` and only ever appended.
//! Merging two versions of a file is then set union, which is associative,
//! commutative and idempotent — a CRDT G-Set. Two CI jobs appending
//! concurrently cannot conflict semantically, and `.gitattributes` can declare
//! `merge=union` so they do not conflict textually either.
//!
//! The directory holding those files is separately bounded to the newest 100
//! PRs, with older ones removed from the tree and indexed by commit hash in
//! `governance/archive.csv` (see that directory's README). That bounds the
//! DIRECTORY, not the ledgers: eviction removes a whole file, never a line
//! from one, so the G-Set property described here is unaffected — and the
//! evicted content is still in git history, byte-identical, at the recorded
//! hash.
//!
//! It is the same order-theoretic shape as the gate's required set in
//! `atlas-plugin`'s `gate::coverage` (not linked: this crate deliberately does
//! not depend on it, since the ledger must never be able to reach the gate):
//! things are added, never removed, so the result does not depend on the order
//! they arrived in.
//!
//! # It is advisory, permanently
//!
//! Nothing here is read by `--pull-request-gate-check`. The gate's verdict is a
//! function of the tree, git history and committed records — adding a ledger
//! read would make it depend on a file any job can append to, which is the
//! property the gate is careful not to have.

pub mod eigenvector;
pub mod event;
pub mod ledger;

#[cfg(test)]
#[path = "ledger_tests.rs"]
mod ledger_tests;

pub use event::{Event, EventKind, Verdict};
pub use ledger::{Journey, append, materialize, read_all};
