// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the per-sequence scheduler state the port sprint's
//! (`port/consumer-opt`) lanes add, held as ONE field (`port`) on `ActiveSeq`.
//!
//! One field per lane; each field's type is defined in the lane's own module
//! and implements `Default`, so the `ActiveSeq { .. }` literal sites never
//! change again (`port: Default::default()`). Lines are inserted immediately
//! BEFORE the marker below by the coordinator's fragment apply pass.

/// The port lanes' per-sequence state. Empty until a lane adds a field; a
/// row-off lane's field is its type's `Default` and is never read.
#[derive(Default)]
pub struct SeqPortState {
    /// `atlas_speculative: false` on the request (L05): this sequence never
    /// enters a speculative step. Set by `SeqPortState::for_request` at the
    /// sequence's creation sites; a round planner (L13) must honor it.
    pub no_drafts: bool,
    /// D-Cut's stash for the step in flight (L06): written by `mtp_dcut::plan`,
    /// taken by `mtp_stats::account_step`.
    pub dcut: crate::scheduler::mtp_dcut::cost::SeqDcut,
    /// The copy drafter's per-sequence index and estimators (L13). Never
    /// allocated while `copy_draft` is off.
    pub copy: crate::scheduler::copy_draft::SeqCopyState,
    // @port:fields
}
