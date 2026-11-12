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
    // @port:fields
}
