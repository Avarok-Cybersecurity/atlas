// SPDX-License-Identifier: AGPL-3.0-only

//! Tests extracted from the original `api.rs` `sanitizer_tests` module.
//! Each submodule preserves its original tests verbatim; shared helpers
//! live in `common`.

mod common;
// TODO: stale F-code tests — reference helper functions that have been
// refactored away (`f37FailureClass`, `f49_detect_duplicate_writes`,
// `f44_check_permanent_failure`, etc.). Files left on disk; un-comment
// these once the assertions are updated against the current API.
// mod sanitizer;
// mod watchdog_f7;
// mod f28_f32;
// mod f49;
