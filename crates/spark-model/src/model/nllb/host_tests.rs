// SPDX-License-Identifier: AGPL-3.0-only

//! Non-CUDA test harness for NLLB's host-only language and position helpers.

#[path = "lang.rs"]
mod lang;
pub use lang::NllbLang;

#[path = "util.rs"]
mod util;
