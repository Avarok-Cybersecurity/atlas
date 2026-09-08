// SPDX-License-Identifier: AGPL-3.0-only

//! Named serve presets: a checkpoint plus the validated `spark serve`
//! configuration for it, declared per kernel target in MODEL.toml
//! `[[serve_presets]]` and baked into the binary by `build.rs`.
//!
//! A kernel target already carries what is true of EVERY checkpoint it serves
//! (`[behavior]`, `[sampling.*]`). A preset carries what is true of ONE
//! checkpoint's validated deployment — the HF id and branch, the flags that
//! were measured together, and the `ATLAS_*` gates its code path needs — so
//! that `spark serve <preset-name>` reproduces a gate-passing configuration
//! without a wall of flags, while every entry stays a DEFAULT the operator can
//! override on the command line or in the environment.
//!
//! Resolution and application live in spark-server
//! (`main_modules::serve_presets`): this crate only declares and carries the
//! data.

/// One `[[serve_presets]]` entry from a target's MODEL.toml.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServePreset {
    /// The servable name: `spark serve <name>`. Unique across every compiled
    /// target, and never equal to a kernel-target directory name (build.rs
    /// enforces both), so a lookup by name is unambiguous.
    pub name: &'static str,
    /// HuggingFace repo id of the checkpoint this preset was validated on.
    pub hf_id: &'static str,
    /// HF branch / revision to resolve in the hub cache (`refs/<revision>`).
    /// Empty = the repo's default revision (`refs/main`). Quantizers that ship
    /// several bit-widths as branches (turboderp's EXL3 repos) need this: the
    /// default branch is not the validated checkpoint.
    pub hf_revision: &'static str,
    /// One line for listings and the startup log.
    pub description: &'static str,
    /// `spark serve` flag DEFAULTS, keyed exactly like a recipe `defaults:`
    /// block (the `ServeArgs` field name, e.g. `max_seq_len`) with the value as
    /// text. Rendered to argv through the recipe schema and re-parsed by clap,
    /// so clap stays the single source of truth for the flag surface; a flag
    /// the operator passed is NOT rendered, so the command line always wins.
    pub flags: &'static [(&'static str, &'static str)],
    /// `ATLAS_*` DEFAULTS the preset's code path needs. Applied only to
    /// variables that are unset when the process starts — an operator's value
    /// is kept and the deviation is logged. Values may reference the effective
    /// serve flags as `{max_seq_len}` / `{max_prefill_tokens}` for caps that
    /// must track a flag rather than a literal.
    pub env: &'static [(&'static str, &'static str)],
}
