// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash drafter levers, resolved once per head and then carried.
//!
//! # ★ THE ENVIRONMENT IS READ EXACTLY ONCE PER HEAD. KEEP IT THAT WAY.
//!
//! `forward_block` runs once per DECODE STEP, and its per-layer helpers run
//! `num_layers` times inside that. Every `std::env::var` on that path
//! allocates a `String` and takes the PROCESS-WIDE environment lock, so
//! concurrent decode threads serialise against each other on it —
//! MEASURED on GB10: a 30-variable resolve costs 0.57 us single-threaded but
//! **4.00 us at 8 threads and 5.76 us at 16**. The cost grows with
//! concurrency, which is exactly why no single-stream benchmark shows it.
//!
//! Before this module `forward_block` read **31** variables per propose, ten
//! of them duplicates of a variable it had already read in the same call, and
//! eleven of them in a single `&&` chain whose only job was to answer one
//! question: is any diagnostic armed?
//!
//! These are resolved at head construction rather than in a `OnceLock`
//! static, for the reason [`crate::layers::ops::ModelLevers`] gives: a static
//! outlives the model whose flags it encodes, so a second model silently
//! keeps the first one's branches. A field on the head cannot go stale,
//! because a new head is a new resolution.

/// Diagnostic and A/B levers for one loaded DFlash drafter.
///
/// Plain `Copy` data. Every field is a pure function of one `ATLAS_*`
/// variable except [`Self::any_diagnostic_armed`], which is a function of
/// eleven of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct DFlashLevers {
    /// Any of the eleven diagnostic variables is **SET**, at any value.
    ///
    /// This is the CUDA-graph kill switch: a diagnostic that dumps or logs
    /// from inside the captured region would be captured with it and then
    /// replayed silently, so an armed diagnostic forces the eager path.
    ///
    /// ★ Presence, not truth — `ATLAS_DFLASH_BLOCK_DUMP=0` suppresses graph
    /// capture while enabling no dump at all. That is the shipped behaviour
    /// (the chain this replaces tested `std::env::var(..).is_err()`), and it
    /// is pinned by a test rather than quietly fixed: the variables are
    /// operator-facing A/B switches, and a graph capture that appears only
    /// when a flag is spelled a particular way is a worse surprise than an
    /// over-eager kill switch.
    ///
    /// One deliberate divergence from the `is_err()` chain this replaces: it
    /// tested `std::env::var`, which reports a NON-UTF-8 value as absent, so
    /// `ATLAS_DFLASH_BLOCK_DUMP=<invalid utf-8>` used to leave capture ON.
    /// This uses `var_os`, so such a value now suppresses capture — the same
    /// direction as every other spelling, and the same `present` idiom
    /// `ModelLevers` uses.
    pub any_diagnostic_armed: bool,

    // ── One-shot dumps ──
    /// `ATLAS_DFLASH_DEBUG_DUMP=1` — first 10 BF16 floats of each key
    /// intermediate, for element-wise comparison against a Python reference.
    pub debug_dump: bool,
    /// `ATLAS_DFLASH_DEBUG_DUMP_FULL=1` — full tensors, not the first 10.
    pub debug_dump_full: bool,
    /// `ATLAS_DFLASH_LOG_DRAFTS=1` — log the γ drafts each propose returns.
    pub log_drafts: bool,
    /// `ATLAS_DFLASH_BLOCK_DUMP=1` — per-layer `.bin` dumps of the block
    /// inputs and every layer's output.
    pub block_dump: bool,
    /// `ATLAS_DFLASH_BLOCK_DUMP_AT_POS=<n>` — arm the block dump only at
    /// decode position ≥ n, so the dump can be taken in the regime where
    /// absolute positions have diverged from ctx slot indices. Default 0
    /// (dump at the first propose).
    pub block_dump_at_pos: usize,
    /// `ATLAS_DFLASH_OPTION_B_DIAG=1` — read back layer 0's first cached
    /// K/V row from the paged drafter cache.
    pub option_b_diag: bool,

    // ── Forced inputs (reference-comparison A/B) ──
    /// `ATLAS_DFLASH_DEBUG_FORCE_PATTERN=1` — overwrite the captured target
    /// hidden with a deterministic pattern the PyTorch reference also makes.
    pub force_pattern: bool,
    /// `ATLAS_DFLASH_DEBUG_FORCE_NOISE_PATTERN=1` — same, for the noise rows.
    pub force_noise_pattern: bool,
    /// `ATLAS_DFLASH_DEBUG_CTX_OFF=1` — drop ctx conditioning entirely
    /// (`eff_ctx = 0`), the A/B for whether the drafter responds to ctx.
    pub force_no_ctx: bool,
    /// `ATLAS_DFLASH_DEBUG_CTX_USED=<n>` — pin `eff_ctx` to exactly n.
    pub force_ctx_used: Option<usize>,

    // ── Precompute diagnostics ──
    /// `ATLAS_DFLASH_PRECOMPUTE=1` — run the ctx K/V precompute chain from
    /// `forward_block` (the production path runs it from `propose`).
    pub precompute: bool,
    /// `ATLAS_DFLASH_PRECOMPUTE_COMMIT=1` — let that diagnostic run write to
    /// the paged cache. Off by default because `forward_block` does not
    /// guarantee a valid block table.
    pub precompute_commit: bool,

    // ── Graph capture ──
    /// `ATLAS_DFLASH_PROPOSE_WARMUP_N=<n>` — eager warm-up passes before
    /// capture. Default 2: two passes warm the PTX→SASS cache, ramp GB10
    /// clocks, and pull hot weight tiles into L2 before capture freezes the
    /// SASS variants the driver picked.
    pub propose_warmup_n: usize,

    // ── DSpark ──
    /// `ATLAS_DSPARK_SHIFT=1|0` forces the SpecForge shifted-row convention
    /// on or off; unset (`None`) defers to the drafter config.
    pub dspark_shift: Option<bool>,
    /// Row 0 carries the Markov anchor bias. Ships ON;
    /// `ATLAS_DSPARK_ANCHOR_BIAS=0` exempts it. Confidence truncation reads
    /// this too — rows without the chain never write their confidence slot.
    pub dspark_anchor_bias: bool,
    /// `ATLAS_DSPARK_CONF_TRACE=1` — log the confidence logits and sigmoids.
    pub dspark_conf_trace: bool,
}

/// The eleven variables whose mere PRESENCE forces the eager path.
///
/// Named as one list because they are one predicate. Adding a diagnostic that
/// writes from inside `forward_block` and forgetting to add it here means the
/// diagnostic gets captured into the graph and replayed — which reads as a
/// dump that never updates, not as an error.
const GRAPH_SUPPRESSING_DIAGNOSTICS: [&str; 11] = [
    "ATLAS_DFLASH_PROPOSE_NO_GRAPH",
    "ATLAS_DFLASH_DEBUG_DUMP_FULL",
    "ATLAS_DFLASH_OPTION_B_DIAG",
    "ATLAS_DFLASH_PRECOMPUTE_DUMP",
    "ATLAS_DFLASH_VERIFY_TRACE",
    "ATLAS_DFLASH_LOG_DRAFTS",
    "ATLAS_DFLASH_DEBUG_FORCE_PATTERN",
    "ATLAS_DFLASH_DEBUG_FORCE_NOISE_PATTERN",
    "ATLAS_DFLASH_DEBUG_CTX_OFF",
    "ATLAS_DFLASH_DEBUG_CTX_USED",
    "ATLAS_DFLASH_BLOCK_DUMP",
];

fn from_values(
    mut value: impl FnMut(&str) -> Option<String>,
    mut present: impl FnMut(&str) -> bool,
) -> DFlashLevers {
    fn opt_in(value: Option<&str>) -> bool {
        value == Some("1")
    }
    fn parsed<T: std::str::FromStr>(value: Option<&str>) -> Option<T> {
        value.and_then(|v| v.parse().ok())
    }

    DFlashLevers {
        any_diagnostic_armed: GRAPH_SUPPRESSING_DIAGNOSTICS.iter().any(|var| present(var)),

        debug_dump: opt_in(value("ATLAS_DFLASH_DEBUG_DUMP").as_deref()),
        debug_dump_full: opt_in(value("ATLAS_DFLASH_DEBUG_DUMP_FULL").as_deref()),
        log_drafts: opt_in(value("ATLAS_DFLASH_LOG_DRAFTS").as_deref()),
        block_dump: opt_in(value("ATLAS_DFLASH_BLOCK_DUMP").as_deref()),
        block_dump_at_pos: parsed(value("ATLAS_DFLASH_BLOCK_DUMP_AT_POS").as_deref()).unwrap_or(0),
        option_b_diag: opt_in(value("ATLAS_DFLASH_OPTION_B_DIAG").as_deref()),

        force_pattern: opt_in(value("ATLAS_DFLASH_DEBUG_FORCE_PATTERN").as_deref()),
        force_noise_pattern: opt_in(value("ATLAS_DFLASH_DEBUG_FORCE_NOISE_PATTERN").as_deref()),
        force_no_ctx: opt_in(value("ATLAS_DFLASH_DEBUG_CTX_OFF").as_deref()),
        force_ctx_used: parsed(value("ATLAS_DFLASH_DEBUG_CTX_USED").as_deref()),

        precompute: opt_in(value("ATLAS_DFLASH_PRECOMPUTE").as_deref()),
        precompute_commit: opt_in(value("ATLAS_DFLASH_PRECOMPUTE_COMMIT").as_deref()),

        propose_warmup_n: parsed(value("ATLAS_DFLASH_PROPOSE_WARMUP_N").as_deref()).unwrap_or(2),

        dspark_shift: match value("ATLAS_DSPARK_SHIFT").as_deref() {
            Some("1") => Some(true),
            Some("0") => Some(false),
            _ => None,
        },
        dspark_anchor_bias: value("ATLAS_DSPARK_ANCHOR_BIAS").as_deref() != Some("0"),
        dspark_conf_trace: opt_in(value("ATLAS_DSPARK_CONF_TRACE").as_deref()),
    }
}

impl DFlashLevers {
    /// Resolve from the environment. Called ONCE, when the head is built.
    ///
    /// ★ Do not call this from `forward_block`, `propose`, or anything they
    /// reach. Take `self.levers` from the head instead — that is why the
    /// field exists, and `dflash_levers_are_resolved_once` fails the build if
    /// a raw `std::env::var` reappears on those paths.
    pub fn from_env() -> Self {
        from_values(
            |var| std::env::var(var).ok(),
            |var| std::env::var_os(var).is_some(),
        )
    }

    /// What a head resolves to with no `ATLAS_*` set: every diagnostic off,
    /// the anchor bias on, two warm-up passes. Tests construct this rather
    /// than mutating the process environment, which `set_var` makes unsafe
    /// and which would race every other test in the binary.
    pub fn defaults() -> Self {
        Self {
            // Opt-out: ships ON, `ATLAS_DSPARK_ANCHOR_BIAS=0` disables.
            dspark_anchor_bias: true,
            // Not a boolean default — the warm-up count is load-bearing for
            // graph capture, so it is spelled out rather than derived from
            // `usize::default()`.
            propose_warmup_n: 2,
            ..Self::default()
        }
    }

    /// The block dump is armed for this decode position.
    ///
    /// Three sites asked this question with two env reads each; it is one
    /// predicate over already-resolved data.
    pub fn block_dump_armed_at(&self, position: usize) -> bool {
        self.block_dump && position >= self.block_dump_at_pos
    }
}

#[cfg(test)]
#[path = "levers_tests.rs"]
mod tests;
