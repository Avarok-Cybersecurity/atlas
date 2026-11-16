// SPDX-License-Identifier: AGPL-3.0-only
// Design after the TierKV pager of arczhi/llama-tierkv (kv-paging-design.md; MIT, Copyright (c) 2023-2026 The ggml authors): head protection of the first 2,048 positions, a recent window decoupled from the logical context, and staging rate-limited to once per 1,024 positions and only after evictions, taken from a written description of the design; no code copied or fetched.

//! Responsibility: the window desk's pure policy. For one long sequence it decides which
//! logical KV blocks stay on the GPU (the head sinks, the recent window, the recalled pages),
//! which full blocks go to host RAM and when, and which pages a recall round fetches and
//! retires. No GPU, no locks, no globals: every function is a function of the geometry and the
//! per-block [`Slot`]s, so the identity and rate-limit properties are CPU unit tests
//! (`tests.rs`).
//!
//! Units: a page is one Atlas KV block (`block_size` tokens, 16 on Qwen3). Logical block `i`
//! covers positions `[i * bs, (i + 1) * bs)`; demoting it never changes any position.

use std::collections::HashSet;
use std::ops::Range;

/// Tokens between two demotion batches and between two recall rounds (TierKV stages pages at
/// most once per 1,024 positions, and only after evictions).
pub const STEP_TOKENS: usize = 1024;

/// At most this many blocks are demoted by one call (bounds one step's host work; the rest go
/// on the following steps).
pub const MAX_DEMOTE_BLOCKS: usize = 4096;

/// Where one logical block of a sequence lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    /// On the GPU in its own block (the sequence's original block).
    Resident,
    /// On the host only; the table holds the desk's hole block at this index.
    Hidden,
    /// On the host; a copy into a fresh GPU block is running (the table still holds the hole).
    Fetching,
    /// On the host and back on the GPU in a fresh block the table holds.
    Recalled,
}

impl Slot {
    /// The attention reads this block.
    pub fn on_gpu(self) -> bool {
        matches!(self, Self::Resident | Self::Recalled)
    }

    /// The block was demoted at least once (it has a host copy).
    pub fn demoted(self) -> bool {
        !matches!(self, Self::Resident)
    }
}

/// The desk's shape for one model: block size and the three token budgets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub block_size: usize,
    /// Head protection: the first `head_tokens` positions never leave the GPU (system prompt,
    /// task statement; attention sinks).
    pub head_tokens: usize,
    /// The recent window: the last `window_tokens` positions never leave the GPU.
    pub window_tokens: usize,
    /// Recalled pages a sequence may hold on the GPU besides the sinks and the window.
    pub recall_tokens: usize,
    /// Demotion batch and recall interval, in positions ([`STEP_TOKENS`]).
    pub step_tokens: usize,
}

impl Geometry {
    pub fn new(
        block_size: usize,
        head_tokens: usize,
        window_tokens: usize,
        recall_tokens: usize,
    ) -> Self {
        Self {
            block_size: block_size.max(1),
            head_tokens,
            window_tokens,
            recall_tokens,
            step_tokens: STEP_TOKENS,
        }
    }

    pub fn head_blocks(&self) -> usize {
        self.head_tokens.div_ceil(self.block_size)
    }

    /// Blocks one demotion batch waits for.
    pub fn batch_blocks(&self) -> usize {
        (self.step_tokens / self.block_size).max(1)
    }

    /// Recalled pages a sequence may hold.
    pub fn recall_blocks(&self) -> usize {
        self.recall_tokens / self.block_size
    }

    /// First logical block of the recent window of a KV of `kv_tokens` positions: the block
    /// holding position `kv_tokens - window_tokens` and every later block stay resident.
    pub fn window_start_block(&self, kv_tokens: usize) -> usize {
        kv_tokens.saturating_sub(self.window_tokens) / self.block_size
    }

    /// Blocks that may leave the GPU: full blocks after the head and before the window.
    pub fn demotable(&self, kv_tokens: usize) -> Range<usize> {
        let full = kv_tokens / self.block_size;
        let end = self.window_start_block(kv_tokens).min(full);
        let start = self.head_blocks().min(end);
        start..end
    }

    /// The desk changes nothing for this length (sinks and window cover every block).
    pub fn is_identity(&self, kv_tokens: usize) -> bool {
        self.demotable(kv_tokens).is_empty()
    }
}

/// The resident blocks to demote now: every `Resident` block of the demotable range, but only
/// once at least one batch of them waits (so demotion happens at most once per
/// [`STEP_TOKENS`] positions), and at most [`MAX_DEMOTE_BLOCKS`].
pub fn demote_batch(g: &Geometry, kv_tokens: usize, slots: &[Slot]) -> Vec<u32> {
    let mut cand: Vec<u32> = g
        .demotable(kv_tokens)
        .filter(|&i| slots.get(i) == Some(&Slot::Resident))
        .map(|i| i as u32)
        .collect();
    if cand.len() < g.batch_blocks() {
        return Vec::new();
    }
    cand.truncate(MAX_DEMOTE_BLOCKS);
    cand
}

/// A recall round is due: the sequence has demoted pages, a recall budget exists, and the last
/// round was at least [`STEP_TOKENS`] positions ago.
pub fn recall_due(
    g: &Geometry,
    kv_tokens: usize,
    last_round: Option<usize>,
    slots: &[Slot],
) -> bool {
    if g.recall_blocks() == 0 || !slots.iter().any(|s| s.demoted()) {
        return false;
    }
    match last_round {
        None => true,
        Some(p) => kv_tokens >= p + g.step_tokens,
    }
}

/// What one recall round changes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RecallPlan {
    /// Hidden pages to copy back (ascending).
    pub fetch: Vec<u32>,
    /// Recalled pages to put back on the shelf (ascending; their host copy is still there).
    pub retire: Vec<u32>,
}

/// Turn a ranking (best first, logical block indices) into a delta: the target set is the
/// first [`Geometry::recall_blocks`] ranked pages that were demoted; pages already recalled or
/// on their way stay, recalled pages outside the target retire, and hidden target pages are
/// fetched as far as the budget allows (a copy still running outside the target counts against
/// it until it lands). Indices outside `slots` or never demoted are ignored.
pub fn plan_recall(g: &Geometry, slots: &[Slot], ranked: &[u32]) -> RecallPlan {
    let budget = g.recall_blocks();
    let mut target: Vec<u32> = Vec::new();
    let mut seen = HashSet::new();
    for &p in ranked {
        if target.len() >= budget {
            break;
        }
        if slots.get(p as usize).is_some_and(|s| s.demoted()) && seen.insert(p) {
            target.push(p);
        }
    }
    let in_target = |i: usize| seen.contains(&(i as u32));
    let mut retire = Vec::new();
    let mut kept = 0usize;
    let mut stray = 0usize;
    for (i, s) in slots.iter().enumerate() {
        match s {
            Slot::Recalled if in_target(i) => kept += 1,
            Slot::Recalled => retire.push(i as u32),
            Slot::Fetching if in_target(i) => kept += 1,
            Slot::Fetching => stray += 1,
            _ => {}
        }
    }
    let room = budget.saturating_sub(kept + stray);
    let mut fetch: Vec<u32> = target
        .iter()
        .copied()
        .filter(|&p| slots[p as usize] == Slot::Hidden)
        .take(room)
        .collect();
    fetch.sort_unstable();
    RecallPlan { fetch, retire }
}

/// Positions the attention reads for a KV of `kv_tokens` (every block not on the GPU is a full
/// block, so each one removes exactly `block_size` positions).
pub fn resident_tokens(g: &Geometry, kv_tokens: usize, slots: &[Slot]) -> usize {
    let off = slots.iter().filter(|s| !s.on_gpu()).count();
    kv_tokens.saturating_sub(off * g.block_size)
}
