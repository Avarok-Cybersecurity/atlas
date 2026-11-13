// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the MTP head's bounded attention window
//! (`MtpHead::set_kv_window`), the API L36's windowed MTP KV consumes.
//!
//! With a window W the drafter's single attention layer reads the sink rows
//! (the first [`SINK_ROWS`] rows of its KV, one block) plus the blocks that
//! hold the last W rows, instead of every row. The view is built over the
//! drafter's own block table, so every key keeps the RoPE position it was
//! written with and the attention is exact over the selected rows; only the
//! set of rows shrinks. Blocks are the unit, so the window is W rounded up to
//! whole blocks. With no window (the default) the drafter reads its whole
//! table, byte-identical to the pre-sprint path.
//!
//! A head-level setting wins over the process default
//! ([`set_default_kv_window`]). The default exists because the drafter sits
//! behind `Arc<dyn DraftProposer>`, which offers no downcast: a consumer that
//! cannot reach the `MtpHead` (L36 from `port_apply`) sets the default before
//! the model is built, or at any later time, and every head follows it.

use std::borrow::Cow;
use std::sync::atomic::{AtomicU32, Ordering};

use super::MtpHead;

/// Rows kept at the start of the drafter KV under a window (attention
/// sinks): one block of the drafter pool (`block_size: 16` in `new.rs`).
pub const SINK_ROWS: usize = 16;

/// Process default window in drafter KV rows; 0 = none.
static DEFAULT_WINDOW: AtomicU32 = AtomicU32::new(0);

/// Set the window every MTP head without its own setting follows
/// (`None` or `Some(0)` = no bound).
pub fn set_default_kv_window(window: Option<u32>) {
    DEFAULT_WINDOW.store(window.unwrap_or(0), Ordering::Relaxed);
}

/// The process default window, `None` when unbounded.
pub fn default_kv_window() -> Option<u32> {
    match DEFAULT_WINDOW.load(Ordering::Relaxed) {
        0 => None,
        w => Some(w),
    }
}

impl MtpHead {
    /// Bound this head's attention to the sink rows plus the last `window`
    /// drafter KV rows (`None` = the whole KV). Takes effect at the next
    /// draft; overrides [`set_default_kv_window`] for this head.
    pub fn set_kv_window(&self, window: Option<u32>) {
        self.v2.set_kv_window(window);
    }

    /// The window this head attends with, `None` when unbounded.
    pub fn kv_window(&self) -> Option<u32> {
        self.v2.kv_window()
    }
}

/// The block table and row count the paged attention reads for a drafter KV
/// of `len` rows over `table` (blocks of `block_size` rows) under a window of
/// `window` rows and `sinks` sink rows.
///
/// Returns the table unchanged (borrowed) and `len` when the sinks and the
/// window already cover every block. Otherwise the view is the first
/// `ceil(sinks / block_size)` blocks followed by the last blocks that hold at
/// least `window` rows, in table order, and the row count is that of the
/// view: every block but the last is full, the last keeps its real fill, so
/// the kernel reads exactly the selected rows.
pub fn view(
    table: &[u32],
    len: usize,
    block_size: usize,
    window: usize,
    sinks: usize,
) -> (Cow<'_, [u32]>, usize) {
    if len == 0 || block_size == 0 || window == 0 {
        return (Cow::Borrowed(table), len);
    }
    let nblocks = len.div_ceil(block_size).min(table.len());
    if nblocks == 0 {
        return (Cow::Borrowed(table), len);
    }
    let tail = len - (nblocks - 1) * block_size;
    let sink_blocks = sinks.div_ceil(block_size);
    let window_blocks = 1 + window.saturating_sub(tail).div_ceil(block_size);
    if sink_blocks + window_blocks >= nblocks {
        return (Cow::Borrowed(table), len);
    }
    let mut v = Vec::with_capacity(sink_blocks + window_blocks);
    v.extend_from_slice(&table[..sink_blocks]);
    v.extend_from_slice(&table[nblocks - window_blocks..nblocks]);
    let rows = (sink_blocks + window_blocks - 1) * block_size + tail;
    (Cow::Owned(v), rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BS: usize = 16;

    #[test]
    fn no_window_or_a_covering_window_is_the_identity() {
        let table: Vec<u32> = (100..110).collect();
        for (len, window) in [(150, 0), (150, 150), (150, 10_000), (20, 4), (0, 8)] {
            let (v, n) = view(&table, len, BS, window, SINK_ROWS);
            assert!(matches!(v, Cow::Borrowed(_)), "len={len} window={window}");
            assert_eq!(&*v, &table[..]);
            assert_eq!(n, len);
        }
    }

    #[test]
    fn the_view_keeps_the_sinks_and_the_newest_blocks() {
        // 150 rows = 9 full blocks + 6 rows in block 9 (table ids 100..=109).
        let table: Vec<u32> = (100..110).collect();
        let (v, n) = view(&table, 150, BS, 40, SINK_ROWS);
        // Last rows needed: 40 -> tail 6 + 34 more = 3 more blocks (6+48=54).
        assert_eq!(&*v, &[100, 106, 107, 108, 109]);
        assert_eq!(n, 4 * BS + 6);
    }

    #[test]
    fn the_window_always_covers_at_least_the_requested_rows() {
        let table: Vec<u32> = (0..64).collect();
        for len in 1..=1000usize {
            for window in [1usize, 15, 16, 17, 100, 333] {
                let (v, n) = view(&table, len, BS, window, SINK_ROWS);
                match v {
                    Cow::Borrowed(t) => {
                        assert_eq!(t.len(), table.len());
                        assert_eq!(n, len, "len={len} window={window}");
                    }
                    Cow::Owned(ref blocks) => {
                        // Rows read beyond the sink block cover the window.
                        assert!(n - BS >= window, "len={len} window={window}");
                        assert!(n < len);
                        assert_eq!(blocks[0], table[0], "the sink block stays");
                        // The last block is always the real last block.
                        assert_eq!(*blocks.last().unwrap(), table[len.div_ceil(BS) - 1]);
                    }
                }
            }
        }
    }

    #[test]
    fn extra_allocated_blocks_past_the_rows_are_never_read() {
        // After a trim the drafter keeps blocks it no longer fills.
        let table: Vec<u32> = (0..20).collect();
        let (v, n) = view(&table, 100, BS, 20, SINK_ROWS);
        assert_eq!(&*v, &[0, 5, 6]); // 100 rows: blocks 0..=6, tail 4
        assert_eq!(n, 2 * BS + 4);
    }

    #[test]
    fn the_process_default_round_trips() {
        assert_eq!(default_kv_window(), None);
        set_default_kv_window(Some(4096));
        assert_eq!(default_kv_window(), Some(4096));
        set_default_kv_window(Some(0));
        assert_eq!(default_kv_window(), None);
        set_default_kv_window(None);
        assert_eq!(default_kv_window(), None);
    }
}
