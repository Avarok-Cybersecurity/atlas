// SPDX-License-Identifier: AGPL-3.0-only
// Partial-hit scheme translated from vLLM (https://github.com/vllm-project/vllm, commit 8a36f08),
// vllm/v1/core/single_type_kv_cache_manager.py (`_has_partial_local_hit`, `allocate_new_blocks`,
// `_apply_cow`), Copyright contributors to the vLLM project, Apache-2.0. Changes: re-expressed
// over Atlas's radix tree (the sub-block hit is found by the two existing sub-block arms, a
// child-key prefix or the node's partial suffix, instead of a hash unit smaller than a block);
// the walk only names the source block and the matched rows, the caller owns the copy; by Atlas
// contributors. See VLLM-COW-ATTRIBUTION.md.

//! Responsibility: the block-aligned walk prefix that both walks share, the
//! copy-on-write sub-block walk (row `prefix_cow_tail`, issue #1193), and the
//! touch that marks a copy source as used.
//!
//! [`RadixTreeInner::walk`]'s two legacy sub-block arms hand the requester a
//! block that another owner still writes. [`RadixTreeInner::walk_cow`] finds
//! the SAME sub-block hits but returns the source block separately, outside
//! `matched_blocks`: the caller allocates a private block, copies the source
//! into it, and appends the copy (`prefill_b/prefix_cow.rs`). A sub-block
//! match therefore never puts a shared block at the requester's writable
//! tail. The radix tree takes no reference on the source; the caller holds a
//! KV reference on it for the duration of the copy enqueue.
//!
//! Rows `[0, rows)` of the source hold K/V for exactly the requester's tokens
//! at exactly the requester's positions: the parent chain matched by context
//! hash, and the source's tokens start with the requester's remainder. Rows
//! past `rows` are copied too (whole-block copy) and are overwritten by the
//! requester before it reads them, because attention reads only positions it
//! has already written (`matched..`) or inherited (`..matched`).

use super::{NodeId, RadixTreeInner, context_hash_combine};

/// The block-aligned prefix of a walk ([`RadixTreeInner::walk_full`]).
pub(in crate::radix_tree) struct FullWalk {
    /// Physical blocks of the matched full chunks, in order.
    pub(in crate::radix_tree) blocks: Vec<u32>,
    /// Disk ids parallel to `blocks` (`u32::MAX` without HSS).
    pub(in crate::radix_tree) disk: Vec<u32>,
    /// Matched tokens (a multiple of the block size).
    pub(in crate::radix_tree) matched: usize,
    /// The deepest matched node (the adapter's root when nothing matched).
    pub(in crate::radix_tree) node: NodeId,
    /// That node's context hash (0 for a root).
    pub(in crate::radix_tree) ctx_hash: u64,
}

/// The source of a copy-on-write tail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::radix_tree) struct CowSource {
    /// Physical block to copy from.
    pub(in crate::radix_tree) block: u32,
    /// Leading rows of that block the requester matched (`1..block_size`).
    pub(in crate::radix_tree) rows: usize,
    /// The radix node that stores `block` (child-key arm); `None` for a
    /// partial-suffix source, which is metadata on the deepest matched node.
    pub(in crate::radix_tree) node: Option<NodeId>,
}

impl RadixTreeInner {
    /// The block-aligned part of every walk: follow full chunks while the
    /// context chain matches and the node is live. `None` when `adapter_id`
    /// has no root yet. Shared by `walk` and `walk_cow` so the two cannot
    /// disagree on the aligned match (this is the loop `walk` always ran).
    pub(in crate::radix_tree) fn walk_full(
        &self,
        tokens: &[u32],
        block_size: usize,
        adapter_id: u64,
    ) -> Option<FullWalk> {
        let mut current = self.root_for_read(adapter_id)?;
        let mut blocks = Vec::new();
        let mut disk = Vec::new();
        let mut matched = 0;
        let mut parent_ctx_hash: u64 = 0; // root context hash

        let num_full_blocks = tokens.len() / block_size;
        for i in 0..num_full_blocks {
            let chunk = &tokens[i * block_size..(i + 1) * block_size];
            let expected_hash = context_hash_combine(parent_ctx_hash, chunk);
            match self.nodes[current].children.get(chunk) {
                Some(&child)
                    if self.nodes[child].context_hash == expected_hash
                        && self.nodes[child].ref_count > 0 =>
                {
                    // Context chain matches AND block is still live: safe to reuse.
                    blocks.push(self.nodes[child].block_idx);
                    disk.push(self.nodes[child].disk_block_id);
                    matched += block_size;
                    parent_ctx_hash = expected_hash;
                    current = child;
                }
                _ => break, // Token match but context mismatch or freed block: stop.
            }
        }
        Some(FullWalk {
            blocks,
            disk,
            matched,
            node: current,
            ctx_hash: parent_ctx_hash,
        })
    }

    /// [`Self::walk`] with copy-on-write tails: `(matched_blocks,
    /// matched_disk, matched_tokens, source)`. With a source,
    /// `matched_tokens == matched_blocks.len() * block_size + source.rows`;
    /// without one the result equals `walk` with the legacy arms off.
    ///
    /// A source qualifies only where the legacy arms would have matched: the
    /// prompt's remainder after its last full block (`0 < remainder <
    /// block_size`, every full block matched) is a prefix of a live child's
    /// key whose context hash continues the chain, or of the matched node's
    /// partial suffix. Among several children the lowest block index wins, so
    /// the pick is deterministic (the legacy arm takes the first in hash-map
    /// order; every candidate holds the same leading rows). A source or
    /// matched block with a `--high-speed-swap` disk id is refused: a private
    /// copy would have no disk image.
    pub(in crate::radix_tree) fn walk_cow(
        &self,
        tokens: &[u32],
        block_size: usize,
        adapter_id: u64,
    ) -> (Vec<u32>, Vec<u32>, usize, Option<CowSource>) {
        let Some(full) = self.walk_full(tokens, block_size, adapter_id) else {
            return (Vec::new(), Vec::new(), 0, None);
        };
        let num_full_blocks = tokens.len() / block_size;
        let remainder = tokens.len() - full.matched;
        let eligible = remainder > 0
            && remainder < block_size
            && full.matched == num_full_blocks * block_size
            && full.disk.iter().all(|&d| d == u32::MAX);
        let source = if eligible {
            self.cow_source(full.node, full.ctx_hash, &tokens[full.matched..])
        } else {
            None
        };
        let matched = full.matched + source.map_or(0, |s| s.rows);
        (full.blocks, full.disk, matched, source)
    }

    /// The copy source for `suffix` under node `at` (context hash `ctx`).
    fn cow_source(&self, at: NodeId, ctx: u64, suffix: &[u32]) -> Option<CowSource> {
        let mut best: Option<(u32, NodeId)> = None;
        for (key, &child) in &self.nodes[at].children {
            if key.len() < suffix.len() || key[..suffix.len()] != *suffix {
                continue;
            }
            let n = &self.nodes[child];
            if n.context_hash != context_hash_combine(ctx, key)
                || n.ref_count == 0
                || n.block_idx == u32::MAX
                || n.disk_block_id != u32::MAX
            {
                continue;
            }
            if best.is_none_or(|(b, _)| n.block_idx < b) {
                best = Some((n.block_idx, child));
            }
        }
        if let Some((block, node)) = best {
            return Some(CowSource {
                block,
                rows: suffix.len(),
                node: Some(node),
            });
        }
        // Partial suffix of the deepest matched node. It has no context hash
        // of its own: it is reachable only through the matched parent chain.
        // `insert` never stores one on a root.
        match &self.nodes[at].partial_suffix {
            Some((toks, block, disk))
                if self.nodes[at].parent.is_some()
                    && toks.len() >= suffix.len()
                    && toks[..suffix.len()] == *suffix
                    && *block != u32::MAX
                    && *disk == u32::MAX =>
            {
                Some(CowSource {
                    block: *block,
                    rows: suffix.len(),
                    node: None,
                })
            }
            _ => None,
        }
    }

    /// Mark a copy source as used by the lookup that just ran (the tick
    /// `inc_refs` took), so the allocation that follows does not pick the
    /// block being copied as its eviction victim. Only the copy-on-write
    /// lookup calls this.
    pub(in crate::radix_tree) fn touch(&mut self, id: NodeId) {
        self.nodes[id].last_access = self.access_counter;
        self.lru_refresh(id);
    }
}
