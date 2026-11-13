// SPDX-License-Identifier: AGPL-3.0-only

//! LRU eviction + entry counting for [`RadixTreeInner`]. Split out of
//! `inner.rs` during the ≤500-line cap enforcement; a child module of
//! `inner` so it keeps field access to the private node arena.
//!
//! The victim comes from [`RadixTreeInner::pick_victim`] (`lru_index.rs`):
//! the full scan, or with `radix_lru_index` on the ordered index that picks
//! the same node. Removal is one code path for both, so the two modes leave
//! identical trees.

use super::{NodeId, RadixTreeInner};

impl RadixTreeInner {
    /// Evict up to `num_blocks` LRU zero-ref leaf nodes.
    /// Returns physical block indices that were freed plus parallel
    /// disk-block IDs (Phase 6.1.e). When HSS isn't in use, every disk_id
    /// in the result is `u32::MAX` and the caller should ignore them; the
    /// public-trait wrapper filters those out into the returned
    /// `EvictedBlocks::disk_block_ids`.
    pub(in crate::radix_tree) fn evict(&mut self, num_blocks: usize) -> (Vec<u32>, Vec<u32>) {
        let mut freed_phys = Vec::new();
        let mut freed_disk = Vec::new();
        if num_blocks == 0 {
            return (freed_phys, freed_disk);
        }

        loop {
            if freed_phys.len() >= num_blocks {
                break;
            }
            match self.pick_victim() {
                Some(node_id) => self.remove_victim(node_id, &mut freed_phys, &mut freed_disk),
                None => break,
            }
        }

        (freed_phys, freed_disk)
    }

    /// Unlink one victim, hand back its block (and its partial-suffix block,
    /// if any), and refresh the index for the victim and its parent, which
    /// may have just become a leaf.
    fn remove_victim(
        &mut self,
        node_id: NodeId,
        freed_phys: &mut Vec<u32>,
        freed_disk: &mut Vec<u32>,
    ) {
        let block = self.nodes[node_id].block_idx;
        let disk = self.nodes[node_id].disk_block_id;
        freed_phys.push(block);
        freed_disk.push(disk);

        if let Some((_, partial_block, partial_disk)) = self.nodes[node_id].partial_suffix.take() {
            freed_phys.push(partial_block);
            freed_disk.push(partial_disk);
        }

        let parent = self.nodes[node_id].parent;
        if let Some(parent_id) = parent
            && let Some(key) = self.nodes[node_id].parent_key.clone()
        {
            self.nodes[parent_id].children.remove(&key);
        }

        self.nodes[node_id].block_idx = u32::MAX;
        self.nodes[node_id].disk_block_id = u32::MAX;
        self.nodes[node_id].children.clear();
        self.nodes[node_id].parent = None;
        self.nodes[node_id].parent_key = None;
        self.nodes[node_id].partial_suffix = None;
        self.free_nodes.push(node_id);
        self.evicted_nodes_total += 1;

        self.lru_refresh(node_id);
        if let Some(parent_id) = parent {
            self.lru_refresh(parent_id);
        }
    }

    pub(in crate::radix_tree) fn num_entries(&self) -> usize {
        // Count non-root, non-deleted nodes. Roots (per adapter_id) and freed
        // nodes both carry `block_idx == u32::MAX`, so this filter excludes them.
        self.nodes
            .iter()
            .filter(|n| n.block_idx != u32::MAX)
            .count()
    }
}
