// SPDX-License-Identifier: AGPL-3.0-only

//! LRU eviction + entry counting for [`RadixTreeInner`]. Split out of
//! `inner.rs` during the ≤500-line cap enforcement; a child module of
//! `inner` so it keeps field access to the private node arena.

use super::RadixTreeInner;

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

            let mut best: Option<(usize, u64)> = None;
            for (id, node) in self.nodes.iter().enumerate() {
                // Roots (one per adapter_id) carry `parent == None` and
                // `block_idx == u32::MAX`; the block_idx guard below already
                // excludes them, but skip explicitly for clarity.
                if node.parent.is_none() {
                    continue;
                }
                if node.ref_count <= 1 && node.children.is_empty() && node.block_idx != u32::MAX {
                    match best {
                        None => best = Some((id, node.last_access)),
                        Some((_, best_access)) if node.last_access < best_access => {
                            best = Some((id, node.last_access));
                        }
                        _ => {}
                    }
                }
            }

            match best {
                Some((node_id, _)) => {
                    let block = self.nodes[node_id].block_idx;
                    let disk = self.nodes[node_id].disk_block_id;
                    freed_phys.push(block);
                    freed_disk.push(disk);

                    if let Some((_, partial_block, partial_disk)) =
                        self.nodes[node_id].partial_suffix.take()
                    {
                        freed_phys.push(partial_block);
                        freed_disk.push(partial_disk);
                    }

                    if let Some(parent_id) = self.nodes[node_id].parent
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
                }
                None => break,
            }
        }

        (freed_phys, freed_disk)
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
