// SPDX-License-Identifier: AGPL-3.0-only

//! Single-threaded radix-tree state — wrapped behind a `Mutex` in the
//! [`super::RadixTree`] public API. Token sequences are chunked at
//! `block_size` granularity; each node holds one physical KV block index.

use std::collections::HashMap;

type NodeId = usize;

pub(super) struct RadixNode {
    /// Children keyed by the token chunk (block_size tokens).
    children: HashMap<Vec<u32>, NodeId>,
    /// Physical KV cache block index stored at this node.
    block_idx: u32,
    /// `--high-speed-swap` disk-block ID (Phase 6.1.e). `u32::MAX` when HSS
    /// is not in use. The cache holds a refcount on this disk_id (bumped
    /// at insert time, dropped at evict time) parallel to its physical-
    /// block ref bookkeeping.
    disk_block_id: u32,
    /// Context hash: FNV-1a chain of this node's tokens + parent's context_hash.
    /// Ensures a block is only reused when the ENTIRE causal prefix matches,
    /// preventing cross-request KV contamination (vLLM-style parent-hash chain).
    context_hash: u64,
    /// Number of active sequences referencing this node.
    ref_count: u32,
    /// Monotonic access counter for LRU eviction.
    last_access: u64,
    /// Parent node (None for root).
    parent: Option<NodeId>,
    /// Token chunk that leads from parent to this node (for cleanup).
    parent_key: Option<Vec<u32>>,
    /// Partial (sub-block) suffix: the last incomplete block in a cached
    /// sequence. Stores `(partial_tokens, block_idx, disk_block_id)`
    /// where `partial_tokens.len() < block_size`. `disk_block_id == u32::MAX`
    /// when HSS isn't in use. Matched after all full blocks to cover the
    /// trailing remainder.
    partial_suffix: Option<(Vec<u32>, u32, u32)>,
}

/// Combine a parent context hash with a token chunk to produce a child context hash.
/// Uses FNV-1a for speed (~ns per block). The chain guarantees that two nodes with
/// identical token chunks but different prefixes get different context_hashes.
fn context_hash_combine(parent_hash: u64, tokens: &[u32]) -> u64 {
    let mut h = parent_hash;
    for &t in tokens {
        h ^= t as u64;
        h = h.wrapping_mul(0x100000001b3); // FNV-1a prime
    }
    h
}

/// Inner state protected by mutex.
pub(super) struct RadixTreeInner {
    nodes: Vec<RadixNode>,
    /// Indices of deleted nodes available for reuse.
    free_nodes: Vec<NodeId>,
    root: NodeId,
    access_counter: u64,
}

impl RadixTreeInner {
    pub(super) fn new() -> Self {
        let root = RadixNode {
            children: HashMap::new(),
            block_idx: u32::MAX,     // sentinel — root has no block
            disk_block_id: u32::MAX, // sentinel — root has no disk slot either
            context_hash: 0,         // root context hash is 0
            ref_count: 0,
            last_access: 0,
            parent: None,
            parent_key: None,
            partial_suffix: None,
        };
        Self {
            nodes: vec![root],
            free_nodes: Vec::new(),
            root: 0,
            access_counter: 0,
        }
    }

    pub(super) fn next_access(&mut self) -> u64 {
        self.access_counter += 1;
        self.access_counter
    }

    pub(super) fn alloc_node(&mut self, node: RadixNode) -> NodeId {
        if let Some(id) = self.free_nodes.pop() {
            self.nodes[id] = node;
            id
        } else {
            let id = self.nodes.len();
            self.nodes.push(node);
            id
        }
    }

    /// Walk the tree matching block-aligned token chunks.
    /// Returns (matched_blocks, matched_disk_block_ids, matched_tokens). The
    /// disk-id vec parallels matched_blocks; entries are `u32::MAX` on nodes
    /// that were inserted before HSS engaged (caller's responsibility to
    /// filter / treat MAX as "no disk-side ref to bump"). Snapshot lookup is
    /// now handled by the separate SsmSnapshotIndex after the walk completes.
    pub(super) fn walk(&self, tokens: &[u32], block_size: usize) -> (Vec<u32>, Vec<u32>, usize) {
        let mut current = self.root;
        let mut matched_blocks = Vec::new();
        let mut matched_disk = Vec::new();
        let mut matched_tokens = 0;
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
                    // Context chain matches AND block is still live — safe to reuse.
                    matched_blocks.push(self.nodes[child].block_idx);
                    matched_disk.push(self.nodes[child].disk_block_id);
                    matched_tokens += block_size;
                    parent_ctx_hash = expected_hash;
                    current = child;
                }
                _ => break, // Token match but context mismatch or freed block — stop.
            }
        }

        // Sub-block matching: when remaining tokens don't fill a full block,
        // check two locations for a match:
        // 1. A child node whose key starts with the remaining tokens (the child
        //    covers a full block but we only need a prefix of it).
        // 2. A partial suffix stored on the current node.
        // This enables warm-cache TTFT optimization by matching ALL prompt tokens
        // even when total % block_size != 0.
        let remainder = tokens.len() - matched_tokens;
        if remainder > 0 && remainder < block_size && matched_tokens == num_full_blocks * block_size
        {
            let suffix = &tokens[matched_tokens..];
            let mut found = false;
            for (key, &child_id) in &self.nodes[current].children {
                if key.len() >= suffix.len() && &key[..suffix.len()] == suffix {
                    let expected = context_hash_combine(parent_ctx_hash, key);
                    if self.nodes[child_id].context_hash == expected
                        && self.nodes[child_id].ref_count > 0
                    {
                        matched_blocks.push(self.nodes[child_id].block_idx);
                        matched_disk.push(self.nodes[child_id].disk_block_id);
                        matched_tokens += remainder;
                        found = true;
                        break;
                    }
                }
            }
            if !found
                && let Some((ref partial_toks, partial_block, partial_disk)) =
                    self.nodes[current].partial_suffix
                && partial_toks.len() >= suffix.len()
                && &partial_toks[..suffix.len()] == suffix
            {
                // Partial suffix doesn't have context_hash — only match if parent chain matched.
                // ref_count check not applicable to partial suffix (it's metadata, not a node).
                matched_blocks.push(partial_block);
                matched_disk.push(partial_disk);
                matched_tokens += remainder;
            }
        }

        (matched_blocks, matched_disk, matched_tokens)
    }

    /// Increment ref_count on all nodes along the matched path.
    pub(super) fn inc_refs(&mut self, tokens: &[u32], block_size: usize, num_matched: usize) {
        let access = self.next_access();
        let mut current = self.root;
        let num_blocks = num_matched / block_size;

        for i in 0..num_blocks {
            let chunk = &tokens[i * block_size..(i + 1) * block_size];
            if let Some(&child) = self.nodes[current].children.get(chunk) {
                self.nodes[child].ref_count += 1;
                self.nodes[child].last_access = access;
                current = child;
            } else {
                break;
            }
        }
    }

    /// Decrement ref_count on all nodes along the matched path.
    pub(super) fn dec_refs(&mut self, tokens: &[u32], block_size: usize) {
        let mut current = self.root;
        let num_full_blocks = tokens.len() / block_size;

        for i in 0..num_full_blocks {
            let chunk = &tokens[i * block_size..(i + 1) * block_size];
            if let Some(&child) = self.nodes[current].children.get(chunk) {
                if self.nodes[child].ref_count > 0 {
                    self.nodes[child].ref_count -= 1;
                }
                current = child;
            } else {
                break;
            }
        }
    }

    /// Insert blocks into the tree. Skips blocks that already exist.
    /// Snapshot storage is handled externally by SsmSnapshotIndex.
    ///
    /// `matched_tokens` is the prefix length the inserting sequence already
    /// acquired via `lookup()`'s `inc_refs` (those nodes' ref_count was
    /// bumped by walk and will be balanced by the eventual `release`). For
    /// blocks past that offset the inserting sequence holds no walk-ref but
    /// `release` will still dec_ref them — so we pre-bump here to keep the
    /// cache's baseline ref (=1) alive across the release. Without this the
    /// very first sequence to cache a prompt immediately renders its own
    /// cache entry un-findable on exit (`walk` requires `ref_count > 0`).
    pub(super) fn insert(
        &mut self,
        tokens: &[u32],
        block_table: &[u32],
        disk_block_ids: &[u32],
        block_size: usize,
        matched_tokens: usize,
    ) -> Vec<u32> {
        let access = self.next_access();
        let mut current = self.root;
        let mut parent_ctx_hash: u64 = 0; // root context hash
        let num_full_blocks = tokens.len() / block_size;
        let num_blocks = num_full_blocks.min(block_table.len());
        // disk_block_ids may be empty (HSS off) or parallel to block_table.
        // When empty, every node's disk_block_id stays at u32::MAX sentinel.
        let hss_active = !disk_block_ids.is_empty();
        debug_assert!(
            !hss_active || disk_block_ids.len() == block_table.len(),
            "disk_block_ids length mismatch: {} vs {}",
            disk_block_ids.len(),
            block_table.len(),
        );

        // Issue #17: collect disk_ids on which the cache newly takes an
        // ownership ref. Caller `inc_disk_ref`s these so the swap allocator's
        // refcount tracks the cache's reachability. Two acquisition events:
        //   1. New node creation with a real disk_id.
        //   2. Existing node, MAX → real disk_id (first-time HSS population).
        // Re-insertion of an already-cached (node, disk_id) pair is NOT an
        // acquisition — the cache's ref already covers it.
        let mut newly_acquired: Vec<u32> = Vec::new();

        for i in 0..num_blocks {
            let chunk = &tokens[i * block_size..(i + 1) * block_size];
            let ctx_hash = context_hash_combine(parent_ctx_hash, chunk);
            let token_start = i * block_size;
            let is_seq_owned = token_start >= matched_tokens;
            let disk_id = if hss_active {
                disk_block_ids[i]
            } else {
                u32::MAX
            };

            if let Some(&child) = self.nodes[current].children.get(chunk) {
                // Node exists — update access time, context_hash, and ensure
                // the cache's ref is still held (ref_count >= 1).
                self.nodes[child].last_access = access;
                self.nodes[child].context_hash = ctx_hash;
                if self.nodes[child].ref_count == 0 {
                    self.nodes[child].ref_count = 1; // restore cache's ref
                }
                if is_seq_owned {
                    self.nodes[child].ref_count += 1;
                }
                // First-time HSS population on a pre-existing node: stash
                // the disk_id. (This happens when the cache was built
                // pre-HSS and a later HSS-enabled sequence touches the
                // same prefix — defensive, shouldn't fire under normal
                // operation since cache_blocks_per_seq is set at startup.)
                if hss_active && self.nodes[child].disk_block_id == u32::MAX && disk_id != u32::MAX
                {
                    self.nodes[child].disk_block_id = disk_id;
                    newly_acquired.push(disk_id);
                }
                parent_ctx_hash = ctx_hash;
                current = child;
            } else {
                self.nodes[current].partial_suffix = None;
                let node = RadixNode {
                    children: HashMap::new(),
                    block_idx: block_table[i],
                    disk_block_id: disk_id,
                    context_hash: ctx_hash,
                    ref_count: if is_seq_owned { 2 } else { 1 },
                    last_access: access,
                    parent: Some(current),
                    parent_key: Some(chunk.to_vec()),
                    partial_suffix: None,
                };
                let child_id = self.alloc_node(node);
                self.nodes[current]
                    .children
                    .insert(chunk.to_vec(), child_id);
                if hss_active && disk_id != u32::MAX {
                    newly_acquired.push(disk_id);
                }
                parent_ctx_hash = ctx_hash;
                current = child_id;
            }
        }

        let remainder = tokens.len() % block_size;
        if remainder > 0 && block_table.len() > num_full_blocks && current != self.root {
            let partial_toks = tokens[num_full_blocks * block_size..].to_vec();
            let partial_block = block_table[num_full_blocks];
            let partial_disk = if hss_active && disk_block_ids.len() > num_full_blocks {
                disk_block_ids[num_full_blocks]
            } else {
                u32::MAX
            };
            // partial_suffix overwrites any prior partial slot. If the prior
            // slot held a real disk_id distinct from the new one, the cache
            // is silently dropping a ref to that old id; surface it on the
            // returned acquisitions only when we actually accept a NEW real
            // disk_id (matching the same "new acquisition" definition as the
            // full-block path above). This branch fires rarely — partial
            // slots are tail-only and typically unique per-prefix.
            let prior = self.nodes[current].partial_suffix.as_ref().map(|p| p.2);
            self.nodes[current].partial_suffix = Some((partial_toks, partial_block, partial_disk));
            if hss_active && partial_disk != u32::MAX && prior != Some(partial_disk) {
                newly_acquired.push(partial_disk);
            }
        }

        newly_acquired
    }

    /// Evict up to `num_blocks` LRU zero-ref leaf nodes.
    /// Returns physical block indices that were freed plus parallel
    /// disk-block IDs (Phase 6.1.e). When HSS isn't in use, every disk_id
    /// in the result is `u32::MAX` and the caller should ignore them; the
    /// public-trait wrapper filters those out into the returned
    /// `EvictedBlocks::disk_block_ids`.
    pub(super) fn evict(&mut self, num_blocks: usize) -> (Vec<u32>, Vec<u32>) {
        let mut freed_phys = Vec::new();
        let mut freed_disk = Vec::new();
        if num_blocks == 0 {
            return (freed_phys, freed_disk);
        }

        loop {
            if freed_phys.len() >= num_blocks {
                break;
            }

            let mut best: Option<(NodeId, u64)> = None;
            for (id, node) in self.nodes.iter().enumerate() {
                if id == self.root {
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

    pub(super) fn num_entries(&self) -> usize {
        // Count non-root, non-deleted nodes.
        self.nodes
            .iter()
            .enumerate()
            .filter(|(id, n)| *id != self.root && n.block_idx != u32::MAX)
            .count()
    }
}
