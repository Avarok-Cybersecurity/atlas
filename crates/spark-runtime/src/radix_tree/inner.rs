// SPDX-License-Identifier: AGPL-3.0-only

//! Single-threaded radix-tree state — wrapped behind a `Mutex` in the
//! [`super::RadixTree`] public API. Token sequences are chunked at
//! `block_size` granularity; each node holds one physical KV block index.

use std::collections::HashMap;

mod evict;

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
    /// Task #24 (adapter-correct KV): one radix ROOT per stable `adapter_id`.
    /// Adapter A's and adapter B's node chains live in physically disjoint
    /// subtrees, so a cross-adapter walk can never reach the wrong adapter's
    /// blocks (correct MISS) and an insert under one root can never clobber
    /// another adapter's node (the children map is keyed by token chunk, so
    /// seeding the context-hash alone would NOT prevent that cross-adapter
    /// insert-collision — disjoint roots do). `adapter_id == 0` is the base /
    /// no-adapter root created in `new()`, so base keying is byte-identical to
    /// the pre-LoRA single-root tree.
    roots: HashMap<u64, NodeId>,
    access_counter: u64,
    /// Whether `walk` may match a PARTIAL tail block (the sub-block arm).
    /// Default FALSE — that arm is unsound for a block the requester will
    /// write into; see the comment at the match site. `ATLAS_PREFIX_PARTIAL_TAIL=1`
    /// restores the old behaviour for A/B only.
    allow_partial_tail: bool,
}

impl RadixTreeInner {
    pub(super) fn new() -> Self {
        Self::with_partial_tail_sharing(
            std::env::var("ATLAS_PREFIX_PARTIAL_TAIL").as_deref() == Ok("1"),
        )
    }

    pub(super) fn with_partial_tail_sharing(allow_partial_tail: bool) -> Self {
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
        let mut roots = HashMap::new();
        roots.insert(0u64, 0usize); // base / no-adapter root
        Self {
            nodes: vec![root],
            free_nodes: Vec::new(),
            roots,
            access_counter: 0,
            allow_partial_tail,
        }
    }

    pub(super) fn next_access(&mut self) -> u64 {
        self.access_counter += 1;
        self.access_counter
    }

    /// Read-side root for `adapter_id`: `None` when this adapter has no cached
    /// blocks yet (walk/inc_refs/dec_refs then no-op → clean MISS).
    fn root_for_read(&self, adapter_id: u64) -> Option<NodeId> {
        self.roots.get(&adapter_id).copied()
    }

    /// Insert-side root for `adapter_id`: creates a fresh disjoint root the first
    /// time an adapter caches anything. Roots carry `block_idx == u32::MAX` and
    /// `parent == None`, so they are excluded from eviction and `num_entries`.
    fn root_for_insert(&mut self, adapter_id: u64) -> NodeId {
        if let Some(&id) = self.roots.get(&adapter_id) {
            return id;
        }
        let id = self.alloc_node(RadixNode {
            children: HashMap::new(),
            block_idx: u32::MAX,
            disk_block_id: u32::MAX,
            context_hash: 0,
            ref_count: 0,
            last_access: 0,
            parent: None,
            parent_key: None,
            partial_suffix: None,
        });
        self.roots.insert(adapter_id, id);
        id
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
    pub(super) fn walk(
        &self,
        tokens: &[u32],
        block_size: usize,
        adapter_id: u64,
    ) -> (Vec<u32>, Vec<u32>, usize) {
        let mut current = match self.root_for_read(adapter_id) {
            Some(r) => r,
            None => return (Vec::new(), Vec::new(), 0), // adapter has nothing cached
        };
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
        //
        // ★ DEFAULT OFF (`allow_partial_tail`), because both arms hand the
        // caller a block it will WRITE INTO while another owner still maps it.
        // The matched partial block becomes the requester's TAIL: its very
        // next KV rows (positions matched..block_end) are written into that
        // shared physical block. Arm 2's donor is a LIVE sequence still
        // decoding into the same rows; arm 1's donor block is full, but the
        // requester's tail writes clobber committed rows other sequences map
        // read-only. There is no copy-on-write anywhere in the paged KV
        // manager, so this is two writers, one block.
        //
        // Measured consequence (2026-08-21, qwen3.8-27B + DFlash2, GB10):
        // with identical prompts in flight the writers stay byte-identical in
        // lockstep and the tear is BENIGN — which is why serial decode looked
        // clean — but any desync (the batched verify's band-dependent
        // rounding was the trigger) makes the writes differ, the block holds
        // a mix of two streams, and every reader derails mid-token at the
        // block boundary: C=4 went 0/4 with three sequences emitting
        // IDENTICAL wrong text (all reading the same torn block).
        //
        // The same arm also has a refcount hole: `inc_refs` walks full-block
        // NODES only, so a matched partial block is pushed into the
        // requester's block_table WITHOUT a ref — the eventual free
        // underflows the donor's count.
        //
        // Cost of OFF: up to block_size-1 tokens of prefill recompute per
        // hit, and non-block-aligned exact-repeat prompts no longer reach the
        // Marconi exact-leaf shortcut (matched < total). Correct first;
        // a copy-on-write tail (fresh block + per-layer D2D of the matched
        // rows, event-ordered against the donor's stream) is the way to win
        // the tokens back if anyone wants them.
        let remainder = tokens.len() - matched_tokens;
        if self.allow_partial_tail
            && remainder > 0
            && remainder < block_size
            && matched_tokens == num_full_blocks * block_size
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
    pub(super) fn inc_refs(
        &mut self,
        tokens: &[u32],
        block_size: usize,
        num_matched: usize,
        adapter_id: u64,
    ) {
        let access = self.next_access();
        let mut current = match self.root_for_read(adapter_id) {
            Some(r) => r,
            None => return,
        };
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

    /// Decrement ref_count on up to `matched_tokens` along the matched path.
    pub(super) fn dec_refs(
        &mut self,
        tokens: &[u32],
        block_size: usize,
        matched_tokens: usize,
        adapter_id: u64,
    ) {
        let mut current = match self.root_for_read(adapter_id) {
            Some(r) => r,
            None => return,
        };
        let num_full_blocks = (matched_tokens / block_size).min(tokens.len() / block_size);

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
        adapter_id: u64,
    ) -> crate::prefix_cache::InsertAcquired {
        let access = self.next_access();
        let root_id = self.root_for_insert(adapter_id);
        let mut current = root_id;
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
        // Blocks the cache starts / stops storing here; the caller takes and
        // releases exactly one KV ref each. See `InsertAcquired`.
        let mut newly_owned_blocks: Vec<u32> = Vec::new();
        let mut released_blocks: Vec<u32> = Vec::new();

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
                // A real child supersedes the partial slot; release its ref.
                released_blocks.extend(self.nodes[current].partial_suffix.take().map(|p| p.1));
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
                newly_owned_blocks.push(block_table[i]);
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
        if remainder > 0 && block_table.len() > num_full_blocks && current != root_id {
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
            // The cache owns a KV ref on the partial block exactly as it does on
            // a node's own block — `walk` hands it out and `evict` hands it back,
            // so an unreferenced one sits on the free list while still cached.
            // See `InsertAcquired` for the failure this prevents.
            let prior_block = self.nodes[current].partial_suffix.as_ref().map(|p| p.1);
            self.nodes[current].partial_suffix = Some((partial_toks, partial_block, partial_disk));
            if prior_block != Some(partial_block) {
                newly_owned_blocks.push(partial_block);
                released_blocks.extend(prior_block);
            }
            if hss_active && partial_disk != u32::MAX && prior != Some(partial_disk) {
                newly_acquired.push(partial_disk);
            }
        }

        crate::prefix_cache::InsertAcquired {
            disk_block_ids: newly_acquired,
            blocks: newly_owned_blocks,
            released_blocks,
        }
    }
}
