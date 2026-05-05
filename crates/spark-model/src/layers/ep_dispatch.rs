// SPDX-License-Identifier: AGPL-3.0-only

//! EP=2 token dispatch/combine routing for MoE expert parallelism.
//!
//! Instead of dense all-reduce after local expert compute, this module
//! partitions tokens by expert ownership and dispatches only the tokens
//! that need remote experts. Communication is O(dispatched_tokens * hidden)
//! rather than O(total_tokens * hidden).
//!
//! See `tasks/ep2-token-dispatch-design.md` for the full design.

/// Routing table for EP token dispatch.
///
/// Partitions top-K expert assignments into local (this rank owns the expert)
/// and remote (partner rank owns the expert) buckets. Each entry is a
/// (token_index, expert_id, weight) triple.
#[derive(Debug)]
pub struct EpRoutingTable {
    /// Token indices routed to this rank's experts.
    pub local_token_indices: Vec<u32>,
    /// Expert IDs for local tokens (absolute, not rank-relative).
    pub local_expert_ids: Vec<u32>,
    /// Routing weights for local tokens.
    pub local_weights: Vec<f32>,
    /// Token indices routed to the remote rank's experts.
    pub remote_token_indices: Vec<u32>,
    /// Expert IDs for remote tokens (absolute, not rank-relative).
    pub remote_expert_ids: Vec<u32>,
    /// Routing weights for remote tokens.
    pub remote_weights: Vec<f32>,
}

impl EpRoutingTable {
    /// Number of (token, expert) pairs handled locally.
    pub fn local_count(&self) -> usize {
        self.local_token_indices.len()
    }

    /// Number of (token, expert) pairs dispatched to remote rank.
    pub fn remote_count(&self) -> usize {
        self.remote_token_indices.len()
    }

    /// Total number of (token, expert) pairs (should equal num_tokens * top_k).
    pub fn total_count(&self) -> usize {
        self.local_count() + self.remote_count()
    }
}

/// Build EP routing table from flattened gate indices and weights.
///
/// Given the top-K expert assignments for M tokens, partitions them into
/// local vs remote based on which rank owns each expert.
///
/// # Arguments
/// * `gate_indices` - Flattened [M * top_k] expert indices from top-K selection
/// * `gate_weights` - Flattened [M * top_k] routing weights from top-K selection
/// * `num_tokens`   - Number of tokens (M)
/// * `top_k`        - Number of experts per token
/// * `local_expert_start` - First expert index owned by this rank (inclusive)
/// * `local_expert_end`   - Last expert index owned by this rank (exclusive)
///
/// # Panics
/// Panics if `gate_indices.len() != num_tokens * top_k` or
/// `gate_weights.len() != num_tokens * top_k`.
pub fn build_ep_routing_table(
    gate_indices: &[u32],
    gate_weights: &[f32],
    num_tokens: usize,
    top_k: usize,
    local_expert_start: usize,
    local_expert_end: usize,
) -> EpRoutingTable {
    let total = num_tokens * top_k;
    assert_eq!(gate_indices.len(), total, "gate_indices length mismatch");
    assert_eq!(gate_weights.len(), total, "gate_weights length mismatch");

    // Pre-allocate with worst-case capacity (all local or all remote).
    let mut local_token_indices = Vec::with_capacity(total);
    let mut local_expert_ids = Vec::with_capacity(total);
    let mut local_weights = Vec::with_capacity(total);
    let mut remote_token_indices = Vec::with_capacity(total);
    let mut remote_expert_ids = Vec::with_capacity(total);
    let mut remote_weights = Vec::with_capacity(total);

    for token_idx in 0..num_tokens {
        for k in 0..top_k {
            let flat_idx = token_idx * top_k + k;
            let expert_id = gate_indices[flat_idx];
            let weight = gate_weights[flat_idx];
            let eid = expert_id as usize;

            if eid >= local_expert_start && eid < local_expert_end {
                local_token_indices.push(token_idx as u32);
                local_expert_ids.push(expert_id);
                local_weights.push(weight);
            } else {
                remote_token_indices.push(token_idx as u32);
                remote_expert_ids.push(expert_id);
                remote_weights.push(weight);
            }
        }
    }

    EpRoutingTable {
        local_token_indices,
        local_expert_ids,
        local_weights,
        remote_token_indices,
        remote_expert_ids,
        remote_weights,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_local() {
        // 2 tokens, top_k=2, all experts in local range [0, 256)
        let indices = vec![3u32, 7, 100, 200];
        let weights = vec![0.6f32, 0.4, 0.55, 0.45];
        let table = build_ep_routing_table(&indices, &weights, 2, 2, 0, 256);

        assert_eq!(table.local_count(), 4);
        assert_eq!(table.remote_count(), 0);
        assert_eq!(table.total_count(), 4);
        assert_eq!(table.local_token_indices, vec![0, 0, 1, 1]);
        assert_eq!(table.local_expert_ids, vec![3, 7, 100, 200]);
    }

    #[test]
    fn test_all_remote() {
        // 2 tokens, top_k=2, all experts in remote range [256, 512)
        let indices = vec![300u32, 400, 256, 511];
        let weights = vec![0.6f32, 0.4, 0.55, 0.45];
        let table = build_ep_routing_table(&indices, &weights, 2, 2, 0, 256);

        assert_eq!(table.local_count(), 0);
        assert_eq!(table.remote_count(), 4);
        assert_eq!(table.remote_token_indices, vec![0, 0, 1, 1]);
        assert_eq!(table.remote_expert_ids, vec![300, 400, 256, 511]);
    }

    #[test]
    fn test_mixed_routing() {
        // 3 tokens, top_k=2, experts split across ranks
        // Rank 0 owns [0, 256), Rank 1 owns [256, 512)
        let indices = vec![
            10u32, 300, // token 0: expert 10 (local), expert 300 (remote)
            255, 256, // token 1: expert 255 (local), expert 256 (remote)
            400, 500, // token 2: expert 400 (remote), expert 500 (remote)
        ];
        let weights = vec![0.7f32, 0.3, 0.5, 0.5, 0.6, 0.4];
        let table = build_ep_routing_table(&indices, &weights, 3, 2, 0, 256);

        assert_eq!(table.local_count(), 2);
        assert_eq!(table.remote_count(), 4);
        assert_eq!(table.local_token_indices, vec![0, 1]);
        assert_eq!(table.local_expert_ids, vec![10, 255]);
        assert_eq!(table.local_weights, vec![0.7, 0.5]);
        assert_eq!(table.remote_token_indices, vec![0, 1, 2, 2]);
        assert_eq!(table.remote_expert_ids, vec![300, 256, 400, 500]);
        assert_eq!(table.remote_weights, vec![0.3, 0.5, 0.6, 0.4]);
    }

    #[test]
    fn test_rank1_perspective() {
        // Same scenario but from rank 1's perspective [256, 512)
        let indices = vec![
            10u32, 300, // token 0: expert 10 (remote for rank1), 300 (local)
            255, 256, // token 1: expert 255 (remote), 256 (local)
        ];
        let weights = vec![0.7f32, 0.3, 0.5, 0.5];
        let table = build_ep_routing_table(&indices, &weights, 2, 2, 256, 512);

        assert_eq!(table.local_count(), 2);
        assert_eq!(table.remote_count(), 2);
        assert_eq!(table.local_token_indices, vec![0, 1]);
        assert_eq!(table.local_expert_ids, vec![300, 256]);
        assert_eq!(table.remote_token_indices, vec![0, 1]);
        assert_eq!(table.remote_expert_ids, vec![10, 255]);
    }

    #[test]
    fn test_single_token() {
        let indices = vec![5u32, 260, 100];
        let weights = vec![0.5f32, 0.3, 0.2];
        let table = build_ep_routing_table(&indices, &weights, 1, 3, 0, 256);

        assert_eq!(table.local_count(), 2); // experts 5, 100
        assert_eq!(table.remote_count(), 1); // expert 260
        assert_eq!(table.total_count(), 3);
    }

    #[test]
    #[should_panic(expected = "gate_indices length mismatch")]
    fn test_length_mismatch_panics() {
        let indices = vec![1u32, 2, 3]; // 3 elements
        let weights = vec![0.5f32, 0.5]; // 2 elements — mismatch with num_tokens * top_k
        build_ep_routing_table(&indices, &weights, 2, 2, 0, 256);
    }

    #[test]
    fn test_boundary_experts() {
        // Expert exactly at boundary: 255 is local, 256 is remote for rank 0
        let indices = vec![255u32, 256];
        let weights = vec![0.6f32, 0.4];
        let table = build_ep_routing_table(&indices, &weights, 1, 2, 0, 256);

        assert_eq!(table.local_count(), 1);
        assert_eq!(table.remote_count(), 1);
        assert_eq!(table.local_expert_ids, vec![255]);
        assert_eq!(table.remote_expert_ids, vec![256]);
    }
}
