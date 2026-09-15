// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_runtime::kv_cache::PagedKvCache;

/// Keep the existing two-block tail allowance independently for every
/// sequence slot. The pool is shared, while each slot owns its block table.
pub(super) fn draft_pool_blocks(
    max_seq_len: usize,
    block_size: usize,
    slots: usize,
) -> Result<usize> {
    anyhow::ensure!(
        max_seq_len > 0 && block_size > 0 && slots > 0,
        "Qwen MTP pool requires positive context, block size and sequence capacity"
    );
    (max_seq_len / block_size)
        .checked_add(2)
        .and_then(|blocks| blocks.checked_mul(slots))
        .ok_or_else(|| anyhow::anyhow!("Qwen MTP KV pool capacity overflow"))
}

pub(super) fn release_blocks(pool: &mut PagedKvCache, blocks: &mut Vec<u32>) {
    pool.free_blocks(blocks);
    blocks.clear();
}

impl super::Qwen4ExpMtpHead {
    pub(in crate::layers) fn release_draft_state(
        &self,
        state: &mut super::Qwen4ExpMtpState,
        gpu: &dyn spark_runtime::gpu::GpuBackend,
    ) -> Result<()> {
        // Return the host-managed blocks even if releasing device state fails.
        // A repeated teardown sees an empty table and cannot double-release.
        release_blocks(
            &mut *self
                .kv_cache
                .lock()
                .map_err(|_| anyhow::anyhow!("Qwen MTP KV pool poisoned"))?,
            &mut state.block_table,
        );
        state.seq_len = 0;
        state.pending_draft = None;
        state.last_num_drafted = 0;
        state.pending_rewind = 0;
        // The draft attention body owns per-sequence QSA device buffers just
        // like target attention. Dropping LayerState only drops their pointers.
        self.module
            .body
            .release_state(state.body_state.as_mut(), gpu)
    }
}

impl super::Qwen4ExpMtpState {
    pub(in crate::layers) fn begin_round(&mut self) -> Result<()> {
        // after_verify clears last_num_drafted. A nonzero count here means
        // the scheduler discarded an unverified round (gate/thinking change),
        // or sampling failed after some draft bodies completed successfully.
        self.pending_rewind = self
            .pending_rewind
            .checked_add(self.last_num_drafted)
            .ok_or_else(|| anyhow::anyhow!("Qwen MTP draft rewind overflow"))?;
        self.last_num_drafted = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;
    use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype};

    fn state() -> super::super::Qwen4ExpMtpState {
        super::super::Qwen4ExpMtpState {
            block_table: Vec::new(),
            seq_len: 3,
            body_state: Box::new(crate::layer::EmptyLayerState),
            pending_draft: None,
            last_num_drafted: 0,
            pending_rewind: 0,
        }
    }

    #[test]
    fn abandoned_round_queues_every_completed_draft_for_rewind() {
        for completed in 1..=3 {
            let mut state = state();
            state.last_num_drafted = completed;
            state.begin_round().unwrap();
            assert_eq!(state.pending_rewind, completed);
            assert_eq!(state.last_num_drafted, 0);
            state.begin_round().unwrap();
            assert_eq!(state.pending_rewind, completed);
        }
    }

    #[test]
    fn accepted_round_preserves_only_its_existing_rejection_rewind() {
        let mut state = state();
        state.pending_rewind = 2;
        state.begin_round().unwrap();
        assert_eq!(state.pending_rewind, 2);
        assert_eq!(state.last_num_drafted, 0);
    }

    #[test]
    fn draft_pool_covers_every_live_slot_with_tail_space() {
        assert_eq!(draft_pool_blocks(32, 16, 2).unwrap(), 8);
        assert_eq!(draft_pool_blocks(33, 16, 2).unwrap(), 8);
    }

    #[test]
    fn draft_pool_rejects_empty_and_overflowed_capacity() {
        assert!(draft_pool_blocks(32, 16, 0).is_err());
        assert!(draft_pool_blocks(0, 16, 1).is_err());
        assert!(draft_pool_blocks(32, 0, 1).is_err());
        assert!(draft_pool_blocks(usize::MAX, 1, 2).is_err());
    }

    #[test]
    fn finished_draft_sequences_return_blocks_for_repeated_requests() {
        let gpu = MockGpuBackend::new();
        let config = KvCacheConfig {
            block_size: 16,
            num_kv_heads: 1,
            head_dim: 8,
            num_layers: 1,
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        };
        let mut pool = PagedKvCache::new(config, 4, &gpu).unwrap();
        for _ in 0..8 {
            let mut first = vec![pool.alloc_block().unwrap(), pool.alloc_block().unwrap()];
            let mut second = vec![pool.alloc_block().unwrap(), pool.alloc_block().unwrap()];
            assert_eq!(pool.num_free_blocks(), 0);
            release_blocks(&mut pool, &mut first);
            assert!(first.is_empty());
            assert_eq!(pool.num_free_blocks(), 2);
            release_blocks(&mut pool, &mut first);
            assert_eq!(pool.num_free_blocks(), 2);
            release_blocks(&mut pool, &mut second);
            assert_eq!(pool.num_free_blocks(), 4);
        }
    }
}
