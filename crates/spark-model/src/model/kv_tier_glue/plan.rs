// SPDX-License-Identifier: AGPL-3.0-only
// Design after LMCache lmcache/v1/kv_load_policy.py and the hybrid rule of lmcache/v1/api.py:636-656
// (commit 3aefae1, Apache-2.0); no code copied.

//! Responsibility: the host KV tier's restore plan, as a pure function.
//!
//! Given what the radix tree already holds on the GPU, how many of the next
//! blocks the host tier holds, and how deep the deepest SSM snapshot for this
//! prefix reaches, decide which block range (if any) to restore. The rule is
//! the hybrid-model boundary rule of the plan (3.7.5, L26b):
//!
//! - attention KV needs the whole prefix, so a restore covers blocks
//!   `[gpu, to)` contiguous with what the GPU already holds;
//! - a hybrid model's recurrent (GDN) state needs a snapshot at or before the
//!   restored depth; restoring KV past the deepest snapshot saves nothing,
//!   because the prefill recomputes from the snapshot anyway, and with no
//!   snapshot past the GPU match the restore saves nothing at all, so it loads
//!   nothing;
//! - a restore shorter than `min_tokens` cannot repay its fixed cost.
//!
//! The cost gate (copy time against prefill time) and the in-flight cap are
//! the tier's (`KvHostTier::gate`, `KvHostTier::max_restore_blocks`, lane
//! L26a): the cap bounds `host_hits` before this plan runs, the gate judges the
//! span this plan picked.

/// Why a prefetch restored nothing, or that it restored.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PrefetchOutcome {
    /// The tier is not serving (row off, no budget, prefix cache off, or the
    /// request is not eligible).
    #[default]
    Off,
    /// The GPU already holds every usable block.
    GpuHit,
    /// The host tier holds none of the next blocks.
    Miss,
    /// A hybrid model with no SSM snapshot past the GPU match: loading KV
    /// would save nothing, so nothing is loaded.
    NoSnapshot,
    /// The restorable span is shorter than `--kv-tier-min-tokens`.
    BelowMinTokens,
    /// The tier's cost model judged a recompute cheaper.
    Declined,
    /// No free GPU block could be found for the restore.
    NoBlocks,
    /// The restore failed; every GPU block stayed valid and the prefill
    /// recomputes.
    Failed,
    /// The blocks were restored into the radix tree.
    Restored,
}

/// What the planner sees.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PlanInput {
    /// Full blocks a restore may cover: at most `(prompt_len - 1) / bs`, so at
    /// least one prompt token is always computed for the first logits.
    pub usable_blocks: usize,
    /// Full blocks the radix tree already holds on the GPU.
    pub gpu_blocks: usize,
    /// Leading host-tier hits for blocks `gpu_blocks..usable_blocks`.
    pub host_hits: usize,
    /// The model has recurrent (SSM / GDN) layers.
    pub has_ssm: bool,
    /// Depth in tokens of the deepest SSM snapshot (resident or tiered) whose
    /// prefix matches, at most `(gpu_blocks + host_hits) * block_size`.
    /// Ignored when `has_ssm` is false.
    pub snapshot_tokens: usize,
    /// Tokens per block.
    pub block_size: usize,
    /// `--kv-tier-min-tokens`.
    pub min_tokens: usize,
}

/// The planner's answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Plan {
    /// Restore nothing, for this reason.
    Nothing(PrefetchOutcome),
    /// Restore blocks `[from, to)` from the host tier.
    Restore { from: usize, to: usize },
}

/// The restore plan for one prompt.
pub(crate) fn plan(i: &PlanInput) -> Plan {
    let bs = i.block_size.max(1);
    if i.gpu_blocks >= i.usable_blocks {
        return Plan::Nothing(PrefetchOutcome::GpuHit);
    }
    let host = i.host_hits.min(i.usable_blocks - i.gpu_blocks);
    if host == 0 {
        return Plan::Nothing(PrefetchOutcome::Miss);
    }
    let available = i.gpu_blocks + host;
    let to = if i.has_ssm {
        let snap = i.snapshot_tokens.min(available * bs);
        if snap <= i.gpu_blocks * bs {
            return Plan::Nothing(PrefetchOutcome::NoSnapshot);
        }
        // The block that holds the snapshot's last token is needed whole: the
        // prefill reads KV below the snapshot from it.
        snap.div_ceil(bs).min(available)
    } else {
        available
    };
    let tokens = (to - i.gpu_blocks) * bs;
    if tokens < i.min_tokens {
        return Plan::Nothing(PrefetchOutcome::BelowMinTokens);
    }
    Plan::Restore {
        from: i.gpu_blocks,
        to,
    }
}

/// Full blocks a restore may cover for a prompt of `prompt_len` tokens.
pub(crate) fn usable_blocks(prompt_len: usize, block_size: usize) -> usize {
    if block_size == 0 {
        return 0;
    }
    prompt_len.saturating_sub(1) / block_size
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(gpu: usize, host: usize, has_ssm: bool, snap: usize) -> PlanInput {
        PlanInput {
            usable_blocks: 64,
            gpu_blocks: gpu,
            host_hits: host,
            has_ssm,
            snapshot_tokens: snap,
            block_size: 16,
            min_tokens: 32,
        }
    }

    #[test]
    fn at_least_one_prompt_token_is_left_to_compute() {
        assert_eq!(usable_blocks(0, 16), 0);
        assert_eq!(usable_blocks(16, 16), 0);
        assert_eq!(usable_blocks(17, 16), 1);
        assert_eq!(usable_blocks(32, 16), 1);
        assert_eq!(usable_blocks(33, 16), 2);
        assert_eq!(usable_blocks(100, 0), 0);
    }

    /// The boundary-rule table of the brief: each row is (GPU blocks, host
    /// hits, hybrid?, snapshot depth) -> plan.
    #[test]
    fn boundary_rule_table() {
        let rows: &[(usize, usize, bool, usize, Plan)] = &[
            // Attention-only: everything the host holds, contiguous with the GPU.
            (0, 10, false, 0, Plan::Restore { from: 0, to: 10 }),
            (4, 6, false, 0, Plan::Restore { from: 4, to: 10 }),
            // Nothing on the host.
            (4, 0, false, 0, Plan::Nothing(PrefetchOutcome::Miss)),
            // Hybrid, KV on the host but no snapshot at all: load nothing.
            (0, 10, true, 0, Plan::Nothing(PrefetchOutcome::NoSnapshot)),
            // Hybrid, snapshot only inside the GPU match: load nothing.
            (4, 6, true, 64, Plan::Nothing(PrefetchOutcome::NoSnapshot)),
            // Hybrid, block-aligned snapshot at block 8: restore up to it, not past.
            (4, 6, true, 128, Plan::Restore { from: 4, to: 8 }),
            // Hybrid, snapshot mid-block 8 (token 130): block 8 is needed whole.
            (4, 6, true, 130, Plan::Restore { from: 4, to: 9 }),
            // Hybrid, snapshot at the very end of the host run.
            (4, 6, true, 160, Plan::Restore { from: 4, to: 10 }),
            // Hybrid, a probe answer deeper than the run is clamped to the run.
            (4, 6, true, 400, Plan::Restore { from: 4, to: 10 }),
            // Below --kv-tier-min-tokens (32): one block of 16 tokens.
            (
                4,
                1,
                false,
                0,
                Plan::Nothing(PrefetchOutcome::BelowMinTokens),
            ),
            (
                4,
                6,
                true,
                80,
                Plan::Nothing(PrefetchOutcome::BelowMinTokens),
            ),
        ];
        for &(gpu, host, ssm, snap, want) in rows {
            assert_eq!(
                plan(&input(gpu, host, ssm, snap)),
                want,
                "gpu={gpu} host={host} ssm={ssm} snap={snap}"
            );
        }
    }

    #[test]
    fn a_full_gpu_match_needs_no_restore() {
        let mut i = input(64, 5, false, 0);
        assert_eq!(plan(&i), Plan::Nothing(PrefetchOutcome::GpuHit));
        i.gpu_blocks = 70;
        assert_eq!(plan(&i), Plan::Nothing(PrefetchOutcome::GpuHit));
    }

    #[test]
    fn host_hits_are_clamped_to_the_usable_blocks() {
        let mut i = input(60, 50, false, 0);
        i.min_tokens = 0;
        assert_eq!(plan(&i), Plan::Restore { from: 60, to: 64 });
    }

    #[test]
    fn a_zero_min_tokens_restores_a_single_block() {
        let mut i = input(4, 1, false, 0);
        i.min_tokens = 0;
        assert_eq!(plan(&i), Plan::Restore { from: 4, to: 5 });
    }
}
