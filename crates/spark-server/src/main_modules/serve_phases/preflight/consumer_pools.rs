// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: run the consumer-card verify-slot fit before the reserve is
//! priced, and publish its cap (port lane L24; the fit and its cell live in
//! `spark_model::ssm_reserve::consumer_fit`).
//!
//! Called from `preflight_reserve` right after the FP16 h-state preconditions
//! and before `ssm_reserve::mtp_state_slots` is first read, so the reserve
//! priced here, the pools `SsmStatePool::new` allocates and the scheduler's
//! spec-dispatch guard all see the same slot count.
//!
//! Inert unless every one of these holds: the target declares
//! `memory_fit_consumer` (ada only), the serve is a Qwen MTP serve
//! (`--speculative`, no `--dflash`, whose pools are gamma-sized), the model has
//! GDN layers, `AVAROK_MTP_POOL_FULL_WIDTH` and `AVAROK_MTP_CONSUMER_FIT_OFF`
//! are absent, the checkpoint's size can be read, and the fit keeps fewer slots
//! than the pool diet would. Otherwise nothing is published and every number
//! downstream is the pre-sprint one.

use avarok_core::config::ModelConfig;

use super::headroom::{PostLoadInputs, checkpoint_bytes, kv_floor_bytes};
use crate::cli;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// What the fit decided, for the log and the tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct VerifyFit {
    pub(super) requested: usize,
    pub(super) cap: usize,
    pub(super) budget: usize,
    pub(super) requested_bytes: usize,
    pub(super) cap_bytes: usize,
}

/// The fit's inputs that preflight already knows, gathered so the arithmetic
/// below is a pure function a CPU test can drive.
#[derive(Clone, Copy, Debug)]
pub(super) struct FitInputs {
    pub(super) free_mem: usize,
    pub(super) resident_weights: usize,
    pub(super) max_batch_size: usize,
    pub(super) per_seq_blob: usize,
    pub(super) kv_floor: usize,
    pub(super) headroom: usize,
    /// `ssm_reserve::mtp_state_slots(max_batch_size)` before any cap.
    pub(super) requested: usize,
}

/// The fit over `verify_bytes(n)` (the verify-pool term of `n` slots).
pub(super) fn decide(i: &FitInputs, verify_bytes: impl Fn(usize) -> usize) -> VerifyFit {
    let budget = spark_model::ssm_reserve::verify_budget_bytes(
        i.free_mem,
        i.resident_weights,
        i.max_batch_size.saturating_mul(i.per_seq_blob),
        i.kv_floor,
        i.headroom,
    );
    let cap = spark_model::ssm_reserve::fit_verify_slots(i.requested, budget, &verify_bytes);
    VerifyFit {
        requested: i.requested,
        cap,
        budget,
        requested_bytes: verify_bytes(i.requested),
        cap_bytes: verify_bytes(cap),
    }
}

/// Run the fit for this serve and publish the cap when it binds.
pub(super) fn publish_verify_slot_fit(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    free_mem: usize,
    post_load: &PostLoadInputs<'_>,
) -> Option<VerifyFit> {
    let consumer = spark_model::layers::ops::target_defaults::resolved()
        .memory_fit_consumer
        .value;
    if !consumer
        || !args.speculative
        || args.dflash
        || config.num_ssm_layers() == 0
        || spark_model::ssm_reserve::mtp_pool_full_width()
        || spark_model::ssm_reserve::consumer_fit_disabled()
    {
        return None;
    }
    let Some(checkpoint) = checkpoint_bytes(post_load.model_dir) else {
        tracing::info!(
            "MTP verify pools: consumer fit skipped (the checkpoint's size could not be read)"
        );
        return None;
    };
    let embed_freed = spark_model::port::embed_host::predicted_freed_bytes(
        config.vocab_size,
        config.hidden_size,
        config.tie_word_embeddings,
    );
    let resident = (checkpoint as usize)
        .saturating_sub(post_load.fp8_ffn_freed)
        .saturating_sub(embed_freed);
    let h_blob = config.num_ssm_layers() * config.ssm_h_state_bytes();
    let conv_blob = config.num_ssm_layers() * config.ssm_conv_state_bytes();
    let num_drafts = args.resolved_num_drafts();
    let f16_pool = spark_model::layers::qwen3_ssm::ssm_h_f16_pool_enabled();
    let rollback = spark_model::ssm_reserve::ssm_rollback_mode();
    let inputs = FitInputs {
        free_mem,
        resident_weights: resident,
        max_batch_size: args.max_batch_size,
        per_seq_blob: spark_model::ssm_reserve::ssm_h_stored_bytes(h_blob, f16_pool) + conv_blob,
        kv_floor: kv_floor_bytes(args, config, post_load.kv_dtype),
        headroom: super::spec_headroom::cuda_headroom(args, config).0,
        requested: spark_model::ssm_reserve::mtp_state_slots(args.max_batch_size),
    };
    let fit = decide(&inputs, |n| {
        spark_model::ssm_reserve::ssm_pool_reserve_bytes(
            0, h_blob, conv_blob, true, num_drafts, n, false, f16_pool, rollback,
        )
    });
    if fit.cap >= fit.requested {
        return Some(fit);
    }
    let cap = spark_model::ssm_reserve::set_consumer_verify_slot_cap(fit.cap);
    tracing::warn!(
        "MTP verify pools: consumer fit covers {cap} of {} slots ({:.2} GiB instead of {:.2} GiB) \
         so they take at most 1/{} of the {:.2} GiB left beside {:.2} GiB of resident weights, the \
         base state, the KV floor and the CUDA headroom; the other slots plain-decode \
         (memory_fit_consumer; AVAROK_MTP_CONSUMER_FIT_OFF to size every slot)",
        fit.requested,
        fit.cap_bytes as f64 / GIB,
        fit.requested_bytes as f64 / GIB,
        spark_model::ssm_reserve::VERIFY_BUDGET_DIVISOR,
        (fit.budget * spark_model::ssm_reserve::VERIFY_BUDGET_DIVISOR) as f64 / GIB,
        resident as f64 / GIB,
    );
    Some(VerifyFit { cap, ..fit })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: usize = 1024 * 1024;

    /// The decision on the 27B at batch 8 on the 4060 Ti (77 section 12.1):
    /// one covered slot, the pools shrink from 8 slots to 1.
    #[test]
    fn the_27b_at_batch_8_covers_one_slot() {
        let per_slot = 643_301_376usize;
        let i = FitInputs {
            free_mem: 15_660 * MIB,
            resident_weights: 14_252_845_984 - 2_542_796_800,
            max_batch_size: 8,
            per_seq_blob: 158_859_264,
            kv_floor: 8 * 4096 * 32 * 1024,
            headroom: 1_157_627_904,
            requested: 8,
        };
        let fit = decide(&i, |n| n * per_slot);
        assert_eq!(fit.cap, 1);
        assert_eq!(fit.requested_bytes, 8 * per_slot);
        assert_eq!(fit.cap_bytes, per_slot);
        assert_eq!(fit.budget, 604_203_568);
    }

    /// A GB10-sized budget keeps every slot, which is why the fit could not
    /// publish there even if the row were on.
    #[test]
    fn a_large_part_keeps_every_slot() {
        let i = FitInputs {
            free_mem: 115 * 1024 * MIB,
            resident_weights: 17 * 1024 * MIB,
            max_batch_size: 32,
            per_seq_blob: 158_859_264,
            kv_floor: 32 * 4096 * 32 * 1024,
            headroom: 4096 * MIB,
            requested: 32,
        };
        let fit = decide(&i, |n| n * 643_301_376);
        assert_eq!(fit.cap, 32);
        assert_eq!(fit.cap_bytes, fit.requested_bytes);
    }
}
