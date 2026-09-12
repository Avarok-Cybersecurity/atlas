// SPDX-License-Identifier: AGPL-3.0-only

//! The pre-load reserve REFUSAL: what it suggests, and why it asked for what
//! it asked for.
//!
//! A sibling of `preflight.rs` (which sits at the 500-line cap), same
//! precedent as `per_sequence_state.rs` and `decode_ring.rs`.
//!
//! Refusal is now the LAST resort, not the first answer: the decode-rollback
//! ring shrinks to fit before this runs (#915), so reaching here means the
//! configuration does not fit even with no rollback depth at all. That makes
//! the text load-bearing — it is the only thing an operator has to act on —
//! which is why it carries the ring formula as well as a suggested batch.

use atlas_core::config::ModelConfig;

use crate::cli;

use super::decode_ring;

/// The terms the refusal text needs that are not derivable from `args` /
/// `config`. Grouped so the call site stays one statement.
pub(super) struct Refusal {
    /// `inference_reserve + buffer_arena_bytes`.
    pub(super) total_reserve: usize,
    pub(super) free_mem: usize,
    /// The terms that do NOT scale with `--max-seq-len`: SSM pools, h stage,
    /// both snapshot regions, CUDA headroom. What is left of `free_mem` after
    /// them is what the suggested sequence length is derived from.
    pub(super) seq_len_independent: usize,
    /// Ring depth the flags asked for, and the depth actually reserved for
    /// (they differ only when an explicit depth was pinned — the auto-fit
    /// path never reaches this refusal).
    pub(super) ring_requested: usize,
    pub(super) ring_slots: usize,
    pub(super) per_seq_blob: usize,
}

/// Build the refusal error. Pure formatting over already-computed bytes.
pub(super) fn reserve_refusal(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    r: Refusal,
) -> anyhow::Error {
    let need_gb = r.total_reserve as f64 / (1024.0 * 1024.0 * 1024.0);
    let free_gb = r.free_mem as f64 / (1024.0 * 1024.0 * 1024.0);
    let budget_for_seq_term = r.free_mem.saturating_sub(r.seq_len_independent) / 2;
    let per_tok_bytes = {
        let key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let nv = config.linear_num_value_heads;
        let conv_dim = key_dim * 2 + value_dim;
        if conv_dim > 0 && config.num_ssm_layers() > 0 {
            (conv_dim * 2) + (nv * 2 * 4) + (value_dim * 2) + (value_dim * 2)
        } else {
            0
        }
    };
    let suggested = budget_for_seq_term
        .checked_div(per_tok_bytes)
        .map(|q| q.max(2048))
        .unwrap_or(0);
    let hint = if suggested > 0 && suggested < args.max_seq_len {
        format!(
            " Try --max-seq-len {} (or lower --max-batch-size / --num-drafts).",
            suggested
        )
    } else if args.max_batch_size > 1 {
        " Reduce --max-batch-size.".to_string()
    } else {
        " Use a smaller model or a GPU with more memory.".to_string()
    };
    anyhow::anyhow!(
        "Preflight failed: inference buffers alone need {:.2} GB but only {:.2} GB is free on the GPU \
         (before weights load). SSM pool + GDN chunked prefill scales with --max-seq-len={} × --max-batch-size={}.{}{}",
        need_gb,
        free_gb,
        args.max_seq_len,
        args.max_batch_size,
        hint,
        ring_note(args, &r),
    )
}

/// Issue #915's third bullet: an operator staring at a 45.8 GiB refusal could
/// not see that 37.9 GiB of it was 8 rollback snapshots x batch 32. Print the
/// arithmetic, and say why shrinking the ring did not rescue the boot.
fn ring_note(args: &cli::ServeArgs, r: &Refusal) -> String {
    if r.ring_requested == 0 {
        return String::new();
    }
    format!(
        " {} — {}.",
        decode_ring::formula(r.ring_slots, args.max_batch_size, r.per_seq_blob),
        if spark_model::ssm_reserve::published_decode_ring_slots().is_some() {
            "an explicit --ssm-decode-ring-slots pins the depth, so it was not shrunk"
        } else {
            "even 0 ring slots does not fit, so shrinking it cannot rescue this boot"
        },
    )
}
