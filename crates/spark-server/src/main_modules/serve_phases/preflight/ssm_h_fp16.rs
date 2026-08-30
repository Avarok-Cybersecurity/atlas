// SPDX-License-Identifier: AGPL-3.0-only

//! `--ssm-h-dtype f16` / `f16-pool` boot preconditions.
//!
//! A sibling of `preflight.rs` (which is at the 500-line cap): this is the
//! one self-contained refusal list in it, and it is the list that grows every
//! time a stage of the FP16 h-state lands.

use anyhow::Result;
use atlas_core::config::ModelConfig;

use crate::cli;

/// `ATLAS_SSM_H_FP16` refuses rather than degrades.
///
/// The flag narrows the GDN h-state to FP16. Stage 1 shipped twins of the two
/// NON-speculative decode kernels — `gated_delta_rule_decode_f16_strided_norm_half`
/// (batched) and `..._f16_norm` (per-sequence, taken at n == 1 and whenever
/// pool slots fragment out of slice order). Stage 2 adds twins of the MTP
/// verify path — `gated_delta_rule_wy{2,3,4}_f16` plus the register-resident
/// `wy{2,3}_resident_f16` — so the state stays FP16 through a verify step and
/// the flag composes with `--speculative`. That matters because the ladder's
/// low and middle rungs are all spec-ON, so before stage 2 they could not use
/// the lever at all.
///
/// Every remaining h-state reader is still FP32, and an FP32 kernel pointed at
/// an FP16 pool produces fluent garbage rather than an error — so the
/// unsupported configurations are rejected here, at boot, instead of being
/// discovered in a benchmark.
///
/// Stage 3 (`--ssm-h-dtype f16-pool`) is where SIZING changes: every h pool
/// is allocated at 2 bytes/element, and prefill — whose six GDN kernel
/// families are FP32 writers and stay that way — runs over a per-slot FP32
/// STAGING blob that the layer widens into and narrows back
/// (`spark_model::layers::qwen3_ssm::ssm_h_fp16`). The refusal list below is
/// UNCHANGED by that, because the narrowing is transparent to every
/// combination in it: what makes a kernel unsafe here is reading FP16 bits
/// as FP32, not how wide the slot holding them is.
pub(super) fn ssm_h_fp16_preconditions(args: &cli::ServeArgs, config: &ModelConfig) -> Result<()> {
    // SSOT: the same resolution the kernels dispatch on — `--ssm-h-dtype`,
    // falling back to `ATLAS_SSM_H_FP16`. This check used to decode the
    // environment independently, which is how a preflight could pass on a
    // reading the kernels did not share.
    if !spark_model::layers::qwen3_ssm::ssm_h_fp16_enabled() || config.num_ssm_layers() == 0 {
        return Ok(());
    }
    // ── `--dflash` (γ=17) ──
    //
    // DFlash dispatches the verify chain at γ+1 = 17 rows, which lands on
    // `gated_delta_rule_wy17` — an FP32-only kernel with no FP16 h-state twin,
    // AND the one WY family that takes an explicit FP32-element intermediate
    // stride. Both halves are wrong over an FP16 h-state, and neither faults:
    // the same fluent-garbage failure the `--num-drafts > 3` refusal below
    // exists for. It applies to stage 1/2 f16 as much as to the f16-SIZED
    // pool, which is why it sits above the stage-3 branch.
    // NARROWED 2026-08-30. This was a blanket refusal of `--dflash`, written
    // when the ONLY DFlash verify width was gamma+1 = 17 and wy17 had no FP16
    // twin. #817 added `gated_delta_rule_wy{5..16}_f16`, so every width up to
    // K=16 now has one — but #817 updated only the CLI validator and left this
    // refusal standing, which made its own headline change unreachable: a
    // `--dflash --ssm-h-dtype f16-pool` serve died here, on #817's branch as
    // much as on this one. Measured 2026-08-30: three gamma values, three
    // refusals, zero tokens.
    //
    // What is still refused is the width with no twin. K = gamma + 1, so the
    // last covered gamma is MAX_F16_TWIN_DFLASH_GAMMA (15); gamma 16 reaches
    // wy17 and must not run under an FP16 pool.
    //
    // An UNSET gamma is allowed through: it resolves from the drafter
    // checkpoint, which is not readable at preflight, and the runtime is
    // fail-closed for it — a width whose twin handle is zero returns None from
    // `wyn_kernel`, lands on the sequential fallback, and that fallback bails
    // under f16 (trait_decode_batched_conv_gdn.rs) rather than reading FP16
    // bits as FP32. Guessing here would refuse working configurations; the
    // backstop refuses broken ones.
    if args.dflash
        && args
            .dflash_gamma
            .is_some_and(|g| g > spark_model::layers::qwen3_ssm::MAX_F16_TWIN_DFLASH_GAMMA)
    {
        anyhow::bail!(
            "--ssm-h-dtype f16 with --dflash-gamma {} : the verify width is gamma + 1 = {}, \
             which dispatches gated_delta_rule_wy17 — the one WY family with no FP16 h-state \
             twin, and the one that strides its h intermediates in FP32 elements. An FP32 \
             kernel over an FP16 h-state emits fluent garbage rather than an error. Use \
             --dflash-gamma <= {}, drop --dflash, or use --ssm-h-dtype f32.",
            args.dflash_gamma.unwrap_or_default(),
            args.dflash_gamma.unwrap_or_default() + 1,
            spark_model::layers::qwen3_ssm::MAX_F16_TWIN_DFLASH_GAMMA,
        );
    }
    // STAGE 2 lifted the blanket refusal on `--speculative`: the MTP verify
    // path's WY kernels now have FP16 h-state twins
    // (`gated_delta_rule_wy{2,3,4}_f16` and the register-resident
    // `wy{2,3}_resident_f16`), so the h-state stays FP16 end-to-end through a
    // verify step. What is still refused is any configuration that can reach a
    // K with NO twin, because the fallback is an FP32 kernel over an FP16 pool
    // — which does not fault, it emits fluent garbage.
    //
    // The reachable K is bounded by the draft count: the ladder's draft count
    // is capped by `--num-drafts`, and K = drafts + 1 rows per sequence. Twins
    // exist for K = 2, 3, 4, so up to 3 drafts is supported. Above that the
    // width lands on the wyN (K=5..8) or wy17 DFlash arms, which are FP32-only.
    if args.self_speculative || args.ngram_speculative {
        anyhow::bail!(
            "--ssm-h-dtype f16 supports --speculative (MTP) only. The self-speculative and \
             ngram-speculative verify paths still write the h-state as FP32, and an FP32 \
             kernel over an FP16 pool produces fluent garbage rather than an error. Run \
             without --self-speculative/--ngram-speculative, or use --ssm-h-dtype f32."
        );
    }
    if args.speculative && args.resolved_num_drafts() > 3 {
        anyhow::bail!(
            "--ssm-h-dtype f16 supports up to 3 drafts (K <= 4 verify rows); --num-drafts is \
             {}. Wider verify widths dispatch the wyN (K=5..8) / wy17 arms, which have no \
             FP16 h-state twin. Lower --num-drafts to 3, or use --ssm-h-dtype f32.",
            args.resolved_num_drafts()
        );
    }
    if !spark_model::layers::qwen3_ssm::gdn_fused_norm_enabled() {
        anyhow::bail!(
            "--ssm-h-dtype f16 requires --gdn-fused-norm — the non-fused decode arms \
             (gated_delta_rule_decode, ..._decode_f32_strided) have no FP16 twin in stage 1."
        );
    }
    if std::env::var("ATLAS_GDN_FUSED_CONV").ok().as_deref() == Some("1") {
        anyhow::bail!(
            "--ssm-h-dtype f16 is incompatible with ATLAS_GDN_FUSED_CONV=1 —              gated_delta_rule_decode_f32_conv_norm has no FP16 twin in stage 1."
        );
    }
    if config.linear_key_head_dim != 128 || config.linear_value_head_dim != 128 {
        anyhow::bail!(
            "--ssm-h-dtype f16 needs linear head dims 128/128 (the FP16 twins size their shared              memory for k_dim == 128); this model is {}/{}",
            config.linear_key_head_dim,
            config.linear_value_head_dim
        );
    }
    if spark_model::layers::qwen3_ssm::ssm_h_f16_pool_enabled() {
        tracing::info!(
            "--ssm-h-dtype f16-pool: GDN h-state stored FP16 AND every h pool SIZED at 2 \
             bytes/element. Prefill runs its unchanged FP32 kernels over a per-slot FP32 \
             staging blob and narrows back, so the pool holds FP16 at all times. NUMERICS: \
             the recurrence now carries FP16 state across prefill chunk boundaries as well \
             as decode steps, with round-to-nearest-even and no stochastic rounding — do NOT \
             publish a number from this mode without ssm-state-poisoning-gate, decode-floor, \
             bfcl-subset and the agentic gate."
        );
    } else {
        tracing::info!(
            "--ssm-h-dtype f16: GDN h-state stored FP16 during decode AND MTP verify (pool \
             stays FP32-sized; prefill unchanged). Scan replica at n=128: 183 -> 84 ms/step."
        );
    }
    Ok(())
}
