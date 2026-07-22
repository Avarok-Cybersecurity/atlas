// SPDX-License-Identifier: AGPL-3.0-only

//! MTP speculative draft proposal step.

use super::*;

/// MTP-aware step: bootstrap sequences without drafts, then verify via CUDA graph.
/// Supports K=2 (num_drafts=1) and K=3 (num_drafts=2).
///
/// `verify_ctx` carries the tokenizer special-token IDs the verify
/// pipeline needs (`<think>` / `</think>` / `<tool_call>` /
/// `</tool_call>`). Threaded down to every verify call site so the
/// 8-stage [`crate::scheduler::logit_processors`] pipeline can run on
/// each verify-position's logits — the fix for MTP-emitted tokens
/// bypassing all pre-sample masks. See `verify_pipeline_helper`.
pub fn step_mtp(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    num_drafts: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    dflash_verify_raw_argmax: bool,
) {
    let mut bootstrap_idxs: Vec<usize> = Vec::new();
    let mut verify_idxs: Vec<usize> = Vec::new();
    for (i, a) in active.iter().enumerate() {
        if !a.pending_drafts.is_empty() {
            verify_idxs.push(i);
        } else {
            bootstrap_idxs.push(i);
        }
    }

    // ── Phase A: Bootstrap decode for sequences without a draft ──
    if !bootstrap_idxs.is_empty() {
        // The previous verify commit's live-state restore runs async on the
        // secondary stream; order it before the bootstrap decode reads
        // h_state/conv_state (and before start_checkpoint_async snapshots
        // the live state). GPU-side event wait, zero CPU cost.
        if let Err(e) = model.sync_secondary() {
            tracing::error!("bootstrap sync_secondary: {e:#}");
        }
    }
    for &idx in &bootstrap_idxs {
        let a = &mut active[idx];

        // DFlash path: skip the standalone M=1 decode. The fused pass already
        // computes every position's logit in one weight sweep, so the "next
        // decoded token" is the bonus token at result[num_accepted] — the logit
        // at the position immediately after the accepted prefix (§8 vLLM
        // bonus-token pattern). Propose initial drafts using the DFlash hidden
        // already captured at row 0 by the previous step's fused pass (or
        // prefill), then route through step_verify_k3/k2 which handles the
        // fused forward, accept/reject, bonus-token emit, and re-propose for
        // the next step. This replaces the two-sweep sequence (M=1 decode here
        // + M=1+k fused in Phase B) with a single M=1+k fused sweep.
        if dflash_verify_raw_argmax
            && !crate::scheduler::verify_pipeline_helper::dflash_seam_serial_enabled()
            && crate::scheduler::adaptive_spec::spec_allowed(a)
        {
            let eff = if a.grammar_state.is_some() {
                1
            } else {
                num_drafts
            };
            let _gmask = mtp_grammar_mask_for(a);
            match model.run_mtp_propose_multi(
                a.last_token,
                a.seq.seq_len,
                eff,
                &mut a.seq,
                0,
                _gmask.as_deref(),
            ) {
                Ok(init) if !init.is_empty() => {
                    if eff >= 3 && init.len() >= 3 {
                        step_verify_k4(
                            model,
                            a,
                            &init,
                            num_drafts,
                            verify_ctx,
                            dflash_verify_raw_argmax,
                        );
                    } else if eff >= 2 && init.len() >= 2 {
                        step_verify_k3(
                            model,
                            a,
                            &init,
                            num_drafts,
                            verify_ctx,
                            dflash_verify_raw_argmax,
                        );
                    } else {
                        step_verify_k2(
                            model,
                            a,
                            &init,
                            num_drafts,
                            verify_ctx,
                            dflash_verify_raw_argmax,
                        );
                    }
                    continue;
                }
                Ok(_) => {
                    tracing::warn!(
                        "DFlash bootstrap propose returned empty; falling back to standalone decode"
                    );
                }
                Err(e) => {
                    tracing::error!("DFlash bootstrap propose: {e:#}");
                }
            }
            // Rare fallback: propose failed or returned empty (e.g. drafter not
            // yet primed). Fall through to the standalone decode below so the
            // sequence emits its next token rather than stalling.
        }

        // Non-DFlash path (or DFlash-propose fallback): EP broadcast + standalone decode.
        // EP: broadcast token to worker before decode (worker runs decode in lockstep).
        if let Err(e) = model.ep_broadcast_cmd_for_seq(a.seq.slot_idx as u32, a.last_token) {
            tracing::error!("EP broadcast bootstrap token: {e:#}");
            a.finished = true;
            continue;
        }
        let logits = match model.decode(a.last_token, &mut a.seq, 0) {
            Ok(l) => l,
            Err(e) => {
                tracing::error!("bootstrap decode error: {e:#}");
                a.finished = true;
                continue;
            }
        };
        // Build the seq's configured penalties (rep/presence/frequency/LZ/DRY)
        // so the MTP bootstrap token sees the SAME penalties+history the
        // non-MTP path applies — the root-cause fix for repetition_penalty /
        // dry_multiplier never reaching MTP-emitted tokens. Cloned before the
        // mutable `grammar_state` borrow to satisfy the borrow checker.
        let penalties = crate::scheduler::sample_step::penalty_params_for(
            a,
            crate::scheduler::sample_step::PositionKind::Verify,
            0.0,
            None,
            Vec::new(),
        );
        // #192: same per-tool-call-segment scoping as the main pipeline
        // (`penalty_history_scope`) so MTP bootstrap tokens see the identical
        // penalty landscape.
        let history = crate::scheduler::sample_step::penalty_history_scope(
            &a.output_tokens,
            a.tool_call_end_token,
        )
        .to_vec();
        // P1-4 (2026-07-09): the bootstrap token is one of only two
        // stochastic sample points under MTP, and its stochastic branch
        // previously sampled with a hardcoded `min_p: 0.0` deep inside
        // `sample_token_with_grammar` — bypassing the MODEL.toml
        // `min_p_floor` (0.05 on this family) that exists precisely to stop
        // FP8/NVFP4 argmax-flip tail tokens. The sampler now reads
        // `penalties.min_p`, which `penalty_params_for` copies from
        // `a.min_p` (request value + floor, resolved in `sampling_setup`) —
        // SSOT, no new channel. Kill-switch: ATLAS_NO_MTP_MINP=1.
        let tok = match sample_token_with_grammar(
            model,
            logits,
            a.temperature,
            a.top_k,
            a.top_p,
            &[],
            a.grammar_state.as_mut(),
            &penalties,
            &history,
        ) {
            Ok(t) => t,
            Err(e) => {
                tracing::error!("bootstrap sample error: {e:#}");
                a.finished = true;
                continue;
            }
        };

        // Extract logprobs from bootstrap decode logits (single position).
        let lp = if let Some(k) = a.top_logprobs {
            extract_single_logprobs(model, logits, tok, k)
        } else {
            None
        };

        emit_token(a, tok, lp);
        if a.finished {
            continue;
        }
        a.last_token = tok;
        // Adaptive speculation: count serial tokens toward the re-probe window.
        crate::scheduler::adaptive_spec::tick_serial(a);

        // Ctx-holes fix (ATLAS_DFLASH_SERIAL_APPEND=1), COMPLEMENT-GATED:
        // the serial ctx-append fires iff propose() will NOT run this
        // iteration, so append and propose decode-append can never both
        // cover one token — double-append impossible by construction
        // (that was the cuMemcpyDtoDAsync status-1 crash).
        // `spec_allowed` is evaluated exactly once (it mutates re-probe
        // state); its verdict is reused for the propose gate below.
        // Exception — re-probe RESUME: the token decoded on the un-suspend
        // iteration would otherwise fall in a hole (the stale
        // `skip_next_decode_append` set by the last suspended token makes
        // the propose below skip its decode-append). Append it here; the
        // skip flag this sets is consumed by that propose — one append,
        // no duplicate, seam covered.
        let was_suspended = crate::scheduler::adaptive_spec::is_suspended(a);
        let will_propose = crate::scheduler::adaptive_spec::spec_allowed(a);
        let reprobe_resume = was_suspended && will_propose;
        if crate::scheduler::adaptive_spec::unified_ctx_enabled() {
            // Unified ctx commit: same complement-gate as the old serial
            // append — fire iff propose() will NOT run (or re-probe resume),
            // so commit and propose decode-append never both cover a token.
            if !will_propose || reprobe_resume {
                let base_pos = a.seq.seq_len.saturating_sub(1);
                if let Err(e) = model.commit_ctx(&mut a.seq, 1, base_pos) {
                    tracing::error!("commit_ctx (mtp serial): {e:#}");
                }
            }
        } else if crate::scheduler::adaptive_spec::serial_append_enabled()
            && (!will_propose || reprobe_resume)
            && let Err(e) = model.dflash_serial_ctx_append(&mut a.seq)
        {
            tracing::error!("dflash_serial_ctx_append: {e:#}");
        }

        if let Err(e) = model.save_hidden_for_mtp(0, 0) {
            tracing::error!("save_hidden_for_mtp: {e:#}");
            continue;
        }
        let _mtp_grammar_mask = mtp_grammar_mask_for(a);
        // BUG#4 (2026-06-02): when a grammar is active, generate only ONE draft.
        // run_mtp_propose_multi (mtp_multi.rs) masks only draft[0] with the
        // position-0 bitmask and leaves draft[1..] UNMASKED, so multi-draft +
        // grammar desyncs — a draft[1] token can violate its true per-position
        // mask, get verified+accepted, then be refused by the matcher later
        // (→ truncation). A single draft uses its own up-to-date mask and is
        // sound; drafts.len()==1 routes verify to the K=2 path. Mask is a no-op
        // when grammar is inactive, so NVFP4/non-tool paths keep full K.
        // 2026-07-09: hoisted to the `effective_drafts_under_grammar` SSOT,
        // now also applied at the five verify-path re-propose sites that
        // previously bypassed this clamp (the "mask held fixed" warn spam).
        let effective_num_drafts =
            crate::scheduler::spec_step::effective_drafts_under_grammar(a, num_drafts);
        // Adaptive speculation: a suspended seq skips proposing entirely and
        // stays on this serial bootstrap path until the re-probe fires.
        // (`will_propose` is the single spec_allowed evaluation above.)
        if will_propose {
            match model.run_mtp_propose_multi(
                tok,
                a.seq.seq_len,
                effective_num_drafts,
                &mut a.seq,
                0,
                _mtp_grammar_mask.as_deref(),
            ) {
                Ok(drafts) if !drafts.is_empty() => {
                    tracing::debug!("MTP bootstrap: tok={tok} → drafts={drafts:?}");
                    a.pending_drafts = drafts;
                }
                Ok(_) => tracing::warn!("MTP propose returned empty"),
                Err(e) => {
                    tracing::error!("run_mtp_propose_multi: {e:#}");
                }
            }
        }

        if let Err(e) = model.start_checkpoint_async(&mut a.seq) {
            tracing::error!("bootstrap start_checkpoint_async: {e:#}");
        }
    }

    // ── Phase B: Verify with pipelined checkpoint ──
    for &idx in &verify_idxs {
        let a = &mut active[idx];
        let mut drafts: Vec<u32> = std::mem::take(&mut a.pending_drafts);
        if drafts.is_empty() {
            continue;
        }

        // Spec-decode boundary awareness (arXiv:2512.15834): when a
        // grammar is active, validate the draft sequence against the
        // matcher and truncate at the first token that crosses a
        // grammar transition. Without this, a draft span that crosses
        // `</function>` (or any other structural boundary) gets
        // accepted by the verifier and emitted, but the post-emit
        // `accept_token` silently fails — desync'ing the grammar
        // from the output stream. Truncating here downgrades K=4 →
        // K=3 → K=2 cleanly.
        if let Some(ref mut gs) = a.grammar_state {
            let kept = truncate_drafts_at_grammar_boundary(gs, &drafts);
            if kept < drafts.len() {
                drafts.truncate(kept);
            }
            if drafts.is_empty() {
                continue;
            }
        }

        // DFlash γ-block drafters return ≥4 drafts per step (γ=16 typical).
        // The K=2/3/4 graphed paths are MTP-shaped and don't generalize past
        // K=4 cleanly, so γ-block verify routes through `step_verify_dflash`.
        // MTP keeps using the existing graphed paths; this dispatch is purely
        // additive.
        if drafts.len() >= 4 {
            step_verify_dflash(
                model,
                a,
                &drafts,
                num_drafts,
                verify_ctx,
                dflash_verify_raw_argmax,
            );
        } else if num_drafts >= 3 && drafts.len() >= 3 {
            step_verify_k4(
                model,
                a,
                &drafts,
                num_drafts,
                verify_ctx,
                dflash_verify_raw_argmax,
            );
        } else if num_drafts >= 2 && drafts.len() >= 2 {
            step_verify_k3(
                model,
                a,
                &drafts,
                num_drafts,
                verify_ctx,
                dflash_verify_raw_argmax,
            );
        } else {
            step_verify_k2(
                model,
                a,
                &drafts,
                num_drafts,
                verify_ctx,
                dflash_verify_raw_argmax,
            );
        }
    }
}
