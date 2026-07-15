// SPDX-License-Identifier: AGPL-3.0-only

//! Scheduler: batched concurrent decode on a single GPU thread.
//! Architecture:
//! - Receiver thread: blocks on request channel, pushes to pending queue,
//!   signals condvar (instantaneous wake, zero polling).
//! - Scheduler thread: prefills new requests sequentially, then runs
//!   batched decode steps via `model.decode_batch()` — weights loaded once
//!   per step for all active sequences.
//!
//! When idle (no active sequences): blocks on condvar (zero CPU).
//! When busy: drains pending queue (mutex lock) after each decode step.

// ── Submodules (split for ≤500 LoC files) ──────────────────────────────────
mod adaptive_spec;
mod confidence;
mod decode_logits_content;
mod decode_logits_seq;
mod decode_logits_step;
mod decode_step;
mod emit_step;
mod fast_greedy;
mod helpers;
mod lifecycle;
mod logit_dump;
mod logit_processors;
mod logprobs;
mod mod_helpers;
mod mtp_gate;
mod mtp_step;
pub(crate) mod mtp_timing;
mod phase_continue_prefills;
mod phase_promote_prefills;
mod phase_start_prefills;
mod prefill_a_step;
mod prefill_a_step_params;
mod prefill_b_step;
mod repetition;
mod rollback;
mod sample_step;
mod spec_step;
mod ssm_decode_ring;
mod types;
mod verify_dflash_step;
mod verify_k2_step;
mod verify_k3_step;
mod verify_k4_step;
mod verify_pipeline_helper;

use confidence::*;
use decode_logits_content::*;
use decode_logits_seq::*;
use decode_logits_step::*;
use decode_step::*;
use emit_step::*;
pub use helpers::disable_watchdogs;
pub use helpers::set_boundary_token_mask;
pub use helpers::set_enable_loop_watchdog;
pub use helpers::set_im_start_hard_stop;
pub use helpers::set_mid_word_token_mask;
pub use helpers::set_numeric_token_mask;
pub use helpers::set_tool_response_hard_stop;
use helpers::*;
pub use helpers::{CONTENT_LOOP_PERIOD_MAX, CONTENT_LOOP_PERIOD_MIN};
pub use helpers::{WatchdogParams, set_watchdog_params};
use lifecycle::*;
use logprobs::*;
use mod_helpers::*;
use mtp_step::*;
use phase_continue_prefills::continue_in_progress_prefills;
use phase_start_prefills::start_new_requests;
use prefill_a_step::*;
use prefill_b_step::*;
use repetition::*;
use rollback::{RollbackOutcome, rollback_to_boundary};
use sample_step::*;
use spec_step::*;
use ssm_decode_ring::SsmDecodeRing;
use types::*;
use verify_dflash_step::*;
use verify_k2_step::*;
use verify_k3_step::*;
use verify_k4_step::*;
// verify_pipeline_helper is referenced via fully-qualified
// `crate::scheduler::verify_pipeline_helper::...` from sibling step
// files (verify_k2/k3/k4/dflash + spec_step), so no `use` import.

// Re-exports threaded through `use super::*;` in sibling step files —
// keep these imports here even though `run` itself doesn't reference all
// of them directly (see scheduler/decode_step.rs etc.).
use anyhow::Result;
use parking_lot::{Condvar, Mutex};
use spark_model::traits::{Model, SequenceState};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_spill::KvSpillManager;
use spark_runtime::sampler::{
    SamplingParams, apply_penalties_and_bias, sample_with_params, sample_with_params_history,
};

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use crate::api::{GrammarSpec, InferenceRequest, InferenceResponse, StreamEvent};
use crate::grammar::{GrammarEngine, GrammarState};
use crate::ngram::NgramProposer;
use crate::scheduling_policy::SchedulingPolicy;

/// Run the scheduler loop on the current thread.
#[allow(clippy::too_many_arguments)]
pub fn run(
    model: Box<dyn Model>,
    request_rx: tokio::sync::mpsc::Receiver<InferenceRequest>,
    eos_tokens: Vec<u32>,
    max_batch_size: usize,
    use_speculative: bool,
    dflash_verify_raw_argmax: bool,
    num_drafts: usize,
    policy: Box<dyn SchedulingPolicy>,
    max_prefill_tokens: usize,
    max_batch_tokens: usize,
    use_self_speculative: bool,
    use_ngram_speculative: bool,
    swap_space_gb: usize,
    high_speed_swap_cfg: Option<spark_storage::HighSpeedSwapConfig>,
    block_size: usize,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    code_fence_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    mut grammar_engine: Option<GrammarEngine>,
    adaptive_sampling: bool,
    mut session_manager: crate::session_manager::SessionSsmManager,
    spontaneous_think_budget: u32,
) {
    model
        .bind_gpu_to_thread()
        .expect("Failed to bind CUDA context to scheduler thread");
    let use_mtp = use_speculative && model.has_proposer();
    let num_drafts = if use_mtp || use_self_speculative || use_ngram_speculative {
        num_drafts.max(1)
    } else {
        0
    };
    let chunked = max_prefill_tokens > 0;
    // Throughput-aware MTP gate: when MTP is requested, measure the verify-step
    // cost multiplier over the first decode steps of the first lone-sequence
    // session and auto-disable MTP if it is provably net-negative. Only armed
    // for the pure-MTP path (not ngram/self/dflash, which have their own
    // economics and proposers).
    let mut mtp_gate = if use_mtp && !mtp_timing::gate_forced() {
        Some(mtp_gate::MtpGate::new(num_drafts))
    } else {
        if use_mtp && mtp_timing::gate_forced() {
            tracing::warn!(
                "ATLAS_MTP_GATE_FORCE=1: MTP throughput gate DISARMED (diagnostic; \
                 verify runs even where the gate would measure it net-negative)"
            );
        }
        None
    };
    let mut ngram_proposer = if use_ngram_speculative {
        Some(NgramProposer::new(4)) // 4-gram context
    } else {
        None
    };
    tracing::info!(
        "Scheduler started (batched mode, max_batch={max_batch_size}, mtp={}, ngram={}, num_drafts={num_drafts}, policy={}, chunked_prefill={}, max_prefill_tokens={})",
        use_mtp,
        use_ngram_speculative,
        policy.name(),
        chunked,
        if chunked { max_prefill_tokens } else { 0 },
    );

    // Holo "always-on fused mixed step" gate (default OFF). When OFF the
    // scheduler behaves EXACTLY as today (binary should_prefill, no slice
    // budget). When ON, an active decode + an in-progress prefill always
    // takes a fused mixed step sized by the policy's prefill_slice_budget
    // so decode never starves during a prefill burst. Read once at startup.
    let always_mixed = std::env::var("ATLAS_HOLO_ALWAYS_MIXED")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if always_mixed {
        tracing::info!("ATLAS_HOLO_ALWAYS_MIXED=on: fused mixed step always-on (slice-budget)");
    }

    let pending = Arc::new((
        Mutex::new(PendingQueue {
            requests: Vec::new(),
            closed: false,
        }),
        Condvar::new(),
    ));

    // Receiver thread: blocks on channel, signals scheduler via condvar.
    let p = Arc::clone(&pending);
    std::thread::spawn(move || {
        let mut rx = request_rx;
        while let Some(req) = rx.blocking_recv() {
            p.0.lock().requests.push(req);
            p.1.notify_one();
        }
        p.0.lock().closed = true;
        p.1.notify_one();
    });

    // Dedicated CUDA stream + event for prefill compute-copy overlap.
    let prefill_stream = model
        .create_stream()
        .expect("Failed to create prefill CUDA stream");
    let prefill_event = model
        .create_event()
        .expect("Failed to create prefill CUDA event");

    let mut active: Vec<ActiveSeq> = Vec::new();
    let mut prefilling: Vec<PrefillInProgress> = Vec::new();
    let mut swapped: Vec<SwappedSeq> = Vec::new();
    let mut spill_manager: Option<KvSpillManager> = if swap_space_gb > 0 {
        let max_bytes = swap_space_gb as u64 * 1024 * 1024 * 1024;
        match KvSpillManager::new(PathBuf::from("/tmp/atlas-swap"), max_bytes) {
            Ok(mgr) => {
                tracing::info!("Swap space: {swap_space_gb} GB at /tmp/atlas-swap/");
                Some(mgr)
            }
            Err(e) => {
                tracing::error!("Failed to initialize swap space: {e:#}");
                None
            }
        }
    } else {
        None
    };

    install_high_speed_swap(&*model, high_speed_swap_cfg);

    loop {
        // ── Drain pending → start prefill (chunked or full) ──
        let new_reqs =
            drain_pending_requests(&pending, &active, &prefilling, &*policy, max_batch_size);
        if new_reqs.is_empty() && active.is_empty() && prefilling.is_empty() {
            // Receiver thread was closed (shutdown).
            let pending_closed = pending.0.lock().closed;
            if pending_closed {
                break;
            }
        }

        // ── Swap-out: evict active sequences to disk when blocks run low ──
        if let Some(ref mut spill) = spill_manager {
            for req in &new_reqs {
                let prompt_len = req.prompt_len();
                let blocks_needed = prompt_len / block_size + 1;
                while model.num_free_blocks() < blocks_needed && !active.is_empty() {
                    let victim_idx = active
                        .iter()
                        .enumerate()
                        .filter(|(_, a)| a.grammar_state.is_none())
                        .max_by_key(|(_, a)| a.seq.block_table.len())
                        .map(|(i, _)| i);
                    let Some(victim_idx) = victim_idx else {
                        tracing::warn!("No swappable sequences (all grammar-active)");
                        break;
                    };
                    match swap_out_sequence(&*model, &mut active, victim_idx, spill) {
                        Ok(s) => {
                            tracing::info!(
                                "Swap-out: evicted seq (seq_len={}, blocks={}) to disk",
                                s.seq_len,
                                s.num_blocks,
                            );
                            swapped.push(s);
                        }
                        Err(e) => {
                            tracing::error!("Swap-out failed: {e:#}");
                            break;
                        }
                    }
                }
            }
        }

        // ── Start new requests ──
        start_new_requests(
            &*model,
            new_reqs,
            chunked,
            always_mixed,
            max_prefill_tokens,
            max_batch_tokens,
            &eos_tokens,
            prefill_stream,
            prefill_event,
            &mut grammar_engine,
            spontaneous_think_budget,
            think_end_token,
            think_start_token,
            tool_call_start_token,
            tool_call_end_token,
            &mut active,
            &mut prefilling,
        );

        // ── Continue in-progress prefills ──
        let did_mixed_step = continue_in_progress_prefills(
            &*model,
            &*policy,
            &mut active,
            &mut prefilling,
            max_prefill_tokens,
            max_batch_tokens,
            always_mixed,
            prefill_stream,
            prefill_event,
            use_mtp,
            use_self_speculative,
            use_ngram_speculative,
            think_end_token,
            think_start_token,
            code_fence_token,
            tool_call_start_token,
            tool_call_end_token,
            adaptive_sampling,
        );

        if active.is_empty() {
            continue;
        }

        // Skip decode when mixed_forward already processed decode logits.
        if !did_mixed_step {
            // Ensure any in-flight prefill work on the prefill stream is complete
            // before decode starts on the default stream.
            if !prefilling.is_empty() {
                let _ = model.record_event(prefill_event, prefill_stream);
                let _ = model.stream_wait_event(model.default_stream(), prefill_event);
            }

            // Build the verify-time LogitsContext once per step: the
            // tokenizer special-token IDs the verify pipeline needs to
            // run the same 8-stage logits processors the non-MTP path
            // applies (mid-word/post-close/tool-during-think/forced-
            // think-end/pin-tool-call/forced-token/grammar). Without
            // this context the MTP/spec verify path emits unmasked
            // GPU-argmax tokens (Phase C-2 root cause, 2026-05-24).
            let verify_ctx = crate::scheduler::logit_processors::LogitsContext {
                think_end_token,
                think_start_token,
                tool_call_start_token,
                tool_call_end_token,
            };
            // Spec-resume guard (ATLAS_DFLASH_RESUME_GUARD=N, default 0 = off):
            // keep the first N post-`</think>` tokens on plain serial decode.
            // The T=0 verify-vs-decode low-margin flips measured 2026-07-07
            // concentrate in the answer's opening tokens; serial-decoding that
            // window sidesteps them while leaving the high-accept answer body
            // speculated. N=0 preserves exact prior behavior.
            static DFLASH_RESUME_GUARD: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
            let dflash_resume_guard = *DFLASH_RESUME_GUARD.get_or_init(|| {
                std::env::var("ATLAS_DFLASH_RESUME_GUARD")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0)
            });
            // ATLAS_DFLASH_SPEC_THINK=1: speculate INSIDE think blocks (vLLM
            // semantics — reference measures 45% draft acceptance on thinking,
            // 2026-07-07 calibration). Bypasses the think-gate AND the resume
            // guard: output is coherent but not byte-lossless vs no-spec (the
            // batch-K numerics floor can flip a low-margin token mid-think),
            // and thinking-budget forced-end is not enforced on the raw-argmax
            // verify path. Throughput mode; leave OFF for byte-proof runs.
            static DFLASH_SPEC_THINK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let dflash_spec_think = *DFLASH_SPEC_THINK.get_or_init(|| {
                std::env::var("ATLAS_DFLASH_SPEC_THINK").ok().as_deref() == Some("1")
            });
            if use_ngram_speculative && active.len() == 1 && active[0].grammar_state.is_none() {
                // N-gram speculative: CPU proposer + CUDA-graphed K=2 verify.
                if let Some(ref mut proposer) = ngram_proposer {
                    step_ngram(&*model, &mut active, proposer, &verify_ctx);
                }
            } else if use_self_speculative && active.len() == 1 && active[0].grammar_state.is_none()
            {
                // Self-speculative: draft via layer-skipping, verify with full model.
                step_self_spec(&*model, &mut active, num_drafts, &verify_ctx);
            } else if use_mtp
                && active.len() == 1
                && (
                    // SPEC_THINK: speculate everywhere EXCEPT the first
                    // `dflash_resume_guard` generated tokens — every observed
                    // T=0 flip (2026-07-07/08) fires within ~7 tokens of spec
                    // ENTRY (sequence start or post-think resume); serial-
                    // decoding the entry window dodges the divergence while
                    // leaving the body speculated.
                    (dflash_spec_think
                        && active[0].output_tokens.len() as u32 >= dflash_resume_guard)
                        || (!active[0].inside_thinking
                            && active[0].post_think_emitted >= dflash_resume_guard)
                )
                && !active[0].suppress_tool_call
                && !active[0].disable_mtp
            {
                // Throughput-aware MTP gate: while still measuring, run the
                // step type the gate asks for and time it. Both step types emit
                // real, correct tokens (MTP verify and plain decode are greedy-
                // equivalent), so the measurement window does not waste work.
                //
                // The verify/decode multiplier is DEPTH-DEPENDENT (weight-bound
                // at short context vs KV/SSM-bound at depth — see mtp_gate module
                // docs), so a decision is only valid for the regime it was
                // measured in: re-open measurement when the live depth leaves it.
                if let Some(gate) = mtp_gate.as_mut() {
                    gate.maybe_remeasure(active[0].seq.seq_len);
                }
                if let Some(gate) = mtp_gate.as_mut().filter(|g| g.is_measuring()) {
                    gate.note_depth(active[0].seq.seq_len);
                    match gate.next_step() {
                        mtp_gate::GateStep::MeasureDecode => {
                            let t0 = std::time::Instant::now();
                            step_decode_only(
                                &*model,
                                &mut active,
                                think_end_token,
                                think_start_token,
                                code_fence_token,
                                tool_call_start_token,
                                tool_call_end_token,
                                adaptive_sampling,
                            );
                            mtp_gate
                                .as_mut()
                                .expect("gate present")
                                .record_decode(t0.elapsed());
                        }
                        mtp_gate::GateStep::MeasureVerify => {
                            // Only a true verify forward (drafts already
                            // proposed) is a representative sample; a bootstrap-
                            // only step proposes the first draft and is skipped.
                            let had_draft = !active[0].pending_drafts.is_empty();
                            let t0 = std::time::Instant::now();
                            step_mtp(
                                &*model,
                                &mut active,
                                num_drafts,
                                &verify_ctx,
                                dflash_verify_raw_argmax,
                            );
                            if had_draft {
                                mtp_gate
                                    .as_mut()
                                    .expect("gate present")
                                    .record_verify(t0.elapsed());
                            }
                        }
                    }
                    // One-time transition work for a freshly-reached DISABLE:
                    // drop pending drafts and order the draft-head state resync
                    // before the next plain decode reads it. NOT permanent —
                    // `maybe_remeasure` above re-opens measurement when the
                    // depth regime changes, so MTP can come back exactly where
                    // it pays (deep agentic contexts).
                    if mtp_gate
                        .as_mut()
                        .and_then(mtp_gate::MtpGate::take_fresh_decision)
                        == Some(mtp_gate::GateDecision::DisableMtp)
                    {
                        for a in active.iter_mut() {
                            a.pending_drafts.clear();
                        }
                        if let Err(e) = model.sync_secondary() {
                            tracing::error!("mtp-gate→decode sync_secondary: {e:#}");
                        }
                    }
                } else if mtp_gate
                    .as_ref()
                    .is_some_and(|g| g.decision() == Some(mtp_gate::GateDecision::DisableMtp))
                {
                    // Measured net-negative in the CURRENT depth regime: plain
                    // decode. Consulted every step (not a latched kill switch)
                    // so the regime re-measurement above can flip it back.
                    step_decode_only(
                        &*model,
                        &mut active,
                        think_end_token,
                        think_start_token,
                        code_fence_token,
                        tool_call_start_token,
                        tool_call_end_token,
                        adaptive_sampling,
                    );
                } else {
                    // MTP wins in this regime (or no gate): speculative decode.
                    step_mtp(
                        &*model,
                        &mut active,
                        num_drafts,
                        &verify_ctx,
                        dflash_verify_raw_argmax,
                    );
                }
            } else {
                // Batch decode (no MTP). Clear stale drafts when transitioning out of MTP mode.
                if use_mtp {
                    for a in active.iter_mut() {
                        a.pending_drafts.clear();
                    }
                    // MTP→decode-only transition: the last verify commit's
                    // live-state restore runs async on the secondary stream;
                    // order it before this decode reads h_state/conv_state
                    // (GPU-side event wait, zero CPU cost).
                    if let Err(e) = model.sync_secondary() {
                        tracing::error!("mtp→decode sync_secondary: {e:#}");
                    }
                }
                step_decode_only(
                    &*model,
                    &mut active,
                    think_end_token,
                    think_start_token,
                    code_fence_token,
                    tool_call_start_token,
                    tool_call_end_token,
                    adaptive_sampling,
                );
            }
        }

        retire_finished_sequences(&*model, &mut active);

        // ── Swap-in: resume swapped sequences when blocks free up ──
        if let Some(ref mut spill) = spill_manager {
            let mut resumed_any = true;
            while resumed_any && !swapped.is_empty() && active.len() < max_batch_size {
                resumed_any = false;
                let free = model.num_free_blocks();
                if let Some(idx) = swapped.iter().position(|s| s.num_blocks <= free) {
                    let s = swapped.remove(idx);
                    match resume_swapped_seq(think_end_token, think_start_token, &*model, s, spill)
                    {
                        Ok(a) => {
                            tracing::info!(
                                "Swap-in: restored seq (seq_len={}, blocks={})",
                                a.seq.seq_len,
                                a.seq.block_table.len(),
                            );
                            active.push(a);
                            resumed_any = true;
                        }
                        Err(e) => {
                            tracing::error!("Swap-in failed: {e:#}");
                        }
                    }
                }
            }
        }
    }

    // Periodic session eviction: free SSM snapshots for expired sessions.
    {
        let freed_slots = session_manager.evict_expired();
        if !freed_slots.is_empty() {
            tracing::info!(
                "Session eviction: freed {} SSM snapshot slot(s), {} sessions active",
                freed_slots.len(),
                session_manager.session_count()
            );
        }
    }

    // Drain any remaining active sequences on shutdown.
    for mut a in active {
        finish_sequence(&*model, &mut a);
    }
    if let Some(ref mut spill) = spill_manager {
        for s in swapped {
            let _ = spill.remove_file(s.swap_id);
        }
    }
    for p in prefilling {
        let mut seq = p.seq;
        let _ = model.free_sequence(&mut seq);
        let _ = model.ep_broadcast_cmd_for_seq(seq.slot_idx as u32, 0xFFFFFFF1);
    }
    // Shutdown applies to every slot the worker has; seq_id is ignored.
    let _ = model.ep_broadcast_cmd_for_seq(0, 0xFFFFFFFF);
    tracing::info!("Scheduler stopped");
}
