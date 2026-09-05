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
mod adaptive_rung;
mod adaptive_spec;
mod admission;
mod beam_prefill;
mod confidence;
mod decode_logits_content;
mod decode_logits_seq;
mod decode_logits_step;
mod decode_step;
mod emit_step;
mod fast_greedy;
#[cfg(test)]
mod finish_guard_tests;
mod helpers;
mod lifecycle;
#[cfg(test)]
mod lifecycle_tests;
mod logit_dump;
mod logit_processors;
mod logprobs;
mod mod_helpers;
pub use mod_helpers::capture_runtime_handle;
pub mod dumps;
pub mod levers;
pub mod limits;
mod mtp_accept_debug;
mod mtp_bootstrap_step;
mod mtp_dcut;
mod mtp_gate;
mod mtp_step;
pub(crate) mod mtp_timing;
mod phase_continue_prefills;
mod phase_promote_prefills;
mod phase_start_prefills;
mod preempt;
#[cfg(test)]
mod preempt_tests;
mod prefill_a_step;
mod prefill_a_step_params;
mod prefill_b_step;
#[cfg(test)]
mod prefill_timing_tests;
mod repetition;
mod rollback;
mod sample_step;
pub mod sched_ctx;
pub mod snapshot;
mod spec_capacity;
pub mod spec_stats;
mod spec_step;
mod ssm_decode_ring;
mod teardown;
#[cfg(test)]
mod test_support;
#[cfg(test)]
mod think_skip_tests;
mod types;
mod verify_dflash_batch_step;
mod verify_dflash_step;
mod verify_k2_step;
mod verify_k3_step;
mod verify_k4_batch_step;
mod verify_k4_step;
mod verify_k4_verdict;
mod verify_pipeline_helper;
pub mod vocab_masks;

use beam_prefill::resolve_beam_hyp;
use confidence::*;
use decode_logits_content::*;
use decode_logits_seq::*;
use decode_logits_step::*;
use decode_step::*;
use emit_step::*;
pub use helpers::WatchdogParams;
pub(crate) use helpers::parse_disable_watchdogs;
pub use helpers::resolve_content_loop_watchdog;
use helpers::*;
pub use helpers::{CONTENT_LOOP_PERIOD_MAX, CONTENT_LOOP_PERIOD_MIN};
use lifecycle::*;
use logprobs::*;
use mod_helpers::*;
use mtp_bootstrap_step::*;
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
use verify_dflash_batch_step::*;
use verify_dflash_step::*;
use verify_k2_step::*;
use verify_k3_step::*;
use verify_k4_batch_step::*;
use verify_k4_step::*;
use verify_k4_verdict::*;
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

use std::sync::Arc;
use std::time::Instant;

use crate::api::{GrammarSpec, InferenceRequest, InferenceResponse, StreamEvent};
use crate::grammar::{GrammarEngine, GrammarState};
use crate::ngram::NgramProposer;
use crate::scheduling_policy::SchedulingPolicy;

/// A runtime LoRA adapter control command, applied by the scheduler at a
/// QUIESCENT point (no in-flight decode) so it never races a graph replay or a
/// live delta read.
pub enum LoraCommand {
    /// Rotate the globally-active adapter to a RESIDENT slot by NAME.
    Rotate(String),
    /// Dynamically LOAD the adapter at `dir` into pool `slot` (pool-size-1
    /// per-request weight change) and make it that slot's resident adapter.
    LoadIntoSlot {
        name: String,
        dir: std::path::PathBuf,
        slot: usize,
    },
    /// Task #27: demand-driven RDMA PROMOTE of a stageable-but-not-resident
    /// adapter from the peer into a cache pool slot (victim chosen on the model
    /// thread), then make it active. The chosen slot + any evicted name flow
    /// back through the ack. `peft` supplies the r/alpha the peer manifest lacks.
    Promote {
        peer_addr: String,
        adapter_id: String,
        name: String,
        peft: atlas_core::config::PeftAdapterConfig,
    },
    /// No-RDMA sibling of [`Self::Promote`]: demand-driven DISK promote of a
    /// stageable-but-not-resident adapter loaded from `dir` into a cache pool
    /// slot (victim chosen on the model thread), then made active. The chosen
    /// slot + any evicted name flow back through the ack. No `peft`: the disk
    /// swap re-parses the dir's `adapter_config.json`.
    PromoteDisk {
        name: String,
        dir: std::path::PathBuf,
    },
}

/// Successful result of a [`LoraCommand`] applied at quiescence. Rotate/Load
/// return [`LoraAck::Done`]; a Promote returns the resolved cache slot (which the
/// HTTP miss path uses as the request's `adapter_slot`) and any evicted adapter
/// name (so the caller drops its stale name->slot overlay entry).
#[derive(Debug, Clone)]
pub enum LoraAck {
    Done,
    Promoted {
        slot: usize,
        evicted: Option<String>,
    },
}

/// A LoRA control command plus the oneshot ack the HTTP handler awaits
/// (`Ok(ack)` on success, `Err(reason)` on unknown adapter / rotation not armed /
/// load failure / pool full).
pub type LoraRotation = (
    LoraCommand,
    tokio::sync::oneshot::Sender<Result<LoraAck, String>>,
);

/// Run the scheduler loop on the current thread.
#[allow(clippy::too_many_arguments)]
/// How many concurrent sequences may speculate. Default 16, override with
/// `ATLAS_MTP_MAX_SEQS` (`=1` restores the single-sequence-only gate).
///
/// `step_mtp` is index-correct over the active slice, so raising this runs
/// MTP over n sequences per step. With the batched K=4 verify wired
/// (`verify_k4_batch_step.rs`), verify-ready K=4 grammarless sequences are
/// verified in ONE eager n*4-row forward (weights read once); anything the
/// model can't batch (EP, HSS, LoRA, grammar, non-uniform K, DFlash) falls
/// back to the serialized per-seq loop that MEASURED (2026-07-27) collapses
/// throughput: cap=4 at C=4 25.8 vs 48.5 MTP-off. Kill switch for A/B:
/// `ATLAS_NO_MTP_BATCH_VERIFY` (presence) forces that serialized loop.
fn mtp_max_seqs() -> usize {
    // SSOT moved to `spark_model::speculative::mtp_max_seqs()` (batched-MTP
    // E1/E2): the model-side single-sequence MTP structures (catchup ring,
    // refeed labels, carry slot) gate on the SAME value the scheduler gates
    // dispatch on. Same parse; default 16 since 2026-07-29 (was 8, and 1
    // before the batched multi-seq verify + propose in
    // `verify_k4_batch_step.rs` removed the serialization that made cap=1
    // mandatory — C=4 cap=4: 25.8 serialized -> 49.0 batched vs 48.5
    // MTP-off). `ATLAS_MTP_MAX_SEQS=8` restores the round-3 cap, `=1` the
    // old single-sequence-only gate.
    spark_model::speculative::mtp_max_seqs()
}

pub fn run(
    mut model: Box<dyn Model>,
    request_rx: tokio::sync::mpsc::Receiver<InferenceRequest>,
    rotation_rx: tokio::sync::mpsc::Receiver<LoraRotation>,
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
    // Per-token masks for THIS model's vocabulary. Carried rather than read
    // from a process-wide static: they are indexed by token id and are
    // meaningless against a different tokenizer.
    vocab_masks: crate::scheduler::vocab_masks::VocabMasks,
    // This model's hard stops: two tokenizer-resolved token ids and the
    // served-context ceiling. Carried for the same reason as `vocab_masks`.
    limits: crate::scheduler::limits::SchedLimits,
    // This model's MODEL.toml `[behavior]` watchdog tunables.
    watchdog: crate::scheduler::helpers::WatchdogParams,
    // Shared with the dashboard, which toggles the loop watchdog mid-run.
    levers: std::sync::Arc<crate::scheduler::levers::SchedLevers>,
    // Shared with the dashboard, which polls it for the queue/KV display.
    snapshot: std::sync::Arc<crate::scheduler::snapshot::SnapshotCell>,
) {
    // Everything this run needs that is derived from the model rather than the
    // request. The levers were twenty-odd `ATLAS_*` statics; they are resolved
    // once here and read through `sched` from every step function.
    let sched =
        crate::scheduler::sched_ctx::SchedCtx::new(vocab_masks, levers, snapshot, limits, watchdog);
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
    let mut mtp_gate = if use_mtp && !sched.levers.mtp_gate_force {
        Some(mtp_gate::MtpGate::new(num_drafts))
    } else {
        if use_mtp && sched.levers.mtp_gate_force {
            tracing::warn!(
                "--mtp-gate force: MTP throughput gate DISARMED (diagnostic; \
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
    // MTP verify-pool slot coverage (bs>32 reserve diet): spec dispatch is
    // additionally gated on every active slot being < this cap — the SAME
    // number `SsmStatePool::new` sizes the intermediate/checkpoint pools to
    // and preflight reserves for (SSOT: `ssm_reserve::mtp_state_slots`).
    // Only SSM models have those pools; pure-attention models keep spec
    // ungated. Equals max_batch at bs<=32 (guard vacuous). Read once at
    // startup like every other env-derived policy value.
    let spec_slot_cap = if model.has_ssm_layers() {
        spark_model::ssm_reserve::mtp_state_slots(max_batch_size)
    } else {
        max_batch_size
    };
    if spec_slot_cap < max_batch_size {
        tracing::info!(
            "MTP verify pools cover {spec_slot_cap}/{max_batch_size} SSM slots — \
             sequences on uncovered slots plain-decode until compaction moves them \
             down (kill switch ATLAS_MTP_POOL_FULL_WIDTH restores full width)"
        );
    }

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

    // Depth-aware admission watermark (PCND: explicit default = the served
    // context ceiling, i.e. reserve each request's own max_tokens; see
    // `admission` module docs and ATLAS_KV_ADMIT_WATERMARK).
    let admit_watermark = admission::resolve_admit_watermark(sched.limits.max_seq_len);

    let pending = Arc::new((
        Mutex::new(PendingQueue {
            requests: Vec::new(),
            closed: false,
            rotations: Vec::new(),
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

    // Rotation receiver thread: LoRA adapter-rotation control requests land in
    // `pending.rotations` (never the sequence queue) and wake the scheduler via
    // the SAME condvar. The scheduler applies them at a quiescent point.
    let pr = Arc::clone(&pending);
    std::thread::spawn(move || {
        let mut rx = rotation_rx;
        while let Some(rot) = rx.blocking_recv() {
            pr.0.lock().rotations.push(rot);
            pr.1.notify_one();
        }
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
    let mut preempted: Vec<PreemptedSeq> = Vec::new();
    let mut spill_manager: Option<KvSpillManager> = if swap_space_gb > 0 {
        let max_bytes = swap_space_gb as u64 * 1024 * 1024 * 1024;
        // Per-PROCESS directory. `KvSpillManager::new` wipes stale `swap_*` files
        // on construction, which is correct for a restart and correct across a
        // hot-swap (the old scheduler is joined before the new one is built, so
        // they never overlap) — but a SHARED path means two `spark serve`
        // processes on one box wipe each other's live spill files. That is a
        // pre-existing hazard this work surfaced rather than introduced.
        let spill_dir = std::env::temp_dir().join(format!("atlas-swap-{}", std::process::id()));
        match KvSpillManager::new(spill_dir.clone(), max_bytes) {
            Ok(mgr) => {
                tracing::info!("Swap space: {swap_space_gb} GB at {}", spill_dir.display());
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

    let mut snapshot_steps: u64 = 0;
    loop {
        // ── Drain pending → start prefill (chunked or full) ──
        // The `t_loop_*` brackets attribute the out-of-step GAP the
        // ATLAS_MTP_TIMING summary reports (see mtp_timing::Phase::Gap): each
        // records one scheduler-tick section. `record` no-ops when the env is
        // unset; the Instant::now() reads are the documented residual cost.
        let t_loop = std::time::Instant::now();
        let new_reqs = drain_pending_requests(
            &pending,
            &active,
            &prefilling,
            &*policy,
            max_batch_size,
            // Parked sequences (spilled or requeued) are waiting on blocks,
            // not on new requests — never block on the request condvar while
            // any exist, or an empty active set would strand them forever.
            !(swapped.is_empty() && preempted.is_empty()),
        );
        // ── Depth-aware admission: queue what cannot fit its decode depth ──
        // (admit-then-preempt was the C=128 thrash; see `admission` docs).
        let new_reqs = admission::gate_admissions(
            &*model,
            &pending,
            new_reqs,
            &active,
            &prefilling,
            &swapped,
            &preempted,
            admit_watermark,
            sched.limits.max_seq_len,
            block_size,
        );
        sched.timing.record(mtp_timing::Phase::LoopDrain, t_loop);

        // ── Publish the observability snapshot (one uncontended lock + a
        // ~72-byte memcpy per tick; see scheduler/snapshot.rs). ──
        snapshot_steps += 1;
        let t_loop = std::time::Instant::now();
        {
            let (mtp_mode, delivered_tps) = match mtp_gate.as_ref() {
                Some(g) => g.observe(),
                None => (snapshot::MtpModeSnap::Off, 0.0),
            };
            sched.snapshot.publish(snapshot::SchedulerSnapshot {
                active_seqs: active.len() as u32,
                prefilling_seqs: prefilling.len() as u32,
                swapped_seqs: (swapped.len() + preempted.len()) as u32,
                pending_len: new_reqs.len() as u32,
                kv_blocks_free: model.num_free_blocks() as u32,
                kv_blocks_total: model.num_total_blocks() as u32,
                ssm_slots_used: session_manager.session_count() as u32,
                ssm_slots_total: session_manager.total_slots() as u32,
                mtp_mode,
                delivered_tps,
                steps_total: snapshot_steps,
                published_at: std::time::Instant::now(),
            });
        }
        sched.timing.record(mtp_timing::Phase::LoopSnapshot, t_loop);

        // ── Apply queued LoRA adapter rotations at a QUIESCENT point ──
        // Only when nothing is in flight (no active decode, no in-progress
        // prefill, no just-drained request, AND no sequence spilled to disk) so
        // the re-point/promote never races a live delta read or a graph replay.
        // `swapped` MUST be empty too: a spilled sequence has RELEASED its adapter
        // ref (#25), so without this gate a Promote/swap could evict/re-stage the
        // slot its KV was computed under and corrupt it on resume (#27 FINDING 1 /
        // #31). Otherwise the commands stay queued and retry once the batch drains.
        if active.is_empty()
            && prefilling.is_empty()
            && new_reqs.is_empty()
            && swapped.is_empty()
            && preempted.is_empty()
        {
            let rotations = std::mem::take(&mut pending.0.lock().rotations);
            for (cmd, ack) in rotations {
                let res = match cmd {
                    LoraCommand::Rotate(name) => {
                        let r = model
                            .set_active_lora(&name)
                            .map(|()| LoraAck::Done)
                            .map_err(|e| format!("{e:#}"));
                        if let Err(ref e) = r {
                            tracing::warn!("LoRA rotation to '{name}' failed: {e}");
                        }
                        r
                    }
                    LoraCommand::LoadIntoSlot { name, dir, slot } => {
                        let r = model
                            .swap_lora_from_disk(&dir, &name, slot)
                            .map(|()| LoraAck::Done)
                            .map_err(|e| format!("{e:#}"));
                        if let Err(ref e) = r {
                            tracing::warn!("LoRA disk swap '{name}' -> slot {slot} failed: {e}");
                        }
                        r
                    }
                    LoraCommand::Promote {
                        peer_addr,
                        adapter_id,
                        name,
                        peft,
                    } => {
                        let r = model
                            .promote_lora_from_peer(&peer_addr, &adapter_id, &name, peft)
                            .map(|(slot, evicted)| LoraAck::Promoted { slot, evicted })
                            .map_err(|e| format!("{e:#}"));
                        if let Err(ref e) = r {
                            tracing::warn!("LoRA promote '{name}' failed: {e}");
                        }
                        r
                    }
                    LoraCommand::PromoteDisk { name, dir } => {
                        let r = model
                            .promote_lora_from_disk(&dir, &name)
                            .map(|(slot, evicted)| LoraAck::Promoted { slot, evicted })
                            .map_err(|e| format!("{e:#}"));
                        if let Err(ref e) = r {
                            tracing::warn!("LoRA disk-promote '{name}' failed: {e}");
                        }
                        r
                    }
                };
                let _ = ack.send(res);
            }
        }
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
                // Reclaim from the prefix cache BEFORE paying disk I/O for a live
                // sequence. Cached blocks are a pure optimization; a swapped-out
                // sequence is in-flight work that costs a write now and a read
                // plus realloc later. This gate reads `num_free_blocks()`, which
                // counts only UNHELD blocks — and the cache legitimately pins
                // everything it caches, so "free" sits near zero on a warm server
                // while thousands of blocks remain reclaimable. Measured at C=4 on
                // a 8869-block pool: 10 swap-outs of ~40-block sequences followed
                // by 16 swap-ins, all of it avoidable churn.
                loop {
                    let free = model.num_free_blocks();
                    if free >= blocks_needed
                        || model.reclaim_prefix_blocks(blocks_needed - free) == 0
                    {
                        break;
                    }
                }
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
        let t_loop = std::time::Instant::now();
        start_new_requests(
            &*model,
            &sched,
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
        sched.timing.record(mtp_timing::Phase::LoopAdmit, t_loop);

        // ── Continue in-progress prefills ──
        let t_loop = std::time::Instant::now();
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
            &sched,
        );
        sched.timing.record(mtp_timing::Phase::LoopPrefill, t_loop);

        if active.is_empty() {
            // Every sequence can be parked (decode-preempted/spilled) with
            // nothing new admitted; the resume passes at the tail of the
            // loop sit AFTER decode, which this `continue` skips — so give
            // requeued victims their resume chance here or an otherwise
            // idle server would strand them.
            if !preempted.is_empty() {
                preempt::resume_preempted_seqs(
                    &*model,
                    &mut active,
                    &mut preempted,
                    max_batch_size,
                    block_size,
                );
            }
            if active.is_empty() {
                continue;
            }
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
                watchdog: sched.watchdog,
                scratch: &sched.scratch,
                dumps: &sched.dumps,
                stats: sched.stats.clone(),
                think_end_token,
                think_start_token,
                tool_call_start_token,
                tool_call_end_token,
                boundary_mask: sched.masks.boundary.clone(),
                mid_word_mask: sched.masks.mid_word.clone(),
                sampling: sched.levers.sampling(),
                timing: sched.timing.clone(),
            };
            // Spec-resume guard (ATLAS_DFLASH_RESUME_GUARD=N, default 0 = off):
            // keep the first N post-`</think>` tokens on plain serial decode.
            // The T=0 verify-vs-decode low-margin flips measured 2026-07-07
            // concentrate in the answer's opening tokens; serial-decoding that
            // window sidesteps them while leaving the high-accept answer body
            // speculated. N=0 preserves exact prior behavior.
            let dflash_resume_guard = sched.levers.dflash_resume_guard;
            // ATLAS_DFLASH_SPEC_THINK=1: DFlash raw-argmax may speculate
            // inside `<think>`. Standard MTP already verifies during think.
            // DFlash raw-argmax does not run ForcedThinkEnd, so it stays
            // serial-in-think unless this lever is on. Resume guard still
            // serial-decodes the spec-entry window.
            let dflash_spec_think = sched.levers.dflash_spec_think;
            // Spec dispatch additionally requires every active sequence's
            // SSM slot to be covered by the MTP verify state pools
            // (intermediates + checkpoints), which are sized to
            // `ssm_reserve::mtp_state_slots(max_batch_size)` slots — the
            // bs>32 reserve diet. Vacuously true at bs<=32 (slots are
            // always < bs <= cap). At bs>32 a transiently high-slotted
            // sequence (LIFO free-list claim after churn) plain-decodes
            // until retirement-time compaction migrates it under the cap;
            // the `else` branch below already clears its stale drafts.
            // Kill switch ATLAS_MTP_POOL_FULL_WIDTH (presence) restores
            // full-width pools and makes this guard vacuous at any bs.
            let spec_slots_covered = active.iter().all(|a| a.seq.slot_idx < spec_slot_cap);
            // WIDTH half of the runtime speculation regime (wave 47). The
            // depth once engaged is `adaptive_rung::drafts_for`; whether we
            // engage at all is this predicate, and it is what lets ONE serve
            // cover the whole concurrency ladder. Recorded (not decided) in
            // `adaptive_rung` so both halves of the regime report from one
            // place — no parallel accounting, the value below is the one the
            // dispatch chain actually uses.
            let spec_width_ok = active.len() <= mtp_max_seqs();
            if use_mtp {
                adaptive_rung::note_width_regime(active.len(), spec_width_ok);
            }
            if use_ngram_speculative
                && active.len() == 1
                && spec_slots_covered
                && active[0].grammar_state.is_none()
            {
                // N-gram speculative: CPU proposer + CUDA-graphed K=2 verify.
                if let Some(ref mut proposer) = ngram_proposer {
                    step_ngram(&*model, &mut active, &sched, proposer, &verify_ctx);
                }
            } else if use_self_speculative
                && active.len() == 1
                && spec_slots_covered
                && active[0].grammar_state.is_none()
            {
                // Self-speculative: draft via layer-skipping, verify with full model.
                // Clamped to the active slot's tiered verify capacity, like
                // the MTP lane (see `spec_capacity`).
                let nd = spec_capacity::clamp_drafts_to_slot_capacity(
                    num_drafts,
                    active
                        .iter()
                        .map(|a| model.mtp_slot_draft_capacity(a.seq.slot_idx)),
                );
                step_self_spec(&*model, &mut active, &sched, nd, &verify_ctx);
            } else if use_mtp
                && spec_width_ok
                && spec_slots_covered
                && (
                    // Both lanes stay serial inside `<think>` unless
                    // ATLAS_DFLASH_SPEC_THINK=1. Resume guard still
                    // serial-decodes the spec-entry window. EVERY active
                    // sequence must be eligible, not just active[0].
                    active.iter().all(|a| {
                        mtp_gate::spec_dispatch_eligible(
                            a.inside_thinking,
                            a.post_think_emitted,
                            a.output_tokens.len() as u32,
                            a.suppress_tool_call,
                            a.disable_mtp,
                            dflash_spec_think,
                            dflash_resume_guard,
                            dflash_verify_raw_argmax,
                        )
                    })
                )
            {
                // Throughput-arbitrated MTP gate: EVERY batch step is timed
                // and reported at its real delivered token count, and the
                // gate picks whichever mode (MTP verify vs plain decode)
                // DELIVERS more tokens/sec —
                // with hysteresis, dwell, and periodic probing of the other
                // mode. Both step types emit real, correct tokens, so
                // arbitration never wastes work. See mtp_gate module docs for
                // why component-time economics were replaced (webserver_ok
                // A/B 2026-07-20: always-on Σ1028s/10-10 vs timing-gated
                // Σ1846s/9-10).
                if let Some(gate) = mtp_gate.as_mut() {
                    if gate.maybe_remeasure(active[0].seq.seq_len) {
                        for a in active.iter_mut() {
                            a.mtp_acct.note_regime_reprobe();
                        }
                    }
                    gate.note_depth(active[0].seq.seq_len);
                    // Spec-entry pin: the first N post-`</think>` tokens run
                    // the verify path even when the gate's arbitration says
                    // Serial, so the answer OPENING cannot flip between the
                    // serial and batch-K forwards on wall-clock luck (see
                    // `mtp_gate::entry_pin_forces_verify` for the measured
                    // flip evidence). Pinned steps are deliberately NOT
                    // recorded: charging verify walls to a Serial-mode
                    // accumulator would corrupt the arbitration baselines,
                    // and ≤N unmeasured steps per sequence merely delay the
                    // reprobe cadence by the same handful of tokens.
                    let min_post_think_emitted = active
                        .iter()
                        .map(|a| a.post_think_emitted)
                        .min()
                        .unwrap_or(u32::MAX);
                    // DFlash C<=2 pin: hold the verify arm whenever <=2
                    // sequences are active. Arbitration is par on tok/s
                    // there, but its serial<->batch-K forward flips fork
                    // the temp-0 token stream mid-answer and the bad
                    // attractor degenerates into repetition loops (see
                    // `dflash_gate_pin_c2` lever doc for the measurements).
                    // Pinned steps skip recording exactly like the
                    // entry pin: charging verify walls to a Serial-mode
                    // accumulator would corrupt the arbitration baselines
                    // it resumes with at C>=3.
                    let dflash_pin = dflash_verify_raw_argmax
                        && active.len() <= 2
                        && sched.levers.dflash_gate_pin_c2;
                    if (dflash_pin || mtp_gate::entry_pin_forces_verify(min_post_think_emitted))
                        && gate.next_step() == mtp_gate::GateStep::MeasureDecode
                    {
                        let lens_before: Vec<usize> =
                            active.iter().map(|a| a.seq.seq_len).collect();
                        step_mtp(
                            &*model,
                            &mut active,
                            &sched,
                            num_drafts,
                            &verify_ctx,
                            dflash_verify_raw_argmax,
                        );
                        for (a, &b) in active.iter_mut().zip(lens_before.iter()) {
                            a.mtp_acct
                                .record_verify_emitted(a.seq.seq_len.saturating_sub(b));
                        }
                    } else {
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
                                    &sched,
                                    spill_manager.as_mut(),
                                    &mut swapped,
                                    &mut preempted,
                                );
                                // A plain decode step emits ONE token PER
                                // ACTIVE SEQUENCE. Charging 1 regardless of
                                // width under-read the serial EWMA ~n× at
                                // batch n (the verify arm below was already
                                // multi-seq summed), biasing arbitration
                                // toward Mtp exactly under concurrency.
                                gate.record_decode(t0.elapsed(), active.len());
                                for a in active.iter_mut() {
                                    a.mtp_acct.record_serial();
                                }
                                // ATLAS_MTP_CATCHUP: ring the serially decoded
                                // token's hidden so the next MTP re-probe can
                                // batch-feed the drafter over the serial gap
                                // (no-op when the feature is off).
                                //
                                // LABEL CONVENTION (off-by-one fixed 2026-07-21).
                                // The reader feeds drafter pair key `k` from ring
                                // label `k + 1`, because pair key k is
                                // `(embed(t_{k+1}), hidden_k)` — so label n must
                                // hold `hidden_{n-1}`, the hidden that PREDICTED
                                // token n. `step_decode_only` forwards
                                // `last_token` at the OLD `seq_len` and only then
                                // pushes that input token and increments
                                // (`decode_a2.rs` / `decode_b.rs`: `tokens.push`
                                // + `seq_len += 1`). So the hidden now in row 0 is
                                // `hidden_{seq_len - 1}` and its label is
                                // `seq_len`, not `seq_len - 1`.
                                //
                                // This previously wrote `seq_len - 1`, which handed
                                // every serially-fed pair key the hidden of the
                                // NEXT position. It is the same quantity the K=3
                                // re-feed labels `base + t + 1` for verify row t at
                                // position `base + t` — that convention is verified
                                // by dumped hidden fingerprints (93/93 cross-step,
                                // see `speculative::mtp_refeed_accepted_enabled`),
                                // so the serial hook was the side that disagreed.
                                //
                                // Multi-seq guard (batched-MTP E2): the catchup
                                // ring is a SINGLE-sequence structure (one ring,
                                // one label space). With n active sequences the
                                // hidden in row 0 belongs to an arbitrary member
                                // of the batch, so ringing it would interleave
                                // unrelated hiddens under one label space.
                                // (`mtp_catchup_enabled` is also force-off when
                                // ATLAS_MTP_MAX_SEQS > 1 — this guard keeps the
                                // save itself single-seq-only regardless.)
                                if active.len() == 1
                                    && let Err(e) =
                                        model.save_hidden_for_catchup(0, active[0].seq.seq_len)
                                {
                                    tracing::warn!("save_hidden_for_catchup: {e:#}");
                                }
                            }
                            mtp_gate::GateStep::MeasureVerify => {
                                // A bootstrap-only step (no pending drafts) emits
                                // 1 token and proposes; its cost is charged to the
                                // MTP mode — proposing IS part of what MTP costs.
                                // Sum over ALL speculating sequences: the gate arbitrates
                                // on tokens-per-second, so counting only active[0]
                                // under-reports MTP's throughput by a factor of n and
                                // biases the gate toward serial decode.
                                let lens_before: Vec<usize> =
                                    active.iter().map(|a| a.seq.seq_len).collect();
                                let t0 = std::time::Instant::now();
                                step_mtp(
                                    &*model,
                                    &mut active,
                                    &sched,
                                    num_drafts,
                                    &verify_ctx,
                                    dflash_verify_raw_argmax,
                                );
                                let emitted: usize = active
                                    .iter()
                                    .zip(lens_before.iter())
                                    .map(|(a, &b)| a.seq.seq_len.saturating_sub(b))
                                    .sum();
                                gate.record_verify_step(t0.elapsed(), emitted, active.len());
                                for (a, &b) in active.iter_mut().zip(lens_before.iter()) {
                                    a.mtp_acct
                                        .record_verify_emitted(a.seq.seq_len.saturating_sub(b));
                                }
                            }
                        }
                    }
                    // One-time transition work when the gate switches to
                    // Serial: drop pending drafts and order the draft-head
                    // state resync before the next plain decode reads it.
                    // Serial->Mtp needs nothing (the next MTP step
                    // bootstraps from empty pending_drafts).
                    if gate.take_fresh_decision() == Some(mtp_gate::GateDecision::DisableMtp) {
                        for a in active.iter_mut() {
                            a.pending_drafts.clear();
                            a.pending_draft_conf.clear();
                        }
                        if let Err(e) = model.sync_secondary() {
                            tracing::error!("mtp-gate→decode sync_secondary: {e:#}");
                        }
                    }
                } else {
                    // Gate bypassed (ATLAS_MTP_GATE_FORCE=1): plain MTP.
                    let lens_before: Vec<usize> = active.iter().map(|a| a.seq.seq_len).collect();
                    step_mtp(
                        &*model,
                        &mut active,
                        &sched,
                        num_drafts,
                        &verify_ctx,
                        dflash_verify_raw_argmax,
                    );
                    for (a, &b) in active.iter_mut().zip(lens_before.iter()) {
                        a.mtp_acct
                            .record_verify_emitted(a.seq.seq_len.saturating_sub(b));
                    }
                }
            } else {
                // Batch decode (no MTP). Clear stale drafts when transitioning out of MTP mode.
                if use_mtp {
                    for a in active.iter_mut() {
                        a.pending_drafts.clear();
                        a.pending_draft_conf.clear();
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
                    &sched,
                    spill_manager.as_mut(),
                    &mut swapped,
                    &mut preempted,
                );
                for a in active.iter_mut() {
                    a.mtp_acct.record_serial();
                }
            }
        }

        let t_loop = std::time::Instant::now();
        // Deadline sweep BEFORE retirement, so a timed-out sequence retires
        // on this same iteration. Placed here rather than in a decode step
        // because the MTP/speculative path does not run `process_decode_logits`.
        enforce_request_deadlines(&mut active);
        retire_finished_sequences(&*model, &mut active, sched.limits.max_seq_len);
        sched.timing.record(mtp_timing::Phase::LoopRetire, t_loop);

        // ── Swap-in: resume swapped sequences when blocks free up ──
        let t_loop = std::time::Instant::now();
        if let Some(ref mut spill) = spill_manager {
            let mut resumed_any = true;
            while resumed_any && !swapped.is_empty() && active.len() < max_batch_size {
                resumed_any = false;
                let mut free = model.num_free_blocks();
                // Nothing fits: the blocks this sequence needs may be sitting in
                // the prefix cache, which holds one ref per radix node and so
                // never volunteers them. Prefill and decode reclaim implicitly
                // (`try_alloc` → evict → retry); swap-in gates on free blocks
                // BEFORE restoring, so it has to ask. Without this a swapped-out
                // sequence waits forever on capacity that is reclaimable but not
                // free — the scheduler goes idle with clients still connected.
                if let Some(smallest) = swapped.iter().map(|s| s.num_blocks).min()
                    && smallest > free
                {
                    // Evicting N radix nodes frees FEWER than N blocks whenever a
                    // live sequence still holds one (eviction returns only the
                    // cache's own ref), so a single pass sized to the shortfall
                    // undershoots and the gate below still fails — measured as
                    // "reclaimed 303 ... to restore a 305-block sequence" with
                    // zero restores. Keep asking until the sequence fits or the
                    // cache has nothing evictable left; each pass either frees at
                    // least one block or returns 0, so this terminates.
                    let was = free;
                    let mut total = 0usize;
                    while free < smallest {
                        let got = model.reclaim_prefix_blocks(smallest - free);
                        if got == 0 {
                            break;
                        }
                        total += got;
                        free = model.num_free_blocks();
                    }
                    if total > 0 {
                        tracing::info!(
                            "Swap-in: reclaimed {total} block(s) from the prefix cache for a \
                             {smallest}-block sequence (free {was} -> {free})",
                        );
                    }
                }
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
        // ── Resume requeued (decode-preempted) sequences when blocks free ──
        preempt::resume_preempted_seqs(
            &*model,
            &mut active,
            &mut preempted,
            max_batch_size,
            block_size,
        );
        sched.timing.record(mtp_timing::Phase::LoopSwap, t_loop);
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
        finish_sequence(&*model, &mut a, sched.limits.max_seq_len);
    }
    if let Some(ref mut spill) = spill_manager {
        for s in swapped {
            let _ = spill.remove_file(s.swap_id);
        }
    }
    for mut p in preempted {
        send_error_to_sink(
            &mut p.a.sink,
            "server shutting down before preempted resume",
        );
    }
    for p in prefilling {
        let mut seq = p.seq;
        let _ = model.free_sequence(&mut seq);
        let _ = model.ep_broadcast_cmd_for_seq(seq.slot_idx as u32, 0xFFFFFFF1);
    }
    // Shutdown applies to every slot the worker has; seq_id is ignored.
    let _ = model.ep_broadcast_cmd_for_seq(0, 0xFFFFFFFF);

    // Release the model's device memory HERE, in order and able to report a
    // failure, before the `Box` drops.
    //
    // This is the point the whole `ModelResource`/`Teardown` mechanism was
    // built for, and until now nothing called it: `Model::teardown` had no
    // caller anywhere in production, so the ordered release was dead code and
    // every allocation fell through to the backend's `Drop` sweep — thousands
    // of them per swap (3370, then 3973, on two successful swaps). The sweep is
    // the intended BACKSTOP for what no owner claims, not the mechanism. `Drop`
    // is neither ordered nor able to fail, which is precisely why `Teardown`
    // exists.
    //
    // Every sequence above has been freed and no request can arrive — but that
    // is HOST-side quiescence, and it is not the quiescence a free needs.
    //
    // `Model::teardown`'s contract says it runs "after the scheduler has
    // drained AND THE STREAM IS SYNCHRONISED". Draining was honoured; the
    // synchronise was not, and nothing else supplied it. Every launch above
    // (`finish_sequence`, `free_sequence`, the decode loop, the EP broadcasts)
    // is ASYNCHRONOUS: it returns once the work is queued, not once the GPU has
    // run it. So teardown could start freeing pools while kernels were still
    // reading them, and on GB10 a freed mapping is unmapped, not merely reused.
    //
    // That is not theoretical. A hot-swap on 2026-08-07 took
    //   NVRM: Xid 31, name=atlas-swap
    //   MMU Fault: ENGINE GRAPHICS GPC0 ... FAULT_PTE ACCESS_TYPE_VIRT_READ
    // — a graphics-engine read of an address with no page table entry, on the
    // thread that drives swap → join → teardown. A kernel reading memory this
    // function had already handed back is exactly that fault.
    //
    // Synchronise BOTH streams the scheduler submits to. The decode path uses
    // the default stream; prefill has its own since the compute/copy overlap
    // (`prefill_stream` above), and work outstanding on either one can still be
    // touching the pools. A failure here is reported and teardown proceeds
    // regardless: refusing to free would leak the whole model, and a stream
    // that cannot be synchronised is already in a state teardown will not
    // improve — but the operator needs it in the log either way.
    let streams = [
        ("default", model.default_stream()),
        ("prefill", prefill_stream),
    ];
    let unsynced = teardown::quiesce_streams(&streams, |s| model.synchronize(s));
    for name in unsynced {
        tracing::error!(
            "could not synchronise the {name} stream before teardown — freeing \
             anyway, but device memory may still be in use"
        );
    }
    // ★ These two must stay adjacent and in this order. See `teardown`.
    if let Err(e) = model.teardown() {
        tracing::error!("model teardown reported a failure: {e:#}");
    }
    tracing::info!("Scheduler stopped");
}
