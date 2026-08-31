// SPDX-License-Identifier: AGPL-3.0-only

//! Batched Phase-A MTP bootstrap: ONE multi-sequence decode forward for every
//! draftless sequence, then ONE batched drafter forward per draft position.
//!
//! Phase A of `step_mtp` ran a per-sequence `model.decode()` — an M=1 forward
//! that re-reads every weight — for each sequence without pending drafts, and
//! then a per-sequence `run_mtp_propose_multi`. At n sequences that is n full
//! weight sweeps of the target plus n of the drafter, on an engine that is
//! ~87% bandwidth-saturated. Every sequence bootstraps on its first decode
//! step and again after any step that could not re-propose, so at C=8/16 this
//! lands repeatedly in the steady state. Third of the three eager costs the
//! n=16 finalizer matrix named (the K=1 verify step measured ~1.9x a plain
//! batch-16 decode step vs the ~1.72x break-even at p1~0.72).
//!
//! The batched form is the SAME machinery the non-MTP decode step already
//! uses: `decode_batch` (n rows, one weight read, slot-ordered so the batched
//! recurrent/graph paths engage), per-row logits at `row*vocab*elem`, then the
//! batched-verify Phase 2/4 pattern for the hiddens — stash every row BEFORE
//! any propose clobbers the shared `hidden_states` buffer, then propose from
//! the stash.
//!
//! Envelope (falls back to the per-sequence loop, never silently degrades):
//! * >= 2 draftless sequences and multi-seq MTP mode (`ATLAS_MTP_MAX_SEQS>1`)
//!   — at cap 1 the per-seq path must stay byte-identical;
//! * not the DFlash bootstrap (its fused pass replaces the standalone decode);
//! * `can_batch_verify(&[2; n])` — the same non-EP / non-HSS / no-LoRA /
//!   stash-allocated envelope the batched verify self-gates on, and the stash
//!   is what makes a cross-sequence propose possible at all;
//! * the DFlash serial-append / unified-ctx commit modes are OFF (their
//!   per-sequence ctx appends are ordered against a per-sequence decode).
//!
//! Kill switch `ATLAS_NO_MTP_BATCH_BOOTSTRAP` (PRESENCE check per the house
//! convention — `=0` is NOT off) forces the per-sequence loop.

use super::*;

/// Kill switch, PRESENCE check, read once per process.
pub(super) fn bootstrap_batch_disabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("ATLAS_NO_MTP_BATCH_BOOTSTRAP").is_some())
}

/// Log a `run_mtp_propose_batched` failure at the right level. Shared by the
/// only two callers (this file and `verify_k4_batch_step.rs`).
///
/// The Err arm carries BOTH real failures — drafter-pool "KV cache exhausted"
/// (the capacity signal the mtp_head pool sizing relied on seeing), CUDA
/// launch/copy errors, "can_propose_batch lied" bugs — which MUST stay at
/// ERROR, and the deterministic meta-stride overflow: a sequence whose
/// drafter block table outgrew its `propose_meta` slab. The old fixed 2048
/// stride = 448 entries = 7,168 tokens was sized in the 4K era; agentic
/// contexts of 10-20K re-fired it for every group on every step — permanent
/// ERROR spam for a permanent, known degradation (PROGRESS_LOG 5.2/6.17).
/// The stride is now computed from `max_seq_len`, so the overflow only
/// remains reachable under an `ATLAS_PROPOSE_META_STRIDE` override or a
/// sequence past `max_seq_len`; it logs at DEBUG per occurrence (silent at
/// the production INFO level, still diagnosable at RUST_LOG=debug — a
/// once-per-process gate would hide that the degradation is permanent).
///
/// Discrimination is a string match on the single producing `ensure!`
/// ("exceeds meta stride", spark-model mtp_head/forward_batch.rs) rather than
/// a typed error: a pub error type would couple the scheduler to mtp_head
/// internals across crates for one message, and the match follows the
/// existing `{e:#}`-contains precedent (decode_step.rs "KV cache exhausted",
/// phase_start_prefills.rs "pool exhausted"). Matching the alternate `{e:#}`
/// form keeps it robust to `.context()` additions upstream.
pub(super) fn log_propose_batched_err(prefix: &str, e: &anyhow::Error) {
    let msg = format!("{e:#}");
    if msg.contains("exceeds meta stride") {
        tracing::debug!("{prefix}: {msg}");
    } else {
        tracing::error!("{prefix}: {msg}");
    }
}

/// One batched argmax readback instead of n serialized single-CTA scans:
/// **ON** by default, disabled by PRESENCE of `ATLAS_NO_MTP_BOOT_ARGMAX`
/// (house convention — `=0` is NOT off). Read once per process.
fn boot_argmax_batch_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("ATLAS_NO_MTP_BOOT_ARGMAX").is_none())
}

/// Whether [`step_mtp_bootstrap_batched`] can run for these sequences.
pub(super) fn can_batch_bootstrap(
    model: &dyn Model,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    n: usize,
    dflash_verify_raw_argmax: bool,
) -> bool {
    n >= 2
        && !dflash_verify_raw_argmax
        && !bootstrap_batch_disabled()
        && spark_model::speculative::mtp_multi_seq_mode()
        && !sched.levers.dflash_unified_ctx
        && !sched.levers.dflash_serial_append
        // FP32-lm_head models (Gemma-4 dense) are excluded: `logits_ptr_is_fp32`
        // is an exact-pointer identity against the FP32 scratch buffer, so a
        // per-ROW offset pointer would dispatch as BF16 and read garbage. The
        // non-MTP batched decode avoids this by slicing a host copy instead of
        // handing out row pointers.
        && !model.decode_logits_fp32()
        // k=2 is the narrowest verify width; this call asks the model for the
        // batched-MTP envelope (stash allocated, non-EP, non-HSS, no LoRA),
        // not for a verify.
        && model.can_batch_verify(&vec![2usize; n])
}

/// Batched bootstrap for the `idxs` (ASCENDING) draftless sequences.
///
/// Mirrors the per-sequence Phase-A body exactly — same penalties, same
/// history scoping, same sampler, same emit, same adaptive-spec bookkeeping,
/// same effective-draft clamp — with the decode and the propose batched.
///
/// Deviation from the per-seq body, deliberate: when the hidden stash fails
/// the per-seq body's `continue` also skipped `start_checkpoint_async`; here
/// the checkpoint still runs (skipping it was incidental to the control flow,
/// and a missing checkpoint is a correctness hazard, not a saving).
pub(super) fn step_mtp_bootstrap_batched(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    idxs: &[usize],
    ladder_nd: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
) {
    // Disjoint &mut for the (ascending) indices, then reorder by SSM slot:
    // `decode_batch`'s batched-recurrent + graph paths require the batch in
    // pool-slot order (decode_step.rs sorts `active` for the same reason);
    // here `active` cannot be reordered (the caller's verify indices point
    // into it), so the REFS are sorted instead. Row j <-> refs[j].
    let mut refs: Vec<&mut ActiveSeq> = Vec::with_capacity(idxs.len());
    let mut it = active.iter_mut();
    let mut consumed = 0usize;
    for &i in idxs {
        let a = it.nth(i - consumed).expect("bootstrap index within active");
        consumed = i + 1;
        refs.push(a);
    }
    refs.sort_by_key(|a| a.seq.ssm_slot_idx().unwrap_or(a.seq.slot_idx));

    let n = refs.len();
    let tokens: Vec<u32> = refs.iter().map(|a| a.last_token).collect();

    // ── ONE batched decode forward: n rows, weights read once ──
    // EP broadcasts are emitted inside `decode_batch` itself (interleaved
    // with its per-seq calls) — hoisting them here would diverge the head's
    // comm-stream op order from the worker's. A batch-level failure finishes
    // the batch, as on the batched verify path: `decode_batch` gives no
    // "nothing advanced" guarantee, so retrying per-seq could double-decode.
    let logits = {
        let mut seq_refs: Vec<&mut SequenceState> = refs.iter_mut().map(|a| &mut a.seq).collect();
        match model.decode_batch(&tokens, &mut seq_refs, 0) {
            Ok(l) => l,
            Err(e) => {
                tracing::error!("batched bootstrap decode_batch (n={n}): {e:#}");
                for a in refs.iter_mut() {
                    a.finished = true;
                }
                return;
            }
        }
    };

    let vocab = model.vocab_size();
    let elem = if model.decode_logits_fp32() { 4 } else { 2 };

    // ── ONE batched argmax readback for the rows that sample greedily ──
    // Every eligible row's `sample_token_with_grammar` takes its fast-greedy
    // branch: `argmax_on_device` = ONE single-CTA `argmax_bf16` (grid [1,1,1],
    // ~100 us at a 248k vocab) plus a BLOCKING 4-byte D2H — n of them,
    // serialized on one stream, each draining the pipeline. `argmax_batch`
    // runs the identical per-row kernel body (one block per row, same tie
    // resolution — byte-identical by construction) in ONE launch with ONE D2H;
    // it is the same call the non-MTP decode step makes
    // (`decode_logits_step.rs`).
    //
    // Eligibility is a STRICT SUBSET of the fast-greedy branch's own gate
    // (`sample_step.rs`): greedy temperature, no grammar (its bitmask is
    // host-side), and EXACTLY-neutral penalties — `PenaltyGate::ReduceOnly`
    // is excluded because its per-position immunity check needs a per-row
    // logit read anyway. Any row failing it keeps the per-row call verbatim,
    // so this can only change WHICH kernel produced an identical token.
    // `decode_logits_fp32` models never reach here (`can_batch_bootstrap`).
    // Kill switch `ATLAS_NO_MTP_BOOT_ARGMAX` (PRESENCE).
    let pen: Vec<_> = refs
        .iter()
        .map(|a| {
            crate::scheduler::sample_step::penalty_params_for(
                a,
                crate::scheduler::sample_step::PositionKind::Verify,
                0.0,
                None,
                Vec::new(),
            )
        })
        .collect();
    let greedy: Vec<bool> = refs
        .iter()
        .zip(pen.iter())
        .map(|(a, p)| {
            boot_argmax_batch_enabled()
                && verify_ctx.sampling.fast_greedy_grammar
                && (a.temperature == 0.0 || verify_ctx.sampling.force_temp_zero)
                && a.grammar_state.is_none()
                && crate::scheduler::fast_greedy::classify_penalties(p)
                    == crate::scheduler::fast_greedy::PenaltyGate::Neutral
        })
        .collect();
    let n_greedy = greedy.iter().filter(|&&g| g).count();
    let batch_toks: Option<Vec<u32>> = if n_greedy >= 2 {
        match model.argmax_batch(logits, n, 0) {
            Ok(t) => {
                static LOGGED: std::sync::Once = std::sync::Once::new();
                LOGGED.call_once(|| {
                    tracing::info!(
                        "MTP bootstrap batched argmax ENGAGED (n={n}, greedy_rows={n_greedy}): \
                         one launch + one D2H replaces {n_greedy} single-CTA scans + syncs"
                    );
                });
                Some(t)
            }
            Err(e) => {
                // Never fatal: the per-row path below produces the same token.
                tracing::error!("batched bootstrap argmax_batch (n={n}): {e:#}");
                None
            }
        }
    } else {
        None
    };

    // ── Per-row sample + emit (identical to the per-seq Phase-A body) ──
    // `propose_rows[j] = Some(row)` marks a sequence that still needs drafts.
    let mut propose_rows: Vec<Option<usize>> = vec![None; n];
    for (j, a) in refs.iter_mut().enumerate() {
        let row_logits = logits.offset(j * vocab * elem);
        let batched = batch_toks.as_ref().filter(|_| greedy[j]).map(|t| t[j]);
        let tok = match batched {
            Some(t) => t,
            None => {
                let history = crate::scheduler::sample_step::penalty_history_scope(
                    &a.output_tokens,
                    a.tool_call_end_token,
                )
                .to_vec();
                match sample_token_with_grammar(
                    model,
                    row_logits,
                    a.temperature,
                    a.top_k,
                    a.top_p,
                    &[],
                    a.grammar_state.as_mut(),
                    &pen[j],
                    &history,
                    &verify_ctx.sampling,
                ) {
                    Ok(t) => t,
                    Err(e) => {
                        tracing::error!("batched bootstrap sample error: {e:#}");
                        a.finished = true;
                        continue;
                    }
                }
            }
        };
        let lp = if let Some(k) = a.top_logprobs {
            extract_single_logprobs(model, row_logits, tok, k)
        } else {
            None
        };
        emit_token(a, tok, lp, sched);
        if a.finished {
            continue;
        }
        a.last_token = tok;
        crate::scheduler::adaptive_spec::tick_serial(a, sched);
        // `spec_allowed` mutates re-probe state — evaluate exactly once.
        if crate::scheduler::adaptive_spec::spec_allowed(a, sched) {
            propose_rows[j] = Some(j);
        }
    }

    // ── Stash every proposing row BEFORE any propose ──
    // Each drafter forward writes the shared `hidden_states` buffer, so the
    // live decode rows are gone after the first propose (mtp_multi.rs:165).
    // Stash slot s <-> proposing[s].
    let proposing: Vec<usize> = (0..n).filter(|&j| propose_rows[j].is_some()).collect();
    let mut stash_ok = false;
    if !proposing.is_empty() {
        match model.stash_verify_hidden_rows(&proposing, 0) {
            Ok(()) => stash_ok = true,
            Err(e) => tracing::error!("batched bootstrap stash_verify_hidden_rows: {e:#}"),
        }
    }

    // ── Batched cross-sequence propose, grouped by the model's declared
    // width; grammar-constrained sequences fall back per-seq (their mask is
    // per-position and the batched path is grammarless by contract). ──
    if stash_ok {
        let group_cap = model.mtp_propose_batch_max().max(1);
        // One-shot attribution for "why is propose not batching": each of
        // these can individually keep every sequence on the per-sequence
        // propose, which looks like missing amortisation rather than a
        // declined feature.
        {
            static WHY: std::sync::OnceLock<()> = std::sync::OnceLock::new();
            WHY.get_or_init(|| {
                tracing::debug!(
                    group_cap,
                    proposing = proposing.len(),
                    ladder_nd,
                    "DFlash batched propose gate (first tick)"
                );
            });
        }
        let batchable: Vec<usize> = (0..proposing.len())
            .filter(|&s| {
                let a = &refs[proposing[s]];
                a.grammar_state.is_none()
                    && crate::scheduler::spec_step::effective_drafts_under_grammar(a, ladder_nd)
                        == ladder_nd
            })
            .collect();
        let mut done = vec![false; proposing.len()];
        // D-Cut ranking key: only requested when the lever is armed, so the
        // default path keeps the plain batched argmax and its narrower D2H.
        let want_conf = crate::scheduler::mtp_dcut::dcut_enabled();
        let mut conf: Vec<Vec<f32>> = Vec::new();
        if group_cap >= 2 && ladder_nd >= 1 {
            for group in batchable.chunks(group_cap) {
                if group.len() < 2 {
                    continue;
                }
                let tokens: Vec<u32> = group
                    .iter()
                    .map(|&s| refs[proposing[s]].last_token)
                    .collect();
                let positions: Vec<usize> = group
                    .iter()
                    .map(|&s| refs[proposing[s]].seq.seq_len)
                    .collect();
                let stash_idx: Vec<usize> = group.to_vec();
                let result = {
                    let mut seq_refs: Vec<&mut SequenceState> = Vec::with_capacity(group.len());
                    let mut it = refs.iter_mut();
                    let mut prev = 0usize;
                    for (g, &s) in group.iter().enumerate() {
                        let row = proposing[s];
                        let step = if g == 0 { row } else { row - prev - 1 };
                        let a = it.nth(step).expect("group row within batch");
                        seq_refs.push(&mut a.seq);
                        prev = row;
                    }
                    model.run_mtp_propose_batched(
                        &tokens,
                        &positions,
                        &stash_idx,
                        ladder_nd,
                        &mut seq_refs,
                        0,
                        want_conf.then_some(&mut conf),
                    )
                };
                match result {
                    Ok(Some(all)) => {
                        for (g, &s) in group.iter().enumerate() {
                            if !all[g].is_empty() {
                                refs[proposing[s]].pending_drafts = all[g].clone();
                                refs[proposing[s]].pending_draft_conf =
                                    conf.get(g).cloned().unwrap_or_default();
                            }
                            done[s] = true;
                        }
                    }
                    Ok(None) => {}
                    Err(e) => {
                        // Mid-chain failure: `last_num_drafted` tracks exactly
                        // the drafter rows written, so a SECOND propose on top
                        // would append rows the next trim cannot account for.
                        // Skip these sequences this step. Meta-stride
                        // overflow logs at debug, everything else at ERROR
                        // (see `log_propose_batched_err`).
                        log_propose_batched_err("batched bootstrap run_mtp_propose_batched", &e);
                        for &s in group {
                            done[s] = true;
                        }
                    }
                }
            }
        }
        for (s, &row) in proposing.iter().enumerate() {
            if done[s] {
                continue;
            }
            if let Err(e) = model.save_hidden_for_mtp_from_stash(s, 0) {
                tracing::error!("batched bootstrap save_hidden_for_mtp_from_stash({s}): {e:#}");
                continue;
            }
            let a = &mut refs[row];
            let mask = mtp_grammar_mask_for(a);
            let eff = crate::scheduler::spec_step::effective_drafts_under_grammar(a, ladder_nd);
            match model.run_mtp_propose_multi(
                a.last_token,
                a.seq.seq_len,
                eff,
                &mut a.seq,
                0,
                mask.as_deref(),
            ) {
                Ok(d) if !d.is_empty() => a.pending_drafts = d,
                Ok(_) => tracing::warn!("MTP propose returned empty"),
                Err(e) => tracing::error!("run_mtp_propose_multi: {e:#}"),
            }
        }
    }

    for a in refs.iter_mut() {
        if a.finished {
            continue;
        }
        if let Err(e) = model.start_checkpoint_async(&mut a.seq) {
            tracing::error!("batched bootstrap start_checkpoint_async: {e:#}");
        }
    }

    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| {
        tracing::info!(
            "MTP batched bootstrap ENGAGED (n={n}): one decode_batch + batched propose \
             replaces {n} M=1 decodes + {n} drafter forwards per draft position"
        );
    });
}
