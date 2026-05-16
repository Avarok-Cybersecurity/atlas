// SPDX-License-Identifier: AGPL-3.0-only

//! Layer A — DEER (Dynamic Early Exit of Reasoning, arXiv 2504.15895).
//!
//! When the model looks confidently done reasoning inside `<think>`, run a
//! bounded *trial* decode: force `</think>`, then `DEER_PROBE_WINDOW` probe
//! steps, and measure the **min** top-1 softmax confidence of the would-be
//! answer (weakest-link, conservative). The trial is ALWAYS rolled back —
//! KV via `seq_len` rewind (the proven self-spec/MTP-reject contract,
//! `spec_step.rs:36-61`), GDN recurrent state via
//! `rollback_ssm_states(seq, 0)` (the well-defined `num_accepted==0`
//! checkpoint-restore branch, `verify_a.rs:306-315`). A "commit" merely sets
//! `force_end_thinking` so the EXISTING force path
//! (`decode_logits_seq.rs:214-226`) emits `</think>` on the next real step —
//! no new emission path, Component P / finish_reason untouched.
//!
//! Single-stream only: the caller guarantees `active.len()==1` (a per-seq
//! trial decode mid-batch would corrupt other sequences' shared decode).
//! MTP is already disabled while `inside_thinking` (`mod.rs:303`), so DEER
//! and MTP never interact. Inert unless `enable_deer` (other models /
//! tool / grammar paths byte-identical).

use super::*;
use spark_runtime::gpu::DevicePtr;

/// Run the DEER probe for a single thinking sequence. May set
/// `a.force_end_thinking`. Never emits tokens; never keeps trial state.
pub fn deer_probe(model: &dyn Model, a: &mut ActiveSeq, think_end_token: Option<u32>) {
    if !deer_enabled() {
        return;
    }
    // Determinism harness: full short-circuit (no probe at all). Paired
    // with ATLAS_DEER_FORCE_ROLLBACK on the SAME image — equal output
    // proves the checkpoint/trial/rollback is byte-lossless. Single-image
    // determinism gate (bench/reasoning_eval.py / gate_pipeline.sh).
    if std::env::var_os("ATLAS_DEER_DISABLE").is_some() {
        return;
    }
    let Some(te) = think_end_token else {
        return;
    };
    if !a.inside_thinking
        || a.force_end_thinking
        || a.require_tool_call
        || a.grammar_state.is_some()
    {
        return;
    }
    // Same mid-artifact interlock as Component G: never cut an open code
    // fence / HTML document (the structural-completeness safety gate).
    if thinking_artifact_open(
        &a.output_tokens,
        a.inside_thinking,
        a.require_tool_call,
        a.grammar_state.is_some(),
    ) {
        return;
    }
    let min_think = deer_min_thinking_tokens(a.min_thinking_tokens as u32);
    if a.thinking_tokens < min_think {
        return;
    }
    // Cheap pre-filter: only probe once F2's confidence streak already says
    // the model looks done — the bounded trial is non-free, run it rarely.
    if a.consecutive_confident < DEER_TRIGGER_STREAK {
        return;
    }

    let pre_seq_len = a.seq.seq_len;
    let pre_tokens = a.seq.tokens.len();
    if let Err(e) = model.checkpoint_ssm_states(&mut a.seq) {
        tracing::warn!("DEER: checkpoint_ssm_states failed ({e:#}); skipping probe");
        return;
    }

    let vocab = model.vocab_size();
    let mut min_conf = f32::INFINITY;
    let mut tok = te; // feed </think> first; its logits = first answer token
    let mut probe_ok = true;
    for _ in 0..DEER_PROBE_WINDOW {
        let lp = match model.decode(tok, &mut a.seq, 0) {
            Ok(l) => l,
            Err(e) => {
                tracing::warn!("DEER: trial decode failed ({e:#})");
                probe_ok = false;
                break;
            }
        };
        match deer_top1_conf_and_argmax(model, lp, vocab) {
            Some((conf, next)) => {
                min_conf = min_conf.min(conf);
                tok = next;
            }
            None => {
                probe_ok = false;
                break;
            }
        }
    }

    // ALWAYS roll back the trial. A commit only *allows* `</think>` on the
    // next real step; it never keeps probe tokens.
    let rb = model.rollback_ssm_states(&mut a.seq, 0);
    a.seq.seq_len = pre_seq_len;
    a.seq.tokens.truncate(pre_tokens);
    if let Err(e) = rb {
        tracing::error!(
            "DEER: rollback_ssm_states failed ({e:#}) — GDN state may be corrupt; \
             finishing seq to avoid silent corruption"
        );
        a.finished = true;
        return;
    }

    // Determinism harness: force the rollback branch unconditionally so
    // `enable_deer=true`+this == baseline token-for-token (proves the
    // checkpoint/rewind is lossless). See bench/reasoning_eval.py.
    let force_rollback = std::env::var_os("ATLAS_DEER_FORCE_ROLLBACK").is_some();
    let gamma = deer_confidence_threshold();
    if probe_ok && !force_rollback && min_conf >= gamma {
        a.force_end_thinking = true;
        tracing::info!(
            "DEER: commit early </think> — min top-1 conf {min_conf:.3} ≥ γ={gamma:.3} \
             after {} thinking tokens",
            a.thinking_tokens,
        );
    } else if probe_ok {
        tracing::debug!(
            "DEER: continue thinking — min top-1 conf {min_conf:.3} < γ={gamma:.3} \
             (force_rollback={force_rollback})"
        );
    }
}

/// Top-1 softmax probability + argmax token from one logits row. Handles
/// the bf16 (default) and fp32 (Gemma-4 only) device layouts like the
/// existing sampler paths.
fn deer_top1_conf_and_argmax(
    model: &dyn Model,
    logits_ptr: DevicePtr,
    vocab: usize,
) -> Option<(f32, u32)> {
    let fp32 = model.logits_ptr_is_fp32(logits_ptr);
    let elem = if fp32 { 4 } else { 2 };
    let mut buf = vec![0u8; vocab * elem];
    if model.copy_logits_to_host(logits_ptr, &mut buf).is_err() {
        return None;
    }
    let mut logits = vec![0.0f32; vocab];
    let mut max_l = f32::NEG_INFINITY;
    let mut argmax = 0u32;
    for (j, slot) in logits.iter_mut().enumerate() {
        let v = if fp32 {
            f32::from_le_bytes([buf[j * 4], buf[j * 4 + 1], buf[j * 4 + 2], buf[j * 4 + 3]])
        } else {
            bf16_to_f32(buf[j * 2], buf[j * 2 + 1])
        };
        *slot = v;
        if v > max_l {
            max_l = v;
            argmax = j as u32;
        }
    }
    let sum: f32 = logits.iter().map(|&l| (l - max_l).exp()).sum();
    let conf = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    Some((conf, argmax))
}
