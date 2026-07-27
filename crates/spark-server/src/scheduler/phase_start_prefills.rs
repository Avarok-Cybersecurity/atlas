// SPDX-License-Identifier: AGPL-3.0-only

//! Phase: start new requests — either single-shot prefill (legacy) or
//! chunked prefill that pushes onto `prefilling`. Handles SSM-pool-full
//! preemption.

use spark_model::traits::Model;

use super::*;
use crate::api::InferenceRequest;
use crate::grammar::GrammarEngine;

#[allow(clippy::too_many_arguments)]
pub(super) fn start_new_requests(
    model: &dyn Model,
    new_reqs: Vec<InferenceRequest>,
    chunked: bool,
    always_mixed: bool,
    max_prefill_tokens: usize,
    max_batch_tokens: usize,
    eos_tokens: &[u32],
    prefill_stream: u64,
    prefill_event: u64,
    grammar_engine: &mut Option<GrammarEngine>,
    spontaneous_think_budget: u32,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    active: &mut Vec<ActiveSeq>,
    prefilling: &mut Vec<PrefillInProgress>,
) {
    // Co-dispatch (ATLAS_PREFILL_CODISPATCH=1): when >=2 non-vision requests are
    // co-admitted this tick with no active decode to starve, DEFER their chunk-0
    // prefill so they batch into one forward via run_batched_prefill_step (which
    // sees prefilling.len() >= 2 → can_batch_prefill_only). Vision excluded: a
    // shared prepare_vision_embed buffer would cross-contaminate stacked streams.
    let want_codispatch = std::env::var("ATLAS_PREFILL_CODISPATCH")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
        && chunked
        && new_reqs.len() >= 2
        && active.is_empty()
        && prefilling.is_empty()
        && !model.is_ep()
        && !new_reqs.iter().any(|r| r.has_image_pixels());
    // Always-mixed chunk-0 fuse: when decodes are active and ATLAS_HOLO_ALWAYS_MIXED
    // is on, DEFER a new request's chunk-0 (admit it to `prefilling` with
    // chunk_offset=0, skip the inline blocking prefill) so it runs this SAME tick
    // in continue_in_progress_prefills via the FUSED mixed path. Otherwise chunk-0
    // ran here as a monolithic prefill that froze every active decode for the whole
    // first chunk (the residual ~3.6s burst stall). Mutually exclusive with
    // want_codispatch (which requires active.is_empty()). EP/vision excluded (per
    // request, below) — same constraints as the fused mixed path.
    let mixed_defer = always_mixed && chunked && !active.is_empty() && !model.is_ep();

    // ── Vision co-dispatch pre-pass (ATLAS_VISION_CODISPATCH, default on) ──
    // Batch every single-chunk-fit image request's ViT encode into ONE
    // forward_batched call so each block's GEMM weights are read once over
    // Σpatches instead of N× — the concurrent-image win (serialized ViT made
    // image TTFT grow ~linearly with concurrency). Each request then reads its
    // own slice of the shared packed buf_out via set_vision_slice_base.
    // Default OFF: profiling showed the ViT encode is ~94% the per-image
    // vision_attention_rope kernel (which does NOT batch across images), so
    // co-dispatching the encode gives no TTFT win (the batchable GEMMs are
    // only ~6% of the ViT) and adds gather/fence overhead. Kept as opt-in
    // infrastructure — it correctly slices vision per-request, which is the
    // prerequisite for admitting image requests into LLM-prefill co-dispatch.
    let vision_codispatch = std::env::var("ATLAS_VISION_CODISPATCH")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    const VISION_P_MAX: usize = 6400; // VisionEncoder scratch cap (Σ pre-merge patches)
    let mut vision_slices: Vec<VisionSlice> = vec![VisionSlice::default(); new_reqs.len()];
    if vision_codispatch && chunked {
        let mut batched_idx: Vec<usize> = Vec::new();
        let mut per_request_imgs: Vec<Vec<(Vec<f32>, usize, usize)>> = Vec::new();
        let mut running_patches = 0usize;
        let mut overflow = false;
        for (k, req) in new_reqs.iter().enumerate() {
            if !req.has_image_pixels() {
                continue;
            }
            // Single-chunk-fit gate: the splice + MRoPE reset img_idx per chunk,
            // so a pad run must not straddle a chunk boundary. max_prefill_tokens
            // is the conservative per-request budget (active grows as we admit
            // each request, dropping the budget from max_batch_tokens), so
            // fitting it guarantees a single chunk regardless of admit order.
            if req.prompt_len() > max_prefill_tokens {
                continue; // self-encodes per-request (legacy single-image path)
            }
            let imgs = req.image_pixels_ref();
            let req_patches: usize = imgs.iter().map(|(_, gh, gw)| gh * gw).sum();
            if running_patches + req_patches > VISION_P_MAX {
                overflow = true;
                break;
            }
            running_patches += req_patches;
            batched_idx.push(k);
            per_request_imgs.push(imgs.to_vec());
        }
        if overflow {
            // Full-disable for the tick — never mix a batched encode with a
            // per-request self-encode into the same buf_out (a later self-encode
            // on the default stream would clobber it before earlier splices run).
            batched_idx.clear();
            per_request_imgs.clear();
        }
        if batched_idx.len() >= 2 {
            match model.prepare_vision_embed_batched(&per_request_imgs) {
                Ok(descs) if descs.len() == batched_idx.len() => {
                    for (slot, (row_off, grid_off, n_img, row_cnt)) in
                        batched_idx.iter().zip(descs.into_iter())
                    {
                        vision_slices[*slot] = VisionSlice {
                            patch_row_offset: row_off,
                            grid_index_offset: grid_off,
                            num_images: n_img,
                            patch_row_count: row_cnt,
                        };
                    }
                    // ONE fence for the whole batch: prefill_stream waits for the
                    // batched encode (default stream) before any chunk-0 splice.
                    if let Err(e) = model
                        .record_event(prefill_event, model.default_stream())
                        .and_then(|_| model.stream_wait_event(prefill_stream, prefill_event))
                    {
                        tracing::error!("vision co-dispatch fence failed: {e:#}");
                    }
                    tracing::info!(
                        "Vision co-dispatch: batched {} image requests this tick",
                        batched_idx.len()
                    );
                }
                Ok(_) => {
                    tracing::warn!(
                        "vision co-dispatch desc count mismatch; per-request fallback this tick"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        "vision co-dispatch batched encode failed: {e:#}; per-request fallback"
                    );
                }
            }
        }
    }

    // ── Beam co-dispatch pre-pass (ATLAS_BEAM_CODISPATCH, default on) ──
    // Fuse this tick's beam requests (num_beams>1) into ONE generate_beam_batch
    // call so their Σ beams decode as a single batched forward per step — beam
    // aggregate throughput then scales with concurrency instead of serializing
    // per request. Requests are grouped by adapter (the model's single global
    // LoRA gate forces one adapter per fused batch) and packed up to BEAM_C_CAP
    // total beams. Each winning hypothesis is stashed in `beam_hyps[req_idx]`,
    // which the per-request beam branch consumes instead of re-running the search.
    // Non-beam requests are untouched; a single beam request (group of 1) falls
    // through to the identical per-request path. Blocking-only: streaming + beam
    // is rejected upstream, so every beam request here is Blocking.
    let mut beam_hyps: Vec<Option<Vec<u32>>> = (0..new_reqs.len()).map(|_| None).collect();
    let beam_codispatch = model.supports_beam()
        && std::env::var("ATLAS_BEAM_CODISPATCH")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(true);
    if beam_codispatch {
        const BEAM_C_CAP: usize = 320; // Σ beams per fused batch (d≤2048 self-KV cap)
        let items: Vec<(usize, i32, usize)> = new_reqs
            .iter()
            .enumerate()
            .filter(|(_, r)| r.num_beams() > 1 && matches!(r, InferenceRequest::Blocking { .. }))
            .map(|(idx, r)| (idx, r.adapter_slot(), r.num_beams() as usize))
            .collect();
        for chunk in pack_beam_chunks(&items, BEAM_C_CAP) {
            if chunk.len() < 2 {
                continue; // single beam request → identical per-request path
            }
            let reqs: Vec<spark_model::traits::BeamReq> = chunk
                .iter()
                .map(|&i| {
                    let r = &new_reqs[i];
                    spark_model::traits::BeamReq {
                        prompt_tokens: r.prompt_tokens_arc().as_ref().clone(),
                        src_lang_id: r.src_lang_id(),
                        tgt_lang_id: r.tgt_lang_id(),
                        adapter_slot: r.adapter_slot(),
                        num_beams: r.num_beams() as usize,
                        max_new: r.max_tokens(),
                        length_penalty: r.length_penalty(),
                        early_stopping: r.early_stopping(),
                    }
                })
                .collect();
            let total_beams: usize = reqs.iter().map(|r| r.num_beams).sum();
            match model.generate_beam_batch(&reqs) {
                Ok(hyps) if hyps.len() == chunk.len() => {
                    for (&i, h) in chunk.iter().zip(hyps.into_iter()) {
                        beam_hyps[i] = Some(h);
                    }
                    tracing::info!(
                        "Beam co-dispatch: fused {} requests ({total_beams} beams) this tick",
                        chunk.len(),
                    );
                }
                Ok(_) => {
                    tracing::warn!("beam co-dispatch count mismatch; per-request fallback")
                }
                Err(e) => {
                    tracing::warn!("beam co-dispatch failed: {e:#}; per-request fallback")
                }
            }
        }
    }

    for (req_idx, req) in new_reqs.into_iter().enumerate() {
        let precomputed_beam_hyp = beam_hyps[req_idx].take();
        if chunked {
            let defer = want_codispatch || (mixed_defer && !req.has_image_pixels());
            // Pre-encoded by the co-dispatch pre-pass? (num_images>0 ⇒ batched)
            let slice = vision_slices[req_idx];
            let vision_slice = if slice.num_images > 0 {
                Some(slice)
            } else {
                None
            };
            // When no active sequences are decoding, process as much of the
            // prompt as buffers allow — avoids per-token paged decode fallback
            // in chunk 2+. Capped at max_batch_tokens (buffer capacity).
            let budget = if active.is_empty() && prefilling.is_empty() {
                max_batch_tokens
            } else {
                max_prefill_tokens
            };
            match start_chunked_prefill(
                think_end_token,
                think_start_token,
                tool_call_start_token,
                tool_call_end_token,
                model,
                req,
                eos_tokens,
                budget,
                prefill_stream,
                prefill_event,
                grammar_engine,
                spontaneous_think_budget,
                defer,
                vision_slice,
                precomputed_beam_hyp,
            ) {
                Ok(StartPrefillResult::Active(a)) => {
                    tracing::info!(
                        "Prefilled (single chunk): seq_len={}, remaining={}",
                        a.seq.seq_len,
                        a.remaining,
                    );
                    active.push(a);
                }
                Ok(StartPrefillResult::InProgress(p)) => {
                    tracing::info!(
                        "Prefill chunk 0/{}: {}/{} tokens",
                        p.prompt_tokens.len(),
                        p.chunk_offset,
                        p.prompt_tokens.len(),
                    );
                    prefilling.push(p);
                }
                Ok(StartPrefillResult::Finished) => {} // EOS on first token
                Err(e) => {
                    handle_prefill_start_error(model, &e, active);
                }
            }
        } else {
            // Legacy non-chunked path.
            match prefill_request(
                think_end_token,
                think_start_token,
                tool_call_start_token,
                tool_call_end_token,
                model,
                req,
                eos_tokens,
                grammar_engine,
                spontaneous_think_budget,
                precomputed_beam_hyp,
            ) {
                Ok(Some(a)) => {
                    tracing::info!(
                        "Prefilled: seq_len={}, remaining={}",
                        a.seq.seq_len,
                        a.remaining,
                    );
                    active.push(a);
                }
                Ok(None) => {}
                Err(e) => {
                    handle_prefill_start_error(model, &e, active);
                }
            }
        }
    }
}

/// Group beam requests by adapter — the model's single global LoRA gate forces
/// one adapter per fused batch — then pack each group into chunks whose summed
/// beam count does not exceed `cap`. `items` is `(req_idx, adapter_slot,
/// num_beams)`; returns chunks of `req_idx` (adapter order is deterministic).
fn pack_beam_chunks(items: &[(usize, i32, usize)], cap: usize) -> Vec<Vec<usize>> {
    let mut groups: std::collections::BTreeMap<i32, Vec<(usize, usize)>> =
        std::collections::BTreeMap::new();
    for &(idx, adapter, nb) in items {
        groups.entry(adapter).or_default().push((idx, nb));
    }
    let mut chunks: Vec<Vec<usize>> = Vec::new();
    for reqs in groups.into_values() {
        let (mut chunk, mut beams) = (Vec::new(), 0usize);
        for (idx, nb) in reqs {
            if !chunk.is_empty() && beams + nb > cap {
                chunks.push(std::mem::take(&mut chunk));
                beams = 0;
            }
            chunk.push(idx);
            beams += nb;
        }
        if !chunk.is_empty() {
            chunks.push(chunk);
        }
    }
    chunks
}

/// SSM-pool-full preemption: free oldest active sequence and surface a
/// 503-equivalent error to the preempted request. Mirrors vLLM's
/// preemption strategy — never return HTTP 500 for resource exhaustion.
fn handle_prefill_start_error(model: &dyn Model, e: &anyhow::Error, active: &mut Vec<ActiveSeq>) {
    let err_msg = format!("{e:#}");
    if err_msg.contains("pool exhausted") && !active.is_empty() {
        let victim_idx = active.len() - 1;
        let mut victim = active.swap_remove(victim_idx);
        tracing::warn!(
            "SSM pool full: preempting seq (slot={}, tokens={}) for new request",
            victim.seq.slot_idx,
            victim.output_tokens.len(),
        );
        send_error(model, &mut victim, "Preempted: server resource pressure");
    } else {
        tracing::error!("Prefill start error: {err_msg}");
    }
}

#[cfg(test)]
mod tests {
    use super::pack_beam_chunks;

    #[test]
    fn groups_by_adapter_then_packs_by_cap() {
        // Two adapters (-1, 7), five beams each request. cap=15 ⇒ ≤3 per chunk.
        let items = [
            (0, -1, 5),
            (1, 7, 5),
            (2, -1, 5),
            (3, 7, 5),
            (4, -1, 5),
            (5, 7, 5),
            (6, -1, 5),
            (7, 7, 5),
        ];
        let chunks = pack_beam_chunks(&items, 15);
        // adapter -1: idx 0,2,4,6 → [0,2,4] + [6]; adapter 7: 1,3,5,7 → [1,3,5] + [7].
        assert_eq!(chunks, vec![vec![0, 2, 4], vec![6], vec![1, 3, 5], vec![7]]);
        // No chunk exceeds the cap.
        for c in &chunks {
            let beams: usize = c.iter().map(|&i| items[i].2).sum();
            assert!(beams <= 15, "chunk {c:?} exceeds cap");
        }
    }

    #[test]
    fn single_adapter_all_fit_one_chunk() {
        let items = [(0, -1, 5), (1, -1, 5), (2, -1, 5)];
        assert_eq!(pack_beam_chunks(&items, 320), vec![vec![0, 1, 2]]);
    }

    #[test]
    fn heterogeneous_beam_counts_respect_cap() {
        // 8+8+4 with cap 16 ⇒ [8,8] then [4] (adding the 4 would hit 20>16).
        let items = [(0, -1, 8), (1, -1, 8), (2, -1, 4)];
        assert_eq!(pack_beam_chunks(&items, 16), vec![vec![0, 1], vec![2]]);
    }

    #[test]
    fn empty_items_no_chunks() {
        assert!(pack_beam_chunks(&[], 320).is_empty());
    }
}
