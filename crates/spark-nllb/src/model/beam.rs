// SPDX-License-Identifier: AGPL-3.0-only

//! NLLB beam search (HF BeamSearchScorer) — split out of model.rs for the
//! 500-LoC cap.

use super::*;

impl NllbModel {
    /// Beam-search translate — faithful to HuggingFace `BeamSearchScorer`
    /// (length_penalty, `early_stopping=false` heuristic, `2*num_beams`
    /// candidate expansion, EOS finalization). `num_beams = 1` degrades to
    /// greedy. Returns generated token ids (excluding the decoder-start
    /// token); the winning hypothesis has EOS appended to match HF's
    /// `finalize`. NLLB defaults: `num_beams=5`, `length_penalty=1.0`,
    /// `early_stopping=false`.
    pub fn generate_beam(
        &self,
        input_ids: &[u32],
        forced_bos: u32,
        num_beams: usize,
        max_new: usize,
        length_penalty: f32,
        early_stopping: bool,
    ) -> Vec<u32> {
        if num_beams <= 1 {
            return self.generate(input_ids, forced_bos, max_new);
        }
        let enc_out = self.encode(input_ids);
        let cross_kv = self.precompute_cross_kv(&enc_out);
        let eos = self.cfg.eos_token_id;
        let start = self.cfg.decoder_start_token_id;

        // Post-forced-BOS state: every beam is [start, forced_bos]. Only beam 0
        // carries score 0 so the first real branching step draws distinct
        // tokens from a single beam (HF initialises beam_scores = [0,-inf,…]).
        // forced_bos contributes 0 to the running sum-logprob (prob 1).
        let mut running: Vec<(Vec<u32>, f32)> = (0..num_beams)
            .map(|b| {
                (
                    vec![start, forced_bos],
                    if b == 0 { 0.0 } else { f32::NEG_INFINITY },
                )
            })
            .collect();
        let mut hyps = BeamHyps::new(num_beams, length_penalty);

        // `max_new` counts generated tokens (excl. decoder-start); we already
        // emitted forced_bos, so the loop may add up to `max_new - 1` more.
        for _ in 1..max_new {
            let cur_len = running[0].0.len(); // includes decoder-start
            // Expand: top 2*num_beams (beam, token, score) candidates overall.
            let mut cands: Vec<(f32, usize, u32)> = Vec::new();
            for (b, (toks, bscore)) in running.iter().enumerate() {
                if !bscore.is_finite() {
                    continue; // dead init beam contributes nothing
                }
                let hidden = self.decode_hidden(toks, &cross_kv);
                let mut logp = self.last_logits(&hidden);
                ops::log_softmax_inplace(&mut logp);
                for tok in ops::top_k_indices(&logp, 2 * num_beams) {
                    cands.push((bscore + logp[tok], b, tok as u32));
                }
            }
            cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

            // Select num_beams continuing beams; EOS candidates finalize a hyp.
            let mut next: Vec<(Vec<u32>, f32)> = Vec::with_capacity(num_beams);
            for (score, b, tok) in cands {
                if next.len() == num_beams {
                    break;
                }
                if tok == eos {
                    hyps.add(running[b].0.clone(), score);
                } else {
                    let mut nt = running[b].0.clone();
                    nt.push(tok);
                    next.push((nt, score));
                }
            }
            let best_running = next[0].1;
            running = next;

            if hyps.is_done(best_running, cur_len, early_stopping) {
                break;
            }
        }

        // Finalize: fold surviving beams into the pool, take the best.
        for (toks, score) in &running {
            if score.is_finite() {
                hyps.add(toks.clone(), *score);
            }
        }
        let mut best = hyps.best().unwrap_or_else(|| running[0].0.clone());
        best.push(eos); // HF finalize appends EOS
        best[1..].to_vec() // strip decoder-start
    }
}

/// Bounded pool of the best `num_beams` finished hypotheses, scored by
/// length-normalised sum-logprob (HF `BeamHypotheses`).
struct BeamHyps {
    num_beams: usize,
    length_penalty: f32,
    beams: Vec<(Vec<u32>, f32)>, // (tokens incl. decoder-start, normalised score)
}

impl BeamHyps {
    fn new(num_beams: usize, length_penalty: f32) -> Self {
        Self {
            num_beams,
            length_penalty,
            beams: Vec::new(),
        }
    }

    fn norm(&self, tokens_len: usize, sum_logprob: f32) -> f32 {
        sum_logprob / (tokens_len as f32).powf(self.length_penalty)
    }

    fn worst(&self) -> f32 {
        self.beams
            .iter()
            .map(|(_, s)| *s)
            .fold(f32::INFINITY, f32::min)
    }

    fn add(&mut self, tokens: Vec<u32>, sum_logprob: f32) {
        let score = self.norm(tokens.len(), sum_logprob);
        if self.beams.len() < self.num_beams || score > self.worst() {
            self.beams.push((tokens, score));
            if self.beams.len() > self.num_beams {
                // Drop the current worst.
                let (wi, _) = self
                    .beams
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.1.partial_cmp(&b.1.1).unwrap())
                    .unwrap();
                self.beams.swap_remove(wi);
            }
        }
    }

    /// HF `is_done`: with `early_stopping=false`, stop once the worst kept
    /// hypothesis already beats the best score any running beam could still
    /// reach at this length.
    fn is_done(&self, best_running_sum_logprob: f32, cur_len: usize, early_stopping: bool) -> bool {
        if self.beams.len() < self.num_beams {
            return false;
        }
        if early_stopping {
            return true;
        }
        let highest_attainable =
            best_running_sum_logprob / (cur_len as f32).powf(self.length_penalty);
        self.worst() >= highest_attainable
    }

    fn best(&self) -> Option<Vec<u32>> {
        self.beams
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(t, _)| t.clone())
    }
}
