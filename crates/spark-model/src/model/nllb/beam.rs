// SPDX-License-Identifier: AGPL-3.0-only

//! Batched beam search (NLLB default `num_beams=5`) for the served model. The B
//! beams of a request decode as one M=B batch per step ([`super::beam_compute`]);
//! candidate expansion / pruning / hypothesis bookkeeping run on the host. Ported
//! from the milestone-7 `nllb_cuda_beambatch` example (token-exact vs HF-bf16),
//! faithful to the CPU reference in `spark-nllb`.

use anyhow::{Context, Result};
use spark_runtime::gpu::DevicePtr;

use super::NllbGpuModel;
use super::kv::NllbSeqKv;
use super::util::{bf16_bytes, decoder_pos_table_bf16, u32_bytes};
use crate::traits::BeamReq;

/// The fused batch's cross-KV for grouped single-launch cross-attention: `kpad`/
/// `vpad` are per-decoder-layer padded `[C, max_enc, d]` buffers (one slab per
/// request). Cross-attn issues ONE launch/layer; row `i` reads slab `i / group`
/// (its request) bounded to its own enc_len by the per-row `crosstk`. For a
/// single request (C=1) the un-padded `[enc_len,d]` buffers are passed directly
/// with `group = num_beams` (so `i / group == 0` for every beam — no padding).
pub(super) struct CrossBatch<'a> {
    pub kpad: &'a [DevicePtr],
    pub vpad: &'a [DevicePtr],
    pub group: usize,
    pub max_enc: usize,
}

/// Max K (= 2·num_beams) the on-device beam top-k kernel supports; sizes the
/// `topk_*` scratch and caps the fused-beam count (num_beams ≤ 16).
pub(super) const NLLB_TOPK_KMAX: usize = 32;

/// Batched decode scratch for B beams (bf16, except the u32 id/length buffers).
pub(super) struct DecBuf {
    pub dh: DevicePtr,
    pub normed: DevicePtr,
    pub q: DevicePtr,
    pub knew: DevicePtr,
    pub vnew: DevicePtr,
    pub attn: DevicePtr,
    pub proj: DevicePtr,
    pub ff: DevicePtr,
    pub logits: DevicePtr,
    pub id: DevicePtr,
    pub selftk: DevicePtr,
    pub crosstk: DevicePtr,
    pub pos_table: DevicePtr,
    pub cache_rows: usize,
    /// Phase-d on-device top-k outputs: per row `lse` (f32), and `K` `(val,tok)`
    /// pairs in `topk_val`/`topk_idx` (`[rows, NLLB_TOPK_KMAX]`).
    pub topk_lse: DevicePtr,
    pub topk_val: DevicePtr,
    pub topk_idx: DevicePtr,
}

impl DecBuf {
    /// `crosstk_host[row]` is that row's cross key length (its request's
    /// `enc_len`); `b` is the total row count (Σ beams over all fused requests).
    pub(super) fn new(
        m: &NllbGpuModel,
        b: usize,
        crosstk_host: &[u32],
        cache_rows: usize,
    ) -> Result<Self> {
        let (d, gpu) = (m.d, m.gpu.as_ref());
        debug_assert_eq!(crosstk_host.len(), b);
        let crosstk = gpu.alloc(b * 4)?;
        gpu.copy_h2d(u32_bytes(crosstk_host), crosstk)?;
        let pos_table = gpu.alloc(cache_rows * d * 2)?;
        gpu.copy_h2d(
            bf16_bytes(&decoder_pos_table_bf16(cache_rows, d)),
            pos_table,
        )?;
        Ok(Self {
            dh: gpu.alloc(b * d * 2)?,
            normed: gpu.alloc(b * d * 2)?,
            q: gpu.alloc(b * d * 2)?,
            knew: gpu.alloc(b * d * 2)?,
            vnew: gpu.alloc(b * d * 2)?,
            attn: gpu.alloc(b * d * 2)?,
            proj: gpu.alloc(b * d * 2)?,
            ff: gpu.alloc(b * m.ffn * 2)?,
            logits: gpu.alloc(b * m.vocab * 2)?,
            id: gpu.alloc(b * 4)?,
            selftk: gpu.alloc(b * 4)?,
            crosstk,
            pos_table,
            cache_rows,
            topk_lse: gpu.alloc(b * 4)?,
            topk_val: gpu.alloc(b * NLLB_TOPK_KMAX * 4)?,
            topk_idx: gpu.alloc(b * NLLB_TOPK_KMAX * 4)?,
        })
    }
    pub(super) fn free(self, gpu: &dyn spark_runtime::gpu::GpuBackend) -> Result<()> {
        for p in [
            self.dh,
            self.normed,
            self.q,
            self.knew,
            self.vnew,
            self.attn,
            self.proj,
            self.ff,
            self.logits,
            self.id,
            self.selftk,
            self.crosstk,
            self.pos_table,
            self.topk_lse,
            self.topk_val,
            self.topk_idx,
        ] {
            gpu.free(p)?;
        }
        Ok(())
    }
}

pub(super) struct Beam {
    pub tokens: Vec<u32>,
    pub score: f32,
    pub logits: Vec<f32>,
}

/// Finished-hypothesis pool (length-penalty scored), HF `BeamHypotheses`.
pub(super) struct BeamHyps {
    num_beams: usize,
    lp: f32,
    beams: Vec<(Vec<u32>, f32)>,
}

impl BeamHyps {
    pub(super) fn new(num_beams: usize, lp: f32) -> Self {
        Self {
            num_beams,
            lp,
            beams: Vec::new(),
        }
    }
    fn worst(&self) -> f32 {
        self.beams
            .iter()
            .map(|(_, s)| *s)
            .fold(f32::INFINITY, f32::min)
    }
    pub(super) fn add(&mut self, tokens: Vec<u32>, sum_logprob: f32) {
        let score = sum_logprob / (tokens.len() as f32).powf(self.lp);
        if self.beams.len() < self.num_beams || score > self.worst() {
            self.beams.push((tokens, score));
            if self.beams.len() > self.num_beams {
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
    pub(super) fn is_done(&self, best_running: f32, cur_len: usize, early_stopping: bool) -> bool {
        if self.beams.len() < self.num_beams {
            return false;
        }
        if early_stopping {
            return true;
        }
        self.worst() >= best_running / (cur_len as f32).powf(self.lp)
    }
    pub(super) fn best(&self) -> Option<Vec<u32>> {
        self.beams
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(t, _)| t.clone())
    }
}

pub(super) fn logsumexp(x: &[f32]) -> f32 {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    m + x.iter().map(|&v| (v - m).exp()).sum::<f32>().ln()
}

pub(super) fn top_k(x: &[f32], k: usize) -> Vec<(f32, usize)> {
    let mut best: Vec<(f32, usize)> = Vec::with_capacity(k + 1);
    for (i, &v) in x.iter().enumerate() {
        if best.len() < k {
            best.push((v, i));
            if best.len() == k {
                best.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            }
        } else if v > best[k - 1].0 {
            best[k - 1] = (v, i);
            let mut j = k - 1;
            while j > 0 && best[j].0 > best[j - 1].0 {
                best.swap(j, j - 1);
                j -= 1;
            }
        }
    }
    best.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    best
}

impl NllbGpuModel {
    /// Run beam search for one request → the winning hypothesis token ids
    /// (decoder-start stripped, EOS-terminated). Reuses `run_encoder` for the
    /// encoder + cross-KV and applies per-request LoRA via the adapter gate.
    pub(super) fn generate_beam_one(&self, req: &BeamReq) -> Result<Vec<u32>> {
        self.set_lora_active(req.adapter_slot);
        let src_id = if req.src_lang_id != 0 {
            req.src_lang_id
        } else {
            self.lang.src_lang_id
        };
        let tgt_id = if req.tgt_lang_id != 0 {
            req.tgt_lang_id
        } else {
            self.lang.tgt_lang_id
        };
        let src = self.lang.encoder_input_with(src_id, &req.prompt_tokens);

        // Encoder + cross-KV (single [enc_len,d] buffers, shared across beams).
        let gpu = self.gpu.as_ref();
        let mut tmp = NllbSeqKv::new(gpu, self.dec_layers, 2, self.d)?;
        self.run_encoder(&src, &mut tmp)?;
        let enc_len = tmp.enc_len;
        let out = self.beam_batched(req, tgt_id, &tmp.cross_k, &tmp.cross_v, enc_len);
        tmp.free(gpu)?;
        out
    }

    fn beam_batched(
        &self,
        req: &BeamReq,
        forced_bos: u32,
        cross_k: &[DevicePtr],
        cross_v: &[DevicePtr],
        enc_len: usize,
    ) -> Result<Vec<u32>> {
        let b = req.num_beams.max(1);
        let max_new = req.max_new.max(2).min(self.cache_rows);
        let gpu = self.gpu.as_ref();
        let buf = DecBuf::new(self, b, &vec![enc_len as u32; b], max_new)?;
        let alloc_set = |n: usize| {
            (0..n)
                .map(|_| gpu.alloc(b * max_new * self.d * 2))
                .collect::<Result<Vec<_>>>()
        };
        let mut sk = alloc_set(self.dec_layers)?;
        let mut sv = alloc_set(self.dec_layers)?;
        let mut sk2 = alloc_set(self.dec_layers)?;
        let mut sv2 = alloc_set(self.dec_layers)?;
        let perm_dev = gpu.alloc(b * 4)?;

        // Single request (C=1): pass the un-padded [enc_len,d] cross-KV directly
        // with group = num_beams, so every beam's `row / group == 0` reads it.
        let xb = CrossBatch {
            kpad: cross_k,
            vpad: cross_v,
            group: b.max(1),
            max_enc: enc_len,
        };
        let res = self.beam_loop(
            req, b, max_new, forced_bos, &xb, &buf, &mut sk, &mut sv, &mut sk2, &mut sv2, perm_dev,
        );

        for p in sk.into_iter().chain(sv).chain(sk2).chain(sv2) {
            gpu.free(p)?;
        }
        gpu.free(perm_dev)?;
        buf.free(gpu)?;
        res
    }

    #[allow(clippy::too_many_arguments)]
    fn beam_loop(
        &self,
        req: &BeamReq,
        b: usize,
        max_new: usize,
        forced_bos: u32,
        xb: &CrossBatch,
        buf: &DecBuf,
        sk: &mut Vec<DevicePtr>,
        sv: &mut Vec<DevicePtr>,
        sk2: &mut Vec<DevicePtr>,
        sv2: &mut Vec<DevicePtr>,
        perm_dev: DevicePtr,
    ) -> Result<Vec<u32>> {
        let (dec_start, eos) = (self.lang.decoder_start_id, self.lang.eos_id);
        // Seed all beams with [dec_start, forced_bos] into their slots.
        self.beam_forward_step(&vec![dec_start; b], 0, b, sk, sv, xb, buf)?;
        self.beam_forward_step(&vec![forced_bos; b], 1, b, sk, sv, xb, buf)?;
        let init = self.beam_logits_host(buf, b)?;
        let mut beams: Vec<Beam> = (0..b)
            .map(|bi| Beam {
                tokens: vec![dec_start, forced_bos],
                score: if bi == 0 { 0.0 } else { f32::NEG_INFINITY },
                logits: init[bi].clone(),
            })
            .collect();

        let mut hyps = BeamHyps::new(b, req.length_penalty);
        for _ in 1..max_new {
            let cur_len = beams[0].tokens.len();
            let mut cands: Vec<(f32, usize, u32)> = Vec::new();
            for (bi, beam) in beams.iter().enumerate() {
                if !beam.score.is_finite() {
                    continue;
                }
                let lse = logsumexp(&beam.logits);
                for (val, tok) in top_k(&beam.logits, 2 * b) {
                    cands.push((beam.score + (val - lse), bi, tok as u32));
                }
            }
            cands.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());

            let mut perm = vec![0u32; b];
            let (mut new_tokens, mut new_scores, mut cur) = (Vec::new(), Vec::new(), Vec::new());
            for (score, parent, tok) in cands {
                if new_tokens.len() == b {
                    break;
                }
                if tok == eos {
                    hyps.add(beams[parent].tokens.clone(), score);
                } else {
                    perm[new_tokens.len()] = parent as u32;
                    let mut t = beams[parent].tokens.clone();
                    t.push(tok);
                    new_tokens.push(t);
                    new_scores.push(score);
                    cur.push(tok);
                }
            }
            if new_tokens.is_empty() {
                break;
            }
            let best_running = new_scores[0];

            // reorder caches sk2[i] = sk[perm[i]] (rows 0..cur_len), then swap
            self.gpu.copy_h2d(u32_bytes(&perm), perm_dev)?;
            for l in 0..self.dec_layers {
                self.gather(sk[l], sk2[l], perm_dev, b, cur_len, buf.cache_rows)?;
                self.gather(sv[l], sv2[l], perm_dev, b, cur_len, buf.cache_rows)?;
            }
            std::mem::swap(sk, sk2);
            std::mem::swap(sv, sv2);

            self.beam_forward_step(&cur, cur_len, b, sk, sv, xb, buf)?;
            let lh = self.beam_logits_host(buf, b)?;
            beams = (0..cur.len())
                .map(|i| Beam {
                    tokens: new_tokens[i].clone(),
                    score: new_scores[i],
                    logits: lh[i].clone(),
                })
                .collect();
            if hyps.is_done(best_running, cur_len, req.early_stopping) {
                break;
            }
        }
        for beam in &beams {
            if beam.score.is_finite() {
                hyps.add(beam.tokens.clone(), beam.score);
            }
        }
        let mut best = hyps.best().context("nllb beam: no finished hypotheses")?;
        best.push(eos);
        Ok(best[1..].to_vec())
    }
}
