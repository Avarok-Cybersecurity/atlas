// SPDX-License-Identifier: AGPL-3.0-only

//! NLLB-200 / M2M-100 encoder-decoder forward pass (CPU, fp32).
//!
//! Faithful port of HuggingFace `M2M100ForConditionalGeneration`:
//! pre-norm layers, learned-free sinusoidal absolute positions (offset 2,
//! `padding_idx` row zeroed), `sqrt(d_model)` embedding scale, ReLU FFN,
//! biased LayerNorm and biased q/k/v/out/fc projections, decoder cross-
//! attention over the encoder output, and tied `lm_head`.

use std::path::Path;

use anyhow::Result;

use crate::config::NllbConfig;
use crate::lora::LoraSet;
use crate::ops;
use crate::weights::{Tensor, WeightStore};

mod beam;

pub struct NllbModel {
    pub cfg: NllbConfig,
    w: WeightStore,
    /// Optional PEFT LoRA adapter applied as a runtime delta on every adapted
    /// projection (encoder + decoder self/cross attention + FFN). `None` = the
    /// base checkpoint, byte-identical to the pre-LoRA path.
    lora: Option<LoraSet>,
}

/// Cached cross-attention K/V (projected encoder output), one entry per
/// decoder layer — invariant across decode steps.
struct CrossKv {
    k: Vec<f32>, // [src_len, d_model]
    v: Vec<f32>, // [src_len, d_model]
}

impl NllbModel {
    pub fn load_dir(dir: &Path) -> Result<Self> {
        let cfg = NllbConfig::from_json(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let w = WeightStore::load_dir(dir)?;
        Ok(Self { cfg, w, lora: None })
    }

    /// Load the base checkpoint and attach a PEFT LoRA adapter from `lora_dir`.
    pub fn load_dir_with_lora(dir: &Path, lora_dir: &Path) -> Result<Self> {
        let mut m = Self::load_dir(dir)?;
        m.lora = Some(LoraSet::load_dir(lora_dir)?);
        Ok(m)
    }

    /// Attach (or replace) the LoRA adapter on an already-loaded model.
    pub fn set_lora(&mut self, lora: Option<LoraSet>) {
        self.lora = lora;
    }

    /// Number of modules the attached adapter overrides (0 if none).
    pub fn lora_modules(&self) -> usize {
        self.lora.as_ref().map_or(0, LoraSet::adapted_modules)
    }

    fn d(&self) -> usize {
        self.cfg.d_model
    }

    /// `y = x·Wᵀ + b` plus the LoRA residual for `module` when an adapter
    /// overrides it. `module` is the weight path minus `.weight`
    /// (e.g. `model.encoder.layers.0.self_attn.q_proj`); the base weight is
    /// `{module}.weight` and the bias `{module}.bias`.
    fn linear_lora(
        &self,
        module: &str,
        x: &[f32],
        rows: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Vec<f32> {
        let mut y = ops::linear(
            x,
            rows,
            in_dim,
            &self.g(&format!("{module}.weight")).data,
            out_dim,
            Some(&self.g(&format!("{module}.bias")).data),
        );
        if let Some(lora) = &self.lora {
            if let Some(delta) = lora.delta(module, x, rows) {
                ops::add_inplace(&mut y, &delta);
            }
        }
        y
    }

    // ---- embeddings --------------------------------------------------------

    fn embed_table(&self) -> &Tensor {
        // Tied: the converted checkpoint stores only `model.shared.weight`.
        self.w
            .get_any(&[
                "model.shared.weight",
                "model.encoder.embed_tokens.weight",
                "model.decoder.embed_tokens.weight",
            ])
            .expect("embedding table")
    }

    /// Scaled token embeddings + sinusoidal positions → `[seq, d_model]`.
    fn embed_with_positions(&self, ids: &[u32]) -> Vec<f32> {
        let d = self.d();
        let table = self.embed_table();
        let scale = self.cfg.embed_scale();
        let pad = self.cfg.pad_token_id;
        let seq = ids.len();
        let mut h = vec![0f32; seq * d];

        // Masked incremental position ids (fairseq/M2M100 convention):
        // non-pad positions start at padding_idx + 1; pad tokens map to the
        // zeroed padding_idx row.
        let mut running = 0u32;
        for (i, &id) in ids.iter().enumerate() {
            let row = &mut h[i * d..i * d + d];
            let e = &table.data[id as usize * d..id as usize * d + d];
            for j in 0..d {
                row[j] = e[j] * scale;
            }
            let pos = if id != pad {
                running += 1;
                running + pad
            } else {
                pad
            };
            if pos != pad {
                add_sinusoid(row, pos as usize, d);
            }
        }
        h
    }

    // ---- attention ---------------------------------------------------------

    /// Multi-head attention. `q_in`/`kv_in` are `[*, d_model]`. When
    /// `causal`, position `i` may only attend to `j <= i` (decoder self-attn).
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        prefix: &str,
        q_in: &[f32],
        kv_in: &[f32],
        causal: bool,
        precomputed_kv: Option<&CrossKv>,
    ) -> Vec<f32> {
        let d = self.d();
        let heads = self.cfg.encoder_attention_heads; // enc == dec heads
        let hd = self.cfg.head_dim();
        let scaling = (hd as f32).powf(-0.5);
        let tq = q_in.len() / d;
        let tk = match precomputed_kv {
            Some(ckv) => ckv.k.len() / d,
            None => kv_in.len() / d,
        };

        let mut q = self.linear_lora(&format!("{prefix}.q_proj"), q_in, tq, d, d);
        for v in q.iter_mut() {
            *v *= scaling;
        }
        let (k, val) = match precomputed_kv {
            Some(ckv) => (ckv.k.clone(), ckv.v.clone()),
            None => (
                self.linear_lora(&format!("{prefix}.k_proj"), kv_in, tk, d, d),
                self.linear_lora(&format!("{prefix}.v_proj"), kv_in, tk, d, d),
            ),
        };

        // Per-head scaled dot-product attention → context [tq, d].
        let mut ctx = vec![0f32; tq * d];
        for h in 0..heads {
            let off = h * hd;
            for i in 0..tq {
                let qi = &q[i * d + off..i * d + off + hd];
                let kmax = if causal { i + 1 } else { tk };
                let mut scores = vec![f32::NEG_INFINITY; tk];
                for j in 0..kmax {
                    let kj = &k[j * d + off..j * d + off + hd];
                    let mut s = 0f32;
                    for t in 0..hd {
                        s += qi[t] * kj[t];
                    }
                    scores[j] = s;
                }
                ops::softmax_rows_inplace(&mut scores, 1, tk);
                let out = &mut ctx[i * d + off..i * d + off + hd];
                for j in 0..kmax {
                    let p = scores[j];
                    if p == 0.0 {
                        continue;
                    }
                    let vj = &val[j * d + off..j * d + off + hd];
                    for t in 0..hd {
                        out[t] += p * vj[t];
                    }
                }
            }
        }
        self.linear_lora(&format!("{prefix}.out_proj"), &ctx, tq, d, d)
    }

    fn ffn(&self, x: &[f32], rows: usize, layer_prefix: &str, ffn_dim: usize) -> Vec<f32> {
        let d = self.d();
        let mut h = self.linear_lora(&format!("{layer_prefix}.fc1"), x, rows, d, ffn_dim);
        ops::relu_inplace(&mut h);
        self.linear_lora(&format!("{layer_prefix}.fc2"), &h, rows, ffn_dim, d)
    }

    fn ln(&self, x: &mut [f32], rows: usize, prefix: &str) {
        let d = self.d();
        ops::layer_norm_inplace(
            x,
            rows,
            d,
            &self.g(&format!("{prefix}.weight")).data,
            &self.g(&format!("{prefix}.bias")).data,
        );
    }

    fn g(&self, name: &str) -> &Tensor {
        self.w.get(name).expect(name)
    }

    // ---- encoder -----------------------------------------------------------

    /// Run the encoder over `input_ids` → `[src_len, d_model]`.
    pub fn encode(&self, input_ids: &[u32]) -> Vec<f32> {
        let seq = input_ids.len();
        let mut h = self.embed_with_positions(input_ids);
        for l in 0..self.cfg.encoder_layers {
            let lp = format!("model.encoder.layers.{l}");
            // self-attention block (pre-norm, bidirectional)
            let mut normed = h.clone();
            self.ln(&mut normed, seq, &format!("{lp}.self_attn_layer_norm"));
            let attn = self.attention(&format!("{lp}.self_attn"), &normed, &normed, false, None);
            ops::add_inplace(&mut h, &attn);
            // FFN block (pre-norm)
            let mut normed = h.clone();
            self.ln(&mut normed, seq, &format!("{lp}.final_layer_norm"));
            let ff = self.ffn(&normed, seq, &lp, self.cfg.encoder_ffn_dim);
            ops::add_inplace(&mut h, &ff);
        }
        self.ln(&mut h, seq, "model.encoder.layer_norm");
        h
    }

    fn precompute_cross_kv(&self, enc_out: &[f32]) -> Vec<CrossKv> {
        let d = self.d();
        let src = enc_out.len() / d;
        (0..self.cfg.decoder_layers)
            .map(|l| {
                let p = format!("model.decoder.layers.{l}.encoder_attn");
                CrossKv {
                    k: self.linear_lora(&format!("{p}.k_proj"), enc_out, src, d, d),
                    v: self.linear_lora(&format!("{p}.v_proj"), enc_out, src, d, d),
                }
            })
            .collect()
    }

    // ---- decoder -----------------------------------------------------------

    /// One decoder forward over the full current token sequence → decoder
    /// hidden states `[dec_len, d_model]` (before lm_head).
    fn decode_hidden(&self, dec_ids: &[u32], cross_kv: &[CrossKv]) -> Vec<f32> {
        let seq = dec_ids.len();
        let mut h = self.embed_with_positions(dec_ids);
        for l in 0..self.cfg.decoder_layers {
            let lp = format!("model.decoder.layers.{l}");
            // causal self-attention
            let mut normed = h.clone();
            self.ln(&mut normed, seq, &format!("{lp}.self_attn_layer_norm"));
            let sa = self.attention(&format!("{lp}.self_attn"), &normed, &normed, true, None);
            ops::add_inplace(&mut h, &sa);
            // cross-attention over encoder output
            let mut normed = h.clone();
            self.ln(&mut normed, seq, &format!("{lp}.encoder_attn_layer_norm"));
            let ca = self.attention(
                &format!("{lp}.encoder_attn"),
                &normed,
                &[],
                false,
                Some(&cross_kv[l]),
            );
            ops::add_inplace(&mut h, &ca);
            // FFN
            let mut normed = h.clone();
            self.ln(&mut normed, seq, &format!("{lp}.final_layer_norm"));
            let ff = self.ffn(&normed, seq, &lp, self.cfg.decoder_ffn_dim);
            ops::add_inplace(&mut h, &ff);
        }
        self.ln(&mut h, seq, "model.decoder.layer_norm");
        h
    }

    /// Logits for the last decoder position (tied lm_head).
    fn last_logits(&self, dec_hidden: &[f32]) -> Vec<f32> {
        let d = self.d();
        let last = &dec_hidden[dec_hidden.len() - d..];
        let table = self.embed_table();
        ops::linear(last, 1, d, &table.data, self.cfg.vocab_size, None)
    }

    // ---- generation --------------------------------------------------------

    /// Greedy translate: encode `input_ids`, seed the decoder with
    /// `decoder_start_token_id`, force `forced_bos_token_id` (target language)
    /// as the first generated token, then argmax until EOS or `max_new`.
    /// Returns generated token ids (excluding the decoder-start token).
    pub fn generate(&self, input_ids: &[u32], forced_bos: u32, max_new: usize) -> Vec<u32> {
        let enc_out = self.encode(input_ids);
        let cross_kv = self.precompute_cross_kv(&enc_out);
        let eos = self.cfg.eos_token_id;
        let mut dec = vec![self.cfg.decoder_start_token_id];
        let mut generated = Vec::new();
        for step in 0..max_new {
            let hidden = self.decode_hidden(&dec, &cross_kv);
            let next = if step == 0 {
                forced_bos
            } else {
                let logits = self.last_logits(&hidden);
                ops::argmax(&logits) as u32
            };
            dec.push(next);
            generated.push(next);
            if next == eos {
                break;
            }
        }
        generated
    }
}

/// Add the M2M-100 sinusoidal position embedding for `pos` into `row`.
///
/// `emb[j] = sin(pos * f_j)` for `j < d/2`, `cos(pos * f_j)` for the upper
/// half, with `f_j = exp(-j * ln(10000)/(d/2 - 1))`.
fn add_sinusoid(row: &mut [f32], pos: usize, d: usize) {
    let half = d / 2;
    let emb_scale = (10000f32.ln()) / (half as f32 - 1.0);
    let p = pos as f32;
    for j in 0..half {
        let freq = (-(j as f32) * emb_scale).exp();
        let angle = p * freq;
        row[j] += angle.sin();
        row[half + j] += angle.cos();
    }
}
