// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the full-LM oracle of the CPU encoder. It runs every layer of the stock Qwen3.5-0.8B checkpoint plus
//! the tied LM head through the same CPU forward the readouts use, and measures what a correct forward must show:
//! single or low double digit perplexity on ordinary text, fluent greedy continuations, and (against an ARC-1.0 release)
//! byte-identical `model.language_model.*` tensors and an identical layer-12 readout. Nothing here touches a GPU.

use std::path::Path;
use std::time::Instant;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::encoder::{Encoder, EncoderOptions, RopeConfig, RopeVariant, weights_file};
use super::error::{GResult, GateError, value_err};
use super::lm_texts::{PROMPTS, TEXTS};
use super::tensorio::TensorFile;

/// The text stack's tensor prefix in both the release and the stock checkpoint.
pub const LM_PREFIX: &str = "model.language_model.";

/// The layer count the ARC readout is taken after (`block_11`, the first 12 layers).
pub const READOUT_DEPTH: usize = 12;

/// The stock `tokenizer.json` (no special tokens added on encode, none skipped on decode).
pub struct Tok(tokenizers::Tokenizer);

impl Tok {
    /// Loads `dir/tokenizer.json`.
    pub fn load(dir: &Path) -> GResult<Self> {
        let p = dir.join("tokenizer.json");
        tokenizers::Tokenizer::from_file(&p)
            .map(Self)
            .map_err(|e| GateError::Value(format!("{}: {e}", p.display())))
    }

    /// Token ids of `text`.
    pub fn encode(&self, text: &str) -> GResult<Vec<u32>> {
        self.0
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| GateError::Value(format!("tokenizer: {e}")))
    }

    /// Text of `ids`.
    pub fn decode(&self, ids: &[u32]) -> GResult<String> {
        self.0
            .decode(ids, false)
            .map_err(|e| GateError::Value(format!("tokenizer: {e}")))
    }

    /// Id of a special token, when the vocabulary has it.
    pub fn id(&self, token: &str) -> Option<u32> {
        self.0.token_to_id(token)
    }
}

/// Lowercase hex SHA-256 of `bytes`.
pub fn sha_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Per-tensor identity of the release's `model.language_model.*` tensors against the stock checkpoint: SHA-256 of the
/// raw bytes, dtype and shape. Differing tensors also report the largest absolute difference.
pub fn identity(arc_dir: &Path, stock_dir: &Path) -> GResult<Value> {
    let a = TensorFile::open(weights_file(arc_dir)?)?;
    let s = TensorFile::open(weights_file(stock_dir)?)?;
    let (mut same, mut differ, mut missing) = (0usize, 0usize, 0usize);
    let mut rows = Vec::new();
    for name in a.names().into_iter().filter(|n| n.starts_with(LM_PREFIX)) {
        let ha = sha_hex(a.raw(&name)?);
        let mut row = json!({
            "name": name, "dtype": a.dtype(&name)?, "shape": a.shape(&name)?, "sha256_arc": ha,
        });
        if !s.has(&name) {
            missing += 1;
            row["status"] = json!("missing_in_stock");
        } else {
            let hs = sha_hex(s.raw(&name)?);
            let meta = a.dtype(&name)? == s.dtype(&name)? && a.shape(&name)? == s.shape(&name)?;
            row["sha256_stock"] = json!(hs);
            if meta && hs == ha {
                same += 1;
                row["status"] = json!("identical");
            } else {
                differ += 1;
                row["status"] = json!("different");
                if meta {
                    let (_, x) = a.read_f32(&name)?;
                    let (_, y) = s.read_f32(&name)?;
                    let d = x
                        .iter()
                        .zip(&y)
                        .map(|(p, q)| (p - q).abs())
                        .fold(0f32, f32::max);
                    row["max_abs_diff"] = json!(d);
                }
            }
        }
        rows.push(row);
    }
    let stock_only: Vec<String> = s
        .names()
        .into_iter()
        .filter(|n| n.starts_with(LM_PREFIX) && !a.has(n))
        .collect();
    let mut stock_only_layers: Vec<usize> = stock_only
        .iter()
        .filter_map(|n| {
            n.strip_prefix(LM_PREFIX)?
                .strip_prefix("layers.")?
                .split('.')
                .next()?
                .parse()
                .ok()
        })
        .collect();
    stock_only_layers.sort_unstable();
    stock_only_layers.dedup();
    let other_prefixes = |f: &TensorFile| -> Vec<String> {
        let mut p: Vec<String> = f
            .names()
            .iter()
            .filter(|n| !n.starts_with(LM_PREFIX))
            .map(|n| n.split('.').take(2).collect::<Vec<_>>().join("."))
            .collect();
        p.sort();
        p.dedup();
        p
    };
    Ok(json!({
        "arc_lm_tensors": rows.len(), "identical": same, "different": differ, "missing_in_stock": missing,
        "stock_only_lm_tensors": stock_only.len(), "stock_only_layers": stock_only_layers,
        "stock_only_non_layer": stock_only.iter().filter(|n| !n.contains(".layers.")).collect::<Vec<_>>(),
        "stock_has_lm_head": s.has("lm_head.weight"),
        "arc_other_prefixes": other_prefixes(&a), "stock_other_prefixes": other_prefixes(&s),
        "tensors": rows,
    }))
}

/// Natural-log loss of each next token (`ids[1..]` given their prefixes) and whether the argmax was that token.
pub fn token_nll(enc: &Encoder, ids: &[u32]) -> GResult<(Vec<f64>, Vec<bool>)> {
    let t = ids.len();
    if t < 2 {
        return value_err("token_nll: needs two tokens");
    }
    let hid = enc.cfg.hidden;
    let (h, _) = enc.residual(ids, &[])?;
    let normed = enc.final_normed(&h[..(t - 1) * hid]);
    let logits = enc.logits(&normed, t - 1)?;
    let v = enc.cfg.vocab;
    let mut nll = Vec::with_capacity(t - 1);
    let mut hit = Vec::with_capacity(t - 1);
    for i in 0..t - 1 {
        let row = &logits[i * v..(i + 1) * v];
        let (arg, m) = argmax(row);
        let lse = f64::from(m)
            + row
                .iter()
                .map(|&x| f64::from(x - m).exp())
                .sum::<f64>()
                .ln();
        let target = ids[i + 1] as usize;
        nll.push(lse - f64::from(row[target]));
        hit.push(arg == target);
    }
    Ok((nll, hit))
}

/// Index and value of the largest entry (the first on ties).
pub fn argmax(row: &[f32]) -> (usize, f32) {
    row.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv { (i, x) } else { (bi, bv) }
        })
}

/// Greedy continuation of `prompt` (no cache: every step reruns the prefix), stopping after `n` tokens or a stop id.
pub fn greedy(enc: &Encoder, prompt: &[u32], n: usize, stop: &[u32]) -> GResult<Vec<u32>> {
    let hid = enc.cfg.hidden;
    let mut ids = prompt.to_vec();
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let (h, _) = enc.residual(&ids, &[])?;
        let last = enc.final_normed(&h[(ids.len() - 1) * hid..]);
        let (next, _) = argmax(&enc.logits(&last, 1)?);
        let next = next as u32;
        out.push(next);
        ids.push(next);
        if stop.contains(&next) {
            break;
        }
    }
    Ok(out)
}

/// `sqrt(mean(x^2))`.
pub fn rms(x: &[f32]) -> f64 {
    (x.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / x.len().max(1) as f64).sqrt()
}

/// The two RoPE settings the oracle checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeSetting {
    /// What the checkpoint's own config says (`rope_type: default` for the stock model: plain RoPE).
    Stock,
    /// YaRN with factor 4 over `original_max_position_embeddings` 262144 (the ARC config, `yarn-f4`).
    YarnF4,
}

impl RopeSetting {
    pub fn name(self) -> &'static str {
        match self {
            Self::Stock => "stock",
            Self::YarnF4 => "yarn-f4",
        }
    }

    /// Points `enc` at this setting, starting from the checkpoint's own `base` rope config.
    pub fn apply(self, enc: &mut Encoder, base: &RopeConfig) {
        enc.cfg.rope = base.clone();
        match self {
            Self::Stock => {
                enc.opts.rope = if base.factor > 1.0 {
                    RopeVariant::Yarn
                } else {
                    RopeVariant::Plain
                };
            }
            Self::YarnF4 => {
                enc.cfg.rope.factor = 4.0;
                enc.cfg.rope.original_max_position = 262_144.0;
                enc.opts.rope = RopeVariant::Yarn;
            }
        }
    }
}

fn stats(w: &[f32]) -> Value {
    let n = w.len().max(1) as f64;
    let mean = w.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    let sd = (w
        .iter()
        .map(|&v| (f64::from(v) - mean).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();
    json!({ "mean": mean, "sd": sd, "min": w.iter().copied().fold(f32::INFINITY, f32::min),
            "max": w.iter().copied().fold(f32::NEG_INFINITY, f32::max) })
}

/// Value statistics of the norm weights, which tell the `(1 + w)` convention (centered on 0) from plain `w` (on 1).
pub fn norm_stats(enc: &Encoder) -> Value {
    use super::encoder::Mixer;
    let mut v = json!({ "final_norm": stats(&enc.final_norm) });
    if let Some(l) = enc.layers.first() {
        v["layer0_input_norm"] = stats(&l.input_norm);
        v["layer0_post_norm"] = stats(&l.post_norm);
    }
    for l in &enc.layers {
        match &l.mixer {
            Mixer::Gdn(g) if v.get("gdn_gated_norm").is_none() => {
                v["gdn_gated_norm"] = stats(&g.norm);
                v["gdn_a_log"] = stats(&g.a_log);
                v["gdn_dt_bias"] = stats(&g.dt_bias);
            }
            Mixer::Attn(a) if v.get("attn_q_norm").is_none() => {
                v["attn_q_norm"] = stats(&a.q_norm);
                v["attn_k_norm"] = stats(&a.k_norm);
            }
            _ => {}
        }
    }
    v
}

/// Arguments of the `lm` run.
pub struct LmArgs<'a> {
    pub stock: &'a Path,
    pub arc: Option<&'a Path>,
    pub max_new: usize,
    pub settings: Vec<RopeSetting>,
    /// Also write the full layer-12 last-token readout vector of every prompt (the GPU parity reference).
    pub full_vectors: bool,
}

/// `(id, logit)` of the `k` largest logits, largest first.
pub fn top_k(row: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..row.len()).collect();
    idx.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
    idx.into_iter().take(k).map(|i| (i, row[i])).collect()
}

/// `ln(sum(exp(row)))` in f64.
pub fn log_sum_exp(row: &[f32]) -> f64 {
    let m = f64::from(argmax(row).1);
    m + row
        .iter()
        .map(|&x| (f64::from(x) - m).exp())
        .sum::<f64>()
        .ln()
}

fn last_row(x: &[f32], hid: usize) -> &[f32] {
    &x[x.len() - hid..]
}

/// `(||a - b|| / ||b||, cos(a, b))` in f64.
pub fn rel_l2_and_cosine(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut d, mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x), f64::from(y));
        d += (x - y) * (x - y);
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    ((d / bb).sqrt(), ab / (aa * bb).sqrt())
}

/// Layer-12 last-token readout of `ids` (the final norm applied to the residual after 12 layers).
fn readout12(enc: &Encoder, ids: &[u32]) -> GResult<Vec<f32>> {
    let (_, snaps) = enc.residual(ids, &[READOUT_DEPTH])?;
    Ok(enc.final_normed(last_row(&snaps[0], enc.cfg.hidden)))
}

/// How far apart readouts are that a GPU parity tolerance must tell apart (or must allow): for every prompt, f32
/// against the bf16-rounding emulation in each setting, and the stock RoPE against yarn-f4.
fn readout_gaps(enc: &mut Encoder, base: &RopeConfig, tok: &Tok) -> GResult<Value> {
    let mut rows = Vec::new();
    for prompt in PROMPTS {
        let ids = tok.encode(prompt)?;
        let mut per = Vec::new();
        for s in [RopeSetting::Stock, RopeSetting::YarnF4] {
            s.apply(enc, base);
            let f = readout12(enc, &ids)?;
            enc.opts.bf16 = true;
            let b = readout12(enc, &ids)?;
            enc.opts.bf16 = false;
            per.push((f, b));
        }
        let gap = |a: &[f32], b: &[f32]| {
            let (r, c) = rel_l2_and_cosine(a, b);
            json!({ "rel_l2": r, "cosine": c })
        };
        rows.push(json!({
            "prompt": prompt,
            "stock_f32_vs_bf16": gap(&per[0].1, &per[0].0),
            "yarn_f4_f32_vs_bf16": gap(&per[1].1, &per[1].0),
            "yarn_f4_vs_stock_f32": gap(&per[1].0, &per[0].0),
        }));
    }
    Ok(Value::Array(rows))
}

/// The full-LM oracle: perplexity of [`TEXTS`], greedy continuations of [`PROMPTS`], and the layer-12 readout sanity
/// numbers, for each RoPE setting. With an ARC release, its own 12-layer encoder's readout is compared with the stock
/// model's layer-12 readout on the same ids.
pub fn run_lm(a: &LmArgs) -> GResult<Value> {
    let t0 = Instant::now();
    let mut enc = Encoder::load_depth(a.stock, None, EncoderOptions::default())?;
    let base = enc.cfg.rope.clone();
    let tok = Tok::load(a.stock)?;
    let mut arc = match a.arc {
        Some(d) => Some(Encoder::load(d, EncoderOptions::default())?),
        None => None,
    };
    let arc_base = arc.as_ref().map(|e| e.cfg.rope.clone());
    eprintln!(
        "loaded {} layers in {:.1}s",
        enc.layers.len(),
        t0.elapsed().as_secs_f64()
    );
    let hid = enc.cfg.hidden;
    let stop: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|s| tok.id(s))
        .collect();
    let mut out = json!({
        "layers": enc.layers.len(), "norm_stats": norm_stats(&enc),
        "rope_base": format!("{base:?}"), "stop_ids": stop, "settings": {},
    });
    for &s in &a.settings {
        s.apply(&mut enc, &base);
        if let (Some(e), Some(b)) = (arc.as_mut(), arc_base.as_ref()) {
            s.apply(e, b);
            // the ARC config is already yarn-f4; its "stock" here means plain RoPE, as for the stock model
            if s == RopeSetting::Stock {
                e.opts.rope = RopeVariant::Plain;
            }
        }
        let mut texts = Vec::new();
        for (name, text) in TEXTS {
            let ts = Instant::now();
            let ids = tok.encode(text)?;
            let (nll, hit) = token_nll(&enc, &ids)?;
            let mean = nll.iter().sum::<f64>() / nll.len() as f64;
            let acc = hit.iter().filter(|&&h| h).count() as f64 / hit.len() as f64;
            eprintln!(
                "[{}] {name}: {} tokens, ppl {:.3}, top1 {:.3} ({:.1}s)",
                s.name(),
                ids.len(),
                mean.exp(),
                acc,
                ts.elapsed().as_secs_f64()
            );
            texts.push(
                json!({ "text": name, "tokens": ids.len(), "scored": nll.len(),
                               "mean_nll": mean, "ppl": mean.exp(), "top1": acc }),
            );
        }
        let mut conts = Vec::new();
        for prompt in PROMPTS {
            let ts = Instant::now();
            let ids = tok.encode(prompt)?;
            let (h, snaps) = enc.residual(&ids, &[READOUT_DEPTH])?;
            let raw12 = last_row(&snaps[0], hid);
            let normed12 = enc.final_normed(raw12);
            let last_logits = enc.logits(&enc.final_normed(last_row(&h, hid)), 1)?;
            let generated = greedy(&enc, &ids, a.max_new, &stop)?;
            let text = tok.decode(&generated)?;
            let mut row = json!({
                "prompt": prompt, "prompt_ids": ids, "generated_ids": generated, "continuation": text,
                "layer12_last_raw_rms": rms(raw12), "layer12_last_readout_rms": rms(&normed12),
                "final_last_raw_rms": rms(last_row(&h, hid)),
                "layer12_readout_first8": &normed12[..8],
                "last_logits_top5": top_k(&last_logits, 5),
                "last_logits_lse": log_sum_exp(&last_logits),
                "last_logits_rms": rms(&last_logits),
            });
            if a.full_vectors {
                row["layer12_readout_last"] = json!(normed12);
            }
            if let Some(e) = arc.as_ref() {
                let ah = e.hidden(&ids)?;
                let d = last_row(&ah, hid)
                    .iter()
                    .zip(&normed12)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0f32, f32::max);
                row["arc_readout_max_abs_diff"] = json!(d);
                row["arc_readout_rms"] = json!(rms(last_row(&ah, hid)));
            }
            eprintln!(
                "[{}] {prompt:?} -> {text:?} ({:.1}s)",
                s.name(),
                ts.elapsed().as_secs_f64()
            );
            conts.push(row);
        }
        out["settings"][s.name()] = json!({ "texts": texts, "continuations": conts });
    }
    out["readout_gaps"] = readout_gaps(&mut enc, &base, &tok)?;
    out["wall_s"] = json!(t0.elapsed().as_secs_f64());
    Ok(out)
}
