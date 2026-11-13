// SPDX-License-Identifier: AGPL-3.0-only
// Portions derived from the ARC-1.0 release documentation and rayline-ai/vllm-arc (examples/readouts.py), Copyright
// 2026 Atlas Futures, Inc., licensed under the Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0). By
// Atlas contributors. See RAYLINE-ROUTING-ATTRIBUTION.md.
//! Responsibility: the signal-head oracle against the release goldens (`goldens`): the CPU encoder runs the golden
//! requests (serialized by the reconstructed `canonical_v2` unless the stand-in is asked for), the readout is the last
//! token of the last block, the signal heads turn it into probabilities, and the probabilities must match
//! `goldens.signals.jsonl` within [`GOLDEN_TOLERANCE`]. The choices the release does not document (activation, scale
//! mode, platt column order, the RoPE variant, the rounding mode) are searched and the best variant is reported; the
//! readout position is searched only with `--search-readout`. The token-count check is in `token_report`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use serde_json::{Value, json};

use super::canonical::Serializer;
use super::encoder::{Encoder, EncoderOptions, RopeVariant};
use super::mat::Mat;
use super::serializer::{BLOCK_ENDS, HfTokenizer, IM_START, Selection, encode_blocks, select_row};
use super::signals::{self, HEADS, Variant};
use super::stage_one::check_release;
use super::token_report::{request_of, token_report_for};

/// Master plan L04 oracle: signal probabilities within 5e-3.
pub const GOLDEN_TOLERANCE: f64 = 5e-3;

/// Reads a `.jsonl` file of objects.
pub fn load_jsonl(path: &Path) -> Result<Vec<Value>> {
    std::fs::read_to_string(path)
        .with_context(|| format!("read {}", path.display()))?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(Into::into))
        .collect()
}

/// Arguments of `goldens`.
pub struct GoldensArgs {
    pub release: PathBuf,
    pub out: Option<PathBuf>,
    pub shell_files: Vec<PathBuf>,
    pub no_verify: bool,
    /// Try the bf16-rounding encoder as well as plain f32.
    pub try_bf16: bool,
    /// The request serializer (the reconstructed `canonical_v2` unless the stand-in is asked for).
    pub serializer: Serializer,
    /// Also search the readout position (block end and row selection); by default the readout is the last token.
    pub search_readout: bool,
}

struct VariantResult {
    json: Value,
    max: f64,
    mean: f64,
    pass: bool,
}

/// `goldens`: the oracle. Exit code 0 when the best variant is within tolerance, 3 otherwise.
pub fn cmd_goldens(a: &GoldensArgs) -> Result<i32> {
    let t0 = Instant::now();
    let cfg = check_release(&a.release, !a.no_verify)?;
    let heads = signals::load_signal_heads(&a.release, &cfg)?;
    let tok = HfTokenizer::from_release(&a.release)?;
    let mut enc = Encoder::load(&a.release, EncoderOptions::default())?;
    println!(
        "route_gate goldens: encoder loaded in {:.1}s",
        t0.elapsed().as_secs_f64()
    );
    let has_scope = |c: &Value| {
        HEADS.iter().any(|h| {
            c["signals"]
                .get(*h)
                .and_then(|s| s.get("scopes"))
                .and_then(|s| s.get("conversation"))
                .is_some_and(|v| !v.is_null() && v.as_object().is_none_or(|o| !o.is_empty()))
        })
    };
    let sig_cases: Vec<Value> = load_jsonl(&a.release.join("goldens.signals.jsonl"))?
        .into_iter()
        .filter(has_scope)
        .collect();
    let mut shells: Vec<(String, Option<String>)> = vec![("bare".to_string(), None)];
    for p in &a.shell_files {
        shells.push((
            p.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            Some(std::fs::read_to_string(p).with_context(|| format!("read {}", p.display()))?),
        ));
    }
    let gold_tokens = load_jsonl(&a.release.join("goldens.jsonl"))?;
    let mut token_deltas = serde_json::Map::new();
    for (name, text) in &shells {
        let rep = token_report_for(a.serializer, &tok, &gold_tokens, text.as_deref())?;
        token_deltas.insert(
            name.clone(),
            json!({"exact": rep.matched(), "total": rep.cases.len(),
                   "single_message_deltas": rep.distinct_deltas_single_message}),
        );
    }
    // The readout of the release is the last token of the last block; the search is opt-in.
    let ends: Vec<&str> = if a.search_readout {
        BLOCK_ENDS.to_vec()
    } else {
        vec!["nl"]
    };
    let selections: Vec<Selection> = if a.search_readout {
        Selection::ALL.to_vec()
    } else {
        vec![Selection::Last]
    };
    let mut enc_variants: Vec<EncoderOptions> = Vec::new();
    for bf16 in if a.try_bf16 {
        vec![false, true]
    } else {
        vec![false]
    } {
        for rope in [RopeVariant::Yarn, RopeVariant::YarnRatio] {
            enc_variants.push(EncoderOptions { bf16, rope });
        }
    }
    let mut results: Vec<VariantResult> = Vec::new();
    for opts in &enc_variants {
        enc.opts = *opts;
        for (sname, stext) in &shells {
            let mut per_end: Vec<Vec<Vec<Vec<f64>>>> = vec![Vec::new(); ends.len()];
            let mut roles_all: Vec<Vec<String>> = Vec::new();
            for (ci, c) in sig_cases.iter().enumerate() {
                let blocks = a.serializer.render(request_of(c));
                let e = encode_blocks(&tok, &blocks, stext.as_deref())?;
                let h = enc.hidden(&e.ids)?;
                let d = enc.cfg.hidden;
                roles_all.push(
                    blocks
                        .iter()
                        .map(|b| {
                            b[IM_START.len()..]
                                .split('\n')
                                .next()
                                .unwrap_or_default()
                                .to_string()
                        })
                        .collect(),
                );
                for (bi, be) in ends.iter().enumerate() {
                    let rows: Vec<Vec<f64>> = e
                        .pick_offsets(be)?
                        .iter()
                        .map(|&o| {
                            h[o * d..(o + 1) * d]
                                .iter()
                                .map(|&v| f64::from(v))
                                .collect()
                        })
                        .collect();
                    per_end[bi].push(rows);
                }
                println!(
                    "  [{:>4.0}s] bf16={} rope={} shell={sname}: case {}/{} ({} tokens)",
                    t0.elapsed().as_secs_f64(),
                    opts.bf16,
                    opts.rope.name(),
                    ci + 1,
                    sig_cases.len(),
                    e.ids.len()
                );
            }
            for (bi, be) in ends.iter().enumerate() {
                for &sel in &selections {
                    let rows: Vec<Vec<f64>> = per_end[bi]
                        .iter()
                        .zip(&roles_all)
                        .map(|(r, roles)| select_row(r, roles, sel))
                        .collect();
                    let x = Mat::from_rows(&rows)?;
                    if sel == Selection::Last && *be == "nl" {
                        for (hname, head) in &heads {
                            let st = |m| head.standardized_stats(&x, 0, m);
                            let (mul, div) =
                                (st(signals::ScaleMode::Mul), st(signals::ScaleMode::Div));
                            println!(
                                "  readout check bf16={} rope={} shell={sname} head={hname}: z-rms mul {:.2} div {:.2}; corr(row, head mean) {:.3}",
                                opts.bf16,
                                opts.rope.name(),
                                mul.0,
                                div.0,
                                div.1
                            );
                        }
                    }
                    for v in signals::grid() {
                        results.push(score_variant(
                            &heads, &sig_cases, &x, v, opts, sname, be, sel,
                        )?);
                    }
                }
            }
        }
    }
    results.sort_by(|a, b| a.max.total_cmp(&b.max).then(a.mean.total_cmp(&b.mean)));
    let oracle_pass = results.first().is_some_and(|r| r.pass);
    let report = json!({
        "n_cases": sig_cases.len(), "tolerance": GOLDEN_TOLERANCE,
        "encoder": {"implementation": "route_gate rust cpu f32", "layers": enc.layers.len(), "weights": "BF16"},
        "serializer": a.serializer.name(), "readout_search": a.search_readout,
        "variants": results.iter().map(|r| r.json.clone()).collect::<Vec<_>>(),
        "tokens": Value::Object(token_deltas),
        "best": results.first().map(|r| r.json.clone()), "oracle_pass": oracle_pass,
    });
    if let Some(p) = &a.out {
        std::fs::write(p, serde_json::to_string_pretty(&report)?)?;
    }
    for r in results.iter().take(8) {
        let v = &r.json;
        println!(
            "  max {:.5} mean {:.5} pass={} match={}/{} bf16={} rope={} shell={} end={} row={} act={} scale={} platt={}",
            r.max,
            r.mean,
            r.pass,
            v["cases_within_tolerance"],
            v["cases_total"],
            v["bf16"],
            v["rope"],
            v["shell"],
            v["block_end"],
            v["selection"],
            v["activation"],
            v["scale_mode"],
            v["platt_order"]
        );
    }
    if let Some(best) = results.first() {
        println!("  best variant, max abs error per case:");
        for c in best.json["per_case_max_err"]
            .as_array()
            .into_iter()
            .flatten()
        {
            println!(
                "    {:<28} {:.5}",
                c["case_id"].as_str().unwrap_or_default(),
                c["max_abs_err"].as_f64().unwrap_or(f64::NAN)
            );
        }
    }
    println!(
        "route_gate goldens: oracle {} (tolerance {GOLDEN_TOLERANCE}); {} cases, {:.1}s; AUROC results must not be believed until it passes",
        if oracle_pass { "PASSED" } else { "NOT met" },
        sig_cases.len(),
        t0.elapsed().as_secs_f64()
    );
    Ok(if oracle_pass { 0 } else { 3 })
}

#[allow(clippy::too_many_arguments)]
fn score_variant(
    heads: &[(String, signals::SignalHead)],
    cases: &[Value],
    x: &Mat,
    v: Variant,
    opts: &EncoderOptions,
    shell: &str,
    block_end: &str,
    sel: Selection,
) -> Result<VariantResult> {
    let (mut worst, mut sum, mut count) = (0.0f64, 0.0f64, 0usize);
    let mut per_case = vec![0.0f64; cases.len()];
    for (hname, head) in heads {
        let probs = head.probs(x, v)?;
        for (k, c) in cases.iter().enumerate() {
            let Some(exp) = signals::expected_probs(&c["signals"][hname.as_str()], &head.classes)
            else {
                continue;
            };
            for (ci, e) in exp.iter().enumerate() {
                let d = (probs.at(k, ci) - e).abs();
                worst = worst.max(d);
                per_case[k] = per_case[k].max(d);
                sum += d;
                count += 1;
            }
        }
    }
    let mean = if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    };
    let pass = worst <= GOLDEN_TOLERANCE;
    let within = per_case.iter().filter(|w| **w <= GOLDEN_TOLERANCE).count();
    Ok(VariantResult {
        json: json!({
            "cases_within_tolerance": within, "cases_total": cases.len(),
            "bf16": opts.bf16, "rope": opts.rope.name(), "shell": shell, "block_end": block_end, "selection": sel.name(),
            "activation": v.activation.name(), "scale_mode": v.scale_mode.name(), "platt_order": v.platt_order.name(),
            "max_abs_err": worst, "mean_abs_err": mean, "pass": pass,
            "per_case_max_err": cases.iter().zip(&per_case)
                .map(|(c, e)| json!({"case_id": c["case_id"], "max_abs_err": e})).collect::<Vec<_>>(),
        }),
        max: worst,
        mean,
        pass,
    })
}
