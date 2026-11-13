// SPDX-License-Identifier: AGPL-3.0-only
// Portions derived from the ARC-1.0 release documentation and rayline-ai/vllm-arc (examples/readouts.py), Copyright
// 2026 Atlas Futures, Inc., licensed under the Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0). By
// Atlas contributors. See RAYLINE-ROUTING-ATTRIBUTION.md.
//! Responsibility: the serializer length check (`tokens`) and the signal-head oracle against the release goldens
//! (`goldens`): the CPU encoder runs the golden requests, the signal heads turn the readouts into probabilities, and
//! the probabilities must match `goldens.signals.jsonl` within [`GOLDEN_TOLERANCE`]. The choices the release does not
//! document (activation, scale mode, platt column order, which block row and token feed the heads, the serializer
//! shell, the RoPE variant, the rounding mode) are searched and the best variant is reported.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use serde_json::{Value, json};

use super::encoder::{Encoder, EncoderOptions, RopeVariant};
use super::mat::Mat;
use super::serializer::{
    BLOCK_ENDS, HfTokenizer, Selection, TextTokenizer, encode_blocks, render_blocks, select_row,
};
use super::signals::{self, HEADS, Variant};
use super::stage_one::check_release;

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

/// One golden request: the release's token count against this stand-in serializer's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenCase {
    pub case_id: String,
    pub golden: i64,
    pub ours: i64,
    pub delta: i64,
    pub turns: usize,
}

/// [`token_report`] result.
#[derive(Clone, Debug)]
pub struct TokenReport {
    pub cases: Vec<TokenCase>,
    pub distinct_deltas: Vec<i64>,
    pub distinct_deltas_single_message: Vec<i64>,
    pub constant_offset: bool,
}

impl TokenReport {
    fn to_json(&self) -> Value {
        json!({
            "cases": self.cases.iter().map(|c| json!({
                "case_id": c.case_id, "golden": c.golden, "ours": c.ours, "delta": c.delta, "turns": c.turns,
            })).collect::<Vec<_>>(),
            "distinct_deltas": self.distinct_deltas,
            "distinct_deltas_single_message": self.distinct_deltas_single_message,
            "constant_offset": self.constant_offset,
        })
    }
}

fn messages_of(g: &Value) -> Vec<Value> {
    g["request"]["request"]["messages"]
        .as_array()
        .cloned()
        .unwrap_or_default()
}

/// Our token count for every golden request against the release's `expected.token_count`.
pub fn token_report<T: TextTokenizer>(
    tok: &T,
    goldens: &[Value],
    shell: Option<&str>,
) -> super::error::GResult<TokenReport> {
    let mut cases = Vec::new();
    for g in goldens {
        let msgs = messages_of(g);
        let ours = encode_blocks(tok, &render_blocks(&msgs), shell)?.ids.len() as i64;
        let golden = g["expected"]["token_count"].as_i64().unwrap_or(0);
        cases.push(TokenCase {
            case_id: g["case_id"].as_str().unwrap_or_default().to_string(),
            golden,
            ours,
            delta: golden - ours,
            turns: msgs.len(),
        });
    }
    let sorted = |it: Vec<i64>| {
        let mut v = it;
        v.sort_unstable();
        v.dedup();
        v
    };
    let distinct = sorted(cases.iter().map(|c| c.delta).collect());
    let single = sorted(
        cases
            .iter()
            .filter(|c| c.turns == 1)
            .map(|c| c.delta)
            .collect(),
    );
    let constant_offset = single.len() == 1 && distinct.len() == 1;
    Ok(TokenReport {
        cases,
        distinct_deltas: distinct,
        distinct_deltas_single_message: single,
        constant_offset,
    })
}

fn read_shell(path: &Option<PathBuf>) -> Result<Option<String>> {
    path.as_ref()
        .map(|p| std::fs::read_to_string(p).with_context(|| format!("read {}", p.display())))
        .transpose()
}

/// `tokens`: prints the token-count difference per golden request.
pub fn cmd_tokens(
    release: &Path,
    shell_file: &Option<PathBuf>,
    out: &Option<PathBuf>,
) -> Result<i32> {
    let tok = HfTokenizer::from_release(release)?;
    let goldens = load_jsonl(&release.join("goldens.jsonl"))?;
    let shell = read_shell(shell_file)?;
    let rep = token_report(&tok, &goldens, shell.as_deref())?;
    println!(
        "route_gate tokens: {} golden requests; delta (release minus stand-in): all {:?}, single-message {:?}; constant offset: {}",
        rep.cases.len(),
        rep.distinct_deltas,
        rep.distinct_deltas_single_message,
        rep.constant_offset
    );
    if let Some(p) = out {
        std::fs::write(p, serde_json::to_string_pretty(&rep.to_json())?)?;
    }
    Ok(0)
}

/// Arguments of `goldens`.
pub struct GoldensArgs {
    pub release: PathBuf,
    pub out: Option<PathBuf>,
    pub shell_files: Vec<PathBuf>,
    pub no_verify: bool,
    /// Try the bf16-rounding encoder as well as plain f32.
    pub try_bf16: bool,
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
        token_deltas.insert(
            name.clone(),
            json!(
                token_report(&tok, &gold_tokens, text.as_deref())?.distinct_deltas_single_message
            ),
        );
    }
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
            let mut per_end: Vec<Vec<Vec<Vec<f64>>>> = vec![Vec::new(); BLOCK_ENDS.len()];
            let mut roles_all: Vec<Vec<String>> = Vec::new();
            for (ci, c) in sig_cases.iter().enumerate() {
                let msgs = messages_of(c);
                let e = encode_blocks(&tok, &render_blocks(&msgs), stext.as_deref())?;
                let h = enc.hidden(&e.ids)?;
                let d = enc.cfg.hidden;
                roles_all.push(
                    msgs.iter()
                        .map(|m| m["role"].as_str().unwrap_or_default().to_string())
                        .collect(),
                );
                for (bi, be) in BLOCK_ENDS.iter().enumerate() {
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
            for (bi, be) in BLOCK_ENDS.iter().enumerate() {
                for sel in Selection::ALL {
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
        }),
        max: worst,
        mean,
        pass,
    })
}
