// SPDX-License-Identifier: AGPL-3.0-only
//! Shareable result cards: a benchmark record rendered onto the Atlas card.
//!
//! # One template, a mapping per benchmark
//!
//! The card has four generic detail slots and one hero number. Eleven separate
//! SVGs would mean eleven files to keep in sync with one stylesheet; instead the
//! artwork is fixed and each benchmark says which of ITS metrics belong in which
//! slot. Adding a benchmark is a table entry, not a new asset.
//!
//! # Every card carries its configuration, and that is not decoration
//!
//! The card prints the model, the quantization, the recipe and the hardware
//! beside the number, because this repository has already been bitten by a
//! number quoted without them — the 2026-08-15 decode-rate retraction, and the
//! BFCL 87.24 that was only ever valid as an A/B denominator. A card that shows
//! a headline figure with no configuration is the artefact that starts that
//! again, so the fields are not optional and a missing one renders as `—`
//! rather than being silently dropped.
use crate::gate::record::GateRecord;
use std::collections::BTreeMap;

/// How a raw `f64` becomes card text.
#[derive(Clone, Copy, Debug)]
pub enum Fmt {
    /// `115` — counts, sample sizes, pass tallies.
    Int,
    /// `22.7` — throughput, seconds per turn.
    One,
    /// `84.22` — accuracy, where the second digit is the whole argument.
    Two,
    /// `8288 ms` — latencies, which are never interesting below the millisecond.
    Ms,
}

impl Fmt {
    pub fn apply(self, v: f64) -> String {
        match self {
            Fmt::Int => format!("{}", v.round() as i64),
            Fmt::One => format!("{v:.1}"),
            Fmt::Two => format!("{v:.2}"),
            Fmt::Ms => format!("{} ms", v.round() as i64),
        }
    }
}

/// One of the four detail boxes.
#[derive(Clone, Copy, Debug)]
pub struct Slot {
    pub label: &'static str,
    pub key: &'static str,
    pub fmt: Fmt,
}

const fn slot(label: &'static str, key: &'static str, fmt: Fmt) -> Option<Slot> {
    Some(Slot { label, key, fmt })
}

/// What a benchmark puts on a card.
#[derive(Clone, Debug)]
pub struct CardSpec {
    pub hero_label: &'static str,
    pub hero_key: &'static str,
    pub hero_note: &'static str,
    pub hero_fmt: Fmt,
    pub slots: [Option<Slot>; 4],
}

/// The per-benchmark mapping.
///
/// Keys are the metric names the records actually carry — taken from committed
/// records, not from memory, because a key that does not exist renders an empty
/// box and nobody notices until it is on social media.
pub fn spec_for(benchmark_id: &str) -> CardSpec {
    match benchmark_id {
        "decode-floor" => CardSpec {
            hero_label: "Tokens / sec",
            hero_key: "server_decode_tok_s",
            hero_note: "decode, steady state",
            hero_fmt: Fmt::One,
            slots: [
                slot("Accept length", "accept_len_mean", Fmt::Two),
                slot("Output tokens", "output_tokens", Fmt::Int),
                slot("Runs", "runs", Fmt::Int),
                None,
            ],
        },
        "concurrency-sweep" => CardSpec {
            hero_label: "Tokens / sec",
            hero_key: "peak_aggregate_tok_s",
            hero_note: "aggregate, best rung of C=1..128",
            hero_fmt: Fmt::One,
            slots: [
                slot("C=1", "c1_aggregate_tok_s", Fmt::One),
                slot("C=16", "c16_aggregate_tok_s", Fmt::One),
                slot("C=128", "c128_aggregate_tok_s", Fmt::One),
                slot("TTFT p50, C=1", "c1_ttft_p50_ms", Fmt::Ms),
            ],
        },
        // The DFlash2 ladder stops at C=16: the verify pool pins at 32 slots, so
        // the wide rungs its sibling reports do not exist here. Showing C=128 as
        // an empty box would read as a regression rather than an absence.
        "concurrency-sweep-dflash2" => CardSpec {
            hero_label: "Tokens / sec",
            hero_key: "peak_aggregate_tok_s",
            hero_note: "aggregate, best rung of C=1..16, DFlash2 armed",
            hero_fmt: Fmt::One,
            slots: [
                slot("C=1", "c1_aggregate_tok_s", Fmt::One),
                slot("C=4", "c4_aggregate_tok_s", Fmt::One),
                slot("C=16", "c16_aggregate_tok_s", Fmt::One),
                slot("TTFT p50, C=1", "c1_ttft_p50_ms", Fmt::Ms),
            ],
        },
        "bfcl-subset" | "bfcl-subset-echolp" => CardSpec {
            hero_label: "Overall accuracy",
            hero_key: "overall_accuracy",
            // The draw is part of the number. CLAUDE.md's rule, on the card.
            hero_note: "BFCL single-turn — see samples for the draw",
            hero_fmt: Fmt::Two,
            slots: [
                slot("Normalized ST", "normalized_single_turn_score", Fmt::Two),
                slot("Samples", "samples", Fmt::Int),
                None,
                None,
            ],
        },
        "agentic-webserver" => CardSpec {
            hero_label: "Webserver OK",
            hero_key: "webserver_ok",
            hero_note: "runs that built, served and tore down cleanly",
            hero_fmt: Fmt::Int,
            slots: [
                slot("Followed directions", "followed_directions", Fmt::Int),
                slot("Iterations", "iterations", Fmt::Int),
                slot("Seconds / turn", "s_per_turn", Fmt::Two),
                slot("Decode tok/s", "decode_tps", Fmt::One),
            ],
        },
        "ttft-cold-gate" => CardSpec {
            hero_label: "TTFT median",
            hero_key: "median_ms",
            hero_note: "cold, first token after load",
            hero_fmt: Fmt::Ms,
            slots: [
                slot("p90", "p90_ms", Fmt::Ms),
                slot("Samples", "samples", Fmt::Int),
                None,
                None,
            ],
        },
        "ttft-warm-gate" => CardSpec {
            hero_label: "TTFT median",
            hero_key: "median_ms",
            hero_note: "warm, prefix cache primed",
            hero_fmt: Fmt::Ms,
            slots: [
                slot("p90", "p90_ms", Fmt::Ms),
                slot("Samples", "samples", Fmt::Int),
                None,
                None,
            ],
        },
        "vision-fidelity" => CardSpec {
            hero_label: "Geometry cells matched",
            hero_key: "geometry_matched",
            hero_note: "vision fidelity, control held",
            hero_fmt: Fmt::Int,
            slots: [
                slot("Cells asserted", "geometry_asserted", Fmt::Int),
                slot("Probes passed", "probes_passed", Fmt::Int),
                slot("Probes total", "probes_total", Fmt::Int),
                None,
            ],
        },
        "video-fidelity" => CardSpec {
            hero_label: "Legs passed",
            hero_key: "legs_passed",
            hero_note: "video fidelity, control held",
            hero_fmt: Fmt::Int,
            slots: [
                slot("Legs asserted", "legs_asserted", Fmt::Int),
                slot("Skipped", "legs_skipped", Fmt::Int),
                None,
                None,
            ],
        },
        "ssm-state-poisoning-gate" => CardSpec {
            hero_label: "Replays byte-identical",
            hero_key: "invariant",
            hero_note: "SSM state isolation under interleaving",
            hero_fmt: Fmt::Int,
            slots: [
                slot("Rounds", "rounds", Fmt::Int),
                slot("Collapsed", "collapsed", Fmt::Int),
                slot("Jittered", "jittered", Fmt::Int),
                None,
            ],
        },
        // Unknown benchmark: the card still renders. Sorted, so it is stable
        // rather than whatever order a map happened to yield — an unstable card
        // for the same record would be its own small mystery.
        _ => CardSpec {
            hero_label: "Result",
            hero_key: "",
            hero_note: "first metric, alphabetically — no card mapping for this benchmark yet",
            hero_fmt: Fmt::Two,
            slots: [None, None, None, None],
        },
    }
}

/// `author=Ada,website=ada.dev` -> map. Later keys win, empty pairs ignored.
///
/// Deliberately forgiving about spacing and a trailing comma, because this is
/// typed by hand on a command line. Deliberately NOT forgiving about a missing
/// `=`: silently dropping `authorAda` would produce a card with no author and no
/// complaint.
pub fn parse_args(raw: &str) -> Result<BTreeMap<String, String>, String> {
    let mut out = BTreeMap::new();
    for part in raw.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        let (k, v) = p
            .split_once('=')
            .ok_or_else(|| format!("`{p}` is not key=value"))?;
        let k = k.trim();
        if k.is_empty() {
            return Err(format!("`{p}` has an empty key"));
        }
        out.insert(k.to_lowercase(), v.trim().to_string());
    }
    Ok(out)
}

/// XML text escaping. The values are model ids, hostnames and a hand-typed
/// author line; `&` in a model id is enough to produce an SVG no renderer will
/// open.
fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Set an attribute on the element carrying `id`, replacing it if present.
///
/// The template hides a whole box with `display="none"` on its group, which is
/// how an unused slot disappears rather than sitting there as an empty bordered
/// rectangle. Text substitution cannot express that, so this exists.
fn set_attr(svg: &mut String, id: &str, attr: &str, value: &str) {
    let needle = format!("id=\"{id}\"");
    let Some(at) = svg.find(&needle) else { return };
    let Some(open) = svg[..at].rfind('<') else {
        return;
    };
    let Some(close) = svg[at..].find('>').map(|i| at + i) else {
        return;
    };
    let existing = format!(" {attr}=\"");
    if let Some(k) = svg[open..close].find(&existing) {
        let vstart = open + k + existing.len();
        let Some(vend) = svg[vstart..close].find('"').map(|i| vstart + i) else {
            return;
        };
        svg.replace_range(vstart..vend, value);
        return;
    }
    svg.insert_str(close, &format!(" {attr}=\"{value}\""));
}

/// Replace the text body of `<... id="ID" ...>body</...>`.
fn set(svg: &mut String, id: &str, value: &str) {
    let needle = format!("id=\"{id}\"");
    let Some(at) = svg.find(&needle) else { return };
    let Some(gt) = svg[at..].find('>').map(|i| at + i + 1) else {
        return;
    };
    let Some(lt) = svg[gt..].find('<').map(|i| gt + i) else {
        return;
    };
    svg.replace_range(gt..lt, &esc(value));
}

/// Render a record onto the card template.
pub fn render(template: &str, record: &GateRecord, args: &BTreeMap<String, String>) -> String {
    let spec = spec_for(&record.benchmark_id);
    let mut svg = template.to_string();
    let m = &record.metrics;

    // ── The hero ──────────────────────────────────────────────────────────
    let hero = if spec.hero_key.is_empty() {
        m.iter().next().map(|(k, v)| (k.clone(), *v))
    } else {
        m.get(spec.hero_key)
            .map(|v| (spec.hero_key.to_string(), *v))
    };
    match hero {
        Some((key, v)) => {
            set(&mut svg, "value-toks", &spec.hero_fmt.apply(v));
            let label = if spec.hero_key.is_empty() {
                key
            } else {
                spec.hero_label.to_string()
            };
            set(&mut svg, "label-toks", &label);
        }
        // A record with no hero metric is a real thing — a failed run. Say so on
        // the card rather than printing the template's placeholder `000.0`,
        // which would read as a measurement of zero.
        None => {
            set(&mut svg, "value-toks", "—");
            set(&mut svg, "label-toks", spec.hero_label);
        }
    }
    set(&mut svg, "note-toks", spec.hero_note);

    // ── The verdict ───────────────────────────────────────────────────────
    // A card without one invites the reader to assume the number passed. Gold
    // rather than red for FAIL: the brand palette has no red, and #EFB338 is
    // its warning tone.
    match record.verdict.as_deref() {
        Some(_) if record.verdict_passes() => {
            set(&mut svg, "value-verdict", "PASS");
            set_attr(&mut svg, "verdict-chip", "fill", "#12B981");
        }
        Some(_) => {
            set(&mut svg, "value-verdict", "FAIL");
            set_attr(&mut svg, "verdict-chip", "fill", "#EFB338");
            // A large green em dash would read as a result. It is an absence.
            set_attr(&mut svg, "value-toks", "fill", "#82868F");
        }
        // An ungated run has no verdict to report. Hiding the chip is honest; a
        // grey placeholder pill invites the reader to wonder what it means.
        None => set_attr(&mut svg, "field-verdict", "display", "none"),
    }

    // ── Configuration, which travels with the number ──────────────────────
    set(&mut svg, "value-model", &record.target_model);
    set(
        &mut svg,
        "value-quant",
        record
            .params
            .get("quant")
            .or_else(|| record.params.get("quantization"))
            .map_or_else(|| quant_from_model(&record.target_model), |q| q.clone())
            .as_str(),
    );
    set(
        &mut svg,
        "value-recipe",
        record.served_by.as_deref().unwrap_or("—"),
    );
    set(&mut svg, "value-test", &record.benchmark_id);
    set(&mut svg, "value-hardware", &hardware_line(record));

    // ── Detail slots ──────────────────────────────────────────────────────
    for (i, s) in spec.slots.iter().enumerate() {
        let (lid, vid) = (format!("label-m{}", i + 1), format!("value-m{}", i + 1));
        match s.and_then(|s| m.get(s.key).map(|v| (s, *v))) {
            Some((s, v)) => {
                set(&mut svg, &lid, s.label);
                set(&mut svg, &vid, &s.fmt.apply(v));
            }
            // An empty slot is blanked, not left holding the template's dummy
            // value. A card showing "Batch 0" for a benchmark with no batch is
            // worse than a card showing nothing.
            // Hide the whole box. Blanking the text leaves an empty bordered
            // rectangle, which reads as a metric that failed to render rather
            // than as a benchmark that has three numbers, not four.
            None => set_attr(&mut svg, &format!("field-m{}", i + 1), "display", "none"),
        }
    }

    // ── Attribution ───────────────────────────────────────────────────────
    for (field, value_id, value) in [
        ("field-author", "value-author", args.get("author")),
        ("field-handle", "value-handle", args.get("handle")),
        (
            "field-site",
            "value-site",
            args.get("website").or_else(|| args.get("site")),
        ),
    ] {
        match value {
            Some(v) if !v.is_empty() => set(&mut svg, value_id, v),
            _ => set_attr(&mut svg, field, "display", "none"),
        }
    }
    svg
}

/// The quantization, when the record does not carry it as a param.
///
/// Read off the checkpoint id, which is where it lives in practice
/// (`unsloth/Qwen3.8-27B-NVFP4`). A guess would be worse than `—`: the whole
/// point of the configuration row is that it is true.
fn quant_from_model(model: &str) -> String {
    let up = model.to_uppercase();
    for tag in ["NVFP4", "FP8", "W4A4", "W4A16", "BF16", "INT8", "FP16"] {
        if up.contains(tag) {
            return tag.to_string();
        }
    }
    "—".to_string()
}

fn hardware_line(record: &GateRecord) -> String {
    let hw = &record.hardware;
    let gpu = hw.gate_key();
    match record
        .hardware_state
        .as_ref()
        .and_then(|s| s.before.sm_clock_mhz)
    {
        Some(mhz) => format!("{gpu}, SM {mhz:.0} MHz"),
        None => gpu,
    }
}
