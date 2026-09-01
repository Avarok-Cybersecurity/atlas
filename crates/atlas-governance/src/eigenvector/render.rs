// SPDX-License-Identifier: AGPL-3.0-only

//! Markdown for the intent-trajectory Discussion comment.
//!
//! The marker constants are defined here, beside the code that emits them,
//! for the same reason `gate::telemetry` defines its own: the upsert finds our
//! comment by searching for the marker, so the renderer and the publisher must
//! agree on it, and exactly one of them should own it.

use super::{Axis, Pole, RenderInput};

pub const MARKER_START: &str = "<!-- atlas-intent-eigenvector:start -->";
pub const MARKER_END: &str = "<!-- atlas-intent-eigenvector:end -->";

/// Below this, an axis explains so little that showing it invites reading
/// structure into noise.
const MIN_EXPLAINED: f64 = 0.01;

fn pct(x: f64) -> String {
    format!("{:.1}%", x * 100.0)
}

/// Plain-language reading of the drift angle, so the number is not left to be
/// interpreted by whoever happens to open the page.
fn drift_reading(degrees: f64) -> &'static str {
    match degrees {
        d if d < 15.0 => "holding its direction",
        d if d < 40.0 => "bending",
        d if d < 70.0 => "turning markedly",
        _ => "working at right angles to what came before",
    }
}

fn coherence_reading(explained: f64) -> &'static str {
    match explained {
        e if e > 0.60 => "strongly single-minded — most work shares one direction",
        e if e > 0.35 => "focused, with real secondary threads",
        e if e > 0.18 => "several concurrent themes of comparable weight",
        _ => "diffuse — no single direction dominates",
    }
}

fn pole_rows(out: &mut String, label: &str, poles: &[Pole]) {
    if poles.is_empty() {
        return;
    }
    out.push_str(&format!("\n**{label}**\n\n"));
    for p in poles {
        out.push_str(&format!(
            "- #{} `{}` — {}\n",
            p.pr,
            p.intent,
            p.title.replace('|', "\\|")
        ));
    }
}

fn render_axis(out: &mut String, index: usize, axis: &Axis, name: Option<&super::AxisName>) {
    let heading = match name {
        Some(n) => format!(
            "### Axis {} — {} ({} of the variance)",
            index + 1,
            n.label,
            pct(axis.explained)
        ),
        None => format!(
            "### Axis {} ({} of the variance)",
            index + 1,
            pct(axis.explained)
        ),
    };
    out.push_str(&heading);
    out.push('\n');
    if let Some(n) = name {
        out.push_str(&format!("\n{}\n", n.gloss));
    }
    pole_rows(out, "One end", &axis.positive);
    pole_rows(out, "The other end", &axis.negative);
    out.push('\n');
}

/// Render the whole report, marker to marker.
pub fn render(input: &RenderInput) -> String {
    let a = &input.analysis;
    let naming = input.naming.as_ref();
    let mut out = String::new();

    out.push_str(MARKER_START);
    out.push_str("\n## 🧭 Repository intent trajectory\n\n");

    if a.axes.is_empty() {
        out.push_str(&format!(
            "No axes yet. {} PR intent{} in the window — not enough variation to \
             decompose into directions. This resolves on its own as the ledger fills.\n\n",
            a.n,
            if a.n == 1 { "" } else { "s" }
        ));
        out.push_str(MARKER_END);
        out.push('\n');
        return out;
    }

    // Headline numbers first: the two things a reader wants without scrolling.
    out.push_str("| | |\n|---|---|\n");
    if let Some(c) = a.coherence {
        out.push_str(&format!(
            "| **Coherence** | {} — {} |\n",
            pct(c),
            coherence_reading(c)
        ));
    }
    match a.drift_degrees {
        Some(d) => out.push_str(&format!(
            "| **Drift** | {:.0}° — {} |\n",
            d,
            drift_reading(d)
        )),
        None => out.push_str("| **Drift** | not readable yet (needs 4+ PRs) |\n"),
    }
    out.push_str(&format!(
        "| **Window** | newest {} PRs in the ledger; {} carried an intent |\n\n",
        a.window, a.n
    ));

    if let Some(t) = naming.and_then(|n| n.trajectory.as_ref()) {
        out.push_str(&format!("{t}\n\n"));
    }

    for (i, axis) in a.axes.iter().enumerate() {
        if axis.explained < MIN_EXPLAINED {
            continue;
        }
        render_axis(&mut out, i, axis, naming.and_then(|n| n.axes.get(i)));
    }

    out.push_str(
        "<sub>Axes are the principal components of the embedded PR intents in \
         `governance/`, computed by power iteration on the Gram matrix; poles are the \
         PRs furthest along each. Drift is the angle between the principal directions of \
         the older and newer halves of the window. A model names the axes it is shown — \
         it does not choose them. Rolling window, not all-time. Descriptive only: nothing \
         here gates anything.</sub>\n\n",
    );
    out.push_str(MARKER_END);
    out.push('\n');
    out
}
