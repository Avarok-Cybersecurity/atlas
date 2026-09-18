// SPDX-License-Identifier: AGPL-3.0-only

//! Resolving the `--control-vector*` flags into [`ControlVectorSpec`]s.
//!
//! Shape mirrors `--lora-adapter`: `NAME=VALUE`, repeatable, one entry per
//! vector. The modifiers are separate flags keyed by the same name rather than
//! a packed `PATH:SCALE:A-B:MODE` spec string, because a filesystem path may
//! contain `:` or `,` and a mis-split there would silently change the scale or
//! the layer range — the two parameters whose wrongness is hardest to see in
//! the output.
//!
//! Every modifier is optional. Defaults are the published refusal-projection
//! configuration: projection mode at scale 1.0 over every layer the file
//! carries.

use anyhow::{Result, bail};
use spark_model::control_vector::{ControlVectorSpec, CvecMode};

/// `NAME=VALUE`, split on the FIRST `=` so a value may contain more.
pub fn parse_named_value(flag: &str, s: &str) -> Result<(String, String), String> {
    let (name, value) = s
        .split_once('=')
        .ok_or_else(|| format!("{flag} must be NAME=VALUE, got '{s}'"))?;
    if name.is_empty() || value.is_empty() {
        return Err(format!("{flag}: empty name or value in '{s}'"));
    }
    Ok((name.to_string(), value.to_string()))
}

/// Value parser for `--control-vector NAME=PATH`.
pub fn parse_control_vector(s: &str) -> Result<(String, String), String> {
    parse_named_value("--control-vector", s)
}

/// Value parser for `--control-vector-layers NAME=START-END`.
pub fn parse_control_vector_layers(s: &str) -> Result<(String, String), String> {
    let (name, range) = parse_named_value("--control-vector-layers", s)?;
    parse_range(&range).map_err(|e| format!("--control-vector-layers '{s}': {e}"))?;
    Ok((name, range))
}

/// Value parser for `--control-vector-scale NAME=FLOAT`.
pub fn parse_control_vector_scale(s: &str) -> Result<(String, String), String> {
    let (name, v) = parse_named_value("--control-vector-scale", s)?;
    let f: f32 = v
        .parse()
        .map_err(|_| format!("--control-vector-scale '{s}': '{v}' is not a number"))?;
    if !f.is_finite() {
        return Err(format!("--control-vector-scale '{s}': must be finite"));
    }
    Ok((name, v))
}

/// Value parser for `--control-vector-mode NAME=project|add`.
pub fn parse_control_vector_mode(s: &str) -> Result<(String, String), String> {
    let (name, m) = parse_named_value("--control-vector-mode", s)?;
    match m.as_str() {
        "project" | "add" => Ok((name, m)),
        other => Err(format!(
            "--control-vector-mode '{s}': mode must be `project` or `add`, got '{other}'"
        )),
    }
}

/// `START-END`, inclusive both ends.
fn parse_range(s: &str) -> Result<(usize, usize), String> {
    let (a, b) = s
        .split_once('-')
        .ok_or_else(|| "range must be START-END".to_string())?;
    let a: usize = a.trim().parse().map_err(|_| format!("bad start '{a}'"))?;
    let b: usize = b.trim().parse().map_err(|_| format!("bad end '{b}'"))?;
    if a > b {
        return Err(format!("start {a} is past end {b}"));
    }
    Ok((a, b))
}

fn lookup<'a>(pairs: &'a [(String, String)], name: &str) -> Option<&'a str> {
    pairs
        .iter()
        .find(|(n, _)| n == name)
        .map(|(_, v)| v.as_str())
}

/// Turn the four flag vectors into one spec per named vector.
///
/// `n_layer` supplies the default layer range — every layer the file can carry
/// (`1..=n_layer-1`; layer 0 never has a direction). It has to come from the
/// loaded model, which is why this resolves at install time and not at parse
/// time.
/// The layer range to use when the operator gave none, by `model_type`.
///
/// Curated, not inferred: each entry is a range that vectors for that model
/// were actually characterised over, not something derived from how many
/// direction tensors a file happens to contain.
///
/// `qwen4_exp` (Qwen3.8-Flash-Next) is 4..44 because all three known vectors
/// for it use that range — the published refusal projection and the two derived
/// in `examples/control-vectors/` — which makes it a property of the model
/// rather than of any single vector.
///
/// A model absent from this table gets no default and the flag is required.
/// That is the right answer for an unknown model: the useful range is an
/// empirical fact about where a direction is worth acting on, and there is
/// nothing to read it off from.
fn default_layers(model_type: &str, n_layer: usize) -> Option<(usize, usize)> {
    let (lo, hi) = match model_type {
        "qwen4_exp" => (4usize, 44usize),
        _ => return None,
    };
    // Guard the table against a checkpoint smaller than the curated range — a
    // config that would otherwise fail deeper in with a worse message. Written
    // `hi < n_layer` rather than `hi <= n_layer - 1` so it cannot underflow on
    // a degenerate layer count.
    (hi < n_layer).then_some((lo, hi))
}

pub fn resolve(
    vectors: &[(String, String)],
    layers: &[(String, String)],
    scales: &[(String, String)],
    modes: &[(String, String)],
    n_layer: usize,
    model_type: &str,
) -> Result<Vec<(String, ControlVectorSpec)>> {
    if vectors.is_empty() {
        return Ok(Vec::new());
    }
    anyhow::ensure!(n_layer > 1, "model reports {n_layer} layers");

    // A modifier naming a vector that was never declared is always a typo, and
    // silently ignoring it would serve the DEFAULT scale or range while the
    // operator reads their flag back off the command line and believes
    // otherwise. That is the whole failure class this feature has to avoid.
    for (flag, pairs) in [
        ("--control-vector-layers", layers),
        ("--control-vector-scale", scales),
        ("--control-vector-mode", modes),
    ] {
        for (name, _) in pairs {
            if !vectors.iter().any(|(n, _)| n == name) {
                bail!(
                    "{flag} names '{name}', which no --control-vector declares \
                     (declared: {})",
                    vectors
                        .iter()
                        .map(|(n, _)| n.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
        }
    }

    let mut out = Vec::with_capacity(vectors.len());
    for (name, path) in vectors {
        if out.iter().any(|(n, _): &(String, _)| n == name) {
            bail!("--control-vector name '{name}' given twice (names must be unique)");
        }
        // The range comes from the flag, else from a CURATED per-model value,
        // else it is an error. What it must never do again is fall back to
        // `1..n_layer-1` — every layer that can carry a direction — because
        // that is not the configuration anything was validated at. The shipped
        // refusal projection is characterised over 4..44, so the old default
        // silently served 1..47 while the design doc described 4..44.
        //
        // "Every tensor in the file" is not the same claim as "every layer
        // should be steered". A llama.cpp control vector carries a direction
        // for each layer it was derived over; which of those to APPLY is a
        // separate tuned decision, and the early layers in particular tend to
        // carry a direction that is real but not one you want to act on.
        //
        // A curated default is a different thing from an inferred one. 4..44 is
        // what all three known qwen3_exp vectors use — the published refusal
        // projection and both derived here — which makes it a property of the
        // MODEL rather than of any one vector. Recording that is useful;
        // guessing from tensor count is not.
        let (layer_start, layer_end) = match lookup(layers, name) {
            Some(r) => parse_range(r).map_err(|e| anyhow::anyhow!("{name}: {e}"))?,
            None => match default_layers(model_type, n_layer) {
                Some(r) => {
                    tracing::info!(
                        "--control-vector '{name}': no --control-vector-layers given, using the \
                         curated default {}-{} for model_type '{model_type}'. Pass the flag to \
                         override.",
                        r.0,
                        r.1
                    );
                    r
                }
                None => bail!(
                    "--control-vector '{name}' needs --control-vector-layers {name}=START-END. \
                     There is no curated default for model_type '{model_type}', and there is no \
                     generic one: applying a vector to every layer is not the configuration these \
                     are tuned at, so guessing would serve something other than what the flags \
                     say. Check the artifact's model card; the valid range here is 1-{}.",
                    n_layer - 1
                ),
            },
        };
        let scale = match lookup(scales, name) {
            Some(s) => s.parse::<f32>()?,
            None => 1.0,
        };
        let mode = match lookup(modes, name) {
            Some("add") => CvecMode::Add,
            _ => CvecMode::Project,
        };
        out.push((
            name.clone(),
            ControlVectorSpec {
                path: path.into(),
                scale,
                layer_start,
                layer_end,
                mode,
            },
        ));
    }
    Ok(out)
}

#[cfg(test)]
#[path = "control_vector_args_tests.rs"]
mod control_vector_args_tests;
