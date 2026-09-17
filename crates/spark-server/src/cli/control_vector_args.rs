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
pub fn resolve(
    vectors: &[(String, String)],
    layers: &[(String, String)],
    scales: &[(String, String)],
    modes: &[(String, String)],
    n_layer: usize,
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
        let (layer_start, layer_end) = match lookup(layers, name) {
            Some(r) => parse_range(r).map_err(|e| anyhow::anyhow!("{name}: {e}"))?,
            None => (1, n_layer - 1),
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
