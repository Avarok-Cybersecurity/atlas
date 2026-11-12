// SPDX-License-Identifier: AGPL-3.0-only

//! Typed field readers for a `/completion` body, and the fields Atlas cannot
//! honor. llama.cpp reads JSON `null` as "absent"; so do these.

use serde_json::{Map, Value};

use super::error::CompatError;

/// llama.cpp's default DRY sequence breakers.
const DEFAULT_DRY_BREAKERS: [&str; 4] = ["\n", ":", "\"", "*"];

/// The sampler order documented as the default (`--samplers`): the one order
/// Atlas applies its own samplers in, so it is the only order accepted.
const DEFAULT_SAMPLERS: [&str; 7] = [
    "dry",
    "top_k",
    "typ_p",
    "top_p",
    "min_p",
    "xtc",
    "temperature",
];

/// The first of `names` present and not null.
pub(super) fn first<'a>(
    body: &'a Map<String, Value>,
    names: &[&'static str],
) -> Option<(&'static str, &'a Value)> {
    names
        .iter()
        .find_map(|n| body.get(*n).filter(|v| !v.is_null()).map(|v| (*n, v)))
}

/// llama.cpp's per-field message shape: `Field 'x': ...`.
pub(super) fn field_err(name: &str, msg: &str) -> CompatError {
    CompatError::invalid(format!("Field '{name}': {msg}"))
}

pub(super) fn num(
    body: &Map<String, Value>,
    names: &[&'static str],
) -> Result<Option<f64>, CompatError> {
    match first(body, names) {
        None => Ok(None),
        Some((_, Value::Number(n))) => Ok(n.as_f64()),
        Some((name, _)) => Err(field_err(name, "must be a number")),
    }
}

pub(super) fn int(
    body: &Map<String, Value>,
    names: &[&'static str],
) -> Result<Option<i64>, CompatError> {
    match num(body, names)? {
        None => Ok(None),
        Some(x) if x.fract() == 0.0 && x.abs() < 9.0e15 => Ok(Some(x as i64)),
        Some(_) => {
            let name = first(body, names).map_or("", |(n, _)| n);
            Err(field_err(name, "must be an integer"))
        }
    }
}

pub(super) fn flag(
    body: &Map<String, Value>,
    name: &'static str,
) -> Result<Option<bool>, CompatError> {
    match first(body, &[name]) {
        None => Ok(None),
        Some((_, Value::Bool(b))) => Ok(Some(*b)),
        Some((n, _)) => Err(field_err(n, "must be a boolean")),
    }
}

/// A feature Atlas lacks, left at its disabled value: fine. Anything else: 400.
fn require_num(
    body: &Map<String, Value>,
    names: &[&'static str],
    default: f64,
    why: &str,
) -> Result<(), CompatError> {
    if let Some(x) = num(body, names)?
        && (x - default).abs() > 1e-9
    {
        let name = first(body, names).map_or("", |(n, _)| n);
        return Err(CompatError::unsupported(name, why));
    }
    Ok(())
}

fn require_false(
    body: &Map<String, Value>,
    name: &'static str,
    why: &str,
) -> Result<(), CompatError> {
    if flag(body, name)? == Some(true) {
        return Err(CompatError::unsupported(name, why));
    }
    Ok(())
}

fn require_empty_array(
    body: &Map<String, Value>,
    name: &'static str,
    why: &str,
) -> Result<(), CompatError> {
    match first(body, &[name]) {
        None => Ok(()),
        Some((_, Value::Array(a))) if a.is_empty() => Ok(()),
        Some((n, _)) => Err(CompatError::unsupported(n, why)),
    }
}

/// Fields that select a sampler or behavior Atlas does not have. Each is
/// accepted at the value that means "off" and refused otherwise.
pub(super) fn reject_unsupported_features(body: &Map<String, Value>) -> Result<(), CompatError> {
    let dyn_why = "dynamic temperature is not implemented";
    require_num(body, &["dynatemp_range"], 0.0, dyn_why)?;
    require_num(
        body,
        &["typical_p"],
        1.0,
        "locally typical sampling is not implemented",
    )?;
    require_num(
        body,
        &["xtc_probability"],
        0.0,
        "the XTC sampler is not implemented",
    )?;
    require_num(
        body,
        &["mirostat"],
        0.0,
        "Mirostat sampling is not implemented",
    )?;
    require_num(
        body,
        &["repeat_last_n"],
        64.0,
        "the repetition-penalty window is set by the server, not per request; omit the field",
    )?;
    require_num(body, &["min_keep"], 0.0, "min_keep has no equivalent")?;
    require_num(body, &["n_indent"], 0.0, "n_indent has no equivalent")?;
    if let Some(x) = num(body, &["adaptive_target"])?
        && x >= 0.0
    {
        return Err(CompatError::unsupported(
            "adaptive_target",
            "adaptive-p sampling is not implemented",
        ));
    }
    if let Some(ms) = num(body, &["t_max_predict_ms"])?
        && ms > 0.0
    {
        return Err(CompatError::unsupported(
            "t_max_predict_ms",
            "a per-request generation time limit is not implemented; use n_predict",
        ));
    }
    require_false(
        body,
        "ignore_eos",
        "the end-of-sequence token cannot be suppressed per request",
    )?;
    require_false(
        body,
        "post_sampling_probs",
        "only pre-sampling log-probabilities are available; omit post_sampling_probs",
    )?;
    require_false(
        body,
        "timings_per_token",
        "per-token timings are not implemented",
    )?;
    require_false(
        body,
        "return_progress",
        "prompt progress events are not implemented",
    )?;
    require_false(
        body,
        "backend_sampling",
        "backend sampling is a llama.cpp feature",
    )?;
    require_false(
        body,
        "reasoning_control",
        "reasoning budget control is not implemented",
    )?;
    if let Some(b) = int(body, &["reasoning_budget_tokens"])?
        && b >= 0
    {
        return Err(CompatError::unsupported(
            "reasoning_budget_tokens",
            "reasoning budgets are not implemented on /completion",
        ));
    }
    require_empty_array(
        body,
        "lora",
        "per-request LoRA scales are not implemented on /completion",
    )?;
    require_empty_array(
        body,
        "response_fields",
        "response field selection is not implemented; the full body is always returned",
    )?;
    require_empty_array(
        body,
        "image_data",
        "multimodal input is not supported on /completion",
    )?;
    if let Some((n, v)) = first(body, &["samplers"]) {
        let names: Vec<&str> = v
            .as_array()
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        if !v.is_array() || names != DEFAULT_SAMPLERS {
            return Err(CompatError::unsupported(
                n,
                "the sampler order is fixed; only the default order can be requested",
            ));
        }
    }
    Ok(())
}

/// DRY: map the three knobs Atlas has; refuse the two it does not when DRY is on.
pub(super) fn dry_params(body: &Map<String, Value>) -> Result<(f32, f32, u32), CompatError> {
    let multiplier = num(body, &["dry_multiplier"])?.unwrap_or(0.0) as f32;
    let mut base = num(body, &["dry_base"])?.unwrap_or(1.75) as f32;
    if base < 1.0 {
        base = 1.75; // llama.cpp replaces a base below 1 with the default
    }
    let allowed = match int(body, &["dry_allowed_length"])? {
        None => 2,
        Some(n) if (0..=i64::from(i32::MAX)).contains(&n) => n as u32,
        Some(_) => {
            return Err(field_err(
                "dry_allowed_length",
                "must be between 0 and 2147483647",
            ));
        }
    };
    if multiplier > 0.0 {
        require_num(
            body,
            &["dry_penalty_last_n"],
            64.0,
            "the DRY scan window is set by the server",
        )?;
        if let Some((n, v)) = first(body, &["dry_sequence_breakers"]) {
            let got: Vec<&str> = v
                .as_array()
                .map(|a| a.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default();
            if got != DEFAULT_DRY_BREAKERS {
                return Err(CompatError::unsupported(
                    n,
                    "custom DRY sequence breakers are not implemented",
                ));
            }
        }
    }
    Ok((multiplier, base, allowed))
}

/// `stop`: a string or an array of strings.
pub(super) fn parse_stop(body: &Map<String, Value>) -> Result<Vec<String>, CompatError> {
    match first(body, &["stop"]) {
        None => Ok(Vec::new()),
        Some((_, Value::String(s))) => Ok(vec![s.clone()]),
        Some((_, Value::Array(a))) => a
            .iter()
            .map(|w| {
                w.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| field_err("stop", "must be an array of strings"))
            })
            .collect(),
        Some(_) => Err(field_err("stop", "must be a string or an array of strings")),
    }
}

/// `presence_penalty` / `frequency_penalty`: the range the sampler is tested over.
pub(super) fn bounded_penalty(
    body: &Map<String, Value>,
    name: &'static str,
) -> Result<f32, CompatError> {
    match num(body, &[name])? {
        None => Ok(0.0),
        Some(p) if (-2.0..=2.0).contains(&p) => Ok(p as f32),
        Some(p) => Err(CompatError::unsupported(
            name,
            &format!("Atlas accepts -2 to 2, got {p}"),
        )),
    }
}
