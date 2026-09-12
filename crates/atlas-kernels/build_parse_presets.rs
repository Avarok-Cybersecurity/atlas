// SPDX-License-Identifier: AGPL-3.0-only
//
// `[[serve_presets]]` MODEL.toml parsing for build.rs. Split from
// `build_parse.rs` (500-LoC cap), the same way `build_parse_behavior.rs` is:
// a child module of `build_parse`, so `crate::` reaches build.rs types.

use crate::ServePresetRaw;

/// Parse `[[serve_presets]]` from MODEL.toml (see `src/serve_preset.rs`).
///
/// Absent array → empty. Malformed entries PANIC with the offending key
/// named: a preset is a servable name, and a preset that parsed leniently
/// would serve a different configuration than the file says. Flag values may
/// be strings, integers, floats or booleans — each becomes the text clap
/// would have read from the command line (`true`/`false` for presence
/// flags, per the recipe schema). Env values must be strings so a
/// `{max_seq_len}` placeholder cannot be mistaken for a number.
pub(crate) fn parse_serve_presets(model_dir: &std::path::Path) -> Vec<ServePresetRaw> {
    let path = model_dir.join("MODEL.toml");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    let toml: toml::Value =
        toml::from_str(&text).unwrap_or_else(|e| panic!("Bad TOML in {}: {e}", path.display()));
    let Some(arr) = toml.get("serve_presets") else {
        return Vec::new();
    };
    let arr = arr.as_array().unwrap_or_else(|| {
        panic!(
            "{}: serve_presets must be an array of tables ([[serve_presets]])",
            path.display()
        )
    });
    // The needles/values are emitted into generated Rust via `{:?}` (escaped),
    // so any text is representable; the shape rules below are about MEANING.
    let scalar_text = |key: &str, v: &toml::Value| -> String {
        match v {
            toml::Value::String(s) => s.clone(),
            toml::Value::Integer(i) => i.to_string(),
            toml::Value::Float(f) => f.to_string(),
            toml::Value::Boolean(b) => b.to_string(),
            other => panic!(
                "{}: serve_presets.flags.{key} must be a scalar (string/int/float/bool), got {other}",
                path.display()
            ),
        }
    };
    let mut out = Vec::new();
    for entry in arr {
        let tbl = entry.as_table().unwrap_or_else(|| {
            panic!(
                "{}: each [[serve_presets]] entry must be a table",
                path.display()
            )
        });
        let required = |key: &str| -> String {
            let s = tbl.get(key).and_then(|v| v.as_str()).unwrap_or_else(|| {
                panic!(
                    "{}: [[serve_presets]] entry is missing string key {key:?}",
                    path.display()
                )
            });
            assert!(
                !s.trim().is_empty(),
                "{}: [[serve_presets]] {key} must be non-empty",
                path.display()
            );
            s.to_string()
        };
        let name = required("name");
        assert!(
            !name.contains('/') && !name.chars().any(char::is_whitespace),
            "{}: serve preset name {name:?} must not contain '/' or whitespace — it is the \
             positional MODEL argument and must never look like an HF id or a path",
            path.display()
        );
        let hf_id = required("hf_id");
        assert!(
            hf_id.contains('/'),
            "{}: serve preset {name:?} hf_id {hf_id:?} is not an `org/repo` HuggingFace id",
            path.display()
        );
        let hf_revision = tbl
            .get("hf_revision")
            .map(|v| {
                v.as_str()
                    .unwrap_or_else(|| {
                        panic!(
                            "{}: serve preset {name:?} hf_revision must be a string",
                            path.display()
                        )
                    })
                    .to_string()
            })
            .unwrap_or_default();
        let description = tbl
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let mut flags = Vec::new();
        if let Some(f) = tbl.get("flags") {
            let f = f.as_table().unwrap_or_else(|| {
                panic!(
                    "{}: serve preset {name:?} flags must be a table",
                    path.display()
                )
            });
            for (k, v) in f {
                // The checkpoint and the kernel target are what the preset IS;
                // letting a flag default restate them would let the two
                // disagree. `model_from_path` stays an operator override.
                assert!(
                    !matches!(k.as_str(), "model" | "model_from_path" | "kernel_target"),
                    "{}: serve preset {name:?} flags.{k} is not allowed — the checkpoint comes \
                     from hf_id/hf_revision and the kernel target from the declaring MODEL.toml",
                    path.display()
                );
                flags.push((k.clone(), scalar_text(k, v)));
            }
        }
        let mut env = Vec::new();
        if let Some(e) = tbl.get("env") {
            let e = e.as_table().unwrap_or_else(|| {
                panic!(
                    "{}: serve preset {name:?} env must be a table",
                    path.display()
                )
            });
            for (k, v) in e {
                assert!(
                    !k.is_empty()
                        && k.chars()
                            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_'),
                    "{}: serve preset {name:?} env key {k:?} must be an UPPER_SNAKE_CASE variable name",
                    path.display()
                );
                let val = v.as_str().unwrap_or_else(|| {
                    panic!(
                        "{}: serve preset {name:?} env.{k} must be a string (quote numbers: \"1\")",
                        path.display()
                    )
                });
                env.push((k.clone(), val.to_string()));
            }
        }
        out.push(ServePresetRaw {
            name,
            hf_id,
            hf_revision,
            description,
            flags,
            env,
        });
    }
    out
}
