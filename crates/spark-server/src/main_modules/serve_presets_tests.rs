// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for `serve_presets`: the pure rendering rules on a synthetic preset,
//! and the REAL `kernels/gb10/*/MODEL.toml` presets round-tripped through clap
//! and `validate_serve_args` — the same SSOT round trip recipes take, run here
//! on the GPU-free skip-build host where the compiled registry is empty.

use super::*;
use std::path::{Path, PathBuf};

static SYNTHETIC: ServePreset = ServePreset {
    name: "synthetic-preset",
    hf_id: "org/repo",
    hf_revision: "branch",
    description: "a test preset",
    flags: &[
        ("max_seq_len", "4096"),
        ("num_drafts", "2"),
        ("speculative", "true"),
        ("enable_prefix_caching", "false"),
        (
            "default_chat_template_kwargs",
            "{\"reasoning_effort\":\"low\"}",
        ),
    ],
    env: &[
        ("ATLAS_SYNTH_GATE", "1"),
        ("ATLAS_SYNTH_CAP", "{max_seq_len}"),
    ],
};

fn synthetic() -> PresetMatch {
    PresetMatch {
        target: "some-target",
        preset: &SYNTHETIC,
    }
}

fn serve_args(argv: &[&str]) -> ServeArgs {
    let mut full = vec!["spark", "serve"];
    full.extend_from_slice(argv);
    match Cli::try_parse_from(full).expect("parses").command {
        Command::Serve(a) => a,
        other => panic!("not a serve command: {other:?}"),
    }
}

#[test]
fn renders_every_flag_the_operator_did_not_pass_plus_the_kernel_target() {
    let (argv, applied, overridden) = default_argv(synthetic(), &|_| false).unwrap();
    assert_eq!(
        argv,
        [
            "--max-seq-len",
            "4096",
            "--num-drafts",
            "2",
            "--speculative",
            "--default-chat-template-kwargs",
            "{\"reasoning_effort\":\"low\"}",
            "--kernel-target",
            "some-target",
        ]
    );
    // The presence flag defaulted to false renders NOTHING and says so.
    assert!(applied.contains(&"--enable-prefix-caching (off)".to_string()));
    assert!(overridden.is_empty());
    // And the rendered line is a valid serve command line.
    let args = serve_args(&argv.iter().map(String::as_str).collect::<Vec<_>>());
    assert_eq!(args.max_seq_len, 4096);
    assert_eq!(args.num_drafts, Some(2));
    assert!(args.speculative);
    assert!(!args.enable_prefix_caching);
    assert_eq!(args.kernel_target.as_deref(), Some("some-target"));
}

#[test]
fn operator_flags_are_not_re_rendered_even_when_equal_to_clap_defaults() {
    // `--num-drafts 1` is clap's default value; a value comparison could not
    // tell it from "omitted". `ValueSource` can, and the preset must yield.
    let set = |id: &str| matches!(id, "num_drafts" | "kernel_target");
    let (argv, _applied, overridden) = default_argv(synthetic(), &set).unwrap();
    assert!(!argv.contains(&"--num-drafts".to_string()), "{argv:?}");
    assert!(!argv.contains(&"--kernel-target".to_string()), "{argv:?}");
    assert_eq!(overridden, ["--num-drafts", "--kernel-target"]);
}

#[test]
fn a_key_that_is_not_a_serve_flag_is_refused_by_name() {
    static BAD: ServePreset = ServePreset {
        name: "bad",
        hf_id: "o/r",
        hf_revision: "",
        description: "",
        flags: &[("definitely_not_a_flag", "1")],
        env: &[],
    };
    // `flag_for` accepts any key (clap is the typo shield), so the refusal
    // comes from the re-parse: render it and hand it to clap.
    let m = PresetMatch {
        target: "t",
        preset: &BAD,
    };
    let (argv, _, _) = default_argv(m, &|_| false).unwrap();
    let mut full = vec!["spark".to_string(), "serve".to_string(), "bad".to_string()];
    full.extend(argv);
    let err = Cli::try_parse_from(&full).expect_err("unknown flag");
    assert!(err.to_string().contains("definitely-not-a-flag"), "{err}");
}

#[test]
fn placeholders_resolve_against_the_effective_args() {
    let args = serve_args(&[
        "m",
        "--max-seq-len",
        "65536",
        "--max-prefill-tokens",
        "2048",
    ]);
    assert_eq!(substitute("{max_seq_len}", &args).unwrap(), "65536");
    assert_eq!(
        substitute("ple={max_prefill_tokens};qsa={max_seq_len}", &args).unwrap(),
        "ple=2048;qsa=65536"
    );
    assert_eq!(substitute("9216", &args).unwrap(), "9216");
    let err = substitute("{gpu_memory_utilization}", &args).unwrap_err();
    assert!(err.to_string().contains("gpu_memory_utilization"), "{err}");
    assert!(substitute("{max_seq_len", &args).is_err());
}

#[test]
fn env_plan_sets_unset_variables_and_keeps_operator_values() {
    let args = serve_args(&["m", "--max-seq-len", "8192"]);
    let current = |var: &str| (var == "ATLAS_SYNTH_GATE").then(|| "0".to_string());
    let (apply, kept) = env_plan(&SYNTHETIC, &args, &current).unwrap();
    assert_eq!(apply, [("ATLAS_SYNTH_CAP".to_string(), "8192".to_string())]);
    assert_eq!(kept, [("ATLAS_SYNTH_GATE".to_string(), "0".to_string())]);
}

#[test]
fn non_preset_invocations_parse_exactly_as_before() {
    let argv: Vec<OsString> = ["spark", "serve", "org/Some-Model", "--max-seq-len", "1024"]
        .into_iter()
        .map(OsString::from)
        .collect();
    let (cli, applied) = parse_from(argv.clone()).unwrap();
    assert!(applied.is_none());
    let Command::Serve(got) = cli.command else {
        panic!("serve");
    };
    let Command::Serve(want) = Cli::try_parse_from(&argv).unwrap().command else {
        panic!("serve");
    };
    assert_eq!(got, want);
}

#[test]
fn hf_ids_and_directories_are_never_presets() {
    assert!(lookup("turboderp/Qwen3.8-Flash-Next-exl3").is_none());
    let tmp = tempfile::tempdir().unwrap();
    assert!(lookup(tmp.path().to_str().unwrap()).is_none());
}

// ── The real MODEL.toml presets ──

fn gb10_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/spark-server is two levels below the workspace root")
        .join("kernels")
        .join("gb10")
}

fn leak(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}

/// Every `[[serve_presets]]` entry in the tree as `(declaring target, preset)`,
/// parsed with the same value rules `build_parse::parse_serve_presets` applies.
fn real_presets() -> Vec<PresetMatch> {
    let mut out = Vec::new();
    let mut names: Vec<String> = std::fs::read_dir(gb10_dir())
        .expect("kernels/gb10 exists")
        .filter_map(|e| {
            let e = e.ok()?;
            let name = e.file_name().to_string_lossy().to_string();
            e.path().join("MODEL.toml").exists().then_some(name)
        })
        .collect();
    names.sort();
    for target in names {
        let path = gb10_dir().join(&target).join("MODEL.toml");
        let text = std::fs::read_to_string(&path).expect("readable MODEL.toml");
        let toml: toml::Value =
            toml::from_str(&text).unwrap_or_else(|e| panic!("bad TOML in {}: {e}", path.display()));
        let Some(arr) = toml.get("serve_presets").and_then(|v| v.as_array()) else {
            continue;
        };
        for entry in arr {
            let pairs = |key: &str| -> &'static [(&'static str, &'static str)] {
                let v: Vec<(&'static str, &'static str)> = entry
                    .get(key)
                    .and_then(|t| t.as_table())
                    .map(|t| {
                        t.iter()
                            .map(|(k, v)| {
                                let text = match v {
                                    toml::Value::String(s) => s.clone(),
                                    toml::Value::Integer(i) => i.to_string(),
                                    toml::Value::Float(f) => f.to_string(),
                                    toml::Value::Boolean(b) => b.to_string(),
                                    other => panic!("{key}.{k}: non-scalar {other}"),
                                };
                                (leak(k), leak(&text))
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                Box::leak(v.into_boxed_slice())
            };
            let s = |key: &str| leak(entry.get(key).and_then(|v| v.as_str()).unwrap_or(""));
            let preset: &'static ServePreset = Box::leak(Box::new(ServePreset {
                name: s("name"),
                hf_id: s("hf_id"),
                hf_revision: s("hf_revision"),
                description: s("description"),
                flags: pairs("flags"),
                env: pairs("env"),
            }));
            out.push(PresetMatch {
                target: leak(&target),
                preset,
            });
        }
    }
    out
}

/// Render each real preset with nothing overridden and hand it to clap +
/// the cross-flag validator: a preset that cannot boot is a preset that
/// lies, and this is where it fails instead of at an operator's startup.
#[test]
fn every_real_preset_renders_a_valid_serve_command_line() {
    let presets = real_presets();
    assert!(!presets.is_empty(), "no [[serve_presets]] in kernels/gb10");
    for m in presets {
        let (argv, _, _) =
            default_argv(m, &|_| false).unwrap_or_else(|e| panic!("{}: {e}", m.preset.name));
        let mut full = vec![
            "spark".to_string(),
            "serve".to_string(),
            m.preset.name.to_string(),
        ];
        full.extend(argv.iter().cloned());
        let cli = Cli::try_parse_from(&full)
            .unwrap_or_else(|e| panic!("{}: rendered {:?}: {e}", m.preset.name, argv));
        let Command::Serve(args) = cli.command else {
            panic!("serve")
        };
        crate::cli::validate_serve_args(&args).unwrap_or_else(|e| panic!("{}: {e}", m.preset.name));
        assert_eq!(args.kernel_target.as_deref(), Some(m.target));
        for (var, value) in m.preset.env {
            substitute(value, &args)
                .unwrap_or_else(|e| panic!("{}: env {var}: {e}", m.preset.name));
        }
    }
}

/// The entry this feature exists for, pinned field by field against the
/// validated 2026-09-05 configuration (.research/exl3_decode_perf).
#[test]
fn qwen38_flash_next_exl3_preset_carries_the_validated_configuration() {
    let m = real_presets()
        .into_iter()
        .find(|m| m.preset.name == "qwen3.8-flash-next-exl3")
        .expect("qwen3.8-flash-next declares the exl3 preset");
    assert_eq!(m.target, "qwen3.8-flash-next");
    assert_eq!(m.preset.hf_id, "turboderp/Qwen3.8-Flash-Next-exl3");
    assert_eq!(m.preset.hf_revision, "4.05bpw_h6_ng6");

    let (argv, _, _) = default_argv(m, &|_| false).unwrap();
    let args = serve_args(
        &std::iter::once(m.preset.name)
            .chain(argv.iter().map(String::as_str))
            .collect::<Vec<_>>(),
    );
    // 128K context, four sequences (operator's serving envelope, 2026-09-05).
    assert_eq!(args.max_seq_len, 131072);
    assert_eq!(args.max_num_seqs, 4);
    assert_eq!(args.max_batch_size, 4);
    assert_eq!(args.gpu_memory_utilization, 0.72);
    assert_eq!(args.ssm_cache_slots, 64);
    // Four queued 30K prefills exceed the 300 s default (measured 2026-09-05).
    assert_eq!(args.request_timeout, 1800);
    assert!(args.fast_load_prefetch_shards);
    assert!(args.speculative);
    assert_eq!(args.num_drafts, Some(2));
    assert!(args.enable_prefix_caching, "prefix caching is ON by rule");
    // Brief reasoning + prior turns' thinking preserved in the rendered history.
    assert_eq!(
        args.default_chat_template_kwargs.as_deref(),
        Some("{\"reasoning_effort\":\"low\",\"preserve_thinking\":true}")
    );
    // fp8 KV is not a flag default here: `[behavior].default_kv_dtype = bf16`
    // already owns it, and preflight refuses anything else for QSA.
    assert_eq!(args.kv_cache_dtype, None);

    let (apply, kept) = env_plan(m.preset, &args, &|_| None).unwrap();
    assert!(kept.is_empty());
    let get = |var: &str| {
        apply
            .iter()
            .find(|(k, _)| k == var)
            .map(|(_, v)| v.as_str())
            .unwrap_or_else(|| panic!("preset env lacks {var}"))
    };
    for var in [
        "ATLAS_EXL3_NATIVE",
        "ATLAS_EXL3_NATIVE_MOE",
        "ATLAS_EXL3_NATIVE_DENSE",
        "ATLAS_QWEN4EXP_MTP",
        "ATLAS_QWEN4EXP_MTP_VERIFY",
        "ATLAS_DFLASH_SPEC_THINK",
        "ATLAS_QWEN4EXP_MTP_HC_BATCHED",
        "ATLAS_VERIFY_EXL3_ROW_ROUTER",
        "ATLAS_VERIFY_EXL3_STABLE_GRID",
        "ATLAS_NO_VERIFY_ROW_FFN",
        "ATLAS_NO_THINKENDED_GPU_ARGMAX",
    ] {
        assert_eq!(get(var), "1", "{var}");
    }
    assert_eq!(get("ATLAS_INTHINK_TOOL_LEAK_OPENERS"), "0");
    assert_eq!(get("ATLAS_PLE_CACHE_SLOTS"), "4194304");
    // The QSA cap tracks --max-seq-len; the PLE cap covers the default chunk.
    assert_eq!(get("ATLAS_QSA_MAX_TOKENS"), "131072");
    // The private MTP draft KV pool is sized to the preset's sequence slots.
    assert_eq!(get("ATLAS_MTP_MAX_SEQS"), "4");
    // Round-2 prefill levers pinned to their measured values (2026-09-06).
    assert_eq!(get("ATLAS_EXL3_MOE_ROWS_PER_EXPERT"), "1024");
    assert_eq!(get("ATLAS_EXL3_DENSE_RECONSTRUCT_ROWS"), "512");
    assert!(get("ATLAS_PLE_MAX_TOKENS").parse::<usize>().unwrap() >= args.max_prefill_tokens);

    // An operator override of --max-seq-len carries into the QSA cap.
    let (argv, _, _) = default_argv(m, &|id| id == "max_seq_len").unwrap();
    let args = serve_args(
        &[m.preset.name, "--max-seq-len", "65536"]
            .into_iter()
            .chain(argv.iter().map(String::as_str))
            .collect::<Vec<_>>(),
    );
    let (apply, _) = env_plan(m.preset, &args, &|_| None).unwrap();
    assert!(apply.contains(&("ATLAS_QSA_MAX_TOKENS".to_string(), "65536".to_string())));
}
