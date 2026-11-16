// SPDX-License-Identifier: AGPL-3.0-only
//! The full-LM oracle of the route_gate CPU encoder, CPU only:
//!   `encoder_check identity <arc_dir> <stock_dir> [--out F]`
//!   `encoder_check lm <stock_dir> [--arc <arc_dir>] [--max-new 32] [--rope stock,yarn-f4] [--threads 8] [--out F]`
//! Library code lives in `avarok_spark_bench::route_gate::lm_check`; this file only parses the arguments.

use std::path::PathBuf;

use avarok_spark_bench::route_gate::lm_check::{LmArgs, RopeSetting, identity, run_lm};

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn positional(args: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut skip = false;
    for a in args {
        if skip {
            skip = false;
        } else if a == "--full-vectors" {
            // the one flag without a value
        } else if a.starts_with("--") {
            skip = true;
        } else {
            out.push(a.clone());
        }
    }
    out
}

fn run(args: &[String]) -> Result<serde_json::Value, String> {
    let threads: usize = flag(args, "--threads")
        .map_or(Ok(8), |s| s.parse())
        .map_err(|e| format!("--threads: {e}"))?;
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global();
    let pos = positional(args);
    match pos.first().map(String::as_str) {
        Some("identity") if pos.len() == 3 => {
            identity(&PathBuf::from(&pos[1]), &PathBuf::from(&pos[2])).map_err(|e| e.to_string())
        }
        Some("lm") if pos.len() == 2 => {
            let settings = flag(args, "--rope")
                .unwrap_or_else(|| "stock,yarn-f4".into())
                .split(',')
                .map(|s| match s {
                    "stock" => Ok(RopeSetting::Stock),
                    "yarn-f4" => Ok(RopeSetting::YarnF4),
                    other => Err(format!("unknown rope setting {other}")),
                })
                .collect::<Result<Vec<_>, _>>()?;
            let max_new = flag(args, "--max-new")
                .map_or(Ok(32), |s| s.parse())
                .map_err(|e| format!("--max-new: {e}"))?;
            let stock = PathBuf::from(&pos[1]);
            let arc = flag(args, "--arc").map(PathBuf::from);
            run_lm(&LmArgs {
                stock: &stock,
                arc: arc.as_deref(),
                max_new,
                settings,
                full_vectors: args.iter().any(|x| x == "--full-vectors"),
            })
            .map_err(|e| e.to_string())
        }
        _ => Err(
            "usage: encoder_check identity <arc> <stock> | lm <stock> [--arc A] [--out F]".into(),
        ),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(v) => {
            let text = serde_json::to_string_pretty(&v).unwrap_or_default();
            if let Some(out) = flag(&args, "--out") {
                if let Err(e) = std::fs::write(&out, &text) {
                    eprintln!("write {out}: {e}");
                    std::process::exit(2);
                }
                eprintln!("wrote {out}");
            } else {
                println!("{text}");
            }
        }
        Err(e) => {
            eprintln!("encoder_check: {e}");
            std::process::exit(1);
        }
    }
}
