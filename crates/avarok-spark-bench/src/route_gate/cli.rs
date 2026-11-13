// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the command line of the `route_gate` binary: a tiny flag parser and the dispatch to the eight
//! subcommands (`build`, `collect`, `grade`, `summary`, `rows`, `goldens`, `tokens`, `fit`) plus `hints`.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{Result, anyhow, bail};

use super::fit_eval::{DEFAULT_GRID, FitArgs, StudyParams, cmd_fit, cmd_selftest};
use super::goldens::{GoldensArgs, cmd_goldens, cmd_tokens};
use super::prep::{self, BuildArgs, CollectArgs, GradeArgs};
use super::rows::{RowsArgs, cmd_rows};

const USAGE: &str = "route_gate: the L04 routing-gate study (CPU only)

usage: route_gate <command> [flags]

  build    --data-dir D --out prompts.jsonl [--quota gsm8k=120,...] [--seed N] [--require-all]
  collect  --prompts prompts.jsonl --url http://127.0.0.1:18080 --model qwen35-4b --out generations.jsonl
           [--concurrency 4] [--thinking] [--limit N] [--timeout 900] [--max-tokens-scale 1.0]
  grade    --prompts prompts.jsonl --generations generations.jsonl --out labels.jsonl
           [--code-exec auto|off|subprocess|unshare] [--timeout 20]
  summary  --labels labels.jsonl
  hints                                          where each dataset comes from (prints only)
  rows     --release ~/models/ARC-1.0 --prompts prompts.jsonl --out readouts.safetensors
           [--limit N] [--block-end nl|im_end] [--shell-file F] [--max-tokens 8192] [--fresh] [--bf16] [--threads 8]
  goldens  --release ~/models/ARC-1.0 [--out goldens_report.json] [--shell-file F ...] [--no-verify] [--bf16] [--threads 8]
  tokens   --release ~/models/ARC-1.0 [--shell-file F] [--out tokens.json]
  fit      --labels labels.jsonl --readouts readouts.safetensors [--release DIR] [--no-verify] [--regime fixed]
           [--oracle-report goldens_report.json] [--out-dir DIR] [--folds 5] [--repeats 3] [--seed N]
           [--bootstrap 1000] [--grid 1,10,...] [--drop-truncated] [--drop-no-answer]
  fit --selftest                                 the study on planted synthetic data
";

/// Parsed flags: valued flags (repeatable) and switches.
#[derive(Debug, Default)]
pub struct Flags {
    vals: HashMap<String, Vec<String>>,
    switches: HashSet<String>,
}

impl Flags {
    /// Parses `--name value`, `--name=value` and bare switches; anything not declared is an error.
    pub fn parse(args: &[String], valued: &[&str], switches: &[&str]) -> Result<Self> {
        let mut f = Flags::default();
        let mut i = 0;
        while i < args.len() {
            let a = &args[i];
            let Some(body) = a.strip_prefix("--") else {
                bail!("unexpected argument {a:?}");
            };
            let (name, inline) = match body.split_once('=') {
                Some((n, v)) => (n, Some(v.to_string())),
                None => (body, None),
            };
            if switches.contains(&name) {
                f.switches.insert(name.to_string());
            } else if valued.contains(&name) {
                let v = match inline {
                    Some(v) => v,
                    None => {
                        i += 1;
                        args.get(i)
                            .cloned()
                            .ok_or_else(|| anyhow!("--{name} needs a value"))?
                    }
                };
                f.vals.entry(name.to_string()).or_default().push(v);
            } else {
                bail!("unknown flag --{name}");
            }
            i += 1;
        }
        Ok(f)
    }

    pub fn has(&self, name: &str) -> bool {
        self.switches.contains(name)
    }

    pub fn get(&self, name: &str) -> Option<&str> {
        self.vals
            .get(name)
            .and_then(|v| v.last())
            .map(String::as_str)
    }

    pub fn req(&self, name: &str) -> Result<&str> {
        self.get(name)
            .ok_or_else(|| anyhow!("--{name} is required"))
    }

    pub fn path(&self, name: &str) -> Result<PathBuf> {
        Ok(PathBuf::from(self.req(name)?))
    }

    pub fn opt_path(&self, name: &str) -> Option<PathBuf> {
        self.get(name).map(PathBuf::from)
    }

    pub fn many(&self, name: &str) -> Vec<String> {
        self.vals.get(name).cloned().unwrap_or_default()
    }

    /// The value parsed as `T`, or `default` when the flag is absent.
    pub fn num<T: FromStr>(&self, name: &str, default: T) -> Result<T> {
        match self.get(name) {
            None => Ok(default),
            Some(s) => s
                .parse()
                .map_err(|_| anyhow!("--{name}: cannot parse {s:?}")),
        }
    }
}

fn set_threads(f: &Flags) -> Result<()> {
    let n: usize = f.num("threads", 8)?;
    if n > 0 {
        // Leaves cores to whoever else is on the machine; the global pool can only be built once.
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global();
    }
    Ok(())
}

fn dispatch(cmd: &str, rest: &[String]) -> Result<i32> {
    match cmd {
        "build" => {
            let f = Flags::parse(
                rest,
                &["data-dir", "out", "quota", "seed"],
                &["require-all"],
            )?;
            prep::cmd_build(&BuildArgs {
                data_dir: f.path("data-dir")?,
                out: f.path("out")?,
                quota: f.get("quota").map(str::to_string),
                seed: f.num("seed", 20_261_112u64)?,
                require_all: f.has("require-all"),
            })
        }
        "collect" => {
            let f = Flags::parse(
                rest,
                &[
                    "prompts",
                    "url",
                    "model",
                    "out",
                    "concurrency",
                    "limit",
                    "timeout",
                    "max-tokens-scale",
                ],
                &["thinking"],
            )?;
            prep::cmd_collect(&CollectArgs {
                prompts: f.path("prompts")?,
                url: f.req("url")?.to_string(),
                model: f.req("model")?.to_string(),
                out: f.path("out")?,
                concurrency: f.num("concurrency", 4usize)?,
                thinking: f.has("thinking"),
                limit: f.num("limit", 0usize)?,
                timeout: f.num("timeout", 900.0f64)?,
                max_tokens_scale: f.num("max-tokens-scale", 1.0f64)?,
            })
        }
        "grade" => {
            let f = Flags::parse(
                rest,
                &["prompts", "generations", "out", "code-exec", "timeout"],
                &[],
            )?;
            let mode = f.get("code-exec").unwrap_or("auto");
            if !["auto", "off", "subprocess", "unshare"].contains(&mode) {
                bail!("--code-exec must be one of auto, off, subprocess, unshare");
            }
            prep::cmd_grade(&GradeArgs {
                prompts: f.path("prompts")?,
                generations: f.path("generations")?,
                out: f.path("out")?,
                code_exec: mode.to_string(),
                timeout: f.num("timeout", 20.0f64)?,
            })
        }
        "summary" => {
            let f = Flags::parse(rest, &["labels"], &[])?;
            prep::cmd_summary(&f.path("labels")?)
        }
        "hints" => prep::cmd_hints(),
        "rows" => {
            let f = Flags::parse(
                rest,
                &[
                    "release",
                    "prompts",
                    "out",
                    "limit",
                    "block-end",
                    "shell-file",
                    "max-tokens",
                    "threads",
                ],
                &["fresh", "bf16"],
            )?;
            set_threads(&f)?;
            let block_end = f.get("block-end").unwrap_or("nl").to_string();
            if !["nl", "im_end"].contains(&block_end.as_str()) {
                bail!("--block-end must be nl or im_end");
            }
            cmd_rows(&RowsArgs {
                release: f.path("release")?,
                prompts: f.path("prompts")?,
                out: f.path("out")?,
                limit: f.num("limit", 0usize)?,
                block_end,
                shell_file: f.opt_path("shell-file"),
                max_tokens: f.num("max-tokens", 8192usize)?,
                fresh: f.has("fresh"),
                bf16: f.has("bf16"),
            })
        }
        "goldens" => {
            let f = Flags::parse(
                rest,
                &["release", "out", "shell-file", "threads"],
                &["no-verify", "bf16"],
            )?;
            set_threads(&f)?;
            cmd_goldens(&GoldensArgs {
                release: f.path("release")?,
                out: f.opt_path("out"),
                shell_files: f
                    .many("shell-file")
                    .into_iter()
                    .map(PathBuf::from)
                    .collect(),
                no_verify: f.has("no-verify"),
                try_bf16: f.has("bf16"),
            })
        }
        "tokens" => {
            let f = Flags::parse(rest, &["release", "shell-file", "out"], &[])?;
            cmd_tokens(
                &f.path("release")?,
                &f.opt_path("shell-file"),
                &f.opt_path("out"),
            )
        }
        "fit" => {
            let f = Flags::parse(
                rest,
                &[
                    "labels",
                    "readouts",
                    "release",
                    "regime",
                    "oracle-report",
                    "out-dir",
                    "folds",
                    "repeats",
                    "seed",
                    "bootstrap",
                    "grid",
                ],
                &["selftest", "no-verify", "drop-truncated", "drop-no-answer"],
            )?;
            if f.has("selftest") {
                return cmd_selftest();
            }
            let grid: Vec<f64> = match f.get("grid") {
                Some(g) => g
                    .split(',')
                    .map(|v| {
                        v.trim()
                            .parse::<f64>()
                            .map_err(|_| anyhow!("--grid: bad value {v:?}"))
                    })
                    .collect::<Result<_>>()?,
                None => DEFAULT_GRID.to_vec(),
            };
            cmd_fit(&FitArgs {
                labels: f.path("labels")?,
                readouts: f.path("readouts")?,
                release: f.opt_path("release"),
                no_verify: f.has("no-verify"),
                regime: f.get("regime").unwrap_or("fixed").to_string(),
                oracle_report: f.opt_path("oracle-report"),
                out_dir: f
                    .opt_path("out-dir")
                    .unwrap_or_else(|| PathBuf::from("route_gate_out")),
                params: StudyParams {
                    folds: f.num("folds", 5usize)?,
                    repeats: f.num("repeats", 3usize)?,
                    seed: f.num("seed", 20_261_112u64)?,
                    bootstrap: f.num("bootstrap", 1000usize)?,
                    grid,
                },
                drop_truncated: f.has("drop-truncated"),
                drop_no_answer: f.has("drop-no-answer"),
            })
        }
        "help" | "--help" | "-h" => {
            print!("{USAGE}");
            Ok(0)
        }
        other => bail!("unknown command {other:?}\n\n{USAGE}"),
    }
}

/// Runs the CLI with `args` (without the program name); returns the process exit code.
pub fn run(args: &[String]) -> i32 {
    let Some((cmd, rest)) = args.split_first() else {
        eprint!("{USAGE}");
        return 2;
    };
    match dispatch(cmd, rest) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("route_gate {cmd}: {e:#}");
            2
        }
    }
}
