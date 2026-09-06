// SPDX-License-Identifier: AGPL-3.0-only

//! Arguments for `spark benchmark`.
//!
//! The same suite the dashboard runs, without a terminal — so a benchmark can
//! be scripted, run in CI, or driven over SSH on a headless box.

/// `spark benchmark <list|run|history>` — or `--pull-request-gate-check` on
/// its own, which needs no subcommand.
#[derive(clap::Args, Debug)]
#[command(arg_required_else_help = true)]
pub struct BenchmarkArgs {
    /// Check the committed `.benchmarks/` records for THIS commit: every
    /// required gate must have a passing record. Prints what is missing or
    /// failing and exits non-zero when the branch is not fully gated.
    /// Runs without a subcommand (and without an endpoint).
    #[arg(long = "pull-request-gate-check")]
    pub pull_request_gate_check: bool,

    /// PR number, for the ADVISORY intent half of `--pull-request-gate-check`.
    ///
    /// The journey ledger is keyed by PR (`governance/pr-<n>.jsonl`), and the
    /// gate otherwise has only a sha — so without this the classified intent
    /// cannot be found at all.
    ///
    /// ★ PCND, and there is no default: guessing a PR number would attribute
    /// another PR's classification to this one. Absent means `NotRequested`,
    /// which is the honest reading of a local run or a push build, and the
    /// verdict is unchanged either way — `gate::exit_code` takes only the
    /// verdicts, by signature.
    ///
    /// The pairing with `--pull-request-gate-check` is enforced by
    /// [`BenchmarkArgs::reject_orphan_pr`], not by clap's `requires`: the
    /// target is a `SetTrue` flag whose implicit `false` default counts as
    /// "present" to clap 4's requirement check, so the attribute silently
    /// never fires (same reason [`RunArgs::checkpoint`] documents).
    #[arg(long)]
    pub pr: Option<u64>,
    #[command(subcommand)]
    pub command: Option<BenchmarkCommand>,
}

impl BenchmarkArgs {
    /// Refuse `--pr` without `--pull-request-gate-check`.
    ///
    /// `--pr` exists only to key the gate check's advisory intent lookup;
    /// on any other invocation it would be a flag that visibly does nothing
    /// — and, with no subcommand given, it previously parsed clean and then
    /// panicked on the `expect("clap enforces a subcommand here")` in
    /// dispatch. An `Err` here is a usage error, phrased like one.
    pub fn reject_orphan_pr(&self) -> Result<(), String> {
        if self.pr.is_some() && !self.pull_request_gate_check {
            return Err(
                "--pr keys the advisory intent lookup of --pull-request-gate-check and does \
                 nothing anywhere else; pass --pull-request-gate-check with it, or drop --pr."
                    .to_string(),
            );
        }
        Ok(())
    }
}

#[derive(clap::Subcommand, Debug)]
pub enum BenchmarkCommand {
    /// List the suite, or one benchmark's parameter schema.
    List(ListArgs),
    /// Run one benchmark against a served endpoint.
    Run(RunArgs),
    /// Show a benchmark group's aggregate over its committed shard records.
    ///
    /// Pure and GPU-free: it reads what is already in `.benchmarks/` and applies
    /// the same aggregation the gate does, so an operator can see the group's
    /// number — and WHICH shard is missing — without waiting for CI to say so.
    Aggregate(AggregateArgs),
    /// Past runs, from `~/.atlas/runs`.
    History(HistoryArgs),
    /// Render a shareable result card from a committed gate record.
    ///
    /// Separate from `run --output-image` on purpose: a card can be regenerated
    /// from any past record, with different attribution, without spending a GPU
    /// hour re-measuring. It reads a COMMITTED record because that is the only
    /// artefact carrying the hardware and the commit the number belongs to — the
    /// configuration a card exists to print.
    Card(CardArgs),
}

#[derive(clap::Args, Debug)]
pub struct CardArgs {
    /// A benchmark ID (`decode-floor`) or a path to a record.
    ///
    /// The ID form is the one people will use: it takes the newest committed
    /// record for that benchmark, which is almost always the run they just did.
    /// A path is the escape hatch for "that specific older result".
    pub record: String,
    /// Where to write. A NAME becomes `./<name>.svg`; a path is taken literally.
    #[arg(long = "output-image", value_name = "NAME|PATH")]
    pub output_image: Option<String>,
    /// `author=Ada,handle=@ada,website=ada.dev`
    #[arg(long = "output-image-args", value_name = "K=V,...")]
    pub output_image_args: Option<String>,
}

#[derive(clap::Args, Debug)]
pub struct ListArgs {
    /// Benchmark id. Omit for the whole suite.
    pub id: Option<String>,
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    pub format: OutputFormat,
}

#[derive(clap::Args, Debug)]
pub struct RunArgs {
    /// Benchmark id — `spark benchmark list` prints them.
    pub id: String,
    /// The endpoint to drive.
    ///
    /// This does NOT start a server — EXCEPT under `--pull-request-gate`, which
    /// serves the benchmark's own recipe on a free port and tears it down
    /// after. The two are mutually exclusive: a gate run has nowhere to send
    /// this, so passing it is rejected rather than quietly overridden.
    #[arg(
        long,
        default_value = "http://127.0.0.1:8888",
        conflicts_with = "pull_request_gate"
    )]
    pub url: String,
    /// The `model` field sent in every request.
    ///
    /// Required rather than defaulted: it is recorded with the run, and a
    /// result that cannot say what it measured is not worth keeping. Under
    /// `--pull-request-gate` it is supplied by the benchmark's recipe instead,
    /// and passing it is rejected rather than silently ignored — a flag that
    /// looks like it selects the model while the recipe actually does is the
    /// confusion this mode exists to remove.
    #[arg(
        long,
        required_unless_present = "pull_request_gate",
        conflicts_with = "pull_request_gate"
    )]
    pub model: Option<String>,

    /// Which box class the run is for, e.g. `gb10`.
    ///
    /// Only consulted under `--pull-request-gate`, to pick the baseline entry
    /// when a benchmark has thresholds for more than one box class. With a
    /// single entry it is inferred; with several, omitting it is an error
    /// rather than a guess.
    #[arg(long)]
    pub hardware: Option<String>,
    /// Which model VARIANT of the benchmark to run, as the checkpoint id its
    /// `BENCH.toml` entry names — e.g. `unsloth/Qwen3.8-27B-NVFP4`.
    ///
    /// Gate-only, like `--hardware`. Omitted, the run takes the one checkpoint
    /// the benchmark's baseline marks `default = true` — an explicit committed
    /// declaration, not a guess (two defaults, or none, refuse to assemble at
    /// all). A checkpoint the baseline does not carry is an error naming what
    /// exists. Distinct from `--model` on purpose: `--model` names a request
    /// field against a server someone else started, while this selects which
    /// (thresholds, serve recipe) pair the gate provisions and is recorded as
    /// the record's `target_model`.
    ///
    /// The pairing is enforced by [`RunArgs::reject_orphan_checkpoint`], not by
    /// clap's `requires`: `--pull-request-gate` is a `SetTrue` flag whose
    /// implicit `false` default counts as "present" to clap 4's requirement
    /// check, so the attribute silently never fires.
    #[arg(long)]
    pub checkpoint: Option<String>,
    /// Override one parameter, e.g. `--param osl=8`. Repeatable.
    ///
    /// Anything not overridden takes the schema default and is still recorded.
    #[arg(long = "param", value_name = "KEY=VALUE", value_parser = parse_kv)]
    pub params: Vec<(String, String)>,
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    pub format: OutputFormat,
    /// How often to drain the run's channels, in milliseconds.
    #[arg(long, default_value_t = 250)]
    pub poll_ms: u64,
    /// Do not write the run to `~/.atlas/runs`.
    #[arg(long)]
    pub no_save: bool,
    /// Confirm a benchmark with side effects beyond load on the endpoint.
    ///
    /// Required for `agentic-webserver`, which executes model-authored shell.
    #[arg(long)]
    pub yes: bool,
    /// Print only the final report, not per-phase progress.
    #[arg(long)]
    pub quiet: bool,
    /// Exit 0 even when the gate verdict is FAIL.
    #[arg(long)]
    pub no_fail_on_verdict: bool,
    /// Do not ask the endpoint two known-answer questions before measuring.
    ///
    /// The probe only WARNS — it never refuses to start — so this is for
    /// skipping the two extra completions, not for silencing a veto.
    #[arg(long)]
    pub skip_coherence_probe: bool,
    /// Commit this run as a gate record under the repo's `.benchmarks/<id>/`.
    ///
    /// The record carries the metrics, verdict, hardware fingerprint, the
    /// exact command and the current commit sha, so the branch itself can
    /// answer "did this pass" — no `~/.atlas` state required.
    #[arg(long)]
    pub pull_request_gate: bool,
    /// Override one SERVE key from the benchmark's recipe, e.g.
    /// `--serve-override kv_cache_dtype=fp8`. Repeatable.
    ///
    /// Distinct from `--param`, which sets the BENCHMARK's own knobs
    /// (iterations, max_tokens). This one reaches the recipe that starts the
    /// server, so it is how you exercise a code path the recipe's pinned
    /// config never reaches — the case that motivated it: every gate recipe
    /// pins `kv_cache_dtype: bf16`, so a change to the fp8-KV attention
    /// kernel could not be measured by any gate at all. Five greens that
    /// never executed the changed code are worse than no run, because they
    /// read as evidence.
    ///
    /// Keys are recipe `defaults` keys, and `Recipe::argv` REFUSES one that
    /// is absent there — a typo fails loudly instead of silently measuring
    /// the unmodified config. `port` is rejected: the gate picks a free port
    /// and a second opinion about it would race the listener.
    ///
    /// ★ Every override is written into the gate record. A record whose
    /// numbers came from a config other than its recipe's must say so, or it
    /// is a plausible number attached to the wrong provenance — which is the
    /// exact failure this whole record format exists to prevent.
    #[arg(long = "serve-override", value_name = "KEY=VALUE")]
    pub serve_override: Vec<String>,
    /// Write a shareable result card beside the run.
    ///
    /// Takes a NAME (`my-run` -> `./my-run.svg`) or a PATH (`/tmp/x.svg`,
    /// `cards/run.svg`). A name is the common case and a path is the escape
    /// hatch; distinguishing them by "does it contain a separator or an
    /// extension" is guesswork the user should not have to reverse-engineer, so
    /// the rule is written in the help text and in `card_output_path`.
    ///
    /// The card carries the model, quantization, recipe and hardware beside the
    /// number, because this repository has already retracted a figure quoted
    /// without them.
    #[arg(long = "output-image", value_name = "NAME|PATH")]
    pub output_image: Option<String>,

    /// Attribution for the card: `author=Ada Lovelace,handle=@ada,website=ada.dev`.
    ///
    /// Comma-separated `key=value`. Unknown keys are accepted and ignored, so a
    /// future card field does not break an old command line. Requires
    /// `--output-image`; on its own it is a typo worth reporting rather than
    /// silently discarding, which `reject_orphan_image_args` does.
    #[arg(long = "output-image-args", value_name = "K=V,...")]
    pub output_image_args: Option<String>,
}

impl RunArgs {
    /// Refuse `--checkpoint` without `--pull-request-gate`.
    ///
    /// Outside the gate the serve config is whatever the operator started, so
    /// a variant selector would be a flag that visibly does nothing — the same
    /// confusion the `--model`/`--url` conflicts exist to remove. An `Err`
    /// here is a usage error, phrased like one.
    /// `--output-image-args` without `--output-image` renders nothing.
    ///
    /// Same shape and same reason as [`Self::reject_orphan_checkpoint`]: clap's
    /// `requires` cannot express it, because the target is an `Option` whose
    /// `None` still counts as "present" for that check.
    pub fn reject_orphan_image_args(&self) -> Result<(), String> {
        if self.output_image_args.is_some() && self.output_image.is_none() {
            return Err(
                "--output-image-args needs --output-image: there is no card to put them on"
                    .to_string(),
            );
        }
        if let Some(raw) = &self.output_image_args {
            atlas_plugin::gate::card::parse_args(raw)
                .map_err(|e| format!("--output-image-args: {e}"))?;
        }
        Ok(())
    }

    pub fn reject_orphan_checkpoint(&self) -> Result<(), String> {
        if self.checkpoint.is_some() && !self.pull_request_gate {
            return Err(
                "--checkpoint selects a model variant for a GATE run, which serves that \
                 variant's own recipe; without --pull-request-gate the endpoint is whatever \
                 you started, so pass --model to name what it serves instead."
                    .to_string(),
            );
        }
        Ok(())
    }
}

#[derive(clap::Args, Debug)]
pub struct HistoryArgs {
    /// Restrict to one benchmark id.
    #[arg(long)]
    pub id: Option<String>,
    /// Print the whole record for one run id.
    #[arg(long)]
    pub run: Option<String>,
    #[arg(long, default_value_t = 20)]
    pub limit: usize,
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    pub format: OutputFormat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum OutputFormat {
    Text,
    Json,
}

/// Split `KEY=VALUE` on the **first** `=` only.
///
/// An `IntList` value is `isls=128,512` and a `Text` value may legitimately
/// contain `=`, so splitting on every separator would corrupt both.
fn parse_kv(s: &str) -> Result<(String, String), String> {
    match s.split_once('=') {
        Some((k, v)) if !k.is_empty() => Ok((k.to_string(), v.to_string())),
        _ => Err(format!(
            "expected KEY=VALUE, got {s:?} — e.g. --param osl=8 or --param isls=128,512"
        )),
    }
}

#[cfg(test)]
#[path = "bench_args_tests.rs"]
mod tests;

/// `spark benchmark aggregate <group>`.
#[derive(clap::Args, Debug)]
pub struct AggregateArgs {
    /// The GROUP id, e.g. `bfcl-subset`. Not a member id — a shard has no
    /// aggregate of its own.
    pub id: String,
    /// The commit the records must cover. Defaults to HEAD.
    #[arg(long)]
    pub sha: Option<String>,
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    pub format: OutputFormat,
}
