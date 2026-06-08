# Atlas FP8 Drift — Statistical Harness

A small harness for measuring opencode drift modes on a running Atlas
container with statistical reliability (N≥10 per tier; bootstrap 95% CI;
Mann-Whitney U for tier comparison).

## Why

Single-probe comparisons (`n=1`) are unreliable on FP8 Qwen3.6 — three
back-to-back probes routinely produce three different drift modes. Any
A/B claim about an Atlas intervention must rest on a statistical
comparison or it is noise.

## Layout

```
harness/
  run_tier.sh         # N-probe loop against a live container
  score_run.py        # extracts structured metrics from one probe
  aggregate.py        # per-tier mean / std / p50 / p90 + bootstrap CI
  compare_tiers.py    # Mann-Whitney U, Bonferroni-adjusted
  runs/               # per-run JSON (one file per probe)
  reports/            # per-tier .md + .csv; cross-tier comparisons
```

## Workflow

```bash
# 1. Ensure atlas container is live & serving on localhost:8888.
sudo docker ps --filter name=atlas-qwen-final

# 2. Run N=10 probes against a tier. Each probe is the same canonical
#    rust-axum agentic prompt; targets are namespaced by tier+run idx.
#    Expect ~3-6 min per probe → ~30-60 min total per tier.
./run_tier.sh wsbcqr 10

# 3. Aggregate per-tier stats (writes reports/<tier>.{md,csv}).
python3 aggregate.py --tier wsbcqr

# 4. Compare two tiers (writes reports/compare_A_vs_B.md).
python3 compare_tiers.py --base tierA --candidate wsbcqr --alpha 0.05
```

## Metrics

Per run, score_run.py extracts:

| Category | Metric | Definition |
|---|---|---|
| **Outcome** | `files_written` | non-`.git` files in target dir |
| **Outcome** | `cargo_toml_valid` | Cargo.toml parses as TOML + has `[package].name` + `version` |
| **Outcome** | `wall_time_s` | opencode wall time (seconds) |
| **Tool use** | `tool_calls_total` | all tool_use events |
| **Drift #1** | `drift_path_outside_target` | write filePath not under expected target dir |
| **Drift #2** | `drift_path_literal_space` | write filePath contains a literal space (`axu m`) |
| **Drift #7** | `drift_lean_prefix` | write content starts with literal `lean` |
| **Drift #9** | `drift_empty_path` | write filePath empty after trim |
| **Drift #11** | `drift_toml_newlines_collapsed` | section header on same line as a key=value |
| **Drift #5** | `drift_xml_attr_leak` | content has `filePath="…"` or `content="…"` (XML-attr style) |
| **Drift #X** | `drift_bash_as_content` | content starts with a shell verb (`cargo `, `ls `, `rm `, etc) |
| **Atlas** | `atlas_ws1_mask_fires` | diagnostic INFO log `ws1/am1 mask active` |
| **Atlas** | `atlas_b1_drift_fires` | B1 margin-ratio drift gauge summary lines |
| **Atlas** | `atlas_tier5c_retries` | Tier 5c retry success count |
| **Atlas** | `atlas_a2_fuzzy_fires` | A2 fuzzy_repair rescue count |
| **Atlas** | `atlas_tool_call_lines` | total Atlas-side tool_call log lines |

## Statistical interpretation

`aggregate.py` reports:
- **mean ± std** — central tendency + spread.
- **p50 / p90** — order statistics (robust to outliers).
- **95% CI on the mean** — percentile bootstrap (10k resamples).
- **non-zero runs** — for rare-event metrics, the count of runs where
  the metric was non-zero is more informative than the mean.

`compare_tiers.py` reports:
- **median(A), median(B), Δ** — point estimate of effect.
- **p (Mann-Whitney U)** — two-sided, no normality assumption.
- **p_bonf** — Bonferroni-adjusted across all ~18 metrics tested.

A row is significant iff `p_bonf < alpha`.

## Caveats

- One probe ≈ 3-6 min wall time. Atlas serves single agentic requests
  sequentially (max_batch_size=4 means concurrent prefill is supported,
  but opencode sessions step through tool calls serially). Wall time
  dominated by atlas decoding, not opencode logic.
- The prompt is one specific opencode task. Drift profile may differ
  for other tasks — broaden the prompt set before generalizing claims.
- `cargo_toml_valid` is syntactic only; not a guarantee the project
  compiles. Adding `cargo build` would 3-5× the per-probe cost.
- The `bash_as_content` heuristic uses a fixed verb list. Will miss
  rarer shells / Python content drift; treat as a lower bound.
