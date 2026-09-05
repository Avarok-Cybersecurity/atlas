# `bench/campaign/` — the day-of cell kit

Everything needed to run **one cell** of the Hopper campaign end to end and come
away with an artifact that can be checked by a machine. The PRD, the recipe
research and the control-leg report live in
[Avarok-Cybersecurity/atlas#899](https://github.com/Avarok-Cybersecurity/atlas/issues/899).

A **cell** is one `(engine, model, SKU, workload, concurrency, spec, think)`
point. A campaign is a grid of them, and the whole point of this directory is
that a cell is reproducible from its artifact alone.

## What each tool is

| File | What it does |
|---|---|
| `run_cell.sh` | The driver. Preflight → serve → boot gate → coherency gate → latency pack → teardown → assemble → validate, for one cell. Orchestrates the existing tools; rewrites none of them. |
| `vllm_control.sh` | Renders (or runs) the **verified** vLLM recipe command for a `(model, SKU)`. Composes nothing. `--selftest`, `--list`. |
| `vllm_recipes.json` | The 29 rendered vLLM profiles, transcribed verbatim from the recipe evidence captured 2026-09-05. Data only. |
| `vllm_render.py` | JSON arithmetic behind `vllm_control.sh` (spec on/off, flag audit, docker argv). |
| `atlas_recipes.json` | The Atlas serve side: 17 entries from PRD §6 plus the repo's own recipe fixtures. Data only. |
| `atlas_render.py` | Renders `EXTRA_ARGS` for `scripts/start-node-ep.sh`. `--selftest`. |
| `cell_assemble.py` | Turns the stage outputs into the §10 artifact. Every field is copied or null. |
| `artifact.schema.json` | JSON Schema (draft 2020-12) for that artifact. |
| `validate_artifact.py` | Checks an artifact against it, stdlib only. `--selftest`. |
| `campaign_test.sh` | The whole suite: every selftest, both dry runs, every refusal, the lint gates. |
| `fixtures/` | One artifact that validates, six that must not, and stub stage outputs. |

Tools this directory **uses and does not own**: `scripts/start-node-ep.sh`,
`bench/hopper_ab/{time_to_ready.sh,coherency_gate.py,compare.py,workloads.json}`,
`bench/ladder38/harness_w55_conc_ladder.py`.

## One command per cell

```bash
# Atlas leg
bench/campaign/run_cell.sh --engine atlas --model nemotron-3-super-fp8 --sku h200 \
  --workload lat --concurrency 16 --spec off --think on \
  --out out/atlas.nemotron-3-super-fp8.h200.lat.c16 --yes

# vLLM control leg, same box, same shapes, inside 24 h
VLLM_IMAGE_DIGEST=sha256:<64 hex> \
bench/campaign/run_cell.sh --engine vllm --model nemotron-3-super-fp8 --sku h200 \
  --workload lat --concurrency 16 --spec off --think on \
  --out out/vllm.nemotron-3-super-fp8.h200.lat.c16 --yes \
  --paired-artifact out/atlas.nemotron-3-super-fp8.h200.lat.c16/artifact.json
```

Drop `--yes` and add `--dry-run` to see every command without launching
anything. **Nothing starts on a box without `--yes`** — with neither flag the
script exits 2.

## Environment

| Variable | Used by | Meaning |
|---|---|---|
| `VLLM_IMAGE_DIGEST` | vLLM legs | `sha256:<64 hex>`. **Required for a real run.** A vLLM cell's engine identity IS its image digest; a tag can be re-pointed between the two legs of an A/B. |
| `VLLM_IMAGE` | vLLM legs | Override the recipe's image tag. |
| `VLLM_PORT` / `ATLAS_PORT` | both | Client port (default 8000 / 8888). |
| `SPARK_BIN` | Atlas legs | Path to the `spark` binary (default `./target/release/spark`). |
| `IMAGE` | Atlas legs | Atlas container image; empty runs `SPARK_BIN` on the host. |
| `HF_CACHE` | both | Host HF cache to mount (default `~/.cache/huggingface`). |
| `DOCKER` | both | `docker` command (e.g. `sudo docker`). |

## The rule about CERTIFIED

> A cell is **CERTIFIED** only when the validator passes, **every gate reports a
> pass**, **and** the paired cell from the other engine exists, on the same box,
> within 24 hours.

"Every gate" means the recorded evidence, not the exit codes: the boot gate's
`passed`, all four coherency probes, and the three measurement gates the ladder
reports *while exiting 0* — the 80% vacuity floor, request errors, and the 10%
throughput spread. A null gate value is "this gate did not report", which is not
a pass.

"The paired cell" means this cell's other leg: the same model, SKU, workload,
concurrency, spec and think settings on the other engine, with both
`timing.started_utc` and `timing.finished_utc` (copied from the ladder's own
header) inside the 24-hour window. `paired_cell.within_24h` is that whole
check's verdict, and `validate_artifact.py` refuses a `CERTIFIED` artifact whose
own gate values and pairing record do not support it.

`run_cell.sh` cannot certify a cell on its own. A cell whose gates all pass but
whose pair is not yet recorded is written as `PARTIAL` with
`failing_stage: "pair"`; pass `--paired-artifact` once the other leg has run to
promote it. A cell that fails a gate is written as `NO-GO` (preflight, serve,
boot, coherency) or `PARTIAL` (ladder, validate) with the failing stage named —
**the artifact is always written**, because a NO-GO is a result the campaign
needs as much as a win.

## Exit codes

`0` every gate passed · `1` a gate failed, artifact written anyway · `2` usage,
or a refusal to start without `--yes` · `3` no rendered recipe for that
`(model, SKU)` · `4` `--spec on` against a recipe with no speculative profile ·
`5` (vLLM) a real run with no image digest · `6` (vLLM) a real run of a
multi-node profile · `9` a thinking mode excluded by the PRD.

`thinking_policy.json` records request-mode eligibility for each recipe model.
GLM-5.3 and GLM-5.3-Flash require `--think on`; Qwen3-Next Instruct requires
`--think off`. The driver refuses an excluded mode before creating resources,
and the Atlas renderer enforces the same policy. A missing recipe still exits
3, even when the requested thinking mode is excluded. New model keys need an
explicit policy entry; eligibility alone does not add a scored cell or recipe.

## Two limits worth knowing before you book a box

**`--think on` is serve-side only.** `harness_w55_conc_ladder.py` hardcodes
`chat_template_kwargs.enable_thinking=false` in every request body, so a
think-on cell sets an Atlas flag whose effect the client then suppresses.
`run_cell.sh` warns loudly and records it in the artifact's `notes`; the cell is
not evidence of thinking-enabled generation. Lifting this needs a change from
the ladder's owner.

**Percentiles are means of per-rep percentiles, not pooled.** The ladder keeps
one percentile per rep and discards the per-request samples, so pooled cell
percentiles are not reconstructible from its output. `metrics.percentile_method`
records which reduction was used, and the validator **rejects** a
ladder-sourced artifact that claims `pooled_requests`.

## Running the suite

```bash
bash bench/campaign/campaign_test.sh     # selftests, dry runs, refusals, lints
```

No GPU, no engine, no network: every case is a selftest, a `--dry-run`, or a
refusal.
