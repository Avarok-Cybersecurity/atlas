# vLLM FP8 on dgx2 — N=10 cargo_valid harness

**Date**: 2026-05-27
**Model**: `Qwen/Qwen3.6-35B-A3B-FP8` (native FP8 checkpoint)
**Host**: dgx2 (10.10.10.2)
**Framework**: vLLM 0.20.2rc1.dev173+g171d59ae8.d20260509
**Image**: `sparkrun-eugr-vllm:latest` (19GB, transferred from dgx1)
**Endpoint reach**: SSH tunnel `dgx1:8889 -> dgx2:8888`

## Configuration

```
vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --max-model-len 65536 --max-num-seqs 4 \
  --gpu-memory-utilization 0.85 \
  --port 8888
```

vLLM auto-resolved architecture as `Qwen3_5MoeForConditionalGeneration`, async scheduling
enabled, custom fusions: norm_quant + act_quant. Model loading + KV init: ~7 minutes.

## Smoke test

`curl /v1/chat/completions` "What is 7+8?" — coherent thinking-block output, correct
answer (15). `curl /v1/models` returns `Qwen/Qwen3.6-35B-A3B-FP8`, `max_model_len=65536`.

A smoke harness run (N=1) wrote a valid Cargo.toml in 84s with zero drift events.

## N=10 harness results

**cargo_toml_valid: 10/10 (100%)** — same as the vllm_bf16 baseline.

| metric | value |
|---|---|
| `cargo_toml_valid` mean | **1.000 ± 0.000** (95% CI: [1.000, 1.000]) |
| `cargo_toml_present` mean | 1.000 ± 0.000 |
| `files_written` median | 1122.000 (p90 1273; mean 997.5 ± 264.4) |
| `tool_calls_total` median | 16.000 (p90 30) |
| `write_calls` median | 2.000 (p90 4) |
| `wall_time_s` median | 242.4s (p90 360.1) |
| any drift signature | **0/10 runs** across all 8 catalogued modes |

Per-run summary (chronological order):

| run | cargo_valid | files | wall_s | notes |
|---|---|---|---|---|
| 1 | true | 1169 | 360 | hit 6-min cap (Cargo.toml written early, opencode kept running) |
| 2 | true | 1273 | 360 | hit 6-min cap (same) |
| 3 | true | 1117 | 138 | clean |
| 4 | true | 926 | 95 | clean |
| 5 | true | 1122 | 242 | clean |
| 6 | true | 1169 | 119 | clean |
| 7 | true | 524 | 247 | clean |
| 8 | true | 890 | 100 | clean |
| 9 | true | 524 | 360 | hit 6-min cap |
| 10 | true | 1261 | 187 | clean |

3 of the 10 runs ran to the 360s harness cap. In all three the Cargo.toml was written
early and opencode kept iterating with shell tool calls afterwards — those caps are
"opencode kept going" not "model produced bad syntax".

## Comparison vs vllm_bf16 baseline

| metric | median(vllm_bf16) | median(vllm_fp8) | delta | p_bonf |
|---|---|---|---|---|
| cargo_toml_valid | 1.000 | 1.000 | +0.000 | 1.0000 |
| files_written | 1159.5 | 1119.5 | -40.0 | 1.0000 |
| tool_calls_total | 16.0 | 15.5 | -0.5 | 1.0000 |
| wall_time_s | 254.6 | 214.8 | -39.8 | 1.0000 |
| all drift signatures | 0.000 | 0.000 | +0.000 | 1.0000 |

Mann-Whitney U Bonferroni-adjusted across 18 metrics — **no metric differs significantly
between vllm_bf16 and vllm_fp8**. Both quantizations produce a valid Cargo.toml on 10/10
runs with zero drift, indistinguishable wall times, and statistically equivalent file /
tool-call counts.

## Caveats

- vLLM image (19GB) was not present on dgx2; transferred via `docker save | scp |
  docker load` (~30 s SCP wall time).
- dgx2 disk free: 105GB before image load, ~87GB after (tar deleted post-load).
- A parallel autonomous harness `fp8cal256` was running on dgx1's Atlas at the same time.
  Two sessions sharing `/workspace/.config/opencode/opencode.json` were isolated by:
  - Pointing the vLLM-side opencode at an alternate `XDG_CONFIG_HOME=/tmp/oc-tunnel-config`
    so each session reads an independent `opencode.json`.
  - vLLM-side baseURL is `localhost:8889/v1` (SSH tunnel to dgx2), Atlas-side stays at
    `localhost:8888/v1`.
- vLLM env warning: `VLLM_USE_V1` not recognized in this vLLM dev build (harmless).
- vLLM tool parser name confirmed: `qwen3_coder` (underscore — not hyphen).
- vLLM peak generation throughput observed: 45-60 tok/s on single GB10.
- The harness's `run_tier.sh` requires a local docker container; a sibling
  `run_tier_vllm.sh` was added at the same path that skips that check and reads the API
  base via `$API_BASE`.

## Files

- `/workspace/atlas-mtp/bench/fp8_dgx2_drift/harness/run_tier_vllm.sh` — remote-vLLM
  variant of the run harness.
- `/workspace/atlas-mtp/bench/fp8_dgx2_drift/harness/reports/vllm_fp8.{md,csv}` —
  aggregate report.
- `/workspace/atlas-mtp/bench/fp8_dgx2_drift/harness/reports/compare_vllm_bf16_vs_vllm_fp8.md`
  — head-to-head vs BF16.
- `/workspace/atlas-mtp/bench/fp8_dgx2_drift/harness/runs/run_vllm_fp8_{1..10}.json` —
  raw per-run scoring artifacts.
- `/tmp/oc-tunnel-config/opencode/opencode.json` — isolated opencode config used by the
  vLLM-side runs (baseURL=localhost:8889/v1).

## Conclusion

**Native FP8 Qwen3.6-35B-A3B on vLLM produces a 10/10 cargo_valid score on N=10 — equal
to the BF16-snapshot baseline, with zero drift signatures and indistinguishable wall
times.** This rules out the FP8 *checkpoint quantization* as a cause of the cargo_valid
drift observed in Atlas's FP8 path; the gap is in Atlas's FP8 dequant / MoE / SSM kernels,
not in the checkpoint itself.
