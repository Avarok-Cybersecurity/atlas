# D-Cut v2: cost-optimal verify budget and online acceptance calibration

**Headline: the batched MTP verify keeps a fixed fraction of its draft rows; v2 keeps the number that pays.**
D-Cut v1 (`scheduler/mtp_dcut.rs`) ranks every optional draft position by survival and keeps a fixed 0.75 of them,
a ratio picked from a four-point sweep at C=8. v2 (row `mtp_dcut_cost_model`, off) replaces the ratio by the
budget that maximizes expected tokens per second under a cost table learned while serving, with survival taken
from a calibrated acceptance model. It only chooses how many drafts to verify, so accept and reject stay exact.

## Upstream

- Project: vLLM, <https://github.com/vllm-project/vllm>, commit `8a36f08`.
- License: Apache License, Version 2.0 (`LICENSES/Apache-2.0.txt`). Copyright contributors to the vLLM project.
  The repository carries no NOTICE file.
- Files translated (no source copied; Python/Triton rewritten as scalar Rust):

| Upstream | Lines | Atlas |
|---|---|---|
| `vllm/v1/worker/gpu/spec_decode/adaptive_verification.py`, `build_cost_tables_from_curves` | 163-213 | `scheduler/mtp_dcut_cost.rs`, `CostTable::curve`: a step-or-linear cost over row counts, made monotone by a running max |
| `adaptive_verification.py`, `get_num_tokens` | 399-469 | `scheduler/mtp_dcut_cost.rs`, `optimal_budget` and `CostModel::decide`: expected accepted tokens for every budget divided by the draft-plus-verify cost, argmax at line 461 |
| `vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py`, `_accumulate_kernel`, `_refit_kernel`, `OnlineAcceptanceEstimator` | 20-167, 342-420 | `scheduler/mtp_accept_estimator.rs`: shared-slope, per-position-intercept logistic model, arrowhead Newton-IRLS, damping `n/(n+50)`, ridge 1e-3, refit every 100 steps |

## Changes for Atlas

- The cost table is learned online (EWMA of the wall time of a batched MTP round per total verify row count,
  `MIN_SAMPLES = 8` per entry) instead of profiled; below the lowest calibrated row count the lowest cost is used,
  above the highest there is no estimate and D-Cut v1's ratio runs instead, with a periodic full-width probe that
  fills the top of the table.
- Every sequence keeps at least one draft (the v1 envelope: `rows_i` stays in 2..=4), the width cap (8 sequences)
  and the verify row budget stay.
- The feature is read from the drafter's top-1 LOG-probability (`pending_draft_conf`, D-Cut's ranking key) instead
  of recomputed from draft logits; positions are capped at 16.
- The estimator never reads the drafted token, only the drafter's distribution at that position (as upstream), so
  trimming a draft never conditions on the token being trimmed.
- Statistics are host `f64` accumulators; no Triton, no GPU buffers.

## Files carrying the translation

`crates/spark-server/src/scheduler/mtp_dcut_cost.rs`, `crates/spark-server/src/scheduler/mtp_accept_estimator.rs`
(and their tests). Each starts with the vLLM attribution header. NOTICE and CITATIONS carry the matching entries.
