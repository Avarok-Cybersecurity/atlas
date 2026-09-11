# references/verify-matrix.md — the serve-matrix gate

What "the serve matrix passes" actually means, grounded in the real scripts, plus
how to make it a **binary pass/fail** (today it isn't — see §Gate).

## The orchestrator: `tests/run_all_models.py`
Enumerates model×quant as **hand-authored `TestSpec` rounds** (not a cross-product):
9 single-GPU rounds (each a head/worker pair run in parallel), `EP2_ROUNDS` (122B
±MTP, 80B-MTP, nemotron-super-120B), `TPEP_ROUNDS` (minimax-m2.7 tp2/ep2),
`EP4_ROUNDS` (397B, skipped by default). Per spec it:

```bash
ATLAS_IMAGE=<tag> ATLAS_HEAD_IP=127.0.0.1 python3 tests/run_all_models.py 2>&1 | tee /tmp/atlas-full-run.log
```

1. **Boot:** `sudo docker run -d --name atlas-test-<label> --gpus all --ipc=host
   -p PORT:PORT -v <hf>:/root/.cache/huggingface <IMAGE> serve <model>
   --scheduling-policy slai --max-seq-len 32768 --kv-cache-dtype <fp8|nvfp4>
   [--speculative --mtp-quantization <q>]`.
2. **Readiness:** polls `docker logs` every 10s for the substring `Listening on`
   (600s timeout). ⚠️ It only fails fast if **both** `Error:` *and* `ERROR` appear
   — a panic with a different string hangs to timeout.
3. **Warmup:** one throwaway `curl` (HTTP code captured but **ignored** — non-fatal).
4. **Suite:** `python3 tests/single_gpu_suite.py --base-url <url> --model <m>
   --output <label>.json [--skip-longctx]`.
5. Aggregates all JSON into `tests/all_models_results/all_results.json`.
   **Individual failures do not abort the run** (by design).

## What the suite checks (`tests/single_gpu_suite.py`)
- **Health** = `GET /v1/models` returns parseable JSON (`Server OK`). On any
  exception → `[FATAL] Server not reachable`, `exit(1)`. ⚠️ This is the real
  "health" check — **not** a `GET /health` nor an explicit 200 assert. (Among the
  serve-matrix suite scripts, only `scripts/dev/coherence_test.py` hits `/health`;
  several benchmark scripts do too, but the matrix itself uses `/v1/models`.)
- **Coherence** (`run_coherence_tests`): factual "capital of Japan"→`tokyo`
  (temp 0), reasoning "120/2"→`60 km/h` (temp 0, LaTeX-regex fallback), creative
  haiku → repetition-loop check. PASS/FAIL per prompt, repetition-loop = FAIL.
- **Smoke / codegen** (`run_fibonacci_test`): extract the ```python block, exec in
  a subprocess, assert first 10 Fibonacci == `[0,1,1,2,3,5,8,13,21,34]`;
  plain-text fallback. ⚠️ **There is no literal "2+2=4" in the serve matrix** —
  the maintainer's "2+2=4 smoke" actually lives in `scripts/dev/coherence_test.py`
  and in `bench/fp8_dgx2_drift/harness/run_tier.sh` (which `exit 4`s if the
  response lacks `4`). Use the fibonacci exec as the matrix's codegen smoke.
- **Tool calls** (`run_tool_call_tests`): Weather/Search → assert
  `message.tool_calls` present and arguments parse as JSON. Known-gap parsers →
  `N/A` (never hard-fail). ⚠️ Even on supported models a miss is `WARN`, not FAIL.
- **TPS** (`run_tps_benchmark`): 50/150/300/500 tok, pass if tps>0 and not error.
- **Long context** (`run_long_context_tests`): needle `PURPLE-DOLPHIN-42` at
  4k/8k/16k. ⚠️ Returns **PASS even if the needle is missed** — it's an
  infra-survival check, not a retrieval-accuracy gate.

## The four quality signals (what "verified" should mean to users)
The serve matrix is the *operational* gate. The **quality** claim behind an image
rests on four signals; know where each lives and which are in-repo:

| Signal | Script (in-repo unless noted) | Bar |
|--------|------------------------------|-----|
| **Coherence** | `scripts/test_coherence.py` (14 sections, `exit 1` on any FAIL: no CJK/Cyrillic, no `<think>`/tool-tag leak, determinism 5×`7*8` byte-identical, tool reliability ≥8/10) · `scripts/dev/coherence_test.py` (10 tests incl. `/health`, 2+2, Paris, temp-diversity, streaming envelope `[DONE]`+`time_to_first_token_ms`, Korean+emoji no U+FFFD) | both `exit 0` |
| **Agentic** | `scripts/dev/agentic_test.py` (real write/read/run loop → `python3 -m unittest` == ok) · `scripts/test_multiturn_agentic.py` (5-turn w/ opencode+claude-code system prompts from `scripts/fixtures/`) | `exit 0` |
| **KL-divergence** | `bench/fp8_dgx2_drift/c1_final_logit_overlap.py` (KL nats + top-1 argmax agree + top-K Jaccard) · `tests/metal_kv_kld_compare.py` | offline, needs pre-captured logit dumps — owned by **`/coherence-parity`** |
| **ST-subset accuracy** (BFCL ST-995) | ⚠️ **NOT in this repo.** External gorilla/`spark-arena-v2` harness. Only in-repo trace is a comment in `kernels/gb10/qwen3.6-27b/MODEL.toml`. | tiny BFCL subset for config A/Bs; full ST-995 only for milestones |

**[TODO / convergence]** ST-subset and the "core-8 trading eval" the maintainer
named have **no artifact in this repo** (grep for `core-8`/`trading`/
`st-995` → registry comment only). To make "verified" reproducible by the
community, either check a thin BFCL/core-8 driver into `tests/` or document the
external harness + its expected numbers in the deployment guide's methodology
section. Right now no one can reproduce the accuracy claim from the repo alone.

## The `webserver_ok` gate (the parked coherence gate)
`webserver_ok` is **not** part of `run_all_models.py`. It's the outcome metric of
the FP8 agentic-drift harness (`bench/fp8_dgx2_drift/harness/`):
`score_run.py::webserver_test()` has an agent (opencode / `--claude-code`) write a
Rust/Axum project, then `cargo build --release`, `cargo run` on an ephemeral port,
poll `/ping` for ≤15s, and sets `webserver_ok=true` iff the body contains `pong`.
It **gates** via `run_tier.sh --bail` (`exit 5` on first `cargo_valid!=true` OR
`webserver_ok!=true`) and `aggregate.py` (exit = `cargo_failures + webserver_failures`).

It is the strongest *agentic* coherence gate the codebase actually enforces, but
it's coupled to one model (`Qwen/Qwen3.6-35B-A3B-FP8`), the agent binaries, and a
live container. **To automate it as the release gate**, generalize it into a
per-image probe: build the standard project once, `curl /ping`→`pong`, over the
flagship model in the image. Wire its exit code into the Stage-2 verdict below.

## Gate — "verified" as a real exit code, over the *whole* matrix
`run_all_models.py` **never `exit`s non-zero on a test failure** — it only writes
per-model JSON to `tests/all_models_results/`. That's why "verified iff the matrix
passes" used to be a human reading the table. **`tests/gate_results.py`** closes
it: a pure SBIO scorer that reads those JSON files, applies explicit PCND bars,
prints a per-model PASS/FAIL, and **exits 0 (ship) / 1 (block)**.

**Coverage is enforced — the gate serves *all*, not just the survivors.** A model
that crashes at boot writes **no** `<label>.json` (the suite `exit`s before writing
when the server is unreachable). A glob-only gate would never see it and would
score the survivors green — a false ship. So `run_all_models.py` writes
`all_models_results/_manifest.json` (the roster it *planned* to cover, from its
`planned_specs()` — SSOT), and the gate fails any planned model with no passing
result. Missing manifest ⇒ hard FAIL unless you pass `--allow-missing-manifest`
(explicit, loud, coverage-unenforced). `tests/test_gate_results.py` locks the
"planned-but-absent = FAIL" property in.

The per-model bars (constants at the top of the script, chosen up front):
- **present** — the model is in the manifest and has a matching result with
  nonempty required probe groups. The orchestrator clears planned prior results
  before boot attempts and rejects output from failed suite subprocesses. Archive
  completed artifacts before another run; use one orchestrator per checkout.
- **coherence** ≥ 2 of 3 probes PASS (tolerates the one temp>0 creative probe).
- **codegen** — the fibonacci exec smoke PASS.
- **tools** — at least one real PASS, **or** all `N/A` (known-gap parser). An
  all-WARN/FAIL result on a supported parser fails.
- **tps** — liveness (mean > 0) **plus** a regression bar: if a blessed
  `tests/baselines/<label>.json` exists, a run > `TPS_TOLERANCE` (10%) below it
  fails; without a baseline it's liveness-only (loud note) unless
  `--require-baselines`. Refresh deliberately: `gate_results.py --update-baselines`
  writes `{"tps": avg}` per label, then review + commit the diff. (Baseline-snapshot
  approach proposed by @Sujimoshi in #253; folded into the gate here to keep one
  roster/SSOT rather than a parallel suite.)

The detailed evidence and remaining provenance limits are in
[`docs/testing/serve-matrix-rst-audit-2026-09-04.md`](../../../../docs/testing/serve-matrix-rst-audit-2026-09-04.md).

Validated against fixtures (full-pass / planned-but-missing / below-bar /
known-gap-all-N/A / tps-regression / no-baseline / require-baselines) in
`tests/test_gate_results.py` and `tests/test_gate_results_orchestrator.py` (host-only). Run order for `/atlas-release gate`:
```bash
ATLAS_IMAGE=<tag> python3 tests/run_all_models.py               # boots + probes matrix -> *.json + _manifest.json
python3 scripts/test_coherence.py --url http://localhost:8888   # leak/determinism/tools, exit 1 on fail
python3 tests/gate_results.py                                   # coverage + regression verdict, exit 0/1
```
All three green = shippable. Record the verdict + the results table next to the
image tag (`docs/releases/<sha>.md`). **A gate with an unrecorded number is not a
gate.** (Still parked: folding the `webserver_ok` agentic probe and a per-model
determinism/leak check directly into `gate_results.py` so one command is the whole
gate — today `test_coherence.py` covers leak/determinism separately.)
