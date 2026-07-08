---
name: atlas-release
description: The Atlas build → verify → image → publish pipeline, plus the upstream-sync PR automation. Turns a merged commit into a serve-matrix-verified `avarok/atlas-gb10` image that users can pull. Use when cutting an image, closing the main→:latest staleness gap, wiring the release gate, or auto-syncing the fork and opening a MODEL.toml enablement PR. Codifies the REAL commands, containers, tags, and gates already in-tree — it does not invent a new release system.
argument-hint: <build | verify | image | publish | gate | sync-pr> [target]
allowed-tools: Bash, Read, Write, Grep, Glob, Agent, Edit
---

# /atlas-release — Atlas build → verify → ship pipeline

One entry point for turning **merged code into a pullable, verified image**. The
gap this closes: today `avarok/atlas-gb10:latest` is cut by hand on a GB10 build
host and drifts weeks behind `main` because nothing rebuilds/re-verifies/re-tags on merge.
This skill makes each stage explicit, gated, and reproducible so a new image can
ship in minutes with proof it works.

Four stages, each gates the next. **A stage's output is only valid if the prior
stage passed its written bar.**

| Stage | Question it answers | Produces | SSOT |
|-------|--------------------|----------|------|
| **1. Build** | Does the target(s) compile into one `spark` binary? | `spark` (PTX embedded) | `docker/gb10/Dockerfile`, `crates/atlas-kernels/build.rs` |
| **2. Verify** | Does every model×quant boot, stay coherent, and hold the signals? | pass/fail verdict + results JSON | `tests/run_all_models.py`, `references/verify-matrix.md` |
| **3. Image** | Package the *verified* binary, tag it to its exact source. | `avarok/atlas-gb10:<sha>` (+ `.tar` via `docker save`) | `docker/gb10/Dockerfile`, `references/pipeline.md` |
| **4. Publish** | Make it pullable — **human-gated push** (never automated). | pushed `:sha`/`:dev`/`:latest` + recorded shipped SHA | maintainer runs `docker push`; `references/pipeline.md` |

**Iron rule of this skill: no image is tagged shippable until the serve matrix
passes.** `run_all_models.py` only writes per-model JSON — so **`tests/gate_results.py`**
is the verdict: a pure scorer that turns those results into a real PASS/FAIL exit
code. Crucially it enforces **full coverage** — `run_all_models.py` writes a
`_manifest.json` of every model the run planned, and the gate fails any planned
model that produced no result (a boot crash writes no JSON and would otherwise be
an invisible false-green). `tests/test_gate_results.py` locks this behavior in.
`references/verify-matrix.md` specifies the bars and the run order.

**Two hard constraints, honored everywhere in this skill:**
- **Never advance the shippable surface without a human gate.** No `docker push`
  of any `avarok/atlas-gb10` tag and no moving of `:latest`/`:dev`/`:nightly` — the
  pipeline builds, verifies, tags, and `docker save`s, and *prepares* the final
  `docker push` for a maintainer to run (Stage 4 stops there). What ships to users
  is a human decision on a verified artifact. **Additive git pushes are fine:**
  `sync-pr` may push an enablement *branch* and open its PR — that is not a
  shippable-surface change.
- **No takeover PRs.** Sync-pr opens *enablement* PRs (MODEL.toml, new target) on
  the fork, or pushes a trivial fix to an existing branch — never a parallel
  reimplementation of someone's in-flight work. Contribute to the author's branch;
  reference and comment rather than fork their work (see `CONTRIBUTING.md`).

---

## Commands

### `/atlas-release build [<model-slug> | *]`  (default `*`)
Compile the `spark` binary. `*` bakes **all** kernel targets into one multi-model
binary (what `:latest` ships); a slug bakes one slim single-model binary.

```bash
# Multi-model sweep (the shippable binary). Host-native — the '38s incremental'.
ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL='*' ATLAS_TARGET_QUANT='*' \
  cargo build --release -p spark-server

# Or inside the pinned builder image (what the Docker image does):
docker build -f docker/gb10/Dockerfile -t atlas-gb10:build .
```

- `ATLAS_TARGET_MODEL` selects which `kernels/gb10/<model>/<quant>/` PTX + which
  `MODEL.toml` sampling/behavior presets are **embedded** in the binary. It does
  **not** bake weights — those always load at runtime from HF cache or
  `--model-from-path`. Runtime picks the right target from the model's
  `config.json` (`model_type` + `hidden_size`).
- **Footgun (PCND):** a bare `cargo build` with the env vars *unset* defaults to
  `qwen3-next-80b-a3b/nvfp4` **only** — not a sweep. The `*` is mandatory and
  easy to forget. Always pass all three explicitly.
- The "38s" figure is a **host cargo incremental** rebuild after Rust-only edits
  (build.rs `rerun-if-changed` gating + its in-build nvcc dedup cache). A *Docker*
  build is clean each time (~2–3 min; no BuildKit cache mount / sccache in-tree).
- Full command reference, container split, and toolchain pins: `references/pipeline.md`.

### `/atlas-release verify <image-or-url>`
Run the **serve matrix** — the gate. A binary/image is "verified" only if it
passes. Cheapest-sufficient path:

1. **Single-model quick gate** (one model, one box) before the full matrix:
   `scripts/dev/test_all_models.sh <hf-id> 8888` (boots the image, curls 6 fixed
   coherence probes). Or the richer `tests/single_gpu_suite.py`.
2. **Full serve matrix** across all model×quant rounds (head + optional worker):
   `ATLAS_IMAGE=<tag> python3 tests/run_all_models.py`.
3. **Coherence gate** (the `webserver_ok` idea, generalized): determinism (5×
   temp=0 identical), tool reliability ≥8/10, no CJK/Cyrillic/`<think>`/
   `<tool_call>` leak, streaming envelope valid — via `scripts/test_coherence.py`
   and the parked `bench/fp8_dgx2_drift/harness` agentic probe.
4. **Gate + record.** Compute a real pass/fail (Stage 2 in `references/verify-matrix.md`
   turns the JSON into an exit code) and write the verdict next to the image tag.

> Hand the numerical/behavioral faithfulness question to **`/coherence-parity
> gate-release`** — that skill owns the KL / argmax / drift thresholds. This skill
> owns the *operational* gate: does every model in the matrix boot and stay sane.

### `/atlas-release image <git-sha>`
Build the runtime image from the **verified** binary and tag it to its exact
source. **Refuses if Stage 2 didn't pass for this SHA.**

```bash
SHA=$(git rev-parse --short=7 HEAD)
docker build -f docker/gb10/Dockerfile \
  --build-arg ATLAS_GIT_SHA="$SHA" \
  -t avarok/atlas-gb10:"$SHA" .
docker save avarok/atlas-gb10:"$SHA" | zstd -o atlas-gb10-"$SHA".tar.zst   # offline distribution
```

- Tag with the **git SHA first** — it is the only identifier that answers "is
  `:latest` the merged code?" (Today three schemes drift: crate `0.1.0`, image
  `alpha-2.x`/`3.0.0`, distro `vX.Y.Z`. The SHA tag is the anchor; moving tags
  point at it.) See the version-provenance note in `references/pipeline.md`.
- `--build-arg ATLAS_GIT_SHA` now stamps `org.opencontainers.image.revision` into
  the image (the Dockerfile `ARG`+`LABEL` were added by this pass) — so
  `docker inspect` answers "which commit is `:latest`?" directly. See
  `references/pipeline.md` §Provenance.

### `/atlas-release publish <tag>`  (human-gated)
Move the moving tags onto a verified SHA image and **prepare** the push. This
command does **not** run `docker push` — it prints the exact commands for the
maintainer to run (or `docker save`s for offline transfer).

```bash
# Prepared for the maintainer to execute (all three moving tags advance together
# onto the same verified SHA — see references/pipeline.md §Publish):
docker tag avarok/atlas-gb10:$SHA avarok/atlas-gb10:nightly
docker tag avarok/atlas-gb10:$SHA avarok/atlas-gb10:dev
docker tag avarok/atlas-gb10:$SHA avarok/atlas-gb10:latest
docker push avarok/atlas-gb10:$SHA && \
  docker push avarok/atlas-gb10:nightly && \
  docker push avarok/atlas-gb10:dev && \
  docker push avarok/atlas-gb10:latest
```

After the maintainer pushes, record the shipped SHA (so "what's in `:latest`?"
is answerable) — a one-line append to `docs/releases/` or the image label.

### `/atlas-release gate`
Standalone: run only the release gate (serve matrix + coherence probe) against a
running image and return **PASS/FAIL** + the results table. This is the check
that must be green before `publish`. `references/verify-matrix.md` is its spec.

### `/atlas-release sync-pr`
The upstream→fork automation. Watch `monumental/main` (upstream) for merges →
fast-forward `origin` (Avarok fork) → build → verify → **if** the merge enables a
new model (new `kernels/gb10/<model>/MODEL.toml` or loader), open the enablement
PR. AI-attributed, CLA-clean, CI-green **before** submit. Full runbook incl. the
remotes, the CLA allowlist cleanup, and the CI-green preflight: `references/pr-and-ci.md`.

---

## Operating principles
- **SSOT:** serve config is owned by `atlas-recipes` recipe `defaults:` (not
  hand-copied into every doc); the model registry is `kernels/gb10/<model>/MODEL.toml`;
  the CI cap is `file-size-cap.yml` (≤500 LoC). Derive from these — never
  restate a value that lives in one of them.
- **PCND:** no stage runs on an implicit default. `ATLAS_TARGET_MODEL='*'` is
  explicit; the verify bar is written before the run; the image tag is the SHA.
  A bare `cargo build` (single-model default) is a bug, not a shortcut.
- **SBIO:** the gate's decision logic (pass/fail from the results table) is pure;
  the I/O is the results JSON + the container boot. Don't bake a host/port/tag
  into the gate — pass them in (`ATLAS_IMAGE`, `--base-url`).
- **CBD:** when the matrix fails, don't reship-and-pray — bisect. Which model,
  which round, which signal (coherence vs tool vs determinism vs OOM). The
  smoking gun is usually one model×quant, one signal. Hand numeric drift to
  `/coherence-parity diagnose-drift`.
- **Verify before publish, always.** A green CI (fmt/clippy/test) proves the code
  compiles GPU-free; it does **not** prove the image serves. Only the serve
  matrix does. CI-green ≠ shippable.

## Integration
- **`/coherence-parity`** owns numerical faithfulness (KL/argmax/drift + the
  `gate-release` thresholds). `/atlas-release verify` calls it for the numeric
  question and owns the operational "does it boot and stay sane" question.
- **`/hypercompile`** produces the kernels this pipeline ships. A hypercompile
  iteration isn't done until `/atlas-release verify` passes on the resulting image.
- **`atlas-recipes`** is where a winning serve config lands as a recipe; this
  pipeline ships the *engine* those recipes pin
  (`container: avarok/atlas-gb10:latest`). Keeping the image fresh is what keeps
  every `sparkrun run @atlas/*` honest.
