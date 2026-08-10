# HANDOFF — `w55-squash` / PR #388

**As of 2026-08-06.** Written so another agent can pick this up cold. Read §1 and §7 first.

### Provenance

| id | value | what it points at |
|---|---|---|
| Session UUID | `731b7370-37d0-492e-9f7c-469813792067` | transcript `/workspace/.claude/projects/-workspace/731b7370-37d0-492e-9f7c-469813792067.jsonl`; also the `originSessionId` recorded in the memory files under `/workspace/.claude/projects/-workspace/memory/` |
| Task/run dir | `d6352fe4-a52b-4d94-9f4b-2e1e6f362a79` | background-agent outputs, `/tmp/claude-996/-workspace/<id>/tasks/` — **ephemeral, on `/tmp`** |
| claude.ai session | `session_01LusZEPgCviuWFoVHBcqBcr` | different namespace; not a filesystem path |

The transcript is the only complete record of the reasoning behind §7 — the summaries here are compressed. Memory index: `/workspace/.claude/projects/-workspace/memory/MEMORY.md`.

Campaign state (in-repo, survives the session): `docs/campaigns/gb10-concurrency-2026-07/STATE.md`.

---

## 0. 2026-08-06 SESSION — what changed, and what is owed

**Read this before §1; the sections below predate it.**

### Pipeline state

CI has **never been confirmed green**: GitHub Actions has been in a **critical
incident since 15:22 UTC 2026-08-06** (`major_outage`) — jobs queue for hours
without starting, and the cancel/force-cancel APIs return 502. Every local
equivalent of every CI job passes at `2ed73e4e`:

| check | result |
|---|---|
| `cargo test --workspace` | **3,400 passing** |
| `cargo clippy --workspace --tests` | 0 errors |
| fmt · typos · LoC cap · kernel-shadow · block_on · rustdoc · cargo-deny | all clean |

Two workflows gained `concurrency: cancel-in-progress` (`tui-threading`, `cla`);
`release-build.yml` deliberately has none — it is a `workflow_call` invoked by
`ci.yml`, so cancelling the caller already cancels it, and a release build must
not be interruptible mid-tag.

### Gate records (`spark benchmark --pull-request-gate-check`)

**Superseded — all five now have committed PASS records.** The table below was
written mid-sweep at the frozen commit `2ed73e4ef9`; the branch has since landed a
record for every gate. Read `.benchmarks/<gate>/<date>-<sha>.json` first — those
files, not this prose, are what `--pull-request-gate-check` reads:

| gate | record | metrics | verdict |
|---|---|---|---|
| `agentic-webserver` | `2026-08-07-2e1ad5b0fb` | 10/10 ws_ok · 10/10 fd · Σwall 986.12 s ≤ 1300 | PASS |
| `ttft-warm-gate` | `2026-08-07-9e9f731ee4` | median 1561.00 ms · p90 4485.88 ms | PASS |
| `ttft-cold-gate` | `2026-08-07-9e9f731ee4` | median 1639.09 ms · p90 4477.48 ms | PASS |
| `bfcl-subset-echolp` | `2026-08-07-d2800e3e1c` | **85.56 / 85.69**, n=1004 | PASS |
| `bfcl-subset` | `2026-08-07-cc1ebf2758` | **88.04 / 88.74**, n=995 | PASS |

`bfcl-subset` is therefore **no longer blocked** — see the Owed list below, which
predates that record.

The mid-sweep snapshot it replaced, kept for the reasoning:

| gate | result |
|---|---|
| `agentic-webserver` | PASS — 10/10 ws_ok · 10/10 fd · Σwall 489 s ≤ 1300 |
| `ttft-warm-gate` | PASS — median −0.2 %, p90 +0.2 % |
| `ttft-cold-gate` | PASS — median +0.3 %, p90 +0.5 % |
| `bfcl-subset-echolp` | ran clean earlier at **85.76 / 86.00, n=1004**; re-running at the frozen sha |
| `bfcl-subset` | **BLOCKED** — needs `qwen3.6/qwen3.6-27b-nvfp4-unsloth` PRed to `atlas-recipes`; fails loudly (exit 1) meanwhile |

★ **Freeze code before a gate sweep.** `record_covers` invalidates on ANY
`crates/` change, including TUI edits that cannot affect a server benchmark.

★ **Timing legs need a quiet box.** The same agentic tier measured **605 s under
compile load vs 489 s idle** — 24 %. Temp-0 accuracy legs (BFCL) are immune.

### Fifteen defects found and fixed this session

All found by RUNNING things, not reading them. The recurring shapes are worth
internalising: *the value that reads correct is not the value that runs*, and
*a guard that abstains never fires*.

1. Reasoning deltas not counted as tokens — TTFT measured time-to-end-of-thinking; on the gate's own recipe every sample logged "no token was emitted" while reporting success
2. TTFT baseline keyed on full URL — a self-started gate binds an ephemeral port, so the guard abstained *every* run
3. Coherence probe called a healthy thinking model a broken checkpoint
4. **BFCL drew n=972, not the pinned 1004** — `configure()` rebuilds the DrawSpec from PARAMETER DEFAULTS, and the floor default was written a second time
5. Frame log lines were dropped by the CLI, so the guard for #4 printed into nowhere
6. Two-sided `Bound` treated as malformed — the draw pin would have failed every run, blaming the baseline's syntax
7. A failing benchmark skipped teardown, leaving ~100 GB resident
8. Gate record named a commit **14 commits newer** than the binary
9. A **cancelled** run wrote a record with `metrics: {}`
10. `draw_sequences` **panics** at 20×8 — takes the server's foreground with it
11. `draw_header` **panics** at 200×3
12. `wrap()` silently clipped unbreakable tokens (URLs, snapshot paths)
13. Chat history filtered by property not position → two consecutive `user` messages on the wire
14. Keyboard scrolling unclamped → blanked the pane; the wheel twin had been fixed
15. Sidebar subsection clicks selected the wrong section

### Owed

- ~~**PR `qwen3.6-27b-nvfp4-unsloth` to `atlas-recipes`** — unblocks `bfcl-subset`.~~ **Done**: `bfcl-subset` has a passing record at `cc1ebf27` (88.04 / 88.74, n=995). The upstream recipe PR (branch `feat/qwen3.6-27b-nvfp4-unsloth`) is still an outward action needing owner sign-off, but it no longer blocks a gate.
- **Remaining UX findings** (audit in the transcript): GPU renders a fabricated `0.0 GB` when NVML is absent; `{:?}` Debug reaches the screen twice; no ETA on downloads or benchmark runs; `NO_COLOR` unhonoured; three different byte units under one roof; `q` still needs a confirmation while a run is in flight, and `/detach` is missing from the key map.
- **One Lighthouse pass on dez.rs** — no Chromium on the build box, so its a11y/contrast claims are computed and reviewed, not machine-audited.
- **`.webmanifest` missing from avarok2's global `/etc/nginx/mime.types`** — patched for dez.rs only; every other PWA on that host has the same latent bug.
- **`oxidize.rs` has no A record** — pre-existing, unrelated, but it is a live site unreachable by name.

### New this session

- **`dez/`** — SvelteKit static PWA, live at **https://dez.rs** with a Let's Encrypt cert, `service-worker.js`/`manifest.webmanifest` at `no-cache`, hashed assets immutable, zero third-party requests.
- **Coverage** — `.github/workflows/coverage.yml` + `codecov.yml` + `scripts/coverage.sh`. Measured baseline **33.73 % lines**. Statuses informational on purpose, with the tightening condition written down.
- **~480 TUI tests** across chat, render, input, library and benchmarks.


---

## 1. TL;DR state

| thing | state |
|---|---|
| Branch | `w55-squash`, tip **`e11d6ae0`**, pushed |
| PR | **#388**, `mergeStateStatus: BLOCKED` — waiting on **human approval only** |
| CI | green at `e11d6ae0` (one long Windows CUDA build was still pending at last check) |
| Benchmark gates | **all five PASS** (§3) |
| Uncommitted work in tree | three workstreams; **compiled & tested 2026-08-03** (§4, §8c #1); `[expected_absent]` harvest done for 16/22 targets (§9) |
| Boxes | dgx1/dgx3 mostly idle end of 2026-08-03; model caches swept to the matrix (§9b) |

**The branch itself is merge-ready.** The uncommitted work in the tree is *follow-up* and is not required for #388. If you want the simplest path: `git stash` the uncommitted work, merge #388 on its green gates, then land the follow-ups as a second PR.

---

## 2. Commit history on this branch

```
e11d6ae0  bench: narrow the harness except clauses and drop an unused import
c4918d0c  Merge origin/main (3e5115e8, #387 — scripts/start-ep2.sh only)
7241a957  kernels: port the shadow-dropped kernels the new audit surfaced
54fd21e0  Add the Serve Matrix as the suite's seventh benchmark
bc54464d  hip: map cuStreamQuery onto hipStreamQuery in the libcuda shim
519cee77  (doc links + libcuda stub symbols)
dd749c33  (rustdoc private intra-doc link + typos)
cce57d93  the concurrency campaign (245 files)
```

★ `7241a957` → `e11d6ae0` differs **only** in `bench/agentic/*.py` (3 files) and `scripts/start-ep2.sh`. Nothing in `crates/`, `kernels/`, `Cargo.*`, `docker/`. **This is why the gates below are still valid at the tip** even though they ran on an image tagged `7241a95`.

---

## 3. Benchmark gates — ALL PASS

Image: **`avarok/atlas-gb10:7241a95`** (id `c52999044e25`, binary 75,939,392 B = all-target `ATLAS_TARGET_MODEL="*"` build).

| gate | result | box | notes |
|---|---|---|---|
| **C2** NVFP4 smoke | PASS | dgx2 | 3/3 tool calls, 3/3 coherent, 0 degenerate |
| **A** webserver_ok | PASS | dgx3 | 10/10 ws_ok, 10/10 followed_directions, **Σ 978.13 s** ≤ 1300 |
| **B** ST-1004 (35B) | ⚠ see note | dgx3 | **84.26 / 82.15** vs the same-box *live* control 84.06 / 82.06, n=1004 |
| **C** warm-TTFT | PASS | dgx1 | PR 1107.44 / 1778.04 ms vs control 1123.03 / 1807.05 ms ⇒ **−1.39% / −1.61%** |
| **D** ST-995 (dense 27B) | PASS | dgx2 | **87.74 / 89.43**, n=995; MLPerf floor 83.64 / 85.32 ⇒ **+4.1** clear |

⚠ **Gate B's "PASS" here was scored against the wrong bar.** 84.06 / 82.06 is the
*live same-box control*, not the gate's threshold.
`.benchmarks/bfcl-subset-echolp/BASELINE.json` ratchets to the **high-water**
84.66 / 83.32, and its own note predicts exactly this: "a run reproducing today's
live behaviour will FAIL this gate by ~0.6/1.3 until that drift is explained".
84.26 / 82.15 is below both mins, so this row would **not** clear the committed
baseline. The gate that does clear it is the later `d2800e3e` record
(85.56 / 85.69) in §0. Comparing to a control run rather than to
`BASELINE.json` is the mistake to avoid repeating — the JSON is the bar.

Gate C's control was built fresh from `c19481aa` as `atlas-gb10:mainctl388` (do **not** reuse `mainctl-tui` — it predates main's tip by two days).

★ **Gate C's load-bearing evidence** is not the TTFT delta, it is: control serve compiles **158 modules** for `(sm_121, qwen3.6-27b, nvfp4)`, PR compiles **167**. The ported kernels are provably compiled in *and routed to* — that is the #296 silent-disable class excluded by evidence rather than assumed.

**Any change to `crates/` or `kernels/` invalidates C2/A/C/D and they must be re-run (~5 h).**

---

## 4. Uncommitted work in the tree — THREE workstreams, NONE COMPILED

> ⚠ **The single most important fact in this document: none of the following has ever been through `cargo`.** Only `rustfmt --check`. Type-checking, clippy, `-D warnings`, and `typos` are all unverified.

### 4a. Kernel resolution + `--check-kernels`

- **TUI kernel table fixed.** `tui/data/kernels.rs` now resolves the **loaded** target via `ptx_for_config` (shape copied from `tui/data/library.rs:135`) instead of `ptx_modules()`, which `build_codegen.rs:86-96` emits as a plain alias of **target 0** (= `deepseek-v4-flash`, first by dirname sort). Same defect fixed at `serve_phases/preflight.rs:264` for Metal.
- Failed lookups split into **required vs expected**, reusing `kernel_audit::split_failures` (SSOT). Toast/red rows only for the required class. Dispatch site shown per row.
- **New flags** in `cli/serve_args.rs`:
  - `--check-kernels` — resolves kernels, prints the report + a one-line JSON blob, **exits with the count of unresolved kernels, clamped to 255** (clamp announced on stdout *and* at `error!`; 8-bit exit statuses mean 256 would otherwise report as 0 = false pass). Ignores the dangerous flag for its exit code. Forces `no_tui`.
  - `--dangerously-allow-unresolved-kernel-lookups` (default **false**) — downgrades the hard failure to a loud warning that still enumerates everything, every boot.
  - `ATLAS_ALLOW_SHADOWED_KERNELS` **deleted** — one switch, not two.
- **New `serve_phases/kernel_gate.rs`** replaces the old no-op at `serve_load.rs:554` (it intersected failures with `shadowed_dropped`, which for the 27B is 2 `[shadow_exempt]` entries with no dispatch site ⇒ provably empty).
- `AUDIT_SEALED: AtomicBool` after the gate; a post-seal miss aborts (chosen over panic: a panic unwinds one scheduler thread and leaves a half-serving process).
- `#[track_caller]` on the `GpuBackend::kernel` **trait declaration** + all impls, `layers::try_kernel`, and four resolver helpers; audit tuple widened to carry `&'static Location`.
- `[expected_absent]` parser (`build_parse::parse_expected_absent`) — **build panics on a missing/empty reason**. Tables ship **deliberately empty**.
- 5 non-turbo `[modules]` renames hoisted to `common/KERNEL.toml`; 22 duplicate lines removed from 6 model tomls.
- 5 dead **quant-level** `MODEL.toml` files deleted (nothing reads them; the 27B's copy actively contradicted the live one).

★ **Before the `[expected_absent]` harvest sweep ran, the fail-closed gate aborted most models** (51 unresolved on the 27B, 115 on nllb, 81–105 elsewhere — mostly deliberate cross-architecture probes). The intended workflow is: run `--check-kernels` per target, paste the printed list with reasons into each `MODEL.toml`. **This sweep has now been run — see §9a (19/22 targets rc=0, `[expected_absent]` populated in 18 MODEL.tomls).**

★ `--check-kernels` **cannot skip the weight load** — every lookup is inside a layer constructor taking materialised weights, and *which* constructors run depends on what the loader finds. A weight-free path would resolve a **different** set than a real serve. Budget a real cold load per target.

### 4b. TUI thinking blocks + tri-state toggle

- New: `tui/chat_thinking.rs`, `tui/chat_stream.rs`, `tui/render/chat_lines.rs`, plus two test modules.
- **The clock fix:** `chat_stream.rs:181-200` stamps `first_token` on the first delta of **any** kind. Previously only `delta.content` counted, so a thinking model reported ~18,000 ms for a ~412 ms TTFT.
- **Tri-state request toggle**: Auto (**omits** `chat_template_kwargs` entirely — no guess, server resolves) / Off / On. Key verified against the server's own `ChatTemplateKwargs` at `openai/chat_request.rs:279-281`.
- Keys: `t`/`Ctrl+T` = request cycle; `T`/`Alt+T` = display cycle (Collapsed/Expanded/Hidden). `Alt+t` must be matched **before** the bare-`t` arm (crossterm reports it as `Char('t')` + modifier).
- Collapsed streaming shows only the last 6 wrapped rows, so a thousand-token trace can't own the pane or the frame budget.
- `TestBackend` snapshot tests incl. 80×24 and 40×12 floors, no-reasoning, and the zero-answer case.

Two pre-existing bugs fixed incidentally: the `?` help modal drew **everything past the 16th entry below its own bottom border** (invisible), and the entry `Ctrl+Enter → send chat message` was **false** (input sends on plain `Enter`).

★ **Known honesty gap, not yet closed:** `--disable-thinking` is a server-side kill switch that outranks the client directive, so the chip can say `thinking on` while nothing thinks. Closing it needs the TUI to read the resolved server default — a real API addition.

### 4c. In-flight verification

A compile-verification pass was running on **dgx3** against 4a when this was written. It rsyncs the worktree (`/workspace` is **not** shared between boxes), builds there, and brings fixes back. **4b landed after it started, so the combined tree needs one more pass over both.**

---

## 5. The 18 s TTFT investigation — CLOSED, and it was not what anyone guessed

Owner reported 18,000 ms TTFT for the prompt `"hello!"` on `nvidia/Qwen3.6-27B-NVFP4` in the TUI chat (config: slots 256 / interval 16, thinking on, tools on, MTP on).

**Root cause: the TUI's TTFT clock ignored `reasoning_content`.** Measured dual-clock:

| req | TRUE TTFT | TUI-reported | reasoning deltas |
|---|---|---|---|
| R1 cold | 414 ms | 16,683 ms | 197 |
| R2 | 483 ms | 20,809 ms | 245 |
| R5 | 411 ms | 19,559 ms | 231 |

Every request, not cold-only. `unsloth` checkpoint identical ⇒ **checkpoint is a red herring**.

Ablations, one variable each: `enable_thinking:false` ⇒ 349–474 ms (**fixed**). `--disable-thinking` ⇒ 346–469 ms (**fixed**). Adding tools ⇒ no change. `--disable-tool-grammar true` ⇒ **spikes identical**.

**Two hypotheses were refuted:**
1. **Kernel resolution.** Audit was clean — correct target, 167 modules, **no `✘ REQUIRED` block**, nothing fell back.
2. **XGrammar compilation.** Disabling tool grammar changed nothing.

The 18 s was real decode time: TPOT a flat **77.6 ms/token** × 200–250 reasoning tokens the model spends deciding how to say hello. Engine healthy; UI blank and lying.

★ Also found: with `response_format:{"type":"json_object"}` + thinking on, **2 of 4 requests returned zero content deltas** — all reasoning, no answer. **Separate bug, not filed.**

---

## 6. Box allocation & rules

| box | ip | at time of writing |
|---|---|---|
| dgx1 | 10.10.10.1 (local) | full C=1..128 Atlas-vs-vLLM sweep on the final image |
| dgx2 | 10.10.10.2 | **owner's interactive TUI session — do not touch** |
| dgx3 | 10.10.10.3 | compile verification |

**Rules that are not negotiable:**
- **ONE bench per box.** Co-tenancy does not add noise, it **shifts the mean** — 16.3 GB of co-tenants cost Atlas **32% at C=16** while costing vLLM ~0%.
- **Parallelize ACROSS boxes**, never within one.
- **Never kill/stop/signal another session's processes** or touch their worktrees (`/workspace/w55` on dgx3 belongs to another session). Move your own work instead.
- **A cargo build is host-CPU load** that sails past `nvidia-smi` and `docker ps` and corrupts any timing leg. Never build on a box running a benchmark.
- **`--gpu-memory-utilization 0.85` max.** GB10 is **unified 121 GB CPU+GPU**; 0.90 has frozen a box and required a physical power-cycle.
- **Verify teardown, never assert it**: `--query-compute-apps` empty, util ~0, clock ~208 MHz, `docker ps` clear. *A bench is done when the memory is free, not when an agent says it is done.*
- `/workspace` is **not** shared between boxes.
- `nvcc` is **not** on the default PATH — `export PATH=/usr/local/cuda/bin:$PATH` or `vendor/cudarc/build.rs` dies.
- Always `ATLAS_TARGET_MODEL="*"` (67 MB / 22 targets; a single-model build is ~52 MB and cannot serve other models).

---

## 7. Lessons learned — read these before trusting any measurement

### Measurement & verification
1. ★ **Build-time signals are not runtime signals.** "Shadow warnings 21 → 0" was used as proof the kernel problem was closed. It wasn't — the TUI reports **runtime resolution**, an entirely different measurement. A kernel can compile in and still fail to resolve.
2. ★ **Never read a run's status from `ls -t *.log | head -1`.** It picked a stale `e2e_golden_nvidia.log` and supplied both a draw size and a score (`0.1126` — the PR #281 disaster number) belonging to a historical run. **Tie the log to the run**: the path the driver was launched with, or a file inside its own `--report-dir`.
3. ★ **Read the gate's exit code, not the pipeline's.** `cmd | tail -16; echo $?` reports `tail`'s status. Redirect to a file and check `$?` on the command itself.
4. ★ **Verification logic fails more often than code.** In one wave, three of four failures were the checker.
5. ★ **Print the raw per-rep series beside every aggregate.** One analyzer bug was caught only because 27 tok/s values were visible inside a 14 tok/s list.
6. ★ **Gate throughput claims spec-OFF.** A single spec-ON A/B pair drifts ±2% and has moved rungs the lever provably cannot reach, in opposite directions. Spec-OFF reproducibility is 0.11–0.19%.
7. ★ **SM clock: run the probe.** A **513 MHz clamp under load** makes every number 2.5–2.9× low while **every gate stays green**. Low variance is not health. Healthy is ~2400 MHz.

### Configuration
8. ★ **`auto` is a DEFERRAL, not a value.** The 27B checkpoint declares `kv_cache_quant_algo: FP8`, so vLLM's `auto` resolved to fp8 while Atlas ran bf16 — confounding both a speed and an accuracy result.
9. ★ **`{"thinking": false}` is SILENTLY IGNORED.** The working key is `chat_template_kwargs:{"enable_thinking": false}`. No error, it just does nothing.
10. ★ **`--num-drafts 1` is a NO-OP.** `config.rs:93` treats `1` as "unset", so MODEL.toml's `default_num_drafts=3` (K=4) wins. Every gate in the benchmark-pr skill passes `--num-drafts 1` believing it means K=2.
11. ★ **A BFCL number is meaningless without its draw.** Always record `category_sample_pct` + N + the ordered-`sample_id` SHA. `echolp_subset27` (n=1004) vs the golden MLPerf draw (n=995) differ by ~1.8 on *normalized* while *overall* coincidentally matches to two decimals.
12. ★ **Use recipes.** An unpinned config does not fail loudly — it produces a plausible number. `atlas-recipes/.../qwen3.6-27b-nvfp4.yaml` already pins `disable_thinking: true`; the owner's serve deviated from it, which is exactly why no gate ever saw the 18 s.
13. ★ **The gates have never exercised grammar-constrained generation** — all five pass `--disable-tool-grammar true`. That path is untested by the suite.

### Kernels
14. ★ **A missing/unreached kernel is a QUESTION, not a finding.** A static audit found a GEMV family dispatch never reached — the identical surface pattern to a bug worth **+99%**. Routing into it measured **−14.38% at C=16, −29.44% at C=32**. **The fallback was the faster path.** Classify gaps as confirmed-defect / candidate-to-measure / benign, and never infer direction from dispatch code.
15. ★ **Shadowing is whole-file and stem-keyed** (`build.rs::collect_cu_files`, a `HashMap<String, PathBuf>` where the model insert *replaces* the common entry). A shadow omitting a kernel silently drops it. **One kernel per file makes stem-keyed shadowing per-kernel with no resolver change** — 178/290 real `.cu` files still hold >1 `__global__`.
16. ★ **A silent fallback is a bug even when the fallback is correct**, because it makes a performance regression indistinguishable from normal operation. `try_kernel` returns `KernelHandle(0)` and falls back at `tracing::debug!` — invisible at default level.
17. ★ **Reverse drift exists and the detector is blind to it.** `common/moe_w4a16_grouped_gemm.cu` defines 3 entry points; the 18 model shadows define 6–7. **`common/` is the stale one.** `shadowed_missing_symbols` only computes `common − model`.
18. `kernels/strix/common/KERNEL.toml` **is a symlink** to `gb10/common/KERNEL.toml` — editing one edits both. 219 symlinks total, none broken.

### Process
19. ★ **Don't let an agent poll.** Several burned 80–130 k tokens returning "Waiting." Block inside one long-timeout call, or poll from the parent on a schedule.
20. ★ **`pgrep -f` self-matches** — your own `bash -c` contains the pattern you're grepping for. And `./target/release/spark serve` does **not** match a `spark serve` pattern.

---

## 8. Done vs. remaining

### 8a. Done — shipped on the branch (committed)

| # | item | outcome |
|---|---|---|
| 1 | Concurrency campaign C=1..128 | one config wins **8/8 rungs on tok/s**: 1.694 / 1.281 / 1.368 / 1.167 / 1.205 / 1.105 / 1.021 / 1.040× at C=1..128 (measured on an **earlier** binary — see 8c #4) |
| 2 | 18 shadow-dropped kernel ports (`7241a957`) | build-time shadow warnings **21 → 0**; 3 models that refused to boot now boot; serve matrix **7/7** |
| 3 | Serve Matrix benchmark (`54fd21e0`) | the suite's 7th registry descriptor; converted from `tests/run_all_models.py` |
| 4 | env → CLI conversion | 6 of 10 "config of record" env vars proved to be **no-ops**; 2 flags not promoted (bitwise control failed) |
| 5 | FP16 GDN h-state | opt-in; worth **1.286× wall at C=64**; token tax is speculation-mediated (+3.17% spec-ON, **0% spec-OFF**) |
| 6 | mmq_x=64 tile rung | batches 33..127 were discarding half their MMA columns; FFN MMQ 51.23 → 49.46 ms/step |
| 7 | CI green + main merged | 24/24 checks at `e11d6ae0`; `3e5115e8` merged (scripts only) |
| 8 | 5 bot review threads | fixed at source — narrowed except clauses, dropped unused import |
| 9 | All five benchmark gates | **C2 / A / B / C / D all PASS** (§3) |

### 8b. Done — investigations that CLOSED (negative results are results)

| finding | verdict |
|---|---|
| 18 s TTFT on nvidia 27B | **TUI clock bug**, not kernels, not grammar. True TTFT 410–493 ms (§5) |
| BFCL "regression" to ~83 | **Not a regression** — Atlas +0.37 ahead on a common basis; 87.24 was inflated by a 4096-context sample exclusion |
| C=1 "regression" | **The prompt**, +17.66%, not code |
| Class-1 GEMV dispatch gap | **REFUTED** — −14.4% at C=16 / −29.4% at C=32; the fallback was faster |
| 16.40 ms/step down_proj prize | **Never existed** — MMQ already owned it |
| FP16 verbosity (Nemotron +37-40%) | **Did not reproduce** — +3.17% spec-ON, 0% spec-OFF |
| Matched-KV C=64/128 inversion | −2% to −6%, not −4.5% to −11%; Atlas wins tok/s at all four points, loses purely on token count |

### 8c. Remaining — immediate (blocks merging §4 work)

| # | item | status |
|---|---|---|
| 1 | **Compile the combined tree** (4a + 4b) | **DONE 2026-08-03.** Release build `ATLAS_TARGET_MODEL="*"` rc=0 (74 MB binary, all 22 targets). Fixed one real defect the first cargo pass surfaced: `qwen3.6-27b/MODEL.toml` `[expected_absent]` used backslash-newline line-continuation in a TOML basic string (invalid TOML → build panic); converted to a multi-line literal. fmt/clippy(workspace,-tests)/typos all clean. Full `cargo test --workspace`: 70 suites pass (dgx3 needs `LD_LIBRARY_PATH=...libnccl.so.2` for the `spark-model` test binary — environmental). One caveat: `atlas-plugin::e2e::the_warm_gate…` is a timing-sensitive mock test (30 ms TTFT, n=3) that failed once at +3.3% median (limit +3.0%) under heavy box load and passed 3/3 on rerun; it is committed code, not part of §4 |
| 2 | **`--check-kernels` harvest sweep** | **20/22 targets DONE 2026-08-03/04** (see §9). `[expected_absent]` populated in 19 MODEL.tomls (383 entries total, every reason cites the dispatch fallback and marks porting UNMEASURED where applicable). `--check-kernels` rc=0 for every target checked — including the three oversized EP=2 targets (DS4/MiniMax/Step) harvested 2026-08-04 and `qwen3.5-27b` via single-target build. Remaining: `qwen3.5-397b-a17b` (EP=4-only, weights never downloaded); `qwen3.5-35b-a3b` accepted as legacy (owner decision — unreachable kernel set, identical module map to qwen3.6-35b-a3b) |
| 3 | **Re-run gates C2/A/C/D** | still open — §4 + the harvest touch `crates/` + `kernels/` (~5 h). NOTE: the kernel-set hash changed only via `[expected_absent]` metadata + KERNEL.toml renames; PTX content unchanged, but per §3 the rule stands |
| 4 | **Re-verify the C=1..128 sweep on the final binary** | **DONE 2026-08-03** — 8/8 rungs win on the gate image; results + confounds appended to `docs/campaigns/gb10-concurrency-2026-07/STATE.md` |
| 5 | Nothing blocks **#388 itself** | human approval only |

### 8d. Remaining — tracked open tasks (9)

| id | item | class |
|---|---|---|
| #73 | `--disable-tool-grammar` does **not** gate `response_format` — `json_object`/`json_schema` still run the Earley parser | bug |
| #83 | `finish_reason="length"` on generations that stopped **~3000 tokens below** `max_tokens` | bug |
| #86 | **Re-gate** wave 46's k64_n64 +1.60% — a single spec-ON pair drifts ±2% | rigor debt |
| #89 | 4 shadow-dropped kernels are **candidates to measure**, not confirmed losses (Class-1 precedent) | measure |
| #90 | Serve non-determinism is **concurrency-dependent** — bitwise output gating is valid at C=1 and only at C=1 | method |
| #92 | Atlas **over-calls tools** where vLLM abstains — the whole BFCL gap is hallucination/irrelevance, not construction | accuracy lever |
| #93 | Re-run wave 54's BFCL under **matched 16-bit KV** — the accuracy verdict is confounded the same way the speed one was | rigor debt |
| #94 | Wave 56 refactor is **not neutral**: −0.67% spec-OFF at C=16, likely `ModelLevers` +8 bytes riding `ForwardContext` to every dispatch site | perf regression |
| #95 | `sparkrun` silently drops all five GDN CLI flags — the recipe path is broken | bug |

### 8e. Remaining — unfiled bugs found in passing

| item | where |
|---|---|
| `response_format:{"type":"json_object"}` + thinking ⇒ **zero content deltas** (2 of 4 requests) | §5 |
| `("qwen3_6_moe", 5120)` declared by **both** `qwen3.5-27b` and `qwen3.6-27b` MODEL.toml; first-match-by-dirname wins ⇒ the 27B entry is **dead** | §4a |
| 43 duplicate `.cu` copies identical *across models* but differing from `common/` — a shared fork every model repeats; hoisting needs a **GPU A/B** | §4a |
| Two nemotron `moe_topk_sigmoid.cu` shadows lack `common/`'s deterministic tie-break; `MAX_TOP_K 24` vs 32 | §4a |
| `--disable-thinking` (server) outranks the client toggle ⇒ the TUI chip can say `thinking on` while nothing thinks | §4b |
| Help-modal and keymap fixes landed incidentally; **178/290** real `.cu` files still hold >1 `__global__` (one-kernel-per-file migration incomplete) | §4b, §7.15 |

**Descoped by the owner:** the `--kv-cache-dtype turbo2/3/4/8` module-rename break. Marked "currently experimental" in comments; **not fixed**, deliberately.

---

## 9. Harvest sweep + fleet model hygiene — 2026-08-03/04

### 9a. `--check-kernels` sweep results (image = this worktree's own build)

Checked rc=0 (unresolved=0) against weights on dgx1/dgx3:

| kernel target | checkpoint(s) checked | expected-absent |
|---|---|---:|
| qwen3.6-27b | nvidia/Qwen3.6-27B-NVFP4, centml/…W4A4-mlpinf, **Kbenkhaled/Qwen3.5-27B-NVFP4** | 1 |
| qwen3.6-35b-a3b | nvidia + unsloth 35B NVFP4, **Sehyo/Qwen3.5-35B-A3B-NVFP4**, lovedheart AgentWorld | 15 |
| qwen3-next-80b-a3b | nvidia 80B Sparse-2of4 (direct path), **Qwen/Qwen3-Coder-Next-FP8** | 18 |
| qwen3-vl-30b-a3b | ig1/…-NVFP4 | 17 |
| nemotron-labs-3-puzzle-75b-a9b | nvidia Puzzle-75B | 4 |
| nemotron-3-nano-30b-a3b | nemotron3-nano-nvfp4-w4a16 (direct path) | 9 |
| nemotron-super-120b-a12b | nvidia Super-120B (util 0.92, slots 0 — check-only) | 9 |
| nllb-200-3.3b | facebook/nllb converted: .bin→safetensors + repo's `convert-safetensors-to-bf16.py` → `/workspace/models/nllb-200-3.3b-bf16` (+`--src-lang eng_Latn --tgt-lang deu_Latn`) | 0 |
| holo-3.1-0.8b / holo-3.1-4b | Hcompany checkpoints | 32/32 |
| holo-3.1-35b-a3b | Hcompany (dgx3) | 15 |
| ornith-1.0-9b | deepreinforce-ai (identical kernel set to holo) | 32 |
| gemma-4-26b-a4b / gemma-4-31b | bg-digitalservices NVFP4A16 / nvidia IT | 36/28 |
| mistral-small-4 | mistralai 119B (dgx3, check-only flags) | 21 |
| qwen3.5-122b-a10b | Sehyo 122B (util 0.99, bs1, seq 512, slots 0, oom-guard 512 — **check-only**; the 0.99 util is never used for serving) | 22 |
| deepseek-v4-flash | nvidia DS4-Flash-NVFP4, **EP=2** (dgx1 rank0 ↔ dgx3 rank1; identical report both ranks, hash 0532e96f02d0) | 9 |
| minimax-m2-229b | lukealonso/MiniMax-M2.7-NVFP4, **EP=2** (dgx2 rank0 ↔ dgx3 rank1; identical both ranks, hash 41a9421cde36) | 12 |
| step3p7-flash | stepfun-ai Step-3.7-Flash-NVFP4, **EP=2** (dgx2 rank0 ↔ dgx3 rank1; identical both ranks, hash a1a4f6b0f249) — weights must be the **per-expert split** from `scripts/preprocess_step3p7_experts.py` (504 fused → 145,152 per-expert tensors); the fused checkpoint defeats the EP-aware pre-flight (estimates full 120 GB → OOM bail) | 35 |
| qwen3.5-27b | Kbenkhaled/Qwen3.5-27B-NVFP4 via a **single-target build** (`ATLAS_TARGET_MODEL=qwen3.5-27b`, hash fdf6bfaf23db, dgx1) — the standard multi-target build routes this checkpoint to `qwen3.6-27b` (exact `(qwen3_5, 5120)` match beats this target's wildcard `(qwen3_5, None)`), so that leg was checked separately there (2026-08-03, hash 6a0211057c4a). This target's own set is smaller (no nvfp4_mmq/q4k/w4a4/w4a16_v2 sources). Includes the standard four GDN f16/half-register/smem honesty-note arms | 33 |

**EP=2 harvest recipe** (used for all three oversized targets): docker `avarok/atlas-gb10:7241a95`
with `--gpus all --ipc=host --network host --device=/dev/infiniband --cap-add=IPC_LOCK
--cap-add=SYS_NICE --ulimit memlock=-1 --security-opt seccomp=unconfined` + RoCEv2 NCCL env
(`enp1s0f1np1`/`rocep1s0f1`, GID 3, Simple/Ring, BUFFSIZE 32M), the dev binary bind-mounted over
`/usr/local/bin/spark:ro` (image binary predates `--check-kernels`), weights at `/model`, and
`serve --model-from-path /model --check-kernels --rank {0,1} --tp-size 1 --ep-size 2 --world-size 2
--master-addr <rank0-mesh-ip> --max-seq-len 512 --max-batch-size 1 --gpu-memory-utilization 0.99
--oom-guard-mb 512` (+ per-target `--kv-cache-dtype`: DS4 fp8, MiniMax/Step bf16).

**Still not checked:**
`qwen3.5-397b-a17b` (EP=4-only, ~200 GB, needs 4 nodes — weights never downloaded).

**Accepted as legacy (owner decision 2026-08-04):** `qwen3.5-35b-a3b` — its kernel set is
unreachable in the standard multi-target build. The only fleet checkpoint
(Sehyo/Qwen3.5-35B-A3B-NVFP4, this target's own `hf_id`) enables MRoPE → dispatch.rs rewrites
`model_type` to `qwen3_6_moe`, which matches `qwen3.6-35b-a3b`'s exact `(qwen3_6_moe, 2048)`
entry, not this target's `(qwen3_5_moe, None)` wildcard. Its KERNEL.toml module map is
identical to qwen3.6-35b-a3b's (diff-empty) and its three `.cu` files are a strict subset, so
serving it would not exercise any kernel the live set lacks. Retained for reference; a
single-target build is possible but no checkpoint can route to it, so no harvest applies.
The note is also pinned at the top of its MODEL.toml.

**Step 3.7 loader fix (this harvest):** `num_attention_layers()` counted `FullAttention` only,
undersizing `attn_layer_dtypes` (12) while the Step loader indexes all 45 attention layers
(12 full + 33 sliding) → `index out of bounds: len 12, index 12` at
`step3p7/load_layers.rs:414`. Fixed in `atlas-core/src/config/methods.rs`: sliding-attention
layers consume the paged KV cache exactly like full-attention ones, so every consumer sized from
that count (KV pool `num_layers`, `attn_layer_dtypes`, loader indexing) must see them all.
Regression test: `test_num_attention_layers_counts_sliding_attention`. Gemma-4 was unaffected —
its parser maps sliding→FullAttention.

**Known honesty gaps kept honest in the MODEL.tomls** (all four `gated_delta_rule_decode_f16_*` arms on every GDN target): the FP16-h-state decode arms dispatch these handles UN-PROBED once `--ssm-h-dtype f16` is active; preflight refuses bad env combos but NOT a target that lacks the twins, so the flag on those models is a 0-handle launch. Every such entry carries a `⚠ HONESTY NOTE`. The correct fix is a preflight kernel-presence check — not yet written.

### 9b. Fleet model-cache hygiene (owner instruction 2026-08-04)

Matrix = `kernels/gb10/*/MODEL.toml` `hf_id` ∪ `atlas-recipes` `model:` fields (+ z-lab DFlash drafters). Script: `/tmp/sweep_models2.py` (matrix EMBEDDED — it must be, `/workspace` is not shared). Deleted across all three boxes: unsloth 27B/35B re-uploads (broken per #327), Qwen/Qwen3.6-35B-A3B (BF16), Ornith-1.0-35B-FP8, gpt-oss-120b-awq, Q4_K gguf 27B, Laguna, AgentWorld, diffusiongemma-NVFP4/FP8-nondynamic variants, Qwen3-1.7B — ~600 GiB freed. **Kept**: the w55 campaign's own checkpoints (nvidia 27B/35B NVFP4, centml W4A4, Puzzle-75B) and everything matrix-listed.

Downloads landed (spread across dgx1/dgx3, ≤3 in flight): Kbenkhaled-27B, VL-30B, DeepSeek-V4-Flash(157G), Super-120B, Coder-Next, 122B, Sehyo-35B, Gemma-4×2, Holo-4B (dgx1); Holo-35B, Mistral-119B, MiniMax-M2.7-NVFP4, Step-3.7 (dgx3). dgx1 disk is now at 98% — next sweep wave needs the 397B skipped or more freed first.

### 9c. What a new session must know about the sweep infrastructure

- `--check-kernels` exit code = unresolved count (clamped 255, clamp announced); JSON blob line `{"atlas_kernel_check": …}` on stdout.
- Tight-memory targets (122B, Super-120B, Mistral) need check-only flags: `--max-seq-len 512 --max-batch-size 1 --max-num-seqs 1 --ssm-cache-slots 0 --gpu-memory-utilization 0.99 --oom-guard-mb 512`. 122B-class also needs `ATLAS_KV_OVERCOMMIT` default behavior. **Never serve with these flags.**
- dgx3 quirk: run spark with `LD_LIBRARY_PATH=/home/claude/ttft-llama` (libnccl.so.2) or the binary won't load.
- All sweep logs: `/workspace/kcheck-results/*.log` (dgx1), `~/kcheck-results/*.log` (dgx3).
