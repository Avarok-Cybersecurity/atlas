<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Campaign: Apple Metal Kernel Mapping and Pipeline support

> Owner brief (2026-09-17): map EVERY kernel 1:1 onto Apple Metal, with any
> exception to 1:1 written as a comment saying why; reach byte-identical parity
> with MLX and ideally beat it; 100% Metal support; no `--features=metal` —
> target_os gating so a plain `cargo build` needs no flags; a Metal benchmark
> set that becomes required when Metal kernels are edited, added gradually and
> LEFT OPEN for now. One PR.

## Measured on the target box (apple-48gb-metal.local, 2026-09-17)

Facts taken from the machine, not from the tree — several contradict what the
tree currently claims:

| fact | measured | the tree says |
|---|---|---|
| chip | **Apple M4 Pro**, `applegpu_g16s` | `kernels/metal/HARDWARE.toml`: an M2 Pro's numbers |
| unified memory | **48 GiB** (`memory_size = 51539607552`) | `memory_gb = 16` — **wrong** |
| max recommended working set | **37.44 GiB** (`40200896512`) | not recorded |
| max single buffer | **28.08 GiB** (`30150672384`) | not recorded — and it is a harder bound than the working set for one monolithic weight tensor |
| Metal compiler | `Apple metal version 32023.921`, target `air64-apple-darwin27.0.0` | the advisory job still says the box has no Metal compiler |
| `-ffp-contract=off` | **accepted** | assumed unavailable |
| `-fno-fast-math` | **accepted** | — |
| `simdgroup_matrix<bfloat,8,8>` + `simdgroup_multiply_accumulate` | **compiles** | two `.metal` headers promise it as future work |
| `simd_sum` / `simd_max` / `simd_shuffle_xor` | **compile** | — |
| MLX | **0.29.3** in `~/mlx-parity` (0.32.2 does not exist; 0.29.3 is the newest published) | — |
| Qwen3.5-4B-MLX-8bit | present, **4.78 GB** blob (the weights live in the shared `hub/blobs`, not under the model dir) | — |

The servable-model rule therefore uses the **measured** 37.44 GiB working set and
28.08 GiB per-buffer ceiling, not an estimated 75% of 48.

---

# THE PLAN — one PR: "Apple Metal Kernel Mapping and Pipeline support"

Repo read: `/tmp/claude-996/wt-certify` (main @ `cb6a592e2`). All paths below are relative to that root. Nothing was built, edited, or pushed.

---

## 0. Facts this plan rests on (counted here, with the command)

| fact | value | command |
|---|---|---|
| `#[cfg(feature="cuda")]` sites in `crates/**.rs` | **130**, in 38 files, 6 crates (spark-model 16 files, spark-runtime 7, spark-server 6, spark-storage 5, avarok-core 3, avarok-spark-bench 1) | `grep -rn 'feature *= *"cuda"' --include=*.rs crates/ \| wc -l` |
| distinct cfg *shapes* wrapping it | **15** — 78 plain, 12 `not()`, 13 `all(cuda, avarok_rdma_verbs)`, 4 `all(cuda, target_os="linux")`, 3 `all(cuda, unix)`, 2 `any(cuda, test)`, … | `grep -rhoP '#!?\[cfg[^]]*feature *= *"cuda"[^]]*\]' … \| sort \| uniq -c` |
| `#[cfg(feature="metal")]` sites | **5** | same, `"metal"` |
| `target_os = "macos"` in `crates/**.rs` | **0** | `grep -rn 'target_os *= *"macos"' --include=*.rs crates/` |
| `default = ["cuda"]` crates | avarok-core, spark-runtime, spark-model, spark-storage, avarok-spark-bench; `["cuda","nccl"]` spark-comm, spark-server | `grep -A3 '^\[features\]' crates/*/Cargo.toml` |
| `spark-server` has a `build.rs` | **NO** (the other five do) | `[ -f crates/spark-server/build.rs ]` |
| Metal `kernel void` entry points | **83**, of which **31** are `nllb_*` | `grep -rhoP '^kernel\s+void\s+\K\w+' kernels/metal --include='*.metal' \| sort -u \| wc -l` |
| `.metal` sources / symlinks under `kernels/metal` | **43** / **0** | `find kernels/metal -name '*.metal' \| wc -l`; `-type l` |
| gb10 `.cu` / `.cuh` / symlinks | 347 / 15 / 91 | `find kernels/gb10 …` |
| distinct literal `__global__ void NAME` texts in gb10 | **514** — and this number is WRONG as a denominator (see below) | `grep -rhoP '__global__\s+(?:__launch_bounds__\([^)]*\)\s+)?void\s+\K\w+' kernels/gb10 --include='*.cu' --include='*.cuh' \| sort -u \| wc -l` |
| files binding an entry point via `#define KERNEL_NAME` | **21** | `grep -rl '#define KERNEL_NAME' kernels/gb10 --include='*.cu'` |
| instantiation macros / invocations | 3 macros — `AVAROK_WYN_INSTANTIATE` + `AVAROK_WYN_F16_INSTANTIATE` (**24** invocations), `AVAROK_MOE_BATCHM_ENTRY` (**7**) | `grep -n 'AVAROK_WYN_INSTANTIATE(\|…' … \| grep -v define \| wc -l` |
| kernel lookups in `crates/**.rs` (2-string form) | **1104** sites, **584** distinct `(module,func)` pairs, **578** distinct func names, **213** distinct modules | python scan of `.kernel("m","f")` / `try_kernel(x,"m","f")` / `has_module("m")`, DOTALL |
| lookups whose name is built with `format!` (invisible to that scan) | **5** — 4 in examples, **1 in production**: `crates/spark-model/src/layers/w4a16_gemv_tiers.rs:144` `format!("w4a16_gemv_batch{w}")` | `grep -rn 'kernel(.*&format!' crates/ --include=*.rs` |
| `kernel_audit::record` call sites | **2**, both `crates/spark-runtime/src/cuda_backend/gpu_impl.rs:376,383`. `MetalGpuBackend::kernel` (`crates/spark-runtime/src/metal_backend.rs:511`) records **nothing** | `grep -rn 'kernel_audit::record' crates/` |
| `metal_backend` tests / `maybe_backend()` skip sites / `#[ignore]` | **41** `#[test]`, **43** `maybe_backend()` calls, **7** ignore attributes (5 `real_model_*` legs) | `grep -rh … crates/spark-runtime/src/metal_backend/tests/*.rs \| wc -l` |
| Metal kernels with a **silent** context truncation | **8** files declare `threadgroup float scores[MAX_SEQ_* = 4096]`; **6** clamp via `min(seq_len, MAX_SEQ_*)`, **2** via `s < seq_len && s < MAX_SEQ_*` | `grep -rl 'threadgroup float scores\[MAX_SEQ' kernels/metal/common/*.metal` |
| `metal` in `crates/avarok-plugin/src/gate/coverage.rs` | **0 occurrences**. `PERF_PATHS` contains bare `"kernels"` **and bare `"crates"`**. `REQUIRED` = 12 gates | `grep -c metal …/coverage.rs`; `sed -n '63,73p'` |
| record attestation coverage | `.benchmarks/bfcl-subset/2026-09-06-da016237db.json` `closure` has **26 keys, all `gb10/*`**, zero metal → `closure::excuses` (`gate/closure.rs:117`) returns false for any `kernels/metal` path | python json read |
| `[benchmarks.limits]` for metal | absent, and **already pinned absent** by `crates/avarok-plugin/src/hardware/limits.rs:~284` (`for hw in ["hopper","b200","strix","strix-hip","metal"] assert_eq!(limits(root,hw).unwrap(), None)`) | `sed -n '260,295p'` |
| `BENCH.toml` files in tree | **4**, all under `kernels/gb10/` | `find kernels -name BENCH.toml` |

**Could not look** (stated, not guessed):
- **The true gb10 entry-point denominator.** 514 is a text count: it includes macro placeholders (`KERNEL_NAME`, `AVAROK_PREFILL_ENTRY`) as if they were kernels, and excludes the 31 macro-generated names. Only `build_shadow::entry_points` resolves `#define KERNEL_NAME`+`#include`, `CONCAT` pastes and instantiation macros (`crates/avarok-kernels/build_shadow.rs:134`). I may not run cargo on this box. **Neither 595 nor 514 may be quoted.** Phase 0 exists to make this number a machine output.
- **Whether any Metal parity test currently passes on a device.** No Mac access (not my box in this workflow), no builds. Every "this test fails today" below is derived from source, and labelled as such.
- **Which flags `xcrun metal` accepts** on `apple-48gb-metal`. Not testable from here.
- **Whether the MLX Qwen3.5 checkpoint stores RMS weights offset-from-1 or standard.** This decides whether the divergence in §3 is a live serving bug or only a mapping hazard.
- `scripts/mac` is **not in the repo** (it lives at `/tmp/claude-996/scripts/mac`, outside the checkout). No PR artifact may reference it.

---

## 1. The spine

### Phase 0 — Pin the denominator and the dispatch set (no kernel, no cfg, no GPU)

**Goal.** Both sides of the 1:1 map become machine output from *the resolver the build uses*, never a hand count.

**Work.**
- New `crates/avarok-kernels/tests/entry_point_census.rs`, on the exact pattern of `crates/avarok-kernels/tests/kernel_shadow_detector.rs:25` — `#[path = "../build_shadow.rs"] mod build_shadow;`. Walk `kernels/gb10` and `kernels/metal`, dedupe symlinks (91 in gb10, 0 in metal), print both resolved sets keyed by **(entry point, source path)**.
- Second half, same test binary: resolve the **dispatch** set. Parse every `.kernel("m","f")`, `try_kernel(g,"m","f")`, `has_module("m")` literal in `crates/`. Then handle the 5 `format!` sites explicitly as **declared prefix families** in a small table in the test (`w4a16_gemv_{t}`, `gated_delta_rule_wy{k}`, `w4a16_gemv_sw_moe_batchm_m{r}`, `w4a16_gemv_batch{w}`): a family entry claims a regex over resolved names, and the test fails if a family matches **zero** resolved names (a dead family) or if a new `format!`-built lookup appears that no family covers.
- Emit the census to `$GITHUB_STEP_SUMMARY` when the env var is present; commit **no generated file** (the manifest is the hand-edited artefact; a committed generated report drifts).

**Proof.** Test asserts: every resolved gb10 name is reachable from exactly one (path, name) pair; every literal lookup resolves in *some* hardware tree or is listed in an `unresolved_lookups` table with a reason; the census totals are printed, not asserted against a literal (a pinned total is a merge conflict per kernel added).

**Negative control.** In a scratch commit, comment out one `AVAROK_WYN_INSTANTIATE(9)` line: the census total must fall by exactly 1 **and** the dispatch cross-check must go red naming `gated_delta_rule_wy9` (the `gdn_wyn_bench` family claims it). If the total does not move, the resolver is not expanding paste macros and any manifest built on it is short by ≥31 rows. Second control: delete the `w4a16_gemv_batch{w}` family row — the test must fail naming the production site `crates/spark-model/src/layers/w4a16_gemv_tiers.rs:144`.

**Lands as** commit 1.

---

### Phase 1 — Make the four existing Metal checks capable of failing

**Goal.** Before any parity claim, remove the four ways a Metal check reports on something other than Metal.

**Work.**
1. **The skip hole.** `crates/spark-runtime/src/metal_backend/tests/helpers.rs:19` `maybe_backend()` returns `None`; **43** call sites take `else { return }`; libtest counts that as PASS. Add a process-wide `AtomicUsize` bumped on every skip, an env var `AVAROK_METAL_NO_DEVICE=1` the device-less leg must set explicitly, and a final test `every_leg_reached_a_device_or_the_run_declared_it_had_none` that fails when `skips > 0` and the var is unset. Set the var on the hosted `macos-14` step in the same commit, so the required context stays green *while saying* it did not execute.
2. **The audit hole.** Add the two `crate::kernel_audit::record(module, func, loaded, site)` calls to `MetalGpuBackend::kernel` (`metal_backend.rs:511`), with `#[track_caller]` on the fn and `let site = std::panic::Location::caller();` as its first line — copied from `cuda_backend/gpu_impl.rs:364-383`. `audit_and_gate` (`crates/spark-server/src/main_modules/serve_phases/kernel_gate.rs:115`) is **not** cfg-gated on cuda, so this gives Metal the boot gate, `seal()`, and `atlas_kernel_lookups_unresolved` it has never had. A missing Metal kernel becomes a boot refusal instead of a mid-decode `Metal: unknown module`.
3. **The ignore hole.** `ci.yml:1023-1027` currently emits `::warning` when `ignored != 0` with the model present. Make it `exit 1`.
4. **The corpus hole.** `ci.yml:986` emits `::warning title=No MLX model on this runner` and lets four `real_model_*` legs skip. Add `AVAROK_METAL_REQUIRE_CORPUS=1` on the self-hosted leg; absence becomes a hard failure there (hosted leg keeps the warning, having declared `AVAROK_METAL_NO_DEVICE=1`).
5. **Poison fill.** In the parity harness allocator: fill every output allocation with `0xA5` before launch. This is what makes "never written" stop being spelled the same way as "computed zero" — the reason `real_model_misc.rs:234`'s `nonzero_count >= n_rows/2` heuristic exists at all.
6. **The launch contract.** New `crates/avarok-kernels/tests/metal_launch_contract.rs`, modelled on `crates/avarok-kernels/tests/kernel_arity.rs` (which exists because an 8-arg launch of a 9-param kernel shipped). Per Metal kernel: pin `threads_per_threadgroup` and the rows/elements-per-threadgroup divisor. The harness derives, from that pin, the index set the kernel *promises* to write; poison surviving inside the promised set is a hard failure. Rule attached: **no parity test may hand-roll `launch_typed`** — every case goes through the production wrapper (`MlxInt8Weight::gemv` at `crates/spark-runtime/src/weights/mlx_int8.rs:202-235` and its siblings).

**Proof.** On the device leg: 0 skips, 0 ignored, 0 poison inside any promised index set. On the hosted leg: red unless `AVAROK_METAL_NO_DEVICE=1` is set on the step. In `cargo test --workspace`: a new `metal_lookup_is_audited` unit test on a mock backend.

**Negative control.** Delete `AVAROK_METAL_NO_DEVICE` from the hosted step → the required context must go RED (proving it was previously green-by-skip). Remove one `kernel_audit::record` call → `metal_lookup_is_audited` must fail by name. Declare a launch-contract pin of 64 threads for `mlx_int8_gemv` → the coverage assertion must fail (proving the pin is load-bearing, not decorative). Point `AVAROK_MLX_MODEL_DIR` at an empty dir on the device leg → red, not `ignored`-and-green.

**Lands as** commits 2 (skip/ignore/corpus + audit) and 3 (poison + launch contract). *The instrument must be red before it is green* — commit 3's CI run is expected to fail on `mlx_int8_gemv`, which is the whole point.

---

### Phase 2 — Fix the live defect the instrument just found

**Goal.** Close the geometry bug, in its own commit, after the instrument reported it.

**The defect, from source.** `kernels/metal/common/mlx_int8_gemv.metal:35,52` — `ROWS_PER_TG = 4u`, `row = tg_idx*ROWS_PER_TG + simd_group_id`; the header at :29-30 states the contract ("`ceil(N/4)` threadgroups, 128 threads"). Production obeys it (`mlx_int8.rs:217-220`: `[out_features.div_ceil(4),1,1] × [128,1,1]`). Two tests do not: `real_model_gemv.rs:132-136` and `real_model_misc.rs:186-190` launch `[n,1,1] × [64,1,1]`. 64 threads = 2 simdgroups, so `simd_group_id ∈ {0,1}` and **every row with `r mod 4 ∈ {2,3}` is never written** — 32 of 64 at `n_rows = 64` (`real_model_misc.rs:77`).

**Work.** Route both tests through `MlxInt8Weight::gemv`. Delete the two hand-rolled launches. Keep `nonzero_count` **only** if poison fill has made it redundant — prefer deleting it and asserting the promised index set instead.

**Proof.** With poison fill the pre-fix diagnosis prints as *"32 poisoned runs at stride 4, offsets {2,3}"* — a defect that names itself. Post-fix: zero poison, `max_abs_diff < 0.1` holds.

**Negative control.** Re-introduce the `[64,1,1]` block in a scratch commit → the launch-contract test must fail *and* the poison report must name stride 4.

**Lands as** commit 4.

---

### Phase 3 — Gate reach, paid once, and the reason it is free here

**Goal.** Stop a Metal-only diff from re-opening all 12 GB10 gates, forever, and stop paying for it on every future Metal PR.

**The measurement that decides the schedule — and that both candidate plans and both judges missed.** `PERF_PATHS` (`coverage.rs:63`) contains **`"crates"`** as well as `"kernels"`. The target_os migration touches **38 `.rs` files across 6 crates**. So **this PR invalidates all 12 required gates no matter what** — the driver is the cfg migration, not the Metal kernels. `coverage.rs` being `BOUNDARY_FILES[0]` therefore costs **nothing additional in this PR**. Plan 1's proposal to schedule that commit separately, and its "this is the one commit that costs GPU time", are both wrong: the campaign is already bought. Do it here.

**Work.** Two shared consts beside `GATE_MACHINERY` (`coverage.rs:344`), listed in all 12 `*_EXCLUDES` arrays:
```rust
const METAL_KERNELS: Exclusion = Exclusion {
    prefix: "kernels/metal",
    rationale: "no gb10 target compiles a .metal source (0 symlinks under kernels/metal; \
                build_target.rs routes vendor=apple to xcrun only), so a Metal kernel edit \
                cannot move a GB10 number",
};
const METAL_BACKEND: Exclusion = Exclusion {
    prefix: "crates/spark-runtime/src/metal_backend",
    rationale: "compiled only under cfg(avarok_metal); a gb10 release build never includes it",
};
```
plus a **third** entry `crates/spark-runtime/src/metal_backend.rs` — `under()` is component-wise (`coverage.rs:912`), so the directory prefix does **not** match the sibling `.rs` file. Getting this wrong is the exact trap that function's `★` comment warns about.

**Do NOT exclude** `crates/spark-runtime/src/weights/mlx_int8.rs`: it is `pub mod mlx_int8;` **unconditionally** (`crates/spark-runtime/src/weights.rs:469`), so it *is* compiled into a gb10 release binary and the exclusion's argument does not hold for it.

**Proof.** New cases in `crates/avarok-plugin/src/gate/coverage_tests.rs`: `invalidated_by(["kernels/metal/common/rms_norm.metal"])` is empty; same for both metal_backend paths; `invalidated_by(["crates/spark-runtime/src/weights/mlx_int8.rs"])` is **all 12** (the deliberate non-exclusion); and `kernels/metal` appears in no `BOUNDARY_FILES` entry.

**Negative control.** Drop the exclusion from exactly one gate's array → the test must fail naming that gate id. Change the dir prefix to `crates/spark-runtime/src/metal_back` → the test must fail (proving `under()` is component-wise and not `starts_with`).

**Lands as** commit 5.

---

### Phase 4 — The target_os migration

See **§4** for the full commit order and trap list. Lands as commits 6-9.

---

### Phase 5 — The manifest, the checker, the checker's self-test, the mutation registry

**Goal.** The 1:1 map becomes a reviewable artefact **and** a verdict, with no row able to read "mapped" on the strength of a test that cannot fail.

See **§2**. Lands as commits 10-12.

---

### Phase 6 — The oracle: measure MLX's arithmetic before declaring any bit-exact tier

**Goal.** Turn "byte-identical with MLX" into a per-row, measured verdict — and settle `-ffast-math` once, tree-wide.

**Work.**
- **Probe kernels** in a new `kernels/metal/common/parity_probe.metal`, three entry points whose outputs *differ* between candidate behaviours on adversarial inputs: `probe_fma` (`a*b+c` where contracted-FMA and rounded-mul-then-add differ in the last bit), `probe_transcendental` (`exp`/`rsqrt` at inputs where `metal::precise::*` and `metal::fast::*` differ), `probe_reduce_order` (a permuted fp32 vector summed in two declared orders). A small python driver computes the same three expressions through `mlx.core`. **The measured answers are committed beside the manifest** and select each row's tier. Nothing about MLX's arithmetic is assumed.
- **The flag decision, tree-wide, not per-file.** `kernels/metal/common/KERNEL.toml [build] extra_metal_flags = ["-ffast-math", "-DTQ_PLUS_SIGNS"]`. Under `-ffast-math` the compiler may contract and reassociate, so bit-identity is not a property we control. **A per-verdict-class flag split is not implementable as either candidate described it**: `build_flags::merge_extra_flags` (`crates/avarok-kernels/build_flags.rs:~78`) merges hardware → `common/KERNEL.toml` → model-quant KERNEL.toml, **dedupes first-position-wins, and forwards one list to every source in the target** (`build.rs:1200-1234` → `job.extra_flags` → `build_target.rs:191`). Mixing `-ffast-math` with `-fno-fast-math` in that list is order-dependent, and its own doc-comment records the last time this drifted ("the Metal per-quant toml lost `-ffast-math`"). So: **one decision for the whole metal target** — recommend dropping `-ffast-math` and adding `-ffp-contract=off` (the Metal analogue of gb10's `--fmad=false`), asserted in `crates/avarok-kernels/tests/kernel_build_flags.rs`. Owner decision (§7).
- **Pin the oracle.** MLX version, checkpoint sha, prompt, token list and step count into a committed fixture under `tests/fixtures/`. Use the **teacher-forced** mode the existing harness already supports (`AVAROK_FORCE_TOKENS_FILE`, `tests/metal_kv_kld_compare.py --teacher-forced`), so the comparison does not stop at the first argmax divergence. Add `--exact` to that script: first differing **byte offset**, both values, and the **stride pattern** of differing offsets.

**Proof.** Probe cases pass with the committed MLX answers. One T2 kernel (`mlx_int8_gemv`) bit-exact against MLX over the corpus, as the existence proof that T2 is reachable at all. One T3 run in the PR body: first N logit bytes identical, or the exact offset and delta.

**Negative control.** Recompile with `-ffast-math` restored → the probe cases must fail (this is the control that separates "we match" from "we happen to match at these inputs"). Swap one `metal::precise::exp` for `metal::fast::exp` → the transcendental probe must fail. Perturb one byte of the reference dump → `--exact` must name that offset.

**Lands as** commits 13-14 (probes + oracle fixture; the flag change alone, since it recompiles every Metal kernel).

---

### Phase 7 — Write kernels, one family per commit, debt ratchets down

**Goal.** Convert `unported` rows to `mapped` in reviewable units; each commit green; each commit lowers exactly one number and grows the mutation registry.

**Order, by leverage measured in the tree, not by entry-point count.**
1. **Paged decode attention.** gb10's paged decode is warp-shuffle + online softmax with no `mma.sync`/`cp.async`/TMA, so `__shfl_xor_sync` → `simd_shuffle`/`simd_max`/`simd_sum` is 1:1 with nothing to emulate. It is also where today's Metal is worst: `attention_decode.metal` runs a three-pass softmax with single-threaded (`if (tid == 0)`) sweeps over `seq_len`, and **8 files** carry a `threadgroup float scores[4096]` (16 KB of a 32 KB budget) with a **silent** truncation — 6 via `min(seq_len, MAX_SEQ_*)`, 2 via a loop bound. Replace with online/flash softmax + tiled K/V staging.
2. **Norms + elementwise.** Where the shadow trap lives (§3) and where fusion/launch-count wins are.
3. **GEMM prefill.** `dense_gemm_bf16.metal:12` and `mlx_int8_gemm.metal:10` both promise a `simdgroup_matrix` follow-on, and those are the **only two** occurrences of that string in all 43 files.
4. **FLA / GDN decomposition**, including the rollback-snapshot forms.
5. **MoE ptrtable**, MLA, vision, sampling.
6. **LAST: TurboQuant.** Three families (prefill / paged decode / kv-append) whose byte layouts are hand-encoded and must agree bit for bit; Metal today covers only the symmetric `turbo2/3/4/8` and `bf16k_turbo2/3/4v` decode+append axes.

**Per-commit rules.**
- Port the **shadow**, not `common/`, wherever a model shadow is the optimized source, and name the source path in the row. Each row's `source` field is checked against the census, and the checker refuses a row whose source is a `common/` file that has a diverged shadow for a model the Metal target serves.
- **Delete the kernel being replaced in the same commit.** A rewrite that leaves the old entry point dispatched proves nothing.
- Also port the **launch-count** paradigms, each as its own row with its own case: the three-way `fused_k_norm_rope_cache_write_*`, `residual_add_rms_norm`, `embed_from_argmax` (removes a device→host→device round trip per token), and the `_batchN` / `_batched` specializations. "Ported" must not be allowed to mean "a slow equivalent exists".
- Every `exception` row gets its `why` at creation, never retro-fitted.

**Proof, per commit.** The family's rows are `mapped`; their named test fns exist and **ran on a device** (Phase 1's skip counter makes that checkable); every mutation registered for the family is caught; `RATCHET.toml` decreased. Plus one structural assertion, once: **no `.metal` file declares a `threadgroup` array sized by a literal ≥ 4096** — that closes all 8 truncating kernels at once, where a single `seq_len = 4097` case closes one.

**Negative control, per commit.** Each family's registered mutations (§3) must each be caught by at least one case, and a `seq_len = 4097` case must fail before the decode rewrite and pass after.

**Lands as** commits 15..N, one per family shard.

---

### Phase 8 — Promote the real-device context

**Goal.** Make device parity required without the queue-forever failure the file already carries a scar for.

**Work.** `metal-device-parity` (`ci.yml:907`) is `continue-on-error: true` and pinned to bare `[self-hosted, macOS, ARM64]` (`ci.yml:939`). Its stated reason — no Metal compiler on the box — is contradicted 130 lines above by `test-macos-metal`'s `★ THE BOX NOW HAS THE METAL COMPILER (2026-09-17, owner-installed)`. Resolve it: drop `continue-on-error`, and route it **by variable exactly as `test-macos-metal` does** (`ci.yml:802`: `${{ fork && 'macos-14' || vars.MACOS_RUNNER_LABEL || 'macos-14' }}`). Keep the same-repo guard (`ci.yml:936-938`) so fork PRs never execute on the Mac. Add the context + its env pins to `.github/scripts/assert-gates-are-wired.py` `REQUIRED_CONTEXTS` (line 76) and `STUB_FREE_STEPS` (line 186).

**Proof.** `assert-gates-are-wired.py` passes with the new entry; the context is red on a PR that breaks one parity case.

**Negative control.** Delete one `AVAROK_SKIP_BUILD: "0"` line → `assert-gates-are-wired.py` must fail (it already does this for the existing job). Route the new job to a label no runner carries → the wiring assertion must refuse the required entry.

**Lands as** commit N+1. **Branch protection is a string no committed file can set — owner action (§7).**

---

## 2. The mapping mechanism

**Where.** `kernels/metal/parity/<family>.toml`, one shard per family. Derive the family list from the **module map** the census produces (213 distinct modules are looked up) — not from a count I did not make. Each shard ≤ ~60 rows so it is reviewable; `.github/workflows/file-size-cap.yml` only caps `.rs` under `crates/`, so reviewability, not the cap, is the reason to shard.

**Key.** `(gb10 source path, entry point)` — **never the bare name.** Proof this is required, measured: `kernels/gb10/common/rms_norm.cu:106` computes `xv0*rms*(1.0f + wv0)` and its header says *"Qwen3-Next uses offset-from-1 normalization … weight is initialized to 0 and stored as offset"*; `kernels/gb10/gemma-4-26b-a4b/nvfp4/rms_norm.cu:100` computes `xv0*rms*wv0` and its header says *"Gemma-4 uses STANDARD RMS normalization"*. Same name, two different formulas, selected by target. A name-keyed manifest lets one Metal `rms_norm` be declared "mapped" to both.

**Row schema.**
```toml
[[kernel]]
source  = "kernels/gb10/common/rms_norm.cu"   # census-checked
entry   = "rms_norm"
state   = "mapped"        # mapped | exception | unported
metal   = "rms_norm"      # required for mapped; the Metal entry-point NAME
tier    = "T1"            # T1 | T2 | T3 — required for mapped
test    = "metal_rms_norm_matches_reference"  # required for mapped; must exist
mutations = ["reduce_order", "weight_offset_form"]  # required for mapped; must be registered
# why:  required for exception, >=80 chars, placeholder-blocklisted, and repeated
#       as a TOML comment directly above the row
```
Metal kernels with no gb10 twin (the 6 `mlx_int8_*`, the **31** `nllb_*`, the 2 self-declared `lora_bgmv_*_stub`) live in a `[[metal_only]]` table that also requires a `why`.

**Namespace collisions are manifest data, not renames.** `kernels/metal/common/KERNEL.toml`'s `[modules]` table is **empty**, so every `.metal` registers under its file stem, while gb10's `[modules]` renames 14+ stems (`paged_decode_attn → paged_decode`, `rms_norm → norm`, …). So `attention_decode` vs `paged_decode_attn`, `kv_cache_append_turbo2` vs `reshape_and_cache_flash_turbo2`, `add_rms_norm` vs `residual_add_rms_norm`, `rope_apply` vs `rope_forward`, `selective_scan_decode` vs `mamba2_ssm_decode`, `gdn_compute_gate` vs `compute_gdn_gates`, `silu_gate` vs `fused_silu_mul` all become rows with an explicit `metal =` value. Renaming live Metal kernels would break `Qwen35Kernels::resolve` (`crates/spark-model/src/forward/qwen3_5/mod.rs:172`, 29 handles) for no correctness gain.

**The checker is a Rust test, not a Python script.** `crates/avarok-kernels/tests/metal_parity.rs` (+ a `tests/support/` parser module to stay under the 500-LoC cap), using `#[path = "../build_shadow.rs"] mod build_shadow;`. It therefore **reuses the resolver the build uses** and reports under the already-required context **`cargo test --workspace`** (`ci.yml:720,753`) with **zero workflow change and no branch-protection change**. `scripts/check_kernel_shadows.py`'s own header explains why a second Python copy of the resolution rule would itself be the defect — that argument applies here verbatim, which is why this is not a Python step on `kernel-structure`.

**It fails on seven classes:**
1. a resolved gb10 (path, entry) pair in no shard, or in more than one;
2. a resolved Metal entry point that is neither some `mapped` row's `metal` value nor a `[[metal_only]]` row;
3. a `mapped` row whose `test` fn does not exist under `crates/spark-runtime/src/metal_backend/tests/`;
4. a `mapped` row naming a mutation that is not in the registry, or a `mapped` row with an empty `mutations` list;
5. an `exception` whose `why` is < 80 chars or matches the placeholder blocklist (`TODO`, `N/A`, `see above`, `not needed`, `later`);
6. a `mapped` row covering two divergent gb10 sources (the `rms_norm` trap);
7. either counter in `kernels/metal/parity/RATCHET.toml` above its committed value.

**Exceptions must be machine-checkable where the class permits it** — this is the hole in a bare ratchet, and the ratchet alone does not close it. Two classes are decidable and are checked:
- `no_dispatch_site` — the census finds no literal lookup **and** no `format!` prefix family matching the name. The 5 dynamic sites are why this is a census question and not a grep.
- `declared_superseded` — the name appears in `kernels/gb10/common/KERNEL.toml [shadow_exempt]` (verified: `w4a16_dequant`, `moe_w4a16_grouped_gemm`, `rms_norm_residual_vanilla`, `residual_add_rms_norm_vanilla`, `gated_rms_norm_f32_input_strided`).

Every other class (`no_hardware_analogue`, `capability_probe`, `model_not_servable`) is prose, so **`RATCHET.toml` carries TWO monotone counters**: `unported`, and `unverifiable_exceptions`. Moving a row from `unported` to a prose `exception` lowers one and raises the other — it cannot lower the total. That is what stops a verbose 80-character rationale from being a route to a green 1:1 gate.

**Refusing to pass when it could not read either side.** Three explicit refusals, because an empty read must never look like agreement:
- resolved gb10 set empty → fail (`kernels/gb10` unreadable or the resolver broke);
- resolved Metal set empty, or `< 83` with no shard change explaining it → fail;
- the shard directory missing or parsing to zero rows → fail. The absence of the manifest is not a pass.

**The checker's own self-test.** `metal_parity_selftest` cases in the same test binary, on the precedent of `scripts/check_cross_hardware.py::selftest()` (*"A rule that cannot fire is a rule that is not a check"*, line 307) — RED/GREEN temp trees, one per defect class above, each of which the checker **must** reject, plus a clean tree it must accept, plus the three refusal cases. A missing fixture must fail rather than exit 0.

---

## 3. The parity harness and the per-family contract

**How both sides are driven over the same bytes.**
- `tests/mlx_capture.py` (run on the Mac by the designated agent or the CI leg — **not** via a path outside the repo): drive `mlx.core` over a deterministic case corpus and write per case `input.bin` + `expected.bin` as **raw little-endian device bytes, with no float parsing anywhere**, plus a manifest recording `mx.__version__`, `mx.metal.device_info()`, dtype, shape, seed, and a sha256 per blob. Small cases committed under `tests/fixtures/`; large cases regenerated in `--live` mode.
- `crates/spark-runtime/src/metal_backend/parity/`, split SBIO-style — `compare.rs` (pure verdict), `driver.rs` (all IO), `corpus.rs` (case loading), `mutate.rs` — so each file stays under the 500-LoC cap.
- **What is compared:** `memcmp` over the output buffer. On mismatch: first differing index, count, and the **stride pattern** of differing indices. Poison (`0xA5`) surviving inside the launch contract's promised index set is reported separately, as poison, never as a value mismatch.
- **Every case goes through the production wrapper.** A harness that encodes its own launch geometry tests a launch nothing ships — that is exactly the bug in §Phase 2.

**The three tiers, and why one word cannot cover this tree.**

| tier | oracle | bar | why this family gets it |
|---|---|---|---|
| **T2** | MLX, per kernel | **bit-exact** (`memcmp` == 0) | Only where MLX has a counterpart: its int8 GEMM/GEMV, RMSNorm, RoPE, SDPA. |
| **T1** | FP32 CPU reference | ≤ 2 BF16 ULPs **plus every registered mutation caught** | The only oracle for GDN/KDA/SSM, TurboQuant, MoE-expert-GEMM, sparse attention — MLX has no equivalent. |
| **T3** | `mlx_lm.generate`, end to end | logit **bytes**, teacher-forced | One shared model; pinned fixture; `metal_kv_kld_compare.py --exact --teacher-forced`. |

**Why T1 needs the mutation registry, with a live instance.** `docs/METAL_BACKEND.md:89-90` claims *"Every kernel has an FP32 CPU-reference parity test within ≤2 BF16 ULPs"*. On bf16 (8 mantissa bits) 2 ULPs is ~0.8 % relative slack — wide enough to absorb a wrong accumulation order, a dropped K tile, or a mis-encoded TurboQuant nibble. Worse, the reference can be written *from* the kernel: `crates/spark-runtime/src/metal_backend/tests/parity_norms.rs:236` computes `v * inv_rms * w`, which is exactly what `kernels/metal/common/rms_norm.metal:68-69` computes — while `kernels/gb10/common/rms_norm.cu:106` computes `x * rms * (1 + w)` for Qwen3-Next. **The existing "≤2 ULP" test agrees with the kernel and cannot see a whole-formula divergence.** So a `mapped` T1 row without registered mutations is not evidence, and the checker refuses it (class 4 above).

**The mutation registry.** `AVAROK_METAL_MUTATE=<name>`: the harness reads the `.metal` **source** from the repo (tests run in the checkout), applies one named textual mutation, compiles it, and runs the family's parity cases against the mutant. Mechanism: prefer `xcrun -sdk macosx metal` → AIR → `metallib` into a temp dir, then `newLibraryWithData_error` — **both halves already exist in tree** (`crates/avarok-kernels/build_target.rs:172-226` does exactly that two-step; `crates/spark-runtime/Cargo.toml`'s comment records that `newLibraryWithData_error` is the API used and that `dispatch2` ships the wrapper). `newLibraryWithSource` is a cleaner alternative **if** the objc2-metal 0.3 binding exists — I could not check that from here.

`crates/spark-runtime/src/metal_backend/tests/metal_mutations.rs` asserts **every registered mutation is caught by at least one parity case, and a mutation nothing catches FAILS THE BUILD.** Starter set: `drop_last_k_tile`, `group_index_off_by_one`, `softmax_two_pass`, `reorder_accumulation`, `reinstate_seq_clamp_4096`, `flip_turboquant_nibble_order`, `weight_offset_form` (`w` ↔ `1+w`), `swap_precise_for_fast`.

**The substitute contract for every non-bitwise row** — so a tier downgrade is not an escape hatch. All four, or the row is a defect:
(a) run-to-run byte determinism on the same device; (b) **byte-invariance across two legal grid shapes** — this is what catches an order-dependent reduction posing as deterministic, and nothing else in the plan does; (c) a tight ULP bound vs an fp64 CPU reference; (d) permutation control — reorder K in the **input**; the bound must still hold while the bytes may change.

**How bit-identity with MLX is actually attempted, in order** (for `mlx_int8_gemv` vs MLX `qmv`): (1) the per-element dequant expression `byte*s + b`, with contraction forced off unless `probe_fma` says MLX contracted it, in which case an explicit `fma()`; (2) the **K partition** — ours is lane-strided (`for (k4 = simd_lane_id; k4 < K4; k4 += 32)`, `mlx_int8_gemv.metal:67`), which is a different addition tree from a block-contiguous partition and will differ in the low bits for the same data; matching MLX requires adopting its partition exactly; (3) the combine — `simd_sum`'s butterfly order is unspecified in the language spec, so it coincides only if MLX also uses `simd_sum` over the same partition; (4) the epilogue — fp32 accumulate, one RNE narrowing to bf16, no intermediate. If (2) or (3) cannot be matched — and anything MLX routes through `simdgroup_matrix` cannot be, since Apple does not specify its internal accumulation order — the row is **T1 with the substitute contract**, and the reason is written in the row *and* at the kernel head.

---

## 4. The target_os migration, in commit order, with every trap named

Cargo cannot make a feature target-conditional, so selection moves into build-script cfgs — following the repo's own precedent, not a new invention: `crates/avarok-rdma/build.rs:23-30` and `crates/spark-runtime/build.rs:16-19` already emit `cargo:rustc-cfg=avarok_{scale,hip,cutlass,flashinfer,rdma_verbs}` with `rustc-check-cfg` declared **first, before any early return**.

**Commit 6 — emit the cfgs; rewrite nothing.**
Each of the five crates with a `build.rs` emits, unconditionally and first:
`println!("cargo:rustc-check-cfg=cfg(avarok_cuda)"); println!("cargo:rustc-check-cfg=cfg(avarok_metal)");`
then `avarok_cuda` = (`CARGO_FEATURE_CUDA` set) **AND** `CARGO_CFG_TARGET_OS != "macos"`; `avarok_metal` = `CARGO_CFG_TARGET_OS == "macos"`. Test: `crates/avarok-kernels/tests/` case asserting the resolution table for the 4 (feature × target) combinations, computed by a pure fn extracted into a `build_backend.rs` included by each build script — same `#[path = …] mod` shape as `build_flags.rs`, *"so the rule does not live only inside build.rs, where nothing tests it."*

**Commit 7 — `spark-server` gets a `build.rs`.**
**Trap:** it has **none** today, and 6 of its `.rs` files carry `feature="cuda"` cfgs. `rustc-check-cfg` **does not cross crates** (`crates/spark-storage/build.rs:32-36` says so explicitly, and records the macOS ordering landmine where the early return preceded the check-cfg line). Without its own build script, every rewritten site there is a hard error under `[workspace.lints.rust] warnings = "deny"`. Either add the build script or leave spark-server on `feature = "cuda"`; recommend adding it.

**Commit 8 — rewrite the 130 + 5 sites.**
**Trap:** it is **not** a `sed`. There are **15 distinct shapes**, including 13 × `all(feature="cuda", avarok_rdma_verbs)`, 4 × `all(feature="cuda", target_os="linux")`, 3 × `all(feature="cuda", unix)`, 3 × `not(all(...))`, 2 × `any(feature="cuda", test)`, 2 × `cfg_attr(not(all(...)), allow(dead_code))`, 1 × `all(feature="cuda", not(feature="nccl"))`, and 2 × `all(feature="metal", not(feature="cuda"))`. Rewrite per shape, and assert the anchor before each write (a mid-script failure that discards edits already made is a known trap here). Gate afterwards: a grep check that no `feature = "cuda"` / `feature = "metal"` cfg remains in `crates/**/*.rs`, wired into the required `cargo test --workspace` as a test rather than a shell step.

**Commit 9 — `cudarc` target-gated; build.rs defaults metal.**
- Move `cudarc` from optional `[dependencies]` to `[target.'cfg(not(target_os = "macos"))'.dependencies]`. Precedent in tree: `crates/spark-storage/Cargo.toml:49` already has a `[target.'cfg(target_os = "linux")'.dependencies]` table. `default = ["cuda"]` may then stay: on macOS the feature resolves, the dependency is inactive, and `avarok_cuda` is off. **This is the correction that matters most.** Keeping `cuda` as a default-on **empty alias** — as the winning candidate proposed — activates all **130** `cfg(feature="cuda")` blocks on macOS while `cudarc` has been removed from the graph: a hard compile failure in exactly the flagless `cargo build` requirement 6 exists to fix, and the proposed proof (`cargo tree` contains `objc2-metal`, not `cudarc`) stays green straight through it.
- `crates/avarok-kernels/build.rs:204` — `auto_skip_macos = target_os == "macos" && !hw_explicit` writes a stub whose `metallib_modules()` is `Vec::new()`; `MetalGpuBackend::new` accepts an empty slice and the binary dies at first lookup with `Metal: unknown module` (the #907/#909/#911 incident, written up at `ci.yml:848-877` and guarded by `.github/scripts/assert-gates-are-wired.py` `STUB_FREE_STEPS` and `embedded_metallib_set_is_not_empty`). Turn that branch into target **selection**: default `AVAROK_TARGET_HW=metal`, `AVAROK_TARGET_MODEL=qwen3-5-4b-vlm-mlx-int8`, `AVAROK_TARGET_QUANT=mlx_int8`. Keep `AVAROK_SKIP_BUILD=1` honoured **first**, for the ubuntu type-check jobs.
- **Keep `metal` and `cuda` as empty no-op alias features.** 40+ live references pass `--features metal` (`ci.yml` ×6 steps, `docs/METAL_BACKEND.md`, three `[[example]] required-features` blocks in `crates/spark-runtime/Cargo.toml`, `/tmp/claude-996/scripts/mac_metal_test.sh`). Deleting the feature breaks all of them for no correctness gain — but the features must now select **nothing**.
- Fix the facts in `kernels/metal/HARDWARE.toml`: `memory_gb = 16` and `memory_bandwidth_gbps = 200` describe an M2 Pro; the box is a 48 GB Mac. The OOM watchdog reads this as its hint.

**Proof.** On the Mac: `cargo build -p spark-server --bin spark` with **no flags** succeeds, and `otool -L target/debug/spark | grep -i 'cuda\|nccl'` is empty — the assertion `ci.yml:886-905` already makes, now against a flagless build. On Linux: plain `cargo build` still produces a CUDA binary. In `cargo test --workspace`: the resolution-table test plus the no-feature-cfg-remains test.

**Negative controls.** (a) Force `avarok_metal` off on macOS via a control-only env override → the metal backend must fail to compile, proving the cfg selected it. (b) Force `avarok_cuda` on on macOS → must fail at `cudarc`, proving the target table excluded it. (c) Re-introduce one `cfg(feature="cuda")` → the grep test must name the file. (d) A Linux leg with `AVAROK_SKIP_BUILD=1` must still type-check, so the build-default change has not broken the ubuntu jobs. (e) Set `AVAROK_SKIP_BUILD=1` on the Mac leg → `embedded_metallib_set_is_not_empty` must fail **by name**, not as 41 lookup failures.

---

## 5. The benchmark gate: open, declared, and incapable of reading green

Four enforced layers. None of them can be read as a pass, and each fails on a *half*-closure.

1. **No thresholds.** `kernels/metal/qwen3-5-4b-vlm-mlx-int8/BENCH.toml` with `[[benchmarks]]` entries carrying `status = "unmeasured"` and **no `[benchmarks.metrics]` table**. Schema-enforced, not convention: `crates/avarok-plugin/src/gate/bench.rs:129` **bails** on an unmeasured entry carrying thresholds (*"A guessed number a run can clear is worse than no number — it reports PASS for something nobody measured"*), and `bench.rs:~245` **drops** unmeasured entries from the baseline, so `resolve()` fails loudly with "no baseline" rather than passing with empty bounds.
2. **No envelope.** Deliberately do **not** add `[benchmarks.limits]` to `kernels/metal/HARDWARE.toml`. `crates/spark-server/src/cli/bench_certify/mod.rs:104` **bails** for a class that declares none, naming the missing thermal envelope, memory floor and timing allowances. On a Mac with no `--hardware`, `Hardware::probe().gate_key()` (`crates/avarok-plugin/src/hardware.rs:104-107`) returns `"unknown"` and `bench_certify` bails first. **Both paths are refusals.** Paste the exact refusal text into the PR body — the record then shows the gate refusing, not passing.
3. **No candidate row.** Do **not** add `metal-kernel-bench` to `coverage::PROMOTION_CANDIDATES`: `coverage.rs:863` states the rule (a candidate naming an unregistered id is *"a debt row nobody can ever discharge"*) and `coverage_promotion_tests.rs:18` `every_promotion_candidate_is_a_registered_benchmark` pins it. The id does not exist yet.
4. **The openness is a reviewed decision, not an absence.** `metal_bench_gate_is_open_and_says_so` in `crates/avarok-plugin/src/gate/coverage_tests.rs` asserts five facts at once: no `[benchmarks.limits]` for metal (**extending** the existing pin at `crates/avarok-plugin/src/hardware/limits.rs:~284`, not duplicating it — that loop already covers `"metal"`, so add the assertion there and reference it, per SSOT); no metric table on any `kernels/metal/**/BENCH.toml` entry; `metal-kernel-bench` in neither `REQUIRED` nor `NOT_REQUIRED` nor `PROMOTION_CANDIDATES`; `kernels/metal` excluded from all 12 gates; **and a zero-entry Metal benchmark set is REFUSED rather than reported as `0 of 0 passed`.** The failure message names the closing sequence: measure the envelope on the 48 GB box → declare the limits → register the driver → list the candidate → require it.

**Negative controls.** Add a `[benchmarks.metrics]` table to one unmeasured Metal entry → `bench.rs`'s existing rejection fires **and** the openness test fires. Add a plausible `[benchmarks.limits]` block → the existing `limits.rs` pin must go red, proving the open state is *asserted* and not merely unconfigured. Add a `metal-kernel-bench` candidate row → `every_promotion_candidate_is_a_registered_benchmark` must fire.

**The speed answer (requirement 5) is a receipt, not a gate.** An `avarok-spark-bench` leg times our path and MLX's over the same corpus on the same box, writes JSON to a CI artifact, and commits nothing to `.benchmarks/` — a committed record is a claim the certification machinery then treats as a baseline. Two structural levers make the receipt readable: `MetalGpuBackend::launch_typed` (`crates/spark-runtime/src/metal_backend.rs:441-492`) opens a **fresh `MTLComputeCommandEncoder` per launch**, and it loops `useResource_usage` over **every live allocation on every dispatch** while cloning the whole allocation map twice (`self.allocations.lock()` at :441-442). Both are O(allocations)-per-launch costs that the Phase 7 fusion work pays down twice. Measure them before defending them.

---

## 6. The first five commits, in order, each with its test

| # | commit | test that closes it | negative control |
|---|---|---|---|
| **1** | `metal: census both sides from the build's own resolver` — `crates/avarok-kernels/tests/entry_point_census.rs` via `#[path = "../build_shadow.rs"]`; `(entry, source path)` sets for gb10 + metal; dispatch-site resolution incl. the 5 `format!` prefix families | the test itself: every resolved name reachable once; every literal lookup resolves or is tabled with a reason; every `format!` family matches ≥1 resolved name; **empty-set refusals** on either tree | comment out `AVAROK_WYN_INSTANTIATE(9)` → total falls by exactly 1 **and** the dispatch check names `gated_delta_rule_wy9`; delete the `w4a16_gemv_batch{w}` family → fails naming `w4a16_gemv_tiers.rs:144` |
| **2** | `metal: a skipped leg, an ignored leg and a missing corpus must not read as a pass` — skip counter + `AVAROK_METAL_NO_DEVICE`; `kernel_audit::record` in `MetalGpuBackend::kernel`; `ignored != 0` → exit 1; `AVAROK_METAL_REQUIRE_CORPUS` | `every_leg_reached_a_device_or_the_run_declared_it_had_none`; `metal_lookup_is_audited` (mock backend, in `cargo test --workspace`) | drop `AVAROK_METAL_NO_DEVICE` from the hosted step → required context RED; remove one `record` call → `metal_lookup_is_audited` fails by name; empty `AVAROK_MLX_MODEL_DIR` on the device leg → RED |
| **3** | `metal: poison fill + a launch contract, and the gemv leg goes red` — `parity/{compare,driver,corpus}.rs` (SBIO split, ≤500 LoC each); `0xA5` prefill; byte-index + stride diff; `crates/avarok-kernels/tests/metal_launch_contract.rs` pinning threads/TG and the rows-per-TG divisor per kernel | the device leg **must fail** on `mlx_int8_gemv`, reporting *32 poisoned runs at stride 4, offsets {2,3}* | declare `mlx_int8_gemv`'s pin as 64 threads → the coverage assertion must fail, proving the pin is load-bearing |
| **4** | `metal: route the real-model legs through the production gemv wrapper` — delete the hand-rolled `[n,1,1]×[64,1,1]` launches at `real_model_gemv.rs:132` and `real_model_misc.rs:186`; call `MlxInt8Weight::gemv` | commit 3's leg goes green: zero poison inside the promised index set; `max_abs_diff < 0.1` | re-introduce `[64,1,1]` → launch-contract test fails **and** the poison report names stride 4 |
| **5** | `gate: a Metal-only diff cannot invalidate a GB10 gate` — `METAL_KERNELS`, `METAL_BACKEND` and the `metal_backend.rs` sibling Exclusion in all 12 `*_EXCLUDES`; `kernels/metal/HARDWARE.toml` memory facts corrected | `coverage_tests.rs`: `invalidated_by` empty for all three metal prefixes; **all 12** for `crates/spark-runtime/src/weights/mlx_int8.rs` (the deliberate non-exclusion — that file is `pub mod` unconditionally) | drop the exclusion from one gate → test names that gate id; shorten the prefix to `…/metal_back` → test fails, proving `under()` is component-wise |

Commits 1-5 are GPU-free and touch no kernel source. Commit 3 is **expected red** and commit 4 turns it green: the PR history then shows the instrument failing before it passed, which is the cheapest available proof that it is not decorative.

---

## 7. What needs the OWNER's decision

1. **What "byte-identical with MLX" means.** MLX has no GDN, no TurboQuant, no MoE-expert-GEMM, so per-kernel MLX comparison is empty for most of the tree. End-to-end logit bytes on `mlx-community/Qwen3.5-4B-MLX-8bit` only, per-kernel bit-exactness only where MLX has a counterpart, or both — and **which MLX version and checkpoint sha pin the oracle**?
2. **Drop `-ffast-math` tree-wide from `kernels/metal/common/KERNEL.toml`, in favour of `-ffp-contract=off`?** It is the Metal analogue of gb10's `--fmad=false` and a precondition for any bit-exactness claim; it costs speed against requirement 5. A per-kernel split is **not** available (§Phase 6).
3. **What is the denominator for "100 % Metal support"** — every resolved gb10 entry point, or every kernel an Apple-servable model dispatches? I counted 584 distinct `(module,func)` pairs looked up in `crates/` against a gb10 declaration count I could not resolve. If it is the dispatched set, the port is materially smaller and the never-dispatched gb10 names become `no_dispatch_site` exceptions.
4. **Which models must serve on the 48 GB Mac?** That single answer decides how many rows are legitimately `model_not_servable` versus real work. `Qwen35Kernels::resolve` wires 29 handles for one 4B MLX-int8 model; the optimized gb10 shadows are 27B/35B NVFP4.
5. **Is the offset-from-1 divergence a live serving bug?** `kernels/metal/common/rms_norm.metal:68` computes `x*rms*w`; `kernels/gb10/common/rms_norm.cu:106` computes `x*rms*(1+w)` for Qwen3-Next. If the MLX Qwen3.5 checkpoint stores offset weights, Metal is renormalising wrongly today and the existing `metal_rms_norm_matches_reference` cannot see it (its CPU reference computes the same formula as the kernel).
6. **Promote `metal-device-parity` to a required context?** It needs a **branch-protection string no committed file can set**, plus an entry in `assert-gates-are-wired.py::REQUIRED_CONTEXTS`, and it must route by `vars.MACOS_RUNNER_LABEL` — never the bare `[self-hosted, macOS, ARM64]` label it carries today.
7. **Accept a two-counter ratchet as the representation of debt** (`unported` + `unverifiable_exceptions`, both monotone down), or must every row reach `mapped` inside this PR? The ratchet is what makes the pipeline green at every commit while hundreds of rows are open; the alternative is a PR that is red until its last commit.
8. **Scheduling.** This PR invalidates all 12 required gates because the cfg migration touches `crates/**` — a full campaign, whatever else it does. A campaign is measuring on this box right now. When does it run, and on which box?

---

## 8. What I would NOT do, and why

1. **I would not keep `cuda` as a default-on empty alias feature.** It activates 130 `cfg(feature="cuda")` blocks on macOS with `cudarc` removed from the graph. That is a hard compile failure in exactly the build requirement 6 asks to work, and the dependency-graph proof proposed for it stays green through the failure.
2. **I would not write the 1:1 checker in Python.** It would need a second copy of the entry-point resolution rule; `scripts/check_kernel_shadows.py`'s own header explains that the duplicate *is* the defect. A Rust test reusing `build_shadow.rs` reports under the already-required `cargo test --workspace` with no workflow or branch-protection change.
3. **I would not attempt a per-kernel or per-verdict-class `-ffast-math` split.** `build_flags::merge_extra_flags` resolves flags per **target**, dedupes first-position-wins, and forwards one list to every source; its doc-comment records the last drift this caused. Mixing `-ffast-math` and `-fno-fast-math` in that list is order-dependent. One tree-wide decision, tested.
4. **I would not schedule the `coverage.rs` exclusion as a separate commit or PR "to control the GPU cost".** The cost is per-PR and already bought by the `crates/**` cfg migration. Splitting it buys a second campaign for nothing.
5. **I would not accept `≤2 BF16 ULPs` as sufficient evidence for a `mapped` row.** ~0.8 % relative slack, and the tree contains a reference written from the kernel it checks (`parity_norms.rs:236`). Every T1 row carries registered mutations or the checker refuses it.
6. **I would not rename the 30 live Metal kernels to match gb10 names.** It breaks `Qwen35Kernels::resolve`'s 29 handles and the NLLB call sites for no correctness gain. Collisions are manifest data (`metal = ` values).
7. **I would not port from `kernels/gb10/common/` by default.** For the flagship models `common/` is the slow baseline by design; each row names its source and the checker refuses a `common/` source that has a diverged shadow for a served model.
8. **I would not pin a required real-device check to the bare `[self-hosted, macOS, ARM64]` label.** `ci.yml:785-801` and `release-build.yml:236` already record a required check **queueing forever** on a label nothing answers — unmergeable with nothing showing broken.
9. **I would not fix the 4096-token truncation per kernel with a `seq_len = 4097` case alone.** I counted **8** kernels carrying that clamp. The structural assertion (no `threadgroup` array sized by a literal ≥ 4096) closes all eight; the case proves one.
10. **I would not port TurboQuant early.** 3 families with hand-encoded byte layouts that must agree bit for bit across prefill, paged decode and kv-append. It is the easiest place in the port to ship a silent layout mismatch, so it goes last, behind a working mutation registry.
11. **I would not reference `/tmp/claude-996/scripts/mac` from any committed file.** It is outside the checkout. Mac execution belongs to the designated Mac agent or to the CI job, and a PR artefact that names a path only one machine has is a step nobody else can run.
12. **I would not commit a Metal benchmark record, threshold, `[benchmarks.limits]` table, or promotion-candidate row.** Every one of those converts a deliberately open gate into a number nobody measured — and `bench.rs:129`, `bench_certify/mod.rs:104` and `coverage_promotion_tests.rs:18` already refuse each of them by name.
13. **I would not quote 595, 514, or any hand count as the gb10 denominator**, or repeat "30/64" as an observation (I found no such record in the tree — only the assert format string it would be derived from). The denominator is commit 1's output or it is not a number.