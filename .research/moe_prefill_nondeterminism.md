# qwen4_exp temp-0 nondeterminism (prefix cache OFF) — ROOT CAUSE FOUND AND PROVEN

Date 2026-09-02, box dgx-00, worktree `/home/ms/atlas/.claude/worktrees/exl3-research`
@ 75e5026a8 (tree left clean; the one throwaway patch is saved as
`det_experiment.patch` in this directory and was reverted + the binary rebuilt
pristine). All artifacts (serve logs, probe logs, scripts) live in
`/home/ms/.claude/jobs/5a7bd33d/tmp/nondet/`.

## 1. Verdict

**The nondeterminism is the fp32 `atomicAdd` epilogue of the fused EXL3 MoE
PREFILL kernel, whose commit ORDER is nondeterministic because experts are
assigned to concurrent expert-groups by a dynamic ticket scheduler.** Each
prompt token's routed-MoE output row receives its top-k=10 expert
contributions as `atomicAdd`s into the fp32 accumulator in whatever order the
groups finish; fp32 addition is not associative, so the prefill hidden states
differ at the bit level on every run, and greedy decode amplifies near-tied
argmaxes into different text. Decode is bit-deterministic; the entire defect
is injected during prefill.

Mechanism (file:line):
* `kernels/gb10/common/exl3_vendor/exl3_moe_kernel.cuh:281-305` — the down
  epilogue `had_d_out` calls `had_hf_r_128_d_inner`, which ends in
  `atomicAdd(output_ptr + {0,32,64,96} + t, ...)`
  (`kernels/gb10/common/exl3_vendor/hadamard_inner.cuh:490-493`) into the fp32
  `output_state` accumulator (`[T, hidden]`, one row per token, shared by all
  of that token's experts).
* `kernels/gb10/common/exl3_vendor/exl3_moe_kernel.cuh:309` — expert-to-group
  assignment is `sched[2+group_idx] = num_groups + atomicAdd(&sched[0], 1)`:
  a dynamic ticket draw, so WHICH group processes WHICH expert, and hence the
  arrival order of the adds into any token row, varies run to run.
* Host launch: `crates/spark-model/src/layers/ops/exl3_matmul/moe_prefill.rs:277-286`
  — `num_groups = min(pf_concurrency=SMs/8=6, num_active)`, so ≥2 groups run
  concurrently at every realistic shape.
* Same bug class, second site (NOT exercised by the short repro, fires at
  count>128 experts i.e. long prefill chunks):
  `exl3_moe_scatter_add_f32` (`kernels/gb10/common/exl3_matmul.cu:447-465`),
  the overflow-expert weighted scatter-add into the same accumulator.

Why the text only diverges some runs / some positions: the fp32 accumulator is
cast to BF16 downstream, which swallows most 1-ulp reorder deltas; divergence
escapes only when a delta crosses a BF16 rounding boundary, then grows through
the layers/positions. Hence bit-identical prompt logprobs for the first 3-4
positions, stochastic onset after, and more distinct outputs the longer the
generation — exactly the reported length dependence (<=64-token generations
"stable", 250-token generations 2-5 distinct of 6, 300-token worse).

## 2. The experiment that proves it

A one-hunk, env-gated, arithmetic-preserving patch (`det_experiment.patch`,
now reverted): `ATLAS_EXL3_MOE_DET=1` forces `num_groups = 1` in the fused
kernel launch. One group processes experts strictly in ticket order with a
group barrier between experts, so every token row receives its expert
contributions in a FIXED order — same adds, same values, ordered commit.
No other code changed; kernels untouched.

| arm (all: prefix cache OFF, C=1, temp 0, identical requests) | prefill probe (echo, 8 reps) | e2e text (shortdet, 6 reps, 250 tok) |
|---|---|---|
| base (native MoE+dense, stock binary) | **7/8 distinct** prompt-logprob vectors | **2/6 distinct** (5/6 on 2026-08-31) |
| ATLAS_QWEN4EXP_NO_PLE=1 | **5/8 distinct** | **4/6 distinct** |
| ATLAS_EXL3_MOE_DET=1 (ordered epilogue) | **1/8 — bit-identical** | **1/6 — six identical completions** |

Logs: `echo_base.log`/`probe_base` (= `serve_probe.log` session + task
bragorkua), `echo_nople.log`/`probe_nople.log`, `echo_det.log`/`probe_det.log`,
`lp_*.log`. The det arm's `lpprobe` (8 reps × 4 decode steps, top-5 logprobs)
is value-bit-identical at every step; the only residual variation is the SORT
ORDER of exactly-tied entries inside the reported top-5 list (e.g. "It"/"The"
both at -2.9092941) — a cosmetic unstable tie sort in the logprob reporting
path, not a numeric difference, and the chosen token is unaffected.

Probes (in this directory): `echoprobe.py` — N identical `/v1/completions`
requests with `echo:true, logprobs:1`, compares the PREFILL's own per-position
logprobs bit-for-bit (this is what isolates prefill from decode without any
rebuild); `lpprobe.py` — same for the first 4 decode steps' top-5 logprobs;
plus the original `warmdiff/shortdet.py` e2e hash probe. Boot:
`boot_nd.sh` = `.research/boot/boot_native_dense.sh` with `--enable-prefix-caching`
dropped (PREFIX=0) and levers overridable; driver `run_arm2.sh`.

## 3. Minimal repro

```
PREFIX=0 CTX=8192 SEQS=1 LOG=<log> bash /home/ms/.claude/jobs/5a7bd33d/tmp/nondet/boot_nd.sh
python3 -u /home/ms/.claude/jobs/5a7bd33d/tmp/nondet/echoprobe.py http://127.0.0.1:8890 8
# -> 5-8 distinct prompt-logprob vectors from 8 identical 89-token requests.
```
No decode needed: the prompt's own logprobs already differ, typically from
position ~4-5 on, at up to ~1e-2 magnitude by mid-prompt.

## 4. What was ruled out, and how

* **Decode (all of it: mgemm MoE tier, GDN recurrence, attention, PLE decode,
  sampling).** With only the prefill epilogue ordered, 6/6 e2e completions and
  all decode-step logprob VALUES are bit-identical. Consistent with code:
  the decode-tier `exl3_mgemm` reduces expert slots in a fixed j=0..stride-1
  loop (`exl3_gemm_kernel.cuh:270-305`) and the gemm split-K reduction is
  lock-ordered by slice index (`exl3_gemm_inner.cuh:600-640`); no float
  atomics anywhere else on this model's path (verified by grep: zero
  `atomicAdd` in `gated_delta_rule*.cu`, `hyper_connection.cu`, `ple.cu`,
  all `paged_decode_attn*` kernels).
* **PLE / n-gram row cache (prior hypothesis 4).** `ATLAS_QWEN4EXP_NO_PLE=1`
  still gives 5/8 distinct prefill vectors and 4/6 distinct texts. Also
  code-level: rows gather by slot from a pinned arena faulted synchronously
  under a mutex before launch; the documented in-flight-eviction race
  (`ple/gather.rs:80-93`) needs ~65k misses in one resolve — unreachable here.
* **Sampling (hypothesis 5).** temp 0 takes the GPU `argmax_batch` fast path
  (`decode_logits_step.rs:174-176`); det arm proves it stable given stable
  logits. The unstable TIE ordering seen in `top_logprobs` listings is in the
  reporting path only.
* **Scheduler/batch shape (hypothesis 2).** All probes are C=1 sequential with
  identical spacing; the det arm is deterministic under the same scheduler, so
  shape/timing is not a numeric input. (First-request-after-boot effects were
  dodged by a warmup request in every probe.)
* **Stale scratch / cross-request leakage (hypothesis 3).** The echo probe's
  runs 2 and 3 (and 5 and 7) were bit-identical to each other mid-sequence
  with different neighbors — inconsistent with content leaking from the
  previous request; and the det arm is deterministic across requests with no
  scratch changes. (`out_f32` is also explicitly zeroed every call,
  `moe_prefill.rs:406`.)
* **Prefix cache / Marconi.** Disabled throughout (`--enable-prefix-caching`
  omitted; prompt is 89 tokens, below marconi_min_tokens=256 anyway).
* **QSA.** Inactive below 2051 context tokens; repro is 89.

## 5. Relation to the prior "RESOLVED (3 causes)" investigation

The 2026-07-28 nondeterminism memo (atlas-decode-nondeterminism) was on the
Nemotron/Puzzle stack: (1) float `atomicAdd` in `gated_rms_norm`
(commit 3cf579a6), (2) warm-only chunk split (a21d1dd4), (3) unsound
exact-full-prompt shortcut (a009242). None regressed — this model never runs
those kernels/paths. This is a FOURTH source, in the EXL3 kernel family that
did not exist then, but it is the same bug class as cause 1: float atomics
with nondeterministic commit order. The `mamba2_ssd_chunk.cu` atomicAdd flagged
there as a probable second source is likewise not on this model's path.

Downstream consequences: the warmdiff/mangle investigations' warm-vs-cold and
prefix-ON/OFF A/Bs were run on a baseline that could not produce two identical
runs — this is the confound their reviews flagged (`fix_review.txt`, INFO
entry "det_noprefix_warm2.log"). The phantom-typo "byte-identical strings look
different" symptom is what jittering prefill activations over a re-read of the
model's own text should produce, but that link is inference, not measured here.

## 6. Fix design

The throwaway `num_groups=1` experiment is NOT the fix: it serializes experts
and measured TTFT 433 ms vs 230 ms on the 89-token prompt (~+88%), and would
be far worse on long prefills (hundreds of active experts serialized).

Recommended fix, in order:
1. **Deterministic epilogue via per-slot scratch + fixed-order reduce.** Have
   `had_d_out` (and the overflow `exl3_moe_scatter_add_f32`) write each
   expert's weight-scaled row to its SLOT in a `[T*top_k, hidden]` scratch
   (slot index = the token's sorted-slot position, already available as
   `token_sorted`'s index), then one grid-stride kernel reduces each token's
   top_k slots in fixed slot order — exactly the contract the decode tier
   already uses (`exl3_gemm_kernel.cuh:262` grouped reduction), which this
   investigation proved deterministic. Memory: prefill batch cap × top_k ×
   hidden × 4B ≈ 50-100 MB fp32 at pf_t_cap≈512 — allocate once next to the
   existing `out_f32`. Expected cost: one extra read+write of the routed rows
   per layer call — low single-digit % of prefill, nowhere near the 88%
   serialization number. Gate it as default-ON with a lever to restore the
   atomic epilogue for A/B.
2. Alternatively (smaller change, more contention): keep `atomicAdd` but make
   group→expert assignment STATIC (`group g takes experts g, g+num_groups,...`)
   — this removes the ticket lottery but NOT the inter-group completion-order
   race on a shared token row, so it is NOT sufficient; listed only to record
   that it was considered and rejected on analysis.
3. While (1) is built: `ATLAS_EXL3_MOE_DET=1`-style serialization is a correct
   opt-in determinism mode for debugging/A-B work (every future qwen4_exp
   quality A/B should run under it or be N>=10 sampled).

## 7. Unsettled / caveats

* **The materialized-MoE control (`ATLAS_EXL3_NATIVE_MOE=0`) never completed**:
  it OOMs at util 0.6 (materialization ≈ 91 GB), and at util 0.8 the serve was
  SIGTERM'd twice right after model build by something on the box I could not
  identify (not my watchdogs — none alive, none logged; not earlyoom — journal
  shows 89% avail; `serve_moe0.log` retained). The det arm supersedes it for
  the verdict, but the "materialized path is deterministic" claim remains
  code-read only, and the mystery SIGTERM-er on dgx-00 is worth knowing about.
* **Long-context divergence** (the 300-token/3-distinct observation) is
  explained by the same mechanism scaling with steps, but at >128 rows/expert
  the overflow scatter-add joins in and QSA (whose indexer has its own
  unordered `atomicAdd` emit at `qsa_indexer.cu:649`) activates past 2051
  tokens — if any nondeterminism survives the epilogue fix at long context,
  those two are the next suspects, in that order.
* **C>1 concurrent batches untested** — batch composition changes routing
  batch shapes; re-run `echoprobe` at C>1 after the fix.
* The det arm was validated at one prompt/one length (89 tokens in, 250 out,
  6+8+8 reps). Rate on the stock binary varies day to day (5/6 vs 2/6 distinct)
  as expected for a scheduling race; any future regression check should use
  `echoprobe.py` (bit-level, catches it in 8 requests) rather than text hashes.
* The unstable tie-order in reported `top_logprobs` is real but cosmetic;
  worth a `sort_by(key, then token_id)` in the logprob formatter some day.

## 8. Hygiene

Worktree clean (`git status` empty), patch saved + reverted,
`target/release/spark` rebuilt from the pristine tree, all serves killed
(verified `pgrep`), no changes committed/pushed/stashed, gx10 untouched,
earlyoom untouched.
