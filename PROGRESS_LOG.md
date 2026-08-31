# perf/enterprise-concurrency-v3 — progress log

Rolling notes for the v3 concurrency branch: what is on it, how to reproduce
every measurement, and what has been settled vs what is still open.

Newest entries at the **bottom** of each section. Every number here was measured
on one GB10 (dgx-00) unless stated otherwise — treat single-box results as
indicative, and say so when quoting them.

---

## 1. What is on this branch

```
2848205c  perf(mtp): D-Cut depth pruning + verify cost cuts     <- PR #379 head
a91390e4  perf(ssm-tier): 22x cheaper snapshot eviction ...     <- PR #381
6e68fa8b  chore(ssm-tier): satisfy clippy and the 500-LoC cap
aa233714  fix(ssm-tier): reap dead tier keys ...                <- PR #382
1810854e  chore(ssm-tier): keep the reaping change under the cap
```

PR #381 and #382 cherry-picked onto #379 with **zero conflicts**. The two lines
of work do not overlap: #379 is MTP/verify/kernels, #381+#382 are the SSM
snapshot spill tier. They can land in either order.

Verified on this branch: release build clean, `atlas-tier` 8+5+16+2+5 = 36 tests,
`spark-model model::ssm` 76 tests, `cargo clippy --tests` clean.

---

## 2. Build

```bash
export CUTLASS_HOME=/home/ms/cutlass                 # REQUIRED — without it the
                                                     # binary builds but refuses to
                                                     # serve: "CUTLASS support was
                                                     # not built"
export RUSTFLAGS="-L <dir containing libnccl.so>"    # symlink libnccl.so -> libnccl.so.2
export LD_LIBRARY_PATH="<same dir>"                  # only needed for `cargo test`
ATLAS_TARGET_MODEL='*' cargo build -p spark-server --release
```

`ATLAS_TARGET_MODEL='*'` is mandatory — a single-target binary dies on any other
model. The binary is `target/release/spark` (not `spark-server`).

---

## 3. Reproduce: serve configs

### 3a. Qwen3.6-27B (centml W4A4) — the #379 benchmark checkpoint

```bash
docker run -d --name combo27 --gpus all --ipc=host --network host \
  -v <binary>:/spark:ro -v /tank/hf:/tank/hf \
  -e RUST_LOG=info -e ATLAS_TARGET_HW=gb10 \
  -e ATLAS_TARGET_MODEL=qwen3.6-27b -e ATLAS_TARGET_QUANT=nvfp4 \
  -e ATLAS_KV_OVERCOMMIT=1 \
  -e ATLAS_SSM_TAIL_MIDCHUNK=0 -e ATLAS_MTP_CATCHUP=0 \
  -e ATLAS_MTP_DRAFT_CONF=0.0 -e ATLAS_MTP_GATE_FORCE=1 -e ATLAS_SSM_TAIL_PROTECT=1 \
  -e ATLAS_SSM_TAIL_LEASE_TTL=128 \
  atlas-gb10:gdnf32-build \
  /spark serve <centml-snapshot> --model-name qwen3.6-27b --host 0.0.0.0 --port 8888 \
    --max-seq-len 131072 --max-batch-size 8 --max-num-seqs 8 --kv-cache-dtype bf16 \
    --gpu-memory-utilization 0.80 --scheduling-policy slai \
    --enable-prefix-caching --ssm-cache-slots 32 --ssm-checkpoint-interval 32 \
    --speculative --num-drafts 3 --mtp-quantization bf16 \
    --tool-call-parser qwen3_xml --disable-tool-grammar true --disable-thinking \
    --tool-max-tokens 32768 --request-timeout 0
```

**MMQ W4A4 is now ON in this config (2026-07-30):** `ATLAS_NO_FFN_NVFP4_MMQ=1`
is REMOVED — `c9965ce9` shows it existed only to let `BF16_TC_PREFILL` engage
(the accuracy campaign), and it outlived that reason when BF16_TC was dropped.
The checkpoint is W4A4-native (`NVFP4-W4A4-mlpinf`). Measured: prefill_short
C=16 63.2 -> 69.7 (best TTFT/TPOT of the probe series). Quality gate:
oc_harness 3/3 (calc 55s, sorter 49s clean; webserver builds+serves).
**Direction-following RESOLVED (2026-07-30, verified from opencode's event
log):** the recurring `followed=3/6, rc=1` webserver signature is NOT a
quality effect — the agent's conversation hits a hard 400 right after
`cargo test` ("Prompt too long: 17175 tokens exceeds max_seq_len 16384"),
killing the session before run/curl/teardown. Reproduced with MMQ OFF
(control: followed=2/6, same rc=1, 3/3 still passing) — W4A4 exonerated.
Actionable: agentic serving needs --max-seq-len >= 32K for this task class
(the 16K rep that scored 6/6 merely squeaked under the ceiling).
**CLOSED (2026-07-30): the agentic config of record is now 64K**
(`combo_conc_mmq64k.sh`: --max-seq-len 65536, KV_OVERCOMMIT=1, MMQ on,
funded by the 20.3 GB decode-ring reclaim — the ring skip + 64K coexist:
startup shows both). Verified: webserver task followed **6/6, rc=0**, 12
tools, 110 s — the complete write/test/run/curl/teardown sequence. Known
caveat at long contexts: past ~7.2K tokens batched MTP propose falls back
per-seq (the still-open 5.2 propose_meta sizing item) — functional, noisy,
and the next code item if 64K becomes the daily driver.

**CANONICAL TEST CONFIG IS NOW 4K (2026-07-30, user decision):**
`--max-seq-len 4096 --max-prefill-tokens 4096`, `ATLAS_KV_OVERCOMMIT` removed
(it existed only because 16 x 16K exceeded strict reservation; 16 x 4K = 64K
tokens fits the ~243K-token pool with room). Matches the #379 config-of-record
context, keeps the MTP drafter pool small (~4.1K blocks), and avoids paging
pressure from max_seq_len-scaled buffers during <=2K-token benches.
COMPARABILITY: every number in sections 6.2-6.8 was measured on the 16K
config — expect small deltas on the first 4K sweep; re-baseline before
attributing them to code. Scratch script: combo_conc_canonical4k.sh.

**Deliberate deviations from #379's own config of record** (`bench/phaseA_c_sweep.sh`):

| | #379 config of record | here | why |
|---|---|---|---|
| `ATLAS_BF16_TC_PREFILL=1` | set | **REMOVED** | crashes, see §5.1 |
| scheduling | fifo | slai | ours; works once BF16_TC is out |
| `--max-batch-size` | 16 (20 in the log) | 8 | ours |
| `--max-seq-len` | 4096 | 131072 | 128K context; see §5.2 for what this exposes |
| `ATLAS_KV_OVERCOMMIT` | unset | 1 | 8x128K exceeds a strict KV reservation |
| util | 0.70 | 0.80 | ours |

### 3b. Holo-3.1-35B-A3B-NVFP4

Full flag set + rationale is in the recipe PR
(`Avarok-Cybersecurity/atlas-recipes#13`). Two things that silently degrade it:
`ATLAS_HOLO_LOW_MEMORY_MOE=1` is the unlock (verify `CUTLASS grouped SFB: built
256 experts` ×40), and **KV must be `bf16`** — paged FlashInfer requires it.

### 3c. SSM spill tier (optional, #381/#382)

```bash
-e ATLAS_SSM_TIER=1 -e ATLAS_SSM_TIER_UNIFIED=1 \
-e ATLAS_SSM_TIER_SWAP_DIR=/ssm-swap -e ATLAS_SSM_TIER_DISK_GB=10 \
-e ATLAS_SSM_TIER_SLOTS=8 -e ATLAS_SSM_TIER_TIMING=1
```

All four of the first envs are required: the cap is inert without `_UNIFIED`, and
without `_SWAP_DIR` it caps **RAM, not disk**. Confirm from the log that the
**O_DIRECT** arm was taken and not the silent host-RAM fallback.

Disk writes only begin once the hot arena is full — first write is spill number
`ATLAS_SSM_TIER_SLOTS + 1`. A 0-byte swap file with a large `_SLOTS` is expected,
not a bug.

---

## 4. Harnesses

```bash
# agentic concurrency (opencode-driven, distinct task per slot)
ATLAS_CONTAINER=<name> python3 bench/agentic/conc_harness.py --levels 1,4,8 --model <id>
# sequential 3-task scorecard (calc / sorter / Rust axum webserver)
python3 oc_harness.py --model <id> --timeout 600
# prefill grid (TTFT-based, unique prefix per request)
python3 bench/agentic/prefill_matrix.py <id> --conc 1,2,4,8 --isl 1024,2048,4096,8192,16384
```

**Single-rep agentic pass rates are noise at C>=4.** Measured spreads of 4/8–8/8
on one fixed config. Use `--repeats 3` before quoting a pass rate. I reported a
single-rep number twice and had to retract both.

---

## 5. Findings

### 5.1 `ATLAS_BF16_TC_PREFILL=1` crashes — ROOT-CAUSED to the **v2** kernel

In #379's config of record. Symptom:

```
Prefill chunk layer 0 failed: cuLaunchKernel: CUDA_ERROR_INVALID_VALUE (1)
  (grid=[136,3,1], block=[128,1,1], shared_mem=0)
```

`grid.x=136` = `div_ceil(N,128)` with N=17408 (the 27B `intermediate_size`), so it
is the prefill FFN GEMM. Under **slai this becomes a SIGSEGV (exit 139) that kills
the process**; under fifo it degrades to a per-request HTTP 500. That escalation
is a separate bug worth fixing: a failed prefill launch should not take the
server down.

**BISECTED — three legs, one request each:**

| leg | env | result |
|---|---|---|
| A | `BF16_TC_PREFILL=1` + `DISABLE_PREFILL_V2=1` (forces **v1**) | **OK**, 55 tok, 0 launch errors |
| B | `BF16_TC_PREFILL=1` alone (uses **v2**) | **FAIL** HTTP 500, `grid=[136,3,1]` |
| C | neither (control) | **OK**, 35 tok |

So the lossless BF16 prefill path is FINE; `w4a16_gemm_t_m128_bf16_v2` is broken
on this target.

**The latent defect — the feature gates on one kernel and launches another**
(`layers/dense_ffn.rs`):

```rust
// guard: checks V1's handle
let bf16_tc_prefill = self.w4a16_gemm_t_m128_bf16_k.0 != 0 && env::var_os("ATLAS_BF16_TC_PREFILL").is_some();
// selection: prefers V2 whenever V2 is loaded
let use_v2 = self.w4a16_gemm_t_m128_bf16_v2_k.0 != 0 && env::var_os("ATLAS_DISABLE_PREFILL_V2").is_none();
let bf16_kernel = if use_v2 { ...v2_k } else { ...bf16_k };
```

The flag is admitted because v1 exists, then dispatches to a v2 that fails at
launch. The in-code comment calls v2 "bit-identical to v1" and preferred, which
is presumably why it shipped.

**Fix 1 (one line, correctness):** gate on the handle actually launched
(`bf16_kernel.0 != 0`, after selection), not v1's handle before it.
**Fix 2 (the real bug):** v2's launch failure itself. RULED OUT by reading: null
handle (it resolves; the kernel is in
`kernels/gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu`), static shared memory (v1 33,920 B
/ v2 30,336 B vs the 49,152 limit), and grid/block (matches the kernel's own
`blockIdx` arithmetic). Remaining suspect: the launcher's parameter pack vs v2's
compiled signature — v1 and v2 are documented to share "the same launch helper
(identical grid/block/args)", so a signature drift between them is the thing to
check first.

**OPERATIONAL ANSWER — do not drop the flag, pair it:**
`ATLAS_BF16_TC_PREFILL=1 ATLAS_DISABLE_PREFILL_V2=1` gives the lossless BF16
prefill (the point of the flag: the FP8 crush causes "length-truncations /
accuracy risk on Qwen3.6-27B") with no crash. Strictly better than removing it,
which is what §3a's config currently does.

**RESOLVED (2026-07-29) — fix 2 found and fixed: a MISSING 9th KERNEL ARG.**
The suspicion above was right. `e7daea87` (the #379 head commit) retrofitted a
9th param `unsigned int ldb` (transposed-B row stride, for lm_head's odd vocab
stride) onto exactly three kernels, all in the 27B dir only: `w4a16_gemm_t`,
`w4a16_gemm_t_p3` and `w4a16_gemm_t_m128_bf16_v2`. The first two got their
launchers updated (`w4a16_gemm_n128` forwards `ldb = n`); **the v2 bf16 kernel
did not** — `dense_ffn.rs` kept launching it through the 8-arg
`w4a16_gemm_n128_m128_bf16` helper. With `cuLaunchKernel`'s `void**` param
form the driver reads one host word per COMPILED param, so it dereferences
`params[8]` — one past the end of the 8-element arg array. Neighboring heap
word null → `CUDA_ERROR_INVALID_VALUE` (the fifo HTTP 500); non-null garbage
pointer → host SIGSEGV (the slai exit 139). One UB, both symptoms.

Fixed on this branch: dense_ffn routes v2 through
`w4a16_gemm_n128_m128_bf16_ldb` with `ldb = N` (the FFN transposed twins are
unpadded), the gate now checks the handle actually launched (fix 1), and the
same missing arg was fixed in `w4a16_bf16_v2_microtest.rs` /
`w4a16_bf16_v2_bench.rs` — the microtest that "proved v2 bit-identical" had
the identical UB, which is why v2 passed validation and shipped.
(`w4a16_m17_bench.rs` already passed a trailing `ldb` and documents that CUDA
ignores extra trailing args; only MISSING args are UB.) After this,
`ATLAS_BF16_TC_PREFILL=1` alone (v2, the faster variant) is expected to work —
re-run `v2_bisect.sh` leg B to confirm; the `DISABLE_PREFILL_V2` pairing
remains as the escape hatch.

### 5.2 `PROPOSE_META_STRIDE` caps batched MTP propose at ~7.2K context

`mtp_head/forward_batch.rs`: `ensure!(256 + block_table.len()*4 <= 2048)` →
448 block-table entries → **~7,168 tokens** at block_size 16.

It fails safe (`Err` → scheduler logs → `continue` → non-batched fallback), but
the batched path silently stops applying above ~7K context. Measured at C=4 on
the 27B:

| `--max-prefill-tokens` | pass | makespan | agg tok/s | `exceeds meta stride` |
|---|---|---|---|---|
| 4096 | 4/4 | 248.1 s | 18.1 | 0 |
| **8192 (default)** | **4/4** | **182.9 s** | **23.1** | **0** |
| 16384 | 4/4 | 334.0 s | 14.6 | **1650** |

At 16K prefill it fires 1,650 times with block tables of 3,592–3,728 entries —
**8× over the limit** — and costs ~45% of makespan. #379's own config of record
is `--max-seq-len 4096`, so its published table cannot hit this by construction.
The slab is only 32 KB total (`PROPOSE_META_SEQS × PROPOSE_META_STRIDE`), so
sizing it from `max_seq_len` looks cheap. Also: it logs at `ERROR` per step.

**8192 is the sweet spot.** Both 4K and 16K were worse.

**And #379's benchmark cannot reach this cap by construction.**
`bench/bench-atlas-concurrency.py` defines 6 ISL/OSL regimes but then filters
`[... if isl + osl <= MAX_SEQ_LEN]` with `MAX_SEQ_LEN` defaulting to 4096. At the
config of record that silently drops the only two long-context regimes:

| regime | ISL/OSL | total | survives at 4096 |
|---|---|---|---|
| prefill_short  | 1024/128  | 1152 | yes |
| balanced_short | 256/256   |  512 | yes |
| balanced_long  | 1024/1024 | 2048 | yes |
| decode_short   | 128/1024  | 1152 | yes |
| prefill_long "RAG / document 8K/1K" | 8192/1024 | 9216 | **DROPPED** |
| decode_long "Long reasoning 1K/8K"  | 1024/8192 | 9216 | **DROPPED** |

So every measured regime has **ISL <= 1024, total <= 2048** -> a block table of
`2048/16 = 128` entries against the 448-entry limit, **3.5x of headroom**. The
`exceeds meta stride` path is unreachable there, which is why 1650 hits at 16K
prefill never showed up upstream. Nothing logs the filter; `TEST_CONFIGS` just
comes back with 4 entries instead of 6.

Consequence for reading #379's table: "four of five levels beat vLLM" is a
short-prompt, short-generation result. It does not speak to >7.2K context, and
the batched-verify path it optimises is the one we measured falling back to
`verify_k2_step` once agentic contexts reach ~15K.

### 5.3 `ATLAS_SSM_TAIL_PROTECT=1` is inert in this config

(2026-08-05: the variable was renamed to the opt-out
`ATLAS_DISABLE_SSM_TAIL_PROTECT` — default ON — and dropped from every launch
script, since §5.3 itself proved it guards an empty set in this config. The
finding below stands as written for the old name.)

`radix_tree/snapshot.rs` says so in-code: the lease only shields `is_tail`
entries, whose sole production writer is reachable only when
`ATLAS_SSM_TAIL_MIDCHUNK != 0` — and the golden set pins `MIDCHUNK=0`. The
comment explicitly warns: *"Do not read a `ATLAS_SSM_TAIL_PROTECT=1` in a launch
script as evidence that tail protection is doing work."* #379's config sets both.

### 5.4 Spill cost: 412 ms → 19 ms (22×), and it holds under load

`CudaBackend::copy_d2h` synchronizes **inside every call**; `spill_slot` issued
60 of them. Fixed in #381 (async gather + one sync + a reusable pinned staging
blob). `ATLAS_SSM_TIER_TIMING=1`:

```
before: gather+sync=392936..411336us  store.put=19397us  total=412334us
after:  gather+sync=  1384..  1419us  store.put=17421us  total= 18841us  staging=pinned
```

Confirmed **under agentic load** (65 spills, Holo, C=1/4/8): gather 1398–1404 µs,
i.e. ±3 µs of the quiet-box number. So the win is not load-dependent — the
original cost really was the 60 blocking syncs. `store.put` is now 93% of what
remains (a host memcpy into the arena).

### 5.5 The disk cap works; reaping is still UNTESTED

Holo, `ATLAS_SSM_TIER_DISK_GB=1`, `_SLOTS=8`, agentic C=1/4/8: **350 spills**, cap
engaged once (latched WARN), swap file **1,002,700,800 B = exactly 15 records ×
66,846,720** — never exceeded the 0.93 GiB budget.

But **0 reaps and 1 fault-in in the whole run.** Reaping triggers on a fault-in
*miss*, and this workload almost never re-reads a spilled snapshot. So #382's
actual fix remains unexercised across three attempts. It needs a probe that
repeats the *same* long prefix across turns while the cap churns — the agentic
harness uses distinct prompts per slot by design.

Also note the economics on that run: **350 spills × ~19 ms ≈ 6.6 s of work for
one successful fault-in.** The tier is a clear net loss without genuine
multi-turn prefix reuse.

### 5.6 The spill gate is principled; the CHECKPOINT INTERVAL is what conflicts with it

On the 27B with `--ssm-checkpoint-interval 32`, **every** eviction candidate is
refused and the tier is fully inert (0 spills, 0 disk):

```
SSM spill SKIPPED (cost gate): victim depth 629 < ATLAS_SSM_SPILL_MIN_TOKENS=1024 — dropped
```

**The 1024 default is derived, not arbitrary** (`model/ssm_spill_gate.rs`):

```
spill_min ~= R * (C_s / p_target + C_f)
  R        ~= 6500 tok/s   measured SSM prefill throughput
  C_s      =  45 ms        budgeted spill cost (POST-gather-fix)
  C_f      ~= 50 ms        fault-in cost
  p_target =  0.3          fault-back probability a spill must earn
  6500 * (0.045/0.3 + 0.05) ~= 1300  ->  1024, block-aligned, biased to spill
```

The gate exists because **spilling is slower than recomputing** below that depth:
a spill is charged in full to whichever request triggered the eviction, while the
benefit accrues to a different, later request that may never come. Same file
notes the coupling: at the PRE-fix 400 ms spill cost the formula gives **~9000
tokens**, which is why the gate and the gather fix ship together.

It is also *more* right on the 27B than on Holo: the blob there is
**158,859,264 B (151 MB, 48 SSM layers)** vs Holo's 66,846,720 B, so `C_s` scales
~2.4x and the break-even depth rises with it. Refusing a 620-token victim is
correct.

**The real defect is that nothing ties the checkpoint interval to the gate.**
`--ssm-checkpoint-interval 32` = a checkpoint every **512 tokens**, so victim
depths cluster at 500-650 and can *never* clear a 1024-token gate — the tier is
silently disabled by a config that looks reasonable. There is no warning for this
(there IS one for `spill_min < fault_min`, so the precedent exists).

Options, unresolved:
- interval >= 64 (>=1024-token spacing) so victims can clear the gate — costs more
  SSM replay on partial hits;
- lower `ATLAS_SSM_SPILL_MIN_TOKENS` to match the interval — but on a 151 MB blob
  that is exactly the loss the gate was derived to prevent;
- warn at startup when `interval * block_size < spill_min`, i.e. when the config
  can never spill. **This one looks unambiguously worth doing.**

### 5.7 `bench-atlas-concurrency.py` measures PREFIX-CACHE HITS, not prefill

`make_prompt(target_tokens)` is **deterministic** — the same filler text for every
request at a given ISL, no per-request uniqueness. With `--enable-prefix-caching`
(which BOTH our config and #379's `phaseA_c_sweep.sh` set), only the first request
of a cell does real prefill; every later one is a full cache hit.

The evidence is in the prefill_long row: **TTFT p50 = 340 ms for an 8192-token
prompt** (~24K tok/s, implausible) against **p99 = 8051 ms** — the p99 is the one
cold request, the p50 is a cache hit.

Consequences:
* The **TTFT column is not a prefill measurement** in any regime. Aggregate tok/s
  is less affected (decode-dominated) but is still flattered.
* It likely explains why `prefill_long` recorded **0** `exceeds meta stride` hits
  despite 9216-token sequences (576 block-table entries vs the 448 limit): if the
  prefill is skipped via cache, the batched propose path is not exercised the way
  a cold 8K prefill would exercise it. **UNVERIFIED — worth confirming.**
* Applies to #379's published numbers too, same script, same caching flag.

Our own `bench/agentic/prefill_matrix.py` does this correctly and says why:
"Every request gets a unique random prefix: with prefix caching on, a repeated
prompt would be skipped entirely and the run would measure cache hits." Use that
for any prefill claim, or add a per-request nonce to `make_prompt`.

## 6. Measurements

27B (centml), C=4 agentic, this branch, tier inert per §5.6:

| config | pass | makespan | decode med | agg |
|---|---|---|---|---|
| #379 alone, default prefill | 4/4 | 182.9 s | 7.2 | 23.1 |
| + SSM stack, interval 32 | 4/4 | 207.3 s | 4.2 | 20.8 |

The second row is **not** a tier measurement (nothing spilled); the delta is the
denser checkpoint interval plus run-to-run variance. Needs repeats.

27B sequential scorecard (`oc_harness.py`), #379 latest head, stock sampling:
**3/3** — calc PASS 3 tools 44 s, sorter PASS 7 tools 106 s, webserver
`cargo_valid=True webserver_ok=True followed=6/6` 150 s.

Sampling A/B on the 27B (`--default-min-p 0.0 --default-top-n-sigma 0.0` vs the
0.08/1.0 CLI defaults): **keep the defaults.** Zeroing them left pass rates
nominally 3/3 but degraded the process — sorter tool errors 4 → 10, webserver
timed out at 600 s having followed 3/6 directions vs 6/6 in 150 s. The
`--default-min-p` doc explains why this checkpoint cares: it truncates the noisy
FP-quantization tail. Note requests arrive `temp=None`, so MODEL.toml
`[sampling.tools]` (temp 0.6 / top_p 0.95 / top_k 20) applies, layered under
these CLI defaults.

---

### 6.1 Concurrency baseline, 27B centml, 16K context (this branch)

Config per §3a but `--max-seq-len 16384`, `--max-num-seqs/--max-batch-size 16`,
`--max-prefill-tokens 8192`, SSM tier OFF. Aggregate tok/s:

| regime | C=1 | C=2 | C=4 | C=8 | C=16 | C=1->16 |
|---|---|---|---|---|---|---|
| decode_short 128/1024   | 29.7 | 43.2 | 69.6 | 109.3 | **153.3** | 5.2x |
| balanced_short 256/256  | 28.6 | 41.2 | 64.5 | 105.9 | **148.6** | 5.2x |
| prefill_short 1024/128  | 26.2 | 38.7 | 58.9 |  87.8 | **127.4** | 4.9x |
| balanced_long 1024/1024 | 26.4 | 38.8 | 63.2 |  96.6 | **150.2** | 5.7x |
| prefill_long 8192/1024  | 25.7 | 38.7 | 59.3 |  85.8 | **123.9** | 4.8x |

`n=32/32` at C=16 in every regime — zero dropped requests. TPOT degrades
gracefully (33 -> 96 ms on decode_short); TTFT p99 tracks p50, so no long-tail
stall. **decode_long (1024/8192) deliberately NOT run** — ~80 min for one regime
on this model; use `BENCH_SKIP_REGIMES=decode_long`.

For reference #379 reports 88.9 tok/s @C=8 and 131.9 @C=16 on this model; we
measure 109.3/153.3 (decode_short) and 96.6/150.2 (balanced_long). NOT
like-for-like (different flags, scheduler, context, and see §5.7 on caching) —
the useful read is only that nothing in this config regressed concurrency.

### 6.2 HONEST 1K baseline (nonce bench), slai/16K config — and what it broke

Same §6.1 serve config (slai, 16K, batch 16), binary at `de129a3c`
(v2-ldb fix + logits-aliasing pick, NO KV fixes yet), bench WITH the
per-request nonce (d09d6516). Aggregate tok/s:

| regime | C=8 | C=16 | §6.1 (cached) C=16 | delta |
|---|---|---|---|---|
| decode_short 128/1024   | 108.5 | **138.0** | 153.3 | -10% |
| balanced_short 256/256  |  90.2 | **123.1** | 148.6 | -17% |
| prefill_short 1024/128  |  55.2 |  **63.8** | 127.4 | **-50%** |
| balanced_long 1024/1024 |  21.9* |  24.9* | 150.2 | broken |

*balanced_long dropped requests (15/16, 30/32) and collapsed. Root cause in
the log: `KV cache exhausted: no free blocks` ERROR-spamming every ~180 ms
(1,074 hits) from `mtp_bootstrap_step` — with unique prompts every request
populates NEW radix blocks, and this branch lacked every one of the
prefix-cache refcount/eviction fixes, so the pool wedges. The cached bench
could never see this (one shared prompt = one block set).

Reading the deltas:
* decode/balanced_short: the honest cost of removing cache flattering is
  ~10-17% at C=16. Against #379's published vLLM bar (98.8 @C=8 / 168.9
  @C=16, treated as FIXED per direction — we do not re-measure vLLM):
  decode_short is 108.5/98.8 = **1.10x at C=8** and 138.0/168.9 = **0.82x at
  C=16**. C=16 remains the gap; C=8 is at parity honestly.
* prefill_short -50%: with real prefills, TTFT p50=p99≈18.8 s at C=16 — the
  aggregate is TTFT-dominated (only 128 out-tokens) and prefill admission is
  the bottleneck, not decode.
* PRIOR C=16 CLAIMS (§6.1 and #379's table) carried the cache flattering;
  quote 6.2 numbers from now on.

### 6.3 KV-lifecycle fixes brought onto the branch (2026-07-29)

Cherry-picked, in order, after 6.2 exposed the wedge:

* `493df3ea` (= wip-laguna-lora d27ec6fd, = #373's first commit): stop
  exhaustion crashing CUDA-700 + wedging the pool. VERIFIED on its own
  binary: balanced_long C=8 recovered 21.9 (drops) → **94.6 tok/s, 16/16**.
  Exhaustion ERROR lines still fire (~1000/run) — pressure is real, the pool just
  no longer wedges.
* #375 complete (`2fe52169`, `000c9ba2`, `d36a54b8`): preempt on decode-time
  and prefill-chunk KV exhaustion instead of failing the batch; make the
  chunk-0 prefix lookup idempotent under preempt-retry. One additive struct
  conflict union-resolved with the d27ec6fd fields.
* #373 complete (`75d1398e`..`52222db9`, 8 commits): re-inc leak fix, the
  InsertAcquired API (cache ref follows the radix NODE's block, not the
  sequence's), HSS slid-window mis-filing, keep-evicting-until-alloc-succeeds
  (this is the one that should quiet the 1000 ERROR spam), partial-suffix
  block ownership, reclaim-from-prefix-cache before swap-out. Ported around
  this branch's #381/#382 prefix_cache layout (`no_caching.rs` module,
  tier methods); compile-clean. **Runtime validation on the full-family
  binary still pending** — rebuilt but not yet benched.

### 6.4 Scheduler/context A/B at C=8/16 (IN FLIGHT at write time)

fifo + 4K max-seq-len (the #379 config-of-record shape) vs 6.2's slai/16K,
same honest bench, binary at `493df3ea`. Preliminary (first two regimes):

| regime | fifo/4K C=8 | slai/16K C=8 | fifo/4K C=16 | slai/16K C=16 |
|---|---|---|---|---|
| decode_short | 78.8 | 108.5 | **90.6** | **138.0** |
| balanced_short | 90.3 | 90.2 | pending | 123.1 |

**slai beats fifo decisively in the decode-heavy regime** (TPOT p50 110 ms
vs 92 ms at C=16) — the phaseA "slai starves prefill admission" note argued
for fifo in throughput sweeps, but that reasoning only bites where PREFILL
is the bottleneck. Waiting on prefill_short/balanced_long halves before
concluding; the emerging picture is regime-dependent scheduling, not a
global fifo win.

### 6.4b Scheduler A/B FINAL: slai wins all four regimes

fifo/4K completed: prefill_short C=16 50.8 (vs slai 63.8), balanced_long
C=16 78.6 (vs 93.1-93.9 on the KV-fixed binary). Even the prefill-heavy
regimes — the ones the phaseA "slai starves prefill admission" note said
fifo should win — prefer slai on this branch. **Keep slai.** (Confound
disclosure: the fifo arm also carried 4K max-seq-len; given fifo lost by
20-45% everywhere, the ordering is safe even if the split between the two
variables is not measured.)

### 6.5 PR-survey verdict (what else is worth bringing in)

Checked content-level (not merge-base) against this branch:

* **Absorbed / superseded — do not pick:** #332 (lm_head batched-GEMV
  tiering + batched multi-seq FFN default: branch's copies are NEWER, with
  measured MIN_N=5 crossover data #332 lacks; only its `decode_b2`
  extension has no equivalent, and that is the B-model family, not the 27B),
  #330 (its w4a16_gemv kernel state is BEHIND this branch's #366/#369
  lineage), #352 (drafter-context defaults landed via #356), #266 (batched
  QKV tiling largely subsumed; 415-line rewrite of a file both branches
  touched — re-derive, don't pick).
* **Model-specific, irrelevant here:** #380 (deepseek-v4), #296 (Holo GDN).
* **Brought in today:** logits-aliasing fix (wip-laguna-lora 1e85cb94),
  d27ec6fd + full #373 + full #375 (above).
* **Still open elsewhere:** #372 (fp8-KV calibration; only matters for fp8
  KV configs — this config runs bf16 KV).

### 6.6 Speculative on/off at C=16 (full-KV binary, slai/16K) — SPEC WINS

Single-variable A/B, only the `--speculative --num-drafts 3
--mtp-quantization bf16` flags dropped:

| C=16 regime | spec ON | spec OFF | spec delta |
|---|---|---|---|
| decode_short   | **125.2** (repeat 123.8) | 96.7 | +29% |
| balanced_short | **122.0** | 93.6 | +30% |
| prefill_short  |  **63.4** | 57.4 | +10% |
| balanced_long  |   93.1 | 93.1 | tie |

The "verify is 1.77x a plain step so spec might lose at C=16" hypothesis is
**REFUTED** — #379's batched verify + K-ladder keep MTP a large net win at
C=16. Do NOT wire a C-dependent auto-disable.

Two side findings:
* **All KV-exhaustion pressure is MTP's.** Spec-OFF ran with ZERO
  `KV cache exhausted` errors; spec-ON logs ~1.1K (down from 10.6K
  pre-#373). The draft/bootstrap KV appetite is the exhaustion driver —
  worth a targeted look at what run_mtp_propose_batched allocates per step.
* **The KV-family stack costs ~9% on decode_short C=16**: pre-KV binary
  138.0 (1 rep) vs full-KV 125.2/123.8 (2 reps), same config. block_trace
  is ATLAS_KV_TRACE-gated so it is not that. Needs a bisect inside the 12
  KV commits (suspects: keep-evicting retry loop under MTP pressure,
  InsertAcquired per-insert allocs, node-ref changes) — or a pre-KV rebuild
  rep to rule 138.0 an outlier. The trade today is unambiguous (balanced_long
  went from BROKEN to 93 tok/s) but the 9% should not be silently accepted.

### 6.6b Second wave of picks (2026-07-30, late session)

Prompted by a survey of `wip-laguna-lora` / `port/lora-moe-avarok` for
transferable work (user-directed):

* `a1d889f2` (= 4046dcad, LoRA-branch peel-off): **rayon host sampling**
  (ATLAS_PARALLEL_SAMPLE, default ON, n>1 only) + token-major/exact-N MoE
  decode. The sampling half is the 27B-relevant piece: this branch sampled
  16 sequences SERIALLY over the ~250K vocab per decode step at C=16. The
  MoE half only touches the qwen3 MoE family (35B) — the 27B is dense-FFN.
  One both-sides-add conflict union-resolved next to the think-ended
  GPU-argmax helper; the par_iter body composes behind the GPU-argmax fast
  path (greedy rows go to GPU first, the host remainder now fans out).
* `7c6f3845` (= b3e6b4fe): charge the Q12 batched arena by STAGED tokens,
  not raw chunk len — admits more streams per batched-prefill cohort when
  prefix hits shrink the real work.
* `7bead3d6` (= 02c6ea37): reject zero-token streams from the Q12 cohort.
* `c08b3d1b` (= d13538cf, ported): **cohort KV pre-flight** — compute
  the whole batch's block need (via read-only `peek_matched_tokens`),
  evict-loop to cover it, and DECLINE the batched path pre-mutation when it
  cannot, so one stream's exhaustion no longer fails all N cohort-mates.
  Port note: this branch lacks laguna's `KernelBatchResult::NotAdmitted` and
  up-front radix reservation — decline is a pre-mutation `bail!` that
  batch.rs already routes to the per-stream fallback, where #375's
  preempt-and-retry lives. The reservation half was deliberately NOT ported.

A/B in flight at write time: all four regimes C=16 rayon-ON, then
`ATLAS_PARALLEL_SAMPLE=0` control on decode_short. Baselines to beat:
decode_short 123.8-125.2, prefill_short 63.4, balanced_long 93.1.

Surveyed and NOT picked (with reasons): laguna's `dd370b5e`/`cf5c4765`/
`66a309b2` are earlier versions of the #373 content already here (the tests
commit `66a309b2` is cheap insurance worth adding later);
`a21d1dd4`/`a0092429`/`ec1a74f8` (prefix-cache soundness / cold-warm
numerics) are correctness picks for a quality pass, not throughput;
`990c1537`/`8aa4f401`/`9e66c6b9` (tool-call parsing) help agentic
scorecards; `3cf579a6` (gated_rms_norm atomicAdd determinism) touches
Nemotron kernel dirs only.

### 6.6c A/B results for the second wave + the K4 ladder analysis

C=16, slai/16K, spec ON, one binary (all second-wave picks), single reps:

| C=16 regime | pre-wave | rayon ON | serial control | read |
|---|---|---|---|---|
| decode_short | 123.8-125.2 | 131.3 | 129.4 | **trio +~5%; rayon ~neutral** |
| balanced_short | 122.0 | 121.4 | — | unchanged |
| prefill_short | 63.4 | 63.3 | — | unchanged |
| balanced_long | 93.1 | 93.5 | — | unchanged |

**Why rayon is neutral here (structural, not a bad port):** with spec-decode
ON, C=16 decode flows through `verify_k4_step`, not `decode_logits_step`
where ATLAS_PARALLEL_SAMPLE lives. The rayon win applies to the NON-spec
path (and to the 35B MoE it was measured on). Keep it: it costs nothing
here and protects the spec-off path (96.7 in 6.6 would presumably improve).
The decode_short recovery 124→~130 attributes to the prefill trio
(cohort pre-flight + arena-by-staged-tokens), clawing back roughly half of
6.6's ~9% KV-stack cost. Exhaustion ERROR lines unchanged (~1.1K, MTP-driven).

**K4 acceptance at C=16 — "only good at first token" is BY DESIGN** (user
observation, confirmed): the default K-ladder is `8:3, 16:1`, so at n<=16
every sequence gets ONE draft — `0 accept-2 / 0 accept-3` is structural.
Observed p1 at C=16: 0.74-0.90 per 100-step window (better than the 0.72
#379 quotes). The ladder's own doc records why depth is dead at n=16:
16:1 = 152.01 vs 131.40 MTP-off (+15.7%) while 16:3 = 120.76 (WORSE than
MTP-off), because the 32-row verify budget is exactly consumed by 16 x 2
mandatory D-Cut rows. The two real levers, in order:
1. **Raise p1** (expected tokens/step at K=1 is `1 + p1`; refeed reached
   0.90 on strix — worth porting/attempting here).
2. **Grow the verify row budget past 32** so `16:2` becomes physically
   testable (same `min(m,32)` logits-arena rows the aliasing fix
   bounds-checks) — a memory/latency trade needing its own ladder matrix.

Also picked this wave (user-directed, correctness for the quality pass):
`9498edd7` (bypass decided before the SSM restore), `68c02fc0` (cold-vs-warm
prefill numerics), `cd163142` (exact-full-prompt shortcut default-OFF — it
is unsound by construction; `ATLAS_MARCONI_EXACT=1` re-enables), `b37669b5`
(insert-contract tests). Ports kept this branch's newer tail-only session
gate + `marconi_min_tokens` threshold with the picked semantics layered in.

### 6.6d The MTP drafter KV pool was sized for ONE sequence — fixed, +10-33% at C=16

Root cause of every `KV cache exhausted` ERROR left after #373/#375: the
drafter's own 1-layer paged pool (`mtp_head/new.rs`) was
`max_seq_len/block_size + 1` blocks — 1,025 on the 16K config, sized before
MTP propose went batched. 16 drafting seqs x ~2K tokens needs ~2x that,
while the 15K-block MAIN pool never fills on a 1K workload. Every hit
degraded batched propose to the per-step fallback. Fix (`bf2fec8a`): scale
by `mtp_max_seqs`, capped at the main pool's block count (drafter can never
need more live tokens than main KV holds). 0.9 GB at this config.

Validation (C=16, slai/16K, spec ON, single reps):

| regime | before | after | delta |
|---|---|---|---|
| decode_short | 131.3 | **144.3** | +10% |
| balanced_long | 93.5 | **123.9** | **+33%** (TPOT p50 148.1 -> 98.9 ms) |
| exhaustion ERROR lines | ~1,100 | **0** | |

This was the largest single C=16 lever found today. Honest decode_short
now 144.3 vs the 168.9 bar = 0.85x (was 0.73-0.78 at session start).
Note TTFT drifted up in both cells (5.5->6.8 s, 18.8->21.1 s) — plausibly
the drafter now prefilling contexts it previously failed to admit; watch
in the next full sweep rather than assuming noise.

### 6.7 Where C=16 parity stands after today

Best honest config (slai/16K, spec ON, full-KV binary), vs the published
vLLM bars (treated as fixed; NOT re-measured):

| | C=8 | C=16 |
|---|---|---|
| Atlas decode_short | 108.5 | 124-138 |
| vLLM bar | 98.8 | 168.9 |
| ratio | **1.10x — parity MET honestly** | **0.73-0.82x** |

The C=16 gap decomposes as: verify-step cost + accept rate (the #379 "Still
open" arithmetic: verify needs <=1.34x a plain step, measured ~1.77x; p1
accept ~0.72 vs 0.90 demonstrated on strix via refeed) for the
decode-heavy regimes, plus prefill admission (TTFT p50=p99=18.8 s walls in
prefill_short) for the prefill-heavy ones.

### 6.8 The prefill wall: full gate-chain autopsy + the real bottleneck

prefill_short C=16 sat at ~63-65 tok/s (TTFT p50=p99 ~17-19 s) through every
config change. Root-caused in layers, each verified by a serve+bench probe
(all single-cell C=16, prefill_short, unique prompts):

1. **1K prompts are single-chunk** → chunk-0 runs as a monolithic inline
   prefill in `phase_start_prefills`, one per tick. 16 concurrent cold
   prefills SERIALIZE. Q12 batched dispatch: provably never invoked
   (validated debug filter: scheduler DEBUG lines present, zero q12 events).
2. `ATLAS_PREFILL_CODISPATCH=1` defers chunk-0 into a cohort → cohort
   ENTERS dispatch (2 entries/run) but falls to per-stream: 65.1 (flat).
   Gate chain peeled from the bail logs:
   a. arena cap: 16x950=15,200 stacked tokens vs arena 4,112 (=
      max-prefill-tokens) → raise to 16,384 for the probe;
   b. first-chunk gate: satisfied by CODISPATCH (either-flag helper);
   c. **prefix-cache blanket gate** (fix #4 in eligible.rs): kernel-batched
      co-dispatch is disabled OUTRIGHT whenever the radix cache is active —
      rationale "with caching on most requests hit", which is the CACHED
      bench worldview; honest/enterprise traffic is all-cold. Probed with
      caching off →
   d. **admit/execute gate MISMATCH** (same species as §5.1): eligibility
      admits chunk-0 on `CODISPATCH || FIRST_CHUNK` but
      `prefill_inner` only honors the explicit `ATLAS_Q12_BATCHED_FIRST_CHUNK`
      → cohort admitted, bails at layer 0 ("requires seq_len_start > 0"),
      62.2 (flat). With the explicit flag →
3. **kernel-batched SUCCEEDED end-to-end — and throughput did not move**
   (63.2). One batched forward over 15,200 tokens costs the same ~18 s as 16
   serial 950-token prefills. Scheduling was NEVER the wall.
4. **The wall is the 27B's cold-prefill compute rate: ~800-900 tok/s flat**
   (even C=1: 1.3 s TTFT for 1K tokens). A 27B dense-ish hybrid needs
   ~54 GFLOP/token, so ~800 tok/s ≈ ~43 effective TFLOPS — the BF16-class
   arms are near their practical roofline on GB10. Arm A/Bs at C=16:
   * default (int8/FP8 arms, MMQ off): 63.2
   * + `ATLAS_BF16_TC_PREFILL=1` (v2, fixed this morning; 0 launch errors
     under load): **53.4 — SLOWER.** The flag buys losslessness, not speed;
     `c9965ce9` shows `NO_FFN_NVFP4_MMQ=1` was added to the golden set
     PURELY to let BF16_TC engage — when BF16_TC was dropped (crash), the
     MMQ disable stayed behind with its reason gone.
   * MMQ re-enabled (drop `ATLAS_NO_FFN_NVFP4_MMQ=1`): **69.7**, best TTFT
     (16.9 s) and TPOT (92.1 ms). +10%, W4A4 is LOSSY — needs the
     oc_harness quality gate before adoption.
   The remaining ~16 s is the non-FFN prefill cost — on this SSM-hybrid the
   prime suspect is the SSM-layer prefill scan (cf. the Holo measurement:
   Atlas chunk_delta_h 11-13x slower than FlashInfer GDN at matched shape).
   Needs an nsys decomposition to confirm; the structural fix is the
   FI-GDN-class kernel program, not a scheduler or config change.

Actionable follow-ups extracted:
* fix the admit/execute first-chunk gate mismatch (align both on
  `first_chunk_batched_enabled()`);
* replace the prefix-cache blanket gate with all-cold cohort admission
  (read-only `peek_matched_tokens` across the cohort — same probe the
  ported KV pre-flight already uses);
* codispatch window default 10 ms -> 100 ms (DONE this session — a burst
  spreads over tens of ms; 10 ms co-admitted only the first arrivals);
* decide the MMQ-on question with a quality gate;
* m128 kernel program (ultracode, 2026-07-30): 11-agent workflow designed
  the port; executed same session. OUTCOME — **honestly-buried lever**: the
  proven minimax/step3p7 8-warp m128_v2 ports cleanly (bit-identity 100% on
  8 shapes) but measures **0.78-0.82x of v1** on the 27B FFN shapes — the
  judges' realism critique ("the v2 file ships zero measured numbers")
  called it. Shipped as opt-in only (`ATLAS_W4A16_VARIANT=v2`), commits
  74eca5a0 + 195a69c6, WITH two durable wins regardless: the PTX arity CI
  test (pins launcher-vs-kernel param counts for the whole w4a16 family —
  the 5.1 bug class is now a CPU test failure) and the single-resolver
  kill-switch. Remaining prefill-GEMM levers: widen W4A4 MMQ past
  gate/up/down (qkvz/o still on the slow arm), or NEW kernel engineering
  (not a port) — the existing v2/v3 schedules do not transfer to these
  shapes.
* nsys the 1K cold prefill to size the SSM-vs-FFN-vs-attn split.
  **DONE (2026-07-30): the split OVERTURNS the SSM hypothesis.** Cold
  717-token prefill, GPU busy 1084/1270 ms wall: w4a16 tile GEMMs 60.9%
  (w4a16_gemm_t_m128 ALONE 52.1%, ~1.86 ms/launch ~= ~90 effective TFLOPS
  at the FFN shape), GDN/SSM 15.6%, dense BF16 8.3%, FP8 6.1%, attention
  0.6%. The prefill wall is the W4A16 GEMM family. Levers: widen W4A4 MMQ
  coverage past gate/up/down (the measured +10% left qkvz/o and residual
  m128 on the slow arm; needs quality gate), m128 kernel efficiency (v2/v3
  pipeline playbook exists for other targets), int8 W4A8 middle ground.
  Report: /home/ms/.claude/jobs/c91b191d/tmp/prefill_1k.nsys-rep.

### 6.10 Canonical-4K baseline of record + the decode-rollback ring reclaim

Full stack (0e8f15b7: MMQ on, KV family, drafter pool, aliasing, picks),
canonical 4K config, honest bench, single reps unless noted:

| regime | C=8 | C=16 | vs vLLM bar |
|---|---|---|---|
| decode_short   |  99.6 | 150.8-161.6 (3 reps, median ~151) | C=8 **1.01x**; C=16 **0.90-0.96x** |
| balanced_short | 101.8 | 133.3 | |
| balanced_long  | 104.9 | 140.2 | |
| prefill_short  |  56.4 |  66.9 | compute-walled (6.8) |

Zero KV-exhaustion errors. TPOT p50 82-84 ms at C=16 (was 91-95 on the 16K
config). Session start was 0.73x at C=16; the remaining gap is ~4-10%
depending on rep. NOTE the decode_short C=16 single-rep spread is ~7% —
quote the median, use >=3 reps for any future claim.

**Decode-rollback ring: ~20.3 GB reclaimed (user-directed).** The ring's
only writer/reader live on the PLAIN decode path (spec does rejection
rollback via the verify snapshot), and the reader additionally needs
watchdogs enabled — under the production spec-ON config the ring was
structurally unreachable yet allocated 8 slots x 16 seqs x the 27B's
158.9 MB blob up front. impl_a1 now skips it when `use_speculative` or
watchdogs are disabled (`ATLAS_SSM_DECODE_RING=1/0` force-overrides;
skip logs the GB saved; rollback fail-opens to decline as documented).
Verified live: startup logs "SKIPPED ... Saves 20.3 GB"; KV pool grew
15,203 -> 17,355 blocks. Perf-neutral on decode (TPOT unchanged-to-better
across 3 cells).

### 6.11 Verify-step cost research (user-directed): decomposition + every config lever measured

Method: two nsys profiles, identical 16-way 300-token decode burst, spec-ON
vs spec-OFF, same binary/config (canonical 4K). The kernel-sum diff IS the
verify tax: **+2,078 ms GPU (4,288 vs 2,210) for the same tokens.**

| component | dt ms | share | note |
|---|---|---|---|
| dense_gemm_bf16_pipelined (960 x ~1.03 ms) | 991 | 48% | the DRAFTER forward (--mtp-quantization bf16) |
| w4a16_gemv_batch16 (159 x 4.25 ms) | 676 | 33% | verify lm_head; streams 636 MB at ~150 GB/s vs 230 roofline |
| drafter per-seq launch chain (~16K launches: 5,200 gemv + 2,555 attn + 6,146 norms) | ~350 | 17% | per-seq loops; the propose GEMMs are batched, the rest is not |

Config levers — ALL measured, ALL settled (decode_short C=16, 2-3 reps,
baseline median ~151 / TPOT 82 ms / p1 0.867):

* `16:2` ladder rung: **BURIED** — 103-108 (-29%). Depth WORKS (mean
  accepted 1.43-1.53 = the p1*p2 projection) but the 32-row verify forward
  is row-linear: +31% tokens at ~+75% step cost. Depth at n=16 is now
  exhaustively dead: 16:1 (151) > 16:3 (120.8, #379) > 16:2 (107).
* `--mtp-quantization fp8`: **BURIED** — 123.8/125.7, TPOT 98-99 ms.
  Acceptance only dipped 0.867->0.827; the fp8 drafter path simply runs
  slower kernels than the pipelined BF16 GEMM. #379's bf16 pick was for
  SPEED, not caution.
* `--lm-head-dtype fp8`: **NEUTRAL** — 149.6/151.1, acceptance 0.866. The
  batched verify head already runs #378's FP8-ACTIVATION path; this flag
  rewires a path verify does not use.
* `ATLAS_MTP_REFEED_ACCEPTED` (mined from #378 — the p1 0.72->0.90 strix
  lever, implemented and never set in any config): **STRUCTURAL NO-OP at
  C=16** — refeed rebuilds drafts 2..K only; at the 16:1 rung draft 1
  already consumes the target's verified hidden. Applies to the C<=8 deep
  rungs; worth an A/B there, not here.
* Acceptance itself is NOT the problem (user hypothesis checked): p1 at
  C=16 is 0.867 vs 0.82 at C=1 on the same workload — the C=1 advantage is
  purely depth (~2.7 tokens/forward at K=3 vs 1.87 at K=1).

#378 mining (user-directed): drafter-context blindness fix and the batched
bootstrap/argmax are content-present in our base; the accept telemetry
(`ATLAS_MTP_ACCEPT_DEBUG`) is available; **"graphing the Phase-A bootstrap"
was declared WIP in #378's body and never landed anywhere** — it is a real
open lever.

RANKED CODE LEVERS (the config space is exhausted):
1. Drafter forward (48% of tax): graph-capture the batched propose and
   batch its per-seq attention chain (the ~16K-launch disease the main
   model's decode already had cured by #379).
2. Verify lm_head (33%): M=32-tuned head kernel — the batch16 GEMV runs at
   ~65% of the memory roofline at this row count; ~1.5x headroom.
3. Phase-A bootstrap graphing (#378's unlanded WIP).

Artifacts: verify_specon/specoff.nsys-rep + kern_*.csv in the job scratch.

### 6.12 LOOP iteration 1: the C=16 decode wall is HOST-SIDE (readbacks + allocs)

Lever-1 slice (drafter small-N K/V projections: per-row GEMV -> tile GEMM at
m>=8; the old M=4 crossover was stale after a83627a2 widened n to 16):
**NEUTRAL e2e** — 146.8/149.2/150.7 (median 149.2) vs baseline median ~151.
Kept (kill switch ATLAS_MTP_KV_GEMV=1): launch-count reduction, no cost.

Computing WHY it was invisible produced the discovery that reframes the
whole C=16 decode program: the spec-ON burst shows only **4.3 s of GPU-busy
across ~25 s of decode wall** — the step is ~80% host-bound. CUDA API sums
(existing 6.11 profiles):

| | spec-ON | spec-OFF |
|---|---|---|
| cuMemcpyDtoHAsync total | **25.6 s / 4,439 calls (~27/step, avg 5.8 ms)** | **40.1 s / 1,848 calls (avg 21.7 ms — full 7.9 MB logits/step)** |
| cuMemAlloc AT RUNTIME | 3.5 s / 4,580 calls | 4.2 s / 4,466 calls |
| cuStreamSynchronize | 2.0 s / 18,094 | 0.4 s / 7,604 |
| GPU kernel busy | 4.3 s | 2.2 s |

Decode in BOTH modes is serialized on blocking device-to-host readbacks
(pageable D2H syncs the stream) and per-step allocations. Kernel-time
levers (6.11's ranked list) cap at ~+6% e2e by Amdahl; THIS is the 12%.

NEXT (loop iteration 2): convert the per-step readbacks to a reusable
PINNED staging pool (#381's proven pattern). Site inventory (initial):
verify_e.rs:470 Phase-5 argmax readback — fresh pageable `vec!` per step,
self-described "the step's one host sync"; decode_a2.rs:130 per-seq logits
pulls (n>1 eager loop) + :359 per-row 16-B probes after a full sync;
drafter forward_batch LP/argmax scratch reads; ~27 D2H/step total. Also
hunt the ~4.5K runtime cuMemAllocs (device-side, per-step). Deeper rung if
pinning alone is not enough: overlap next-step launch with the readback
(double-buffered step).

### 6.13 LOOP iteration 2: grammarless verify fast-greedy — +9%, C=16 at 0.973x

The 6.12 histogram pinned the wall: 992,308 B x 2,541 DtoH = each sequence's
[2, vocab] BF16 verify rows copied to host EVERY step for the pre-sample
pipeline. The #237 fast-greedy arm skips exactly this — but was gated on
`grammar_state.is_some()`, so grammar-bound sequences got the GPU-argmax
fast path while plain chat (the bench, most enterprise greedy traffic) paid
16 blocking ~1 MB stream-drain waits per step.

Fix: the grammarless sibling of the #237 gate — same eligibility (greedy,
not in thinking, penalties Neutral or ReduceOnly with the same per-token
immunity proof + 2-byte probe), no bitmask to consult, GPU argmax IS the
pick, zero D2H. Same tie-breaking trade #237 already shipped for grammar.
Kill switch: ATLAS_NO_FAST_GREEDY_CHAT=1.

decode_short C=16, canonical 4K: **164.4 / 164.2 / 164.6 (median 164.4)**,
TPOT p50 78.8-79.2 ms, coherence smoke clean. Session arc: 0.73x -> 0.90x
-> **0.973x** of the 168.9 bar. Remaining ~4.5 tok/s: the D2H tail
(per-request 44.5 MB x 12 + 41.9 MB x 3 copies — ~13 GB per 16 requests,
unidentified, likely SSM/drafter lifecycle), ~4.5K runtime cuMemAllocs,
drafter launch chain.

### 6.13b Full C-ladder on the current stack (single reps, user-requested)

decode_short, canonical 4K, binary at 8075c2c6:

| C | Atlas | vLLM bar | ratio | #379 published |
|---|---|---|---|---|
| 1 | 30.2 | 14.2 | **2.13x** | 25.6 |
| 2 | 44.0 | 27.8 | **1.58x** | 35.2 |
| 4 | 72.9 | 53.3 | **1.37x** | ~57.0 |
| 8 | 117.8 | 98.8 | **1.19x** | 88.9 |
| 16 | 163.9 (median-of-4 family: 164.2) | 168.9 | 0.97x | 131.9 |

Every level gained 18-33% over #379's published numbers from today's stack
(KV family, drafter pool, prefill trio, MMQ, fast-greedy-chat). FOUR OF
FIVE levels now beat vLLM outright; C=16 is the last, ~5 tok/s short.

### 6.14 NEXT LEVER DESIGN (loop iteration 3): verify-step pipelining

Observation driving the design: removing ~90 ms/step of D2H wait (6.13)
moved TPOT only 82 -> 79 ms — the waits were OVERLAPPED with other host
serialization. The step wall (~148 ms at 1.87 tok/forward) is a serial
chain: verify forward (GPU) -> Phase-1 host verdict loop -> stash ->
per-seq verdict/propose -> re-propose, GPU idle through every host phase.

Design — overlap verify N+1 with verdict N:
* Double-buffer the logits/verify arena using the row_base machinery the
  aliasing fix introduced (alternate two row windows per step).
* Launch step N+1's verify forward as soon as step N's GPU argmax lands;
  the drafts for N+1 depend only on N's accepted tokens (already on GPU),
  not on the host verdict/emit bookkeeping.
* Serial fallback + kill switch; grammar/logprobs/temp>0 stay serial.
* Hazards: the documented shared-buffer clobber set (mtp_multi.rs:165) —
  the double buffer exists to break exactly those.

Prerequisite measurement (queued): fresh profile of the CURRENT binary
with --cudabacktrace to (a) re-derive the phase-gap timeline post-6.13 and
(b) attribute the weight-shaped 42.5 MB D2H copies (predequant-adjacent,
~13 GB/burst, still unexplained). Design adjusts if the gaps moved.

### 6.15 LOOP: the C=16 step anatomy is now exact — one 130 ms stall left

In-process phase timers (ATLAS_MTP_TIMING now wired into the batched K4
step) + the backtraced trace agree on the per-step structure at C=16:

* GPU: **idle ~140-152 ms -> dense 11.1 ms kernel span (145 kernels, 10.7
  busy — the verify graph replays with NO internal bubbles) -> argmax**.
* Host: launch region 0.16 ms; then **d2h_wait 126-134 ms** blocked in the
  stream sync of a 128-320 B argmax readback; propose 6-10 ms; verdict/emit
  ~0.3 ms; Phase-1 pipeline ~0 (the 6.13 fast-greedy holding).

Buried this iteration with evidence:
* cuMemAlloc-per-step theory: ZERO in-window allocs (all 9,160 were load).
* Default-stream entanglement theory: switching Phase-5 to
  `copy_d2h_on_stream(stream)` (semantically the right dependency; kill
  switch ATLAS_VERIFY_D2H_DEFAULT_STREAM=1) reproduced the SAME 130 ms.
* Host-bookkeeping theory: those phases total <12 ms.

Remaining question, exactly one: why does a LAUNCHED verify graph not
START for ~130 ms while the host blocks in the sync? Discriminator queued:
cuGraphLaunch runtime timestamp vs first kernel start in a fresh
containerized trace — (a) gap after launch = GPU-side dependency (event
edge signaled late, capture wiring); (b) late launch = the timer story is
incomplete somewhere upstream. Also note "Captured CUDA graph" fired 42x
in one burst — capture CHURN (key = ssm-slot vector + ks) is a suspect:
a capture mid-stream serializes and may leave cross-stream event edges.

### 6.16 LOOP CONCLUSION: the last 130 ms/step is a PLATFORM stall — code levers measured-dead against it

The burial ledger for the ~130 ms/step verify stall (each theory killed by a
direct experiment, all kill-switched arms retained in-tree):

| # | hypothesis | experiment | result |
|---|---|---|---|
| 1 | pageable D2H staging | pinned staging buffer | 130 ms unchanged |
| 2 | default-stream entanglement | copy_d2h_on_stream(verify stream) | unchanged |
| 3 | submission-ring flush | cuStreamQuery spin instead of blocking sync | unchanged |
| 4 | CUDA-graph replay quirk | ATLAS_NO_MTP_VERIFY_GRAPHS (eager) | 119-121 ms (worse agg) |
| 5 | GPU power-gating | external 3 ms keep-awake kernel pinger | WORSE (142 ms) |
| 6 | copy-engine pickup of tail D2H | MAPPED-ARGMAX (kernel writes host-mapped pinned; no copy op at all) | unchanged |
| 7 | per-step device allocs | trace: zero in-window cuMemAllocs | n/a |
| 8 | dashboard as second CUDA client | nvidia-smi: /spark is the ONLY compute app | n/a |
| 9 | MPS de-quantization | user-scope MPS daemon | WORSE (3.8 ms avg) |

**Platform characterization (the decisive artifact):** a 10-line nop-kernel
launch+sync client (`katimed.cu`, job scratch) on an OTHERWISE IDLE GB10:
avg 1.07 ms, worst 208 ms per cycle (healthy platforms: ~5 us). During
decode: 4.6 ms avg, worst 438 ms. Driver 580.173.02 open kernel module,
aarch64. The serve's per-step ~130 ms is this platform behavior expressed
at step cadence; the trace shows queued work (kernels AND copies) sitting
~130 ms on an idle device before pickup, then executing in a dense burst.

**Standing:** C=16 decode_short median ~164 (0.973x of the 168.9 bar; four
of five ladder levels beat vLLM outright). The remaining ~4.5 tok/s is
inside the platform stall — with a healthy launch/sync path the measured
step would be ~25 ms and C=16 would clear the bar several times over,
which also implies vLLM's own 168.9 on this box ate the same stall.

**Recommended next actions (not code):** escalate to the driver/platform
channel with katimed.cu as the repro (assert: idle-box launch+sync avg
1 ms / worst 200 ms on 580.173.02); retest on newer driver drops; compare
against the cuda-compat-13-2 stack the FI-GDN work already uses. The
verify-path plumbing landed this session (mapped argmax, pinned staging,
on-stream copy, spin probe, telemetry) is all default-safe and becomes
free upside the day the platform stall is fixed.

### 6.17 Agentic concurrency (conc_harness) — 27B vs the fleet

64K config, full stack, single rep (levels 1/4/8; pass-rate caveat per s4):

| model | agg tok/s C=1/4/8 | pass | makespan C=8 |
|---|---|---|---|
| qwen3.6-27B (this branch) | 17.0 / 24.4 / 25.1 | **1/1, 4/4, 8/8** | 346.6 s |
| Holo 35B-A3B (2026-07-27) | 42.8 / 53.2 / 53.5 | 0/1, 3/4, 4/8 | 194 s |
| Laguna-XS (2026-07-27) | 21.1 / 45.4 / 59.3 | 1/1, 4/4, 8/8 | 133 s |
| Laguna-S (2026-07-27) | 11.0 / 25.1 / 46.0 | 1/1, 3/4, 8/8 | 210 s |

13/13 tasks passed; KV ledger CLEAN over the whole run (0 exhaustions,
0 decref, 0 unowned-evicts, 0 preempts — the day's KV/drafter work visible
end-to-end). One repeat-loop cut at C=8 (task still passed). Speed is
Laguna-S-class — expected dense-27B physics vs A3B actives — but the
C=4->C=8 aggregate FLATLINE (+2.9%) contrasts with the synthetic bench's
healthy scaling to C=16: agentic contexts run 10-20K tokens, past the
~7.2K batched-propose fallback (5.2). **The 5.2 propose_meta sizing fix is
now the top agentic-performance lever**, ahead of everything else in 7.

## 7. Open

Ordered by what I would pick up first.

0. **C=16 @ 1K to vLLM parity (168.9)** — the active mission. Honest position
   after 6.2: decode_short 138.0 @C=16 (0.82x). Next levers, in order:
   (a) finish the 6.4 scheduler A/B and keep slai unless prefill regimes
   flip it; (b) validate the full KV-family binary (6.3) — the
   keep-evicting fix may recover MTP-batched propose under pressure and
   with it decode speed; (c) speculative on/off at C=16 (verify step
   ~1.77x plain decode at n=16 vs the <=1.34x bar #379 derived) — if OFF
   wins, wire a C-dependent auto-disable; (d) #379's own "Still open"
   items: Phase-A bootstrap graphing, accept-rate (p1 ~0.72, refeed
   reached 0.90 on strix).
1. ~~**§5.1 fix 2 — why does `w4a16_gemm_t_m128_bf16_v2` fail to launch?**~~
   **DONE (2026-07-29):** missing 9th `ldb` arg — see the RESOLVED block in
   §5.1. Both fixes landed AND e2e leg-B REVALIDATED: v2 path serves with 0
   launch errors, output identical to v1 (55 tok vs control 35).
2. **§5.7 — add a per-request nonce to `make_prompt`**: DONE (2026-07-29) —
   `make_prompt` now embeds a per-request nonce and is called per request, not
   per cell. All PRIOR TTFT columns (incl. §6.1 and #379's table) predate this
   and measured cache hits. Still open: re-check whether `prefill_long` really
   produces 0 meta-stride hits now that prefill is cold.
3. **§5.2 — size `propose_meta` from `max_seq_len`** instead of a fixed 2048, and
   demote the per-step `ERROR` to a once-per-sequence `debug`.
4. **§5.6 — warn at startup when `ssm_checkpoint_interval * block_size <
   ATLAS_SSM_SPILL_MIN_TOKENS`**, i.e. when the config can never spill.
5. **§5.3 — `ATLAS_SSM_TAIL_PROTECT=1` is inert** under `TAIL_MIDCHUNK=0`. DONE
   (2026-08-05): dropped from every launch script and renamed to the opt-out
   `ATLAS_DISABLE_SSM_TAIL_PROTECT` (default ON, `=1` disables).
6. SSM tier reaping is still unexercised (§5.5). `bench/ssm_faultin.py` is the
   probe; it needs a SMALL resident pool (`--ssm-cache-slots 1..2`) so 2-3
   requests force eviction, NOT the 48-request/21 GB shape I first built.

Deliberately parked: `put_with` (evaluated and rejected in atlas#382 — only 2 of
4 `SnapshotBlobStore` implementors could support it, and it holds the residency
`Mutex` across 60 async enqueues; a better hypothesis is that the 17-19 ms
`store.put` is lazily-faulted calloc pages, measurable as put wall time vs put
ordinal 1..128).

---

## 8. State at handoff (2026-07-29)

**Branch** `perf/enterprise-concurrency-v3` on `avarok`, all commits signed:

```
30839cb7  test(bench): agentic + correctness harnesses
5c9f9672  docs(progress): #379's benchmark cannot reach the meta-stride cap
299c47e5  docs: PROGRESS_LOG
1810854e  chore(ssm-tier): LoC cap
aa233714  fix(ssm-tier): reap dead tier keys            <- atlas#382
6e68fa8b  chore(ssm-tier): clippy + LoC cap
a91390e4  perf(ssm-tier): 22x cheaper eviction          <- atlas#381
2848205c  perf(mtp): D-Cut depth pruning                <- atlas#379 head
```

Worktree `/home/ms/atlas/.claude/worktrees/combo`, tracking the remote.

**Related PRs** (all draft, all green): atlas#381 (spill tier),
atlas#382 (reaping, stacked on #381), atlas-recipes#13 (Holo-3.1-35B recipe,
from a fork — no write access to atlas-recipes). Review comment on atlas#379:
`#issuecomment-5112240223`.

**Local scratch** `/home/ms/.claude/jobs/c91b191d/tmp/`: serve scripts
(`combo_conc.sh` = the §3a config, `combo_27b.sh`, `holo_pr382.sh`,
`v2_bisect.sh`), binaries (`spark-combo` = this branch), and benchmark output
(`conc_baseline_4regimes.txt`, `conc_prefill_long.txt`). NOT in git.

**Environment gotchas that cost time here:**
* `CUTLASS_HOME=/home/ms/cutlass` at build ONLY to make the CUTLASS
  reference path available for benchmarking. It is NOT required to serve:
  the default build sets neither `CUTLASS_HOME` nor `FLASHINFER_HOME`, and
  the binary serves on Atlas's own kernels. (Corrected 2026-08-31 — the
  earlier "or the binary refuses to serve" was never true of `spark-server`.)
* `RUSTFLAGS="-L <dir with libnccl.so>"`; symlink `libnccl.so -> libnccl.so.2`.
* The binary is `target/release/spark`, not `spark-server`.
* Piping a long-running harness through `tail` buffers ALL output until it
  exits — write to a file and tail the file instead.
* One test fails on `wip-laguna-lora` but passes on every PR branch:
  `radix_tree::snapshot::tests::lease::lookup_tiered_tail_session_gate`.
  Pre-dates this work; `cargo test --workspace` is green on the PRs.

## 8.1 Addendum (2026-07-29, evening session)

Commits added on top of the stack above (newest first):

```
52222db9..75d1398e  8 commits: full atlas#373 (prefix-cache refcount family)
d36a54b8..2fe52169  3 commits: full atlas#375 (KV-exhaustion preempt)
493df3ea  fix(kv): exhaustion wedge (= wip-laguna-lora d27ec6fd)
8a07a672  bench: BENCH_LEVELS env
de129a3c  fix(scheduler): mixed-step logits aliasing (= 1e85cb94 ported)
d09d6516  fix(bench): per-request nonce (§5.7 resolved)
79522031  fix(prefill): v2 ldb param (§5.1 resolved + revalidated)
```

Cross-references: findings 6.2-6.5. Cherry-pick port notes: the aliasing fix
was rebuilt around this branch's older prefill_b types (no KernelBatchResult);
#373's InsertAcquired API was rebuilt around this branch's prefix_cache
layout (`no_caching.rs`, tier methods). `cargo check` clean on
spark-runtime/model/server; **full-workspace `cargo test` on the final stack
still pending** — the KV refcount logic deserves the kv_cache + radix_tree +
prefix_cache suites before any PR is cut from this.

Scratch additions: `combo_conc_fifo.sh`, `combo_1k_fifo4k.sh` (fifo/4K serve),
`conc_1k_honest.txt` (§6.2 raw), `conc_1k_fifo4k.txt` (§6.4 raw, in flight),
`v2fix_bisect.log` (§5.1 revalidation). Binary `spark-combo` = `493df3ea`
build at the time of the 6.4 run; `target/release/spark` in the worktree =
full stack (post-#373) — NOT yet deployed to `spark-combo`.

New session-level gotcha: launch every long-running harness with
`python3 -u` (block-buffering hid 40 min of §6.2's output), and never `cp`
over `spark-combo` while a container has it mounted — stop the container
first or the copy fails with "text file busy" and the old binary keeps
serving (§6.4's first attempt was invalidated exactly this way).
