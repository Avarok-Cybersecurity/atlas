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
  -e ATLAS_NO_FFN_NVFP4_MMQ=1 -e ATLAS_SSM_TAIL_MIDCHUNK=0 -e ATLAS_MTP_CATCHUP=0 \
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

### 5.1 `ATLAS_BF16_TC_PREFILL=1` fails kernel resolution — REMOVE IT

In #379's config of record but fatal here:

```
Prefill chunk layer 0 failed: cuLaunchKernel: CUDA_ERROR_INVALID_VALUE (1)
  (grid=[136,3,1], block=[128,1,1], shared_mem=0)
```

Valid grid/block with `INVALID_VALUE` = an unresolved (null) kernel handle, not
bad dimensions. Fires on the first request, at prefill layer 0, in the FLA
chunked GDN path. Scheduler-independent: **slai turns it into a SIGSEGV (exit
139) that kills the process; fifo degrades it to a per-request HTTP 500.** That
secondary fault is worth fixing on its own — a failed prefill launch should not
take the server down.

Dropping the one flag fixes it; everything else in the golden set is fine, and
slai then works. Same class as the `rms_norm_strided` resolution bug #379 itself
fixes (`try_kernel` swallows the miss and returns handle 0).

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

### 5.3 `ATLAS_SSM_TAIL_PROTECT=1` is inert in this config

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

## 7. Open

- §5.6 — reconcile `ATLAS_SSM_SPILL_MIN_TOKENS` with `--ssm-checkpoint-interval`.
- §5.5 — build a warm-reuse probe that actually exercises reaping.
- `put_with` to remove the remaining 66 MB `store.put` memcpy was **evaluated and
  rejected** (atlas#382): only 2 of 4 production `SnapshotBlobStore` implementors
  could support it, and it holds the residency `Mutex` across 60 async enqueues.
  A better hypothesis is recorded there: the 17–19 ms may be lazily-faulted calloc
  pages (66,846,720 B = 16,320 fresh 4 KiB pages ≈ 16 ms), not memcpy bandwidth —
  measure `store.put` vs put ordinal 1..128 before touching the trait.
- Stale-key reaping under sustained cap pressure is still the documented
  degradation mode; see atlas#382.
