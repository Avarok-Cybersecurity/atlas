# Replicating the GB10 edge-agentic submission

Everything needed to reproduce the numbers claimed for `centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf`
on a DGX Spark GB10, end to end, from a clean checkout. Written so that someone who was not
in the campaign can run it.

The decode work itself (chain-widened K=4 verify) landed in #366; this document is the
operational half — the exact build, the frozen serve configuration, the gates that must pass
first, and the traps that cost us real time.

---

## 1. Prerequisites

| what | value |
|---|---|
| hardware | DGX Spark GB10, sm_121a, 121 GB unified memory |
| model | `centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf` (dense, already all-NVFP4 including GDN) |
| harness | MLCommons edge-agentic (`inference-endpoint`), 1007 perf + 995 BFCL |
| container | `atlas-gb10:followups` (any image with the CUDA runtime; the binary is bind-mounted in) |
| determinism | temp 0.0, seed 42 |

The harness checkout must already have a base `config.yaml` to derive from — the campaign used
`results/defaults_20260721_173342/config.yaml`. Any base config works as long as the same one is
used for every leg you intend to compare.

---

## 2. Build

```bash
PATH=/usr/local/cuda/bin:$PATH \
ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.6-27b \
  cargo build --release -p spark-server --bin spark --features cuda
```

**`ATLAS_TARGET_MODEL=qwen3.6-27b` is not optional.** Without it the build defaults to
`qwen3-next-80b` and compiles a different kernel set, so a kernel edit produces an
md5-identical binary and you measure nothing. Confirm the build log says:

```
atlas-kernels: compiled 158 kernels for target 0 (gb10, qwen3.6-27b, nvfp4)
```

---

## 3. Run it

One entry point:

```bash
ATLAS_BIN=$PWD/target/release/spark \
HARNESS_DIR=/workspace/endpoints-fresh \
BASE_CONFIG=/workspace/endpoints-fresh/results/defaults_20260721_173342/config.yaml \
  ND=3 bash scripts/mlperf-edge/run_golden_e2e.sh
```

It serves with the frozen configuration, runs the Gate C2 smoke, then runs both harness phases
(`--mode both`) and prints the report directory. Expect roughly 2.5 hours.

`ND` is `--num-drafts`. **`--num-drafts N` gives verify width K = N+1**, so `ND=3` is the K=4
submission config. This off-by-one has caused mis-labelled sweeps more than once.

---

## 4. The frozen configuration

These are pinned submission values, not tunables. Changing any of them invalidates comparison
with the recorded results.

### Serve flags

```
--max-seq-len 32768  --max-batch-size 1  --kv-cache-dtype bf16
--gpu-memory-utilization 0.70  --enable-prefix-caching
--ssm-cache-slots 128  --ssm-checkpoint-interval 32
--speculative --num-drafts 3 --mtp-quantization bf16
--tool-call-parser qwen3_xml --disable-tool-grammar true --disable-thinking
```

- `--gpu-memory-utilization 0.70` — GB10's memory is unified. Above ~0.70 the box OOM-freezes
  rather than failing cleanly. Do not raise it.
- `--kv-cache-dtype bf16` — fp8-KV was measured and rejected; see §6.
- `--mtp-quantization bf16` — the MTP head is forced to bf16. It is NVFP4 in the checkpoint but
  runs ~4× slower that way, and it is only ~3% of the step.
- `--disable-tool-grammar true` — required for BFCL. With grammar on, tool-call scoring collapses.
- `--max-batch-size 1` — every speculative path is gated on `active.len() == 1`. There is no MTP
  at concurrency, so a larger batch silently disables the thing being measured.

### Environment

```
ATLAS_NO_FFN_NVFP4_MMQ=1     ATLAS_SSM_TAIL_MIDCHUNK=0    ATLAS_MTP_CATCHUP=0
ATLAS_MTP_DRAFT_CONF=0.0     ATLAS_MTP_GATE_FORCE=1       ATLAS_SSM_TAIL_PROTECT=1
ATLAS_SSM_TAIL_LEASE_TTL=128 ATLAS_BF16_TC_PREFILL=1
```

**Trap — presence flags.** Several `ATLAS_*` flags are read with `is_some()` / `is_none()`, so
setting one to `0` still turns it **on**. `ATLAS_NO_FFN_NVFP4_MMQ` and `ATLAS_BF16_TC_PREFILL`
are presence-checked; to disable them you must unset them, not set them to zero. Others
(`ATLAS_SSM_TAIL_MIDCHUNK`, `ATLAS_MTP_CATCHUP`) do compare against `"0"`/`"1"`. Check the
callsite before assuming.

Kill-switches that are deliberately left at their defaults: `ATLAS_GDN_WYN=0` disables the
wy5–wy8 batched GDN verify and falls back to the serial path.

---

## 5. Gates — run these before believing any timing

Order matters. Gate C2 is first because an NVFP4 build can pass the correctness gates while
emitting garbage, and the expensive gates take hours.

1. **Gate C2 — NVFP4 coherence + tool call.** Seconds. `run_golden_e2e.sh` runs it automatically
   and prints both outputs. It must produce coherent code and a real `get_weather` call on
   Paris. If it does not, stop; nothing downstream is meaningful.
2. **Output neutrality.** For any change that should not alter output, run both legs through
   `scripts/mlperf-edge/ab_ttft_probe.sh <tag> <binary>` and compare `combined_sha`. Greedy at
   temp 0 with a fixed seed, an output-neutral change gives a byte-identical hash.
3. **KL logit drift** for changes that *are* numeric:
   `python3 scripts/mlperf-edge/kl_coherence_gate.py <baseline_port> <candidate_port>`.
   Threshold: mean KL < 1e-3, token match ≥ 0.99.
4. **K=3 control leg** as the regression guard — the new dispatch arms are all `try_kernel`-gated,
   so K=3 must be unchanged (38.49 ms).

### Known harness noise — not a regression

Every run on this harness emits **exactly two** of these near startup:

```
jinja2.exceptions.UndefinedError: 'dict object' has no attribute 'name'
TypeError: Can only get item pairs from a mapping.
apply_chat_template failed for Qwen/Qwen3.6-27B (TypeError);
  falling back to whitespace tokenization. Tool-call OSL/TPOT may diverge.
```

This is client-side: the metrics aggregator probes the chat template with a tool-call message
shape the Qwen3.6-27B jinja template does not accept. It fires twice, at startup, not per
sample, and it is identical in the 4104 s reference run — so it does not break comparability
between legs. It is not an Atlas fault and not a signal that anything regressed. If you see a
count other than 2, that *is* worth investigating.

---

## 6. Do not re-run these — they are settled

Each was measured, not argued. Details and raw data in `DECODE_FOLD_LEDGER.md`.

| lever | verdict |
|---|---|
| **fp8-KV** | Neutral-to-worse at e2e on a same-binary control: TPOT +2.4%, TPS −2.7%, IoU 0.6285→0.6223. An earlier "−9% wall" claim was a stale-baseline error and is retracted. |
| **W4A8 int8-activation decode GEMV** (the Strix +25% trick) | 0.99–1.01× on GB10, plus ~0.5% accuracy cost. GB10 decode is weight-once and roofline-bound; activations are <1% of traffic at M≤8. |
| **Native FP4-MMA verify** | M≤8 is far too small for the tiles. |
| **NVFP4-ing the GDN weights** | Inert — this checkpoint's GDN is already NVFP4. |
| **Tree drafting** | Chains beat trees by 17% on measured acceptance. `speculative/tree_shape.rs` is kept as the Phase-0 evidence; nothing dispatches to it. |
| **K ≥ 5** | The ladder puts K=5–8 at 32–34 ms against K=4's 31.69 ms. K=5 is wall/tps-optimal within noise but has no full-e2e accuracy validation. |
| **SSM snapshot-lookup fix** | Real fix, but **neutral on GB10** — the pathology does not reproduce under this config. See `SNAPSHOT_FIX_AB.md`. |

---

## 7. Methodology rules that caught real errors

These are not ceremony; each one caught a wrong conclusion during this campaign.

- **Always run a same-binary control leg.** The fp8-KV "win" was a comparison against a stale
  baseline. The control leg is what exposed it.
- **Speculative TPOT is trajectory-dependent.** Different conditions emit different token
  sequences, so per-token timings are not comparable until you have diffed the emitted output.
  Compare `combined_sha` first.
- **IoU has an MDE of roughly 0.022–0.042 at N=1.** Swings smaller than that are undecidable.
  Do not chase them, and do not report them as wins or losses.
- **Never `ncu` a full GB10 serve** — it froze the box. Use `nsys` at kernel granularity, or a
  standalone microbenchmark.
- **Wall time is partly client-side** (~456 s of it across 1007 samples). Compare wall only
  between runs on an otherwise-idle box.
- **Bracket your `pgrep`** when scripting around a serve, and confirm the previous serve is gone
  before starting the next — overlapping serves OOM-wedge the box.

---

## 8. What is still open

- **TPOT tail dispersion.** p50 is at parity with a tuned vLLM; the gap widens down-tail
  (p90 +12.9%, p99 +13.7%) with no cliff. This is variable acceptance — on a reject-heavy step
  you pay the full K=4 verify for one accepted token. It is inherent to fixed-K speculation, and
  needs adaptive-K or a higher-acceptance drafter.
- **TTFT p99.** The one latency axis a tuned vLLM still wins. The snapshot-lookup fix does not
  close it (§6), and the evidence points at SSM snapshot *eviction* under multi-session
  pressure rather than lookup. A single-conversation probe structurally cannot reproduce it —
  it never contends for the 128 SSM cache slots.
- **W4A4 prefill.** Decode-W4A4 is dead because decode is weight-roofline-bound at M≤8. Prefill
  is the opposite regime — compute-bound, M = seqlen ≫ 8 — so native sm_121a FP4 MMA on the
  prefill FFN/attention GEMMs is the one place FP4 activations could buy tensor-core throughput.
  Never tested. Accuracy is the gate, and GDN is numerically sensitive, so scope it to
  FFN/attention projections first.
