---
title: 'DFLASH-2: the fastest single-machine numbers Atlas has produced'
dek: 66.6 tokens per second on a stock build, one DGX Spark, one stream, and every figure reproducible from a commit.
categories: [engineering, benchmarks]
date: 2026-09-01
keywords: [speculative decoding, dflash2, dgx spark, gb10, qwen3.8-27b, single stream, atlas inference]
og-image: ''
author: ronald-stesiak
draft: false
---

Single machine, single session is what most Atlas users actually run: one
DGX Spark, one model, one stream. From a CUDA/C centric tuning background,
that number is the one I care about most, and this post is about the layer
that decides it, what it measures on current main, and how to reproduce
every figure from a commit.

## What DFLASH-2 is

DFLASH-2 is Atlas's SOTA and fastest measured speculative-decode layer for
Qwen3.8-27B, with more targets to follow: a block-diffusion draft head
proposes a window of tokens, the target model verifies them in one pass,
and the engine keeps only the tokens the target verifies.

Every accepted token passes the target's verification in that run, and the
output is byte-reproducible.

It landed on main in the #797 integration stack (from PR #648), alongside
the register-tiled batch-8 W4A16 GEMV and the FP8 propose path. The GDN
verify kernels that carry the current numbers landed with the #818 stack.

## The protocol

Every number below follows the same discipline: single stream, temperature
0, median of 3 runs, completion-token counts quoted next to tok/s, fresh
server boot, first requests raw (no warmup), every run server-attested with
accept telemetry in the log. Code-bound prompt: a MinHeap class with
docstrings (839-879 token completions). Prose-bound prompt: automotive
history (300-token completions). Sets at different completion lengths are
never compared.

Hardware: NVIDIA DGX Spark (GB10), driver 580.126.09, clocks unpinned.

## The numbers

Measured on the PR #831 build: current main plus the change that makes the
configuration below the default. The benchmark gate ran on that head and
all 11 required gates pass.

| workload | r1 | r2 | r3 | median | tokens |
|---|---|---|---|---|---|
| MinHeap (code) | 66.6 | 66.6 | 66.5 | **66.6** | 839 |
| Volvo (prose) | 24.7 | 24.7 | 24.7 | **24.7** | 300 |
| serial floor, speculation OFF | 16.23 | 16.23 | 16.21 | **16.23** | 870 |

Against the serial floor, DFlash-2 turns 16.23 tok/s into 66.6: a **4.10x**
speedup on identical hardware, identical model, identical prompt class,
with the only difference being whether speculation is on.

The floor is a strict one-variable A/B: the serve line with the two
speculation flags removed (`--dflash`, `--draft-model`) and nothing else
changed. It is also the most reproducible measurement in this post: all
three runs returned 16.2 tok/s, 870 completion tokens, and the same sha256
`5f87e3d9`. The record sets hold the same standard: one sha256 per
workload, three runs each (`e23c9083` for MinHeap, `90c1fca8` for Volvo).

On patterned code the ceiling is higher still. A memoized-Fibonacci prompt
in the 198-215 token class produced a 70.8 median under the same
configuration served with an explicit `--dflash-gamma 10`, which is the
value the defaults now resolve on their own. That is a different
completion-length class and is quoted separately rather than mixed into
the table above, per the protocol.

For reference, the path here is public: 54.5 tok/s published with PR #648
and independently reproduced at 54.6 on separate hardware the same day,
63.0 after the #817 kernel stack, and 66.6 once the whole stack plus the
#831 defaults landed on a stock build.

Prose runs slower than code by design of the workload, not the engine: a
drafter feeds on structure. On code the verify path accepts ~5.5 tokens
per step; on prose ~1.6. Speculative decode is a code-and-math instrument
first.

## The reproducibility receipt

The strongest property of these runs is not the speed. Each workload above
produced one output hash across all three runs. And the 875-token MinHeap
output of the earlier record configuration is **byte-identical** (sha256
`25e95ba0585c66ca...`) across the pre-merge #648 branch on 08-19 and
merged main eleven days later, straight through a 342-file integration
squash, under two different context-commit configurations. Same answer to
the byte. Reproducibility here is a claim you can hash.

## Configuration: what these numbers require

Reaching these numbers previously required the record launch environment,
a set of DFlash flags that shipped opt-in. As of PR #831 they are the
defaults: a stock build serves this configuration with no DFlash flags and
no environment variables. The A/B that justified the change is stark:
identical binary and box, defaults vs record env, produced 505-token
instruction-dropping output at degraded quality versus the record-class
output above. The full flag list and per-flag rationale are in the PR.

Serve line:

```bash
ATLAS_TARGET_MODEL=qwen3.8-27b ./target/release/spark serve <Qwen3.8-27B-NVFP4> \
  --dflash --draft-model <Qwen3.8-27B-DFlash2> \
  --max-seq-len 4096 --gpu-memory-utilization 0.55 --max-batch-size 1 \
  --ssm-cache-slots 0 --kv-cache-dtype bf16 --scheduling-policy slai \
  --disable-thinking
```

Checkpoints: unsloth Qwen3.8-27B-NVFP4 (target), incoai/Qwen3.8-27B-DFlash2
(drafter, stock from HF). The kv-cache bf16 override is deliberate and part
of the record configuration; the server will warn about it.

## Reproduce it

Build the #831 head (or main once it merges), serve as above, and run the
two prompts at temperature 0 with median-of-3. The receipts behind this
post (request JSONs with usage counts and output hashes, plus the server
logs with accept telemetry) are the standard we hold every number to. If
your median lands meaningfully away from these figures on a GB10, that is
worth a report: this configuration is deterministic enough that two builds
eleven days apart agreed to the byte.

The number at the top of this post is current as of publication. It is not
expected to remain the record for long.
