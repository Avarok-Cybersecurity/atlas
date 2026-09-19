---
name: bench-parity-oracle
description: B.E.N.C.H P.A.R.I.T.Y O.R.A.C.L.E — decides whether two engine invocations (Atlas vs vLLM/SGLang/...) measure the same thing. Give it two JSONL lines of {engine, command}; it returns a per-axis verdict and refuses to call a comparison fair when it is not.
---

# B.E.N.C.H P.A.R.I.T.Y O.R.A.C.L.E

**B**enchmark **E**quivalence **N**ormaliser for **C**omparing **H**arnesses — **P**arity **A**djudicator **R**eporting **I**nvocation **T**iers, **Y**ielding **O**nly **R**eliable **A**pples-to-apples **C**omparison **L**egitimacy **E**vidence.

A published "we beat X by 1.33×" is only worth the parity behind it. This oracle
reads two engine commands and says, axis by axis, whether they measure the same
workload — and **refuses to return a parity verdict when an axis is invisible to
it**, because an unexamined axis is exactly where an unfair comparison hides.

## Input

Two (or more) JSONL lines on stdin, one object per line:

```jsonl
{"engine": "vllm",  "command": "vllm serve unsloth/Qwen3.8-27B-NVFP4 --max-model-len 2048 --max-num-seqs 128 ..."}
{"engine": "atlas", "command": "spark serve unsloth/Qwen3.8-27B-NVFP4 --max-seq-len 2048 --max-batch-size 128 ..."}
```

`engine` is one of `atlas`, `vllm`, `sglang`, `trtllm`. `command` is the full
serve invocation, exactly as run. Optional `env` carries the environment
(`"ATLAS_MTP_K_LADDER=1:3,2:1 ..."`), and optional `harness` carries the client
invocation — the oracle will otherwise tell you those axes are unexamined.

## Run

```sh
python3 .claude/skills/bench-parity-oracle/parity.py legs.jsonl
python3 .claude/skills/bench-parity-oracle/parity.py legs.jsonl --json   # machine-readable
cat legs.jsonl | python3 .claude/skills/bench-parity-oracle/parity.py     # stdin also works
```

Exit code: `0` only for **IN PARITY**; `1` for the other two.

## Output

A per-axis table and one of three verdicts:

| verdict | meaning |
|---|---|
| **IN PARITY** | every axis the oracle knows is equal, and none is unexamined |
| **NOT IN PARITY** | at least one axis differs — each difference is named with both values |
| **UNDETERMINED** | an axis is unexamined on at least one side; parity cannot be claimed |

`UNDETERMINED` is not a softer `IN PARITY`. It means the comparison has not been
shown to be fair, which is the same practical standing as unfair.

A `match (default)` row is a match the oracle derived from a **documented engine
default** because neither command set the flag, not from two observed values.
Those axes are named again in the verdict line, because a reader deciding
whether to trust a chart should know which parity claims rest on a default:

```
VERDICT: IN PARITY — every known axis matches and none is unexamined;
2 rest(s) on a documented engine default rather than an observed value:
tensor_parallel, scheduling
```

### The verdict on this repo's own published pair

Run against `bench/ladder38/published.json`'s `atlas` and `vllm-mtp` legs — the
two series behind the site's "Faster than vLLM at every concurrency" chart —
the oracle returns **IN PARITY**. Its `vllm-nospec` leg returns NOT IN PARITY on
three axes (`context_len` 2048 vs 4096, `kv_dtype` fp8 vs bfloat16,
`speculation` mtp:3 vs none), which is what a second baseline is *for*.

## The axes it adjudicates

Serve-side, normalised across engines:

| axis | atlas | vllm | sglang |
|---|---|---|---|
| checkpoint | positional / `--model-name` | positional / `--model` | `--model-path` |
| context | `--max-seq-len` | `--max-model-len` | `--context-length` |
| batch cap | `--max-batch-size` | `--max-num-seqs` | `--max-running-requests` |
| KV dtype | `--kv-cache-dtype` | `--kv-cache-dtype` | `--kv-cache-dtype` |
| GPU mem fraction | `--gpu-memory-utilization` | `--gpu-memory-utilization` | `--mem-fraction-static` |
| prefix caching | `--enable-prefix-caching` | `--enable-prefix-caching` / `--no-…` | `--enable-prefix-caching` |
| speculation | `--speculative --num-drafts N` | `--speculative-config` / `--num-speculative-tokens` | `--speculative-num-draft-tokens` |

**Speculation is normalised to `<method>:<k>`, never compared as a string.** The
published pair spells one setting two ways — Atlas `--speculative --num-drafts 3
--mtp-quantization bf16`, vLLM `--speculative-config
'{"method":"mtp","num_speculative_tokens":3}'` — and a string comparison called
that pair NOT IN PARITY. That is the worst failure this tool can have: it would
have us "fix" a fair comparison until it became unfair. Three states, kept
distinct:

| state | when | verdict effect |
|---|---|---|
| `mtp:3`, `draft-model:4`, … | a method **and** a width are both readable | compared |
| `none` | **no** speculation flag of that engine appears at all — an observed off | compared |
| unexamined | some speculation flag appears but no method/width pair can be read from it | forces UNDETERMINED |

A draft width with no method (`--num-speculative-tokens 3` alone) is the third
case, not the first: it does not name a setting the other engine can be held to.
| tensor parallel | `--tensor-parallel` | `--tensor-parallel-size` | `--tp-size` |
| thinking | `--disable-thinking` | client `chat_template_kwargs.enable_thinking` | same |
| scheduling | `--scheduling-policy` | `--scheduling-policy` | `--schedule-policy` |

Harness-side (client) axes it will ask for and mark unexamined when absent:
**ISL**, **OSL**, **concurrency**, **reps/warmup**, **temperature**, **seed**,
**presence/frequency penalty**, **prompt fixture**.

## Why those axes and not others

Each one has cost a real comparison in this repo:

- **OSL** — the published ladder runs OSL 1024, the gate runs 320; the same
  checkpoint reads 478 tok/s and 116 tok/s. Nothing was wrong with either run.
- **Speculation levers** — the published ladder pinned
  `ATLAS_MTP_K_LADDER=1:3,2:1,4:2,8:2,16:1` and `DCUT_RATIO=1.0`; the gate takes
  recipe defaults. Same engine, same model, different machine.
- **Sampling penalties** — `presence_penalty`/`frequency_penalty` had to be
  pinned to 0.0 on BOTH engines before the ladder was trustworthy
  (`bench/ladder38/RESULTS.md`), because one engine defaulted them non-zero.
- **Thinking** — enabled on one side silently multiplies output tokens.
- **KV dtype** — an fp8-KV engine against a bf16-KV engine is a memory
  comparison wearing a throughput comparison's clothes.

## What it deliberately does NOT do

It does not judge *which* configuration is better, or whether a lever is
"allowed". Parity is a question about two commands, not about merit. If you want
the best number each engine can produce, that is a different exercise and the
oracle will not bless it as parity.
