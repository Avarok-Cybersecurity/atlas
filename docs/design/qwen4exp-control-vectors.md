# Control vectors (activation steering) for Qwen3.8-Flash-Next on NVFP4

WIP exploration note. Branch `wip/qwen4exp-control-vector`, cut from the #1063
head (`4a253306e`). Nothing here is implemented yet — this records the
mechanism, the exact insertion points in Atlas, and the traps, so the
implementation can be costed and reviewed before it is written.

Motivating artifact:
[`Cudecnik/Qwen3.8-Flash-Next-refusal-projection`](https://huggingface.co/Cudecnik/Qwen3.8-Flash-Next-refusal-projection)
— a per-layer refusal direction for `qwen4exp`, published as a llama.cpp GGUF
control vector plus two llama.cpp patches. We want the same capability against
Atlas's NVFP4 serve.

## 1. What the artifact actually is

**Not a LoRA.** It is a set of unit direction vectors in the residual stream.
No weight deltas, no dequantize/requantize, nothing touches a packed NVFP4
tensor. Verified by parsing the file:

```
general.architecture      = controlvector
controlvector.model_hint  = qwen4exp
controlvector.layer_count = 47
47 tensors: direction.1 .. direction.47, each F32 [2560], every one |v| = 1.000000
```

2560 is exactly `qwen4_exp` `hidden_size` (16 n-gram heads x 160), and 48 is
its layer count, so the file is shaped for the model we already serve.

Cross-layer structure, measured on the file: adjacent-layer cosine **0.968**,
all-pairs off-diagonal cosine **0.722** over the active range. That is one
coherent feature carried across depth, not 41 independent per-layer fits — a
good sign it is a real representational direction rather than noise, and it
means a `dir=single` variant (one layer's direction reused for the whole range)
would likely work almost as well.

### The intervention

Mode is `project`, scale 1.0, layers 4..44 (from the shipped `.json`):

```
h  <-  h - s * (h . v_il) * v_il            v_il unit norm, s = 1.0
```

At s=1 this fully ablates the component of the residual along `v_il` — a rank-1
projection. `mode=add` (`h += v`) is the other arm; the repo ships a separate
additive vector for it at scale ~0.1 over layers 16..32. **The two files are
not interchangeable** — loading the additive vector at scale 1.0 in project
mode is meaningless.

### Where llama.cpp hooks it

From `01-qwen4exp-cvec-hooks.patch`, in `llama_model_qwen4exp::graph`:

```c
res_hc = build_hc_combine(res_hc, cur, inject, il);
res_hc = build_cvec(res_hc, il);          // <-- the hook
cb(res_hc, "l_last", il);
```

Two properties matter:

- It is applied **after the hyper-connection combine, at the end of the layer**
  — not at the intra-layer (attention/GDN sublayer) combine.
- It is applied to **all `hc` streams** of `[n_embd, hc, n_tokens]`, broadcast
  over the stream axis, while the vector was *derived* from the stream **mean**
  (the patch's `LLAMA_QSA_L_OUT=1` export). That is self-consistent because
  projection is linear: the mean of the projected streams equals the projection
  of the mean. Do not "fix" this by collapsing to the mean first.

## 2. Why NVFP4 is a non-issue

The intervention lives entirely in the residual highway. In Atlas that highway
is **FP32 `[T, hc_mult, H]`** (`hyper_connection_post_fold.rs:6-12`: "per site
it reads `[T, hc*H]` FP32 and writes it straight back"), with `hc_mult = 4`,
`H = 2560`. Weight quantization never enters the arithmetic. There is no such
thing as an "NVFP4 control vector" — the vector stays F32 (480 KB for all 48
layers) and the dot product accumulates in FP32.

The *one* genuinely NVFP4-dependent question is *provenance*: this direction was
extracted from **unsloth UD-Q2_K_XL** activations under llama.cpp, not from
Atlas's NVFP4 graph. Whether it transfers is empirical, and §5 gives the cheap
test. Refusal directions are generally robust across quantization, so the
expectation is that it transfers — but that is a hypothesis, not a result.

## 3. Atlas insertion points

### 3.1 The funnel — do NOT sprinkle the hook

Every qwen4_exp forward path finishes a hyper-connection site through one
function, `ops::hc_post_site` (`layers/ops/hyper_connection_dispatch.rs:188`).
The call sites are:

| Path | File:line |
|---|---|
| prefill | `qwen3_attention/trait_impl/prefill_inner.rs:792` |
| decode | `qwen3_attention/trait_impl/decode_inner.rs:600, 632, 742` |
| GDN/SSM decode | `qwen3_ssm/trait_decode_hc.rs:128, 163` |
| batched multi-seq decode | `qwen3_attention/trait_impl/multi_seq/mod.rs:272, 411, 433` |
| batched verify | `qwen3_attention/trait_impl/verify_rows_hc.rs:247` |
| batched verify (inner) | `qwen3_attention/trait_impl/verify_rows_hc_inner.rs:269, 338` |
| batched verify (attn) | `qwen3_attention/trait_impl/verify_rows_hc_attn.rs:296` |
| **MTP drafter** | `qwen4_exp_mtp_combine.rs:98` — *different kernel*, `qhc_mtp_combine_streams` |

Roughly half of these are the intra-layer (attention/GDN sublayer) `hc_post`
and half are the end-of-layer (FFN) one. **Only the end-of-layer site takes the
hook.**

Wiring thirteen call sites by hand is the exact failure mode this codebase has
already paid for twice — the LoRA slot layout with four consumers that silently
disagreed, and GDN `out_proj`, which was deliberately folded *inside*
`ssm_tp_all_reduce` precisely so that "all five SSM dispatch paths finish
`out_proj` through that one function, so folding there makes it impossible to
wire some paths and forget one."

**Follow that precedent.** Fold the projection inside `hc_post_site`, and add
the two facts it is missing as **required** parameters:

```rust
pub fn hc_post_site(
    ...,
    site: HcSite,          // Attn | Ffn — only Ffn takes the cvec
    layer_idx: u32,        // selects direction.<il>
    cvec: Option<&ControlVector>,
    ...
)
```

Making them required (not defaulted, not read from a global) means the compiler
enumerates the call sites for you. You cannot miss one. `num_tokens`,
`hidden_size`, `hc.hc_mult`, `out` and `stream` are already in scope there, so
the kernel launch needs nothing else.

The MTP drafter is the one genuine second site: it combines streams through
`qhc_mtp_combine_streams` rather than `hc_post_site`, so it must be wired
explicitly. Two places, not thirteen.

### 3.2 Ordering against the final collapse

`hc_head_site` collapses the highway into `hidden` on the last layer
(`decode_inner.rs:773`, `verify_rows_hc.rs:264`, `multi_seq/mod.rs:469`). The
cvec must be applied **before** that collapse. With the shipped range (4..44 of
0..47) the last layer is outside the range anyway, but the code must be correct
for a range that includes layer 47.

### 3.3 The kernel

One kernel, in place on `hc_streams`, grid over `(token, stream)`:

```
acc = dot(h[t, r, :], v_il)        // FP32, 2560 elements
h[t, r, :] -= s * acc * v_il
```

Two passes over 2560 FP32. Per layer per token this is `hc_mult * 2560 * 2`
reads + `hc_mult * 2560` writes against a highway that already moves
`[T, hc*H]` FP32 twice per site. It is a rounding error on a memory-bound step
— the published artifact reports 51.19 tok/s decode with it on, and llama.cpp's
implementation is not even fused (it materializes `repeat(v)`). Do not
pre-optimize; measure it under `measurement-discipline` before claiming a cost
either way.

### 3.4 Loading

Keep GGUF as the container — `crates/spark-runtime/src/weights/gguf.rs` is
already a generic GGUF walker, and `avarok-core/src/config/gguf.rs` already
parses GGUF metadata. The control-vector file needs a small dedicated reader
rather than the model loader, because the model loader's contract is
"dequantize to BF16, key by HuggingFace name" and we want F32 kept as F32 and
keyed by layer index. ~80 lines, CPU-only, unit-testable in CI (which is
CPU-only).

Load once at boot into a single `[48, 2560]` F32 device buffer, zero rows
outside the active range so the kernel needs no range branch.

## 4. Traps

These are Atlas-specific and each maps to an incident already in the record.

**Prefix caching must be considered poisoned across a cvec change.** KV computed
with the projection applied differs from KV computed without it. There are four
separate prefix-cache contamination incidents on record. **Make the control
vector a boot-time flag, not a per-request parameter** — llama.cpp does the
same (`--control-vector-scaled` at server start). A per-request cvec needs the
prefix cache keyed by cvec identity, and that is a much larger and much more
dangerous change. Defer it.

**Draft and verify must agree.** If the projection is applied in MTP verify but
not in the drafter, acceptance collapses; if in the drafter but not verify, the
output stays correct (verify is authoritative) and acceptance collapses
silently. Both are invisible to a correctness gate and show up only as a tok/s
regression — the same signature as the highway-row defect that held MTP p1 at
0.19 while every gate passed. Watch `p1` / `tok_step` on the first run.

**TP=2: apply exactly once, on the full-width residual.** The highway after
`hc_post` is full 2560-wide and identical on both ranks, so applying it inside
`hc_post_site` is safe. Applying it anywhere upstream of an all-reduce would
either double-apply it or project a partial sum — the GDN `out_proj` trap
verbatim.

**EP=2: both ranks must load the same file and apply identically**, or the ranks
diverge mid-sequence. Ship the vector path in the same config both ranks read;
log the file's SHA on each rank at boot and compare.

**CUDA graphs.** The vector buffer must live at a stable address across
captures. A boot-time allocation satisfies this; do not allocate per-request.

**Fails-open is the enemy.** If the file is missing, malformed, or shaped for a
different `hidden_size`, **fail the boot loudly**. A silently-skipped projection
serves an un-ablated model while the operator believes otherwise — the
"capability fails closed into fallback" pattern, and here the failure mode is
user-visible behaviour, not just speed.

## 5. Validation

The upstream repo ships the right oracle and we should copy it rather than
gating on "did it refuse".

**Primary gate — the cosine probe.** Export the per-layer stream mean before and
after the hook and report `mean|cos(h, v_il)|`. With the vector loaded, `post`
must be < 0.01; without it, `post` must equal `pre` and be clearly non-zero.
This is a direct, deterministic check that the arithmetic landed on every
forward path, and it is far more sensitive than a behavioural eval. The diag tap
already sitting immediately after `hc_post_site` (`decode_inner.rs:753-769`,
`diag_norm_f32` on `hc_streams` with both single-stream and ALL_STREAMS extents)
is most of the plumbing.

Run the probe on **every** path — prefill, decode, multi-seq decode, batched
verify, MTP draft — not just decode. That is the only thing that proves no call
site was missed.

**Secondary — does the Q2-derived direction transfer to NVFP4?** Run the probe
on the NVFP4 serve with the published vector. If `post` ~ 0 and behaviour
shifts, it transferred and no derivation work is needed. Also report KL against
the un-ablated serve on a neutral corpus; the artifact quotes `kl_mean 0.186`
on its own base, which is a useful order-of-magnitude reference (its
`kl_max 18.3` says some tokens move a great deal).

**If it does not transfer**, derive our own against NVFP4 — that needs an
activation-capture path (the `LLAMA_QSA_L_OUT` equivalent: dump the per-layer
stream mean for a prompt corpus), then an offline mean-difference and
normalize. Bigger lift, but the apply path and the probe are unchanged, and a
useful cross-check falls out for free: **cosine between our NVFP4-derived
direction and the published Q2-derived one.** High cosine would be strong
evidence the direction is a property of the model rather than of the quant.

## 6. What Atlas gets that llama.cpp cannot

The upstream `AGENTS.md` pins `-np 1` — "the model's indexer asserts on batched
requests" — so the published artifact is strictly single-stream. Atlas serves
this model at C=4 with MTP and multi-rank prefix caching. If the hook is wired
through `hc_post_site` it covers the batched and speculative paths by
construction, which is a capability the reference implementation does not have.

## 7. Scope note

This is a generic activation-steering mechanism; refusal ablation is one vector
among many (style, verbosity, format adherence, and refusal *calibration* in
either direction all use the same hook). The implementation should be named and
documented for the general capability. Note also that the shipped `.json`
reports movement on the harmless set as well as the harmful one
(`refusal_harmless_* ` 4.0/6.0 vs `refusal_harmful_*` 2.0/2.0, scale
undocumented) — a rank-1 ablation at s=1.0 is a blunt instrument and is not
free of collateral behavioural change. Whatever we ship should be off by
default and explicit at boot.
