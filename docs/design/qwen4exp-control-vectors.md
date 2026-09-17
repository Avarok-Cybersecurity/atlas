# Control vectors (activation steering) for Qwen3.8-Flash-Next on NVFP4

WIP note. Branch `wip/qwen4exp-control-vector`, cut from the #1063 head
(`4a253306e`).

**Status.** The apply path is implemented and the boot-time arm works:
`control_vector.cu` (kernel, proven against a CPU reference),
`spark-model/src/control_vector.rs` (GGUF load + validation, 15 CPU tests),
`model/control_vector_hook.rs` (the hook), and 14 call sites across the 11
model-level layer loops. Not yet done: the serve flags (§3.5), the cosine-probe
gate wired into a run (§5), per-request selection (§8), and any measurement on
real weights — **nothing here has been run against the model**. The transfer
question in §2 is still open.

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

`scripts/inspect_control_vector.py` reproduces those numbers.

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
  of the mean. Do not "fix" this by collapsing to the mean first — and note
  that Atlas's own collapse (§3.3) is a learned low-rank sigmoid mix, *not* a
  mean, so you cannot reason about the head output that way either.

## 2. Why NVFP4 is a non-issue

The intervention lives entirely in the residual highway. In Atlas that highway
is **FP32 `[T, hc_mult, H]`** — token-major with the `hc_mult` rows contiguous
per token; buffer `hc_streams`, sized `m * hc_mult * h * 4` at
`spark-runtime/src/buffers/sizes.rs:636`, accessor
`spark-runtime/src/buffers/accessors.rs:237` (whose doc comment still says
BF16 — stale, it is FP32). `hc_mult = 4`, `H = 2560`, 48 layers, 3:1 GDN:
attention interleave.

Weight quantization never enters the arithmetic. There is no such thing as an
"NVFP4 control vector" — the vector stays F32 (480 KB for all 48 layers) and
the dot product accumulates in FP32.

The *one* genuinely NVFP4-dependent question is *provenance*: this direction was
extracted from **unsloth UD-Q2_K_XL** activations under llama.cpp, not from
Atlas's NVFP4 graph. Whether it transfers is empirical, and §5 gives the cheap
test. Refusal directions are generally robust across quantization, so the
expectation is that it transfers — but that is a hypothesis, not a result.

For the record, LoRA is separately unwired here: `lora/loading.rs:29` rejects
every family but `qwen3_5` dense, `holo3_1_moe` and `qwen3_6_moe`, and the
qwen4_exp weight loader has no LoRA references at all. None of that matters for
this feature, but it does mean "just ship it as a LoRA" is not a shortcut.

## 3. Atlas insertion points

### 3.1 There is no per-layer `l_out`

The residual stream **is** the 4-row highway. The only `[T, hidden]` collapsed
vectors inside a layer are the two `hc_pre` outputs, which are the *inputs* to
the attention/GDN and FFN sublayers and are explicitly scratch ("`hidden` is
scratch from here on: the highway carries the state between layers" —
`qwen3_ssm/trait_prefill_hc.rs:158-160`). So the intervention must be applied
to all 4 rows per token. Four dots and four AXPYs, not one.

### 3.2 Two candidate sites

**(a) The model-level layer loop**, on `ctx.buffers.hc_streams()` immediately
after `layer.prefill()/decode()` returns. This is where `AVAROK_DUMP_HYPER_RMS`
already reads the highway (`prefill_b/forward_layers.rs:236-253`). Fires
exactly once per layer per token, and the "end of layer" semantics are
unambiguous. Eleven loops to patch:

| Path | Loop |
|---|---|
| whole-prompt prefill | `model/trait_impl/prefill_a.rs:426` |
| chunked prefill (shipping path) | `model/trait_impl/prefill_b/forward_layers.rs:170` |
| kernel-batched chunked prefill | `model/trait_impl/prefill_b/batch_kernel.rs:569` |
| two-phase GDN prefill | `model/trait_impl/prefill_c.rs:493` |
| decode, single seq (graph-captured) | `model/trait_impl/decode_a3.rs:41` |
| decode, batched multi-row | `model/trait_impl/decode_a2.rs:546` |
| mixed decode+prefill | `model/trait_impl/decode_b.rs:542` |
| decode, profiled | `model/impl_b1.rs:490` |
| self-speculative draft decode | `model/impl_b1.rs:680` |
| MTP verify, single seq | `model/trait_impl/verify_hc_rows.rs:306` |
| MTP batched verify | `model/trait_impl/verify_e.rs:667` |

**(b) Inside `ops::hc_post_site`** (`layers/ops/hyper_connection_dispatch.rs:188`),
the single function every path finishes a hyper-connection site through. Thirteen
call sites, but roughly half are the *intra-layer* `hc_post` and must not take
the hook, so it needs a required `site: HcSite` parameter to discriminate.

**Recommendation: (a).** It is fewer sites, needs no site discrimination, and
the "once per layer" property is structural rather than asserted. The appeal of
(b) was the `ssm_tp_all_reduce` precedent — one funnel so no path can be
forgotten — but that argument is weaker here because (b) still requires
correctly classifying which of the two sites per layer is the layer end, and
getting that wrong is silent. Whichever is chosen, the parameters that select
behaviour should be **required**, not defaulted or read from a global, so the
compiler enumerates the sites.

Note that `verify_b.rs`, `verify_c.rs`, `verify_c2.rs`, `verify_d.rs` and
`verify_fused.rs` are **dead for qwen4_exp** — `verify_needs_hc_path()` is
`config.hc_mult > 0` (`verify_hc.rs:229`) and every single-sequence K-verify
entry short-circuits to `decode_verify_hc`. Do not wire them.

### 3.3 Ordering against the final collapse

`ops::hc_head_site` collapses the highway into `[T, hidden]` BF16 on the last
model layer (`decode_inner.rs:614/773`, `verify_rows_hc.rs:264`,
`multi_seq/mod.rs:469`, `prefill_inner.rs:1051`). On Qwen **this collapse is
also the final norm** — `config.final_norm_identity = true`
(`parsers/qwen4_exp.rs:63`), so `final_norm_apply` degenerates to a `copy_d2d`
and `lm_head` reads that copy. The cvec must land before the collapse. With the
shipped range (4..44 of 0..47) the last layer is outside it anyway, but the code
must be correct for a range that includes layer 47.

### 3.4 The kernel

One fused kernel, one block per (token, row), FP32 highway, two passes over
2560 elements with a shared-memory reduction between them:

```
acc = dot(h[t, r, :], v_il)        // FP32
h[t, r, :] -= s * acc * v_il
```

Close templates already exist in
`kernels/gb10/qwen3.8-flash-next/nvfp4/`:

- `hyper_connection.cu:390` `hc_pre_down` — `dot(down_w[r], normed[t])` with
  lane-strided FP32 accumulate over the `[T, hc*H]` highway and a
  `__shfl_down_sync` reduction. **At `rank = 1` this is literally `dot(h, v)`.**
- `ple.cu:252` `ple_add_highway` — `highway[i] += ple_out[i]`, FP32 in place,
  grid-stride, eight lines. The writeback extends trivially to
  `highway[t*hcH + i] -= s * dot[t] * v[i]`.

Launch plumbing to copy: `layers/ops/ple.rs:92` and
`layers/ops/hyper_connection.rs:17+`. A new `.cu` in that directory gets a
module name for free from its file stem (`KERNEL.toml`). It is a new file in
the qwen3.8-flash-next shadow, so no CHKI exposure — but do **not** reach into
`kernels/gb10/common/`, which `strix` symlinks.

Cost: two passes over 2560 FP32 per (token, row), against a highway that
already moves `[T, hc*H]` FP32 twice per site. Expected to be a rounding error
on a memory-bound step — the published artifact reports 51.19 tok/s decode with
it on, and llama.cpp's version is not even fused (it materializes `repeat(v)`).
Do not pre-optimize; measure under `measurement-discipline` before claiming a
cost either way.

### 3.5 Loading

**The reader already exists — but not the one an earlier draft named.**
`spark_nllb::gguf::read_gguf_f32` (`crates/spark-nllb/src/gguf.rs:127`) does
the right thing, but `spark-nllb` pulls in `tokenizers` (with `onig`), which is
the wrong dependency to hang on `spark-model` for a 480 KB sidecar.

Use `spark_runtime::weights::gguf::container::GgufFile::parse` instead:
spark-model already depends on spark-runtime, everything inside `container` is
already `pub`, and only the `mod container;` declaration
(`weights/gguf.rs:27`) needed opening. `GgufFile::parse` + `tensor_abs_offset`
gives the header and each tensor's byte range; reading F32 rows out is ~40
lines.

(`GgufLoader` itself is a whole-model loader that dequantizes to BF16 into a
`WeightStore` keyed by HuggingFace name — far too heavy, and the wrong
contract, for a sidecar vector file.)

Load once at boot into a single `[48, 2560]` F32 device buffer, zeroing rows
outside the active range so the kernel needs no range branch.

## 4. Traps

Each maps to an incident already in the record.

**CUDA-graph capture.** The decode and verify bodies are graph-captured
(`decode_a.rs`, `verify_e.rs` module docs). The hook must be a **pure
stream-ordered kernel launch against fixed device addresses** — no
`std::env::var` per layer, no sync, no H2D at launch time. `model_levers.rs`'s
module doc is an explicit prohibition on reading env per layer or per token:
read once, carry it. `try_dflash_capture_all` (`model/impl_b3.rs:451`) is the
existing precedent for a capture-legal per-layer hook; copy its shape. A
boot-time device allocation satisfies the address requirement.

**`hc_row_offset` is not always zero.** Mixed steps and the K-row verify place
rows at non-zero offsets (`prefill_inner.rs:561`, `decode_inner.rs:467`); the
highway is addressed as `hc_streams().offset(hc_row_offset * hc_mult * h * 4)`.
A hook that assumes row 0 repeats the batched-verify highway-row defect, which
held MTP p1 at 0.19 while every correctness gate passed because the output
stayed byte-correct.

**`AVAROK_HC_FUSE_POST=1`** folds the attention `hc_post` into the FFN `hc_pre`
(`hyper_connection_post_fold.rs:186+`, default OFF), which makes the mid-layer
highway stale. The fold predicate already carries a `taps_inert` conjunct for
exactly this class of reader — an armed cvec belongs in that conjunct.

**Prefix caching is poisoned across a cvec change** — KV computed with the
projection differs from KV computed without it. An earlier draft concluded from
this that the vector had to be boot-time-only. **That was wrong about Atlas**,
and the correction is in §8: the prefix cache is already partitioned by an
adapter identity, and a cvec identity can ride the same channel. The MVP is
still boot-time because that is the smaller change, not because per-request is
unsound.

**MTP: the drafter is ONE layer, not 48.** `Qwen4ExpMtpHead::draft_hidden`
(`layers/qwen4_exp_mtp_hidden.rs:9`) runs a single `self.module.body.decode(...)`
at `:126`; the batched form `draft_bodies_batched`
(`qwen4_exp_mtp_combine.rs:159`) likewise runs one `decode_multi_seq` at `:269`.
So this is **not** a symmetric "apply it in both halves" problem — there is no
layers-4..44 sweep in the drafter to mirror. The open question is whether the
MTP body layer should carry a direction at all, and the published artifact gives
no guidance because llama.cpp does not run MTP here. **Decide it deliberately
and measure `p1` / `tok_step` on the first run**: a drafter/verifier mismatch
does not corrupt output (verify is authoritative) but silently collapses
acceptance, which no correctness gate will catch.

**TP=2: apply only on post-all-reduce buffers.** The highway, the `hc_pre`
outputs and the `hc_head` output are all post-reduce and rank-replicated, so the
kernel runs redundantly on each rank producing identical bytes — safe, provided
`v` is uploaded identically and the reduction order is fixed. `hidden_size` is
never sharded (`tp_shard.rs` shards only QKV/O/gate/up/down dims). **Avoid**
`attn_out` between O-proj and `comm.all_reduce_async` (`prefill_inner.rs:748`,
`decode_inner.rs:112`, `decode_inner.rs:581`) and the MoE `output` before its
in-forward all-reduce (`moe/forward_prefill_exl3.rs:262`, `forward_exl3.rs:300`,
and siblings) — both are row-parallel partial sums, and intervening there is the
GDN `out_proj` trap verbatim.

**EP=2: both ranks must load the same file**, or they diverge mid-sequence. Log
the file's SHA per rank at boot and compare.

**Fails-open is the enemy.** If the file is missing, malformed, or shaped for a
different `hidden_size`, **fail the boot loudly**. A silently-skipped projection
serves an un-ablated model while the operator believes otherwise — the
"capability fails closed into fallback" pattern, except here the failure mode is
user-visible behaviour, not just speed.

## 5. Validation

The upstream repo ships the right oracle and we should copy it rather than
gating on "did it refuse".

**Primary gate — the cosine probe.** Report `mean|cos(h, v_il)|` over the
highway before and after the hook. With the vector loaded, `post` must be
< 0.01; without it, `post` must equal `pre` and be clearly non-zero. This is
deterministic, and far more sensitive than a behavioural eval. Existing
plumbing to build on:

- `diag_norm_f32` (`qwen3_attention/trait_impl/diag.rs:47`, gated by
  `AVAROK_DIAG_V4_ALL_LAYERS=1`) already fires around every hc site on both
  single-stream and ALL_STREAMS extents.
- `hidden_probe_layer` (`model/impl_a3.rs:615`, `AVAROK_LOGIT_PROBE=1`) is the
  per-layer fingerprint precedent, gated first thing for zero cost when off.

Run the probe on **every** path — all four prefill loops, decode, multi-seq
decode, both verify loops — not just decode. That is the only thing that proves
no site was missed.

**Secondary — does the Q2_K_XL-derived direction transfer to NVFP4?** Run the
probe on the NVFP4 serve with the published vector. If `post` ~ 0 and behaviour
shifts, it transferred and no derivation work is needed. Also report KL against
the un-ablated serve on a neutral corpus; the artifact quotes `kl_mean 0.186` on
its own base as an order-of-magnitude reference (its `kl_max 18.3` says some
tokens move a great deal).

**If it does not transfer**, derive our own against NVFP4. Most of the capture
path already exists: `AVAROK_QWEN4EXP_DUMP=<dir>` / `tap_highway`
(`layers/ple/dump.rs:76`) writes `<dir>/L{layer:02}_{tag}.bin` as raw LE FP32
`[T, hc*H]` at named points. Caveats: it is **prefill-only**, `claim()` is
one-shot per tag, and it **synchronizes and copies D2H at every tap** — its own
doc calls it "a debug aid, not a serving mode." That is fine for an offline
corpus run, which is all derivation needs. Then mean-difference harmless vs
harmful over the stream mean, normalize, write a GGUF in the same layout.

A free cross-check falls out: **cosine between our NVFP4-derived direction and
the published Q2-derived one**, per layer —
`scripts/inspect_control_vector.py --against`. High cosine would be strong
evidence the direction is a property of the model rather than of the quant.

## 6. What Atlas gets that llama.cpp cannot

The upstream `AGENTS.md` pins `-np 1` — "the model's indexer asserts on batched
requests" — so the published artifact is strictly single-stream. Atlas serves
this model at C=4 with MTP and multi-rank prefix caching, and site (a) covers
the batched and speculative paths by construction. That is a capability the
reference implementation does not have.

## 7. Scope note

This is a generic activation-steering mechanism; refusal ablation is one vector
among many (style, verbosity, format adherence, and refusal *calibration* in
either direction all use the same hook). The implementation should be named and
documented for the general capability. Note also that the shipped `.json`
reports movement on the harmless set as well as the harmful one
(`refusal_harmless_*` 4.0/6.0 vs `refusal_harmful_*` 2.0/2.0, scale
undocumented) — a rank-1 ablation at s=1.0 is a blunt instrument and is not free
of collateral behavioural change. Whatever we ship should be off by default and
explicit at boot.

## 8. Per-request opt-in

The goal is request-level selection, the way a LoRA adapter is selected. An
earlier draft of this note assumed that was blocked by the prefix cache. It is
not — Atlas already solved this problem for LoRA, and the machinery generalises.

**The prefix cache is not adapter-blind. It is adapter-keyed, twice over:**

- `hash_token_prefix(tokens, count, adapter_id)`
  (`spark-runtime/src/radix_tree.rs:37`) folds the id into the FNV-1a state
  before the tokens, and the fold is a strict no-op at `adapter_id == 0`, so
  base keying stays byte-identical to the old token-only hash.
- `RadixTreeInner` holds `roots: HashMap<u64 /*adapter_id*/, NodeId>`
  (`radix_tree/inner.rs:60-126`) — physically **disjoint radix roots**. The
  comment there is explicit that the hash alone would not be enough, because
  the children map is keyed by token chunk; disjoint roots are what stop a
  cross-adapter insert collision. Asserted by
  `radix_tree/tests/adapter.rs:13,48,71`.

The discriminator is `adapter_id_hash(name, generation)`
(`spark-model/src/lora/key.rs:31`) — name-derived so it survives pool-slot
reuse, with a generation folded in so a re-staged slot misses the stale prefix.
It is stamped onto the sequence at prefill (`scheduler/prefill_a_step.rs:159`),
carried across preempt and restore, and threaded through every cache entry
point.

**There is a second belt.** `filter_adapter_cohort`
(`scheduler/admission.rs:155`) restricts a batch to ONE adapter identity,
re-queueing the others at the front rather than failing them. That exists
because v0 LoRA decode does not route per row, so a batch mixing adapters is
refused wholesale. **A control vector has exactly the same property** — the
hook applies one vector to the whole highway, so a batch cannot mix cvec
selections either. The cohort filter is therefore not an obstacle; it is the
mechanism this needs, already built.

### What to build

Compose, in ONE function, a `variant_id: u64` from the adapter id and the cvec
id, and use it for both consumers:

1. the prefix-cache key (in place of the bare `adapter_id`), and
2. the admission cohort key.

Today there is exactly one `u64` slot for such a discriminator, and both
consumers read it. If the two are computed in different places they will drift,
and the failure is silent in the worst way: a cache hit that returns another
variant's KV. This is the same four-consumer hazard the LoRA slot layout was
bitten by.

Keep `0` meaning "base, nothing applied", so a serve with neither an adapter
nor a cvec hashes exactly as it does today.

`ControlVector` should then become a small registry on the model (name → vector)
rather than a single `Option`, with the per-request selection resolved at
admission and carried on `ForwardContext` — `cvec_after_layer` would read
`ctx`, not `self`. The call sites do not change; only what the hook reads.

**One asymmetry worth noting before building it.** A LoRA adapter is selected
by the `adapter` field or by `model`; there is no selector meaning "no
adapter" once a pool is resident (`lora_control.rs:27`), which is a known wart.
A refusal-steering vector should not inherit it: "off" has to be expressible
per request, which means the cvec id must have a real zero and the request
field must distinguish absent from "none".
