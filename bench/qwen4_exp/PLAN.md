# Qwen3.8-Flash-Next (`qwen4_exp`) — plan of work to first correct token

Written 2026-08-26, after the load milestone (Avarok #753, PR #754). The model
boots, passes the fail-closed kernel audit, and serves the HTTP API. It does
not generate: a request reaches model layer 0 and is refused by name.

This file is the sequencing decision and the reasoning behind it. Progress is
tracked on #753's checkboxes; this is the *order* and the *why*.

---

## 0. What actually blocks the first token

Three forward mechanisms are unported. From `ARCHITECTURE.md`:

| | mechanism | where | state |
|---|---|---|---|
| B | mHC low-rank residual | all 48 layers | kernel written, **never validated**, wired nowhere |
| C | PLE n-gram injection | model layer **1** | unbuilt |
| D | QSA indexer | 12 full-attn layers | **provably inert at <=2048** — sequenced later, NOT optional |

Only B and C stand between here and a token. D is arithmetic-exempt inside the
budget the model currently fits in (see `ARCHITECTURE.md` §3).

**QSA is required, not negotiable.** It is sequenced after the first token
only because it cannot be exercised before then — see §1.5. Everything that
would otherwise have to be redone for it is threaded in now.

### Five defects, all silent, all live before this branch

`attn_layer_idx` counts **attention layers only** (0..11), not model layers.
The mHC path in `qwen3_attention` was written for DeepSeek-V4, where every
layer is attention and the two indices coincide. On this model they do not:

1. **`hc_expand` fires on the wrong layer.** The guard is
   `attn_layer_idx == 0`, which is model layer **3**. Model layers 0-2 are GDN.
   The 4-stream highway would be seeded three layers late, on top of whatever
   the buffer held.
2. **`hc_head` never fires at all.** The guard is
   `attn_layer_idx + 1 == num_hidden_layers`, i.e. `12 == 48` — always false.
   `hyper_connection_mixer` **is the model's final norm** (the checkpoint has
   no `model.norm.weight`), so the LM head would read an uncollapsed,
   unnormalized stream.

Neither throws. Both are on the list below as part of phase B.

### Two more, in the same family

3. **A second RMS pass on the mHC output.** The attention mHC path runs
   `rms_norm(hidden, input_norm)` after `hc_pre` (`prefill_inner.rs:614`).
   Qwen has **no per-layer input_layernorm** — `hc_norm` inside `hc_pre`
   occupies that role, and the loader supplies ones-filled placeholders. A
   second RMS pass over an already mixed-and-normed vector is a different
   function, and ones-weights do not make it identity. The low-rank path must
   skip it.

4. **`hc_norm` dropped the offset-from-1.** *(Found and fixed in phase A —
   the gate earning its keep before a single golden was compared.)*
   `hyper_connection.cu` hand-rolled its grouped norm as `x * rms * w`.
   `Qwen4ExpTextRMSNorm.forward` is `normed * (1.0 + weight)` with the
   parameter initialised to **zeros** — Gemma's convention — while the
   `Qwen4ExpTextRMSNormGated` used by the GDN block beside it is the ordinary
   `weight * normed` initialised to ones. The checkpoint settles it: every
   plain-RMSNorm tensor centres near 0 and the gated GDN norm centres at 0.97.

   | tensor | mean | std |
   |---|---|---|
   | `layers.3.self_attn.q_norm` | 0.2833 | 0.0610 |
   | `layers.3.self_attn.indexer.q_layernorm` | −0.0372 | 0.0651 |
   | `layers.0.attn_hyper_connection.hc_norm` | −0.0635 | 0.4729 |
   | `layers.1.ple.norm_key` | −0.1067 | 0.0841 |
   | `layers.0.linear_attn.norm` *(gated)* | **0.9668** | 0.0326 |

   For `w ≈ 0` the missing offset is a near-null mix: finite, plausible,
   wrong. Measured against the reference it is `max|diff| = 4.65`.

   Atlas already dispatches this globally through
   `ships_vanilla_norm_weights`, which correctly leaves `qwen4_exp` on the
   offset-from-1 path — so `q_norm`/`k_norm` were never affected. Only the
   hand-rolled norm inside this kernel was. **The same offset applies to
   PLE's `norm_key` / `norm_query` / `norm_conv` in phase D.**

### A fifth, found during bring-up: the mHC diagnostic lies

`diag_norm` decodes its buffer as **BF16** (`vec![0u16; n]`). The mHC highway
is **FP32** — `hc_streams` is sized `m * hc_mult * h * 4` precisely because
BF16 storage collapses generation at scale. Every diag pointed at
`hc_streams` was therefore reading an f32 buffer as u16 pairs, which prints
alternating plausible/garbage values and a `norm=NaN`:

```
DIAG V4-prefill L0 hc_post-ffn: norm=NaN max=1.9e38
  first4=[450560.0000, 0.0583, -1.18e36, -0.0074]
```

The alternation is the tell — high half of each f32 decodes to something
plausible, low half to noise. On a **healthy** highway. This is pre-existing
on the DeepSeek-V4 path, and it cost a bring-up cycle chasing a NaN that was
never there. The twelve `hc_streams` diags now use `diag_norm_f32`, which
already existed beside it for `post`.

---

## 1. Why this is sequenced serially

The instinct is to fan out: attention mHC, GDN mHC, and PLE touch different
files. Three reasons not to.

1. **There is no testable intermediate state.** `layer_types` interleaves
   GDN and attention 3:1 on a shared 10240-wide highway buffer. Finish only
   the attention half and nothing runs; finish only the GDN half and nothing
   runs. Two workstreams that cannot each be verified are one workstream with
   a merge conflict in the middle.
2. **One GPU, one Atlas instance.** `--gpu-memory-utilization` reserves its
   whole fraction, and this model fits at 0.80 with ~0.4 GB of KV to spare.
   Any two streams that need to *serve* the model are serialized by the box
   whatever the branch topology says.
3. **This failure class is silent.** Every open item — the mHC mix, the PLE
   hash, the cross-attention gate — has a plausible-but-wrong implementation
   that produces fluent text. Serial-with-a-gate is how each one gets pinned
   to a number before the next lands on top of it.

**What does parallelize**, and is scheduled to: phase A's goldens are CPU and
checkpoint I/O, phase F is a single serve run reading a log, and phase G's
semantics read is pure reference-reading. All three overlap the Rust work
without contending for anything.

---

## 1.5 QSA is a requirement, and phase F is what unblocks it

The indexer is **not** an optional long-context optimization to bolt on later.
The model was trained with selection; dense attention above the budget is a
different function, and the whole point of a 262K-context model is the context.

It is sequenced after the first token for one reason, and it is not
prioritization: **the box currently fits `--max-seq-len 2048`.** Below the
budget the indexer is mathematically inert —
`topk(min(block_topk, num_complete_blocks))` with `block_topk = 2048/4 = 512`
selects every block — so at the context this model currently fits there is no
observable difference between a correct QSA and no QSA at all. It cannot be
tested, which means it cannot be trusted.

**So phase F is a hard prerequisite for phase G, not a side quest.** Closing
the ~11 GB resident gap is what buys a context long enough for selection to
engage, and therefore what makes QSA verifiable. That is why F runs in
parallel from the start rather than at the end.

### Threaded in now, so G is a wiring job and not a rewrite

Each of these is cheap today and expensive to retrofit:

- **Load the indexer weights now.** `index_qk_proj[640,2560]` (fused q 4×128 +
  k 1×128), `indexer.q_layernorm[128]`, `indexer.k_layernorm[128]`, on the 12
  full-attention layers. 3.3 MB/layer, ~39 MB total — negligible against 74.8
  GB, and skipping them now means redoing the loader, the memory budget and
  the alloc ledger later. **They must carry the offset-from-1 norm
  convention** (defect 4): `q_layernorm` measures mean −0.0372.
- **Keep `index_compress_ratio` off `compress_ratios`.** Already done. This is
  the switch: a non-empty `compress_ratios` turns on `probes.compressed_attn`
  and dispatches DeepSeek-V4's compressor — a *different mechanism*. G flips
  Qwen's own path on without ever routing through V4's.
- **Refuse above the budget rather than attend densely.** The refusal site is
  where G's entry point goes, so it doubles as the seam.
- **Settle the semantics question in tandem** (phase G below). It is a
  reference read that touches no code the other phases touch.

---

## 2. Phases

### A — Module goldens (the gate)  · small · **no GPU**

`ops::hc_pre_lowrank` / `hc_post_lowrank` / `hc_head_lowrank` and
`hyper_connection.cu` were written from the reference and have been compared
to nothing. Validate before wiring, not after — otherwise every phase-B and
phase-C bug is debugged against an unproven kernel.

- [x] `hc_golden.py` — runs the real `Qwen4ExpTextGatedResidual` on real
      checkpoint weights at all **three** sites (`layers.0.attn_`,
      `layers.0.mlp_`, and the model-level `hyper_connection_mixer`, which is
      `use_combine=False` and has no `block_inject_weight`). Dumps
      `mixed_input` / `hyper_input` / `injection_weights`
- [x] **The reference is the shipped one.** `transformers` 5.16.1 carries
      `qwen4_exp` natively and is **byte-identical** to
      `ref/modeling_qwen4_exp.py`, so the golden runs against the real module,
      not a vendored transcription
- [x] Defect 4 above, caught here
- [x] **Rust probe: PASSED.** `hyper_connection_lowrank_tests.rs` launches all
      four entry points against the golden on the real GPU:
      `attn` mixed cos 0.999998528 (max|diff| 1.27e-2, one BF16 ulp at |x|=6.7),
      injection_weights 2.4e-7, `mlp` cos 0.999998644,
      `hyper_connection_mixer` cos 0.999998586, `hc_post` **bit-exact (0.0)**
- [ ] `ple_golden.py` — same for `Qwen4ExpTextPLELayer`: gate pre/post signed
      sqrt, `gated_value`, conv output
- [ ] Rust probe (`#[ignore]` GPU test) loading the `.npz`, launching the four
      entry points, reporting max-abs and cosine per output
- [ ] **Acceptance: `hc_pre_lowrank` within BF16 tolerance of the reference,
      or the kernel is wrong and phase B does not start**

The grouped-RMSNorm detail (`group_size = hidden`, four independent 2560-wide
norms inside the 10240 vector) is the single most likely kernel error, so the
golden's fixed input deliberately gives the four streams **unequal** scales
(0.25 / 1 / 4 / 16). With equal scales a global RMS agrees with the grouped
one and the bug hides; with these, the wrong reading is `max|diff| = 15.2`.

`hc_golden.npz` (0.9 MB — inputs and expected outputs) is tracked;
`hc_golden_weights.npz` (79 MB, pulled verbatim from the checkpoint) is
gitignored and regenerated by the same script.

    /path/to/venv/bin/python -u bench/qwen4_exp/hc_golden.py

### B — mHC on the attention layers · medium

Kernel and ops exist; this is dispatch plus the three defects above.

- [ ] Resolve variant once at init (`HcVariant::{Sinkhorn, LowRank}`) rather
      than branching `site.lowrank.is_some()` at 23 call sites
- [ ] Route the 23 sites — `prefill_inner.rs` (8), `decode_inner.rs` (8),
      `multi_seq/mod.rs` (7) — through one dispatch wrapper per entry point.
      Uniform signature, not 23 if/else blocks; DeepSeek's `comb` argument has
      no low-rank counterpart and must not leak into the shared shape
- [ ] Skip `rms_norm(hidden, input_norm)` under `LowRank`
- [ ] Move `hc_expand` off `attn_layer_idx == 0` onto **model layer 0** —
      which is GDN, so ownership moves to phase C's entry path
- [ ] Fix `hc_head` to fire on the last **model** layer (47 = attention,
      `attn_idx` 11), not `attn_layer_idx + 1 == num_hidden_layers`
- [ ] `deepseek_v4_mtp.rs`'s 2 sites: leave on Sinkhorn, assert not LowRank

### C — mHC on the GDN layers · large · **critical path**

The SSM prefill **fuses its residual adds into its norms**:

```
rms_norm_residual(hidden, input_norm) -> normed, residual     # step 1
  ... block ...                       -> out_proj_buf         # steps 2-10
residual_add_rms_norm(hidden, out_proj_buf, post_attn_norm)   # step 11
ffn.forward_prefill(norm_output)                              # step 12
residual_add(hidden, moe_output)                              # step 13
```

Under mHC the **highway is the residual**, and the block output must reach it
through `hc_post`. Steps 1, 11 and 13 double-count. So this is not a wrapper
around `prefill_inner` — it is a second entry path, exactly as the attention
layer has `prefill_inner_hc` beside `prefill_inner`.

Do it by **extraction, not duplication**: steps 2-10 are ~270 lines that
already read `normed` and write `out_proj_buf` and touch the residual nowhere.

- [x] Extract steps 2-10 verbatim into `prefill_block(normed_in)` ->
      `trait_prefill_block.rs`. **Verified BYTE-IDENTICAL** by diffing the
      moved span against `git show HEAD:` — the only lines dropped are a dead
      `t0` reassignment clippy flagged as never read. Pure code motion, so a
      GDN regression on any existing model could only be a transcription error
- [x] `prefill_inner_hc` (`trait_prefill_hc.rs`): `hc_expand` if model layer 0
      -> `hc_pre` -> `prefill_block` -> `hc_post` -> `hc_pre` -> `ffn` ->
      `hc_post`. No `input_norm`, no `post_attn_norm`, no `residual_add`
- [x] Decode twin (`trait_decode_hc.rs`) — simpler, because `ssm_forward` was
      already a residual-free function of `normed`
- [x] `hc_expand` ownership moved here, and the SSM layer grew the three
      kernel handles, gated on `config.hc_mult > 0` so a plain GDN model
      issues no lookup and leaves no row in the startup audit
- [x] `decode_batched` / `decode_multi_seq` / `decode_verify_multi` refuse via
      `refuse_batched_under_hc` — C=1 only, stated
- [x] `ensure_no_unwired_hc` retired; what is left is the batched refusal
- [x] **PLE refused at LOAD** unless `ATLAS_QWEN4EXP_NO_PLE=1`, with the
      warning naming what is wrong and why

**Milestone: greedy generation with PLE stubbed.** Output is *wrong* — model
layer 1's injection is missing — so it stays behind an explicit
`ATLAS_QWEN4EXP_NO_PLE=1` that logs a loud warning and is refused by default.
It is a diagnostic that proves the mHC spine end to end, not a result.

### D — PLE n-gram injection · large

Spec in `ARCHITECTURE.md` §2 and §4. The row cache, pinned arena, deferred
load and pre-flight exclusion all transfer from #746. **The ID computation
does not** — LongCat is a polynomial rolling hash, Qwen is SplitMix64.

- [ ] Read `layer_multipliers` `[3]` I64 from the checkpoint; use
      `_build_layer_multipliers` only as a cross-check, never as the source
- [ ] 16 head ranges via `ngram_heads_offsets` / `ngram_heads_vocab_sizes`;
      160 dims each, **concatenated** to 2560 — not LongCat's sum-of-projections
- [ ] `key_proj[10240,2560]`, `value_proj[2560,2560]`, `norm_query` /
      `norm_key` / `norm_conv[10240]`
- [ ] Gate kernel: per-stream dot / sqrt(H), **signed sqrt**
      (`sign(g) * sqrt(max(|g|,1e-6))`), sigmoid, broadcast-multiply
- [ ] Depthwise conv1d, `groups=10240`, `kernel_size=4`, **`dilation=3`** ->
      9-step state; SiLU; add
- [ ] Per-sequence conv state for decode (9 steps x 10240) — new state, sized
      into the KV/state budget
- [ ] Inject into the 10240 highway **before** model layer **1**'s attn hyper-connection — `ple_layer_ids` is 1-INDEXED (`ple_layer_ids.index(layer_idx + 1)`), so `[2]` means `layer_idx == 1`, and the checkpoint confirms it: the tensors are at `layers.1.ple.*`
- [ ] Bit-exactness harness for the IDs, in the shape of #746's
      `ngram_parity.py` — a wrong hash returns valid rows from a 320M-row
      table and nothing in the log ever says so

### E — End to end · medium

- [ ] Layer-slice golden (first N layers, `bench/ngram_ref/slice_*.py` shape).
      The full 180B reference does not fit on this box in torch; slices do
- [ ] Stage-by-stage compare: embed -> PLE -> L0 GDN -> L3 attention -> mixer
- [ ] Greedy smoke, `max_tokens >= 250`
- [ ] `logit_quality.py` against the slice golden — top-1, top-k overlap, KL
- [ ] Perf pass and the alloc-ledger recheck at the final resident footprint

### G — QSA indexer · medium-large · **required; gated on F**

The 12 full-attention layers select over their KV with a learned indexer
rather than attending densely. Sequenced here because it is unverifiable
below the budget (§1.5), not because it is optional.

Runs in tandem from the start — it is a reference read, not an edit:

- [ ] Read `Qwen4ExpTextQSAIndexer` (`ref/modeling_qwen4_exp.py` L611
      onward) and write the selection semantics down beside §3 of
      `ARCHITECTURE.md`, the way §1 and §2 were written down
- [ ] **Answer the open question: does Atlas's DeepSeek CSA selection match
      Qwen's?** Atlas has `index_n_heads` / `index_head_dim` / `index_topk`,
      `csa_compress`, `prefill_attn_compressed` and `prefill/cache_skip_v4.rs`.
      If the block scoring and top-k semantics agree, G is wiring; if they
      differ, G needs its own kernel and that must be known before F lands,
      not after

Then, once F has bought a testable context:

- [ ] Wire selection on the 12 full-attention layers; `indexer_budget = 2048`,
      `compress_ratio = 4`, `block_topk = 512`
- [ ] Prefill and decode both — decode selects against the whole cache
- [ ] **Golden it the way phase A goldened mHC**: run the reference indexer on
      real weights at a length ABOVE the budget and compare the selected block
      sets, not just the attention output. A selection that is wrong but
      plausible gives fluent text; a selection compared set-against-set does
      not hide
- [ ] Remove the above-budget refusal only when that golden passes

### F — Memory · small · **runs alongside A; PREREQUISITE FOR G**

Resident is ~85 GB against 73.33 GB of tensor bytes; the model fits only at
`--max-seq-len 2048` with ~0.4 GB of KV. 150,528 of 302,488 tensors are
<=4-byte scalars, but skipping 74k of them recovered 0.8 GB, not the 13.5 GiB
a 64 KB granule predicts — so the allocator pools small blocks and the gap is
somewhere else.

- [ ] Read the alloc ledger table. Do not estimate again
- [ ] Re-check `tag_alloc_owner` coverage on the qwen4_exp loader arms
- [ ] Target: enough headroom for a useful context, not 2048

---

## 3. Scope calls for v1

Stated rather than asked; all reversible, all narrowing.

| | v1 | why |
|---|---|---|
| MTP | **dropped** | #753 item I; saves 4.7 GB and substantial work |
| QSA indexer | **sequenced, not dropped** | required (§1.5); unverifiable until F buys a context above 2048. Weights load NOW |
| batched / multi-seq decode | **refused** | C=1 proves correctness; concurrency is a perf question |
| vision tower | **text only** | `qwen3_vl.rs` is believed to cover it; untested, so refuse rather than guess |
| context > 2048 | **refused** | `index_compress_ratio` is recorded for exactly this refusal |

---

## 4. Critical path

```
A ──> B ──> C ──> [milestone: generates, PLE stubbed] ──> D ──> E
                                                          (correct token)
F ─────────────────────────────────────────────> G
(memory, from the start)          (QSA — needs F's context to be testable)
G's semantics read runs from the start too; only its wiring waits on F.
```

A is the gate. C is the long pole. D is where the silent-failure risk
concentrates, which is why it lands last and behind its own harness. F and G
are the long-context track: they are not after the correctness track, they
run beside it, and G is a requirement rather than a follow-on.

**Two independent tracks, and that is the whole of the parallelism available**
— A/B/C/D/E all touch the same highway buffer and the same GPU, so they are
serial with each other; F and G touch neither until G's wiring, so they run
alongside throughout.
