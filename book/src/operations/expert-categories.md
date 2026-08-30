# Expert Telemetry & Selective Expert Loading

A mixture-of-experts model routes each token to a handful of its experts.
Which handful is not arbitrary: prompts of a kind — Python, SQL, French
translation — reuse a recognizable subset. Atlas can measure that subset and
then serve from it alone, holding a fraction of the expert weights in memory.

Two steps, in order. The first measures; the second uses the measurement.

## Step 1 — Expert categorization

Start the server with expert telemetry enabled. It is a boot flag, not a
request parameter, because the routing capture has to be recorded into the
decode CUDA graph before the model is built:

```bash
spark serve Qwen/Qwen3.6-35B-A3B-FP8 --expert-telemetry
```

Any request can then ask for its own routing back:

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen3.6-35B-A3B-FP8",
  "messages": [{"role": "user", "content": "Write a Python function that reverses a string."}],
  "report_expert_metadata": true
}'
```

The response carries `usage.expert_activation`:

```json
{ "scope": "prefill+decode", "top_k": 8, "num_experts": 256,
  "tokens_routed": 2760, "unattributed_rows": 0,
  "decode_tokens_routed": 1880, "decode_unattributed_rows": 0,
  "layers": [ { "layer": 0, "experts": [4, 7, 11], "counts": [1, 1, 2],
                "mass": [0.09, 0.04, 0.21] } ] }
```

`counts` is how many routed token-slots chose each expert; `mass` is the
summed post-renormalization routing weight, which is what an expert set is
budgeted on. `Σcounts == tokens_routed × top_k` whenever every routed slot
carried weight — the check that separates a quiet prompt from a broken tap.

`scope` says which part of the request the numbers cover.
`"prefill+decode"` is the whole request; `"prefill"` means the serve could
only attribute the prompt. `decode_tokens_routed` is how much of
`tokens_routed` came from generation, and `decode_unattributed_rows` counts
decode positions that ran WITHOUT being attributed — MTP verify rows, which
are staged but not folded, because a rejected draft's routing belongs to a
token that was rolled back. On a speculating serve that number is non-zero
and tells you coverage is partial; on a non-speculating serve it is 0. Across
tool-loop turns a merge keeps the weaker scope, so a restarted serve cannot
launder partial coverage into a full-coverage claim.

The field defaults off and is absent unless asked for, so a consumer can
tell "this serve is not instrumented" from "this prompt used no experts".
Asking for it on a serve without `--expert-telemetry`, or on a dense
checkpoint, is a 400 naming which of the two is missing.

The `expert-categories` benchmark does this over a committed corpus of 2000
short prompts across 20 categories — 100 each, equal counts because EAS
normalizes by the corpus's category entropy — and reduces the result to a
table:

```bash
spark benchmark run expert-categories --url http://localhost:8000 \
  --model Qwen/Qwen3.6-35B-A3B-FP8
```

It writes two artifacts to `~/.atlas/runs/expert-categories/<run>/`: a
paste-ready `expert_categories.toml`, and `stats.json` with the full
per-expert distribution and the cross-category overlap. The overlap is worth
reading before trusting the table — on Qwen3.6-35B-A3B the most similar
category pairs are code-python/code-rust (0.72 Jaccard),
creative-writing/general-chat (0.62) and sql/tool-calling (0.59), which is
what a real register signal looks like.

Per layer and category it keeps the smallest expert set covering `coverage`
of the routing mass (default 0.90). Mass, not how often an expert was
chosen: an expert picked in every token at weight 0.02 contributes less than
one picked in a tenth of them at weight 0.5.

## The Expert Alignment Score

The same run reports **EAS-1.0**: how much of the uncertainty about which
category a prompt came from is resolved by watching which experts its tokens
routed to. 1.0 means expert identity determines the category; 0.0 means
routing says nothing about it.

Per layer it takes each category's KL divergence from the pooled expert
marginal — which averages to exactly the mutual information `I(C;E)`, so
"score each category then average" and "compute the mutual information" are
the same operation. It is then chance-corrected by a permutation null
subtracted from both numerator and denominator, because plug-in mutual
information over 256 experts is biased upward and an uncorrected ratio could
never reach the 0.0 floor it claims. The shuffle is at prompt level: tokens
within a prompt reuse experts, and shuffling tokens would understate the null
and flatter the score.

Scores are comparable across models **on one frozen corpus** and not across
corpora — the category taxonomy sets the ceiling, so the corpus hash is part
of the number's identity. Read EAS next to a quality delta, never alone: a
model whose experts perfectly encoded twenty arbitrary categories would score
1.0 and would probably be a worse language model, so the target is the best
EAS at no quality cost, not the highest EAS.

Measured on Qwen3.6-35B-A3B-FP8 (10 categories, prefill only): **EAS
0.17886**, every layer clearing the null. The per-layer curve is the useful
part — 0.06 at layers 0-2, peaking at 0.277 around layer 20, back to 0.09 by
layer 39. Routing is category-agnostic at both ends of the model. That is
both an argument for per-layer coverage budgets instead of one global
threshold, and a plain explanation of the quality cost below: at 0.18, most
routing is not about the category at all.

## Step 2 — Boot-time expert loading

Paste the emitted block into the model's `kernels/<hw>/<model>/MODEL.toml`
and **rebuild**. The table is read at build time, like `[dflash]`, because a
runtime box does not ship the kernels tree. Editing MODEL.toml also
invalidates that target's gate closure hash, which is intended — the
binary's expert set changed, so earlier benchmark records no longer describe
it.

Then serve from one category:

```bash
spark serve Qwen/Qwen3.6-35B-A3B-FP8 --expert-category code-python
```

A comma-separated list loads the **union** of those categories' experts — a
request does not announce its category, so serving several means holding what
any of them needs:

```bash
spark serve Qwen/Qwen3.6-35B-A3B-FP8 --expert-category code-python,translation,math
```

Unions are strongly sub-additive: those three need 2645, 2998 and 3198
experts alone (8841 if disjoint) but 5015 together, 49% of the routed set.
The boot log reports the union and what each category costs on its own.

Boot reports what it kept:

```
Expert category "code-python" (coverage 0.90): 2645 of 10240 routed experts
across 40 layers — 26% of routed-expert weights
```

Measured on that model: 18626 weight tensors instead of 64196, and 13.05 GB
of weight memory instead of 36.14 GB.

### What it costs

Real, and worth measuring before relying on it. At coverage 0.90 on the
`code-python` category, a Python prompt still answers on-task but degrades —
the reverse-a-string answer kept its signature and docstring and corrupted
the slice expression. A prompt from a different register degrades much
further: "Traduis en français: The weather is beautiful today." returned
`Il天气是 beautiful today.` against `Il fait beau aujourd'hui.` from the
full model.

So this is a memory/quality trade, not a free win. It is also not a cliff you
have to guess at: at the three-category union above (49% of experts) the same
French prompt returns `Il fait beau aujourd'hui.` — identical to the full
model — the Python answer is clean, and even an out-of-union legal prompt
answers coherently. The usable operating point on this checkpoint is
somewhere between 26% and 49%, and a coverage or union sweep brackets it.

Published work on the same model family reports the same shape: "Half the
Experts, All the Code" (arXiv:2607.16721) finds 50% keep statistically tied
with the base model on Qwen3.6-35B-A3B, and 25% keep needing LoRA plus router
self-distillation to recover half the gap. A 26% single-category serve is on
the far side of that knee, which is where our measured corruption sits.

Worth knowing what the failure actually is, because it rules out the obvious
fix. `norm_topk_prob` is on for this architecture, so masked experts
contribute nothing to the softmax denominator and the surviving weights still
sum to 1 — there is no magnitude deficit to rescale away. The damage is
SUBSTITUTION: when some of a token's true top-8 are absent, top-k backfills
those slots with experts the router ranked far lower, and hands them the
absent experts' weight. A confident wrong expert at full strength, which is
what a corrupted span in otherwise on-task text looks like.

### What is refused

Experts outside the category are not loaded at all, and the router is masked
so top-k cannot select one. Where a routing path is not masked, the layer
refuses by name rather than dereferencing a null expert pointer: expert
parallelism, the atomic-C4 and token-major decode variants, and MTP verify.
Boot refuses outright, with the fix named, for an unknown category, a model
with no table, a dense checkpoint, a table measured on a different
checkpoint, a layer keeping fewer experts than top-k selects, and a router
that scores zero-computation experts.
