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
{ "scope": "prefill", "top_k": 8, "num_experts": 256,
  "tokens_routed": 800, "unattributed_rows": 0,
  "layers": [ { "layer": 0, "experts": [4, 7, 11], "counts": [1, 1, 2],
                "mass": [0.09, 0.04, 0.21] } ] }
```

`counts` is how many routed token-slots chose each expert; `mass` is the
summed post-renormalization routing weight, which is what an expert set is
budgeted on. `scope` says which part of the request the numbers cover —
`"prefill"` today, since decode rows cannot yet be attributed to a sequence.
`Σcounts == tokens_routed × top_k` whenever every routed slot carried
weight, which is the check that separates a quiet prompt from a broken tap.

The field defaults off and is absent unless asked for, so a consumer can
tell "this serve is not instrumented" from "this prompt used no experts".
Asking for it on a serve without `--expert-telemetry`, or on a dense
checkpoint, is a 400 naming which of the two is missing.

The `expert-categories` benchmark does this over a corpus of 320 short
prompts in 10 categories and reduces the result to a table:

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

So this is a memory/quality trade, not a free win. Raise `coverage` and
re-measure to find a usable operating point, and expect traffic outside the
category's register to pay the most.

### What is refused

Experts outside the category are not loaded at all, and the router is masked
so top-k cannot select one. Where a routing path is not masked, the layer
refuses by name rather than dereferencing a null expert pointer: expert
parallelism, the atomic-C4 and token-major decode variants, and MTP verify.
Boot refuses outright, with the fix named, for an unknown category, a model
with no table, a dense checkpoint, a table measured on a different
checkpoint, a layer keeping fewer experts than top-k selects, and a router
that scores zero-computation experts.
