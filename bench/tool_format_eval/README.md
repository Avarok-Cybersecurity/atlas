# Tool-format eval harness

Quantifies and root-causes the multi-turn tool-call **format** failure on
Qwen3.6-35B (the "narrate-then-malformed-`<parameter>`" degradation that breaks
agentic tool calling under `thinking_in_tools=true`). Built to answer one
question with statistics instead of N=10 guesswork:

> Is the failure in **parsing**, or in **model inference/decoding**?

## The three layers

| Layer | Script | Needs | Answers |
|-------|--------|-------|---------|
| 1 — behavioral rate | `harness.py` | Atlas only | failure **rate** + Wilson 95% CI across configs |
| 2 — failure classifier | `harness.py` | Atlas only | per-turn **parsing vs generation** bin (from raw text) |
| 3a — logit decision probe | `logit_probe.py` | Atlas only | does Atlas's own next-token dist **rank** `<tool_call>` at the decision point? |
| 3b — dump localization | `dump_analyze.py` | `ATLAS_LOGIT_DUMP` | **model** vs **Atlas-processor** vs **sampler** (within Atlas) |
| 4 — Atlas↔vLLM diff | `compare_endpoints.py` | a running vLLM endpoint | **Atlas inference faithful** vs **forward-pass/quant divergence** |

Layers 1–3a run on this box with no extra deps (stdlib `urllib`). Layer 4 needs a
vLLM reference dump (this box has no torch/transformers).

## Run

```bash
# Layer 1+2 — rate + classifier (N per cell; 50 gives ~±14% CI, 200 ~±7%)
python3 bench/tool_format_eval/harness.py --n 50

# Layer 3a — is the forward pass favoring the wrong token? (parser-free)
python3 bench/tool_format_eval/logit_probe.py --runs 12 --top 10 --thinking on

# Layer 3b — localize within ONE Atlas dump (model vs bias vs sampler)
ATLAS_LOGIT_DUMP=/tmp/atlas.jsonl ./target/release/spark serve <model> --tool-call-parser qwen3_coder ...
#   then send exactly ONE failing tool-turn request, stop the server, then:
python3 bench/tool_format_eval/dump_analyze.py single /tmp/atlas.jsonl

# Layer 4 — definitive: Atlas vs vLLM forward-pass divergence (just needs a vLLM
#   OpenAI endpoint; greedy top_logprobs diff, no dumps / no token-id alignment)
python3 bench/tool_format_eval/compare_endpoints.py \
    --atlas-url http://localhost:8888 --atlas-model nvidia/Qwen3.6-35B-A3B-NVFP4 \
    --vllm-url http://<host>:8000 --vllm-model Qwen/Qwen3.6-35B-A3B-FP8 \
    --thinking on --top 8
```

## How to read it

- **Layer 2**: failures dominated by `malformed_opener` ⇒ the model emitted the
  wrong dialect itself ⇒ **generation/decode**, not parsing. `parser_miss`
  dominated ⇒ **parsing** (model was fine, Atlas dropped it).
- **Layer 3a**: if `<tool_call>` is rarely top-1 at the first post-`</think>`
  token, Atlas's forward pass disfavors the structural opener ⇒ decode/inference.
- **Layer 3b**: `bias`-flips ⇒ Atlas's additive processor stack (think-suppression,
  attractor, WS mask) is steering — vLLM has none of these. Otherwise the wrong
  token is the model's own argmax ⇒ forward-pass/quant ⇒ run Layer 4.
- **Layer 4**: `raw_topk` top-1 **diverges early** from vLLM ⇒ Atlas's forward
  pass (quant kernels / KV) is numerically off. `raw_topk` **matches** vLLM but
  Atlas still picks differently ⇒ the divergence is Atlas's `bias`/sampler, not
  the model.

## Findings to date (2026-06-16, NVFP4 Qwen3.6-35B)

- **Layer 1 (N=50/cell):** thinking ON = 43/50 (tools=2), 29/50 (tools=4); thinking
  OFF = 50/50, 50/50. Degrades with tool count ⇒ context-load effect.
- **Layer 2:** ZERO `parser_miss` in 200 turns ⇒ **parsing exonerated**. Failures
  are `malformed_opener` + `runaway`. Captured failure dialects: `<parameter
  name=…>` (attribute-XML) AND `` ```json {"command":…} `` `` (hermes JSON) — the
  model emits the wrong tool *dialect*, not the qwen3_coder `<tool_call>` XML.
- **Layer 3a:** `<tool_call>` opener TOP-1 at the decision point in **0/12** runs;
  model deterministically narrates first.
- **Layer 3b (ATLAS_LOGIT_DUMP, 229 steps incl. a failing run):** **bias-flips = 0**
  in every block (bias applied on every step but never flipped the argmax) ⇒
  **Atlas additive-processor stack exonerated**. 215/229 picks are the model's own
  raw argmax; 14 are sampler (min_p/penalty, ~15% noisy 2nd-order effect).

- **Parser swap (hermes vs qwen3_coder, N=30/50):** thinking-ON validity is
  comparable and still flaky (hermes 21/30, 24/30; qwen3_coder 43/50, 29/50;
  CIs overlap), thinking-OFF 100% for both. Parser choice does NOT fix it ⇒
  further confirms the defect is not parsing.

**Verdict so far:** not parsing, not Atlas's bias processors, not parser choice.
The malformed tokens are the model's OWN raw-logit argmax under Atlas's forward
pass (it prefers narration + JSON/attr-XML over qwen3_coder XML when thinking).
The single OPEN question is whether that distribution is faithful to the true
model or a forward-pass/quant divergence — **Layer 4 (`compare_endpoints.py`)**,
which needs a running vLLM endpoint.
