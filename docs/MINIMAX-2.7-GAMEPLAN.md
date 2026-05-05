# MiniMax M2.7 — Atlas Support Gameplan

**Status**: Draft, pre-implementation.
**Target**: Feature parity with other inference engines on MiniMax M2.7 plus
Atlas-native MTP and CUDA kernels for GB10. Goal tok/s on a single Spark:
match or beat the Qwen3.5-122B-A10B path (~50 tok/s with MTP+EP=1, ~54
tok/s EP=2) for a similarly-sized model (229B total / 10B active).

## 1. Architecture snapshot (from published `config.json`)

M2.7 is architecturally identical to M2.1 — same `minimax_m2` `model_type`,
same dims. Only the weights differ. So any work done for one drops straight
onto the other.

| Field | Value | Atlas-side implication |
|---|---|---|
| `num_hidden_layers` | 62 | All full-attention (no SSM/Mamba) |
| `attn_type_list` | `[1] * 62` | Confirms uniform attention stack |
| `hidden_size` | 3072 | Matches Qwen3.5-35B/122B/Coder-Next shape |
| `head_dim` | 128 | Standard GQA head |
| `num_attention_heads` | 48 Q | New shape — need attention kernel tuning |
| `num_key_value_heads` | 8 KV | Ratio 6:1 GQA |
| `rotary_dim` | 64 | **Partial RoPE** — only first 64 of 128 head dims rotated. New kernel path. |
| `rope_theta` | 5,000,000 | Long-context rope base |
| `use_qk_norm` + `qk_norm_type=per_layer` | true | Per-layer learned Q/K norm weights |
| `num_local_experts` | 256 | Exact match to Qwen3.5-122B MoE dispatch |
| `num_experts_per_tok` | 8 top-k | Match |
| `intermediate_size` | 1536 | MoE FFN width per expert |
| `shared_intermediate_size` | 0 | **No shared expert** (differs from Qwen3.5) |
| `scoring_func` | sigmoid | **New routing** — most Atlas MoE is softmax/topk-bias |
| `use_routing_bias` | true | Needs `e_score_correction_bias` loaded + applied |
| `hidden_act` | silu | SwiGLU FFN |
| `rms_norm_eps` | 1e-6 | Standard |
| `max_position_embeddings` | 196,608 | 192k context |
| `vocab_size` | 200,064 | BPE tokenizer |
| `dtype` (M2.7) | bfloat16 | Base weights BF16 before quant |
| `quantization_config.fmt` | `float8_e4m3fn` | Native FP8 |
| `quantization_config.weight_block_size` | `[128, 128]` | **Block-scaled FP8**, same shape as Atlas Qwen3-Coder-Next FP8 path |
| `modules_to_not_convert` | gate, correction_bias, lm_head | Keep BF16 for routing + output |
| `use_mtp` | true | MTP speculative decoding |
| `num_mtp_modules` | 3 | **3 draft modules** (vs 1 in Qwen3.5) |
| `mtp_transformer_layers` | 1 | Each draft is a single transformer layer |

**Params**: 229B total, ~10B active (256×8/256 × 62×1536×3072-ish FFN + attn).
Fits in 120 GB at FP8 with room for KV cache at 64k context and below.

## 2. Reuse vs new work

Going module by module in `crates/spark-model`:

### Reuse (large code paths already in Atlas)

- **BPE tokenizer** — `tokenizers` crate handles it. Model ships
  `tokenizer.json`. Coherence harness already validates this path.
- **MoE dispatch kernels** — Atlas already has 256-expert top-8 for
  Qwen3.5-122B. Reusable as-is:
  `kernels/gb10/qwen3.5-122b-a10b/nvfp4/moe_*.cu` and the FP8 variant from
  `qwen3-coder-next-fp8`.
- **GQA decode / prefill attention** — head_dim=128 is the standard path.
  Current Atlas Qwen3.5 attention kernels handle head_dim=128 GQA; the
  new Q/KV shape (48/8) is a tile-shape retune, not new math.
- **FP8 block-scaled GEMM** — `weight_block_size [128,128]` identical to
  Qwen3-Coder-Next FP8. `atlas-gemm` path reusable.
- **RMSNorm / SiLU / SwiGLU FFN** — stock.
- **MTP scheduler** — existing MTP verify/accept loop in `spark-runtime`
  works layer-agnostically; needs adapter for 3 draft modules instead of 1.
- **OpenAI-compatible server + chat template** — stock `tokenizer.rs`.

### New work (real kernels + loader pieces)

1. **Partial RoPE kernel** (`rotary_dim=64` on `head_dim=128`)
   - Current Atlas RoPE assumes `rotary_dim == head_dim`. MiniMax rotates
     only the first 64 dims of each 128-wide head and leaves the trailing
     64 untouched.
   - Either: (a) new `rope_forward_partial` kernel, or (b) parametrize
     existing `rope_forward` with a `rope_dim` arg and have `rope_dim <
     head_dim` skip the non-rotated tail. Option (b) is cleaner — fewer
     kernel specializations.
   - Reference: vLLM's `rotary_embedding.py` handles this via
     `partial_rotary_factor`; llama.cpp's ggml does it via `n_dims` arg.

2. **Per-layer QK norm** (`use_qk_norm=true, qk_norm_type=per_layer`)
   - Atlas has QK norm for Qwen3.5 hybrid SSM models, applied on
     attention-only layers. MiniMax applies it per layer with learned
     weights `q_layernorm` / `k_layernorm` each of shape `[head_dim]`.
   - Add loader entries, apply RMSNorm on Q and K before RoPE.
   - Tiny new work — fused into existing prefill/decode attention.

3. **Sigmoid MoE routing with correction bias**
   - `scoring_func=sigmoid` means scores are `sigmoid(gate_logits)` not
     `softmax`. Atlas Qwen3.5 MoE currently does softmax + top-k.
   - `use_routing_bias=true` applies `e_score_correction_bias` before
     top-k (loss-free balancing technique from DeepSeek-V3 papers).
   - New code path in `moe_router.rs`: compute sigmoid of logits, add
     correction bias, take top-8 by the biased scores, but dispatch with
     **raw sigmoid scores** (not biased ones) — this is the DeepSeek
     trick.
   - Reference: vLLM `fused_moe/fused_moe.py` — `fused_grouped_topk` with
     `scoring_func="sigmoid"` path.

4. **3-module MTP draft loop**
   - Existing Atlas MTP runs one draft module per decode step. MiniMax
     produces 3 drafts per step (3 sequential transformer layers, each
     predicting one future token).
   - Scheduler change in `spark-runtime`: draft 3 tokens with 3 MTP
     modules, verify in one parallel forward pass through the 62 main
     layers (same as K=3 spec decode today). Accept up to 3 tokens per
     step.
   - Reference: DeepSeek-V3 MTP paper, M2 modeling code at
     `modeling_minimax_m2.MiniMaxM2ForCausalLM`.

5. **Model loader** — `crates/spark-model/src/minimax_loader.rs`
   - Parse `config.json` → Atlas `ModelConfig` with new fields:
     `rope_dim`, `qk_norm_per_layer`, `moe_scoring_func`, `moe_routing_bias`,
     `num_mtp_modules`.
   - Weight prefix mapping. Loader pattern copies cleanly from
     `qwen3_loader.rs`.

6. **`kernels/gb10/minimax-m2-229b/fp8/MODEL.toml`**
   - Declares target tuple `(sm_121, minimax-m2-229b, fp8)`.
   - Points at reused MoE kernels + new partial-rope + QK-norm kernels.

### Out of scope for v1

- NVFP4 variant — ship FP8 first since that's what MiniMax published.
- EP=2 cross-node — same story as Qwen3.5-122B. Can extend later with
  expert-parallel sharding once single-node works.

## 3. Speedups to port

Survey of what other engines do well with MiniMax:

### vLLM

- **FlashAttention-3 variant** for head_dim=128 GQA at long context.
  Atlas already has hand-tuned GB10 attention, but worth diffing their
  prefill kernel at 32k+ to see if they handle partial-RoPE more
  efficiently than a parametrized path.
- **CUDA graphs for decode** — vLLM captures full decode-step graphs.
  Atlas already does this for Qwen3.5; needs to be enabled for
  MiniMax once the model path is up.
- **Prefix caching** — vLLM caches KV blocks across requests with same
  system prompt. High-value for agentic workloads (repeat system
  prompts). Atlas has radix-cache/Marconi for SSM; attention-only
  prefix caching isn't wired yet — adjacent win worth bundling.
- **Chunked prefill** — splits long prefill into chunks overlapped with
  decode. Atlas has this for Qwen3-Next but not enabled for every
  model. Turn on for MiniMax from day 1.

### SGLang

- **RadixAttention + prefix cache** — SGLang's radix tree tracks prefix
  overlap across requests. Same concept as vLLM prefix caching but
  more aggressive. If we bring up a generic attention prefix cache for
  MiniMax it subsumes both.
- **FlashInfer** for decode attention. Atlas already beats FlashInfer
  on GB10 per our benchmarks; not a port target.
- **Speculative decoding verify** — their verify kernel batches draft
  tokens efficiently. Worth comparing against our MTP K=3 verify when
  we extend from K=1/K=2 to K=3.

### llama.cpp

- **Q4/Q6 quantization** — not relevant (we're FP8 native).
- **Quantized KV cache** — llama.cpp has FP8 KV cache for attention
  models. Atlas has `--kv-cache-dtype fp8/nvfp4/bf16` already; for
  MiniMax default to FP8 to match the quant scheme.
- **Streaming prefill** — chunk size tuning for agentic. Less
  architectural, more config tuning.

### Things nobody else has yet that Atlas should

- **MTP K=3 native verify** — M2 ships 3 MTP modules; no engine I've
  seen has enabled K=3 verify for it yet (vLLM's PR still in review as
  of this writing). Atlas already has K=2 verify on Qwen3.5 — extending
  to K=3 is a win we can claim.
- **FP8 block-scaled MoE with on-chip dequant** — Atlas Qwen3-Coder-Next
  already does this. Applying it to MiniMax MoE should give the same
  ~20-30% throughput lift we saw on Coder-Next vs FP8 weight-only.

## 4. Milestones + acceptance criteria

Each milestone is a branch, lands a coherent slice, and must pass
acceptance before the next starts.

### M1 — Shape-compatible loader + attention (no MTP)
- Weight loader: `minimax_loader.rs`, weights materialize, vocab
  matches `tokenizer.json`.
- Model runs 1-token decode with greedy sampling at BF16 (no quant).
- Acceptance: "Hello" round-trips correctly. Single token verified
  layer-by-layer against HF reference.

### M2 — Partial RoPE + per-layer QK norm
- New `rope_forward` handles `rope_dim < head_dim` (additive param,
  no kernel duplication).
- Q/K RMSNorm with per-layer learned weights.
- Acceptance: 512-token greedy generation matches HF reference
  byte-for-byte on the first 16 tokens for 5 seed prompts.

### M3 — Sigmoid + correction-bias MoE routing
- New `moe_router_sigmoid` path. Biased scoring for top-k selection,
  raw sigmoid for dispatch weighting.
- Acceptance: routing distribution matches HF reference for 100 random
  tokens (same expert picks, same per-expert weights within 1e-4).

### M4 — FP8 block-scaled end-to-end
- Reuse Qwen3-Coder-Next FP8 path. New MODEL.toml targeting
  `minimax-m2-229b/fp8`.
- Acceptance: 2048-token greedy generation coherent. No NaN/garbage
  across 20 varied prompts (code, reasoning, multilingual).
- Throughput target: ≥40 tok/s decode at batch=1 on single Spark.

### M5 — MTP K=3 with 3 draft modules — **DEFERRED, UNTESTABLE**
Update 2026-04-14: scope was reduced to K=1/K=2 (K=3 not
required). More importantly, **both `MiniMaxAI/MiniMax-M2` and
`MiniMaxAI/MiniMax-M2.7` checkpoints ship zero MTP weights** — 96 103
tensors, max layer index = `num_hidden_layers - 1` = 61. No
`model.layers.62..64.*`, no `enorm` / `hnorm` / `eh_proj`, no `mtp` /
`draft` substrings anywhere. `modeling_minimax_m2.py` contains no MTP
forward either. The `use_mtp: True, num_mtp_modules: 3` config fields
are aspirational — the public weights don't satisfy them.

What landed anyway (scaffold only):
- `MultiModuleMtpHead` + `MultiModuleMtpState` in `layers/mtp_multi.rs`
- `WeightLoader::load_mtp_weights_multi` trait method with default
  single-module adapter
- `Vec<MtpWeights>` plumbed through factory → `TransformerModel::new`
  → proposer dispatcher (Some → `MtpHead`, Vec>1 → `MultiModuleMtpHead`)
- `MinimaxM2WeightLoader::load_mtp_weights_multi` probes the first
  expected MTP tensor; empty on tiny-random & both full checkpoints
  (i.e., speculative decoding cleanly disables), fail-fast bail when
  tensors ARE present (future weights release scenario)

Reopens if MiniMax publishes a separate MTP weights repo. Otherwise
closed as scaffold-complete, validation-impossible.

### M6 — Agentic path + prefix cache
- Prefix cache for attention-only models (generalizes from Marconi).
- Tool calling with Atlas's existing `tool_parser.rs` — MiniMax uses
  XML tool call format per their model card.
- Acceptance: Claude Code / OpenCode agentic session of ≥20 tool calls
  completes without error.

**Partial landing (2026-04-14)**: `MinimaxXmlParser` shipped in
`tool_parser.rs` as a new `ToolCallFormat::MinimaxXml` variant (CLI
name `minimax_xml`). Parses
`<minimax:tool_call><invoke name="X"><parameter name="K">V</parameter></invoke></minimax:tool_call>`
with a pre-pass that normalizes the outer `<minimax:tool_call>` tag
to `<tool_call>` so the shared scanning loop handles it, and a new
`parse_minimax_xml_call` inner parser for the `<invoke>`/`<parameter>`
attribute style. Four unit tests landed (single param, multi param,
content-prefix interleave, format→parse roundtrip). Not auto-assigned
via `tool_defaults.toml` — opt-in with `--tool-call-parser minimax_xml`.
Attention-only prefix cache still pending.

## 5. Sequencing

Do M1-M2-M3 serially (each unblocks the next). M4 can overlap with M2
once M1 is green. M5 starts after M4. M6 can start any time after M4.

Expected timeline: aggressive is 5 working days if model loader cleanly
inherits from Qwen3.5 path. Realistic with debugging: 8-10 days.

## 6. First PR scope

Keep PR #1 small and shippable:
- Add `minimax_m2` to `ModelConfig` parsing (no kernel changes).
- Add the loader skeleton (returns "not yet supported" on serve).
- Add MODEL.toml stub pointing at shared kernels.

This is the "fork the existing Qwen3.5 path" commit. From there each
milestone is a PR on top.

> Historical note: the original staging landed behind an
> `ATLAS_ENABLE_MINIMAX=1` env flag. That gate was removed once the M2.7-NVFP4
> EP=2 suite went green — `model_type = "minimax_m2"` dispatch in
> `crates/spark-model/src/factory.rs` is now the sole selector.

## Regression notes (2026-04-14)

**M2 + M3 reverted** (`4d71dc8` reverting `da9cd9c`). Applying the
three-way qk_norm dispatch + sigmoid MoE branches broke
Mistral-Small-4: outputs collapse to "…" (U+2026) token, pass-24
score 4/13 vs pass-22 13/13. Surprising because:

- Mistral takes the MLA early-return in both
  `qwen3_attention::decode.rs` (line 579) and `prefill.rs`
  (line 150) → never reaches the qk_norm_full branch.
- Mistral has `correction_bias_dev = None` → every new MoE dispatch
  falls through to the existing softmax path, identical to pre-commit.
- Mistral's `KERNEL.toml` shadows the common `gb10/nvfp4/KERNEL.toml`
  per build.rs line 296, so the `moe_topk_sigmoid = "moe_topk_sig"`
  mapping I added to the common file doesn't affect Mistral's
  module resolution.

Re-land plan for next debug session (needs the `tests/single_gpu_suite.py
mistral-small-4` harness to bisect):

1. decode.rs `q_norm_full` branch only (no `k_norm_full`, no MoE changes)
2. decode.rs `k_norm_full` branch
3. prefill.rs `q_norm_full` branches (both deinterleave paths)
4. prefill.rs `k_norm_full`
5. decode.rs primary MoE sigmoid dispatch
6. Batched MoE sigmoid dispatch sites
7. `moe_topk_sigmoid_batched` ops + ctor handle
8. `PagedPrefillArgs` refactor (optional cleanup; not needed for MiniMax)

Run Mistral after each commit. Whichever step introduces "…" is the
regression.

## 7. Open questions to resolve before starting

1. **MTP head structure**: Does each of the 3 MTP modules share the
   embedding matrix with the base model, or do they have their own?
   Reading `modeling_minimax_m2.py` will answer this.
2. **Correction bias**: Is it static (loaded from the checkpoint) or
   dynamic (updated per-token via some aux loss at training)? If static,
   it's just another weight tensor. If dynamic, we have extra state.
   Expected static per the DeepSeek-V3 paper this technique comes from.
3. **Tokenizer**: BPE, 200k vocab. Need to run coherence harness against
   their `tokenizer.json` to confirm encode-path compatibility with our
   existing tokenizer infra.
4. **Benchmarks target**: What's the vLLM number on DGX Spark FP8 today?
   Discord reported ~25 tok/s NVFP4 two-spark. Need a head-to-head
   single-spark FP8 number to size the target.

## 8. Signal to watch

Once M4 lands, run the same TTFT A/B we ran on Qwen3.5 (with fastokens
findings in mind — tokenization won't be the bottleneck). The expected
story at single-request batch=1:
- Decode tok/s ≥40 @ FP8, batch=1, ISL 1024
- Decode tok/s ≥55 with MTP K=3 enabled (85% accept target)
- TTFT ≤1s at ISL 1024, ≤8s at ISL 8192
- 64k+ ISL will expose attention prefill — probably where we're weakest
  initially. OK to address in M7 (long-context tuning pass) rather than
  gate M4 on it.

---

**Next concrete action**: once the remaining fastokens A/B runs finish
(Nemotron Nano→Super, then Mistral / Coder-Next), start M1: loader
skeleton + ModelConfig parsing. Will open a draft PR on a
`minimax-m2-support` branch off current `spec_ssm` HEAD.
