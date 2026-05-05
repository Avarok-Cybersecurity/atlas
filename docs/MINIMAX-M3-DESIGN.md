# MiniMax M2 — M3 Design: Sigmoid MoE Routing + Correction Bias

**Status**: Implementation-ready. All infrastructure exists.
**Acceptance (from gameplan)**: routing distribution matches HF reference
for 100 random tokens — same expert picks, same per-expert weights within
1e-4.

## Good news up front

Atlas already ships the exact kernel MiniMax needs.

- CUDA: `kernels/gb10/nvfp4/moe_topk_sigmoid.cu` (lines 22–80 implement
  the sigmoid + bias + top-k + unbiased-score-gather pipeline).
- Rust binding: `crates/spark-model/src/layers/ops.rs::moe_topk_sigmoid`
  (line 3410). Takes `(gate_logits, bias, ...)`.
- Consumer: Nemotron-H already uses this path (bring-up landed in an
  earlier pass). The kernel comment reads:

  > Nemotron-H uses sigmoid routing (NOT softmax like Qwen3/DeepSeek):
  >   scores = sigmoid(logits)
  >   selection = scores + bias   (bias affects WHICH experts, not their weights)
  >   indices  = topk(selection)
  >   weights  = scores[indices]  (pre-bias sigmoid scores)
  >   weights /= sum(weights)     (if norm_topk_prob)
  >   weights *= scaling_factor   (routed_scaling_factor, e.g., 2.5)

This is bit-for-bit the same pipeline as MiniMax's
`route_tokens_to_experts` (verified by reading
`modeling_minimax_m2.py` — quoted in the M1 smoke-test conversation).

## MiniMax-specific differences from Nemotron-H

| Field | Nemotron-H | MiniMax M2 | Impact |
|---|---|---|---|
| `num_experts` | 128 (Nano) / 512 (Super) | 256 | kernel MAX_EXPERTS=512 already covers |
| `top_k` | 6 (Nano) / 22 (Super) | 8 | kernel MAX_TOP_K=32 already covers |
| `norm_topk_prob` | true | true (weights /= sum(weights)) | identical |
| `routed_scaling_factor` | 2.5 (Super) | 1.0 (no scale in MiniMax config) | identical (pass 1.0) |
| `e_score_correction_bias` shape | `[num_experts]` | `[num_experts]` | identical |
| Bias storage precision | f32 | f32 (safe default; MiniMax config has no override) | identical |

Net: zero kernel changes, zero ops-binding changes. M3 is pure wiring
inside the MinimaxM2 MoE layer's forward path.

## What M3 actually ships

### 1. MoE layer dispatch (`crates/spark-model/src/layers/moe.rs`)

The existing MoeLayer implementation has two branches in its top-k
section: one calls `moe_topk_softmax` (Qwen3.5 path), one calls
`moe_topk_sigmoid` (Nemotron-H path). MiniMax reuses the sigmoid
branch; the dispatcher just needs to select it based on
`config.scoring_func == "sigmoid"` when the MinimaxM2WeightLoader
hands the layer off.

Minimal change: the layer ctor already accepts a `use_sigmoid: bool`
flag in its configuration struct (or a string "sigmoid" vs "softmax");
the loader sets it from `config.scoring_func`. Verify with a 5-line
diff of the Nemotron loader path to confirm the API.

### 2. Correction-bias weight loading (`crates/spark-model/src/weight_map.rs`)

Nemotron-H loads its routing bias under one of these prefixes depending
on vintage:
- `backbone.layers.{i}.mixer.e_score_correction_bias`
- `model.layers.{i}.mlp.e_score_correction_bias`

MiniMax puts the bias at:
- `model.layers.{i}.block_sparse_moe.e_score_correction_bias`

(Source: running `python3 -c "from safetensors import safe_open; f =
safe_open('model.safetensors', framework='pt'); print([k for k in
f.keys() if 'correction' in k])"` on the tiny-random checkpoint.)

One new branch in `load_moe_qwen35_fp8_experts` (or an equivalent
MiniMax-specific `load_moe_minimax` helper) that pulls the bias from
the MiniMax-style weight name. Keep it f32 — the kernel expects f32
and Nemotron's load path already handles the bf16→f32 cast path if we
ever hit a checkpoint that stores it in bf16.

### 3. Loader wire-up (`crates/spark-model/src/weight_loader/minimax.rs`)

Today's MinimaxM2WeightLoader stubs every method. M3 replaces the
`load_layers` stub with a real implementation that:

- For each of the 62 layers:
  1. Load attention weights (plain AttentionWeights + q_norm + k_norm).
  2. Load MoE weights via the new `load_moe_minimax` helper.
  3. Build a MoeLayer with `scoring_func = "sigmoid"`, pass the loaded
     bias tensor.
  4. Build a Qwen3AttentionLayer with `rope_dim = 64` (partial RoPE)
     and the per-layer qk_norm wired in.
  5. Combine into a TransformerLayer.

The attention side depends on M2. If M2 lands the qk_norm and partial
RoPE wiring first, M3 just adds the MoE half.

### 4. No unit-test harness change required

Existing moe_topk tests cover the sigmoid path (Nemotron). Add one
integration test that runs MiniMax-tiny through a single layer and
compares the expert indices + weights to HF's
`route_tokens_to_experts` output on the same input. Host-side unit
test using the tiny-random checkpoint — no 229B weights needed.

## Validation plan (acceptance criteria in detail)

1. **Unit — expert indices match**: For 100 random
   `[num_experts]` gate logit vectors and a fixed bias vector,
   `moe_topk_sigmoid` output indices must equal `torch.topk(
   torch.sigmoid(logits).float() + bias, k=8).indices.sort().values`
   on CPU.

2. **Unit — expert weights match within 1e-4**: Same corpus, weights
   from the kernel must match `sigmoid(logits)[indices] /
   sigmoid(logits)[indices].sum()` within 1e-4 absolute.

3. **End-to-end — tiny-random layer 0 output matches HF**: Run the
   tiny-random MiniMax through layer 0 of MiniMaxM2 on HF (greedy,
   temperature 0, seed fixed), compare the FFN output tensor element-
   wise to Atlas. Difference must be bounded by `rms_norm_eps *
   hidden_size * num_experts_per_tok * bf16_epsilon` (i.e. within
   accumulation slop, not higher).

## Estimated effort

- Weight loader helper: 0.5 day.
- MoE layer sigmoid branch wiring: 0.5 day (mostly finding the
  existing flag that Nemotron flipped).
- Unit + integration tests: 0.5 day.
- **Total: 1–1.5 days.**

## Dependencies

- **M1**: landed. Config parses + dispatch hits MinimaxM2WeightLoader.
- **M2 (partial RoPE + qk_norm)**: needed before end-to-end test works.
  M3 can land behind M2 without breaking M2's tests — the MoE layer
  doesn't depend on the attention output shape, only on `hidden_size`.
- **M4**: FP8 weights. Not needed for M3 correctness (tiny-random is
  BF16). FP8 path validated separately.

## Files touched (estimate)

- `crates/spark-model/src/layers/moe.rs` (+20 lines, scoring_func
  branch in ctor)
- `crates/spark-model/src/weight_map.rs` (+40 lines, load_moe_minimax
  helper)
- `crates/spark-model/src/weight_loader/minimax.rs` (+150 lines,
  replaces the stub load_layers with real loop)
- Tests: `tests/integration/minimax_moe_routing.rs` (new, +100 lines)

All additive. No existing model path modified.
