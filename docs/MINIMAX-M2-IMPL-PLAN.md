# MiniMax M2 — M2 Implementation Plan (BF16 first pass)

**Status**: Ready to code. M1 is green; tiny-random model loads through
the kernel-target gate and the loader stub. M2 fills in real layer
construction for the BF16 path.

## Acceptance (from gameplan)

- Single-token forward pass matches HF `MiniMaxM2ForCausalLM` layer-0
  output for 5 random input ids.
- 512-token greedy generation matches HF byte-for-byte on first 16
  tokens for 5 seeds.

Validation uses `yujiepan/minimax-m2.7-tiny-random` (arch-exact,
BF16, already cached locally). HF reference runs via
`transformers.MiniMaxM2ForCausalLM.from_pretrained` on CPU.

## Weight name mapping (verified against tiny-random checkpoint)

| Atlas abstraction | HF weight name (MiniMax) |
|---|---|
| embed | `model.embed_tokens.weight` |
| final_norm | `model.norm.weight` |
| lm_head | `lm_head.weight` |
| per-layer input_norm | `model.layers.{i}.input_layernorm.weight` |
| per-layer post_attn_norm | `model.layers.{i}.post_attention_layernorm.weight` |
| Q projection | `model.layers.{i}.self_attn.q_proj.weight` |
| K projection | `model.layers.{i}.self_attn.k_proj.weight` |
| V projection | `model.layers.{i}.self_attn.v_proj.weight` |
| O projection | `model.layers.{i}.self_attn.o_proj.weight` |
| q_norm | `model.layers.{i}.self_attn.q_norm.weight` ← shape `[head_dim * num_attention_heads]` |
| k_norm | `model.layers.{i}.self_attn.k_norm.weight` ← shape `[head_dim * num_key_value_heads]` |
| MoE gate | `model.layers.{i}.block_sparse_moe.gate.weight` |
| MoE routing bias | `model.layers.{i}.block_sparse_moe.e_score_correction_bias` |
| Expert gate proj | `model.layers.{i}.block_sparse_moe.experts.{j}.w1.weight` |
| Expert up proj | `model.layers.{i}.block_sparse_moe.experts.{j}.w3.weight` |
| Expert down proj | `model.layers.{i}.block_sparse_moe.experts.{j}.w2.weight` |

Note the Mixtral-convention expert naming (`w1/w2/w3` not
`gate_proj/up_proj/down_proj`). Same as Nemotron-H's MoE layer naming.

## qk_norm shape divergence from existing Atlas models

**This is the subtle bit.** Atlas's existing Qwen3-family attention
holds `q_norm: DenseWeight` sized `[head_dim]` — per-head RMSNorm with
shared learned weight. MiniMax holds weight sized
`[head_dim * num_heads]` — one RMSNorm over the concatenated Q output
before the view-into-heads. These are **mathematically different**:

- Atlas Qwen3.5: each head normalized by its own RMS, weight shared
  across heads.
- MiniMax M2: all heads normalized by the global RMS, weight per
  element of the full projected Q.

Two implementation choices:

### Option A — new `AttentionWeights::q_norm_full` variant

Add a sibling field in `AttentionWeights`:
```rust
pub q_norm_full: Option<DenseWeight>,  // shape [hidden_q] if Some
```
(Keep `q_norm: DenseWeight` for existing Qwen3-family path.) In the
attention forward, branch:
```rust
if let Some(full_norm) = &self.q_norm_full {
    rms_norm_full(q_proj_out, full_norm.weight, hidden_q);
} else {
    // existing per-head path
}
```
Loader sets `q_norm_full = Some(..)` for MiniMax, `q_norm = ..` for
Qwen.

### Option B — pre-broadcast MiniMax weights to per-head shape

Split the `[hidden_q]` weight into `num_heads` slices of `[head_dim]`
at load time. But the normalization math differs (global RMS vs
per-head RMS), so a simple reshape won't match HF output. Option B
produces incorrect results — reject.

**Chosen: Option A.** Additive, 15-line change in the attention
forward, matches HF behavior exactly.

## Partial RoPE

Already handled. `config.rotary_dim()` returns the explicit integer
field when set (M1 change), and the common rope kernel at
`kernels/gb10/nvfp4/rope.cu` already takes `rotary_dim` as a runtime
argument distinct from `head_dim`. No kernel change needed; attention
forward just needs to pass `config.rotary_dim()` down.

## MoE — scoped to M2 or to M3?

M2 can land with a **placeholder softmax routing** that uses Atlas's
existing `moe_topk_softmax`. This will produce incorrect output but
will let the rest of the forward path exercise end-to-end. M3 replaces
with the sigmoid+bias kernel and acceptance criteria for the
whole-forward match flip green.

Alternatively, land M2 and M3 together since both are needed for
layer-0 output match. Recommend: **one PR, both milestones**, since
the tiny-random validation can't pass without both.

## Code change list (single PR, M2 + M3)

### `crates/spark-model/src/weight_loader/minimax.rs` (~200 lines)

Replace stub `load_layers` with a real impl:

```rust
fn load_layers(
    &self,
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layer_kv_dtypes: &[KvCacheDtype],
) -> Result<Vec<Box<dyn TransformerLayer>>> {
    if !enabled() { bail!(...); }  // (existing gate, keep)

    let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    let stream = gpu.default_stream();
    let h = config.hidden_size;
    let mut layers: Vec<Box<dyn TransformerLayer>> = Vec::with_capacity(config.num_hidden_layers);

    for i in 0..config.num_hidden_layers {
        let lp = format!("model.layers.{i}");
        let input_norm = dense(store, &format!("{lp}.input_layernorm.weight"))?;
        let post_attn_norm = dense(store, &format!("{lp}.post_attention_layernorm.weight"))?;

        // ── Attention ──
        let p = format!("{lp}.self_attn");
        let q_bf16 = dense(store, &format!("{p}.q_proj.weight"))?;
        let k_bf16 = dense(store, &format!("{p}.k_proj.weight"))?;
        let v_bf16 = dense(store, &format!("{p}.v_proj.weight"))?;
        let o_proj_nvfp4 = quantize_to_nvfp4(
            &dense(store, &format!("{p}.o_proj.weight"))?,
            h, h, gpu, absmax_k, quantize_k, stream,
        )?;
        let q_norm = dense(store, &format!("{p}.q_norm.weight"))?;
        let k_norm = dense(store, &format!("{p}.k_norm.weight"))?;

        let attn = AttentionWeights {
            q_proj: q_bf16, k_proj: k_bf16, v_proj: v_bf16,
            o_proj: o_proj_nvfp4,
            q_norm,        // full shape [hidden_q] — loader sets q_norm_full below
            k_norm,
            k_scale: 1.0, v_scale: 1.0,
        };

        let q_nvfp4 = quantize_to_nvfp4(&q_bf16, /* dims */, gpu, absmax_k, quantize_k, stream)?;
        let k_nvfp4 = quantize_to_nvfp4(&k_bf16, /* dims */, gpu, absmax_k, quantize_k, stream)?;
        let v_nvfp4 = quantize_to_nvfp4(&v_bf16, /* dims */, gpu, absmax_k, quantize_k, stream)?;

        // ── MoE (sigmoid + correction bias, NO shared expert) ──
        let moe_weights = load_moe_minimax(store, &lp, config.num_experts, gpu, config,
                                           absmax_k, quantize_k, stream)?;
        let gate_nvfp4 = quantize_to_nvfp4(
            &moe_weights.gate, config.num_experts, h,
            gpu, absmax_k, quantize_k, stream,
        )?;
        let moe_layer = MoeLayer::new_sigmoid(                       // ← new ctor variant
            moe_weights, config.num_experts, Some(gate_nvfp4), gpu, config,
        )?;
        let ffn = FfnComponent::Moe(moe_layer);

        let layer = Qwen3AttentionLayer::new_ungated(                // ungated — MiniMax doesn't gate Q
            input_norm, attn, post_attn_norm, ffn, i,
            Some(q_nvfp4), Some(k_nvfp4), Some(v_nvfp4),
            gpu, layer_kv_dtypes[i],
            config.fp8_kv_calibration_tokens, config,
        )?;

        layers.push(Box::new(layer));
    }
    Ok(layers)
}
```

### `crates/spark-model/src/weight_map.rs` (~60 lines)

New helper `load_moe_minimax` — mirror `load_moe_no_shared` but:
- Weight prefix: `{lp}.block_sparse_moe.experts.{j}.{w1,w2,w3}`
- Gate prefix: `{lp}.block_sparse_moe.gate`
- Bias: `{lp}.block_sparse_moe.e_score_correction_bias` as f32 tensor

```rust
pub fn load_moe_minimax(
    store: &WeightStore,
    lp: &str,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    config: &ModelConfig,
    absmax_k: KernelHandle,
    quantize_k: KernelHandle,
    stream: u64,
) -> Result<MoeWeights> {
    let sp = format!("{lp}.block_sparse_moe");
    let gate = dense(store, &format!("{sp}.gate.weight"))?;
    let correction_bias = dense_keep_f32(store, &format!("{sp}.e_score_correction_bias"))?;
    let experts = (0..num_experts).map(|j| {
        let ep = format!("{sp}.experts.{j}");
        Ok(DenseExpert {
            gate_proj: dense(store, &format!("{ep}.w1.weight"))?,   // gate
            up_proj:   dense(store, &format!("{ep}.w3.weight"))?,   // up
            down_proj: dense(store, &format!("{ep}.w2.weight"))?,   // down
        })
    }).collect::<Result<Vec<_>>>()?;
    // NO shared expert
    Ok(MoeWeights { gate, correction_bias: Some(correction_bias), experts, shared: None, ... })
}
```

### `crates/spark-model/src/layers/moe.rs` (~20 lines)

Add `MoeLayer::new_sigmoid` that stores the bias tensor and dispatches
through `ops::moe_topk_sigmoid` in its top-k section. Mirror Nemotron-H's
existing sigmoid path exactly (read the nemotron.rs loader — it already
constructs a sigmoid-routed MoE).

### `crates/spark-model/src/weight_map.rs` — `AttentionWeights` field add (~5 lines)

Extend struct:
```rust
pub struct AttentionWeights {
    pub q_proj: DenseWeight,
    // ... existing fields ...
    pub q_norm: DenseWeight,        // existing — [head_dim]
    pub k_norm: DenseWeight,        // existing — [head_dim]
    pub q_norm_full: Option<DenseWeight>,  // NEW — [hidden_q] for MiniMax-style
    pub k_norm_full: Option<DenseWeight>,  // NEW — [hidden_k]
    pub k_scale: f32,
    pub v_scale: f32,
}
```

Defaults to `None` for existing loaders (no change to Qwen35 / VL / Nemotron
code since struct-update syntax via `..Default::default()` or explicit
`None` at construction).

### `crates/spark-model/src/layers/qwen3_attention/{prefill,decode}.rs` (~30 lines)

In each attention forward path, after Q and K projections and before
RoPE:

```rust
if let Some(q_norm_full) = &self.attn_weights.q_norm_full {
    // Standard RMSNorm over hidden_q last dim.
    rms_norm_bf16(
        ctx.gpu, self.rms_norm_k,
        q_dev_ptr,                              // in-place
        q_norm_full.weight,
        batch_tokens, hidden_q,
        config.rms_norm_eps, stream,
    )?;
}
// (symmetric block for k_norm_full)
```

Existing `rms_norm_bf16` op in atlas/layers/ops.rs already handles this
shape. Verify kernel arg is `[..., D]` with weight `[D]` — it is
(that's what rms_norm.cu does).

### M1 loader stub removal

Replace the `not_yet` error calls in `minimax.rs` `load_{embedding,
final_norm, lm_head, mtp_weights}` with real loads:

```rust
fn load_embedding(...) -> Result<DenseWeight> {
    dense(store, "model.embed_tokens.weight")
}
fn load_final_norm(...) -> Result<DenseWeight> {
    dense(store, "model.norm.weight")
}
fn load_lm_head(...) -> Result<DenseWeight> {
    dense(store, "lm_head.weight")
}
fn load_mtp_weights(...) -> Result<Option<MtpWeights>> {
    Ok(None)  // M5 replaces this
}
```

## Test plan

Local CPU pytest script (new file,
`tests/integration/minimax_tiny_vs_hf.py`):

```python
import torch, json, subprocess, time
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "yujiepan/minimax-m2.7-tiny-random"

# 1. Golden: HF output
hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).eval()
tok = AutoTokenizer.from_pretrained(MODEL)
inp = tok("The quick brown", return_tensors="pt")
with torch.no_grad():
    hf_out = hf(**inp).logits[0, -1].float()
hf_top5 = torch.topk(hf_out, 5)

# 2. Atlas output: via OpenAI API with logprobs=5
atlas_url = "http://localhost:8889/v1/chat/completions"
req = {"model": MODEL, "messages": [{"role":"user", "content":"The quick brown"}],
       "max_tokens": 1, "logprobs": True, "top_logprobs": 5, "temperature": 0}
# ... POST and parse logprobs ...

# 3. Assert top-5 token ids match, logits within 1e-2 abs
assert set(atlas_top5_ids) == set(hf_top5.indices.tolist())
for tok_id in hf_top5.indices:
    assert abs(atlas_logit[tok_id] - hf_logit[tok_id]) < 1e-2
```

Runtime: a few minutes on any CPU + small VRAM. No 229B weights, no
kernel rework.

## Effort estimate (revised with tiny-random in hand)

- `load_moe_minimax` helper: 30 min
- `MoeLayer::new_sigmoid` ctor (mirror Nemotron's existing path): 45 min
- `AttentionWeights::q_norm_full` field add + all other loaders pass
  `None`: 15 min
- `qwen3_attention` forward: add RMSNorm call gated on `q_norm_full`:
  45 min (includes finding the right place in prefill + decode paths)
- `MinimaxM2WeightLoader::load_layers`: 90 min (structural code, plus
  debugging the tiny model boot until first layer runs)
- Test harness: 45 min
- **Total: ~4.5 hours of focused coding, assuming no surprises.**

## Dependencies
- M1: landed, e2e validated.
- M3: merges with M2 (can't test without sigmoid routing).
- M4: separate — BF16 path works first; FP8 lands after.
- M5: separate — non-MTP decode first; K=3 verify lands after.

## Files touched
- `crates/spark-model/src/weight_loader/minimax.rs` (~200 lines net)
- `crates/spark-model/src/weight_map.rs` (~80 lines: `load_moe_minimax` +
  AttentionWeights field)
- `crates/spark-model/src/layers/moe.rs` (~40 lines: `new_sigmoid` ctor)
- `crates/spark-model/src/layers/qwen3_attention/{prefill,decode}.rs`
  (~60 lines total: q_norm_full RMSNorm call)
- `tests/integration/minimax_tiny_vs_hf.py` (new, ~80 lines)

All additive. Existing models untouched — `q_norm_full = None` default
means Qwen/Gemma/Mistral/Nemotron paths are byte-identical.
