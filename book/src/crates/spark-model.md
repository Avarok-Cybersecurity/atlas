# spark-model

**Role:** the model assembly crate. Translates loaded weights and config into `Box<dyn TransformerLayer>` objects, drives the inference engine loop, implements speculative decoding and vision preprocessing.
**Key files:** `engine.rs`, `model.rs`, `factory.rs`, `layer.rs` + `layers/*.rs`, `weight_loader/*.rs`, `weight_map.rs`, `speculative.rs`, `vision_preprocess.rs`, `traits.rs`, `quant_format.rs`, `mistral_loader.rs`, `preflight.rs`.

This is the largest crate in the workspace — ~18k lines — because every model architecture Atlas supports has its own loader here. The design centers on two small, heavily-used traits.

## The central traits

### `Model`

```rust
pub trait Model {
    fn alloc_sequence(&self) -> Result<Sequence>;
    fn free_sequence(&self, seq: &mut Sequence) -> Result<()>;
    fn prefill(&self, seq: &mut Sequence, tokens: &[u32], stream: u64) -> Result<()>;
    fn decode(&self, seq: &mut Sequence, token: u32, stream: u64) -> Result<u32>;
    // + vision, tool, MTP extension methods
}
```

One-model-per-server. `TransformerModel` (in `model.rs`) is the concrete type — owns the loaded `Vec<Box<dyn TransformerLayer>>`, the shared embedding + LM head, the KV cache, the prefix cache, the buffer arena. Threaded via `Arc` into the scheduler.

### `TransformerLayer`

```rust
pub trait TransformerLayer: Send + Sync {
    fn forward(&self, ctx: &mut LayerContext) -> Result<()>;
    fn kind(&self) -> LayerKind; // Attention / SsmAttention / Moe / DenseFfn / Vision
    // + MTP-aware variants, KV cache allocation hooks
}
```

Every layer type is a trait object. The decode-step loop in `engine.rs` iterates `self.layers.iter().map(|l| l.forward(ctx))`. No `match layer.kind()` branches in the hot loop — the virtual call is the entire dispatch.

## The layer menagerie (`layers/*.rs`)

| Layer | Files | Used by |
|---|---|---|
| Qwen3 full attention | `qwen3_attention.rs` | Qwen3 / Qwen3-Next / Qwen3.5 / Qwen3.6 / Qwen3-VL |
| Qwen3 SSM | `qwen3_ssm.rs` | Qwen3 hybrid (SSM branch) |
| Qwen3.5 GDN (gated delta rule) | `qwen3_ssm.rs` + specialised variant | Qwen3.5-35B, Qwen3.5-122B, Qwen3.6 |
| Nemotron Mamba-2 | `nemotron_mamba2.rs` | Nemotron-3 Nano / Super |
| MoE (sparse experts) | `moe.rs`, `moe_prefill.rs`, `moe_shared.rs` | Every MoE model |
| Dense FFN | `dense_ffn.rs` | Dense models (Qwen3.5-27B, Gemma-4-31B) |
| Gemma-4 sliding+full alternating attention | `gemma4_attention.rs` | Gemma-4-31B |
| Mistral attention + MoE | `mistral_attention.rs`, `mistral_moe.rs` | Mistral-Small-4 |
| MiniMax attention + 256-expert sigmoid MoE | `minimax_attention.rs`, `minimax_moe.rs` | MiniMax-M2.7 |
| Vision ViT block + merger | `vision_encoder.rs` | Qwen3-VL, Qwen3.6 |

New models reuse these where possible. Writing a new layer type is rare — the MiniMax 256-expert sigmoid-routed MoE is the most recent example, and it was a new file because the routing semantics genuinely differ from softmax-topk. Gemma-4's sliding+full alternation got a new file because the attention window masks alternate per layer.

## Weight loaders (`weight_loader/*.rs`)

One file per model family. Each implements:

```rust
pub trait ModelWeightLoader {
    fn load_layers(&self, store: &WeightStore, config: &ModelConfig,
                   gpu: &dyn GpuBackend, layer_kv_dtypes: &[KvCacheDtype])
        -> Result<Vec<Box<dyn TransformerLayer>>>;
    fn load_embedding(&self, store: &WeightStore, config: &ModelConfig) -> Result<DenseWeight>;
    fn load_final_norm(&self, store: &WeightStore, config: &ModelConfig,
                       gpu: &dyn GpuBackend) -> Result<DenseWeight>;
    fn load_lm_head(&self, store: &WeightStore, config: &ModelConfig) -> Result<DenseWeight>;
    fn load_mtp_weights(&self, store: &WeightStore, config: &ModelConfig,
                        gpu: &dyn GpuBackend) -> Result<Option<MtpWeights>>;
}
```

Current files:

- `qwen3.rs` — Qwen3-Next (NVFP4, hybrid SSM+Attention+MoE with MTP).
- `qwen35.rs` — Qwen3.5 MoE (35B, 122B) with GDN + MTP.
- `qwen35_dense.rs` — Qwen3.5 Dense (27B), hybrid without MoE.
- `qwen3_vl.rs` — Qwen3-VL (30B, vision + attention + MoE).
- `gemma4.rs` — Gemma-4 (26B MoE + 31B dense; GeGLU; sliding/full alternation).
- `nemotron.rs` — Nemotron-H Nano + Super (Mamba-2 + MoE + attention).
- `minimax.rs` — MiniMax-M2 / M2.7 (256-expert sigmoid MoE).
- `mistral_loader.rs` — the one outlier (lives one level up in `spark-model/src/` because the Mistral-Small-4 loader predates the `weight_loader/` submodule reorganisation).

Each loader knows the HF weight-name patterns for its family and translates them into `Box<dyn TransformerLayer>` via the helpers in `weight_map.rs`:

- `load_attention(store, layer_idx, prefix)` — reads `q_proj`, `k_proj`, `v_proj`, `o_proj` + optional RoPE scales.
- `load_moe(store, layer_idx, num_experts, ...)` — reads the expert weights, gate, optional shared experts.
- `load_ssm(store, layer_idx, config)` — reads the Mamba/GDN A, B, C, D, dt projections.
- `load_dense_ffn(...)` — gate + up + down.
- `dequant_nvfp4_to_bf16`, `dequant_fp8_to_bf16` — on-the-fly quant conversion for layers that run BF16 even if the checkpoint is quantized (e.g. first/last attention layers under `--kv-high-precision-layers`).

## The factory (`factory.rs`)

The single point where a model type becomes a loader:

```rust
fn loader_for_config(config: &ModelConfig) -> Result<Box<dyn ModelWeightLoader>> {
    let normalized = config.model_type
        .to_lowercase()
        .replace('-', "_")
        .replace('.', "_");
    match normalized.as_str() {
        "qwen3_next_for_causal_lm"     => Ok(Box::new(Qwen3WeightLoader)),
        "qwen3_5_next_for_causal_lm"   => Ok(Box::new(Qwen35WeightLoader)),
        "qwen3_5_for_causal_lm"        => Ok(Box::new(Qwen35DenseWeightLoader)),
        "qwen3_vl_for_causal_lm"       => Ok(Box::new(Qwen3VLWeightLoader)),
        "gemma_4_for_causal_lm"        => Ok(Box::new(Gemma4WeightLoader)),
        "nemotron_h_for_causal_lm"     => Ok(Box::new(NemotronHWeightLoader)),
        "minimax_m2_for_causal_lm"     => Ok(Box::new(MinimaxM2WeightLoader)),
        "mistral_small_4_for_causal_lm" => Ok(Box::new(MistralWeightLoader)),
        other => bail!("Unsupported model type: '{}'", other),
    }
}
```

This is the **single code site where `model_type` strings are matched**. Everything downstream of `factory::build` holds `Box<dyn Model>` and is model-agnostic. See the top-level repo [Adding a new model](https://github.com/Avarok-Cybersecurity/atlas/blob/main/README.md#adding-a-new-model) guide.

## The engine (`engine.rs`)

```rust
pub fn generate(
    model: &dyn Model,
    prompt_tokens: &[u32],
    params: &SamplingParams,
) -> Result<GenerateResult>;
```

Prefill → decode loop, sampler integration, finish-reason detection (`"stop"` / `"length"`), EOS + stop-token handling. The scheduler in `spark-server` drives this — `engine.rs` itself is stateless per call; the per-sequence state lives on `Sequence` (allocated/freed around the generate).

## Speculative decoding (`speculative.rs`)

Wraps the MTP draft-then-verify loop. Draft tokens are produced by the MTP head, verified by the main model in one forward pass, and accepted-to-longest-match. Sibling stride bugs (`qwen3_attention` + `qwen3_ssm` with K≠2) were fixed in the Pass-16/Pass-22 bug sweeps; the current code handles K=1, 2, and 3 for the families that support it. See the [MTP chapter](../deep-dives/mtp.md).

## Vision preprocessing (`vision_preprocess.rs`)

For Qwen3-VL and Qwen3.6: accept image input (JPEG/PNG/base64), resize/normalise to the model's patch grid, produce pixel-values tensor + MRoPE position IDs (H/W/T triples). Handles the 3× positions scratch mentioned in [atlas-embed](./atlas-primitives.md#atlas-embed--position--token-embeddings).

## Quant format runtime dispatch (`quant_format.rs`)

Sniffs the checkpoint shape on load and picks the right `Dequantize` implementation. Introduced in the Pass-25 sweep to replace a load-time heuristic that had produced EP=2 CUDA illegal-address errors on the ModelOpt-NVFP4 variant of M2.7.

## Preflight (`preflight.rs`)

Runs a small synthetic decode step before the HTTP server binds. Catches:

- Weight-loading shape mismatches.
- KV-cache budget overruns (pre-OOM).
- Missing kernel modules for the selected `KernelTarget`.

This is what "OOM pre-flight" in the feature matrix is. Failing preflight produces a clear, early error; passing it means the hot path is safe.

## What's explicitly not here

- **No HTTP.** That's `spark-server`.
- **No GPU ops.** Every GPU touch is via `spark-runtime::GpuBackend`.
- **No collective ops.** Every multi-GPU touch is via `spark-comm::CommBackend`.

Adding a new model is almost always: one new `weight_loader/<family>.rs`, one match arm in `factory.rs`, optional reuse of existing `layers/*.rs`, optional new `layers/<family>_attention.rs` if the attention shape genuinely differs.
