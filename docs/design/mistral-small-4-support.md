# Mistral Small 4 119B NVFP4 Support

**Status**: Planning complete, implementation starting
**Date**: 2026-03-27
**HF Model**: `mistralai/Mistral-Small-4-119B-2603-NVFP4`

## Architecture

- MLA (Multi-head Latent Attention) with kv_lora_rank=256, q_lora_rank=1024
- NoPE/RoPE split: qk_nope_head_dim=64, qk_rope_head_dim=64
- 128 experts top-4, 1 shared expert, expert_hidden_dim=2048
- 36 layers, dim=4096, 32 heads, 32 kv_heads, v_head_dim=128
- Vision encoder (multimodal)
- 119B params (6.5B active per token)
- NVFP4 on MoE experts only, attention is FP16

## Key Differences from Qwen3.5

| | Mistral Small 4 | Qwen3.5 |
|--|--|--|
| Config | `params.json` (custom) | `config.json` (HF) |
| Weights | `consolidated-*.safetensors` | `model.safetensors-*` |
| Attention | MLA (compressed KV latent) | Standard GQA |
| KV cache | 320 dims/token (12.8x smaller) | 4096 dims/token |
| Weight names | wq_a/wq_b/wkv_a/wkv_b/wo | q_proj/k_proj/v_proj/o_proj |
| MoE names | w1/w2/w3 | gate_proj/up_proj/down_proj |
| Chat format | [INST]...[/INST] + [TOOL_CALLS] | ChatML |
| Quant scope | MoE only (attention FP16) | All layers |

## Implementation Phases

### Phase A: Infrastructure (~300 lines, 1-2 days)
- [ ] A.1: `params.json` config normalizer in `config.rs`
- [ ] A.2: `consolidated-*.safetensors` support in `weights.rs`
- [ ] A.3: Mistral Jinja chat template
- [ ] A.4: `AttentionType::Mla` in capabilities.rs

### Phase B: MLA Attention (~3500 lines, 1-2 weeks)
- [ ] B.1: `MlaAttentionLayer` struct in new `mla_attention.rs`
- [ ] B.2: MLA KV cache format (compressed latent storage)
- [ ] B.3: MLA decode kernel (`mla_decode_attn.cu`)
- [ ] B.4: MLA prefill kernel (`mla_prefill_attn.cu`)
- [ ] B.5: MLA reshape_and_cache kernel

### Phase C: Weight Loader (~500 lines, 1 day)
- [ ] C.1: `MistralWeightLoader` implementing `ModelWeightLoader`
- [ ] C.2: Factory registration in `loader_for_config()`

### Phase D: Integration (~50 lines, 1 hour)
- [ ] D.1: Kernel target `kernels/gb10/mistral-small-4/`

## Shortcut: GQA Fallback

For initial bring-up without MLA kernels:
- Decompress KV at load time (lose 12.8x cache reduction)
- Use existing paged attention kernels
- Gets the model serving (slowly) while MLA kernels are developed

## Full plan: `.claude/plans/delegated-hatching-ladybug.md`
