# Qwen3.5-35B-A3B-NVFP4 HyperCompilation Journey

**Model**: Sehyo/Qwen3.5-35B-A3B-NVFP4
**Hardware**: NVIDIA DGX Spark GB10 (SM121, 120 GB LPDDR5X @ 273 GB/s)
**Started**: 2026-03-02

---

## Model Architecture Summary

| Parameter | Qwen3.5-35B | Qwen3-Next-80B | Delta |
|-----------|-------------|----------------|-------|
| Model type | `qwen3_5_moe` | `qwen3_moe` | Different HF class |
| Hidden size | 2048 | 2048 | Same |
| Total layers | 40 (30 linear + 10 full attn) | 48 (36 SSM + 12 attn) | -8 layers |
| Attention heads | 16Q / 2KV / 256d | 16Q / 2KV / 256d | Same |
| DeltaNet heads | 16K / 32V / 128d | 16K / 32V / 128d | Same |
| Experts | 256 total, top-8 | 512 total, top-10 | Half as many |
| Expert intermediate | 512 | 512 | Same |
| Vocab size | 248,320 | 151,936 | +63% larger |
| Linear proj quant | **BF16 (not quantized!)** | NVFP4 | 2x bandwidth |
| MTP attention type | Full attention | SSM | Different path |
| Vision | Qwen2VL (27-layer ViT) | None | New modality |
| NVFP4 size on disk | ~25 GB | ~47 GB | -47% smaller |

### Critical Insight: BF16 Linear Attention Weights

The Sehyo NVFP4 quantization **skipped** the 30 linear attention layers. These remain in BF16:

| Component | Per-token reads | % of total |
|-----------|----------------|------------|
| Linear attn projections (BF16) | 2,023 MB | 52% |
| LM head (BF16, 248K vocab) | 1,017 MB | 26% |
| MoE (NVFP4, 8 active experts) | 679 MB | 18% |
| Full attn (NVFP4) | 153 MB | 4% |
| **Total** | **3,871 MB** | 100% |

**Theoretical decode ceiling**: 3,871 MB / 273 GB/s = 14.2 ms = **70.5 tok/s** (at 100% BW)
**Practical estimate** (65% BW): 21.8 ms = **45.9 tok/s**

**Optimization opportunity**: Self-quantize linear_attn weights to NVFP4 at load time → saves ~1,447 MB/token → potential **73 tok/s** at 65% BW.

---

## Phase 0: Model Download

**Status**: COMPLETE

- Downloaded 24 GB to `/workspace/.cache/huggingface/hub/models--Sehyo--Qwen3.5-35B-A3B-NVFP4/`
- 14 files: 1 model shard (23.4 GB) + 1 MTP weights (1.69 GB) + configs
- Weight format issue discovered: linear_attn weights stored as separate `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b` (vLLM expects fused `in_proj_qkvz`, `in_proj_ba`)

---

## Phase 1: vLLM Baseline

**Status**: COMPLETE

### Root Cause: packed_modules_mapping Bug

The v25 image had correct weight stacking code in `Qwen3_5Model.load_weights()` but the
linear_attn weights were being NVFP4-quantized when they should remain BF16.

**Root cause chain**:
1. Checkpoint has separate BF16 weights: `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`
2. Model creates fused parameters: `in_proj_qkvz`, `in_proj_ba` (via `MergedColumnParallelLinear`)
3. Quantization `ignore` list has original names (`in_proj_qkv`, etc.) but NOT fused names
4. `compressed-tensors` uses `packed_modules_mapping` to map fused->unfused for ignore checking
5. `Qwen3_5MoeForConditionalGeneration` inherits `packed_modules_mapping` from
   `Qwen3VLForConditionalGeneration` which only has `qkv_proj`, `gate_up_proj`, `qkv`
6. Missing: `in_proj_qkvz -> [in_proj_qkv, in_proj_z]` and `in_proj_ba -> [in_proj_b, in_proj_a]`
7. Without mapping, quantizer cannot match fused names against ignore list
8. Linear attn projections get NVFP4-quantized with `weight_packed`/`weight_scale` parameters
9. Weight loader looks for `.weight` but only finds `.weight_packed` -> skip loading -> zeros

**Fix**: `/workspace/fix_qwen35_packed_modules.py` adds GDN entries to `packed_modules_mapping`
on both `Qwen3_5ForCausalLMBase` and `Qwen3_5ForConditionalGeneration`.

### Configuration

```bash
sudo docker run -d --name vllm-qwen35 --network host --gpus all --ipc=host \
  -v /workspace/.cache/huggingface:/root/.cache/huggingface \
  -v /workspace/fix_qwen35_packed_modules.py:/tmp/fix_qwen35_packed_modules.py:ro \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e MODEL=Sehyo/Qwen3.5-35B-A3B-NVFP4 -e PORT=8888 -e GPU_MEMORY_UTIL=0.90 \
  -e MAX_MODEL_LEN=4096 -e MAX_NUM_SEQS=128 \
  -e VLLM_EXTRA_ARGS="--attention-backend flashinfer --kv-cache-dtype fp8 --enforce-eager" \
  --entrypoint bash \
  dgx-vllm:v25 -c "
    python3 /tmp/fix_qwen35_packed_modules.py && \
    export PATH=/opt/venv/bin:\$PATH && \
    vllm serve Sehyo/Qwen3.5-35B-A3B-NVFP4 \
      --host 0.0.0.0 --port 8888 \
      --max-model-len 4096 --gpu-memory-utilization 0.90 \
      --max-num-seqs 128 \
      --attention-backend flashinfer --kv-cache-dtype fp8 --enforce-eager
  "
```

### Results

| Metric | Value |
|--------|-------|
| Decode throughput | **35-37 tok/s** (eager mode, no CUDA graphs) |
| TTFT (short prompt) | **0.08-0.12s** |
| Memory usage | **21.86 GiB** (model weights + KV cache) |
| KV cache capacity | **2,219,664 tokens** (847x concurrent at 4K) |
| Startup time | **~3 min** (120s weight loading + 50s KV init) |
| Stability | **4/4 requests** (150 tok each, coherent output) |
| Output quality | Coherent reasoning/thinking output |
| MoE backend | Marlin (NVFP4 W4A16 dequant) |
| Attention backend | FlashInfer |
| CUDA graphs | Disabled (enforce-eager) |
| torch.compile | Disabled (enforce-eager) |

### Benchmark Detail (4 runs, 150 tokens each)

| Run | Decode tok/s | TTFT |
|-----|-------------|------|
| 1 | 35.1 | 0.12s |
| 2 | 35.6 | 0.10s |
| 3 | 36.0 | 0.08s |
| 4 | 37.3 | 0.10s |

### Optimization Opportunities

1. **Enable CUDA graphs + torch.compile**: Should give +30-50% (like Qwen3-Next: 36 -> 60 tok/s)
2. **Self-quantize BF16 linear_attn to NVFP4**: Saves 1,447 MB/token bandwidth (52% of reads)
3. **MTP speculative decoding**: Additional +10-15% if MTP head works
4. **Theoretical ceiling**: 70.5 tok/s (100% BW), ~46 tok/s practical (65% BW)

### Key Observations

- The model is a "thinking" model that shows reasoning process in output
- BF16 linear attention weights are the primary bandwidth bottleneck (52% of per-token reads)
- 256 experts with top-8 routing (vs Qwen3-Next's 512/top-10) means smaller MoE overhead
- No `NVFP4 Marlin negative scales` warning (unlike unpatched version, confirming fix correctness)

---

## Phase 2: Atlas HyperCompilation

**Status**: IN PROGRESS

Target: `kernels/gb10/qwen3.5-35b-a3b/nvfp4/`

### Kernel Reuse Analysis

**Reusable as-is (21 kernel files)**: gated_delta_rule, causal_conv1d, deinterleave_qg, all w4a16_*, dense_gemv_*, dense_gemm_bf16, paged_decode_attn*, reshape_and_cache, kv_cache_append, rms_norm, e2m1_branchless, quantize_bf16_to_nvfp4, argmax_bf16, moe_permute, moe_silu_mul, residual_add, transpose_u8, vector_add, inferspark_*

**Runtime param changes only (3 files)**: moe_topk (256 experts/top-8), moe_expert_gemv* (top-8, 256 entries), moe_w4a16_grouped_gemm (top-8)

**Adaptation needed (2 kernel functions)**:
- `deinterleave_qkvz` → new `deinterleave_qkv` variant (no Z)
- `compute_gdn_gates` → adapt for separate A/B projections vs fused BA

**New functionality**:
- M-RoPE for multimodal (text-only uses standard RoPE)
- MTP with full attention (reuse existing attn kernels, new dispatch in Rust)
- Pre-FC normalization (2 extra RMSNorm calls, trivial)

### Architecture Adaptations Checklist
- [ ] Copy kernels from qwen3-next-80b-a3b target
- [ ] Create MODEL.toml and KERNEL.toml for new target
- [ ] Update Rust model factory for qwen3_5_moe config parsing
- [ ] Adapt DeltaNet projection dispatch (separate vs fused weights)
- [ ] Parameterize MoE for 256 experts / top-8 routing
- [ ] MTP head: full attention path instead of SSM
- [ ] Self-quantize BF16 linear_attn weights to NVFP4 at load time (perf optimization)
- [ ] Vision encoder (future — text-only first)

---

## Changelog

| Date | Event |
|------|-------|
| 2026-03-02 | Journey started. Model downloaded (24 GB). |
| 2026-03-02 | vLLM baseline: v22 garbage output (weight stacking bug), v25 timeout (SM121 kernel issue). Rebuild initiated. |
| 2026-03-02 | Architecture analysis complete. 21/33 kernels reusable as-is. BF16 linear_attn weights identified as primary bandwidth bottleneck (52% of reads). |
| 2026-03-02 | HyperCompilation Phase 2 started. Setting up kernel target directory. |
| 2026-03-02 | **ROOT CAUSE FOUND**: `packed_modules_mapping` missing GDN entries. Fix: `/workspace/fix_qwen35_packed_modules.py`. |
| 2026-03-02 | **Phase 1 COMPLETE**: 35-37 tok/s decode, coherent output, 21.86 GiB memory. No rebuild needed (v25 + 1 Python patch). |
