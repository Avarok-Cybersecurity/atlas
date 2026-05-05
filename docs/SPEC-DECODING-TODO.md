# MTP Speculative Decoding for Atlas Spark

## Context

Atlas Spark achieves **99.1 tok/s peak** (~97 sustained) on Qwen3-Next-80B-A3B with single-token decode. The model has a built-in MTP (Multi-Token Prediction) head — a single transformer decoder layer trained jointly with the target model. vLLM achieves 59.9 tok/s using MTP with this same model (1.65x over its 36.4 tok/s baseline). Applying similar MTP to Atlas's 97 tok/s baseline could reach **140-170 tok/s**.

The design must abstract speculative decoding to support EAGLE-3 in the future.

**Prerequisite**: Model must produce coherent output first (Step 0 below).

---

## Step 0: Output Coherence Verification (Prerequisite) — IN PROGRESS

**Problem**: The integration test verifies finite logits and valid token ranges but never decodes output to text. Optimizations could have introduced subtle numerical errors producing plausible but incoherent output.

**Solution**: Replace the blind 200-token generation with a factual Q&A coherence check.

**Files**:
- `crates/spark-server/tests/integration.rs` — Replace or augment the 200-token generation:
  1. Load tokenizer from model_dir
  2. Encode a factual prompt: "What is the capital of France?" (using chat template if needed)
  3. Run prefill on the encoded prompt tokens
  4. Generate up to 200 tokens (same as now, with throughput timing)
  5. Decode ALL generated token IDs to text
  6. Log the full output text with `tracing::info!`
  7. **Assert the output contains "Paris"** (case-insensitive) — factual correctness check
  8. Assert output contains at least 20 unique tokens (not degenerate repetition)
  9. Keep existing throughput timing and reporting

**Verification**: Run integration test, confirm output says "Paris", inspect for coherent English.

**Strategy**: Bisect through commits to find where coherence broke. Start from ~40 tok/s baseline with coherence test, fix each commit incrementally.

---

## Step 1: Speculative Decoding Abstraction (SDD)

**Problem**: Need a clean trait boundary that MTP implements now and EAGLE-3 can implement later.

**Design**:

```rust
// crates/spark-model/src/speculative.rs (NEW)

pub trait ProposerState: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

pub trait DraftProposer: Send + Sync {
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>>;

    fn propose(
        &self,
        last_token: u32,
        target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Vec<u32>>;

    fn after_verify(
        &self,
        num_accepted: usize,
        state: &mut dyn ProposerState,
        stream: u64,
    ) -> Result<()>;
}
```

**Files**:
- `crates/spark-model/src/speculative.rs` — New: traits above
- `crates/spark-model/src/lib.rs` — Add `pub mod speculative;`
- `crates/spark-model/src/traits.rs` — Add `proposer_state: Option<Box<dyn ProposerState>>` to `SequenceState`

---

## Step 2: MTP Weight Loading

**Problem**: 1553 BF16 MTP weights need to be loaded from safetensors. MoE experts should be quantized to NVFP4 at load time for bandwidth reduction.

**MTP weight inventory** (all BF16 in safetensors, `mtp.*` prefix):
| Weight | Shape | Purpose |
|--------|-------|---------|
| `mtp.fc.weight` | [2048, 4096] | Concat embed+hidden projection |
| `mtp.pre_fc_norm_embedding.weight` | [2048] | RMSNorm on embedding |
| `mtp.pre_fc_norm_hidden.weight` | [2048] | RMSNorm on hidden state |
| `mtp.norm.weight` | [2048] | Final output RMSNorm |
| `mtp.layers.0.self_attn.q_proj.weight` | [8192, 2048] | 2x wider for output gate |
| `mtp.layers.0.self_attn.{k,v}_proj.weight` | [512, 2048] | 2 KV heads |
| `mtp.layers.0.self_attn.o_proj.weight` | [2048, 4096] | Output projection |
| `mtp.layers.0.self_attn.{q,k}_norm.weight` | [256] | Per-head norms |
| `mtp.layers.0.mlp.gate.weight` | [512, 2048] | MoE router |
| `mtp.layers.0.mlp.experts.{0..511}.*` | various | 512 BF16 experts |
| `mtp.layers.0.mlp.shared_expert.*` | various | Shared expert |
| Shared: `model.embed_tokens.weight` | [151936, 2048] | Reuse target embedding |
| Shared: `lm_head.weight` | [151936, 2048] | Reuse target lm_head (NVFP4) |

**Files**:
- `crates/spark-model/src/weight_map.rs` — Add `MtpWeights` struct + `load_mtp()` function
- `crates/spark-model/src/weight_loader.rs` — Add `load_mtp()` to `Qwen3WeightLoader`
  - Quantize fc, o_proj, Q/K/V proj, MoE experts to NVFP4 at load time (reuse `quantize_to_nvfp4()`)
- `crates/spark-model/src/factory.rs` — Call `load_mtp()`, pass to model constructor

**Memory**: ~3.3 GB BF16 raw → ~1.8 GB after NVFP4 quantization of projection weights.

---

## Step 3: MTP Head Forward Pass

**Forward pass (single token)**:
```
embed(token) → pre_fc_norm_embedding → normed_embed [2048]
hidden_states → pre_fc_norm_hidden → normed_hidden [2048]
bf16_concat(normed_embed, normed_hidden) → combined [4096]
fc GEMV [2048, 4096] → hidden [2048]                        ← NVFP4 (quantized at load)
input_layernorm → normed
Q proj GEMV [8192, 2048] → split: Q[4096] + gate[4096]     ← NVFP4
K proj GEMV [512, 2048]                                      ← NVFP4
V proj GEMV [512, 2048]                                      ← NVFP4
Q_norm, K_norm, RoPE (partial=0.25), reshape_and_cache_fp8, paged_decode_attn_fp8
sigmoid_mul(gate, attn_output) → gated[4096]
O proj GEMV [2048, 4096]                                     ← NVFP4
residual_add + post_attn_layernorm
MoE (512 NVFP4 experts, top-10, same pipeline as target)
residual_add
final_norm → shared lm_head (NVFP4) → argmax
```

**New CUDA kernels** (added to `cuda_kernels/residual_add.cu`):
1. `bf16_concat(a, b, out, N)` — Concatenate two [N] BF16 vectors into [2N]

**Kernel modification**:
- `cuda_kernels/dense_gemv_bf16.cu` — `s_A[2048]` → `s_A[4096]` for K=4096 support (fc, o_proj)

**Files**:
- `cuda_kernels/residual_add.cu` — Add `bf16_concat` kernel
- `cuda_kernels/dense_gemv_bf16.cu` — Increase s_A to 4096
- `crates/atlas-kernels/src/lib.rs` — Update kernel count
- `crates/spark-model/src/layers/mtp_head.rs` — **New**: `MtpHead` struct
- `crates/spark-model/src/layers/ops.rs` — Add `bf16_concat()` dispatch
- `crates/spark-model/src/layers/mod.rs` — Add `pub mod mtp_head;`

**KV cache**: Add 1 extra layer to PagedKvCache (13 instead of 12). MTP uses FP8 KV with default scale=1.0.

---

## Step 4: Multi-Token Target Verification

**Key insight**: GEMV kernels are bandwidth-bound. Most per-layer weights fit in L2 cache (32 MB on GB10). Processing N tokens through a layer: first token reads from LPDDR5X, subsequent tokens hit L2. Cost is ~1.1x single decode, not Nx.

**Approach**: Process N tokens sequentially through each layer.

**Files**:
- `crates/spark-model/src/traits.rs` — Add `decode_verify()` to Model trait
- `crates/spark-model/src/model.rs` — Implement `decode_verify()`
- `crates/spark-runtime/src/buffers.rs` — Add draft logits buffers

**Cost estimate**: ~12-15ms for 3 tokens vs ~10ms for single decode (L2 caching).

---

## Step 5: SSM State Checkpoint & Rollback

**State size**: 36 layers × (2.05 MB h_state + 128 KB conv_state) = **78 MB per checkpoint**.

**Strategy**: Single checkpoint before verification. On partial acceptance: restore + replay K accepted tokens through SSM layers only.

**Files**:
- `crates/spark-model/src/layer.rs` — Add checkpoint buffers to `SsmLayerState`
- `crates/spark-model/src/layers/qwen3_ssm.rs` — Add `checkpoint_state()` / `rollback_state()`
- `crates/spark-model/src/model.rs` — Checkpoint/restore in `decode_verify()`

---

## Step 6: Generation Loop Integration

**Speculative decode loop**:
```
1. target.decode(last_token) → logits_0 + hidden_states          (10ms, CUDA graph)
2. token_0 = argmax(logits_0)
3. mtp.propose(token_0, hidden_states, N=2) → [draft_1, draft_2] (2ms)
4. checkpoint SSM states
5. target.decode_verify([token_0, d1, d2]) → [logits_1, logits_2, logits_3]  (12ms)
6. verify: v1=argmax(l1), v2=argmax(l2), bonus=argmax(l3)
7. mtp.after_verify(num_accepted) — trim MTP KV cache
```

**Expected throughput** (N=2, 80% acceptance): ~143 tok/s

**Files**:
- `crates/spark-model/src/engine.rs` — Add `generate_speculative()`
- `crates/spark-model/src/model.rs` — Add `hidden_states()` accessor
- `crates/spark-server/src/main.rs` — Use `generate_speculative()` if MTP weights present

---

## Step 7: CUDA Graph Optimization

Capture separate graphs for verification (fixed N=2) and MTP forward.

**Files**: `crates/spark-model/src/model.rs` — Add `verify_graph: Mutex<Option<GraphHandle>>`

---

## Implementation Order

```
Step 0 (coherence) → Step 1 (abstraction) → Step 2 (weights) → Step 3 (MTP head)
                                                                      ↓
                              Step 6 (gen loop) ← Step 4 (verify) + Step 5 (SSM ckpt)
                                    ↓
                              Step 7 (graphs)
```

## Expected Throughput Progression

| After Step | Expected tok/s | Why |
|-----------|---------------|-----|
| Step 0 | 97 (unchanged) | Verification only |
| Step 3 | 97 (test MTP standalone) | No integration yet |
| Step 6 | 130-170 | Speculative decode active |
| Step 7 | 140-180 | Launch overhead eliminated |
