# GLM-4.7-Flash support in Atlas — implementation plan

## Goal

Add `Glm4MoeLiteForCausalLM` (`model_type = "glm4_moe_lite"`) as a first-class
model target in Atlas so `spark serve GadflyII/GLM-4.7-Flash-NVFP4` boots and
serves OpenAI-compatible completions on GB10. Target throughput: ≥80 tok/s
without MTP, ≥130 tok/s with MTP (matching Qwen3.6-35B-A3B). Current best on
this hardware via vLLM + n-gram: 44 tok/s — Atlas should give us **~2–3×**.

## Architecture summary

GLM-4.7-Flash = DeepSeek-V3-mini at 35B-A3B:

| Property | Value | Atlas reference |
|---|---|---|
| `model_type` | `glm4_moe_lite` | new arm in `factory.rs` |
| Layers | 47 (layer 0 dense FFN, layers 1–46 MoE) | `first_k_dense_replace=1` |
| Hidden | 2048 | matches Qwen3.6 |
| Attention | **MLA** q_lora=768, kv_lora=512, qk_nope=192, qk_rope=64, v=256 | `mistral_loader.rs` (MLA-GQA fallback), `qwen3_attention/decode/attention_forward_mla.rs` |
| Heads | 20 Q, 20 KV (effectively MHA after MLA projection) | new |
| MoE | 64 routed + 1 shared, top-k 4, `noaux_tc` sigmoid + bias correction, scale 1.8 | minimax.rs (sigmoid + correction_bias), qwen35 (shared expert) |
| MTP | 1 nextn layer at index 47 with `eh_proj`, `enorm`, `hnorm`, `embed_tokens`, full attn+MoE block | qwen35 MTP (single-module) + minimax multi-module |
| Quant | NVFP4 compressed-tensors (`weight_packed/scale/global_scale/input_global_scale`), attention left BF16, lm_head + embed BF16 | qwen35 NVFP4 variant |
| Vocab | 154880; rope_theta 1e6; max_pos 202752 | new |
| Chat template | thinking + tool-call w/ `<tool_call>` blocks, `<think>` tags | new `jinja-templates/glm4_5.jinja` |

## Phased delivery

### Phase 0 — Scaffolding (THIS SESSION) ✅
Produce a buildable but unimplemented skeleton so subsequent PRs are surgical.
Files:
- `crates/spark-model/src/factory.rs` — add dispatch arm
- `crates/spark-model/src/weight_loader/mod.rs` — re-export `Glm4LiteWeightLoader`
- `crates/spark-model/src/weight_loader/glm4_lite.rs` — skeleton w/ TODO!()
- `kernels/gb10/glm-4.7-flash-a3b/nvfp4/MODEL.toml` — kernel target config
- `kernels/gb10/glm-4.7-flash-a3b/nvfp4/KERNEL.toml` — kernel manifest
- `jinja-templates/glm4_5.jinja` — chat template (ported from HF)
- `docs/design/GLM-4.7-FLASH-IMPL-PLAN.md` — this plan, in-repo

### Phase 1 — Config plumbing
- Confirm `ModelConfig::parse_config` reads all GLM-specific fields:
  `n_routed_experts`, `n_shared_experts`, `first_k_dense_replace`,
  `num_nextn_predict_layers`, `routed_scaling_factor`, `topk_method`,
  `q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`,
  `v_head_dim`.
- Wire any missing fields with safe defaults.
- Unit test in `crates/atlas-core/src/config.rs::tests` loading the
  GLM-4.7-Flash `config.json`.

### Phase 2 — Weight loader (MLA + dense + MoE)
- `load_embedding`/`load_final_norm`/`load_lm_head` — trivial.
- Layer 0: dense FFN (gate_proj/up_proj/down_proj) via the standard
  NVFP4 compressed-tensors path (`load_dense_ffn_nvfp4`-equivalent —
  reuse `qwen35_dense::load_layers` helpers).
- Layers 1–46: MoE block matching DeepSeek-V3 shapes:
  - Gate: BF16 `[n_routed, hidden]` + `e_score_correction_bias`
    `[n_routed]` (bias-corrected noaux_tc routing).
  - 64 routed experts each {gate_proj, up_proj, down_proj} in NVFP4.
  - 1 shared expert {gate_proj, up_proj, down_proj} in NVFP4.
- Attention (all 47 layers): MLA in BF16 (per `ignore: re:.*self_attn.*`):
  - `q_a_proj [q_lora_rank, hidden]`, `q_a_layernorm [q_lora_rank]`,
    `q_b_proj [num_heads*(qk_nope+qk_rope), q_lora_rank]`.
  - `kv_a_proj_with_mqa [kv_lora_rank + qk_rope_head_dim, hidden]`,
    `kv_a_layernorm [kv_lora_rank]`,
    `kv_b_proj [num_heads*(qk_nope+v_head_dim), kv_lora_rank]`.
  - `o_proj [hidden, num_heads*v_head_dim]`.
  - Use `mistral_loader.rs::gpu_matmul` MLA→GQA expansion as a
    correctness baseline first (no KV-cache compression), then enable
    the native MLA decode path in `qwen3_attention/decode/attention_forward_mla.rs`
    once decode is numerically validated.

### Phase 3 — MoE routing (noaux_tc + sigmoid + bias correction)
- New `MoeLayer::new_noaux_tc(...)` (or extend MiniMax's sigmoid path with
  `routed_scaling_factor`) — formula:
  `scores = sigmoid(gate @ hidden) + bias`
  `topk = argsort_top4(scores)` → renormalize topk values and multiply by
  `routed_scaling_factor`.
- Shared expert output always added (weight = 1).

### Phase 4 — MTP head (1 module, DeepSeek-V3 layout)
- Load tensors at `model.layers.47.*`: `eh_proj`, `embed_tokens`, `enorm`,
  `hnorm`, plus a full attention + MoE block matching layers 1–46.
- `eh_proj` is NVFP4 `[hidden, 2*hidden]`; concat-then-project semantics
  match Qwen3.5's `mtp.fc` but with `eh_` prefix and NVFP4-quantized.
- Wire into `MtpHead` / `MtpMulti` (1 module).

### Phase 5 — MODEL.toml + behavior
- `kernels/gb10/glm-4.7-flash-a3b/nvfp4/MODEL.toml` declares:
  - `model_types = [{ model_type = "glm4_moe_lite", hidden_size = 2048 }]`
  - Sampling presets (GLM card: T=0.6 thinking, T=0.8 non-thinking,
    top_p=0.95, top_k=40).
  - `thinking_default = true`, `max_thinking_budget = 4096`,
    `default_num_drafts = 1`.
- Chat template name aligns with HF (`glm4_5.jinja`).

### Phase 6 — Chat template + tool-call parser
- Port `chat_template.jinja` from the HF repo.
- Add a GLM-4.7 tool-call parser in `crates/spark-server/src/tool_call/`:
  format is `<tool_call>{json}</tool_call>` (similar to Qwen3 but with
  GLM-specific newlines). Confirm with a smoke test.

### Phase 7 — Build, smoke test, benchmark
- `cargo build --release` with `ATLAS_TARGET_MODEL=glm-4.7-flash-a3b`.
- `./target/release/spark serve GadflyII/GLM-4.7-Flash-NVFP4 --port 9999`.
- Validate text completion correctness via the existing `bench/bench-quick.py`.
- Run `benchmark_all_engines.sh` in the GLM repo against Atlas as a 4th
  engine to land the head-to-head comparison.

## Risks / unknowns

1. **MLA forward correctness on GLM dims.** Atlas's MLA kernel was authored
   against DeepSeek/Mistral dimensions. GLM uses 20 heads × {nope=192,
   rope=64, v=256} = q_dim=5120, v_dim=5120. KV-cache shape is
   `[seq, kv_lora + qk_rope] = [seq, 576]`. Verify the kernel's templated
   head_dim accepts 256 and the q-split logic handles `qk_nope != head_dim`.
2. **noaux_tc + sigmoid + bias correction.** MiniMax has the bias correction
   loaded but routing dispatch was still pending at last review (see
   `minimax.rs` line 28: "structurally correct but quantitatively wrong").
   GLM needs this finished, possibly as the first concrete user.
3. **MTP architecture mismatch.** GLM's `eh_proj` is NVFP4 (vs Qwen3.5's BF16
   `mtp.fc`). The existing `load_mtp` helper assumes BF16; needs an NVFP4
   variant.
4. **Vocab size 154880.** Larger than every existing target except Qwen3.6
   (248320) and MiniMax. lm_head + embed BF16 cost ≈ 1.2 GB — fits easily.
5. **Atlas community fit.** Per AGENTS.md, "don't regress on models already
   in the support matrix." Adding GLM should not touch shared kernels;
   keep all changes additive.

## Out of scope for this PR series

- Tool-call regression suite for GLM-4.7-Flash (separate PR after smoke test).
- DFlash drafter for GLM (no public DFlash drafter for GLM exists).
- Multi-GPU TP (GB10 is single GPU; TP=1 covers the use case).
- Vision (GLM-4.7-Flash is text-only; vision sibling is GLM-4.6V-Flash, separate target).

## Tracking

This plan is the design doc. SQL todos track concrete coding tasks; see
`todos` table in the session DB.
