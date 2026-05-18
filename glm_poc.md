# GLM-4.7-Flash on Atlas — POC Status

**Goal:** Run `GadflyII/GLM-4.7-Flash-NVFP4` on the Atlas pure-Rust inference engine to achieve ~2–3× speedup over the vLLM baseline (~44 tok/s).

---

## ✅ Done

### Phase 0 — Scaffolding
- `crates/spark-model/src/factory.rs` — `glm4_moe_lite` dispatch arm + unit test
- `crates/spark-model/src/weight_loader/mod.rs` — registers `Glm4LiteWeightLoader`
- `kernels/gb10/glm-4.7-flash-a3b/nvfp4/KERNEL.toml` — kernel manifest (reuses Qwen3.5 PTX set)
- `kernels/gb10/glm-4.7-flash-a3b/nvfp4/MODEL.toml` — model config (model-level, inside nvfp4/)
- `kernels/gb10/glm-4.7-flash-a3b/MODEL.toml` — model-level MODEL.toml with `[[model_types]]` entry (**untracked**)
- `jinja-templates/glm4_5.jinja` — chat template
- `docs/design/GLM-4.7-FLASH-IMPL-PLAN.md` — 7-phase design doc
- All compile, clippy, 65 tests pass

### Phase 2 — Weight Loader (`glm4_lite.rs`) — committed in `6eefad3`
- `load_embedding` → `model.embed_tokens.weight`
- `load_final_norm` → `model.norm.weight`
- `load_lm_head` → `lm_head.weight`
- `load_layers` — full 47-layer loop:
  - Layer 0: dense FFN (`intermediate_size=10240`)
  - Layers 1–46: noaux_tc MoE (64 experts, top-4, sigmoid routing + `e_score_correction_bias`)
  - MLA phases A–E with GLM tensor names (`model.layers.N.self_attn.*`)
  - Standard RoPE inv_freq (not YaRN)
- `load_mtp_weights` → `Ok(None)` (Phase 4 deferred)

### Config parsing fixes (uncommitted — `git diff HEAD`)
- `crates/atlas-core/src/config.rs` — `nullable_u32_or_array` deserializer for `eos_token_id` (GLM uses an array `[2, 151336, 151338]` not a scalar)
- `crates/atlas-core/src/config/dispatch.rs` — `glm4_moe_lite` parse branch: computes `head_dim`, maps `n_routed_experts`, sets `partial_rotary_factor`, clears `attn_gated`

### MLA kernels (untracked)
- `kernels/gb10/glm-4.7-flash-a3b/nvfp4/paged_decode_attn_mla.cu` — GLM-specific MLA decode kernel (HDIM=576: kv_lora_rank=512 + qk_rope_head_dim=64)
- `kernels/gb10/glm-4.7-flash-a3b/nvfp4/mla_absorbed.cu` — copied from Mistral (runtime-parametric)
- `KERNEL.toml` updated: added `paged_decode_attn_mla`, `mla_absorbed`, `moe_topk_sigmoid` entries + `-DHDIM=256` build flag

### Startup script (untracked)
- `start-glm.sh` — starts `spark serve GadflyII/GLM-4.7-Flash-NVFP4` on port 9999 + LiteLLM proxy on 11111

---

## 🔴 Current Blocker

**Smoke test fails:**
```
No compiled kernel target matches model_type 'glm4_moe_lite' / hidden_size=2048
Available targets: ['glm-4.7-flash-a3b']
```

**Root cause identified:** The `[[model_types]]` entry lives in `kernels/gb10/glm-4.7-flash-a3b/MODEL.toml`. This file was created **after** the last `cargo build --release`, and `atlas-kernels/build.rs` does **not** emit `cargo:rerun-if-changed` for `MODEL.toml` — only for `.cu` files and `KERNEL.toml`. So the binary was compiled without the `glm4_moe_lite` → `glm-4.7-flash-a3b` mapping baked in.

---

## 📋 What Needs To Be Done

### 1. Fix kernel target matching (blocker)
Commit all untracked/modified files, then force rebuild:

```bash
cd /home/sna/ai-projects/atlas

# Stage everything
git add kernels/gb10/glm-4.7-flash-a3b/MODEL.toml
git add kernels/gb10/glm-4.7-flash-a3b/nvfp4/MODEL.toml
git add kernels/gb10/glm-4.7-flash-a3b/nvfp4/KERNEL.toml
git add kernels/gb10/glm-4.7-flash-a3b/nvfp4/paged_decode_attn_mla.cu
git add kernels/gb10/glm-4.7-flash-a3b/nvfp4/mla_absorbed.cu
git add crates/atlas-core/src/config.rs
git add crates/atlas-core/src/config/dispatch.rs
git commit -m "feat(glm): kernel target, MLA kernels, config parsing fixes"

# Force atlas-kernels to rebuild (it won't pick up MODEL.toml changes otherwise)
touch crates/atlas-kernels/build.rs

# Build only the GLM target (faster)
ATLAS_TARGET_MODEL=glm-4.7-flash-a3b cargo build --release -p spark-server
```

> Also consider adding `println!("cargo:rerun-if-changed={}", model_toml_path.display());`
> inside `parse_model_types()` in `build_parse.rs` so this can't happen again.

### 2. Smoke test
```bash
./start-glm.sh   # in one terminal

# In another terminal, wait for "Atlas is up", then:
curl -s http://localhost:9999/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm4_moe_lite","messages":[{"role":"user","content":"hi"}],"max_tokens":20}' \
  | python3 -m json.tool
```

### 3. Benchmark vs vLLM
```bash
# vLLM baseline is ~44 tok/s (with n-gram speculative decoding)
# Run Atlas benchmark on port 9999:
cd /home/sna/ai-projects/atlas
python3 bench/bench-quick.py --port 9999 --model glm4_moe_lite
```

### 4. Deferred phases (lower priority)
| Phase | What | Notes |
|-------|------|-------|
| 4 | MTP head (`model.layers.47.*`) | Would enable speculative decoding; currently `Ok(None)` |
| 6 | GLM tool-call parser in `spark-server` | `<tool_call>` token parsing |
| Fix | Add `rerun-if-changed` for MODEL.toml in `build_parse.rs` | Prevents this class of bug recurring |

---

## File Map

| File | Status | Purpose |
|------|--------|---------|
| `crates/spark-model/src/weight_loader/glm4_lite.rs` | ✅ committed | Full weight loader |
| `crates/spark-model/src/factory.rs` | ✅ committed | `glm4_moe_lite` dispatch |
| `kernels/gb10/glm-4.7-flash-a3b/MODEL.toml` | ⚠️ untracked | `[[model_types]]` → runtime matching |
| `kernels/gb10/glm-4.7-flash-a3b/nvfp4/MODEL.toml` | ⚠️ untracked | Sampling presets |
| `kernels/gb10/glm-4.7-flash-a3b/nvfp4/KERNEL.toml` | ⚠️ modified | MLA + MoE module entries |
| `kernels/gb10/glm-4.7-flash-a3b/nvfp4/paged_decode_attn_mla.cu` | ⚠️ untracked | MLA decode kernel |
| `kernels/gb10/glm-4.7-flash-a3b/nvfp4/mla_absorbed.cu` | ⚠️ untracked | MLA absorbed kernel |
| `crates/atlas-core/src/config.rs` | ⚠️ modified | `eos_token_id` array fix |
| `crates/atlas-core/src/config/dispatch.rs` | ⚠️ modified | `glm4_moe_lite` config parse |
| `start-glm.sh` | ⚠️ untracked | Startup script |
| `jinja-templates/glm4_5.jinja` | ✅ committed | Chat template |

---

## Reference Numbers

| Engine | Model | tok/s |
|--------|-------|-------|
| vLLM baseline | GLM-4.7-Flash NVFP4 | ~40 |
| vLLM + n-gram speculative | GLM-4.7-Flash NVFP4 | ~44 (best) |
| Atlas + MTP (K=2) | Qwen3.5/3.6-35B-A3B NVFP4 | ~131 |
| **Atlas + GLM (target)** | **GLM-4.7-Flash NVFP4** | **~80–130 (estimated)** |
