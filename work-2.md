# GLM-4.7-Flash Atlas Integration — Work Summary

**Date:** 2026-05-19  
**Branch:** `feat/glm-4.7-flash-scaffolding`

---

## ✅ What Works

### Infrastructure / Startup
- **`start-glm.sh`** — complete launch script: starts Atlas (port 9999) + LiteLLM proxy (port 11111), waits for readiness, sets up Ctrl-C cleanup.
- **GPU memory** — with a clean GPU (only ~311 MB Xorg usage), Atlas loads cleanly:
  - 37213 weight tensors in ~90s
  - 90 PTX kernel modules compiled/loaded
  - KV cache: 41k+ blocks × 47 layers = ~66 GB at 26832 seq len; 657k+ max tokens
- **180K context** works when `--gpu-memory-utilization 0.85` is set *and* GPU has ~113 GB free.
- **`--scheduling-policy slai`** is set in `start-glm.sh`.
- **LiteLLM auth disabled** — `disable_auth: true` in `lite_llm_config_glm.yaml`; no API key required from VS Code Copilot side.
- **CUDA_ERROR_ILLEGAL_ADDRESS (700) at layer 33 is fixed** — `crates/spark-runtime/src/buffers/sizes.rs` patch (commit `a15698f`) adds `.max(k_max * config.intermediate_size)` for the GLM layer-0 dense FFN buffer. Server no longer crashes.
- **Server API** — `/v1/models`, `/v1/chat/completions` (non-streaming and streaming) all respond correctly.

### KV Cache Sizing — key rule
```
seq_len=180000 needs ~107 GB free GPU
seq_len=26832  needs ~6 GB   free GPU  (safe fallback)
```
With only 23 GB used at peak weights, 180K works fine when GPU is clean.

---

## ❌ What Does Not Work

### Model output is garbage (primary open bug)

**Symptom:**
```
"content": "\\\\\\               \t\t\t\t\t\t\t\t\t\t\\0\t\t\t\t\t}=\ <|code_suffix|>\\\\1xml202\t\t\t\t\\\\"
```
Every inference response is incoherent garbage — wrong tokens, `<|code_suffix|>` (a FIM token), backslash noise, null bytes.

**Root cause investigation:**

#### Bug 1 (FIXED in binary, but output still garbage)
- **Location:** `kernels/gb10/common/dense_gemm_tc.cu` lines 60–73
- **Bug:** A-tile shared memory loading used `if (idx < 256)` with `idx = tid` (0..127). This only loaded rows 0..7 of a 16-row tile. Rows 8..15 stayed uninitialized (stale SMEM from prior kernel).
- **Effect:** Token index 8+ in any prefill chunk had corrupted Q and K_rope values → overflow to 2.4e31 → inf in KV cache → all decode tokens garbage.
- **Fix applied:** Changed to `for (idx = tid; idx < TC_TM * TC_TK; idx += TC_BLOCK)` so each thread covers 2 elements (all 256 covered).
- **Status:** Fix is in `dense_gemm_tc.cu`, binary rebuilt (timestamp `May 19 11:57`), PTX in binary confirmed to contain the 2-iteration A-tile loop.
- **BUT: output is still garbage after the fix.** The A-tile fix was necessary but not sufficient.

#### Bug 2 (undiagnosed — remaining open issue)
After applying the A-tile fix and rebuilding, inference still returns garbage. The diagnostic env var `ATLAS_DIAG_GLM=1` would print layer-by-layer tensor stats (norm, max, nan/inf), but we could not capture a full diagnostic run successfully due to server instability during rapid kill/restart cycles (OOM watchdog fires). 

**What the garbage tokens suggest:**
- `<|code_suffix|>` is a FIM (fill-in-the-middle) special token — its presence means the top-1 predicted token at decode step 1 is completely wrong.
- This points to either: the KV cache being written with garbage values (same inf-in-cache symptom), OR the lm_head/softmax receiving corrupt hidden states.
- The prior `qg_out` max=2.4e31 was specifically for token index 8 at layer 4. With the A-tile fix, layer 4 token 8 should now be correct — but there could be a *different* layer or a different kernel with the same partial-load bug.

**Likely next bugs to check:**
1. Other kernels that call `dense_gemm_tc` — are there other callers that pass an M > 8? Check all call sites in `cache_skip_mla.rs` and other attention/FFN files.
2. The MoE FFN path for GLM (`ffn.forward_prefill`) — the original CUDA 700 crash was in `ffn.forward_prefill` at layer 33. The buffer sizing fix resolved the crash but the FFN output itself may still be corrupt if another kernel has a similar partial-load bug.
3. The NVFP4 dequant kernels — any kernel that processes a 16-row tile might have the same `if (tid < N)` pattern instead of a strided loop.

---

## Server Stability Notes

**OOM watchdog kills the server if GPU memory < 2 GB** — on unified memory GB10, rapid kill/restart doesn't release GPU memory instantly. Wait 5–10s after killing before restarting.

**Correct restart procedure:**
```bash
./kill.sh
sleep 8
./start-glm.sh
```

**The previous `CUDA_ERROR_ILLEGAL_ADDRESS` crashes** that appeared in earlier sessions were caused by the layer-33 FFN buffer overflow (now fixed), NOT by the A-tile bug. The A-tile bug produced silent numerical corruption instead.

---

## Files Changed (This Session and Prior)

| File | Change |
|------|--------|
| `kernels/gb10/common/dense_gemm_tc.cu` | A-tile loop fix (rows 8–15 now loaded) |
| `crates/spark-runtime/src/buffers/sizes.rs` | Layer-0 dense FFN buffer sizing fix (commit `a15698f`) |
| `crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_mla.rs` | `ATLAS_DIAG_GLM` diagnostic probes |
| `start-glm.sh` | Full launch script (Atlas + LiteLLM) |
| `/home/sna/ai-projects/lunch-model/lite_llm_config_glm.yaml` | `disable_auth: true`, dual model aliases |

---

## Immediate Next Steps

1. **Capture `ATLAS_DIAG_GLM=1` output** — start server, wait for full load, send one request, read log *before* killing. This will show whether qg_out/k_rope still have inf values after the A-tile fix.
2. **Search for other partial-load bugs** — grep all CUDA kernels for `if.*tid.*<` patterns inside K-loops that should be strided loops:
   ```bash
   grep -rn "if.*tid.*<\|if.*threadIdx" kernels/gb10/ --include="*.cu" | grep -v "for ("
   ```
3. **Check MoE FFN kernel** — the `ffn.forward_prefill` path uses a different kernel; inspect its tile loading for the same pattern.
4. **If diagnostics show inf in qg_out still** — the for-loop fix may have been compiled away or there's a second GEMM caller with the same bug.
5. **Commit the A-tile fix** once output is verified clean:
   ```
   kernels: fix dense_gemm_tc A-tile partial load (rows 8–15 uninitialized)
   ```
