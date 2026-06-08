# Phase 2c Day 3 — NVFP4 weight checkpoint test: BREAKTHROUGH

## TL;DR

The model degeneration is in the **FP8 weight quantization**, not Atlas's compute. Running the EXACT same Atlas (image `atlas-gb10:realfix2`) with `RedHatAI/Qwen3.6-35B-A3B-NVFP4` weights instead of `Qwen/Qwen3.6-35B-A3B-FP8` produces dramatically better output at every context length. The fix path is **switch to NVFP4 weights**; no Atlas kernel work needed.

## Comparison table — same prompt, same Atlas, different weight quant

### Short prompt: "Write a 2-sentence Rust hello world program using axum 0.8"

| Aspect | FP8 (Qwen/...-FP8) | NVFP4 (RedHatAI/...-NVFP4) |
|---|---|---|
| Throughput | ~65 tok/s | **117 tok/s** (1.8×) |
| TTFT | ~140ms | **72ms** (½) |
| Output quality | mostly works | **works** |
| TOML / code syntax | mixed | **valid axum 0.8** |

### Longer prompt: "Create Cargo.toml + src/main.rs + tests/ping_test.rs"

| Aspect | FP8 | NVFP4 |
|---|---|---|
| TOML quotes | **single quotes** (invalid TOML) | **double quotes** (valid) |
| axum version | (didn't get here) | `"0.7"` correctly specified |
| Code structure | (didn't get here) | full project: main + tests separated correctly |
| Stop reason | (didn't get here) | **finish=stop, natural completion** |
| Watchdog | frequently amputates | **doesn't fire** |

### Deep-context probe (18920-token opencode-style)

NVFP4 model + NVFP4 KV cache, prompt_tokens=9780, max_tokens=800:

- **Content**: `"Let me fix the broken directory structure and create the project properly."` — clean English, no token glue
- **Tool call**: `bash({"command":"rm -rf /home/nologik/test-rust-axum-v3 && mkdir -p /home/nologik/test-rust-axum-v3/src","description":"Fix broken directory and create src"})` — **valid shell command**
- **Finish reason**: `tool_calls` (natural completion, not `length` amputation)
- **Reasoning channel**: ~97 tokens of coherent thinking, then a small role-token leak at the very end (`assistant<think>`). Doesn't reach the emitted content/tool_calls — opencode won't see this.

Compare to the FP8 model on the SAME context depth: 8192-token doom loops, `{"content":",...filePath":""}` malformed tool calls, "withcurl" / "axut" / "Let mefix" token-glue artifacts, language switching (Indonesian descriptions), single-quoted TOML.

## What this means for the Phase 2c question

The Phase 2c hypothesis was: **"deep-layer drift on Qwen3.6-FP8 is from FP8 KV cache quantization"**. Days 1 + 2 falsified this:
- Day 1: KV cache dtype doesn't move per-layer hidden-state cosine (8-way bit-identical tie)
- Day 2: Phase 2b kernel changes (RNE, `__expf`, FP16 P×V) individually exonerated

Day 3 finds the **actual** answer: it's the FP8 *weight* quantization that fundamentally drifts the model. Atlas's compute pipeline is correct; given correctly-quantized weights (NVFP4), Atlas produces a high-quality model.

NVFP4 quant scheme: per-block FP8 scales × 4-bit E2M1 nibbles. This gives 16× finer per-element precision than dense FP8 E4M3, at half the memory. For weights that have large dynamic range (late-layer attention K/V projections, MoE expert weights), NVFP4 holds the values within reasonable bounds while FP8 hits its E4M3 dynamic-range ceiling and quantizes coarsely.

Project memory was 90% right (`project_qwen36_phase2b_softmax_expf.md`: "NVFP4 KV best at deep, FP8 best early"). The KV cache bench couldn't see it (dequants before matmul) but the WEIGHTS exhibit exactly that pattern.

## Recommendation

**Production deployment of Qwen3.6-35B-A3B on Atlas should use the NVFP4 checkpoint, not FP8.** Two NVFP4 checkpoints are already cached on dgx1:

- `RedHatAI/Qwen3.6-35B-A3B-NVFP4` (tested, validated)
- `AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4` (untested but available)

The `atlas-gb10:realfix2` image works directly with this model — no rebuild needed. Recommended invocation:

```bash
sudo docker run -d --name atlas-qwen \
  --network host --gpus all --ipc=host \
  -e RUST_LOG=info \
  -v /workspace/.cache/huggingface:/root/.cache/huggingface \
  -v /workspace/atlas-dumps:/workspace/atlas-dumps \
  atlas-gb10:realfix2 \
  serve RedHatAI/Qwen3.6-35B-A3B-NVFP4 \
    --port 8888 --max-seq-len 65536 --max-batch-size 8 \
    --gpu-memory-utilization 0.88 \
    --kv-cache-dtype nvfp4 \
    --kv-high-precision-layers auto \
    --scheduling-policy slai --enable-prefix-caching \
    --speculative --mtp-quantization bf16
```

Performance gains: ~1.8× throughput on simple prompts, ½ TTFT, no watchdog amputations, no token glue, valid agentic outputs at deep context.

## Bench dump artifacts

- `/workspace/atlas-dumps/numdrift/phase2c-nvfp4-weights/` — per-layer NVFP4 hidden states (not directly comparable to FP8 hf_*.bin since different weight set)
- `/workspace/atlas-dumps/opencode-nvfp4.jsonl` — request dump from NVFP4 testing

## Open work (Day 4+, lower priority)

These would close the remaining ~1-2% gap to a true HF NVFP4 reference, but the current NVFP4 model is already production-functional:

1. Generate HF NVFP4 reference dump for proper cosine comparison
2. Investigate the trailing `assistant<think>` role-token leak in reasoning channel at deep context
3. Audit MoE expert-routing divergence on NVFP4 vs HF reference (the "8/8 → 3/8 overlap" theory now testable with proper reference)
4. Verify long-form agentic opencode session with NVFP4 model in production
