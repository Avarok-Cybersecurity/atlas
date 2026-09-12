# RST session — S0 vLLM bake-off endpoint

CHARTER
-----------------------------------------------
Find whether the lab Spark vLLM image will serve the same shipped Qwen3.8 NVFP4 MoE as Atlas, and what the instrument does when its attention backend is wrong.

#AREAS
S0-bakeoff
vllm-spark
flashinfer

START
2026-09-11

ORACLE
Comparable-product: greedy completion 200, not merely `GET /v1/models` 200.
Instrument known-bad: FlashInfer `plan()` arity. `/v1/models` 200 with a dead EngineCore.

KNOWN-BAD (observed)
1. Default FlashInfer decode `plan()` Expected 19 got 20. Weights loaded; crash at warmup/first request.
2. `GET /v1/models` 200 while EngineCore is already dead — **models is not a serve oracle**.
3. `VLLM_ATTENTION_BACKEND=FLASH_ATTN` still entered `flashinfer.py` on the first completion.
4. `VLLM_USE_FLASHINFER=0` is an unknown env on this image (`envs.py` warning).
5. `VLLM_ATTENTION_BACKEND=TORCH_SDPA` ignored: log `Using FLASHINFER attention backend out of potential backends: ['FLASHINFER', 'TRITON_ATTN']`.
6. Unknown model id → 404 (this instrument *can* fail). Contrast Atlas #BUG (200 remap).

TEST NOTES
- Image `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` (v0.25.2.dev).
- Missing preprocessor_config first; copied. `--limit-mm-per-prompt image=0` rejected by CLI.
- Atlas S0 smoke (not certified): ISL 128 / OSL 32 / C=1, greedy `ping` + 1..8, ~15.8 tok/s e2e non-stream, `docs/k3/logs/bakeoff-atlas-only-2026-09-11.jsonl`. hopper-lookup-allow binary.
- Retry in flight: `VLLM_ATTENTION_BACKEND=TRITON_ATTN` (the only listed alternative).

BUGS
#BUG
FlashInfer `plan()` 19 vs 20 args on this GB10 image + Qwen3.8 NVFP4. Env knobs that claim to select FLASH_ATTN / disable FlashInfer did not.

#ISSUE
`GET /v1/models` 200 is a lying health check once EngineCore has died.

TEST NOTES (cont.)
- CLI `--attention-backend TRITON_ATTN` is the knob that actually stuck (`Using AttentionBackendEnum.TRITON_ATTN backend`). Env vars did not.
- Greedy with `chat_template_kwargs.enable_thinking=false`: both engines emit `ping` + `1..8`. Without it, vLLM thinks aloud (`We need to respond to user...`) and the JSONL is not comparable.
- Think-off C=1 ISL128/OSL32 e2e (non-stream, not certified): Atlas ~15.7 tok/s, vLLM ~12.4 tok/s. Sibling NVFP4 trees, not byte-identical. Atlas hopper-lookup-allow. `docs/k3/logs/bakeoff-thinkoff-2026-09-11.jsonl`.

STOP
Charter complete enough for S0 "one shipped Atlas MoE and one vLLM produce a comparison JSONL". Residual: FlashInfer broken on this image; env backend knobs lie; `/v1/models` 200 is not a serve oracle; weights not identical.
