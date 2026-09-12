# RST session — S0 vLLM bake-off endpoint (in flight)

CHARTER
-----------------------------------------------
Find whether the lab Spark vLLM image will serve the same shipped Qwen3.8 NVFP4 MoE as Atlas, and what the instrument does when its attention backend is wrong.

#AREAS
S0-bakeoff
vllm-spark
flashinfer

ORACLE
Comparable-product: OpenAI `/v1/models` 200 and a greedy completion.
Instrument known-bad: the first boot *is* the known-bad — observe the crash, then change one variable.

KNOWN-BAD (observed)
Image `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`, default FlashInfer decode:
`TypeError: Mismatched number of arguments when calling: plan(...). Expected 19 but got 20 arguments`
in `flashinfer/decode.py` / `vllm/v1/attention/backends/flashinfer.py`.
Weights had already loaded (21.34 GiB, 137 s). Failure is compile/warmup, not missing files.

TEST NOTES
- First attempts: missing `preprocessor_config.json` (OSError) — copied from the spark1 tree. Then `--limit-mm-per-prompt image=0` rejected by this CLI.
- Retry in flight: `VLLM_ATTENTION_BACKEND=FLASH_ATTN --enforce-eager --max-model-len 4096`.

BUGS
#BUG
Default FlashInfer path in this GB10 vLLM image cannot plan decode for this Qwen3.8 checkpoint (19 vs 20 `plan()` args). Bake-off cannot use the un-flagged image.

STOP
Not yet. Waiting on FLASH_ATTN retry.
