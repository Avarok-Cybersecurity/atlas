# RST session — S0 Atlas serve (shipped Qwen3.8-27B NVFP4)

CHARTER
-----------------------------------------------
Find whether a hopper-era `spark` binary will serve a shipped NVFP4 MoE on spark1, and whether the OpenAI surface tells the truth about model identity and bad requests.

#AREAS
S0-bakeoff
atlas-api
claims-model-id

START
2026-09-11

TESTER
umbrella / lab

ORACLE
Claims: `--model-name` appears in `GET /v1/models`.
World: greedy `Reply with exactly the word pong.` → `pong`.
History: second identical greedy call matches the first.
Product: empty `messages` and `max_tokens=0` should 4xx, not 200 with junk.
Claims (OpenAI): unknown `model` should not silently run a different id.

KNOWN-BAD
- Empty messages → 400 `messages must contain at least one message` (instrument can fail).
- `max_tokens=0` → 400 `max_tokens must be at least 1`.
- Unknown model id → **did not fail** (see #BUG).

TEST NOTES
- First boot died: 11 unresolved Hopper kernel lookups on GB10 (`gdn_*_hopper`, `paged_decode_*_splitk_hopper`, …). Relaunch with `--dangerously-allow-unresolved-kernel-lookups`. That flag is part of the route identity; this is not a clean shipping binary.
- Bound `0.0.0.0:8888` after loopback-only first success.
- `GET /v1/models` → `id=Qwen/Qwen3.8-27B-NVFP4`, `owned_by=atlas-spark`, `max_model_len=8192`.
- Greedy pong: 254 ms then 240 / 231 ms, content `pong`, `finish=stop`. Determinism held for this prompt.
- `/v1/completions` 200, `finish_reason=length` (sanity that the legacy route exists).
- Weights: spark1 iron NVFP4 tree, not byte-identical to spark2's unsloth tree. Bake-off JSONL must say so.

BUGS
#BUG
Unknown model id is accepted (HTTP 200) and served as `Qwen/Qwen3.8-27B-NVFP4`.
Repro: POST `/v1/chat/completions` with `"model":"definitely-not-a-model"`.
Result: 200, response `model` field is the loaded checkpoint.
Expected (OpenAI claims): 404 / invalid_request for an unknown id, *or* documented one-model-per-process behaviour. Today it looks like the client id is ignored.

#ISSUE
Serve required `--dangerously-allow-unresolved-kernel-lookups` because the binary was copied from a hopper-gate tree. S0 bake-off numbers from this process are not a GB10 shipping receipt.

STOP
Charter complete enough to use Atlas as one bake-off endpoint. Residual: unknown-model remap; hopper-lookup flag.
