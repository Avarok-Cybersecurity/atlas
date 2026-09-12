# RST session — C1 HF goldens (0.40B)

CHARTER
-----------------------------------------------
Find whether we can freeze a token-exact HF greedy trace for `inference-optimization/Kimi-K3-0.40B`, and what that trace is actually evidence *of*.

#AREAS
C1
hf-twin
oracles

START
2026-09-12

ORACLE
HF `language_model.generate` greedy `do_sample=False`, 8 prompts × 128 new tokens.
Instrument known-bad: host CPU torch + triton = 0 drivers / cpu tensor in Triton. Control: CUDA `.to("cuda")` in the GB10 vLLM image.

KNOWN-BAD (observed)
1. Host `torch 2.12.0+cpu`: `RuntimeError: 0 active drivers`.
2. CUDA container without `.to(device)`: `Pointer argument cannot be accessed from Triton (cpu tensor?)`.
3. Missing deps: tiktoken, einops, fla-core (error text: "Plese run pip install -U fla-core").
4. vLLM image default entrypoint is `vllm`, not bash.

TEST NOTES
- Goldens: `docs/k3/goldens/kimi-k3-0.40b-greedy.json` (19741 bytes, 8 rows, ~133–137 tokens each including prompt).
- Generated on spark1 GPU inside `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`, `device cuda dtype bfloat16`.
- Continuations are **not** a capability oracle. Prompt 0 starts Bee Movie; several others collapse into Inigo Montoya / Sith paste. C1 is **token identity vs this file**, not "France is Paris".
- Script: `docs/k3/scripts/gen_kimi_k3_0_40b_goldens.py`.

BUGS
#ISSUE
0.40B twin greedy output is prompt-insensitive mush. Do not use it to judge K3 quality. Use it only as a bit-exact HF lock.

STOP
Goldens frozen. CPU bind (`7b1d1b52d`) 26 atlas-core tests green; HF token-exact still skipped until safetensors→`K3CpuModel` ingest (in flight). Twin quality is not an oracle.
