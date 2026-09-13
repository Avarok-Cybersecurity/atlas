# RST session — S6 spark serve 0.40B twin

CHARTER
-----------------------------------------------
Find whether `spark serve` of the 0.40B twin answers OpenAI `/v1` after the CPU-fallback decode, and whether the tokens are the C1 greedy ones.

#AREAS
S6
spark-serve
tokenizer.json

START
2026-09-13

ORACLE
Claims: `GET /v1/models` 200, `POST /v1/completions` 200.
C1: greedy temperature=0 on `According to all known laws of aviation,` should start like HF (first id 1459).
Known-bad: missing `tokenizer.json` → boot fail `Failed to load tokenizer`.

KNOWN-BAD (observed)
No `tokenizer.json` (tiktoken.model only) refused boot. Converter from tiktoken.model matches C1 aviation ids `[18805, 308, 799, 5624, 12524, 318, 57195, 11]`. Vocab 163584 vs model 163840 (256 specials not in json).

TEST NOTES
- Binary: spark2 CUDA release `6d9aaf3` + `117620f8e` fallback, copied to spark1 `spark-k3`.
- Boot: live at :8888 `kimi-k3-0.40b` with `--dangerously-allow-unresolved-kernel-lookups` (w4a16 nvfp4 lookup on a BF16 twin).
- EOS from config.json `text_config.eos_token_id=2` (wrong; tokenizer EOS is 163585).
- Completions temperature=0 max_tokens=16: HTTP 200 in 429 ms, text `自主性!!!!!!!!!!!!!!!` — **not** C1 Bee Movie. Chat and completions same. CPU fallback is not matching C1 on the serve path (likely prefill `seq_len>1` vs token-outer CPU, and/or EOS 2).
- Unresolved: `w4a16::w4a16_gemm_t_p3` for `(kimi-k3, sm_121, nvfp4)`.

BUGS
#BUG
Serve greedy ≠ C1 HF greedy. Do not call S6 product-complete.
#ISSUE
`eos_token_id=2` in twin `text_config`. Real EOS is 163585.
#ISSUE
No native tiktoken.model loader; we generated tokenizer.json.

STOP
Charter complete for "does it answer HTTP". Quality vs C1 is a fail. CUDA KDA still parked.
