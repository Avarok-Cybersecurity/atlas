# RST session — C0 config / factory / weight-map

CHARTER
-----------------------------------------------
Find whether the official Kimi K3 `config.json` is actually parsed as `kimi_k3` (nested `text_config`, not aliased to DeepSeek/GLM), and whether the 96-shard class map refuses a missing required text tensor.

#AREAS
C0
config-parser
weight-map-dry-run

START
2026-09-11

TESTER
umbrella / review of `48b9c58a4` + follow-up known-bads

ORACLE
Claims: PRD geometry vs vendored `docs/k3/fixtures/moonshotai-Kimi-K3-config.json`.
Product: TSV `docs/k3/official-weight-classes.tsv`.
Known-bad (instrument shown to fail):
1. Wrapper JSON with `text_config` removed → `parse_kimi_k3_refuses_wrapper_without_text_config` (missing `linear_attn_config`).
2. 95-shard map → `require_shard_count(..., 96)` errors `got 95`.
3. Drop `output_attn_res_proj` → `missing required text classes`.
4. `load_layers` → `K3-WIP` bail (graph not implemented).

TEST NOTES
- Official parse: 93 layers, 69 KDA / 24 MLA, last layer MLA, 896 experts top-16, 2 shared, situ 4/25, attn_res 12, NoPE + output gate, full-rank gate, conv 4, q/kv lora 1536/512, weight_prefix `language_model`. Ran on this Mac: 5/5 atlas-core tests.
- 1-based HF lists: layers 1–3 KDA, 4 MLA, 93 MLA → 0–2 / 3 / 92.
- Canonicalise inner `kimi_linear` → `kimi_k3`. Dispatch does **not** alias glm5_next / deepseek_v3.
- 0.40B twin fixture parses (hidden 1024, 8 layers, 6 KDA, last MLA). Twin omits `gate_lower_bound`; default **-5.0** so C1 can start. Production JSON still supplies the key.
- Dry-run: 48 text classes match TSV; vision ignored under `language_model_only`. spark-model `--lib` does not compile on macOS (`posix_fallocate` / `O_DIRECT` in spark-storage — pre-existing). Those tests ride Linux CI.
- `load_*` is WIP. C0 is parse + map, not a graph.

BUGS
#ISSUE
`inference-optimization/Kimi-K3-0.40B` `linear_attn_config` has no `gate_lower_bound`. Parser defaults -5.0. Recorded so C1 does not treat the default as "read from the twin JSON".

#ISSUE
spark-model kimi_k3 unit tests cannot run on this Mac (`posix_fallocate`). Closed on spark2 Linux 2026-09-12: **9 passed** (`cargo test -p spark-model --lib kimi_k3`).

STOP
Charter complete. Official parse + 96-shard map + known-bads (dropped text_config, missing class, 95 shards). Residual: no 57MB index in git (by design). C0 **green**.
