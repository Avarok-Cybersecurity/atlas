# RST session — C0 config / factory / weight-map (in flight)

CHARTER
-----------------------------------------------
Find whether the official Kimi K3 `config.json` is actually parsed as `kimi_k3` (nested `text_config`, not aliased to DeepSeek/GLM), and whether the 96-shard class map refuses a missing required text tensor.

#AREAS
C0
config-parser
weight-map-dry-run

ORACLE
Claims: PRD geometry vs vendored `docs/k3/fixtures/moonshotai-Kimi-K3-config.json`.
Product: TSV `docs/k3/official-weight-classes.tsv` (497220 tensors / 96 shards / 59 classes).
Known-bad (must fail before C0 is green):
1. Drop `text_config` / parse only the wrapper → error, not a silent vision-only config.
2. Synthetic weight_map with 95 shards → refuse.
3. Omit `language_model.model.layers.*.block_sparse_moe.experts.*.w1.weight_packed` → refuse missing key.
4. Alias to `deepseek_v3` or `glm5_next` → fail the factory arm test.

STOP
Not yet. Parser implementer still running. This sheet is the charter; fill Notes/Bugs when tests exist.
