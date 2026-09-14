# RST session — C2/C3 CPU self-consistency (not rental-green)

CHARTER
-----------------------------------------------
Find whether OUR CPU graph is internally consistent: decode logits vs full prefill, cache hit vs cold. This is not C2/C3 vs HF.

#AREAS
C2
C3
cpu-cache

ORACLE
Self-consistency on `synthetic_tiny` / `synthetic_small`. Known-bad: skip a layer on decode only; stomp KDA conv slot / MLA KV.

TEST NOTES
`cargo test -p atlas-core --lib kimi_k3`: 37 pass including
- `c2_prefill_decode_logits_match_full_prefill_{tiny,small}` ATOL/RTOL 1e-5
- `c2_skip_layer_on_decode_path_diverges`
- `c3_prefix_cache_hit_matches_nocache_{tiny,small}`
- `c3_stomp_kda_conv_slot0_changes_tokens`, `c3_stomp_mla_kv_changes_tokens`

These gates are on a graph that **fails C1**. They prove cache/prefill wiring, not HF identity.

BUGS
#ISSUE
Do not check C2/C3 on the umbrella rental list until C1 token-exact is green (or the same tests run on a matching engine).

TEST NOTES (spark2, 2026-09-13, `K3_TWIN` 0.40B, `83c0d34`)
`cargo test -p atlas-core --lib -- _twin -- --test-threads=1 --nocapture`
- `c2_prefill_decode_logits_match_full_prefill_twin` **ok** — aviation prompt logits prefill-decode vs full-prefill; skip layer 0 on decode diverged.
- `c3_prefix_cache_hit_matches_nocache_twin` **ok** — 8 greedy ids match cold. Known-bad: trash-all KDA conv+recurrent after prefix write changed tokens. MLA-only kv stomp did **not** move 8 aviation ids (6 KDA / 2 MLA); not used as the mutant.

STOP
C2 and C3 **green** on the C1 twin. GPU paged-cache TODO remains; not required to check these boxes.
