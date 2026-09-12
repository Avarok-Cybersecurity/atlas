# RST session — C7 dummy TP=2 tokens (in-process; not rental-green)

CHARTER
-----------------------------------------------
Find whether a production-width dummy (hidden=7168, 96 heads, 2 layers) greedy-decodes the same tokens at TP=2 as at TP=1 when `o_proj` is column-split and the hidden is allreduced. Not spark1+spark2. Not `nccl_2rank_bench`.

#AREAS
C7
cpu-tp
o_proj-column-split

START
2026-09-12

ORACLE
Self-consistency: same dummy weights, same prompt ids, greedy ≥8 new tokens, f32. TP=1 = unsplit `o_proj`. TP=2 = two in-process ranks, column shards, sum. Identity `o_proj` allreduce is bit-exact vs unsplit GEMV.

KNOWN-BAD
Drop rank-1 `o_proj` shard (zero half the columns) must change greedy tokens. Instrument must fail this mutant before TP=1 vs TP=2 is trusted.

TEST NOTES
Mac `cargo test -p atlas-core --lib kimi_k3::c7 --no-default-features` — **6 passed**, 4.53s.
- `c7_column_tp2_ident_allreduce_is_bit_exact`
- `c7_tiny_tp2_matches_tp1_tokens` / `c7_tiny_drop_rank1_o_proj_shard_changes_tokens`
- `c7_prod_width_dummy_shape` (7168 / 96 / 2 layers / vocab 256 / 8 routed)
- `c7_dummy_tp2_matches_tp1_tokens` — TP=1 == TP=2 greedy 8
- `c7_drop_rank1_o_proj_shard_changes_tokens` — known-bad diverged

This is **not** dual-Spark NCCL TP. `nccl_2rank_bench` measures fabric algbw; it does not compare tokens.

BUGS
#ISSUE
Umbrella C7 (spark1+spark2 == spark1 single-GPU tokens) stays open. Dual-node known-bad is still kill rank 1 mid-decode.

TEST NOTES (spark1+spark2, 2026-09-12)
`docs/k3/scripts/c7_dummy_tp2_nccl.py` in `dspark-vllm-gx10` `--network host`, RoCE pin `enp1s0f1np1` / `rocep1s0f1`.
- Clean: `tp1 == tp2` greedy 8 after prompt `[1,2,3,4]` (`MATCH True`).
- Known-bad `C7_DROP_RANK=1`: `tp2` diverged (`DROP True`).

STOP
Dual-node dummy C7 **green**. Not a full K3 graph TP. Kill-rank-1-mid-decode still a follow-up, not required to check the dummy-token box.
