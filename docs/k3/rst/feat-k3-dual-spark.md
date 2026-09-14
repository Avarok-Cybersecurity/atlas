# RST session — feat/k3-dual-spark (extracted)

CHARTER
-----------------------------------------------
Find whether K3 `supports_tp` is true (the old loader refuse is gone) and whether TP=2 reconstructs the same full q_proj/o_proj plan on rank 0 and rank 1.

AREAS
feat/k3-dual-spark
C7 (lab dual-Spark aviation MATCH + kill rank 1 timeout: umbrella `c7-k3-tp2.md`)

ORACLE
- `supports_tp() == true` — `--tp-size 2` must not fail-fast.
- Twin TP=2: q_proj column `[8*32, 1024]`, o_proj row `[1024, 8*32]`.
- Umbrella lab: spark1+spark2 aviation 16 tokens MATCH TP=1; kill rank 1 → TimeoutError, not bee-fly.

KNOWN-BAD
Old loader: `supports_tp` false. This slice asserts true.
Rank 0 vs rank 1 `tp_rank` must differ; full tensor_plan sizes must match.

TEST NOTES
`cargo test -p spark-model --lib -- kimi_k3::tp` (Linux). Atlas-core plan tests run wherever spark-model compiles.
Do not put lab IPs in git. Pin `enp1s0f1np1` / `rocep1s0f1`.

BUGS
#N/A this slice. Kill-rank-1 hang-forever is a lab serve concern; umbrella timed out.

STOP
Charter complete for TP plan + supports_tp. Full bind/allreduce stays on the umbrella serve path until BoundLayer extract.
