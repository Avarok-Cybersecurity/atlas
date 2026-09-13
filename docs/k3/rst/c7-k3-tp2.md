# RST session — K3 0.40B TP=2 on two Sparks

CHARTER
-----------------------------------------------
Find whether Atlas `spark-k3 --world-size 2 --tp-size 2` actually shards the 0.40B twin across spark1+spark2, and whether greedy tokens match TP=1.

ORACLE
TP=1 aviation greedy (bee-fly, first id 1459). Dual-node TP=2 must match through at least 16 tokens. Known-bad: kill rank 1 or drop rank-1 shard → not bee-fly / error.

KNOWN-BAD (observed 2026-09-13)
`--tp-size 2` **fail-fast** before NCCL:
`TP (--tp-size > 1) is not supported by the kimi_k3 weight loader.`
`supports_tp()` is false. Worker copy of `spark-k3` was missing (`k3-lab/bin` absent) — secondary.

STOP
Charter not complete until `supports_tp` is true, dual-node boots on RoCE `enp1s0f1np1`/`rocep1s0f1`, and TP=2 tokens == TP=1.
