# RST session — K3 0.40B TP=2 on two Sparks

CHARTER
-----------------------------------------------
Find whether Atlas `spark-k3 --world-size 2 --tp-size 2` actually shards the 0.40B twin across spark1+spark2, and whether greedy tokens match TP=1.

ORACLE
TP=1 aviation greedy (bee-fly, first id 1459). Dual-node TP=2 must match through at least 16 tokens. Known-bad: kill rank 1 or drop rank-1 shard → not bee-fly / error.

KNOWN-BAD (loader, 2026-09-13)
`--tp-size 2` fail-fast: `TP (--tp-size > 1) is not supported by the kimi_k3 weight loader` because `supports_tp()` was false. That is why C7 must stay **unchecked** — dummy `o_proj` NCCL is not 0.40B TP.

In-tree (`2ed1455`+) the loader returns true and `slice_for_rank`s Q/K/V/O, KDA companions, MLA q_b/kv_b/g/o, dense gate/up/down, expert w1/w3/w2. Unit tests on mock GPU are not this charter. Dual-node: TP=2 aviation == TP=1 bee-fly; drop rank 1 must not be bee-fly.

Launch: `HEAD_IP=... WORKER_IP=... bash docs/k3/scripts/start-k3-tp2.sh` (IPs from env). Both ranks must run a binary built **after** `2ed1455`. Pin RoCE `enp1s0f1np1` / `rocep1s0f1`. Live spark1 `:8888` is still the TP=1 CUDA-MLA binary.

STOP
Charter not complete. Umbrella C7 unchecked until dual-node 0.40B TP=2 tokens == TP=1.
