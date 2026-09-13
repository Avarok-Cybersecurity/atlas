# RST session — K3 0.40B TP=2 on two Sparks

CHARTER
-----------------------------------------------
Find whether Atlas `spark-k3 --world-size 2 --tp-size 2` actually shards the 0.40B twin across spark1+spark2, and whether greedy tokens match TP=1.

ORACLE
TP=1 aviation greedy (bee-fly, first id 1459). Dual-node TP=2 must match through at least 16 tokens. Known-bad: kill rank 1 or drop rank-1 shard → not bee-fly / error.

KNOWN-BAD (loader, 2026-09-13)
`--tp-size 2` used to fail-fast (`supports_tp()` false). Loader now returns true and `slice_for_rank`s Q/K/V/O, KDA q/k/v/o/g (+ conv/A_log/dt_bias/b_proj/f_b), MLA q_b/kv_b/g/o, dense gate/up/down, expert w1/w3/w2. Unit test `drop_rank1_o_proj_shard_changes_hidden` / rank-0 vs rank-1 `q_proj` bytes: drop rank-1 changes the hidden. Dual-node: `C7_DROP_RANK=1` or kill rank 1 must not be bee-fly.

Launch: `HEAD_IP=... WORKER_IP=... bash docs/k3/scripts/start-k3-tp2.sh` (IPs from env). Both ranks must run this commit's `spark-k3` (`BIN=`). Pin RoCE `enp1s0f1np1` / `rocep1s0f1`.

STOP
`supports_tp` is true. Charter not complete until dual-node boots on RoCE and TP=2 tokens == TP=1.
