# RST session — K3 0.40B TP=2 on two Sparks

CHARTER
-----------------------------------------------
Find whether Atlas `spark-k3 --world-size 2 --tp-size 2` actually shards the 0.40B twin across spark1+spark2, and whether greedy tokens match TP=1.

ORACLE
TP=1 aviation greedy (bee-fly, first id 1459). Dual-node TP=2 must match through at least 16 tokens. Known-bad: kill rank 1 or drop rank-1 shard → not bee-fly / error.

KNOWN-BAD (loader, 2026-09-13)
`--tp-size 2` used to fail-fast (`supports_tp()` false). That binary is gone.

TEST NOTES (2026-09-13, spark1+spark2, binary `afd01e60`, tree `87761b8`+)
- Both ranks: `kimi_k3: TP slice_for_rank rank=0/2` and `rank=1/2`. NCCL init on `enp1s0f1np1` / `rocep1s0f1`. `tp_rank=0/2` and `1/2`.
- TP=1 aviation (old serve): `'there is no way a bee should be able to fly. Its wings are too'`
- TP=2 `/v1/completions` T=0 max_tokens=16: **same 16 tokens**. `MATCH True`. finish=length.
- Known-bad: `pkill -x spark-k3` on rank 1, then the same request → `TimeoutError`, `DROP_STILL_MATCH False`. Rank 0 did not silently emit bee-fly.

Launch: `HEAD_IP=… WORKER_IP=… MASTER_ADDR=<spark1 RoCE> bash docs/k3/scripts/start-k3-tp2.sh` (addresses from env, not git).

BUGS
#N/A dual-node 0.40B TP=2 tokens matched TP=1; kill rank 1 hung instead of matching.

STOP
C7 **green**. Dummy NCCL o_proj is fabric-only and is not this box.
