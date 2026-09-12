# RST session — S0 fabric / NCCL

CHARTER
-----------------------------------------------
Find whether spark1↔spark2 actually carries NCCL over the cabled RoCE rail, and whether the documented Down HCA would lie to us.

#AREAS
S0-fabric
platform-roce
nccl-2rank

START
2026-09-11

TESTER
umbrella / lab

ORACLE
Claims: LAB pin `NCCL_SOCKET_IFNAME=enp1s0f1np1` `NCCL_IB_HCA=rocep1s0f1`.
Product: log line `via NET/IB` and `Connected all rings`.
Instrument known-bad: ICMP already 0% loss is not NCCL; NCCL without the pin is the mutant we did *not* rerun this session (parked).

KNOWN-BAD
`ibdev2netdev` shows `enp1s0f0np0` Down / `enp1s0f1np1` Up on both boxes. A pin of `f0` is the planted wrong iface. Not executed this session (see #ISSUE). Control that *was* hit: ICMP over the Up rail, 0% loss, before NCCL.

TASK BREAKDOWN
#DURATION short
#SESSION SETUP 40
#TEST DESIGN AND EXECUTION 50
#BUG INVESTIGATION AND REPORTING 10
#CHARTER VS. OPPORTUNITY 90/10

DATA FILES
docs/k3/logs/nccl-2rank-spark1-rank0-2026-09-11.log (IPs redacted)
docs/k3/logs/nccl-2rank-spark2-rank1-2026-09-11.log
docs/k3/logs/nccl-2rank-SUMMARY-2026-09-11.txt

TEST NOTES
- Rank 0 spark1, rank 1 spark2, `nccl_2rank_bench`, extra GB10 env from LAB.md.
- Observed: `Channel 00/0 : … via NET/IB/0` both directions, `GDR 0`, `PXN 0`.
- 16 KiB 30.39 µs 0.539 GB/s; 64 KiB 35.86 µs 1.827 GB/s; 1 MiB 121.43 µs 8.635 GB/s.
- 2026-09-03 prior (not this campaign's C7): 1 MiB 6.060 GB/s. Do not treat either number as a K3 kernel result.

BUGS
#BUG
`NCCL_IB_HCA=rocep1s0f0` (Down) does **not** fail. It silently uses `NET/Socket`. 1 MiB all-reduce **1.265 GB/s** vs pinned RoCE **8.635 GB/s**. Logs still print `Connected all rings`. A bake-off that forgets the pin will look "healthy" and be ~7× slow. 2026-09-12: `docs/k3/logs/nccl-downhca-rank0-2026-09-12.log`.

STOP
Charter complete including the known-bad. Pin is mandatory. Socket fallback is not RoCE.
