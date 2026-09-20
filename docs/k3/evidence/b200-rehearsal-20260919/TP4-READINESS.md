# Four-GPU readiness: B200 development candidate

The frozen runtime source is `083260451` and the server SHA256 is
`e36f14822006b4aa19329314c573325bdf97107bdd5b4a4a98f9d04cbfd6bac8`.
The candidate explicitly enables `K3_CUDA_DENSE=1`. This is one host with four
B200 GPUs connected by NVLink, not four hosts. These are development gates;
no certification campaign, seal or merge is claimed.

## Functional gates

- TP1 and TP2 exact-generation canaries passed against the retained baseline.
- TP4 passed 121 lifecycle checks across five cycles after 45 seconds idle:
  repeat and boundary-length generation, streaming agreement, cancellation
  recovery, concurrent clients and recovery requests.
- All owned processes exited zero within the bounded cleanup. Restarting the
  same binary reproduced the prior TP4 baseline's complete 16-token response,
  finish reason and token counts.
- Complete collective signatures agree at 118,883 submissions on every rank.
  The TP2 count cannot be reused: `ep_min_u32` broadcasts once per communicator
  rank per request, adding `(4-2)*122=244` submissions to the 118,639 TP2 count.
- A separate targeted 2/4-client check passed all nine requests against serial
  responses, including the recovery request. This proves request handling,
  not simultaneous multi-sequence kernels or continuous batching efficiency.

Compact receipts: [TP4 lifecycle](tp4-dense-lifecycle.json) and
[2/4-client concurrency](concurrency-2-4.json). Bulk logs are retained outside
Git; the compact receipts identify the binary and scope.

## Rank measurements

The GPU compute/collective-balance gate remains **inconclusive**. Functional
TP4 readiness passed, but the entire five-gate milestone is not claimed closed.
The lifecycle's 269 samples observed
process GPU-memory high-water of 3,786 / 3,784 / 3,784 / 3,784 MiB for ranks
0 / 1 / 2 / 3. Sampling was every 0.5 seconds; these values include context use
and can miss a transient peak. Similar memory or matching outputs alone does
not establish balanced computation.

Process CPU execution was 104.59 / 104.60 / 105.41 / 104.49 seconds over this
lifecycle capture. That is closely matched CPU activity, not GPU kernel time.
Supported Nsight Systems 2026.3.2 captured CUDA kernels, copies and APIs in an
exclusive, compiler-free window (40 occupancy samples, zero compiler overlap).
However, rank 0 emitted a possible CUPTI event-loss diagnostic at shutdown.
The warning was after the measured regions, but it does not locate the lost
buffers; matching collective counts cannot establish complete compute traces.
The capture is therefore rejected for a definitive GPU-balance conclusion;
[rejection receipt](nsys-timing-rejected.json) preserves the diagnostic.
The planned flushing retry was not run when rental experiments were ended.

A separate supported allocation capture has no driver/loss warnings but no
allocation lifetime/size event table. Exact allocator peak is unavailable.
API counts or durations cannot reconstruct it. Its compact audit is
[supported allocation audit](supported-allocation-audit.json).
NCCL kernel elapsed time includes transport and peer waiting; host stream
synchronization also includes other outstanding work. Neither is pure
collective waiting, and neither is substituted for a missing measurement.

## Full-model-dimension composition

The integrated fixture runs the actual host graph and CUDA callbacks for a
KDA+dense layer followed by MLA+LatentMoE/shared MLP at official TP4 rank-local
dimensions. It provisions the 16 selected experts behind the full 896-way
router, not the entire checkpoint. Five token steps include exact fresh-reset
replay and a history-cleared negative control. The B200 run passed; clearing
history changed the output by a maximum 0.06444836.

The fixture uses 2,210,299,904 bytes of host matrix payload and 697,761,792
bytes of tracked persistent uploaded weights. These exclude context,
scratch and temporary allocations. Fixture construction took 0.771 seconds;
the five-step graph took 0.369 seconds. Sparse synthetic values exercise full
physical tensor dimensions but do not establish real-model throughput or
quality. Existing independent kernel arithmetic evidence is reused.

See [full-shape test and fallback accounting](../../tp4-fullshape-preflight.md)
for shapes, reproduction and cost formulas. This composition test does not
exercise checkpoint loading, `K3BoundLayer` allocation, full depth or NCCL;
the separate small-model TP4 lifecycle covers the distributed serving path.

## Remaining full-model constraints

Even with dense/shared MLP and recurrent KDA state on GPU, CPU projections,
normalization, routing, AttnRes and intermediate expert processing remain.
The current MLA wrapper reuploads expanded KV history each token. At official
TP4 dimensions and 4,096 live tokens that history upload alone is an estimated
2.8125 GiB per rank per generated token. Remaining CPU projection weights imply
approximately 58.73 GB per rank per token of FP32 matrix reads. These are layout
estimates, not measured full-model transfers or throughput.

The official checkpoint's current shard payload is about 391.50 GiB per TP4
rank before caches and reserves, exceeding a B200's memory. TP8 reduces that
to approximately 214.61 GiB per rank. The four-GPU rehearsal is not a claim
that the full model fits on this rental or that B300/TP8 is validated.


## Spark follow-up priorities

1. Wire the existing resident MLA cache helper into serving. Cover snapshot,
   restore, prefix reuse, cancellation and reset before claiming the full-history
   uploads are gone. This removes the largest context-dependent transfer found.
2. Move KDA/MLA projections and latent MoE down/up off the CPU, keeping their
   intermediates on device. Preserve independent numerical comparisons and
   measure transfers; dense/shared CUDA alone does not remove host weight copies.
3. Keep expert SiTU and weighted mixing on GPU; reuse scratch and pointer tables
   instead of allocating and round-tripping them for every token.

Use the existing small fixture on one or two Sparks with explicit ownership of
available devices; preserve Deckard and avoid the other active Kimi work.
The newly added dense primitive is B200-specific, so GB10 needs a reviewed
kernel target before enabling that path. GB10/TP2 results do not replace the
outstanding B200/TP4 balance capture or a true four-rank full-dimension test.

The rank-local full-dimension test passed. An unverified four-rank extension
was deferred and is not part of the committed candidate or claimed evidence.
No additional rental tests or certification campaigns were started at wrap-up.

## Rental closeout

The user confirmed Vast instance `51664218` stopped on September 19, 2026
(Chicago date), with **$32.35 remaining account credit**. These are user-reported
status and balance, recorded at `2026-09-20T02:53:50Z`; no billing API verification
is claimed. See [closeout receipt](rental-closeout.json).
