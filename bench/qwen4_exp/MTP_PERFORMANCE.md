# EXL3 performance and concurrency — 2026-09-05

**Current recommendation: two drafts, one active sequence by default.** One and
two drafts passed the intended model-card agentic check with preserve-thinking
on. Three drafts passed deterministic output checks, but its sampled agentic
run exceeded the wall budget and was cancelled; it is not an agentic pass.

Runtime commit: `ab0bdb7461cab728df3a60c9fcda09f847a69a4e`.
Native release binary SHA-256:
`a5b1063554e1e2beb998006cb6f8af5358252ba7633b5a6864e97dce2077d6b9`.
The branch includes `avarok/main` at `8682329cce0dd8bec1d3775704e978533b00bf7a`.

## Changes

- `542886a49`: fuse routing staging and activation replication into one CUDA
  launch. Local/remote IDs, FP16 rounding, slot order and expert GEMM/reduction
  geometry are preserved. `ATLAS_NO_EXL3_FUSED_INGRESS=1` restores two launches.
- `4f6c38078`: refuse the generic multi-sequence verifier for HC layouts; Qwen
  finishes each sequence's specialized verify/commit before another reuses the
  highway. Private draft KV capacity follows actual MTP sequence slots. Finished
  requests return KV blocks and release draft QSA buffers. Abandoned proposals
  rewind before reuse. Active MTP no longer creates a global diagnostic shadow
  state that could accumulate history across requests.
- `ab0bdb746`: remove unused draft snapshot storage and its prefix-sized D2H
  copy. QSA rollback already rewinds position marks. The old copy remains
  available with `ATLAS_QWEN4EXP_MTP_SNAPSHOT_AB=1` for reproducible diagnostics;
  the selected mode is logged. No replacement synchronization is needed on the
  supported serving path, whose draft operations use the same stream.

The concurrency change touches shared batched-verify admission. It does not
introduce a new cross-model trait. Complete EXL3 graph capture remains disabled;
cooperative capture support alone does not establish full-path safety.

## Controlled observations — preliminary, single harness

Same binary, checkpoint, flags and warmup; three repetitions of a greedy LRU
cache coding prompt capped at 512 output tokens, with low thinking effort and
speculation during thinking. All complete choices, including reasoning, match
across these widths and repeats.

| Drafts | Median request wall | Median server response tok/s |
| --- | ---: | ---: |
| 1 | 33.286 s | 15.525 |
| 2 | 31.134 s | 16.611 |
| 3 | 32.529 s | 15.891 |

Two drafts shortened this prompt's median wall time by about 6.5%. This is not
an agentic throughput claim or a general speedup estimate.

Neither small implementation optimization demonstrated a convincing serving
speedup in its isolated A/B. Fused ingress had mixed differences below 0.1% on
two short prompts. The snapshot diagnostic's median wall was 33.079 s versus
33.286 s without it, within the observed repeat variation. Removing unused
work is not sufficient evidence of an end-to-end improvement.

The standalone staging event harness observed approximately 6.15–6.37 us for
two launches versus 4.09–4.10 us fused at rows 1–4, H=2560/top-k=10 (five batches
per arm, 2,000 operations per batch, alternating order). Event timing includes
submission gaps and is separate from serving throughput.

## Agentic model-card sanity

Both `ATLAS_AGENTIC_SAMPLING=model-card` and
`ATLAS_AGENTIC_PRESERVE_THINKING=1` are explicit. Served reasoning effort is
`low`; the requests omit sampling overrides. These are one-shot sampled
trajectories, not comparable speed measurements.

| Drafts | Result | Turns | Wall | Run record |
| --- | --- | ---: | ---: | --- |
| 1 | Pass, all six checks | 11 | 389.9 s | `run-1788627462102414995` |
| 2 | Pass, all six checks | 16 | 671.8 s | `run-1788628364589419312` |
| 3 | Failed/cancelled after exceeding budget | 19 recorded | 1124.8 s elapsed | `run-1788629726444274721` |

The three-draft trajectory repeatedly repaired generated Rust tests, including
an empty-response raw-TCP test. It was stopped after exceeding the configured
1,000-second wall budget. Its official verdict is `Fail: cancelled`, not a
completed correctness verdict. The record and trajectory are retained; it was
not rerolled into a pass. No inference error was observed in that run.

Earlier agentic runs without preserve-thinking remain historical compatibility
observations only. `ATLAS_DFLASH_SPEC_THINK=1` is a separate flag controlling the
current thinking span. A preceding-runtime on/off control matched five complete
responses; its distinct binary fingerprint is retained in the evidence.

## Concurrency and memory

Final binary: two drafts, `--max-num-seqs 2 --max-batch-size 2`,
`ATLAS_MTP_MAX_SEQS=2`, 32K configured context, BF16 KV, GPU utilization 0.72,
no prefix cache. Speculation during thinking is on; fusion is on and snapshot
copying is off. This tests concurrent requests through serialized per-sequence
verification, not a batched-forward throughput gain.

Three repetitions of plain/plain and plain/tool pairs produced 12/12 matching
responses against isolated controls. Generated content, reasoning and tool
arguments match; independently assigned tool-call IDs are excluded. A cancelled
stream's companion and a subsequent request also matched. The server logged the
client disconnect and ended that sequence after nine tokens.

Reported server GPU allocation returned to **87,084 MiB** at every idle sample
between pairs. Minimum host MemAvailable was **20.59 GiB**; swap-out delta was
zero (ten pages swapped in). The unrelated ASR service stayed resident. The
private draft KV pool was 128.125 MiB for two slots. The harness stopped new
submissions below 8 GiB available, but that guard did not trigger.

This covers repeated short requests under a 32K configured limit, not two
fully occupied 32K contexts or concurrency above two. An earlier three-draft C2
run before snapshot elision also matched 24 paired outputs across two runs.

## Verification and reproduction

- Final-source model unit suite: 775 passed, 14 ignored with one test thread.
  The preceding parallel run had a transient hanging-ffmpeg fixture startup
  failure; the complete serial rerun passed. Server suite: 2,361 passed,
  12 ignored. CUDA workspace clippy passes with `-Dwarnings`.
- Final default paths: 15/15 short greedy responses (five per width), plus
  9/9 repeated long responses, match their references. Three additional long
  snapshot-control responses match. These are output checks, not a universal
  logit-byte identity claim.
- GPU staging: 48 exact-buffer cases against old wrappers and an independent
  cast/index oracle, including guard bytes. Stable expert-grid parity: all
  18 cases pass with fused and unfused ingress. The old-grid negative control
  produces 30,149 differences. GPU fixtures passed again at the end.
- Existing PR-wide formatting, quantization identifier spelling, and 12 vendor-header failures remain.
  Rustdoc also finds three existing EXL3 broken/private links. Changed-source
  formatting and diff checks pass. No Docker image was rebuilt or published.

[Raw evidence](mtp-performance-results.jsonl) is JSON Lines, one typed record per
artifact: fingerprints, exact harnesses, raw choices/usage, agentic records,
cancelled trajectory, acceptance logs and memory samples. The ingress serving
A/B has its own earlier binary fingerprint; it is not mixed with final timings.

Local TUI: `/home/ms/run-atlas-pr834-tui.sh`, default two drafts and concurrency
one. Select `MTP_DRAFTS=1|2|3`; `MTP_CONCURRENCY=2` enables the tested two-slot
profile. The compiled binary has its library RUNPATHs embedded. Test servers
are stopped after validation so port 8892 is available for the TUI.
