# Extended Spark readiness rehearsal — 2026-09-19

These are development receipts for draft PR #1150, not benchmark certification.
The initial rental path is raw text/token-array inference and kernel bring-up.
Official XTML chat/reasoning/tool demultiplexing remains unimplemented; see the
[protocol boundary and prepared-prompt commands](../../PROTOCOL.md).

## Findings fixed before rental

1. A worker's ordinary command wait inherited a 30-second collective deadline.
   After an otherwise successful pilot, 30 idle seconds poisoned the worker's
   communicator and the next request timed out. Only the first command-word
   receive now permits idle time; it still polls NCCL errors. Subsequent command
   words and payloads retain their deadline. External supervision remains
   responsible for an idle peer that disappears without a reported NCCL error.
2. Blocking completions trimmed whitespace through the generic reasoning parser,
   while streaming preserved it. Unmarked completion text now remains byte-exact.
   The aviation case reproduced the failure before the fix. The new baseline
   preserves decoded whitespace; all other text, token counts and finish reasons
   were checked against the earlier receipt. Soak comparisons use exact bytes,
   with no stripping or whitespace tolerance.
3. The official checkpoint has no `tokenizer.json`. The derived tokenizer passed
   2,265 differential cases through Atlas's actual tokenizer loader, all 163,584
   base-token decodes, and checks of model EOS versus tokenizer-named EOS. See
   [tokenizer evidence](../official-tokenizer-20260919/README.md).
4. Generic ChatML/Qwen processing is incompatible with official XTML. Chat now
   refuses that unsupported contract, and raw official completions preserve
   literal `<think>` data. Independent official encoder fixtures and prepared
   integer-array requests support numerical bring-up without pretending native
   chat/tool support exists.
5. A serving directory made from ordinary shard symlinks would violate the
   loader's containment checks. The [staging tool](../../../../scripts/k3/TOKENIZER.md)
   verifies the snapshot, hardlinks weights on the same filesystem and copies
   metadata/derived assets. Real fixture staging preserved original hashes and
   kept all canonical loader paths inside the serving root. No second full
   weight copy is made; originals and hardlinks must remain read-only.

## What passed

- Native packed twin, TP2, EP1, protocol v2, two sequence slots, prefill budget 32,
  BF16 KV, max sequence 512, cache reuse off: **436 request checks over 708.93 seconds**.
  Twenty cycles include **20 streaming comparisons, 20 cancellations and 40
  overlapping-client completions**, alongside deterministic repetitions and
  post-cancel canaries. No output/finish/count mismatch occurred.
- A **45-second request-free idle gap** precedes generation successfully.
- Server logs confirm cancellation recovery: cycle 0 closed after the first
  content event; the server logged receiver drop and completion after 2 tokens
  of a 128-token budget. The next canary matched. Concurrent request lifetimes
  overlapped before either finished; this is not proof of fused GPU batching.
- **476,453 collective submissions per rank match**, including shutdown. The
  completed requests provide separate progress evidence; matching log prefixes
  alone cannot prove completion or detect equal truncation.
- Five-second process-RSS samples: in the last two thirds, head RSS ranged
  **1,691,396–1,692,300 KiB** (904 KiB span); worker RSS was **1,668,416 KiB**.
  This is a bounded observation, not a long-term leak proof or GPU allocator
  audit. Unified-memory device telemetry is not substituted with zero.
- A clean rebuilt container passes TP1 completion/stream byte parity and
  cancellation recovery. A synthetic identity overlay retains the twin's
  weights/tokenizer while selecting the official contract: all four
  stream/tools chat combinations return HTTP 400; raw completions still work.
  This validates routing/refusal, not official-model chat semantics.
- The identical final image also passes TP2 with protocol v1 and a 45-second
  idle gap before its canary. Container transport is NCCL Socket, not RDMA or
  NVSwitch; both ranks exit zero and leave both GPUs idle. See
  `container-tp2-idle.json`. This complements native protocol-v2 idle coverage.
- Latest workspace checks: **7,049 Rust tests passed, 120 ignored; 68 Python tests
  passed**. Workspace clippy, rustdoc, formatting, typo and file-size checks
  passed. The previously reported 14 unchanged Hopper vendor license-header
  findings remain; new source files carry the required SPDX header.

`soak.json`, `baseline-v2.json` and `container-api.json` contain compact results.
Full per-request logs and process samples are retained outside Git. Earlier
failed pilots are retained there too; they are not counted as passing checks.

Native soak used binary `14f1f49eff22db38281a2ea1f98883ec510d87e7df016d808251a1eacbc3a91d`.
The later official-XTML literal-output fix does not affect that twin path; the
final container includes it. Both native GB10 and B300 release builds pass;
these are ARM64 host executables, and B300 execution remains unverified.
The final image identity is recorded in `container-api.json`; source labels
are disclosed development snapshots, not clean certification stamps.

## Repeat and extend on the rental

Use [the soak harness](../../../../scripts/k3/SOAK.md), selecting a trusted
baseline for the actual checkpoint. The packed twin's updated baseline is
`baseline-v2.json` here. Keep the official tokenizer, prepared prompts, source
pin, binary/image identity, ranks, topology and memory measurements with each
run. Start with C=1/eager execution, then repeat the boundaries and controlled
C=2 lifecycle checks on TP8. Actual SM103 execution, NVSwitch behavior,
full-checkpoint peaks, semantic quality and useful throughput remain node tests.
