# MTP head v2: frequency-ranked draft vocabulary and device-resident propose chain

Port sprint lane L11 (`port/consumer-opt`). Two rows, both off on every target:
`mtp_vocab_rank` (`id` | `freq`, `AVAROK_MTP_VOCAB_RANK`, `--mtp-vocab-rank`) and
`mtp_propose_graph` (`AVAROK_MTP_PROPOSE_GRAPH`).

## What it does

**Frequency-ranked draft vocabulary.** The draft-only NVFP4 head the MTP proposer
uses on Ada (main head BF16 or FP8) holds `--mtp-vocab` rows, by default the first
100,000 token ids. A draft can never propose a token outside its head, and one miss
rejects that position and every later draft of the chain. Under `freq` the head
holds the 100,000 most frequent tokens instead, with the tokenizer's 26 added
tokens forced in (the first-N head holds none of them, so it can never draft
`<|im_end|>`, `<think>`, `</think>` or a tool-call tag). The drafter's argmax then
returns a head ROW, which `kernels/ada/common/draft_vocab_remap.cu` maps to the
token id on the device before `embed_from_argmax` and before any readback. Grammar
rows translate the token bitmask into row space on the host. Every draft is
verified by the main head, so outputs equal the id head's.

Held-out coverage of committed tokens (`crates/spark-model/data/mtp_vocab_rank_qwen35.json`,
computed by `gen_mtp_vocab_rank.py` from Cinference's held-out counts):

| Rows | First-N ids | Frequency-ranked |
|---:|---:|---:|
| 32,768 | 69.0% | 91.0% |
| 65,536 | 73.0% | 96.0% |
| 100,000 (default) | 86.2% | 98.1% |

**Device-resident propose chain.** Per-sequence `propose` read every intermediate
draft with a blocking 4-byte copy so the next position could copy its embedding
row from a host id; the batched propose read all n ids after every position.
Under `mtp_propose_graph` the next position gathers its embeddings on the device
from the previous position's ids (`batched_embed`, the same rows), the ids of all
positions collect in a device buffer, and the round reads them back once. Same
kernels, same bytes: drafts are bit-identical.

**Bounded MTP attention window** (`MtpHead::set_kv_window`, for L36). With a window
the drafter attends its sink block plus the blocks of the last W rows, through a
block-table view, so each key keeps its RoPE position. Inert by default.

## Sources and license

| Item | Upstream | Kind |
|---|---|---|
| Ranking data | Cinference `tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64`, commit 383e5db (sha256 `c692dc76388132c910547589b4fb4a0503fbd6ad50aaac6a509bbcb192a8afa5`) | data copy (top 131,072 ids of row 0) |
| Selection | Cinference `tools/convert/proposal.py:21-42` (`select_shortlist`) | translation (Python stdlib, Rust) |
| Row gather | Cinference `tools/convert/proposal.py:44-104` (`add_proposal`) | translation (device gather into NVFP4) |
| Remap | Cinference `src/ops/kernel/speculative_round.cuh:956-963` | translation (CUDA) |
| Graphed round | Cinference `src/models/qwen3_5/program/speculative/mtp.cpp:94-201` | design only |

Cinference is derived from NInfer by Neroued and its contributors
(https://github.com/Neroued/ninfer) and is distributed under the Apache License,
Version 2.0 (`LICENSES/Apache-2.0.txt`). The ranking data are token counts by
NInfer authors; the source corpus is not published. The NOTICE entry reproduces
the first paragraph of Cinference's NOTICE.

Source files that carry derived code, each with the attribution header under its
SPDX line: `crates/spark-model/data/gen_mtp_vocab_rank.py`,
`crates/spark-model/src/layers/mtp_head/vocab_rank.rs`, `ranked_head.rs`,
`propose_chain.rs` (design credit), `kernels/ada/common/draft_vocab_remap.cu`. The
binary table `crates/spark-model/data/mtp_vocab_rank_qwen35.u32` carries its
provenance (source, commit, file hash, license, the data note) in
`mtp_vocab_rank_qwen35.json` beside it.

## Tokenizer identity check

The counts were made by teacher-forcing Qwen3.6-27B (its tokenizer). Before the
table is used, two checks hold:

1. Generator, once: every (id, token) pair the three Cinference manifests print
   (128 each) decodes to the same text through the Qwen3.5-4B and Qwen3.5-0.8B
   `tokenizer.json` (768 pairs, 0 mismatches), and the two files are identical
   (sha256 `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`,
   FNV-1a-64 `349dd3b67c2adef5`).
2. Runtime, per model: the served `tokenizer.json` must hash (FNV-1a-64) to a
   verified file and the vocabulary must be 248,070 (tokenizer-capped) or 248,320.
   Otherwise one warning and the id head.

## Changes for Atlas

Selection extended to any head width (the table completed in id order); forced ids
kept at every width; ranked rows quantized with the full head's global scale so a
ranked row is byte-identical to that token's row of the full quantization; the
gather done in 4,096-row chunks on the device; the remap writes in place and can
copy into the propose chain's buffer; a recorder (`AVAROK_MTP_VOCAB_RANK_RECORD`)
produces Atlas's own ranking from committed tokens to replace the v1 table.

## Receipts still owed (GPU)

`mtp_vocab_rank=freq`: acceptance per category up on the L03 corpus
(`atlas_mtp_drafts_accepted_total`, `atlas_mtp_draft_out_of_head_total`) and the
logprob KL inside the screen's bar. Greedy token SHA equal vs `id` is NOT the
receipt: L11 measured 7 of 8 equal at C=1, a verify-width numerics effect at a
bf16 near-tie (`layers/mtp_head/vocab_rank.rs`).
`mtp_propose_graph`: greedy SHA equal at C=1 and C=8; propose phase time down.
