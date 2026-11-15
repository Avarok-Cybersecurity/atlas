# Qwen3.8-27B on a 16 GB Ada card (RTX 4060 Ti)

Status: **written, not measured.** Everything below is code and arithmetic from this port (branch
`port/consumer-opt`, lane L24). No 27B weight has been loaded on Ada yet; every number marked "inferred"
comes from a study of the RTX 5060 Ti recipe and must be replaced by the GPU pass receipts
listed at the end. Nothing here is on by default.

## What the target is

`kernels/ada/qwen3.8-27b/MODEL.toml` declares the checkpoint on Ada. It copies gb10's `qwen3.8-27b`
declaration table for table (architecture facts, sampling presets, behavior, no DFlash section;
`crates/avarok-kernels/tests/ada_27b.rs` pins the equality), with one difference: it compiles the kernel
set of an existing Ada directory, `kernel_source = "holo-3.1-4b"`, instead of gb10's `qwen3.6-27b` tree.

Why not the gb10 tree: there is no `kernels/ada/qwen3.6-27b`, a symlinked copy would add cross-hardware
symlinks (rule S1 of `scripts/check_cross_hardware.py`), and that tree carries `w4a4_gemm.cu` and
`nvfp4_mmq.cu`, which need the FP4 MMA sm_89 lacks. `holo-3.1-4b` is the same architecture family
(qwen3_5 dense: hybrid GDN plus full attention, MRoPE), so every lookup the dense path issues for the 27B
is one the 4B target already makes. The 27B-only kernels are real files in `kernels/ada/common/`: the
keep-packed GGUF IQ4_XS, Q4_K and Q6_K GEMVs and MMQ prefill (lane L19), and the affine and EXL3 families
(lanes L17a, L17c, L20b).

The `[expected_absent]` tables are holo-3.1-4b's, carried because the compiled set is the same. They are
NOT harvested for this directory; row R0 below replaces them.

Resolution: on Ada the 27B is the only declarer of `(qwen3_5, hidden 5120)`, so a GGUF file or an
anonymous directory resolves to it, and the two Holo targets resolve exactly as before (`ada_27b.rs`).

## Why it does not fit without changes

From that study (inferred; 15.29 GiB usable with a desktop session):

| Item | GiB |
|---|---:|
| Weights on device: UD-IQ4_XS layers, Q6_K head, MTP head | 12.50 |
| Embedding table if kept on the device (BF16 `[248320, 5120]`) | 2.37 |
| CUDA context, cuBLASLt, graphs | 0.45 |
| Prefill chunk 2,048, logits, scratch | 0.60 |
| GDN state, one stream (48 layers x 48 heads x 128 x 128 x FP32) | 0.14 |
| MTP verify planes, 3 drafts | 0.28 (FP16 h) to 0.6 (FP32 h) |
| KV resident, the remainder | about 1.3 |

Three things in that table are what the sprint changes: the embedding table leaves the device, the head
stays packed, and the verify pools are sized to what is left.

## The three mechanisms

### 1. Embedding table in mapped host memory (row `embed_host_offload`, `AVAROK_EMBED_HOST_OFFLOAD`)

`crates/spark-model/src/port/embed_host/`. The table is copied once into page-locked host memory and its
device-visible alias becomes the model's `embed_tokens` pointer; the device copy is freed right after the
loader's prune step. Every gather keeps working through the alias, including the MTP head's
`embed_from_argmax`, which embeds token ids that exist only on the device (a host-side gather before a
graph launch could not serve them). The central single-row embed (`model/impl_a3_embed.rs`) copies the row
from the host table directly. Same bytes either way, so the row is bit-identical by construction; it frees
2.37 GiB on the 27B.

It declines (and the table stays on the device) when the head is tied to the embedding, the tensor is not
a BF16 `[rows, hidden]` table, or the backend refuses mapped memory. WSL2 may refuse a 2.4 GiB pinned
region; a refusal is counted in `atlas_embed_host_refusals_total`, and a `--speculative` serve
then refuses to start, because its sizing assumed the freed VRAM. Metrics: `atlas_embed_host_resident_bytes`,
`atlas_embed_host_device_bytes_freed_total`, `atlas_embed_host_row_copies_total`.

### 2. Packed LM head (`--lm-head-packed`)

`crates/spark-model/src/port/packed_head.rs`, dispatched from `model/lm_head_q6k.rs`. The file's Q6_K head
(1.04 GB instead of 2.54 GB in BF16, read once per token) stays as raw super-blocks and is served by lane
L19's Ada `Q6_K x q8_1` GEMV through `W4Linear`, in chunks of 8 rows. The pre-sprint packed head (DeepSeek
V4.1 on GB10) resolves `kquant_moe`, which only the deepseek model directory compiles; it is still used
wherever that module exists, so GB10 is unchanged.

The head stays packed only with `--lm-head-packed` on the keep-packed GGUF route (row `gguf_native_iq4`).
Every other reader of the head pointer expects BF16, so the factory refuses `--lm-head-packed` together
with `--speculative` (the MTP drafter quantizes its draft head from the BF16 head) or with any
`--lm-head-dtype` other than `bf16`. A head that is not Q6_K in the file is served as before, with a
warning.

### 3. Verify pools sized to the card (`memory_fit_consumer`, ada only)

`crates/spark-model/src/ssm_reserve/consumer_fit.rs` and
`crates/spark-server/src/main_modules/serve_phases/preflight/consumer_pools.rs`. One verify slot at
`--num-drafts 3` with FP32 h-state is about 613 MiB on the 27B (three h snapshots, four conv snapshots,
one checkpoint blob). Preflight computes what is left beside the resident weights (the checkpoint minus
what the FP8 FFN route and the embedding offload free), the base state of every stream, the KV floor and
the CUDA headroom, gives the verify pools at most half of it, and publishes the largest slot count that
fits, never below one. The pools, the reserve and the scheduler's spec-dispatch guard all read it through
`ssm_reserve::mtp_state_slots`, so uncovered slots plain-decode. Worked numbers at 15,660 MiB free, FP8
KV, 8,192 tokens and the embedding offloaded (pinned by `consumer_fit_tests.rs`):

| Batch | Budget for verify pools | Slots covered |
|---:|---:|---:|
| 1 | n/a (one slot requested) | 1 |
| 2 | 1.38 GiB | 2 |
| 4 | 1.11 GiB | 1 |
| 8 | 0.56 GiB | 1 |

The 4B on the same card keeps every slot at the default batch, so its measured serves are unchanged; on
GB10, Hopper, B200 and B300 the fit never runs. `AVAROK_MTP_CONSUMER_FIT_OFF` skips it,
`AVAROK_MTP_POOL_FULL_WIDTH` ignores it with the rest of the pool diet.

### Measured ceilings (`preflight/measured_ceiling.rs`)

A table of receipts: per card (total memory), model, weight format, KV dtype and speculation mode, the
longest `--max-seq-len` that booted AND survived a full-length prefill at a stated batch. A measurement
only lowers the answer: a serve above a matching row refuses at preflight (or warns with
`AVAROK_MEASURED_CEILING_OFF`), and the memory formula still runs after it. The table ships empty, since
no context ceiling has been measured on Ada; the GPU pass adds rows.

## FP16 h-state

The 77 budget assumes `--ssm-h-dtype f16` (K at most 4). The FP16 decode twins live in gb10's
`qwen3.6-27b` tree and are not in this target's compiled set, and the FP16 arms dispatch those handles
without probing them. Do not pass `--ssm-h-dtype f16` on this target until an Ada file ships the twins;
serve and size with FP32 state (the fit above already does).

## How to serve it (once the GGUF is downloaded)

```
AVAROK_GGUF_NATIVE_IQ4=1 AVAROK_EMBED_HOST_OFFLOAD=1 \
spark serve --model-from-path ~/models/Qwen3.8-27B-GGUF --model-name qwen38-27b \
  --kernel-target qwen3.8-27b --max-seq-len 8192 --max-batch-size 1 --max-num-seqs 1 \
  --kv-cache-dtype fp8 --lm-head-dtype bf16 --lm-head-packed --gpu-memory-utilization 0.93
```

With MTP, drop `--lm-head-packed` (the head then expands to BF16, about 1.5 GB more) and add
`--speculative --num-drafts 3`.

## What is unmeasured, and the receipt for each

Run by the GPU measurement script (not part of this repository) for the 4B and 27B rows (the 27B rows after the
download), against these criteria:

| Row | What | Pass criterion |
|---|---|---|
| Embed | 4B, `AVAROK_EMBED_HOST_OFFLOAD=0` vs `=1` | greedy token SHA equal solo and C=8; VRAM freed in the boot ledger, or the decline logged for a tied head |
| R0 | `--check-kernels` on `qwen3.8-27b` | resolves, no hard error; harvested `[expected_absent]` replaces the carried tables |
| R1 | load and residency | weights about 12.5 GiB, peak at most 15.3 GiB, no CPU fallback |
| R2, R3 | dequant oracle, GEMV bandwidth | lane L19's rows |
| R4 | no-spec decode, 5-run medians | at least 15 tok/s |
| R5 | KL vs a BF16/FP8 oracle | at most 0.1 (oracle on a larger machine) |
| R6 | MTP on the code replay | 28 to 35 tok/s at 3 drafts |
| R7 | KV capacity and the measured ceiling | at least 40K FP8 tokens; the longest surviving `--max-seq-len` becomes a `MEASURED` row |
| R10, R11 | aggregate and memory per stream | C=8 at least 100 tok/s without MTP; covered slots match the fit |
| R12 | prefill | TTFT at 16K and 64K, reported |

Rows R8 and R9 (selective attention and the host KV tier) belong to lanes L26 to L36.
