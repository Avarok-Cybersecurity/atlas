# DFlash on the Ada 4B: z-lab/Qwen3.5-4B-DFlash

Lane L14 of the consumer-optimization port sprint. This is a wiring lane: the
DFlash engine (`BlockDiffusionDraftHead`, the K = gamma + 1 verify, the gamma
resolver) already exists and runs on GB10 and Hopper. The lane pairs the
Qwen3.5-4B drafter with the Ada 4B target, checks the pairing at startup, and
sizes the 16 GB card's reserve for it. Opt-in only: nothing changes unless the
operator passes `--dflash`.

Status: code and unit tests written; **nothing here has run on the GPU**. Every
speed, acceptance and memory figure below that is not marked "read from the
checkpoint" is a prediction or a number from the drafter's model card (another
card, another engine). The GPU receipts are listed at the end.

## What the drafter is

`z-lab/Qwen3.5-4B-DFlash` (Z Lab and Modal, Apache-2.0, not bundled). Read from
its `config.json` and safetensors header on 2026-11-12:

| Property | Value |
|---|---|
| Layers | 6 Qwen3 layers: 5 `sliding_attention` (window 4096), the last `full_attention` |
| Hidden / FFN | 2560 / 9216 |
| Heads | 32 query, 8 KV, head_dim 128 |
| Block size | 16 (serves at gamma 15, verify width K = 16, via `default_dflash_gamma`) |
| Target taps | 8 layers: `[1, 5, 9, 13, 17, 21, 25, 29]`, so `fc` is `[2560, 8 x 2560]` |
| Mask token | 248077 (the Qwen3.6 drafters use 248070) |
| Vocabulary | 248320, tied embeddings: it shares the target's embedding and LM head |
| RoPE | `rope_parameters {rope_type: default, rope_theta: 1e7}`, plain RoPE |
| `num_target_layers` | 32 |
| Tensors | 69, all BF16, no `embed_tokens` or `lm_head`; 1,268,851,712 bytes of tensors |

It matches `kernels/ada/holo-3.1-4b` (hidden 2560, 32 layers, vocab 248320).
There is no z-lab Qwen3.5 drafter below 4B (HF listing of `z-lab`, 2026-11-12:
smallest are `Qwen3.5-4B-DFlash` and `Qwen3-4B-DFlash-b16`), so
`holo-3.1-0.8b/MODEL.toml` is unchanged.

Caveat on the target: the drafter was trained on Qwen/Qwen3.5-4B hidden states.
`holo-3.1-4b` also serves Hcompany/Holo-3.1-4B (same `(qwen3_5, 2560)` match),
a fine-tune whose hidden states may be off the drafter's distribution. Output is
exact either way (the target verifies every draft); only acceptance, hence
speed, is at risk. The GPU matrix runs on Qwen3.5-4B first.

## What changed

| File | Change |
|---|---|
| `kernels/ada/holo-3.1-4b/MODEL.toml` | new `[dflash]` block (draft model, gamma 16, window 4096, mask 248077, the 8 taps) |
| `weight_loader/dflash_loader/config.rs` | new fields (`sliding_window`, `use_sliding_window`, `layer_types`, `num_target_layers`, `rope_parameters.rope_theta`); `parse_normalized`; `validate_for_target`; `tap_count`, `layer_is_sliding` |
| `weight_loader/dflash_loader/shape_check.rs` | new: checks all 69 tensor shapes against the config before loading; an error for drafters that state `num_target_layers` (this one), a warning for the older drafters GB10/Hopper serve; `AVAROK_DFLASH_SHAPE_CHECK=0` or `=1` forces warn or error |
| `weight_loader/dflash_loader.rs` | hooks only (three lines) |
| `factory/build.rs` (fragment) | refuses a drafter built for another target (layer range, `num_target_layers`, vocabulary) where the `memory_fit_consumer` row is on (Ada only); every other target logs the finding and builds as before; `AVAROK_DFLASH_SKIP_PAIR_CHECK` downgrades the refusal on Ada too |
| `scheduler/dflash_rung.rs` | `AVAROK_DFLASH_RUNG_MULTI` is honored without write-on-accept (it was silently ignored there); defaults unchanged |
| `scheduler/verify_dflash_step.rs` | stale doc comment corrected (see below) |
| `preflight/spec_headroom.rs` | ada-only priced reserve for `--dflash` on Qwen-shaped targets |

The stale comment: `step_verify_dflash`'s doc listed the SSM commit and the
hidden hand-off as "deferred to Phase 6". They are wired: the code commits with
`commit_accepted_prefix(num_accepted + 1, gamma + 1)` (the in-place index-select
documented at `async_chkpt.rs` `commit_accepted_prefix_dispatch`), and the
per-tap hidden states are captured inside the verify forward.

## Memory on a 16 GB card

Drafter weights are 1.27 GB. By default `AVAROK_DFLASH_DRAFTER_FP8` (on unless
`=0`) adds FP8 mirrors of the dense weights (about half the store, 0.63 GB) and
an FP8 mirror of the LM head. Beside that the head's scratch is wider than the
27B/35B drafters' (FFN 9216 vs 6144, 8 taps), which `factory::build` estimates
at 300 MB.

The preflight (`weights.rs` `load_dflash_drafter`) estimates the whole
footprint before loading and refuses when `free < estimate + headroom`. For the
4B at `--max-seq-len 8192` that estimate is about 3.7 GB. If it refuses on a
tight setting, drop `AVAROK_DFLASH_DRAFTER_FP8` or lower `--max-seq-len`.

The speculative CUDA headroom (`spec_headroom.rs`): `--dflash` used to keep a
flat 4 GiB on every target. On ada only (the `memory_fit_consumer` row) a
Qwen-shaped DFlash serve now reserves `512 MiB base + SSM pool dummy slot +
per-sequence ctx accumulators + 1024 MiB margin`, clamped to the flat 4 GiB.

| Case (4B, `--max-seq-len 8192`) | Reserve |
|---|---|
| `--max-batch-size 1` | 2,777 MiB = 512 + 921 + 320 + 1,024 (1,319 MiB under the flat term) |
| default `--max-batch-size 8` | accumulators alone are 2,560 MiB, the sum passes 4 GiB, clamped to 4,096 MiB (no saving) |
| any non-ada target | 4,294,967,296 bytes, as before (pinned by a unit test) |

The dummy slot is the part `preflight_reserve` has never counted (one base blob
plus one full-width verify slot: 51 + 870 MiB at gamma 16 for the 4B's 48 MiB h
and 3 MiB conv per sequence). The accumulators are
`batch x min(max_seq_len, 16384) x 8 taps x 2560 x 2 B`, allocated lazily per
request. **UNMEASURED**: whether the margin is enough is the first GPU receipt.
`AVAROK_SPEC_HEADROOM_FULL` restores the flat reserve.

### Eight streams on the 16 GB card: `AVAROK_DFLASH_FIT=1` (opt-in, off by default)

The default boot at `--max-batch-size 8` is refused ("only 8.08 GB remains but
13.32 GB is needed", `s4-l14-df-c8.log`). The 13.32 GiB is exactly 6,960 MiB of
SSM pools (8 base blobs of 51 MiB, then 8 verify slots of 819 MiB each: 15 h
snapshots x 48 MiB + 16 conv x 3 MiB + a 51 MiB checkpoint, uniform because every
slot verifies the whole block) + 4,096 MiB of CUDA headroom (clamped from 5,017,
2,560 of it ctx accumulators priced at the worst-case window) + 258 MiB of GDN
prefill + 2,327 MiB of buffer arena. The pools alone are 6.8 GiB of the 8.08 GiB.

`AVAROK_DFLASH_FIT=1` prices the serve by what it allocates and picks a width:

1. the pool width follows `--dflash-gamma` (the preflight used to size the pools
   at the drafter's default whatever the flag said);
2. the accumulators are priced at `AVAROK_DFLASH_CTX_CAP` (the head reads that
   env; the preflight did not), `0` = uncapped;
3. with no `--dflash-gamma`, the widest gamma (down to 3, verify width 4) whose
   pools, headroom, arena, the drafter's weights and post-sizing allocations and
   the 4,096-token KV floor all fit beside the resident weights is chosen and
   pinned as if `--dflash-gamma` had been given (the gamma resolver is then off).
   The boot log prints every term ("DFlash fit: gamma N instead of 15 ...").

At 8 streams with `AVAROK_DFLASH_CTX_CAP=1024 --max-prefill-tokens 1024` (arena
394 MiB) the CPU sizing test (`preflight/dflash_fit_tests.rs`) finds gamma 3: 2,064
pools + 2,114 headroom + 32 GDN + 394 arena + 2,407 drafter + 1,024 KV floor + 256
margin = 8,291 MiB against 8,469 MiB beside the weights (the default gamma 15
needs 13,799). That leaves about 1.2 GiB of real KV after the drafter's
allocations. A narrower verify width lowers the tokens one verify step can take
and nothing else (the verify is exact). If no width fits, the log says what even
gamma 3 needs and the default stays. UNMEASURED on the card until the `dfflash8`
row of `bench/rows.fix3` runs.

## How to run

```
spark serve --model-from-path <model dir>/Qwen3.5-4B --model-name qwen35-4b \
  --kernel-target holo-3.1-4b --dflash --draft-model ~/models/Qwen3.5-4B-DFlash \
  --max-seq-len 8192 --max-num-seqs 1 --max-batch-size 1 --kv-cache-dtype bf16 \
  --lm-head-dtype bf16 --gpu-memory-utilization 0.93
```

Without `--draft-model` the id comes from MODEL.toml and the model resolver
fetches it. `--dflash-gamma N` pins the verify width (pinning also switches the
gamma resolver off, which a record run wants). Unpinned, the resolver runs with
cap K = 16: C = 1 starts wide and drops to K = 5 on prose, C >= 2 uses K = 4
only with write-on-accept (`AVAROK_GDN_WOA=1`, lane L10), else the cap.

## Known limits

- **Sliding window per layer is not applied.** The drafter was trained with
  five windowed layers and a full last layer. The head applies its ctx window
  (`AVAROK_DFLASH_CTX_WINDOW`, default 4096, the same number) uniformly. Equal
  up to 4096 tokens of context; beyond that the last layer sees less than it
  was trained on. Acceptance risk only. `layer_is_sliding` is in the config for
  a later head change.
- **The rung thresholds are the 27B's.** Nothing in `dflash_rung.rs` was
  measured for this drafter or card.
- **Raw-argmax verify.** In a DFlash serve the verify judges on raw argmax
  (`dflash_verify_raw_argmax`, `verify_dflash_step.rs`), so inside verify steps
  the output is greedy whatever temperature the request asked for. This is
  existing behavior on every target, not new; the Holo sampling presets use
  temperature 0.6 to 0.7, so a sampled request gets greedy tokens from
  speculative steps. Worth a separate look.
- **Batched verify** is bounded by the 96-row budget: 6 sequences at K = 16.
- The drafter's `d2t` remap path is still unwired (not needed: vocabularies
  are equal).
- The 4B drafter's hidden states are read from taps 1 to 29. Tap 1 and the
  GDN layers among them (layers 0 to 2 are GDN, 3 is full attention, and so
  on) rely on the existing per-layer capture, which the 27B drafter already
  exercises with `[1, 16, 31, 46, 61]`.

## GPU receipts owed (master plan Q11, lane L14)

The GPU measurement script for the 4B matrix (not part of this repository) runs these. In order:

1. **Boot and shape**: `--dflash` boots on the 4B, the log shows the shape
   check line (69 tensors) and the footprint pre-flight; record the preflight
   reserve line and `atlas_gpu_ledger_*` peak against the new reserve.
2. **Greedy SHA vs plain decode**: `qual.py dump` plain, then `--dflash
   --dflash-gamma 15`; the 24-prompt greedy outputs must be identical (first
   divergence at none). A divergence is a bug in the pairing, not noise.
3. **Speed ABBA vs MTP** on prose, code and structured cells at C = 1:
   `bench5.py` for MTP (`--speculative`) and `--dflash` (resolver, and pinned
   15, 7). Acceptance from the `DFLASH K=gamma verify` log lines.
4. **Concurrency** C = 1, 2, 4, 8 with `conc.py`, then the C >= 2 rung sweep
   `AVAROK_DFLASH_RUNG_MULTI` = 4, 8, 16.
5. **Reserve check** at `--max-batch-size 1` and 8: peak VRAM minus ledger
   must stay under the priced headroom.

Flip rule: none. `--dflash` stays opt-in; the receipts decide whether the docs
recommend it on Ada.
