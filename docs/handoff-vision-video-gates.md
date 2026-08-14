# Handoff — vision & video gates (branch `feat/video-support`, PR #516)

Written 2026-08-14. Everything below is measured on dgx-00 unless it says otherwise.

## State

| | |
|---|---|
| Branch | `feat/video-support`, pushed to `avarok` |
| PR | #516, **draft**, based on `feat/qwen3.8-27b-support` (#513) |
| Base sync | merged `avarok/feat/qwen3.8-27b-support` at 2026-08-14; **0 behind** as of the last push |
| Workspace | 4026 tests pass, clippy 0, fmt clean, typos clean |

Retarget #516 to `main` once #513 lands.

## ★ THE ONE OPEN BUG — modality reordering

**Atlas renders vision markers grouped by modality, not in the order the client sent them.**

Proven directly. A request with `video_url` first and `image_url` second renders:

```
<|image_pad|>   <-- the IMAGE, which the client sent SECOND
<|video_pad|>
```

### Why it matters

The pad runs and the encoder rows agree with *each other*, so nothing errors and the token counts are all correct. What is wrong is that the model is shown the items in a different order than the caller wrote them, so any prompt that refers to "the first" or "the video you sent first" is describing something else. It is the same silent-wrong-answer shape as the rest of this branch.

### How it was found

`video-fidelity`'s `video-before-image` leg fails on `qwen3.6-35b-a3b` (12/13) and passes on `qwen3.6-27b` and `qwen3.8-27b`. That is not a model-capability difference in the way it first looks:

* `mixed-media` (image→video) passes **everywhere**, because that order happens to match what Atlas renders.
* `video-before-image` fails only on the 35B because the stronger models recover from being told "the video came first" while seeing the image first. The 35B (A3B, ~3B active) cannot.

So the defect is present on **all three** targets; only the weakest exposes it.

A confound was removed along the way and is worth keeping: the still used to be `01_square_224`, a saturated colour gradient, so a model reading the wrong item could answer with palette colours from the *image*. It now uses `13_gray_224.jpg`; with a grayscale still, any palette colour in the reply can only have come from the clip. That changed the 35B's answer from `[yellow, green, blue]` (borrowed from the image) to `[]` — a much cleaner signal.

### Cause, and it is ours

Introduced in this branch's video wiring. Two places are ordered by modality rather than by content:

1. `crates/spark-server/src/api/chat/msg_entry.rs` — `collect_message_images` walks `m.content` **twice**: once collecting every `ContentPart::Image`, then again collecting every `ContentPart::Video`. So `all_images` / `all_videos` / `image_pad_counts` are grouped, not interleaved.
2. `crates/spark-server/src/api/chat/template.rs` — `build_json_messages_for` emits `image_count` × `{"type":"image"}` and then `video_count` × `{"type":"video"}`.

They are consistent with each other, which is why nothing breaks — and both are wrong about order.

### Shape of the fix

The pad-count vector, the encoder-item vector and the template markers must all be built in **one interleaved pass** over `m.content`:

* `collect_message_images` → a single loop that pushes to one ordered list of media items, each tagged image-or-video, with `image_pad_counts` growing in that same order.
* `MsgEntry` currently carries `image_count: usize` and `video_count: usize`. Those cannot express order — replace with something like `media: Vec<MediaKind>` (or `Vec<enum {Image, Video}>`) so `build_json_messages_for` can emit markers in sequence.
* `build_msg_entries`' preprocessing loop then has to preprocess in that same interleaved order, since `image_pixels` (the `Vec<VisionItem>` handed to the model) must line up index-for-index with the pad runs.

The ordering contract is already written down in the code comments at both sites — they just describe the wrong order today. Update them with the fix.

**Regression test already exists and will go green on its own:** `video-fidelity`'s `video-before-image` leg on `qwen3.6-35b-a3b`. Run it before and after.

## Gate status

| gate | declared on | result |
|---|---|---|
| `vision-fidelity` | qwen3.6-27b, qwen3.6-35b-a3b, qwen3.8-27b | **PASS** — 14/14 geometry, 3/3 probes, 10/10 integrity, control held |
| `video-fidelity` | qwen3.8-27b (default), qwen3.6-27b | **PASS** — 13/13 legs, 0 skipped |

`qwen3.6-35b-a3b` is deliberately **not** declared for `video-fidelity` — see the bug above. Once it is fixed, add the entry (copy the qwen3.6-27b block in its `BENCH.toml`) and re-measure.

### Outstanding: committed gate-proof records

The owner asked for `.benchmarks/<id>/DATE-SHA.json` records for both gates. **Not yet generated.** Vision is green on all three targets and ready; video is green on two. Produce with:

```
spark benchmark run vision-fidelity --pull-request-gate --hardware gb10 \
  --url http://127.0.0.1:PORT --model NAME
```

(dirty tree is refused; the operator commits the record.)

## Running things

No harness scripts are committed — they lived in the job scratch dir. The essentials:

```bash
# Build (ALWAYS all targets — see memory: atlas-build-all-targets)
export ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL='*' ATLAS_TARGET_QUANT='*' \
  CUTLASS_HOME=/home/ms/cutlass FLASHINFER_HOME=/home/ms/flashinfer \
  LIBRARY_PATH=/home/ms/nccl/build/lib LD_LIBRARY_PATH=/home/ms/nccl/build/lib \
  RUSTFLAGS="-L native=/home/ms/nccl/build/lib"
cargo build --release --bin spark

# Serve a vision target (checkpoints are on the NFS mount, not ~/.cache)
SNAP=$(ls -d /mnt/gx10-hf-hub/models--unsloth--Qwen3.6-27B-NVFP4/snapshots/*/ | head -1)
./target/release/spark serve --model-from-path "$SNAP" --model-name m \
  --kernel-target qwen3.6-27b --port 8897 --max-seq-len 32768 --max-num-seqs 4 \
  --gpu-memory-utilization 0.70 --no-tui --video-allow-ffmpeg --video-fps 2

# Run a gate
./target/release/spark benchmark run vision-fidelity --url http://127.0.0.1:8897 \
  --model m --hardware gb10 --no-save
```

Kernel targets: `qwen3.6-27b`, `qwen3.6-35b-a3b`, `qwen3.8-27b`. Hub dirs:
`models--unsloth--Qwen3.6-27B-NVFP4`, `models--Qwen--Qwen3.6-35B-A3B-FP8`,
`models--unsloth--Qwen3.8-27B-NVFP4`.

**When reading benchmark output, grep for `Warn:` as well as `Info:`.** A filter that
omitted `Warn:` hid a failing leg for two runs and sent me chasing a phantom.

## Known non-issues — do not re-investigate

* **`closure_attestation` fails** on `gb10/qwen3.8-27b/nvfp4` (tree-side vs baked hash). **Pre-existing**, reproduces on `50285deb` with none of this branch's commits. Reported on #513.
* **`tokenizer::tests::qwen_dense_parity` is flaky** under `cargo test --workspace`, ~2 runs in 5, a different test each time; passes in isolation and under `-p spark-server --bin spark`. The diff is JSON whitespace in the rendered tools block — two serialization paths and some shared state choosing between them under parallelism. Unrelated to vision/video.
* **The two API surfaces render different prompts** for the same request (81 vs 109 tokens on qwen3.8) because `chat_template_kwargs` is dropped by the Responses lowering. **This is filed as issue #518** and is a parity gap, not a defect — `reasoning.effort` IS honored on `/v1/responses`. Do not let a benchmark compare token counts *across* surfaces; compare each surface against itself.

## Deferred by owner decision

* **LoC cleanup** — parked until the buildout finishes. Currently over the 500-line cap and not allow-listed: `crates/atlas-plugin/src/benchmarks/video/driver.rs` (~700), `crates/spark-server/src/api/chat/msg_entry.rs` (~520), plus pre-existing `crates/atlas-kernels/src/lib.rs` (501, was 484 upstream — the `min_p` field is part of that growth) and `crates/spark-server/src/tui/bench_state.rs` (504).
* **PR #511 (min_p) overlaps #516** — owner said it is folded in; ignore.
* **Anthropic API surface** — not a concern for now. `/v1/responses` is covered.

## Design notes worth not relearning

* **Assert differences, not absolutes, whenever an unknown envelope is involved.** The video geometry leg uses three durations (1×/2×/4×) because two points *always* fit a line — the implied template overhead absorbs any error, so a 2-point ratio check is vacuous. The thinking and Responses legs use two images for the same reason.
* **A leg must vary one thing.** The Responses leg originally compared surface *and* thinking state at once and reported a phantom vision defect on qwen3.8. It now compares `/v1/responses` against itself.
* **Skipped ≠ passed.** Both benchmarks count `measured` separately from `passed`, and a run where everything skipped reports INCONCLUSIVE. A gate that silently stops measuring is the failure mode being guarded against.
* **EXIF orientation is APPLIED** as of this branch (it was ignored before). The pair `15_exif_rot90_224.jpg` / `16_exif_none_224.jpg` pins it: tagged reads "right", untagged reads "top". `Orientation=6` rotates 90° CW, carrying the stored top edge to the *right* — not the left.
* Video needs **ffmpeg** for everything except GIF; legs that need it SKIP rather than fail. Documented in README, QUICKSTART and docs/DEPLOYMENT.md.

## Issues filed from this work

* **#515** — video support (the implementation plan; largely delivered on this branch)
* **#518** — `chat_template_kwargs` dropped on `/v1/responses`
