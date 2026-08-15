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

## ✔ FIXED 2026-08-15 — modality reordering

**Atlas rendered vision markers grouped by modality, not in the order the client sent them.** A request with `video_url` first and `image_url` second rendered `<|image_pad|>` before `<|video_pad|>`.

The pad runs and the encoder rows agreed with *each other*, so nothing errored and every token count was right. What was wrong is that the model was shown the items in a different order than the caller wrote them, so any prompt referring to "the first" or "the video you sent first" described something else — the same silent-wrong-answer shape as the rest of this branch.

### Where it actually lived

The handoff named two sites; there was a **third, upstream of both**, and it was the one that made the other two unfixable on their own:

1. `crates/spark-server/src/openai/chat_message.rs` — `ParsedContent` held `images: Vec<String>` and `videos: Vec<String>`. **Order was already destroyed at the wire parse**, before any IR existed, on both the chat-completions and Responses parsers.
2. `crates/spark-server/src/api/chat/msg_entry.rs` — `collect_message_images` walked `m.content` twice, once per modality.
3. `crates/spark-server/src/api/chat/template.rs` — `build_json_messages_for` emitted `image_count` image markers then `video_count` video markers.

All three were consistent with each other, which is exactly why nothing broke.

### What changed

Media is now **one tagged, ordered sequence** from the wire to the rendered prompt:

* `ParsedContent { text, media: Vec<MediaRef> }`, where `MediaRef { kind: MediaKind, uri: String }`. Both parsers append to that one list in arrival order. `images()` / `has_images()` serve the stored-conversation writers, which replay images only.
* `ir::MediaKind` is the single modality tag, shared by the wire type and `MsgEntry` so the tag and the order cannot drift apart.
* `Message::media_kinds()` replaces `image_count()` / `video_count()` — a count pair cannot express order, which is what made the defect expressible.
* `MsgEntry.media: Vec<MediaKind>` drives the markers; `collect_message_media` does one pass over `m.content` building the encoder inputs and pad counts together; the preprocessing loop dispatches per item **in that same order** instead of running images and then videos.
* URI resolution (base64 / remote-URL policy) is now one helper for both modalities rather than two copies.

No Jinja change was needed: the bundled template already walks the content array in order and numbers `Picture N:` / `Video N:` from running counters. The model side was already order-driven too — `mrope_pos` consumes `grids[item]` in pad-run order with `t_len` distinguishing still from clip, so interleaving needed nothing there.

**Verification.** `video-fidelity` on `qwen3.6-35b-a3b`: **13/13, 0 skipped, control held** (was 12/13). `video-before-image` reports 288 prompt tokens — the same 288 the dense targets record, so the ordering moved and the geometry did not. `vision-fidelity` on the same server still 14/14 geometry, 3/3 probes, all integrity legs. Plus 10 unit tests pinning the order at each hop (wire, IR, collection, markers) — `cargo test -p spark-server --bin spark`, 2220 pass.

## Gate status

| gate | declared on | result |
|---|---|---|
| `vision-fidelity` | qwen3.6-27b, qwen3.6-35b-a3b, qwen3.8-27b | **PASS** — 14/14 geometry, 3/3 probes, 10/10 integrity, control held |
| `video-fidelity` | qwen3.8-27b (default), qwen3.6-27b, qwen3.6-35b-a3b | **PASS** — 13/13 legs, 0 skipped |

All three vision targets now declare both gates.

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

* **LoC cleanup** — parked until the buildout finishes. Currently over the 500-line cap and not allow-listed: `crates/atlas-plugin/src/benchmarks/video/driver.rs` (917), `crates/spark-server/src/api/chat/msg_entry.rs` (578 — down 19 from the ordering fix, which folded two copies of the URL-policy branch into one helper), plus pre-existing `crates/atlas-kernels/src/lib.rs` (501, was 484 upstream — the `min_p` field is part of that growth) and `crates/spark-server/src/tui/bench_state.rs` (504).
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
