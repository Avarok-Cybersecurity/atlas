// SPDX-License-Identifier: AGPL-3.0-only

//! Video fidelity: did the frames reach the model, in the right order?
//!
//! The sibling of `vision`, and it exists because a video has one failure mode
//! a still does not — the frames can arrive out of order, or not at all, while
//! every count still adds up.
//!
//! ## Why the reversed clip is the whole benchmark
//!
//! "Describe this video" has a confident answer available from language priors
//! alone. That is not a hypothetical: on 2026-08-14 the splice matched only
//! `<|image_pad|>`, so a video's pad positions received NO encoder rows and
//! kept the raw token embedding — and the model, shown a featureless field,
//! described a gray one, then on a differently-worded prompt invented "a
//! person performing martial arts". Vision-token counts were exact, prompt
//! length was exact, nothing errored.
//!
//! What separates "the frames arrived in order" from "something arrived" is
//! sending the SAME sequence twice, once reversed, and requiring the answer to
//! reverse with it. Same geometry, same prompt, same token count — only the
//! order differs. No amount of language prior gets that right twice.
//!
//! ## The other checks
//!
//! * **Geometry** as a RATIO: a clip of twice the duration must cost twice the
//!   temporal groups. Stated as a ratio because the absolute count depends on
//!   the server's `--video-fps`, which this benchmark does not set and must
//!   not assume — hard-coding the default would fail correct, tuned servers.
//! * **Backend parity**: an MP4 and a byte-different GIF of identical content
//!   must produce identical geometry, exercising the subprocess and in-process
//!   decoders against each other.
//! * **Mixed media**: an image and a video in one request, which is where the
//!   ordering contract between collection, template markers and pad expansion
//!   would show a desync.
//! * **Integrity**: media order, history, and two opposite clips in flight,
//!   which expose request-specific offset and conversation-state defects.
//! * **Concurrency**: C = 1, 2, 4 of the same request, requiring correct
//!   replies and prompt-token geometry equal to the single-stream baseline.
//!
//! ## Skipping, not failing
//!
//! Every container except GIF needs ffmpeg, which is an OPT-IN deployment
//! choice. Legs that need it are SKIPPED on a server without it, exactly as
//! the image benchmark reports UNMEASURED for a picture beyond the encoder's
//! capacity. A run where everything skipped reports INCONCLUSIVE rather than
//! PASS — measuring nothing must never read as green.
//!
//! Registered in `registry.rs` and required on the model targets whose
//! `BENCH.toml` files carry a `video-fidelity` gate entry.

pub mod concurrency;
pub mod driver;
pub mod geometry;
pub mod provision;
pub mod request;
pub mod score;

pub use driver::{DESCRIPTOR, METADATA};
