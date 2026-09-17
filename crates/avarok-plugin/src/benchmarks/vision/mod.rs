// SPDX-License-Identifier: AGPL-3.0-only

//! Vision fidelity: does the served model see the image it was sent, at the
//! resolution its checkpoint permits?
//!
//! Two legs, and the split is the design.
//!
//! **Geometry** asserts an exact vision-token count from `usage.prompt_tokens`
//! across a resolution/aspect ladder. Deterministic, needs no judgement about
//! model output, and catches the class a capability probe cannot: preprocessing
//! that still produces a coherent answer from the wrong number of patches. The
//! 2026-08-14 resolution-cap defect was exactly that — everything above 1280px
//! on the long side was silently reduced to about a tenth of its permitted
//! area, and every capability check still passed, because a downscaled picture
//! is still recognisably red.
//!
//! **Capability** asserts the model actually saw the picture, on deliberately
//! unambiguous subjects, with a no-image control so a pass cannot come from
//! language priors alone.
//!
//! ## Relationship to the existing harnesses
//!
//! `tests/vision_sweep.py` grades Mona Lisa recognition on a PASS/PARTIAL/FAIL
//! keyword rubric across models, and `tests/vit_reference_check.py` diffs
//! Atlas's ViT layer-by-layer against an HF reference to localise a first
//! divergence. Both remain; neither asserts token geometry, and neither would
//! have caught the resolution cap. This is the third thing — a registered
//! benchmark, so it can become a gate — and it reuses their fixture ladder
//! rather than inventing one.
//!
//! Registered in `registry.rs` and deliberately NOT added to the required set
//! in `gate/required_tests.rs`: runnable and gate-able, gating nothing yet.

pub mod driver;
pub mod geometry;
pub mod probes;
pub mod provision;
pub mod request;
pub mod score;

pub use driver::{DESCRIPTOR, METADATA};
