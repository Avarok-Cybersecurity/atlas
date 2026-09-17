// SPDX-License-Identifier: AGPL-3.0-only

//! Capability probes: did the model actually see the picture?
//!
//! Scored on substring presence over deliberately unambiguous subjects, so a
//! failure cannot be argued away as a wording difference. The fixtures are the
//! synthetic ladder from `scripts/gen_test_images.py` — each image carries
//! distinct geometry and a size label, which is what makes them describable
//! without shipping a copyrighted photo.
//!
//! These are the WEAKER leg and are meant to be. A capability probe answers
//! "did vision run at all"; `geometry.rs` answers "did it run correctly". Both
//! are needed: geometry alone would pass on an encoder producing the right
//! token count from garbage embeddings.

/// A probe: which fixtures to attach, what to ask, what a correct answer
/// must and must not contain.
pub struct Probe {
    pub id: &'static str,
    /// Fixture filenames, resolved against the provisioned artifact dir.
    /// Empty means "send no image" — see [`CONTROL`].
    pub images: &'static [&'static str],
    pub prompt: &'static str,
    /// Lowercased; ALL must appear.
    pub want_all: &'static [&'static str],
    /// Lowercased; NONE may appear. Stops a reply that names everything from
    /// scoring as a pass.
    pub want_none: &'static [&'static str],
}

pub const PROBES: &[Probe] = &[
    Probe {
        id: "sees-an-image",
        images: &["01_square_224.png"],
        // The generator draws a size label into each fixture, so the smallest
        // rung is legible only if the encoder preserved fine detail at the
        // bottom of the ladder.
        prompt: "Describe what you see in this image in one short sentence.",
        want_all: &[],
        // Nothing positive to assert without pinning the generator's exact
        // artwork; the assertion is that a reply arrives and is not a refusal.
        want_none: &["cannot see", "no image", "unable to see", "don't see"],
    },
    Probe {
        id: "reads-the-size-label",
        images: &["07_hd_1280x720.png"],
        // 1280x720 is the rung that sat exactly on the old long-side clamp, so
        // this probe is also the one most likely to change behaviour when the
        // area bound moves.
        prompt: "This image has a size label drawn on it. Read the label exactly.",
        want_all: &["1280"],
        want_none: &["cannot see", "no image"],
    },
    Probe {
        id: "multi-image-order",
        images: &["01_square_224.png", "08_portrait_480x854.png"],
        // Order is the assertion: a splice that concatenates embeddings in the
        // wrong order still describes both images as a SET.
        prompt: "You are shown two images. Is the FIRST one square or portrait? \
                 Answer with one word.",
        want_all: &["square"],
        want_none: &["portrait"],
    },
];

/// The non-vacuity control: the same question with NO image attached.
///
/// A server that has silently stopped splicing vision embeddings will still
/// answer the probes above from language priors — "describe this image" invites
/// a confident description of nothing. If this control produces an
/// image-shaped answer rather than a refusal, the capability leg proved
/// nothing and the run reports VACUOUS instead of PASS.
pub const CONTROL: Probe = Probe {
    id: "control-no-image",
    images: &[],
    prompt: "This image has a size label drawn on it. Read the label exactly.",
    // A correct server has nothing to read; saying so is the pass.
    want_all: &[],
    // Seeing the label from `reads-the-size-label` with no image attached
    // means that probe is not evidence of anything.
    want_none: &["1280"],
};

/// The image-specific probe reused by the concurrency leg. Keeping this tied
/// to a normal capability probe prevents the load test from substituting a
/// stimulus such as "reply OK" that can pass without observing the image.
pub fn concurrency_probe() -> &'static Probe {
    PROBES
        .iter()
        .find(|probe| probe.id == "reads-the-size-label")
        .expect("reads-the-size-label probe is part of the fixed probe set")
}

#[cfg(test)]
#[path = "probes_tests.rs"]
mod probes_tests;
