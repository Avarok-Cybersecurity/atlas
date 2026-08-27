// SPDX-License-Identifier: AGPL-3.0-only

//! The committed video fixtures, and getting them onto disk.
//!
//! Same shape as the image ladder's provisioning (BFCL's pattern, the house
//! convention): assets are `include_bytes!`d into the binary and written to
//! `~/.atlas/artifacts/video` behind a content-derived stamp, so a
//! regenerated clip re-provisions by itself and nothing has to be
//! hand-versioned.

use anyhow::Result;

use crate::artifacts::{ArtifactStore, Stamp, write_asset_bytes};

/// What a clip is, beyond its bytes.
pub struct Clip {
    pub name: &'static str,
    pub bytes: &'static [u8],
    /// The colors it shows, in order. The whole benchmark rests on this
    /// being checkable by eye and unambiguous in English.
    pub colors: &'static [&'static str],
    /// Seconds of footage. Used only for the RATIO assertion, never to
    /// predict an absolute frame count — the server's sampling rate is its
    /// own business.
    pub seconds: u32,
    /// True for the container that needs a subprocess decoder. Legs carrying
    /// one are skipped, not failed, when the server has no ffmpeg.
    pub needs_ffmpeg: bool,
    pub mime: &'static str,
}

/// Four clips, and each pair of them is an assertion.
///
/// * `01` vs `02` — TEMPORAL ORDER. Identical geometry and prompt, frames
///   reversed. If the answer does not reverse too, the model is not reading
///   the sequence. This pair is what caught the splice defect where video pad
///   tokens received no encoder rows at all: every token count was perfect and
///   the model calmly described a gray field.
/// * `01` vs `03` — BACKEND PARITY. Same content, one through ffmpeg and one
///   through the in-process GIF decoder. Identical geometry or the paths
///   disagree.
/// * `05`, `04`, `01` — GEOMETRY at 1x / 2x / 4x. Three durations, not two:
///   two measurements cannot test proportionality at all, because the implied
///   template overhead absorbs any discrepancy and a line always fits two
///   points. With three, `t4 - t2 == 2 * (t2 - t1)` is a claim that can fail,
///   and it is independent of both the overhead and the tokens-per-group
///   figure — so it holds whatever `--video-fps` the server was started with.
pub const CLIPS: &[Clip] = &[
    Clip {
        name: "01_colors_fwd.mp4",
        bytes: include_bytes!("../../../assets/video/01_colors_fwd.mp4"),
        colors: &["red", "green", "blue", "yellow"],
        seconds: 4,
        needs_ffmpeg: true,
        mime: "video/mp4",
    },
    Clip {
        name: "02_colors_rev.mp4",
        bytes: include_bytes!("../../../assets/video/02_colors_rev.mp4"),
        colors: &["yellow", "blue", "green", "red"],
        seconds: 4,
        needs_ffmpeg: true,
        mime: "video/mp4",
    },
    Clip {
        name: "03_colors_fwd.gif",
        bytes: include_bytes!("../../../assets/video/03_colors_fwd.gif"),
        colors: &["red", "green", "blue", "yellow"],
        seconds: 4,
        needs_ffmpeg: false,
        mime: "image/gif",
    },
    Clip {
        name: "05_colors_unit.mp4",
        bytes: include_bytes!("../../../assets/video/05_colors_unit.mp4"),
        colors: &["red"],
        seconds: 1,
        needs_ffmpeg: true,
        mime: "video/mp4",
    },
    Clip {
        name: "04_colors_half.mp4",
        bytes: include_bytes!("../../../assets/video/04_colors_half.mp4"),
        colors: &["red", "green"],
        seconds: 2,
        needs_ffmpeg: true,
        mime: "video/mp4",
    },
];

pub fn clip(name: &str) -> Option<&'static Clip> {
    CLIPS.iter().find(|c| c.name == name)
}

/// Content-derived, so a regenerated clip re-provisions without anyone
/// remembering to bump a version.
fn stamp_value() -> String {
    crate::benchmarks::content_stamp("video-fixtures-v1", CLIPS.iter().map(|c| (c.name, c.bytes)))
}

pub const PLUGIN_ID: &str = "video";

pub fn provision(store: &ArtifactStore) -> Result<std::path::PathBuf> {
    let dir = store.plugin_dir(PLUGIN_ID)?;
    let stamp = Stamp::new(&dir, ".provisioned", stamp_value());
    if stamp.is_current() {
        return Ok(dir);
    }
    for c in CLIPS {
        write_asset_bytes(&dir, c.name, c.bytes)?;
    }
    // LAST: a stamp written before the writes complete turns a partial
    // directory into a permanent "already done".
    stamp.commit()?;
    Ok(dir)
}

#[cfg(test)]
#[path = "provision_tests.rs"]
mod provision_tests;
