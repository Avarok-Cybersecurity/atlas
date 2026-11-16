// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the per-sequence state of one LFM3 conv layer
//! (`Lfm3ConvState`, a `[hidden, 6]` BF16 window on the device) and the host
//! blob that carries it through a Marconi prefix-cache snapshot.
//!
//! Blob format (little endian), version 1:
//!
//! ```text
//! 0   8   magic  b"LFM3CV01"
//! 8   4   hidden (u32)
//! 12  4   window length (u32, always 6)
//! 16  ..  hidden * 6 BF16 values, channel-major, oldest first
//! ```
//!
//! A blob from another geometry is refused, never truncated: restoring a
//! window of the wrong width would silently serve a different prefix.

use std::any::Any;

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::STATE_LEN;
use crate::layer::LayerState;

/// Magic and version of the snapshot blob.
pub const BLOB_MAGIC: &[u8; 8] = b"LFM3CV01";
/// Bytes before the payload.
pub const BLOB_HEADER: usize = 16;

/// One sequence's window for one conv layer. Owned by the layer
/// (`alloc_state` / `release_state`), never pool memory.
pub struct Lfm3ConvState {
    /// `[hidden, 6]` BF16 on the device; NULL after release.
    pub conv: DevicePtr,
    /// Bytes behind `conv`.
    pub bytes: usize,
}

impl LayerState for Lfm3ConvState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Serialize a window read back from the device.
pub fn encode_blob(hidden: usize, payload: &[u8]) -> Result<Vec<u8>> {
    ensure!(
        payload.len() == hidden * STATE_LEN * 2,
        "LFM3 conv snapshot: payload of {} bytes does not match hidden {hidden}",
        payload.len()
    );
    let mut out = Vec::with_capacity(BLOB_HEADER + payload.len());
    out.extend_from_slice(BLOB_MAGIC);
    out.extend_from_slice(&(hidden as u32).to_le_bytes());
    out.extend_from_slice(&(STATE_LEN as u32).to_le_bytes());
    out.extend_from_slice(payload);
    Ok(out)
}

/// The payload of a blob written by [`encode_blob`] for the same geometry.
pub fn decode_blob(hidden: usize, blob: &[u8]) -> Result<&[u8]> {
    ensure!(
        blob.len() >= BLOB_HEADER && &blob[..8] == BLOB_MAGIC,
        "LFM3 conv snapshot: not an LFM3 conv window blob ({} bytes)",
        blob.len()
    );
    let word = |at: usize| u32::from_le_bytes([blob[at], blob[at + 1], blob[at + 2], blob[at + 3]]);
    let (h, w) = (word(8) as usize, word(12) as usize);
    ensure!(
        h == hidden && w == STATE_LEN,
        "LFM3 conv snapshot: blob is for hidden {h} window {w}, this layer is hidden {hidden} \
         window {STATE_LEN}"
    );
    let payload = &blob[BLOB_HEADER..];
    ensure!(
        payload.len() == hidden * STATE_LEN * 2,
        "LFM3 conv snapshot: payload of {} bytes, expected {}",
        payload.len(),
        hidden * STATE_LEN * 2
    );
    Ok(payload)
}
