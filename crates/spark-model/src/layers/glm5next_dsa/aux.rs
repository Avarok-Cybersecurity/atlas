// SPDX-License-Identifier: AGPL-3.0-only

//! The DSA indexer cache as a Marconi aux blob — what a prefix-cache hit must put back.
//!
//! # Why the indexer needs a blob at all
//!
//! A GLM-5.3 sequence carries three kinds of per-sequence state: paged MLA latents (KV blocks,
//! carried by the radix prefix cache), the KDA recurrent `h`/conv state (carried by the Marconi
//! SSM snapshot slot, exactly as qwen4exp's 36 GDN layers are), and THIS — `[len, D]` rows of
//! LayerNorm'd indexer keys and compress-gate projections per DSA layer, which the selector
//! scores every step. Neither is recoverable from the MLA latent (`wk · hidden` cannot be
//! inverted out of a rank-512 compression), so a restore that brings back KV + KDA and leaves
//! the indexer at `len = 0` resumes with rows MISSING behind a KV image that assumes them
//! present. `decode_k`'s lockstep check turns that into a hard bail; without the check it would
//! select over an empty context — a silently wrong answer. This blob is what closes the gap.
//!
//! # Blob kind: image, not marks
//!
//! qwen4exp's two aux carries are the two possible kinds. QSA's `ingested`/`pooled` are
//! contiguous MARKS over buffers written forward from them, so a rewind is exact and the keys
//! themselves still ride the blob. PLE's rolling conv + n-gram history have DISCARDED their
//! oldest entries and must be imaged. The DSA cache is written forward from `len` and never
//! discards, so its counter rewinds exactly (like QSA) — but the rows still have to travel,
//! because a restored sequence lands in a FRESH `Glm5NextDsaState` (`alloc_sequence` allocates
//! it at `len = 0`; a swap-in re-allocates it the same way), and the rows are a function of the
//! hidden stream, which the prefix cache does not keep. So: a 1:1 image of `[0, len)`, header
//! first. `valid` rides too — it is one byte a row, and a memset on restore would have to
//! reason about stream ordering the copy already gets for free.
//!
//! # Sizing is a real constraint, not a detail
//!
//! `4·D + 1 = 513 B` per token per layer at `index_head_dim = 128`, times 11 DSA layers =
//! **5,643 B per token of context**, host-resident per snapshot slot. At 32,768 tokens that is
//! ~176 MiB per snapshot and ~2.75 GiB across the default 16 Marconi slots; at 131,072 it would
//! be 11 GiB. QSA's precedent is 3,072 B/token across 12 layers, so this is ~1.8x heavier. The
//! cap below is what keeps the host budget bounded; above it every DSA layer returns `None`,
//! the snapshot is saved aux-less, and the restore gate declines it (recompute from 0 — slower,
//! never stale).
//!
//! # Save-side cost on this branch
//!
//! Each `snapshot` is three `copy_d2h_on_stream` calls, and that primitive DRAINS the stream
//! per call — 33 drains per checkpoint across the 11 layers, every 64 decode tokens at the
//! decode-checkpoint cadence. Correct, not fast. wip/exl3-research's batched
//! `snapshot_aux_plan`/`snapshot_aux_into` (plan -> enqueue -> ONE sync -> split) is the fix on
//! rebase; the blob layout here is what that gather will emit, so nothing in this file's
//! format needs to change then.

use anyhow::{Result, bail};
use spark_runtime::gpu::GpuBackend;

use super::state::Glm5NextDsaState;

/// `"GDSA"` little-endian — a blob that does not start with this is not ours.
pub const AUX_MAGIC: u32 = 0x4153_4447;
/// Bump when the layout after the header changes. A restore refuses any other version rather
/// than guessing, because the failure mode of a mis-parsed image is a wrong selection, not a
/// crash.
pub const AUX_VERSION: u32 = 1;
/// `[magic u32][version u32][len u64][index_head_dim u32][layer_idx u32]`.
pub const AUX_HEADER_BYTES: usize = 24;

/// Default for `ATLAS_GLM_DSA_AUX_MAX_TOKENS`. Sized by arithmetic, not by a measurement:
/// 32K is ~176 MiB host per slot, already larger than the KDA device slot it travels with
/// (2380 MiB / 16 = ~149 MiB), and 128K would be 11 GiB of host RAM for the default 16 slots
/// — earlyoom territory on a GB10 (two hard reboots on record from over-commit). Untuned;
/// the first GPU soak should say whether warm prefixes ever exceed it.
const DEFAULT_AUX_MAX_TOKENS: usize = 32_768;

/// Longest DSA indexer cache (rows) a Marconi snapshot will carry. Read ONCE from
/// `ATLAS_GLM_DSA_AUX_MAX_TOKENS` and logged at INFO so a serve log states the cap it ran with.
///
/// 🪤 Under TP=2 the value must be identical on both ranks: the indexer is replicated, each
/// rank snapshots its own copy, and a rank that declines while the other restores would
/// diverge on the very next selection. There is no cross-rank check; the launcher's shared
/// environment is what keeps them equal.
///
/// `0` is honoured literally — every non-empty state declines, which is the negative control
/// (`ATLAS_GLM_DSA_AUX_MAX_TOKENS=1` in the first GPU test: every hit must log a decline and
/// still answer correctly).
pub fn aux_max_tokens() -> usize {
    static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        let cap = std::env::var("ATLAS_GLM_DSA_AUX_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(DEFAULT_AUX_MAX_TOKENS);
        tracing::info!(
            "GLM DSA Marconi aux cap: {cap} tokens ({} B/token/layer at D=128; snapshots of a \
             longer indexer cache are saved aux-less and declined on restore)",
            4 * 128 + 1
        );
        cap
    })
}

/// The 24-byte header in front of the row image. Pure data, so the refusal arms can be tested
/// without a GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuxHeader {
    /// Rows imaged: `k_normed[0..len)`, `gate[0..len)`, `valid[0..len)`.
    pub len: usize,
    /// Row width in elements. Pinned so a blob from a differently-configured indexer cannot be
    /// laid over these buffers with the rows misaligned.
    pub index_head_dim: usize,
    /// The layer the rows came from. Every DSA layer's rows look alike byte-wise, so without
    /// this a blob set that was assembled with layers permuted would restore cleanly and
    /// select on the wrong layer's keys.
    pub layer_idx: u32,
}

impl AuxHeader {
    /// Bytes of the whole blob this header describes: header + `len · (4·D + 1)`.
    pub fn blob_bytes(&self) -> usize {
        AUX_HEADER_BYTES + self.len * (4 * self.index_head_dim + 1)
    }

    pub fn encode(&self) -> [u8; AUX_HEADER_BYTES] {
        let mut h = [0u8; AUX_HEADER_BYTES];
        h[0..4].copy_from_slice(&AUX_MAGIC.to_le_bytes());
        h[4..8].copy_from_slice(&AUX_VERSION.to_le_bytes());
        h[8..16].copy_from_slice(&(self.len as u64).to_le_bytes());
        h[16..20].copy_from_slice(&(self.index_head_dim as u32).to_le_bytes());
        h[20..24].copy_from_slice(&self.layer_idx.to_le_bytes());
        h
    }

    /// Parse and refuse: too short, wrong magic, wrong version. Each arm names itself — a
    /// restore refusal is a per-request error in a serve log, and "size mismatch" alone does
    /// not say whether the blob or the receiver is the odd one out.
    pub fn decode(blob: &[u8]) -> Result<Self> {
        if blob.len() < AUX_HEADER_BYTES {
            bail!(
                "DSA aux blob truncated: {} bytes, header alone is {AUX_HEADER_BYTES}",
                blob.len()
            );
        }
        let u32_at = |o: usize| u32::from_le_bytes(blob[o..o + 4].try_into().unwrap());
        let magic = u32_at(0);
        if magic != AUX_MAGIC {
            bail!("DSA aux blob magic {magic:#010x} is not {AUX_MAGIC:#010x} (\"GDSA\")");
        }
        let version = u32_at(4);
        if version != AUX_VERSION {
            bail!("DSA aux blob version {version} is not {AUX_VERSION}; refusing to guess the layout");
        }
        let len = u64::from_le_bytes(blob[8..16].try_into().unwrap());
        let len = usize::try_from(len)
            .map_err(|_| anyhow::anyhow!("DSA aux blob len {len} does not fit a usize"))?;
        Ok(Self {
            len,
            index_head_dim: u32_at(16) as usize,
            layer_idx: u32_at(20),
        })
    }

    /// Every host-side check a restore makes BEFORE the first upload, so a refused blob never
    /// half-writes a state. `layer_idx`/`index_head_dim` are the receiving layer's;
    /// `capacity` is the receiving state's reservation; `blob_len` is the blob as delivered.
    pub fn validate_for(
        &self,
        layer_idx: u32,
        index_head_dim: usize,
        capacity: usize,
        blob_len: usize,
    ) -> Result<()> {
        if self.layer_idx != layer_idx {
            bail!(
                "DSA aux blob was taken from layer {} but is being restored into layer \
                 {layer_idx}; a permuted blob set would select on the wrong layer's keys",
                self.layer_idx
            );
        }
        if self.index_head_dim != index_head_dim {
            bail!(
                "DSA aux blob row width {} does not match this indexer's {index_head_dim}",
                self.index_head_dim
            );
        }
        if self.len > capacity {
            bail!(
                "DSA aux blob holds {} rows but this sequence reserved {capacity}; the blob \
                 was taken under a larger --max-seq-len than this serve",
                self.len
            );
        }
        let want = self.blob_bytes();
        if blob_len != want {
            bail!(
                "DSA aux blob is {blob_len} bytes; a {}-row, D={} image is exactly {want}",
                self.len,
                self.index_head_dim
            );
        }
        Ok(())
    }
}

/// Byte ranges of the three sections after the header, in blob order.
fn sections(h: &AuxHeader) -> [std::ops::Range<usize>; 3] {
    let rows = h.len * h.index_head_dim * 2;
    let k = AUX_HEADER_BYTES..AUX_HEADER_BYTES + rows;
    let g = k.end..k.end + rows;
    let v = g.end..g.end + h.len;
    [k, g, v]
}

/// Image `st`'s rows `[0, len)` into a blob, or `None` when `len` exceeds [`aux_max_tokens`].
///
/// `None` above the cap is ALL-OR-NOTHING across the 11 DSA layers by construction: every DSA
/// layer's `len` is the sequence length at a pass end (the lockstep contract), so either all
/// of them are over the cap or none is. The snapshot is then saved aux-less and the restore
/// gate declines it. Below the cap this ALWAYS returns `Some`, including at `len == 0` — an
/// empty image is a valid, complete statement ("this layer has ingested nothing"), and the
/// restore gate's completeness check needs one blob per aux-carrying layer.
///
/// D2H is `copy_d2h_on_stream`, ordered after the pass that wrote the rows on `stream`. The
/// caller (`collect_aux_states`) takes this at a checkpoint that is a completed pass end, so
/// the rows are position-correct at `len` by construction.
pub fn snapshot(
    st: &Glm5NextDsaState,
    layer_idx: u32,
    gpu: &dyn GpuBackend,
    stream: u64,
) -> Result<Option<Vec<u8>>> {
    snapshot_with_cap(st, layer_idx, aux_max_tokens(), gpu, stream)
}

/// [`snapshot`] with the cap passed in — the env-free core, so the decline arm is testable.
pub fn snapshot_with_cap(
    st: &Glm5NextDsaState,
    layer_idx: u32,
    cap: usize,
    gpu: &dyn GpuBackend,
    stream: u64,
) -> Result<Option<Vec<u8>>> {
    let len = st.len();
    if len > cap {
        return Ok(None);
    }
    let h = AuxHeader {
        len,
        index_head_dim: st.index_head_dim(),
        layer_idx,
    };
    let mut blob = vec![0u8; h.blob_bytes()];
    blob[..AUX_HEADER_BYTES].copy_from_slice(&h.encode());
    if len > 0 {
        let [k, g, v] = sections(&h);
        gpu.copy_d2h_on_stream(st.k_normed, &mut blob[k], stream)?;
        gpu.copy_d2h_on_stream(st.gate, &mut blob[g], stream)?;
        gpu.copy_d2h_on_stream(st.valid, &mut blob[v], stream)?;
    }
    Ok(Some(blob))
}

/// Put a [`snapshot`] blob back into `st` on a prefix-cache hit, BEFORE the resumed prefill.
///
/// Validation is entirely host-side and entirely before the first H2D (see
/// [`AuxHeader::validate_for`]): a refused blob leaves the state untouched and the error
/// propagates through `apply_aux_states` with `?`, so the request fails rather than resuming
/// on a half-written image. `copy_h2d_async` from the transient `blob` is safe — the CUDA
/// backend stages pageable sources before returning (`gpu_impl.rs:394-418`), which is what
/// lets `apply_aux_states` drop its clone right after.
///
/// After the upload the counter is set with [`Glm5NextDsaState::set_len_restored`]; the
/// first resumed row then meets `decode_k`'s lockstep check, which bails if `len` is behind
/// the sequence and rewinds exactly if it is ahead (a decode checkpoint taken right after a
/// partially rejected verify carries a few extra rows — sound, do not "fix" it by refusing).
pub fn restore(
    st: &mut Glm5NextDsaState,
    layer_idx: u32,
    blob: &[u8],
    gpu: &dyn GpuBackend,
    stream: u64,
) -> Result<()> {
    let h = AuxHeader::decode(blob)?;
    h.validate_for(layer_idx, st.index_head_dim(), st.capacity(), blob.len())?;
    if h.len > 0 {
        let [k, g, v] = sections(&h);
        gpu.copy_h2d_async(&blob[k], st.k_normed, stream)?;
        gpu.copy_h2d_async(&blob[g], st.gate, stream)?;
        gpu.copy_h2d_async(&blob[v], st.valid, stream)?;
    }
    st.set_len_restored(h.len)
}

#[cfg(test)]
mod tests;
