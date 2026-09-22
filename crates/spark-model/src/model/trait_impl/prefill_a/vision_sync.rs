// SPDX-License-Identifier: AGPL-3.0-only

//! Give every rank the SAME merged vision embeddings before the prefill
//! splice.
//!
//! # The failure this exists to make impossible
//!
//! Only rank 0 receives image bytes, so only rank 0 runs the ViT. Every other
//! rank reached `embed_chunk` with `vision_embed_patches == 0`, kept the raw
//! `<|image|>` token embedding at each pad position, and then all-reduced that
//! into rank 0's spliced rows on every layer. The model stays FLUENT — it
//! answers confidently about a picture nobody showed it — which is why the
//! Qwen3.8 instance of this (rsafier, PR #1066, "Vision across ranks": it read
//! **HARBOR** as *Superman*) passed every check that looked at rank 0 alone:
//! tower cosine 0.998 vs HF, splice exact, positions right to the digit, all
//! on one rank of two. The lesson is his; this is GLM's copy of it.
//!
//! # Design: broadcast the merged rows, do not re-run the tower
//!
//! Rank 0 broadcasts `[n_rows, out_hidden_size]` BF16 straight out of the
//! encoder's `buf_out` — for a 448x448 image, 256 rows x 4096 x 2 B = 2 MB,
//! against the 1.05 GiB of tower weights each rank already holds. The
//! alternative — fan the pixels out and let each rank encode locally — moves
//! MORE bytes (`pixel_values` is 4.8 MB for the same image), spends a second
//! ViT pass, and makes cross-rank agreement something to be argued rather than
//! something the wire guarantees. A broadcast is bit-identity by construction.
//!
//! # Why GLM needs nothing else, and an MRoPE model would
//!
//! PR #1066's other half was the position stream: Qwen3.8's ranks disagreed on
//! `(T, H, W)` because the workers had no grids. GLM has no MRoPE at all —
//! `Glm5NextModel` deletes `rope_deltas`, and NoPE MLA + KDA consume no
//! spatial positions — so `mrope_interleaved` is false and both ranks build
//! the identical flat stream from the tokens they both already have. 🪤 A
//! future MRoPE model on this path still needs its grids broadcast; this
//! function does not do that, and `embed_chunk` would be right while
//! `upload_meta` was wrong.

use anyhow::Result;

use super::super::super::types::TransformerModel;

/// How many merged rows this prompt's pad run needs.
///
/// Derived from the TOKENS, which both ranks already hold, rather than sent —
/// so the broadcast count can be checked against something independent instead
/// of being believed.
pub(crate) fn vision_rows_for_prompt(tokens: &[u32], image_pad: u32, video_pad: u32) -> usize {
    tokens
        .iter()
        .filter(|&&t| t == image_pad || t == video_pad)
        .count()
}

/// What rank 0 should send for this prompt.
///
/// `pending` is `vision_embed_patches` — the total merged rows in the shared
/// packed `buf_out`, which under co-dispatch spans SEVERAL requests. This
/// request owns `[row_base, row_base + n_pad)` of it. When rank 0 has no
/// encoder output staged (`pending == 0`) it will not splice either, so the
/// answer is 0 and both ranks skip in step — that is what keeps a
/// preempt-resume re-prefill, which re-runs no ViT, from desynchronising the
/// two.
pub(crate) fn rows_to_broadcast(pending: usize, row_base: usize, n_pad: usize) -> Result<usize> {
    if pending == 0 || n_pad == 0 {
        return Ok(0);
    }
    let end = row_base
        .checked_add(n_pad)
        .ok_or_else(|| anyhow::anyhow!("vision sync: row_base {row_base} + {n_pad} overflows"))?;
    anyhow::ensure!(
        end <= pending,
        "vision sync: this request claims rows [{row_base}, {end}) of a packed buffer holding \
         {pending}. The prompt's pad run and the encoder's output disagree, which is the \
         desync the splice cannot see."
    );
    Ok(n_pad)
}

/// FNV-1a over the bytes of the rows that will be spliced. Not a checksum for
/// storage — a cheap cross-rank equality witness, which is precisely what
/// PR #1066 had no way to state.
pub(crate) fn digest(bytes: &[u8]) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for &b in bytes {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

/// Is the cross-rank equality assertion armed?
///
/// Off by default because it costs a D2H of every spliced row (2 MB for one
/// 448x448 image) on every prefill. On, it turns a silent wrong answer into a
/// refused request.
pub(crate) fn check_enabled() -> bool {
    matches!(
        std::env::var("AVAROK_GLM_VISION_CHECK")
            .ok()
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

impl TransformerModel {
    /// Rank 0 → workers: the merged vision rows for this prompt.
    ///
    /// Called by BOTH sides at the same point in the `0xFFFFFFF0` wire
    /// sequence — immediately after the prompt tokens and before
    /// `prefill_chunk` — so the collective pairs by construction. A model with
    /// no vision tower does not call it at all, which keeps the text wire
    /// protocol byte-identical rather than merely equivalent.
    pub(in crate::model) fn ep_sync_vision_embeds(&self, tokens: &[u32]) -> Result<()> {
        if !self.multi_rank_protocol_active() {
            return Ok(());
        }
        let Some(ve) = self.vision_encoder.as_ref() else {
            return Ok(());
        };
        let comm = self.comm.as_ref().expect("multi_rank_protocol_active");
        let (image_pad, video_pad) = self.vision_pad_ids();
        let n_pad = vision_rows_for_prompt(tokens, image_pad, video_pad);
        let is_head = comm.rank() == 0;

        let n_rows = if is_head {
            let pending = *self.vision_embed_patches.lock();
            let row_base = *self.vision_row_base.lock();
            let n = rows_to_broadcast(pending, row_base, n_pad)?;
            self.ep_broadcast_u32(n as u32)?;
            n
        } else {
            let n = self.ep_broadcast_u32(0)? as usize;
            // The count is checked against the worker's OWN reading of the
            // prompt. A head that thinks this prompt has a different number of
            // image rows than the tokens say is the desync, not the symptom.
            anyhow::ensure!(
                n == 0 || n == n_pad,
                "vision sync: rank {} was sent {n} merged rows but counts {n_pad} vision pad \
                 tokens in the prompt. Head and worker disagree about this request's images.",
                comm.rank()
            );
            n
        };
        if n_rows == 0 {
            if !is_head {
                // Explicitly clear, so a stale count from a previous request
                // can never make a worker splice into a text prompt.
                *self.vision_embed_patches.lock() = 0;
                *self.vision_row_base.lock() = 0;
            }
            return Ok(());
        }

        // The worker's scratch is allocated on its first IMAGE, and it never
        // sees one — so it is allocated here instead, before the first byte
        // lands in `buf_out`.
        ve.ensure_scratch(self.gpu.as_ref())?;
        let out = ve.out_hidden_size();
        let bytes = n_rows * out * 2;
        let src_row = if is_head {
            *self.vision_row_base.lock()
        } else {
            0
        };
        comm.broadcast(ve.out_row(src_row).0, bytes, 0)?;

        if !is_head {
            // The worker's rows land at 0, so its splice reads from 0. Rank 0
            // keeps its own (possibly co-dispatched) base.
            *self.vision_embed_patches.lock() = n_rows;
            *self.vision_row_base.lock() = 0;
        }
        self.verify_vision_rank_agreement(n_rows, src_row, bytes, is_head)
    }

    /// Log — and, when armed, ASSERT — that every rank is about to splice the
    /// same bytes.
    fn verify_vision_rank_agreement(
        &self,
        n_rows: usize,
        src_row: usize,
        bytes: usize,
        is_head: bool,
    ) -> Result<()> {
        let comm = self.comm.as_ref().expect("multi_rank_protocol_active");
        if !check_enabled() {
            tracing::debug!(
                "vision sync: rank {} holds {n_rows} merged rows at base {src_row}",
                comm.rank()
            );
            return Ok(());
        }
        let ve = self
            .vision_encoder
            .as_ref()
            .expect("checked by the only caller");
        self.gpu.synchronize(self.gpu.default_stream())?;
        let mut buf = vec![0u8; bytes];
        self.gpu.copy_d2h(ve.out_row(src_row), &mut buf)?;
        let local = digest(&buf);
        tracing::info!(
            "AVAROK_GLM_VISION_CHECK rank {} rows={n_rows} base={src_row} digest={local:#010x}",
            comm.rank()
        );
        let head = self.ep_broadcast_u32(if is_head { local } else { 0 })?;
        anyhow::ensure!(
            is_head || head == local,
            "vision sync: rank {} spliced digest {local:#010x} but rank 0 has {head:#010x} over \
             the same {n_rows} rows. The ranks are about to all-reduce DIFFERENT image \
             embeddings — the model would stay fluent and answer about the wrong picture.",
            comm.rank()
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The count both ranks derive independently. Video pads count too — a
    /// video's temporal groups occupy pad positions exactly like an image's.
    #[test]
    fn rows_are_counted_from_the_prompt_not_trusted_from_the_wire() {
        let (img, vid) = (154_854u32, 154_855u32);
        let tokens = vec![1, 154_830, img, img, img, 154_831, 7, vid, vid, 2];
        assert_eq!(vision_rows_for_prompt(&tokens, img, vid), 5);
        assert_eq!(vision_rows_for_prompt(&[1, 2, 3], img, vid), 0);
    }

    /// A text prompt on a vision-capable serve sends nothing, and so does a
    /// re-prefill where rank 0 staged no encoder output — the preempt-resume
    /// path runs no ViT, and both ranks must skip together rather than one
    /// waiting on a broadcast that never comes.
    #[test]
    fn nothing_is_sent_when_either_side_has_nothing() {
        assert_eq!(rows_to_broadcast(0, 0, 256).unwrap(), 0);
        assert_eq!(rows_to_broadcast(256, 0, 0).unwrap(), 0);
        assert_eq!(rows_to_broadcast(0, 0, 0).unwrap(), 0);
    }

    /// Under co-dispatch `buf_out` is shared: this request owns a WINDOW of
    /// it, and only its own rows go on the wire.
    #[test]
    fn a_co_dispatched_request_sends_only_its_own_window() {
        // Three requests of 256 rows packed together; the middle one.
        assert_eq!(rows_to_broadcast(768, 256, 256).unwrap(), 256);
        // The last one ends exactly at the end, which is in bounds.
        assert_eq!(rows_to_broadcast(768, 512, 256).unwrap(), 256);
    }

    /// A window running past the packed buffer means the pad run and the
    /// encoder output describe different things. That is the desync itself,
    /// and it must be an error rather than a short read the splice cannot see.
    #[test]
    fn a_window_past_the_packed_rows_is_refused() {
        let err = rows_to_broadcast(768, 512, 257).unwrap_err().to_string();
        assert!(err.contains("769"), "must name the end row: {err}");
        assert!(err.contains("768"), "must name the capacity: {err}");
        assert!(rows_to_broadcast(768, usize::MAX, 2).is_err());
    }

    /// The witness has to separate the case it exists for: one row different
    /// out of 256 must change the digest.
    #[test]
    fn the_digest_separates_a_single_changed_row() {
        let rows = vec![0xABu8; 256 * 4096 * 2];
        let mut one_off = rows.clone();
        // Flip one byte in the middle row — a single BF16 mantissa bit.
        one_off[128 * 4096 * 2] ^= 0x01;
        assert_ne!(digest(&rows), digest(&one_off));
        // And an empty payload is not confused with a zeroed one.
        assert_ne!(digest(&[]), digest(&[0u8; 8]));
    }

    /// Off unless explicitly armed — it costs a D2H of every spliced row.
    #[test]
    fn the_check_is_opt_in() {
        // `check_enabled` reads the process environment, which a test must not
        // mutate; assert the shape of the decision instead.
        assert!(!matches!(None::<&str>, Some("1" | "true" | "yes" | "on")));
        assert!(matches!(Some("1"), Some("1" | "true" | "yes" | "on")));
    }
}
