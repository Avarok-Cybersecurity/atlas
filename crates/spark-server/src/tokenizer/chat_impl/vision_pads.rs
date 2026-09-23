// SPDX-License-Identifier: AGPL-3.0-only

//! Vision placeholder tokens and the one-pad-per-image → one-pad-per-patch
//! fan-out.
//!
//! Split from `chat_impl.rs` for the 500-LoC cap, and it earns its own file:
//! this is where the prompt's image placeholders and the vision encoder's
//! output rows are made to be the same count, and a disagreement between them
//! is silent in both directions.

use super::super::ChatTokenizer;

impl ChatTokenizer {
    /// The image placeholder token id.
    ///
    /// `declared` is the checkpoint's own `vision_config.image_token_id`, and
    /// it WINS when non-zero. The literal probe below is the fallback for a
    /// checkpoint that declares nothing.
    ///
    /// 🔴 The probe alone is not enough, and the failure is silent. It looks
    /// for Qwen's `<|image_pad|>`; GLM-5.3 names the same token `<|image|>`
    /// (154854), so on GLM the probe returns `None`, `expand_vision_pads`
    /// becomes a no-op, and the prompt keeps ONE image token where the encoder
    /// produced 256 embedding rows. Nothing errors: the splice overwrites the
    /// single position, the remaining 255 rows are never read, and the model
    /// answers about a 1-token image. The id has been in `VisionConfig` the
    /// whole time — it just was not asked for.
    pub fn image_pad_token_id(&self, declared: u32) -> Option<u32> {
        if declared != 0 {
            return Some(declared);
        }
        self.encode("<|image_pad|>")
            .ok()
            .and_then(|ids| if ids.len() == 1 { Some(ids[0]) } else { None })
    }

    /// `<|video_pad|>`, the temporal sibling, with the same declared-wins rule.
    /// `None` on a tokenizer without it — every text-only model, and any VL
    /// model that predates video.
    pub fn video_pad_token_id(&self, declared: u32) -> Option<u32> {
        if declared != 0 {
            return Some(declared);
        }
        self.encode("<|video_pad|>")
            .ok()
            .and_then(|ids| if ids.len() == 1 { Some(ids[0]) } else { None })
    }

    /// Post-process a rendered token sequence to expand `<|image_pad|>`
    /// placeholders. The Qwen3-VL / Qwen3.6 chat template emits exactly one
    /// `<|image_pad|>` per image, but the vision encoder produces
    /// `grid_h * grid_w` patches per image. At embed-injection time the
    /// server expects one pad token per patch so each patch's embedding
    /// lands at the right hidden-state position — this helper does the
    /// fan-out.
    ///
    /// `pad_counts[i]` is the number of patches the i-th image produces.
    /// Extra or missing `<|image_pad|>` occurrences (vs `pad_counts.len()`)
    /// pass through unchanged, matching counts are replicated in place.
    pub fn expand_vision_pads(
        &self,
        tokens: Vec<u32>,
        pad_counts: &[usize],
        declared: (u32, u32),
    ) -> Vec<u32> {
        if pad_counts.is_empty() || pad_counts.iter().all(|&c| c <= 1) {
            return tokens;
        }
        fan_out_pads(
            tokens,
            pad_counts,
            self.image_pad_token_id(declared.0),
            self.video_pad_token_id(declared.1),
        )
    }
}

/// The fan-out itself, as a free function over EXPLICIT ids so it can be
/// tested without a tokenizer.
///
/// `pad_counts[i]` is the number of merged tokens the i-th media item
/// produces. Occurrences are replaced left to right, one count each; a
/// mismatch in either direction passes the extra tokens through unchanged
/// rather than guessing.
pub(crate) fn fan_out_pads(
    tokens: Vec<u32>,
    pad_counts: &[usize],
    image_pad: Option<u32>,
    video_pad: Option<u32>,
) -> Vec<u32> {
    if image_pad.is_none() && video_pad.is_none() {
        return tokens;
    }
    let extra: usize = pad_counts.iter().map(|c| c.saturating_sub(1)).sum();
    let mut out = Vec::with_capacity(tokens.len() + extra);
    let mut img_idx = 0usize;
    for t in tokens {
        if Some(t) == image_pad || Some(t) == video_pad {
            let count = pad_counts.get(img_idx).copied().unwrap_or(1).max(1);
            for _ in 0..count {
                out.push(t);
            }
            img_idx += 1;
        } else {
            out.push(t);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::fan_out_pads;

    /// GLM-5.3's id (154854) against a tokenizer that has no `<|image_pad|>`.
    /// This is the case the literal-probe could not serve: without the
    /// declared id nothing expands, the prompt keeps ONE image position for
    /// 256 encoder rows, and the model answers about a 1-token image without
    /// anything erroring.
    #[test]
    fn a_declared_id_expands_where_a_literal_probe_would_not_have() {
        let glm_image = 154_854u32;
        let tokens = vec![1, 154_830, glm_image, 154_831, 2];
        let out = fan_out_pads(tokens.clone(), &[256], Some(glm_image), None);
        assert_eq!(out.len(), tokens.len() + 255);
        assert_eq!(out.iter().filter(|&&t| t == glm_image).count(), 256);
        // The surrounding begin/end-of-image markers are untouched and still
        // bracket the run.
        assert_eq!(out[1], 154_830);
        assert_eq!(*out.last().unwrap(), 2);
        assert_eq!(out[out.len() - 2], 154_831);

        // Without an id — the pre-fix behaviour on GLM — nothing moves.
        assert_eq!(fan_out_pads(tokens.clone(), &[256], None, None), tokens);
    }

    /// Multiple items consume their counts in order, and images and videos
    /// share one left-to-right cursor.
    #[test]
    fn counts_are_consumed_in_order_across_both_modalities() {
        let (img, vid) = (100u32, 101u32);
        let out = fan_out_pads(vec![img, 7, vid, img], &[2, 3, 1], Some(img), Some(vid));
        assert_eq!(out, vec![img, img, 7, vid, vid, vid, img]);
    }

    /// More placeholders than counts: the surplus passes through as one token
    /// each rather than reusing the last count.
    #[test]
    fn a_short_count_list_does_not_reuse_the_last_count() {
        let img = 100u32;
        let out = fan_out_pads(vec![img, img], &[3], Some(img), None);
        assert_eq!(out, vec![img, img, img, img]);
    }
}
