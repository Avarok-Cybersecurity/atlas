// SPDX-License-Identifier: AGPL-3.0-only

//! Vision pad-token id resolution.
//!
//! A sibling file rather than more of `tokenizer/tests.rs`, which sits at
//! exactly the 500-LoC cap and is not on the allow-list.

use atlas_core::config::VisionConfig;

use super::chat_impl::resolve_vision_pad_ids;

fn glm_vision() -> VisionConfig {
    VisionConfig {
        // GLM-5.3-Flash `config.json`: `<|image|>` / `<|video|>`.
        image_pad_token_id: 154_854,
        video_pad_token_id: 154_855,
        model_type: "glm5_next_vision".to_string(),
        ..VisionConfig::default()
    }
}

/// Mirrors `spark-model`'s `model/impl_a2.rs::vision_pad_ids`, the resolver
/// that decides which positions the splice overwrites. It is `pub(super)`
/// there, so this test re-states its rule rather than calling it; the two
/// differing is exactly the failure being pinned, so the rule is written out
/// once here in full and compared against the tokenizer's answer.
fn splice_side_ids(vision: Option<&VisionConfig>) -> (u32, u32) {
    let Some(v) = vision else {
        return (u32::MAX, u32::MAX);
    };
    let image = if v.image_pad_token_id != 0 {
        v.image_pad_token_id
    } else {
        spark_model::layers::vision_encoder::IMAGE_PAD_TOKEN_ID
    };
    let video = if v.video_pad_token_id != 0 {
        v.video_pad_token_id
    } else {
        spark_model::layers::vision_encoder::VIDEO_PAD_TOKEN_ID
    };
    (image, video)
}

#[test]
fn pad_ids_agree_between_tokenizer_and_splice() {
    let glm = glm_vision();
    assert_eq!(
        resolve_vision_pad_ids(Some(&glm)),
        Some(splice_side_ids(Some(&glm))),
    );
    assert_eq!(resolve_vision_pad_ids(Some(&glm)), Some((154_854, 154_855)));

    // A checkpoint declaring 0 (the "not stated" encoding) takes the Qwen
    // family fallback on BOTH sides, from the same constants.
    let unstated = VisionConfig::default();
    assert_eq!(
        resolve_vision_pad_ids(Some(&unstated)),
        Some(splice_side_ids(Some(&unstated))),
    );
    assert_eq!(
        resolve_vision_pad_ids(Some(&unstated)),
        Some((
            spark_model::layers::vision_encoder::IMAGE_PAD_TOKEN_ID,
            spark_model::layers::vision_encoder::VIDEO_PAD_TOKEN_ID,
        )),
    );
}

/// 🪤 A text-only checkpoint must resolve to "matches nothing", never to
/// Qwen's 151655/151656. Those are ordinary byte-BPE pieces in other
/// vocabularies (GLM-5.3 spells two emoji with them), and treating a text
/// prompt containing one as a vision prompt silently disables prefix-cache
/// lookup, prefix-cache insert and the Marconi snapshot for that request.
#[test]
fn a_text_only_config_matches_nothing() {
    assert_eq!(resolve_vision_pad_ids(None), None);
    assert_eq!(splice_side_ids(None), (u32::MAX, u32::MAX));
}

/// The regression this whole change exists to prevent: the pad id used to be
/// derived by ENCODING the literal string `"<|image_pad|>"` and accepting it
/// only when it was one token. That spelling is Qwen's and is absent from
/// GLM-5.3's vocabulary entirely, so both getters returned `None`, the
/// fan-out short-circuited, and a 4096-token image shipped ONE pad.
///
/// The id lives in `config.json` for both families; the spelling does not.
/// Asserted structurally — a `VisionConfig` carrying GLM's ids resolves
/// without the tokenizer being consulted at all.
#[test]
fn resolution_does_not_depend_on_the_qwen_spelling() {
    let glm = glm_vision();
    let (image, _) = resolve_vision_pad_ids(Some(&glm)).expect("glm declares pads");
    assert_ne!(
        image,
        spark_model::layers::vision_encoder::IMAGE_PAD_TOKEN_ID,
        "GLM must not resolve to the Qwen id"
    );
}
