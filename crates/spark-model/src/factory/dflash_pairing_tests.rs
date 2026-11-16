// SPDX-License-Identifier: AGPL-3.0-only

//! CPU tests for the DFlash pairing check's view of the target: the table
//! rows it reads from the weight store, and the two outcomes that matter (a
//! padded table pairs with the drafter that states its rows; a drafter for
//! another vocabulary is still refused).

use super::*;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::weights::{DeferredTensor, WeightDtype, WeightTensor};
use std::collections::HashMap;

/// `z-lab/Qwen3.5-4B-DFlash/config.json`: vocab 248320, trained against a
/// 32-layer target, mask token 248077.
const DRAFTER_4B: &str =
    include_str!("../weight_loader/dflash_loader/fixtures/qwen3_5_4b_dflash_config.json");

/// The one embedding tensor of the Qwen3.5-4B checkpoint, `[248320, 2560]`.
const EMBED: &str = "model.language_model.embed_tokens.weight";

/// Rows of that table, and what the tokenizer leaves of them.
const TABLE_ROWS: usize = 248_320;
const TOKENIZER: usize = 248_070;
const HIDDEN: usize = 2560;

fn store(tensors: &[(&str, [usize; 2])]) -> WeightStore {
    let map: HashMap<String, WeightTensor> = tensors
        .iter()
        .map(|(name, shape)| {
            let tensor = WeightTensor {
                ptr: DevicePtr::NULL,
                shape: shape.to_vec(),
                dtype: WeightDtype::BF16,
            };
            ((*name).to_string(), tensor)
        })
        .collect();
    WeightStore::from_map(map)
}

/// The 4B target as `build_model` receives it: `served` is
/// `ModelConfig::vocab_size` after the server's tokenizer cap.
fn target(served: usize) -> ModelConfig {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.hidden_size = HIDDEN;
    config.num_hidden_layers = 32;
    config.vocab_size = served;
    config
}

fn drafter() -> DflashConfig {
    crate::weight_loader::dflash_loader::parse_dflash_config(DRAFTER_4B).expect("fixture parses")
}

#[test]
fn a_padded_table_pairs_with_the_drafter_that_states_its_rows() {
    // An earlier GPU run on this card: the checkpoint's table has 248320 rows, the tokenizer
    // names 248070 of them, and the drafter states 248320.
    let s = store(&[(EMBED, [TABLE_ROWS, HIDDEN])]);
    let t = target(TOKENIZER);
    assert_eq!(drafter().vocab_size, TABLE_ROWS);
    assert_eq!(shared_table_rows(&s, &t), Some(TABLE_ROWS));
    check(&drafter(), &t, &s).expect("the 4B drafter pairs with the 4B target");
    // The comparison that refused it: the served vocabulary.
    let e = drafter()
        .validate_for_target(HIDDEN, 32, t.vocab_size)
        .unwrap_err();
    assert!(e.to_string().contains("vocabulary (248320)"), "{e}");
    // An uncapped config (no tokenizer.json beside the model) pairs too.
    check(&drafter(), &target(TABLE_ROWS), &s).expect("uncapped");
}

#[test]
fn a_drafter_for_another_vocabulary_is_still_refused() {
    // A target whose table has other rows (a 151,936-row Qwen3 embedding).
    let other = store(&[(EMBED, [151_936, HIDDEN])]);
    let e = check(&drafter(), &target(151_665), &other).unwrap_err();
    let msg = e.to_string();
    assert!(msg.contains("vocabulary (248320)"), "{msg}");
    assert!(msg.contains("151936"), "{msg}");
    // The 4B table against a drafter that states another vocabulary.
    let s = store(&[(EMBED, [TABLE_ROWS, HIDDEN])]);
    let mut wrong = drafter();
    wrong.vocab_size = 151_936;
    let e = check(&wrong, &target(TOKENIZER), &s).unwrap_err();
    assert!(e.to_string().contains("vocabulary (151936)"), "{e}");
    // A drafter that states the TOKENIZER count is not the table either: its
    // tail GEMM would stop 250 rows short of what it was trained on.
    let mut short = drafter();
    short.vocab_size = TOKENIZER;
    assert!(check(&short, &target(TOKENIZER), &s).is_err());
    // The other pairing rules are untouched: wrong depth still refuses.
    let mut shallow = target(TOKENIZER);
    shallow.num_hidden_layers = 28;
    let e = check(&drafter(), &shallow, &s).unwrap_err();
    assert!(e.to_string().contains("taps target layer"), "{e}");
}

#[test]
fn the_mask_token_is_checked_against_the_drafters_vocabulary() {
    // The mask token is row 248077: past the tokenizer, inside the table.
    let d = drafter();
    let sub = d.dflash_config.as_ref().expect("dflash block");
    let mask = sub.mask_token_id as usize;
    assert_eq!(mask, 248_077);
    assert!((TOKENIZER..TABLE_ROWS).contains(&mask), "mask token {mask}");
    let s = store(&[(EMBED, [TABLE_ROWS, HIDDEN])]);
    check(&d, &target(TOKENIZER), &s).expect("a mask row in the padding is legal");
    // A mask id past the drafter's own vocabulary is refused whatever the
    // table says.
    let mut bad = drafter();
    let sub = bad.dflash_config.as_mut().expect("dflash block");
    sub.mask_token_id = TABLE_ROWS as u32;
    let e = check(&bad, &target(TOKENIZER), &s).unwrap_err();
    assert!(e.to_string().contains("mask_token_id"), "{e}");
}

#[test]
fn an_untied_lm_head_is_the_table_that_counts() {
    // The head padded to 248320 rows beside a shorter embedding: the tail
    // GEMM walks the head.
    let s = store(&[
        ("lm_head.weight", [TABLE_ROWS, HIDDEN]),
        (EMBED, [248_077, HIDDEN]),
    ]);
    assert_eq!(shared_table_rows(&s, &target(TOKENIZER)), Some(TABLE_ROWS));
    for key in LM_HEAD_KEYS {
        let s = store(&[(key, [TABLE_ROWS, HIDDEN])]);
        assert_eq!(
            shared_table_rows(&s, &target(TOKENIZER)),
            Some(TABLE_ROWS),
            "{key}"
        );
    }
}

#[test]
fn a_packed_head_leaves_the_question_to_the_embedding() {
    // A pre-packed NVFP4 head is not [rows, hidden].
    let s = store(&[
        ("lm_head.weight", [TABLE_ROWS, HIDDEN / 2]),
        (EMBED, [TABLE_ROWS, HIDDEN]),
    ]);
    assert_eq!(shared_table_rows(&s, &target(TOKENIZER)), Some(TABLE_ROWS));
}

#[test]
fn a_deferred_embedding_is_read_from_its_header_shape() {
    let mut s = WeightStore::empty();
    s.defer(
        EMBED.to_string(),
        DeferredTensor {
            path: "model-00001-of-00002.safetensors".into(),
            offset: 0,
            shape: vec![TABLE_ROWS, HIDDEN],
            dtype: WeightDtype::BF16,
        },
    );
    assert_eq!(shared_table_rows(&s, &target(TOKENIZER)), Some(TABLE_ROWS));
    check(&drafter(), &target(TOKENIZER), &s).expect("deferred table");
}

#[test]
fn only_a_tensor_named_as_the_token_embedding_counts() {
    assert!(is_embedding("embed_tokens.weight"));
    assert!(is_embedding(EMBED));
    assert!(!is_embedding("model.embed_tokens.weight_scale"));
    assert!(!is_embedding("model.other_embed_tokens.weight"));
    assert!(!is_embedding("model.visual.patch_embed.proj.weight"));
    // Same width, another name: not the table.
    let s = store(&[("model.visual.pos_embed.weight", [TABLE_ROWS, HIDDEN])]);
    assert_eq!(shared_table_rows(&s, &target(TOKENIZER)), None);
    // The right name at another width (a tower's own table): not it either.
    let s = store(&[("model.tower.embed_tokens.weight", [TABLE_ROWS, 1024])]);
    assert_eq!(shared_table_rows(&s, &target(TOKENIZER)), None);
}

#[test]
fn nothing_is_guessed_where_the_store_does_not_say() {
    let t = target(TOKENIZER);
    // No table at all.
    assert_eq!(shared_table_rows(&WeightStore::empty(), &t), None);
    // Two embeddings of the model's width that disagree.
    let two = store(&[
        (EMBED, [TABLE_ROWS, HIDDEN]),
        ("model.aux.embed_tokens.weight", [100_000, HIDDEN]),
    ]);
    assert_eq!(shared_table_rows(&two, &t), None);
    // Two that agree are one answer.
    let same = store(&[
        (EMBED, [TABLE_ROWS, HIDDEN]),
        ("model.aux.embed_tokens.weight", [TABLE_ROWS, HIDDEN]),
    ]);
    assert_eq!(shared_table_rows(&same, &t), Some(TABLE_ROWS));
    // Fewer rows than the served vocabulary is a slice (a vocab-parallel
    // shard), not the table.
    let shard = store(&[(EMBED, [TOKENIZER / 2, HIDDEN])]);
    assert_eq!(shared_table_rows(&shard, &t), None);
    // Then the served vocabulary stands in, which is the stricter test: the
    // 4B drafter is refused against 248070 and accepted against 248320.
    let e = check(&drafter(), &t, &WeightStore::empty()).unwrap_err();
    assert!(e.to_string().contains("vocabulary (248320)"), "{e}");
    check(&drafter(), &target(TABLE_ROWS), &WeightStore::empty()).expect("uncapped, no table");
}
