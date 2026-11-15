// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from HuggingFace transformers (models/lfm3, Apache-2.0, Copyright 2026 The HuggingFace Team) and vLLM (vllm/model_executor/models/lfm3.py, Apache-2.0, Copyright contributors to the vLLM project). Changes: ported from Python/Triton to Rust/CUDA for Atlas.

//! Responsibility: CPU tests of the LFM3 padded-vocabulary rule. The
//! checkpoint's embedding has 128000 rows (its config.json `vocab_size`); the
//! tokenizer names 125033 tokens, and the server caps `ModelConfig::vocab_size`
//! to that count before the loader runs. The loader checks the table against
//! the checkpoint's own row count, and the served width stays the tokenizer's.

use avarok_core::config::ModelConfig;
use spark_runtime::weights::WeightDtype;

use super::names::{self, EMBED, Geometry};
use super::tests::{problems_with, real};

const TABLE_ROWS: usize = 128_000;
const TOKENIZER_TOKENS: usize = 125_033;
const HIDDEN: usize = 2048;

/// The real config as the server hands it to the loader: `vocab_size` capped
/// to the tokenizer's count (`serve_phases::cap_vocab_size_to_tokenizer`).
fn capped() -> ModelConfig {
    let mut c = real();
    assert_eq!(c.embed_rows, TABLE_ROWS, "the parser records the table");
    c.vocab_size = TOKENIZER_TOKENS;
    c
}

fn with_embed(config: &ModelConfig, rows: usize) -> Vec<String> {
    problems_with(config, |n, f| {
        if n == EMBED {
            *f = Some((vec![rows, HIDDEN], WeightDtype::BF16));
        }
    })
}

#[test]
fn a_padded_embedding_is_accepted_after_the_tokenizer_cap() {
    // The queue-1 S10 failure: "model.embed_tokens.weight has shape
    // [128000, 2048], expected [125033, 2048]".
    let out = with_embed(&capped(), TABLE_ROWS);
    assert!(out.is_empty(), "{out:?}");
}

#[test]
fn a_tied_checkpoint_may_also_ship_a_padded_head() {
    let mut c = capped();
    c.tie_word_embeddings = false;
    let out = problems_with(&c, |n, f| {
        if n == names::LM_HEAD {
            *f = Some((vec![TABLE_ROWS, HIDDEN], WeightDtype::BF16));
        }
    });
    assert!(out.is_empty(), "{out:?}");
}

#[test]
fn an_embedding_that_is_not_the_configured_table_is_still_refused() {
    let c = capped();
    // The tokenizer's count is not the table's height: a table cut to it is
    // not the checkpoint the config describes.
    for rows in [TOKENIZER_TOKENS, 129_000, 127_999] {
        let out = with_embed(&c, rows);
        assert_eq!(out.len(), 1, "{rows}: {out:?}");
        assert!(
            out[0].contains(&format!(
                "model.embed_tokens.weight has shape [{rows}, 2048], expected [{TABLE_ROWS}, 2048]"
            )),
            "{rows}: {out:?}"
        );
    }
    let wrong_width = problems_with(&c, |n, f| {
        if n == EMBED {
            *f = Some((vec![TABLE_ROWS, 1024], WeightDtype::BF16));
        }
    });
    assert_eq!(wrong_width.len(), 1, "{wrong_width:?}");
    let bad_head = problems_with(&c, |n, f| {
        if n == names::LM_HEAD {
            *f = Some((vec![TOKENIZER_TOKENS, HIDDEN], WeightDtype::BF16));
        }
    });
    assert_eq!(bad_head.len(), 1, "{bad_head:?}");
    assert!(
        bad_head[0].contains("expected [128000, 2048]"),
        "{bad_head:?}"
    );
}

#[test]
fn the_served_vocabulary_is_the_tokenizers_and_the_table_is_the_checkpoints() {
    let g = Geometry::of(&capped());
    // Logits, sampling and masking walk `vocab` rows (a prefix of the table),
    // so the 2967 padding rows past the tokenizer can never be sampled.
    assert_eq!(g.vocab, TOKENIZER_TOKENS);
    assert_eq!(g.table_rows, TABLE_ROWS);
    assert!(g.table_rows - g.vocab > 0);
    // An uncapped config serves the whole table.
    let g = Geometry::of(&real());
    assert_eq!((g.vocab, g.table_rows), (TABLE_ROWS, TABLE_ROWS));
}

#[test]
fn a_served_vocabulary_past_the_table_is_refused() {
    let mut c = capped();
    c.vocab_size = TABLE_ROWS + 1;
    let out = problems_with(&c, |_, _| {});
    assert_eq!(out.len(), 1, "{out:?}");
    assert!(out[0].contains("exceeds the embedding table"), "{out:?}");
}

#[test]
fn a_config_with_no_recorded_table_falls_back_to_its_vocabulary() {
    let mut c = real();
    c.embed_rows = 0;
    c.vocab_size = TOKENIZER_TOKENS;
    assert_eq!(Geometry::of(&c).table_rows, TOKENIZER_TOKENS);
    // The old behavior, kept for a config that never recorded its table.
    let out = with_embed(&c, TABLE_ROWS);
    assert_eq!(out.len(), 1, "{out:?}");
}
