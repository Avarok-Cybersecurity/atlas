// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: what the DFlash drafter/target pairing check compares the
//! drafter's vocabulary against. It is the row count of the table the two
//! models share, read from the target's weight store, and never the served
//! vocabulary.
//!
//! # Why not `ModelConfig::vocab_size`
//!
//! A DFlash drafter ships no embedding and no LM head. It indexes the
//! target's `embed_tokens` with its own ids, and its tail GEMM walks
//! `vocab_size` rows of the target's LM head (`dflash_head/from_weights.rs`).
//! So the `vocab_size` its config states has to be the number of ROWS those
//! tensors have in the target's checkpoint: 248320 on Qwen3.5-4B. The rows
//! past the tokenizer are not dead to it either: the drafters put their mask
//! token there (id 248077 on the Qwen3.5-4B drafter, 248070 on the Qwen3.6
//! ones), on a row the tokenizer does not name.
//!
//! By the time a model is built, `ModelConfig::vocab_size` is a different
//! number. The server caps it to the tokenizer's count (248070, "Capping
//! vocab_size from 248320 to 248070 (tokenizer)") so the decode head skips
//! the padding rows. The pairing check compared the drafter against that
//! capped value and refused every correct drafter on a target whose table is
//! padded (an earlier GPU run on this card: "DFlash drafter vocabulary (248320) differs from
//! the target's 248070"). A drafter built for another vocabulary is still
//! refused: its count does not match the table's rows either.

use avarok_core::config::ModelConfig;
use spark_runtime::weights::WeightStore;

use crate::weight_loader::DflashConfig;

/// Keys an untied LM head ships under (the list `lm_head_setup` reads).
const LM_HEAD_KEYS: [&str; 3] = [
    "lm_head.weight",
    "language_model.lm_head.weight",
    "model.lm_head.weight",
];

/// The token embedding's name in every loader: this, bare or after a prefix
/// (`model.language_model.embed_tokens.weight` on Qwen3.5).
const EMBED_NAME: &str = "embed_tokens.weight";

fn is_embedding(name: &str) -> bool {
    name == EMBED_NAME
        || name
            .strip_suffix(EMBED_NAME)
            .is_some_and(|prefix| prefix.ends_with('.'))
}

/// Rows of a `[rows, hidden]` table; `None` for any other shape (a packed
/// NVFP4 head is not `[rows, hidden]` and is left to the embedding).
fn table_rows(shape: &[usize], hidden: usize) -> Option<usize> {
    match shape {
        [rows, width] if *width == hidden => Some(*rows),
        _ => None,
    }
}

/// A tensor's shape whether it is resident or was deferred at load.
fn shape_of<'a>(store: &'a WeightStore, name: &str) -> Option<&'a [usize]> {
    match store.get(name) {
        Ok(t) => Some(t.shape.as_slice()),
        Err(_) => store.deferred(name).map(|t| t.shape.as_slice()),
    }
}

/// Rows of the embedding / LM-head table the drafter shares with the target,
/// as the target's checkpoint ships it.
///
/// An untied LM head wins: it is the table the drafter's tail GEMM walks.
/// Otherwise the token embedding, which a tied head also is. Several tensors
/// of that name and the model's width must agree on the row count; where they
/// do not, nothing is guessed. `None` when the store shows no such table, an
/// ambiguous one, or one with fewer rows than the served vocabulary (a slice
/// of the table, not the table).
pub(super) fn shared_table_rows(store: &WeightStore, config: &ModelConfig) -> Option<usize> {
    let hidden = config.hidden_size;
    let lm_head = LM_HEAD_KEYS
        .iter()
        .find_map(|key| shape_of(store, key).and_then(|shape| table_rows(shape, hidden)));
    let rows = lm_head.or_else(|| {
        let deferred = store.deferred_sorted();
        let mut rows = store
            .names()
            .chain(deferred.iter().map(|(name, _)| name.as_str()))
            .filter(|name| is_embedding(name))
            .filter_map(|name| table_rows(shape_of(store, name)?, hidden));
        let first = rows.next()?;
        rows.all(|r| r == first).then_some(first)
    })?;
    (rows >= config.vocab_size).then_some(rows)
}

/// The pairing check ([`DflashConfig::validate_for_target`]) against the
/// target as it is being built: its hidden size, its layer count and the rows
/// of the table it shares with the drafter.
///
/// When the store shows no such table the served vocabulary stands in, with a
/// warning that names it: that comparison is the stricter one, and a refusal
/// it causes says why.
pub(super) fn check(
    drafter: &DflashConfig,
    config: &ModelConfig,
    store: &WeightStore,
) -> anyhow::Result<()> {
    let rows = shared_table_rows(store, config).unwrap_or_else(|| {
        tracing::warn!(
            "DFlash pairing check: the target's weight store has no embedding or LM-head table \
             [rows, {}] with at least {} rows; comparing the drafter's vocabulary against the \
             served vocabulary instead",
            config.hidden_size,
            config.vocab_size,
        );
        config.vocab_size
    });
    drafter.validate_for_target(config.hidden_size, config.num_hidden_layers, rows)
}

#[cfg(test)]
#[path = "dflash_pairing_tests.rs"]
mod tests;
