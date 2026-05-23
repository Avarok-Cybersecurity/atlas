// SPDX-License-Identifier: AGPL-3.0-only
//
// Tokenizer subsystem — port of `cpp/tokenizer_info.{cc,h}` and
// `tokenizer_info_impl.h`.
//
// Provides the decoded-vocabulary metadata the grammar matcher needs to
// build and mask the logit bitmask. This is the exact public surface
// Atlas's grammar engine consumes:
//   `TokenizerInfo::new(vocab, VocabType, vocab_size, stops, prefix)`
//   `TokenizerInfo::from_huggingface(vocab, tokenizer_json, ..)`
//   `detect_metadata_from_hf(tokenizer_json)`
//
// Module map:
//   vocab_type   — VocabType enum (Raw / ByteFallback / ByteLevel)
//   decoder      — raw-token -> byte-sequence decoders, GPT-2 byte map
//   hf_metadata  — HuggingFace tokenizer.json metadata detection
//   info         — TokenizerInfo: the decoded-vocabulary view

pub mod decoder;
pub mod hf_metadata;
pub mod info;
pub mod vocab_type;

pub use decoder::{byte_to_char_map, char_to_byte_map, decode_token};
pub use hf_metadata::{HfMetadata, detect_metadata_from_hf};
pub use info::TokenizerInfo;
pub use vocab_type::VocabType;
