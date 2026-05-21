// SPDX-License-Identifier: AGPL-3.0-only
//
// Tests for `TokenizerInfo` — ported / adapted from
// `tests/python/test_tokenizer_info.py`. The Python tests load real
// HuggingFace tokenizers; we instead exercise the same behaviors with
// hand-built vocabularies so the tests run with no network / model
// downloads, while still covering every code path.

use super::*;
use crate::tokenizer::decoder::byte_to_char_map;

/// Encode a literal byte string into a ByteLevel vocab string by
/// applying the GPT-2 bytes-to-unicode forward map.
fn byte_level_encode(bytes: &[u8]) -> String {
    let map = byte_to_char_map();
    bytes
        .iter()
        .map(|&b| char::from_u32(map[b as usize]).unwrap())
        .collect()
}

#[test]
fn raw_vocab_literal_decoding() {
    let vocab = vec!["+".to_string(), "regular".to_string(), "我".to_string()];
    let info = TokenizerInfo::new(&vocab, VocabType::Raw, None, None, false);
    assert_eq!(info.vocab_type(), VocabType::Raw);
    assert_eq!(info.vocab_size(), 3);
    assert_eq!(info.decoded_vocab()[0], b"+");
    assert_eq!(info.decoded_vocab()[1], b"regular");
    assert_eq!(info.decoded_vocab()[2], "我".as_bytes());
}

#[test]
fn byte_fallback_vocab_decoding() {
    // <0x01> -> byte, "▁▁" -> "  ", "er" literal.
    let vocab = vec![
        "<0x01>".to_string(),
        "\u{2581}\u{2581}".to_string(),
        "er".to_string(),
        "\u{2581}hello".to_string(),
    ];
    let info = TokenizerInfo::new(&vocab, VocabType::ByteFallback, None, None, true);
    assert_eq!(info.decoded_vocab()[0], vec![0x01]);
    assert_eq!(info.decoded_vocab()[1], b"  ");
    assert_eq!(info.decoded_vocab()[2], b"er");
    assert_eq!(info.decoded_vocab()[3], b" hello");
    assert!(info.add_prefix_space());
}

#[test]
fn byte_level_vocab_decoding() {
    // Ported from test_vocab_conversion's byte_level case.
    let vocab = vec![
        byte_level_encode(b"\""),
        byte_level_encode("我".as_bytes()),
        byte_level_encode(b" automotive"),
    ];
    let info = TokenizerInfo::new(&vocab, VocabType::ByteLevel, None, None, false);
    assert_eq!(info.decoded_vocab()[0], b"\"");
    assert_eq!(info.decoded_vocab()[1], "我".as_bytes());
    assert_eq!(info.decoded_vocab()[2], b" automotive");
}

#[test]
fn empty_vocab() {
    let vocab: Vec<String> = vec![];
    let info = TokenizerInfo::new(&vocab, VocabType::Raw, None, None, false);
    assert_eq!(info.vocab_size(), 0);
    assert!(info.decoded_vocab().is_empty());
    assert!(info.sorted_decoded_vocab().is_empty());
    assert!(info.stop_token_ids().is_empty());
    assert!(info.special_token_ids().is_empty());
}

#[test]
fn special_token_detection() {
    // Ported from test_special_token_detection: only "" is special.
    let vocab: Vec<String> = [
        "", "<s>", "</s>", "[@BOS@]", "regular", "<>", "<think>", "</think>",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    // Explicit stop ids = [2] so "</s>" is a stop, not auto-detected.
    let info = TokenizerInfo::new(
        &vocab,
        VocabType::ByteFallback,
        Some(8),
        Some(vec![2]),
        true,
    );
    assert_eq!(info.special_token_ids(), &[0]);
    assert_eq!(info.stop_token_ids(), &[2]);
}

#[test]
fn auto_detected_stop_tokens() {
    // With no explicit stop ids, well-known markers are auto-detected.
    let vocab: Vec<String> = ["hello", "</s>", "world", "<|eot_id|>", "<|endoftext|>"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let info = TokenizerInfo::new(&vocab, VocabType::Raw, None, None, false);
    assert_eq!(info.stop_token_ids(), &[1, 3, 4]);
}

#[test]
fn explicit_stop_tokens_override_detection() {
    // "</s>" is a detection marker but explicit ids win.
    let vocab: Vec<String> = ["a", "</s>", "b"].iter().map(|s| s.to_string()).collect();
    let info = TokenizerInfo::new(&vocab, VocabType::Raw, None, Some(vec![0, 2]), false);
    // id 1 ("</s>") is NOT a stop because it is not in the explicit set.
    assert_eq!(info.stop_token_ids(), &[0, 2]);
}

#[test]
fn padding_vocab_size_creates_special_tokens() {
    // Ported from test_padding_vocab_size.
    let vocab: Vec<String> = (0..10).map(|i| format!("tok{i}")).collect();
    let info = TokenizerInfo::new(&vocab, VocabType::Raw, Some(15), None, false);
    assert_eq!(info.vocab_size(), 15);
    // Trailing 5 ids are padding special tokens.
    assert_eq!(
        &info.special_token_ids()[info.special_token_ids().len() - 5..],
        &[10, 11, 12, 13, 14]
    );
    // decoded_vocab is padded to full length with empty byte strings.
    assert_eq!(info.decoded_vocab().len(), 15);
    assert!(info.decoded_vocab()[14].is_empty());
}

#[test]
fn model_vocab_size_smaller_than_tokenizer() {
    // Ported from test_model_vocab_size_smaller_than_tokenizer.
    let vocab: Vec<String> = (0..20).map(|i| format!("tok{i}")).collect();
    let info = TokenizerInfo::new(&vocab, VocabType::Raw, Some(12), None, false);
    assert_eq!(info.vocab_size(), 12);
    // decoded_vocab truncates to vocab_size when it is smaller.
    assert_eq!(info.decoded_vocab().len(), 12);
}

#[test]
fn sorted_decoded_vocab_is_lexicographic() {
    let vocab: Vec<String> = ["zebra", "apple", "mango", "banana"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let info = TokenizerInfo::new(&vocab, VocabType::Raw, None, None, false);
    let sorted: Vec<&[u8]> = info
        .sorted_decoded_vocab()
        .iter()
        .map(|(_, t)| t.as_slice())
        .collect();
    assert_eq!(sorted, vec![&b"apple"[..], b"banana", b"mango", b"zebra"]);
}

#[test]
fn sorted_excludes_stop_and_special() {
    let vocab: Vec<String> = ["", "</s>", "regular", "another"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let info = TokenizerInfo::new(&vocab, VocabType::Raw, None, None, false);
    // "" is special (id 0), "</s>" is stop (id 1) — neither sorted.
    let ids: Vec<i32> = info
        .sorted_decoded_vocab()
        .iter()
        .map(|(i, _)| *i)
        .collect();
    assert_eq!(ids.len(), 2);
    assert!(!ids.contains(&0));
    assert!(!ids.contains(&1));
}

#[test]
fn trie_ranges_length_matches_sorted() {
    let vocab: Vec<String> = ["a", "ab", "abc", "b", "bc"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let info = TokenizerInfo::new(&vocab, VocabType::Raw, None, None, false);
    assert_eq!(
        info.trie_subtree_nodes_range().len(),
        info.sorted_decoded_vocab().len()
    );
    // Each subtree end is within bounds and >= its own index.
    for (i, &end) in info.trie_subtree_nodes_range().iter().enumerate() {
        assert!(end as usize > i);
        assert!(end as usize <= info.sorted_decoded_vocab().len());
    }
}

#[test]
fn trie_prefix_subtree_nesting() {
    // Sorted: "a","ab","abc". "a" contains nothing before it; "ab"
    // contains "a"? find("a" in "ab") -> yes -> nested. "abc" contains
    // "ab" -> nested. So subtree of entry 0 spans all 3.
    let vocab: Vec<String> = ["a", "ab", "abc"].iter().map(|s| s.to_string()).collect();
    let info = TokenizerInfo::new(&vocab, VocabType::Raw, None, None, false);
    let ranges = info.trie_subtree_nodes_range();
    assert_eq!(ranges, &[3, 3, 3]);
}

#[test]
fn dump_metadata_format() {
    // Ported from test_dump_metadata_load metadata strings.
    let vocab: Vec<String> = (0..10).map(|i| format!("tok{i}")).collect();
    let info = TokenizerInfo::new(
        &vocab,
        VocabType::ByteFallback,
        Some(32000),
        Some(vec![2]),
        true,
    );
    assert_eq!(
        info.dump_metadata(),
        r#"{"vocab_type":1,"vocab_size":32000,"add_prefix_space":true,"stop_token_ids":[2]}"#
    );
}

#[test]
fn dump_metadata_raw_no_stops() {
    let vocab: Vec<String> = vec!["a".to_string()];
    let info = TokenizerInfo::new(&vocab, VocabType::Raw, Some(100352), Some(vec![]), false);
    assert_eq!(
        info.dump_metadata(),
        r#"{"vocab_type":0,"vocab_size":100352,"add_prefix_space":false,"stop_token_ids":[]}"#
    );
}

#[test]
fn dump_metadata_byte_level() {
    // Stop id must fall within the encoded vocab to be recorded — the
    // C++ only flags stop ids while iterating actual vocab entries.
    let vocab: Vec<String> = (0..3).map(|i| format!("tok{i}")).collect();
    let info = TokenizerInfo::new(
        &vocab,
        VocabType::ByteLevel,
        Some(128256),
        Some(vec![2]),
        false,
    );
    assert_eq!(
        info.dump_metadata(),
        r#"{"vocab_type":2,"vocab_size":128256,"add_prefix_space":false,"stop_token_ids":[2]}"#
    );
}

#[test]
fn from_huggingface_byte_level() {
    // Qwen-style: ByteLevel decoder, no prefix space (F68-critical).
    let json = r#"{"decoder":{"type":"ByteLevel"}}"#;
    let vocab = vec![byte_level_encode(b" hi"), byte_level_encode(b"world")];
    let info = TokenizerInfo::from_huggingface(&vocab, json, None, Some(vec![1])).unwrap();
    assert_eq!(info.vocab_type(), VocabType::ByteLevel);
    assert!(!info.add_prefix_space());
    assert_eq!(info.decoded_vocab()[0], b" hi");
    assert_eq!(info.stop_token_ids(), &[1]);
}

#[test]
fn from_huggingface_byte_fallback_with_prefix() {
    // LLaMA-2 style: ByteFallback + Prepend "▁" normalizer.
    let json = r#"{"decoder":{"type":"ByteFallback"},
        "normalizer":{"type":"Prepend","prepend":"▁"}}"#;
    let vocab = vec!["<0x0A>".to_string(), "er".to_string()];
    let info = TokenizerInfo::from_huggingface(&vocab, json, None, None).unwrap();
    assert_eq!(info.vocab_type(), VocabType::ByteFallback);
    assert!(info.add_prefix_space());
    assert_eq!(info.decoded_vocab()[0], vec![0x0A]);
}

#[test]
fn from_huggingface_invalid_json_is_err() {
    let vocab = vec!["a".to_string()];
    assert!(TokenizerInfo::from_huggingface(&vocab, "{bad", None, None).is_err());
}

#[test]
fn from_huggingface_unknown_decoder_falls_back_to_raw() {
    let json = r#"{"decoder":{"type":"WordPiece"}}"#;
    let vocab = vec!["hello".to_string()];
    let info = TokenizerInfo::from_huggingface(&vocab, json, None, None).unwrap();
    assert_eq!(info.vocab_type(), VocabType::Raw);
}

#[test]
fn clone_and_eq() {
    let vocab: Vec<String> = vec!["a".to_string(), "b".to_string()];
    let info = TokenizerInfo::new(&vocab, VocabType::Raw, None, None, false);
    let cloned = info.clone();
    assert_eq!(info, cloned);
}

#[test]
fn detect_metadata_standalone() {
    // The free function is re-exported and usable directly.
    let json = r#"{"decoder":{"type":"ByteLevel"}}"#;
    let meta = crate::tokenizer::detect_metadata_from_hf(json).unwrap();
    assert_eq!(meta.vocab_type, VocabType::ByteLevel);
    assert_eq!(
        meta.to_json(),
        r#"{"vocab_type":2,"add_prefix_space":false}"#
    );
}

#[test]
fn deepseek_style_stop_marker_detected() {
    // DeepSeek-V2 end-of-sentence marker is in the detection set.
    let marker = "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>";
    let vocab: Vec<String> = ["a", marker, "b"].iter().map(|s| s.to_string()).collect();
    let info = TokenizerInfo::new(&vocab, VocabType::Raw, None, None, false);
    assert_eq!(info.stop_token_ids(), &[1]);
}
