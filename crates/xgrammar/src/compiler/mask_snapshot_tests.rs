// SPDX-License-Identifier: AGPL-3.0-only
//
// #918 — cross-process persistence of the Tier-2 rule-level mask cache.

use std::time::{Duration, Instant};

use super::*;
use crate::compiler::GrammarCompiler;
use crate::tokenizer::{TokenizerInfo, VocabType};

/// A deterministic synthetic vocabulary big enough that mask generation
/// is the dominant cost (so the cold/warm ratio below is meaningful),
/// small enough that the cold leg stays well under a second even in a
/// debug `cargo test`. Real deployments are 30-60x wider — Qwen3.6-35B
/// is 248,320 tokens — so this test asserts a RATIO, never a time.
fn synthetic_vocab(n: usize) -> Vec<String> {
    let alphabet: Vec<char> = (b' '..=b'~').map(|c| c as char).collect();
    let mut vocab: Vec<String> = alphabet.iter().map(|c| c.to_string()).collect();
    let mut i = 0usize;
    while vocab.len() < n {
        let a = alphabet[i % alphabet.len()];
        let b = alphabet[(i / alphabet.len()) % alphabet.len()];
        let c = alphabet[(i / (alphabet.len() * alphabet.len())) % alphabet.len()];
        vocab.push(format!("{a}{b}{c}"));
        i += 1;
    }
    vocab.truncate(n);
    vocab
}

fn tokenizer(n: usize) -> TokenizerInfo {
    TokenizerInfo::new(&synthetic_vocab(n), VocabType::Raw, None, None, false)
}

/// No cap in the unit tests — every warmed mask is persisted.
const MAX: usize = usize::MAX;

fn compiler(info: &TokenizerInfo) -> GrammarCompiler {
    // 1 worker + caching on — the serve default restored by `37228e85`.
    GrammarCompiler::new(info.clone(), 1, true, 1024 * 1024 * 1024)
}

/// The coherency gate's `get_weather` arguments (#918), as a bare JSON
/// schema so this test needs nothing from `spark-server`.
const WEATHER_SCHEMA: &str = r#"{"type":"object","properties":{
    "city":{"type":"string"},"days":{"type":"integer"}},
    "required":["city","days"]}"#;

/// A schema this process has never compiled: different field names,
/// different types, one extra property.
const SEARCH_SCHEMA: &str = r#"{"type":"object","properties":{
    "query":{"type":"string"},"limit":{"type":"integer"},
    "fuzzy":{"type":"boolean"}},
    "required":["query","limit"]}"#;

fn compile_and_prewarm(c: &GrammarCompiler, schema: &str) -> (Duration, usize) {
    let started = Instant::now();
    let compiled = c
        .compile_json_schema(schema, true, None, None, true, Some(8))
        .expect("schema compiles");
    let masks = compiled.compile_top_k_masks(512);
    (started.elapsed(), masks)
}

/// Take the best of `reps` — the cheapest run is the one least polluted
/// by scheduler noise on a shared CI box, and the claim ("the warm leg
/// is much cheaper") is one-sided.
fn best_of<F: FnMut() -> Duration>(reps: usize, mut f: F) -> Duration {
    (0..reps).map(|_| f()).min().expect("reps > 0")
}

#[test]
fn a_snapshot_round_trips_every_mask_byte_for_byte() {
    let info = tokenizer(2048);
    let c = compiler(&info);
    compile_and_prewarm(&c, WEATHER_SCHEMA);
    let identity = c.snapshot_identity();
    let original = c.rule_cache().expect("cache enabled").entries();
    assert!(!original.is_empty(), "prewarm populated no rule masks");

    let decoded = decode(&encode(identity, &original), identity).expect("round trip decodes");
    assert_eq!(decoded.len(), original.len());
    for ((k_a, m_a), (k_b, m_b)) in original.iter().zip(decoded.iter()) {
        assert_eq!(k_a, k_b, "rule key changed across the snapshot");
        assert_eq!(**m_a, **m_b, "mask changed across the snapshot");
    }
}

#[test]
fn a_snapshot_written_for_another_tokenizer_is_a_miss() {
    let info = tokenizer(1024);
    let c = compiler(&info);
    compile_and_prewarm(&c, WEATHER_SCHEMA);
    let entries = c.rule_cache().unwrap().entries();
    let bytes = encode(c.snapshot_identity(), &entries);

    // Same vocabulary size, different contents.
    let other = TokenizerInfo::new(
        &synthetic_vocab(1024)
            .iter()
            .map(|t| format!("x{t}"))
            .collect::<Vec<_>>(),
        VocabType::Raw,
        None,
        None,
        false,
    );
    assert_ne!(info.fingerprint(), other.fingerprint());
    let foreign = SnapshotIdentity {
        tokenizer_fingerprint: other.fingerprint(),
        vocab_size: other.vocab_size(),
    };
    assert!(
        decode(&bytes, foreign).is_none(),
        "a snapshot from a different tokenizer must not be reused"
    );
    // ...and the same tokenizer with a different vocab_size cap is also
    // a miss: mask indices are positions in the sorted decoded vocab.
    assert!(
        decode(
            &bytes,
            SnapshotIdentity {
                vocab_size: info.vocab_size() + 1,
                ..c.snapshot_identity()
            }
        )
        .is_none()
    );
}

#[test]
fn a_corrupt_or_truncated_snapshot_is_a_miss_not_a_wrong_mask() {
    let info = tokenizer(1024);
    let c = compiler(&info);
    compile_and_prewarm(&c, WEATHER_SCHEMA);
    let identity = c.snapshot_identity();
    let bytes = encode(identity, &c.rule_cache().unwrap().entries());

    assert!(
        decode(&bytes, identity).is_some(),
        "control: intact decodes"
    );
    for cut in [0usize, 1, 9, bytes.len() / 2, bytes.len() - 1] {
        assert!(
            decode(&bytes[..cut], identity).is_none(),
            "truncation at {cut} must be a miss"
        );
    }
    let mut flipped = bytes.clone();
    let mid = flipped.len() / 2;
    flipped[mid] ^= 0x01;
    assert!(
        decode(&flipped, identity).is_none(),
        "a single flipped body byte must fail the checksum"
    );
}

#[test]
fn loading_a_snapshot_removes_the_cold_mask_prewarm() {
    let dir = std::env::temp_dir().join(format!("xgrammar-918-{}", std::process::id()));
    let path = dir.join("masks.bin");
    let _ = std::fs::remove_file(&path);
    let info = tokenizer(2048);

    // COLD: a fresh compiler with an empty rule cache.
    let cold = best_of(3, || {
        let c = compiler(&info);
        compile_and_prewarm(&c, WEATHER_SCHEMA).0
    });
    let source = compiler(&info);
    let (_, masks) = compile_and_prewarm(&source, WEATHER_SCHEMA);
    assert!(masks > 0);
    let written = source
        .save_mask_snapshot(&path, MAX)
        .expect("snapshot writes");
    assert!(written > 0, "nothing persisted");

    // WARM: a fresh compiler seeded from the snapshot only.
    let mut imported = 0;
    let warm = best_of(3, || {
        let c = compiler(&info);
        imported = c.load_mask_snapshot(&path).expect("snapshot loads");
        compile_and_prewarm(&c, WEATHER_SCHEMA).0
    });
    assert_eq!(imported, written, "every persisted mask should import");

    // Byte-exactness first — a fast wrong answer is not a fix.
    let fresh = compiler(&info);
    let (fresh_masks, warm_masks) = (
        {
            compile_and_prewarm(&fresh, WEATHER_SCHEMA);
            fresh
                .compile_json_schema(WEATHER_SCHEMA, true, None, None, true, Some(8))
                .unwrap()
        },
        {
            let c = compiler(&info);
            c.load_mask_snapshot(&path).unwrap();
            compile_and_prewarm(&c, WEATHER_SCHEMA);
            c.compile_json_schema(WEATHER_SCHEMA, true, None, None, true, Some(8))
                .unwrap()
        },
    );
    assert_eq!(
        *fresh_masks.inner().mask_cache.lock().unwrap(),
        *warm_masks.inner().mask_cache.lock().unwrap(),
        "snapshot-warmed masks differ from freshly computed ones"
    );

    // Hit/miss oracle alongside the ratio: the warm compiler must have
    // served its masks from the imported snapshot, not recomputed them.
    let seeded = compiler(&info);
    assert_eq!(seeded.rule_cache().unwrap().hit_miss(), (0, 0));
    seeded.load_mask_snapshot(&path).unwrap();
    compile_and_prewarm(&seeded, WEATHER_SCHEMA);
    let (hits, misses) = seeded.rule_cache().unwrap().hit_miss();
    assert!(
        hits > 0 && hits > misses,
        "snapshot hits={hits} misses={misses}"
    );

    // The measured M5 Max ratio on the real Qwen3 vocabulary was
    // 596.7 ms cold vs 0.0-16.2 ms warm (issue #918). A 4x floor is a
    // deliberately loose gate for a 2,048-token synthetic vocabulary on
    // a contended CI box; it still fails outright if the snapshot stops
    // being consulted.
    assert!(
        cold > warm * 4,
        "snapshot did not cut cold prewarm: cold={cold:?} warm={warm:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_snapshot_also_warms_a_schema_it_was_never_compiled_for() {
    // The finding behind #918: the rule cache is keyed STRUCTURALLY, so
    // a schema the snapshot never saw still reuses its string / number /
    // whitespace / punctuation sub-rule masks. A snapshot keyed by
    // (tokenizer, schema) — the shape #918 suggests — would miss exactly
    // this. Asserted on the HIT/MISS counters rather than wall clock:
    // the effect size varies with how much structure two schemas share
    // (observed 2.7% of the cold cost on the qwen3_coder tool-grammar
    // path, ~70% here on a bare JSON schema with renamed properties and
    // an extra type), but "the snapshot was consulted and hit" is exact.
    let dir = std::env::temp_dir().join(format!("xgrammar-918x-{}", std::process::id()));
    let path = dir.join("masks.bin");
    let _ = std::fs::remove_file(&path);
    let info = tokenizer(1024);

    let cold = compiler(&info);
    compile_and_prewarm(&cold, SEARCH_SCHEMA);
    let (cold_hits, _) = cold.rule_cache().unwrap().hit_miss();

    let source = compiler(&info);
    compile_and_prewarm(&source, WEATHER_SCHEMA); // a DIFFERENT schema
    source
        .save_mask_snapshot(&path, MAX)
        .expect("snapshot writes");

    let warm = compiler(&info);
    assert!(warm.load_mask_snapshot(&path).unwrap() > 0);
    compile_and_prewarm(&warm, SEARCH_SCHEMA);
    let (warm_hits, warm_misses) = warm.rule_cache().unwrap().hit_miss();

    assert!(
        warm_hits > cold_hits,
        "the imported snapshot was never hit for the unseen schema: \
         warm={warm_hits} hits / {warm_misses} misses, cold={cold_hits} hits"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_cache_disabled_compiler_neither_loads_nor_saves() {
    let info = tokenizer(256);
    let c = GrammarCompiler::new(info.clone(), 1, false, 1024 * 1024);
    let path = std::env::temp_dir().join("xgrammar-918-never-written.bin");
    let _ = std::fs::remove_file(&path);
    assert_eq!(c.save_mask_snapshot(&path, MAX).unwrap(), 0);
    assert_eq!(c.load_mask_snapshot(&path).unwrap(), 0);
    assert!(!path.exists(), "a disabled cache must not write a file");
    assert_eq!(c.rule_cache_len(), 0);
}

#[test]
fn a_missing_snapshot_file_is_a_miss_not_an_error() {
    let info = tokenizer(256);
    let c = compiler(&info);
    let path = std::env::temp_dir().join("xgrammar-918-absent-file.bin");
    let _ = std::fs::remove_file(&path);
    assert_eq!(c.load_mask_snapshot(&path).unwrap(), 0);
}

#[test]
fn a_compiled_grammar_can_cross_a_thread_boundary() {
    // #918 candidate (4) moves `compile_top_k_masks` onto a background
    // thread so it overlaps prefill; that needs `CompiledGrammar: Send`.
    fn assert_send<T: Send + 'static>() {}
    assert_send::<crate::compiler::CompiledGrammar>();
}
