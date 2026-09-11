// SPDX-License-Identifier: AGPL-3.0-only

//! #918 — cold grammar preparation for a new tool schema.
//!
//! The phase split this module measures (M5 Max, `--release`, Qwen3
//! ByteLevel-BPE tokenizer with 151,669 tokens, the coherency gate's
//! `get_weather` schema through `compile_qwen3_coder_tool_grammar` —
//! reproduce with
//! `QWEN_TOKENIZER_JSON=... cargo test --release ... phase_timings -- --ignored --nocapture`):
//!
//! ```text
//! engine build (TokenizerInfo)               101.9-128.0 ms   once per process
//! grammar construction                           4.9-5.5 ms   per schema
//! top-k mask prewarm, first grammar         596.7-621.2 ms   <-- the cold cost
//! top-k mask prewarm, same schema again              0.0 ms
//! top-k mask prewarm, a DIFFERENT schema       14.2-16.2 ms
//! matcher construction + first mask fill         0.1-0.3 ms
//! ```
//!
//! End to end, "tool schema in hand" to "first constrained mask filled",
//! same box and build, n=3 (`cold_vs_snapshot_warm_with_a_real_tokenizer`):
//!
//! ```text
//! cold, nothing on disk                            624.8 ms
//!   of which the request thread is held               4.9 ms
//! warm from an on-disk snapshot, same schema     5.2-5.3 ms
//! warm from an on-disk snapshot, unseen schema  5.0-19.1 ms
//! ```
//!
//! Two facts drive the fix: the prewarm is ~99% of the cold path, and it
//! is paid once per PROCESS, not once per schema — a second, unrelated
//! schema costs ~2.5% of the first because xgrammar's Tier-2 rule cache
//! keys masks structurally. So: persist that cache across processes
//! (`super::super::mask_cache`) and overlap what is left with prefill
//! (`super::super::prewarm`).

use std::time::{Duration, Instant};

use super::*;
use crate::grammar::GrammarState;

/// A deterministic synthetic vocabulary: single printable ASCII plus
/// three-character combinations, then the tool-call control tokens the
/// qwen3_coder grammar needs. Wide enough that mask generation dominates
/// (so the ratios below mean something), narrow enough to stay fast in a
/// debug `cargo test`. Production vocabularies are 100-250x wider, which
/// is why every assertion here is a RATIO and never a time.
fn wide_vocab(n: usize) -> Vec<String> {
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
    vocab.extend(["<tool_call>", "</tool_call>", "<eos>"].map(String::from));
    vocab
}

const VOCAB: usize = 1024;

fn engine() -> GrammarEngine {
    let vocab = wide_vocab(VOCAB);
    let eos = (vocab.len() - 1) as i32;
    GrammarEngine::new(&vocab, &[eos]).expect("engine builds")
}

/// The coherency gate's tool, and a second tool that shares no name.
fn tool(name: &str, a: &str, b: &str) -> Vec<ToolDefinition> {
    serde_json::from_value(serde_json::json!([{
        "type": "function",
        "function": {
            "name": name,
            "description": "Look up the current weather for a city.",
            "parameters": {"type": "object", "properties": {
                a: {"type": "string", "description": "City name."},
                b: {"type": "integer", "description": "Forecast horizon in days."}},
                "required": [a, b]},
        }
    }]))
    .unwrap()
}

/// What one request pays before its first constrained token, split at
/// the boundary the fix acts on.
struct Prepared {
    /// Schema -> EBNF -> parse -> normalize -> optimize -> decompose.
    /// Unaffected by #918 and, on a narrow synthetic vocabulary in a
    /// debug build, large enough to swamp the phase under test — which
    /// is why the ratios below are asserted on `masks`, not on the sum.
    construct: Duration,
    /// Matcher construction + the top-k mask prewarm + the first
    /// constrained fill (which joins the overlapped prewarm). This is
    /// the ~99% of the cold path #918 is about.
    masks: Duration,
    state: GrammarState,
}

fn prepare(engine: &mut GrammarEngine, tools: &[ToolDefinition]) -> Prepared {
    let started = Instant::now();
    let compiled = engine
        .compile_qwen3_coder_tool_grammar(tools, true, "</parameter>")
        .expect("tool grammar compiles");
    let construct = started.elapsed();

    let started = Instant::now();
    let hook = engine.mask_snapshot_hook();
    let mut state =
        GrammarState::new_with_hook(&compiled, engine.vocab_size(), hook).expect("grammar state");
    // The first constrained sample — this is what joins the overlapped
    // prewarm, so it is the honest end of the mask phase.
    state.fill_bitmask();
    Prepared {
        construct,
        masks: started.elapsed(),
        state,
    }
}

/// Size of the snapshot file under `dir`, once it appears.
///
/// The write runs on its own thread (so a request's first constrained
/// fill never waits for I/O — see `mask_cache::mask_snapshot_hook`), so
/// the test polls instead of assuming the file is already there.
fn await_snapshot(dir: &std::path::Path) -> Option<u64> {
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        if let Ok(entries) = std::fs::read_dir(dir.join(".atlas-grammar-cache")) {
            for entry in entries.filter_map(Result::ok) {
                if entry.file_name().to_string_lossy().starts_with("masks-") {
                    let len = entry.metadata().ok()?.len();
                    if len > 0 {
                        return Some(len);
                    }
                }
            }
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    None
}

fn scratch(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "atlas-918-{tag}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

#[test]
fn a_second_distinct_schema_reuses_the_first_schemas_masks() {
    // The measurement that redirects #918 from "per schema" to "per
    // process": on the real Qwen3 vocabulary an unrelated second schema
    // cost 16.2 ms against the first's 596.7 ms, because xgrammar's
    // Tier-2 rule cache keys masks STRUCTURALLY and every JSON tool
    // schema shares its string / number / whitespace / punctuation
    // sub-rules.
    //
    // Asserted on the cache's hit/miss COUNTERS, not on wall clock. The
    // size of the saving is cost-weighted, not count-weighted: the
    // shared rules are the broad free-form value scanners (one mask
    // each, most of the time), while the misses are cheap per-schema
    // literal keys. Observed here: 31 hits / 94 misses yet 1.5x faster
    // at a 1,027-token vocabulary; 37x faster at 151,669, where those
    // value scanners are almost the whole bill. The invariant worth
    // gating is therefore "a schema this engine never saw still reuses
    // masks at all" — which goes to zero the moment the cross-grammar
    // cache is keyed per schema, disabled, or its FSM hashing breaks.
    let mut engine = engine();
    let cache = engine
        .compiler
        .rule_cache_handle()
        .expect("serve compilers enable the rule cache");

    prepare(&mut engine, &tool("get_weather", "city", "days"));
    let (hits_after_first, misses_after_first) = cache.hit_miss();
    assert!(
        misses_after_first > 0,
        "the first schema should have computed masks, not found them"
    );

    prepare(&mut engine, &tool("search_docs", "query", "limit"));
    let (hits, misses) = cache.hit_miss();
    let (new_hits, new_misses) = (hits - hits_after_first, misses - misses_after_first);
    assert!(
        new_hits > 0,
        "cross-schema mask reuse lost: the second schema hit {new_hits} / missed {new_misses}"
    );
}

#[test]
fn a_persisted_snapshot_removes_the_cold_prewarm_for_the_next_process() {
    let dir = scratch("hit");
    let tools = tool("get_weather", "city", "days");

    // "Process" 1: nothing on disk. Pays the cold path, then the
    // background prewarm persists the cross-grammar masks.
    let mut first = engine();
    first.attach_mask_cache(&dir);
    let cold = prepare(&mut first, &tools);
    let snapshot = await_snapshot(&dir).expect("snapshot written by the background prewarm");
    assert!(snapshot > 0, "snapshot file is empty");

    // "Process" 2: same model directory, fresh engine.
    let mut second = engine();
    second.attach_mask_cache(&dir);
    let warm = prepare(&mut second, &tools);

    // Correctness before speed: the warm path must produce the same
    // first-token mask, bit for bit.
    assert_eq!(
        cold.state.bitmask_data(),
        warm.state.bitmask_data(),
        "snapshot-warmed grammar admits a different first token set"
    );
    // Measured 621.2 ms -> ~0.3 ms on the real Qwen3 vocabulary; a 3x
    // floor is the loose version for a 1,027-token synthetic vocabulary
    // in a debug build on a contended box.
    assert!(
        cold.masks > warm.masks * 3,
        "the snapshot did not remove the cold prewarm: cold={:?} warm={:?} \
         (construct {:?} / {:?})",
        cold.masks,
        warm.masks,
        cold.construct,
        warm.construct,
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_snapshot_from_a_different_tokenizer_is_a_miss() {
    let dir = scratch("miss");
    let tools = tool("get_weather", "city", "days");

    let mut writer = engine();
    writer.attach_mask_cache(&dir);
    prepare(&mut writer, &tools);
    await_snapshot(&dir).expect("the writing engine persisted its masks");

    // A different vocabulary: same count, different bytes. Its
    // fingerprint differs, so it must not adopt the other's masks —
    // it looks for (and finds) no file of its own.
    let other_vocab: Vec<String> = wide_vocab(VOCAB).iter().map(|t| format!("z{t}")).collect();
    let eos = (other_vocab.len() - 1) as i32;
    let mut reader = GrammarEngine::new(&other_vocab, &[eos]).expect("engine builds");
    reader.attach_mask_cache(&dir);
    let names: Vec<String> = std::fs::read_dir(dir.join(".atlas-grammar-cache"))
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        names.len(),
        1,
        "a second tokenizer must not reuse the first's file: {names:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The benchmark-style harness that produced the phase table above.
/// Ignored by default: it needs a real 100K+ tokenizer on disk and is a
/// diagnostic, not a gate.
#[test]
#[ignore = "CPU diagnostic: set QWEN_TOKENIZER_JSON to an existing tokenizer.json"]
fn phase_timings_with_a_real_tokenizer() {
    let path = std::env::var("QWEN_TOKENIZER_JSON").expect("QWEN_TOKENIZER_JSON");
    let tokenizer = tokenizers::Tokenizer::from_file(path).expect("tokenizer loads");
    let stop = tokenizer.token_to_id("<|im_end|>").expect("<|im_end|>") as i32;
    let started = Instant::now();
    let mut engine =
        GrammarEngine::from_tokenizer(&tokenizer, None, &[stop]).expect("engine builds");
    println!(
        "vocab={} engine_build_ms={:.1}",
        engine.vocab_size(),
        started.elapsed().as_secs_f64() * 1000.0
    );
    let cases = [
        (
            "get_weather(city, days)",
            tool("get_weather", "city", "days"),
        ),
        (
            "get_weather(city, days) again",
            tool("get_weather", "city", "days"),
        ),
        (
            "search_docs(query, limit) NEW",
            tool("search_docs", "query", "limit"),
        ),
        (
            "run_cmd(command, timeout) NEW",
            tool("run_cmd", "command", "timeout"),
        ),
    ];
    for (label, tools) in cases {
        let t = Instant::now();
        let compiled = engine
            .compile_qwen3_coder_tool_grammar(&tools, true, "</parameter>")
            .expect("tool grammar compiles");
        let construct = t.elapsed().as_secs_f64() * 1000.0;
        let t = Instant::now();
        let masks = compiled.compile_top_k_masks(512);
        let prewarm = t.elapsed().as_secs_f64() * 1000.0;
        let t = Instant::now();
        let mut state = GrammarState::new(&compiled, engine.vocab_size()).expect("grammar state");
        state.fill_bitmask();
        let first_fill = t.elapsed().as_secs_f64() * 1000.0;
        println!(
            "[{label}] construct_ms={construct:.1} masks={masks} prewarm_ms={prewarm:.1} \
             matcher_and_first_fill_ms={first_fill:.1} mask_bytes={}",
            compiled.memory_size_bytes(),
        );
    }
}

/// End-to-end cold path on a real tokenizer: what a request pays from
/// "tool schema in hand" to "first constrained mask filled", with and
/// without a snapshot on disk. Ignored for the same reason as
/// [`phase_timings_with_a_real_tokenizer`].
#[test]
#[ignore = "CPU diagnostic: set QWEN_TOKENIZER_JSON to an existing tokenizer.json"]
fn cold_vs_snapshot_warm_with_a_real_tokenizer() {
    let path = std::env::var("QWEN_TOKENIZER_JSON").expect("QWEN_TOKENIZER_JSON");
    let tokenizer = tokenizers::Tokenizer::from_file(path).expect("tokenizer loads");
    let stop = tokenizer.token_to_id("<|im_end|>").expect("<|im_end|>") as i32;
    let dir = scratch("real");
    let tools = tool("get_weather", "city", "days");
    let build = || GrammarEngine::from_tokenizer(&tokenizer, None, &[stop]).expect("engine");

    let mut cold_engine = build();
    cold_engine.attach_mask_cache(&dir);
    let cold = prepare(&mut cold_engine, &tools);
    await_snapshot(&dir).expect("snapshot persisted");

    for rep in 0..3 {
        let mut warm_engine = build();
        warm_engine.attach_mask_cache(&dir);
        let warm = prepare(&mut warm_engine, &tools);
        let mut unseen_engine = build();
        unseen_engine.attach_mask_cache(&dir);
        let unseen = prepare(&mut unseen_engine, &tool("run_cmd", "command", "timeout"));
        await_snapshot(&dir);
        // How long the REQUEST thread itself is held before prefill can
        // start — the part candidate (4) moves behind the forward pass.
        let mut admit_engine = build();
        let admitted = Instant::now();
        let compiled = admit_engine
            .compile_qwen3_coder_tool_grammar(&tool("ping", "host", "count"), true, "</parameter>")
            .unwrap();
        let mut state =
            GrammarState::new_with_hook(&compiled, admit_engine.vocab_size(), None).unwrap();
        let admit_ms = admitted.elapsed().as_secs_f64() * 1000.0;
        state.fill_bitmask();
        let joined_ms = admitted.elapsed().as_secs_f64() * 1000.0;
        println!(
            "rep={rep} cold_ms={:.1} warm_same_schema_ms={:.1} warm_unseen_schema_ms={:.1} \
             cold_admit_ms={admit_ms:.1} cold_admit_to_first_fill_ms={joined_ms:.1}",
            (cold.construct + cold.masks).as_secs_f64() * 1000.0,
            (warm.construct + warm.masks).as_secs_f64() * 1000.0,
            (unseen.construct + unseen.masks).as_secs_f64() * 1000.0,
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
