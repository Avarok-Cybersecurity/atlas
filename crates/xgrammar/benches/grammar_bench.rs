// SPDX-License-Identifier: AGPL-3.0-only
//
// Criterion micro-benchmark for the pure-Rust `xgrammar` crate.
//
// Exercises the two hot paths the Tier-1 perf work targeted:
//   * grammar compilation  — `GrammarCompiler::compile_json_schema`
//     / `compile_builtin_json_grammar`
//   * the per-token mask path — `GrammarMatcher::fill_next_token_bitmask`
//
// Everything is deterministic: a fixed synthetic vocabulary, a fixed
// tool-call JSON schema, and a fixed valid token sequence driven through
// `accept_token`. The bench only touches the public API façade
// (`xgrammar::{GrammarCompiler, GrammarMatcher, TokenizerInfo,
// VocabType, allocate_token_bitmask}`) so it stays inside `forbid(unsafe)`.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use xgrammar::{GrammarCompiler, GrammarMatcher, TokenizerInfo, VocabType, allocate_token_bitmask};

// ── Synthetic vocabulary ───────────────────────────────────────────
//
// A realistic-sized byte-level vocabulary so the bitmask words
// (`ceil(vocab/32)`) match production mask sizes. The first 256 entries
// are the single bytes (so the matcher can spell any JSON character);
// the remainder are deterministic filler tokens. `VOCAB_SIZE` is fixed
// at 32k — large enough for realistic mask work without making each
// criterion sample prohibitively slow in the container.
const VOCAB_SIZE: usize = 32_768;

/// Build the fixed synthetic vocabulary. Token 0..256 are the raw byte
/// values (byte-level encoding uses the `<0xNN>` form upstream, but the
/// pure-Rust tokenizer decodes ByteLevel tokens directly — single
/// printable ASCII chars are stored verbatim). Tokens 256.. are unique
/// filler strings that never collide with grammar terminals.
fn build_vocab() -> Vec<String> {
    let mut vocab: Vec<String> = Vec::with_capacity(VOCAB_SIZE);
    // 0..256 — every single byte as a one-char token.
    for b in 0u32..256 {
        vocab.push(char::from_u32(b).unwrap_or('\u{FFFD}').to_string());
    }
    // 256.. — deterministic multi-char filler tokens.
    for i in 256..VOCAB_SIZE {
        vocab.push(format!("tok{i}"));
    }
    vocab
}

/// Construct the `TokenizerInfo` over the synthetic vocab. ByteLevel so
/// the matcher's decoded-vocab path matches a Qwen/MiniMax tokenizer.
fn tokenizer_info() -> TokenizerInfo {
    let vocab = build_vocab();
    TokenizerInfo::new(&vocab, VocabType::ByteLevel, &None, false)
        .expect("synthetic TokenizerInfo construction")
}

// ── Representative tool-call JSON schema ───────────────────────────
//
// A `get_weather`-style tool: an object with several typed properties
// (string / number / enum / boolean) and a `required` list — the shape
// a real function-calling grammar compiles.
const TOOL_SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "location": {"type": "string"},
    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
    "days": {"type": "integer"},
    "detailed": {"type": "boolean"}
  },
  "required": ["location", "unit"],
  "additionalProperties": false
}"#;

/// Compile the tool-call schema into a `CompiledGrammar`.
fn compile_tool_schema(compiler: &mut GrammarCompiler) -> xgrammar::CompiledGrammar {
    compiler
        .compile_json_schema(TOOL_SCHEMA, true, None, None::<(&str, &str)>, true, None)
        .expect("compile tool-call schema")
}

// ── A fixed valid token sequence ───────────────────────────────────
//
// The matcher is driven through the characters of a concrete valid JSON
// object that satisfies `TOOL_SCHEMA`. Each character is one byte-level
// token (id == byte value, as built above), so the sequence is a stable
// list of single-byte token ids. `fill_next_token_bitmask` is timed at
// the *first* generation step (root just opened) — the heaviest mask
// because the largest set of tokens is still reachable.
const VALID_JSON: &str = r#"{"location": "Paris", "unit": "celsius", "days": 3, "detailed": true}"#;

/// The valid JSON as a sequence of single-byte token ids.
fn valid_token_ids() -> Vec<i32> {
    VALID_JSON.bytes().map(|b| b as i32).collect()
}

// ── Benchmarks ─────────────────────────────────────────────────────

fn bench_compile(c: &mut Criterion) {
    let info = tokenizer_info();
    let mut group = c.benchmark_group("compile");
    group.sample_size(20);

    // Tool-call JSON schema — cache disabled so every iteration does the
    // full compile (the path Tier 1 optimised), not a cache hit.
    group.bench_function("tool_schema", |b| {
        b.iter(|| {
            let mut compiler =
                GrammarCompiler::new(&info, 1, false, -1).expect("GrammarCompiler::new");
            black_box(compile_tool_schema(&mut compiler));
        });
    });

    // Built-in standard-JSON grammar.
    group.bench_function("builtin_json", |b| {
        b.iter(|| {
            let mut compiler =
                GrammarCompiler::new(&info, 1, false, -1).expect("GrammarCompiler::new");
            black_box(
                compiler
                    .compile_builtin_json_grammar()
                    .expect("builtin json"),
            );
        });
    });

    group.finish();
}

fn bench_fill_bitmask(c: &mut Criterion) {
    let info = tokenizer_info();
    let mut compiler = GrammarCompiler::new(&info, 1, false, -1).expect("GrammarCompiler::new");
    let compiled = compile_tool_schema(&mut compiler);
    let words = allocate_token_bitmask(1, VOCAB_SIZE).len();

    let mut group = c.benchmark_group("fill_bitmask");
    group.sample_size(50);

    // Per-token hot path: a fresh matcher (root just opened) filling the
    // first next-token bitmask. This is the heaviest single mask.
    group.bench_function("first_step", |b| {
        b.iter_batched(
            || {
                (
                    GrammarMatcher::new(&compiled, None, false, -1).expect("GrammarMatcher::new"),
                    vec![0i32; words],
                )
            },
            |(mut matcher, mut bitmask)| {
                black_box(matcher.fill_next_token_bitmask(&mut bitmask, 0, false));
            },
            criterion::BatchSize::SmallInput,
        );
    });

    // Mid-generation step: accept the first half of the valid sequence,
    // then time the bitmask fill at that interior matcher state.
    let tokens = valid_token_ids();
    let half = tokens.len() / 2;
    group.bench_function("mid_step", |b| {
        b.iter_batched(
            || {
                let mut matcher =
                    GrammarMatcher::new(&compiled, None, false, -1).expect("GrammarMatcher::new");
                for &t in &tokens[..half] {
                    matcher.accept_token(t);
                }
                (matcher, vec![0i32; words])
            },
            |(mut matcher, mut bitmask)| {
                black_box(matcher.fill_next_token_bitmask(&mut bitmask, 0, false));
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_accept_fill_loop(c: &mut Criterion) {
    let info = tokenizer_info();
    let mut compiler = GrammarCompiler::new(&info, 1, false, -1).expect("GrammarCompiler::new");
    let compiled = compile_tool_schema(&mut compiler);
    let words = allocate_token_bitmask(1, VOCAB_SIZE).len();
    let tokens = valid_token_ids();

    let mut group = c.benchmark_group("accept_fill_loop");
    group.sample_size(30);

    // Simulate a full short generation: for each token, fill the mask
    // then accept the token — the exact per-step server loop.
    group.bench_function("full_generation", |b| {
        b.iter_batched(
            || {
                (
                    GrammarMatcher::new(&compiled, None, false, -1).expect("GrammarMatcher::new"),
                    vec![0i32; words],
                )
            },
            |(mut matcher, mut bitmask)| {
                for &t in &tokens {
                    black_box(matcher.fill_next_token_bitmask(&mut bitmask, 0, false));
                    matcher.accept_token(t);
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_compile,
    bench_fill_bitmask,
    bench_accept_fill_loop
);
criterion_main!(benches);
