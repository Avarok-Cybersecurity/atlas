// SPDX-License-Identifier: AGPL-3.0-only

//! Explicit CPU diagnostic using an already-present Qwen tokenizer. Never downloads.

use super::*;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use xgrammar::compiler::{CompiledGrammar, CompiledGrammarImpl, RuleLevelCache};
use xgrammar::matcher::GrammarMatcher;

fn fresh(source: &CompiledGrammar, threads: usize) -> CompiledGrammar {
    let s = source.inner();
    CompiledGrammar::from_impl(Arc::new(CompiledGrammarImpl {
        prewarm_max_threads: threads,
        grammar: Arc::clone(&s.grammar),
        tokenizer_info: s.tokenizer_info.clone(),
        mask_cache: Mutex::new(Default::default()),
        tag_slice: Arc::clone(&s.tag_slice),
        rule_cache: Some(RuleLevelCache::new(1024 * 1024 * 1024 / 3)),
        decomposition: s.decomposition.clone(),
    }))
}

#[test]
#[ignore = "CPU diagnostic: requires QWEN_TOKENIZER_JSON pointing to existing pinned tokenizer"]
fn native_qwen_prewarm_exactness_and_cpu_timing() {
    let path = std::env::var("QWEN_TOKENIZER_JSON").expect("explicit existing tokenizer path");
    let tokenizer = tokenizers::Tokenizer::from_file(path).unwrap();
    let stop = tokenizer.token_to_id("<|im_end|>").unwrap() as i32;
    let end = tokenizer.token_to_id("<|endoftext|>").unwrap() as i32;
    assert_eq!([stop, end], [248046, 248044]);
    let mut engine = GrammarEngine::from_tokenizer(&tokenizer, Some(248320), &[stop, end]).unwrap();
    let tools: Vec<ToolDefinition> = serde_json::from_value(serde_json::json!([{
        "type":"function", "function": {"name":"get_weather",
        "description":"Look up the current weather for a city.",
        "parameters":{"type":"object","properties":{
            "city":{"type":"string","description":"City name."},
            "days":{"type":"integer","description":"Forecast horizon in days."}},
            "required":["city","days"]}}
    }]))
    .unwrap();
    let output = "<tool_call>\n<function=get_weather>\n<parameter=city>\nReykjavik\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
    let ids = tokenizer.encode(output, false).unwrap().get_ids().to_vec();
    for auto in [true, false] {
        let started = Instant::now();
        let source = engine
            .compile_qwen3_coder_tool_grammar(&tools, auto, "</parameter>")
            .unwrap();
        println!(
            "native compile auto={auto} ms={:.3}",
            started.elapsed().as_secs_f64() * 1000.0
        );
        if let Ok(dir) = std::env::var("QWEN_GRAMMAR_EXPORT_DIR") {
            std::fs::write(
                format!("{dir}/native-{auto}.ebnf"),
                xgrammar::grammar::print_grammar(source.grammar()),
            )
            .unwrap();
            std::fs::write(
                format!("{dir}/output-ids.json"),
                serde_json::to_string(&ids).unwrap(),
            )
            .unwrap();
        }
        for rep in 0..1 {
            let serial = fresh(&source, 1);
            let parallel = fresh(&source, 4);
            let start = Instant::now();
            let n_serial = serial.compile_top_k_masks(512);
            let serial_ms = start.elapsed().as_secs_f64() * 1000.0;
            let start = Instant::now();
            let n_parallel = parallel.compile_top_k_masks(512);
            let parallel_ms = start.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(n_serial, n_parallel);
            assert_eq!(
                *serial.inner().mask_cache.lock().unwrap(),
                *parallel.inner().mask_cache.lock().unwrap(),
                "adaptive masks differ before sampling"
            );
            let mut a = GrammarMatcher::new(serial, None, false, -1);
            let mut b = GrammarMatcher::new(parallel, None, false, -1);
            let mut am = vec![0; 248320usize.div_ceil(32)];
            let mut bm = am.clone();
            let mut fill_seconds = 0.0;
            for &id in &ids {
                am.fill(0);
                bm.fill(0);
                let start = Instant::now();
                a.fill_next_token_bitmask(&mut am, 0, false).unwrap();
                fill_seconds += start.elapsed().as_secs_f64();
                b.fill_next_token_bitmask(&mut bm, 0, false).unwrap();
                assert_eq!(am, bm, "next-token masks differ");
                assert_ne!(
                    am[id as usize / 32] & (1 << (id % 32)),
                    0,
                    "native continuation rejected"
                );
                assert!(a.accept_token(id as i32, false));
                assert!(b.accept_token(id as i32, false));
                a.rollback(1);
                b.rollback(1);
                am.fill(0);
                bm.fill(0);
                a.fill_next_token_bitmask(&mut am, 0, false).unwrap();
                b.fill_next_token_bitmask(&mut bm, 0, false).unwrap();
                assert_eq!(am, bm, "rollback masks differ");
                assert!(a.accept_token(id as i32, false));
                assert!(b.accept_token(id as i32, false));
            }
            println!(
                "native auto={auto} rep={rep} masks={n_serial} serial_ms={serial_ms:.3} parallel_ms={parallel_ms:.3} fill_ms_per_token={:.3} tokens={}",
                fill_seconds * 1000.0 / ids.len() as f64,
                ids.len()
            );
            // Both paths reject the same malformed function name in required mode.
            if !auto {
                a.reset();
                b.reset();
                assert!(!a.accept_string("<tool_call>\n<function=unknown>", false));
                assert!(!b.accept_string("<tool_call>\n<function=unknown>", false));
            }
        }
    }
}
