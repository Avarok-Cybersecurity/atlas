// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-only timing of grammar exported from Atlas's native tool compiler.
use std::sync::{Arc, Mutex};
use std::time::Instant;
use xgrammar::compiler::{CompiledGrammar, CompiledGrammarImpl, GrammarCompiler, RuleLevelCache};
use xgrammar::matcher::GrammarMatcher;
use xgrammar::tokenizer::TokenizerInfo;

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

fn main() {
    let dir = std::env::args()
        .nth(1)
        .expect("existing CPU fixture directory");
    let raw = std::fs::read_to_string(format!("{dir}/qwen-tokenizer.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let mut vocab = vec![String::new(); 248320];
    for (tok, id) in json["model"]["vocab"].as_object().unwrap() {
        vocab[id.as_u64().unwrap() as usize] = tok.clone();
    }
    for tok in json["added_tokens"].as_array().unwrap() {
        vocab[tok["id"].as_u64().unwrap() as usize] = tok["content"].as_str().unwrap().to_owned();
    }
    let stop = vocab.iter().position(|t| t == "<|im_end|>").unwrap() as i32;
    let end = vocab.iter().position(|t| t == "<|endoftext|>").unwrap() as i32;
    assert_eq!([stop, end], [248046, 248044]);
    let meta = xgrammar::detect_metadata_from_hf(&raw).unwrap();
    let info = TokenizerInfo::new(
        &vocab,
        meta.vocab_type,
        None,
        Some(vec![stop, end]),
        meta.add_prefix_space,
    );
    let ids: Vec<i32> =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/output-ids.json")).unwrap())
            .unwrap();
    for auto in [true, false] {
        let ebnf = std::fs::read_to_string(format!("{dir}/native-{auto}.ebnf")).unwrap();
        let compiler = GrammarCompiler::new(info.clone(), 1, true, -1);
        let start = Instant::now();
        let source = compiler.compile_grammar_from_ebnf(&ebnf, "root").unwrap();
        println!(
            "compile auto={auto} ms={:.3}",
            start.elapsed().as_secs_f64() * 1000.0
        );
        if std::env::args().nth(2).as_deref() == Some("--warm-only") {
            let warmed = fresh(&source, 4);
            let count = warmed.compile_top_k_masks(512);
            let started = Instant::now();
            for _ in 0..100 {
                assert_eq!(warmed.compile_top_k_masks(512), count);
            }
            println!(
                "warm auto={auto} selected={count} us_per_call={:.3}",
                started.elapsed().as_secs_f64() * 1e6 / 100.0
            );
            continue;
        }
        for rep in 0..3 {
            let serial = fresh(&source, 1);
            let parallel = fresh(&source, 4);
            let start = Instant::now();
            let n_serial = serial.compile_top_k_masks(512);
            let serial_ms = start.elapsed().as_secs_f64() * 1000.0;
            let start = Instant::now();
            let n_parallel = parallel.compile_top_k_masks(512);
            let parallel_ms = start.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(n_serial, n_parallel);
            assert!(
                *serial.inner().mask_cache.lock().unwrap()
                    == *parallel.inner().mask_cache.lock().unwrap(),
                "adaptive mask/state sets differ"
            );
            let mut a = GrammarMatcher::new(serial, None, false, -1);
            let mut b = GrammarMatcher::new(parallel, None, false, -1);
            let mut am = vec![0; 248320usize.div_ceil(32)];
            let mut bm = am.clone();
            let mut fill_seconds = 0.0;
            let mut positions = Vec::new();
            for (position, &id) in ids.iter().enumerate() {
                am.fill(0);
                bm.fill(0);
                let start = Instant::now();
                a.fill_next_token_bitmask(&mut am, 0, false).unwrap();
                let fill_us = start.elapsed().as_secs_f64() * 1e6;
                fill_seconds += fill_us / 1e6;
                b.fill_next_token_bitmask(&mut bm, 0, false).unwrap();
                assert!(am == bm, "next-token masks differ");
                assert_ne!(am[id as usize / 32] & (1 << (id % 32)), 0);
                let start = Instant::now();
                assert!(a.accept_token(id, false));
                let accept_us = start.elapsed().as_secs_f64() * 1e6;
                positions.push(serde_json::json!({"position":position,"token_id":id,"token":String::from_utf8_lossy(&info.decoded_vocab()[id as usize]),"fill_us":fill_us,"accept_us":accept_us}));
                assert!(b.accept_token(id, false));
                a.rollback(1);
                b.rollback(1);
                am.fill(0);
                bm.fill(0);
                a.fill_next_token_bitmask(&mut am, 0, false).unwrap();
                b.fill_next_token_bitmask(&mut bm, 0, false).unwrap();
                assert!(am == bm, "rollback masks differ");
                assert!(a.accept_token(id, false));
                assert!(b.accept_token(id, false));
            }
            println!(
                "auto={auto} rep={rep} masks={n_serial} serial_ms={serial_ms:.3} parallel_ms={parallel_ms:.3} fill_ms_per_token={:.3} tokens={}",
                fill_seconds * 1000.0 / ids.len() as f64,
                ids.len()
            );
            println!(
                "positions {}",
                serde_json::json!({"auto":auto,"rep":rep,"positions":positions})
            );
            if !auto {
                a.reset();
                b.reset();
                assert!(!a.accept_string("<tool_call>\n<function=unknown>", false));
                assert!(!b.accept_string("<tool_call>\n<function=unknown>", false));
            }
        }
    }
}
