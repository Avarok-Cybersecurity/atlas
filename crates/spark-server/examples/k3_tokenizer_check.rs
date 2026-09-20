// SPDX-License-Identifier: AGPL-3.0-only
//! Compare a derived tokenizer against the offline tiktoken oracle (no GPU/model weights).
use anyhow::{Context, Result, ensure};
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    text: String,
    ids: Vec<u32>,
    decoded: String,
    decoded_skip_special: String,
}

fn main() -> Result<()> {
    let directory = std::env::args()
        .nth(1)
        .context("usage: k3_tokenizer_check DERIVED_DIRECTORY SERVING_DIRECTORY")?;
    let serving = std::env::args()
        .nth(2)
        .context("missing serving directory")?;
    let serving = std::path::Path::new(&serving);
    let config =
        avarok_core::config::parse_config(&std::fs::read_to_string(serving.join("config.json"))?)?;
    ensure!(
        config.eos_ids() == vec![163586],
        "official model EOS changed"
    );
    let chat = spark_server::tokenizer::ChatTokenizer::from_model_dir(
        serving,
        config.eos_token_id,
        true,
        &config.model_type,
        None,
        true,
    )?;
    ensure!(
        chat.eos_token_id() == 163586,
        "ChatTokenizer changed model EOS"
    );
    ensure!(
        chat.inner().token_to_id("[EOS]") == Some(163585),
        "named EOS changed"
    );
    ensure!(
        chat.decode(&[163587, 163588, 163589, 163586])? == "<|open|><|close|><|sep|>",
        "XTML marker decoding changed"
    );
    let path = std::path::Path::new(&directory);
    let tokenizer = tokenizers::Tokenizer::from_file(path.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;
    let cases: Vec<Case> =
        serde_json::from_slice(&std::fs::read(path.join("tokenizer-oracle.json"))?)?;
    ensure!(!cases.is_empty(), "empty oracle");
    for (i, case) in cases.iter().enumerate() {
        ensure!(
            chat.encode(&case.text)? == case.ids,
            "ChatTokenizer encode differs at case {i}"
        );
        ensure!(
            chat.decode(&case.ids)? == case.decoded_skip_special,
            "ChatTokenizer decode differs at case {i}"
        );
        let encoded = tokenizer
            .encode(case.text.as_str(), false)
            .map_err(|e| anyhow::anyhow!("encode case {i}: {e}"))?;
        ensure!(
            encoded.get_ids() == case.ids,
            "token IDs differ at case {i}"
        );
        let decoded = tokenizer
            .decode(&case.ids, false)
            .map_err(|e| anyhow::anyhow!("decode case {i}: {e}"))?;
        ensure!(decoded == case.decoded, "decoded text differs at case {i}");
        let skipped = tokenizer
            .decode(&case.ids, true)
            .map_err(|e| anyhow::anyhow!("decode skip-special case {i}: {e}"))?;
        ensure!(
            skipped == case.decoded_skip_special,
            "skip-special text differs at case {i}"
        );
    }
    println!(
        "PASS: {} official K3 tiktoken cases match Atlas Rust and ChatTokenizer; model EOS preserved",
        cases.len()
    );
    Ok(())
}
