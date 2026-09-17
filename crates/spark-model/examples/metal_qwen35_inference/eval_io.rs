// SPDX-License-Identifier: AGPL-3.0-only
//! Eval-comparison IO knobs for the inference driver: per-step logit
//! dumps (`AVAROK_LOGITS_OUT`), argmax-sequence dumps
//! (`AVAROK_DUMP_TOKENS`), and teacher-forced token lists
//! (`AVAROK_FORCE_TOKENS_FILE`). Used by `tests/metal_kv_kld_compare.py`
//! to compare KV dtypes with identical contexts at every position.

use anyhow::Result;

/// Open the per-step logits sink when `AVAROK_LOGITS_OUT` is set. Raw
/// little-endian bf16, `[n_steps, vocab]`.
pub fn logits_writer() -> Option<std::cell::RefCell<std::io::BufWriter<std::fs::File>>> {
    std::env::var("AVAROK_LOGITS_OUT").ok().map(|p| {
        std::cell::RefCell::new(std::io::BufWriter::new(
            std::fs::File::create(p).expect("create AVAROK_LOGITS_OUT"),
        ))
    })
}

/// Load the teacher-forcing list from `AVAROK_FORCE_TOKENS_FILE`
/// (newline-separated token ids; position 0 replaces the first sampled
/// token). Two runs fed the same list share an identical context at
/// every position, so their logit dumps are KLD-comparable end to end.
pub fn forced_tokens() -> Option<Vec<u32>> {
    std::env::var("AVAROK_FORCE_TOKENS_FILE").ok().map(|p| {
        std::fs::read_to_string(&p)
            .unwrap_or_else(|e| panic!("read AVAROK_FORCE_TOKENS_FILE {p}: {e}"))
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.trim().parse().expect("token id"))
            .collect()
    })
}

/// Write the argmax sequence (one id per line) when `AVAROK_DUMP_TOKENS`
/// is set — the output feeds `AVAROK_FORCE_TOKENS_FILE` in later runs.
pub fn maybe_dump_tokens(generated_ids: &[u32]) -> Result<()> {
    if let Ok(path) = std::env::var("AVAROK_DUMP_TOKENS") {
        let body: String = generated_ids.iter().map(|t| format!("{t}\n")).collect();
        std::fs::write(&path, body)?;
        println!("  (dumped {} token ids to {path})", generated_ids.len());
    }
    Ok(())
}
