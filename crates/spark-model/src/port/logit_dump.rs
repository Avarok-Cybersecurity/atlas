// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the prompt-logits dump writer (`AVAROK_PROMPT_LOGITS_DUMP=<dir>`),
//! the server half of the teacher-forced KL measurements (bench `tf_kl.py`).
//!
//! # What it writes
//!
//! A `/v1/completions` request with `echo` and `logprobs` projects EVERY prompt
//! position through the LM head (`prefill_b/prompt_logprobs.rs`), in batches of
//! up to 32 rows, before the top-k is taken. With the variable set, each host
//! batch's full vocabulary rows are appended here as FP16 (converted from the
//! BF16 logits, little endian), so a reference serve and a candidate serve can
//! be compared over the whole distribution at every position, not only over
//! the top-k.
//!
//! Per request, with `id = request_id(prompt tokens)`:
//!
//! * `<dir>/<id>.f16`: row `p` is the logits at prompt position `p`, which
//!   predict `tokens[p + 1]`; `vocab_size` FP16 values per row, rows in order,
//!   `(tokens.len() - 1)` rows when the prompt is fully scored.
//! * `<dir>/<id>.json`: `{"request_id", "format": "f16-le", "vocab_size",
//!   "rows", "row_i_predicts": "tokens[i+1]", "tokens": [...]}`, written with
//!   the first batch. A reader checks `file size == rows * vocab_size * 2`.
//!
//! The id is a hash of the prompt token ids (FNV-1a 64 over the length and
//! every id, little endian), so a client that knows the tokens it sent computes
//! the same name, and two runs over identical prompts land on identical names
//! (the first batch truncates, so a rerun overwrites rather than appends).
//!
//! # Cost
//!
//! Unset, [`dump_batch`] is one `OnceLock` read. Set, it costs a conversion and
//! a write per batch on a path that already copies 16+ MB device to host. A
//! write error logs once and disables the dump; it never fails the request.

use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

/// The environment variable that names the dump directory.
pub const ENV: &str = "AVAROK_PROMPT_LOGITS_DUMP";

/// The dump directory, read once; `None` when unset or empty.
pub fn dump_dir() -> Option<&'static Path> {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| {
        std::env::var_os(ENV)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    })
    .as_deref()
}

/// FNV-1a 64 over the token count and every token id (little endian), as 16
/// lowercase hex digits.
pub fn request_id(tokens: &[u32]) -> String {
    fn eat(h: &mut u64, bytes: &[u8]) {
        for &b in bytes {
            *h ^= u64::from(b);
            *h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    eat(&mut h, &(tokens.len() as u64).to_le_bytes());
    for t in tokens {
        eat(&mut h, &t.to_le_bytes());
    }
    format!("{h:016x}")
}

/// The JSON index of one request.
pub fn index_json(id: &str, vocab: usize, rows: usize, tokens: &[u32]) -> String {
    let ids: Vec<String> = tokens.iter().map(u32::to_string).collect();
    format!(
        "{{\"request_id\":\"{id}\",\"format\":\"f16-le\",\"vocab_size\":{vocab},\
         \"rows\":{rows},\"row_i_predicts\":\"tokens[i+1]\",\"tokens\":[{}]}}\n",
        ids.join(",")
    )
}

/// Write `rows` rows of BF16 logits (`bf16_rows`, `vocab` values per row,
/// little endian) at absolute row `first_row` of the request's `.f16` file as
/// FP16. `first_row == 0` truncates the file and writes the index.
pub fn write_batch(
    dir: &Path,
    tokens: &[u32],
    first_row: usize,
    rows: usize,
    vocab: usize,
    bf16_rows: &[u8],
) -> std::io::Result<()> {
    let want = rows * vocab * 2;
    if bf16_rows.len() != want {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "expected {want} bytes ({rows} rows x {vocab}), got {}",
                bf16_rows.len()
            ),
        ));
    }
    std::fs::create_dir_all(dir)?;
    let id = request_id(tokens);
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(first_row == 0)
        .open(dir.join(format!("{id}.f16")))?;
    file.seek(SeekFrom::Start((first_row * vocab * 2) as u64))?;
    let mut out = Vec::with_capacity(want);
    for c in bf16_rows.chunks_exact(2) {
        let v = half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32();
        out.extend_from_slice(&half::f16::from_f32(v).to_bits().to_le_bytes());
    }
    file.write_all(&out)?;
    if first_row == 0 {
        let scored = tokens.len().saturating_sub(1);
        std::fs::write(
            dir.join(format!("{id}.json")),
            index_json(&id, vocab, scored, tokens),
        )?;
    }
    Ok(())
}

static FAILED: AtomicBool = AtomicBool::new(false);

/// The hook `collect_prompt_logprobs_chunk` calls after each host batch.
/// `first_row` is the absolute prompt position of the batch's first row.
/// No-op unless `AVAROK_PROMPT_LOGITS_DUMP` is set; a write error logs once and
/// disables the dump.
pub fn dump_batch(tokens: &[u32], first_row: usize, rows: usize, vocab: usize, bf16_rows: &[u8]) {
    let Some(dir) = dump_dir() else {
        return;
    };
    if FAILED.load(Ordering::Relaxed) {
        return;
    }
    if let Err(e) = write_batch(dir, tokens, first_row, rows, vocab, bf16_rows)
        && !FAILED.swap(true, Ordering::Relaxed)
    {
        tracing::warn!(
            "{ENV}: writing to {} failed, dump disabled: {e}",
            dir.display()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    fn scratch() -> PathBuf {
        static N: AtomicU32 = AtomicU32::new(0);
        let d = std::env::temp_dir().join(format!(
            "atlas-logit-dump-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn bf16_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_bits().to_le_bytes())
            .collect()
    }

    fn read_f16(path: &Path) -> Vec<f32> {
        std::fs::read(path)
            .unwrap()
            .chunks_exact(2)
            .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect()
    }

    #[test]
    fn two_batches_round_trip_as_fp16_rows_with_an_index() {
        let dir = scratch();
        let tokens = [10u32, 11, 12, 13, 14];
        let b0 = [1.0, 2.0, -3.5, 0.25, 0.0, 100.0, -0.5, 7.0];
        let b1 = [4.0, -1.0, 8.0, 0.5, -2.0, 3.0, 6.0, -8.0];
        write_batch(&dir, &tokens, 0, 2, 4, &bf16_bytes(&b0)).unwrap();
        write_batch(&dir, &tokens, 2, 2, 4, &bf16_bytes(&b1)).unwrap();
        let id = request_id(&tokens);
        let rows = read_f16(&dir.join(format!("{id}.f16")));
        let want: Vec<f32> = b0.iter().chain(b1.iter()).copied().collect();
        assert_eq!(rows, want, "row p is the logits at prompt position p");
        let idx = std::fs::read_to_string(dir.join(format!("{id}.json"))).unwrap();
        assert!(idx.contains(&format!("\"request_id\":\"{id}\"")), "{idx}");
        assert!(idx.contains("\"vocab_size\":4"), "{idx}");
        assert!(idx.contains("\"rows\":4"), "{idx}");
        assert!(idx.contains("\"tokens\":[10,11,12,13,14]"), "{idx}");
        assert!(idx.contains("\"format\":\"f16-le\""), "{idx}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_first_batch_of_a_rerun_truncates_instead_of_appending() {
        let dir = scratch();
        let tokens = [1u32, 2, 3, 4, 5];
        let b = bf16_bytes(&[1.0; 8]);
        write_batch(&dir, &tokens, 0, 2, 4, &b).unwrap();
        write_batch(&dir, &tokens, 2, 2, 4, &b).unwrap();
        write_batch(&dir, &tokens, 0, 2, 4, &b).unwrap();
        let id = request_id(&tokens);
        let len = std::fs::metadata(dir.join(format!("{id}.f16")))
            .unwrap()
            .len();
        assert_eq!(len, 2 * 4 * 2, "a rerun overwrites the earlier dump");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_wrong_sized_batch_is_an_error_not_a_corrupt_file() {
        let dir = scratch();
        let err = write_batch(&dir, &[1, 2, 3], 0, 2, 4, &[0u8; 15]).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        assert!(!dir.exists(), "nothing is created for a rejected batch");
    }

    #[test]
    fn the_id_is_a_stable_function_of_the_prompt_tokens() {
        assert_eq!(request_id(&[1, 2, 3]), request_id(&[1, 2, 3]));
        assert_ne!(request_id(&[1, 2, 3]), request_id(&[1, 2, 4]));
        assert_ne!(request_id(&[1, 2]), request_id(&[1, 2, 0]));
        assert_eq!(request_id(&[]).len(), 16);
        // FNV-1a 64 of the empty prompt: the offset basis folded over the
        // 8-byte zero length. Pinned so the bench's twin of this function can be
        // checked against a constant.
        assert_eq!(request_id(&[]), "a8c7f832281a39c5");
    }

    #[test]
    fn unset_the_hook_does_nothing() {
        if dump_dir().is_some() {
            return; // the environment armed the dump
        }
        dump_batch(&[1, 2, 3], 0, 1, 2, &[0u8; 4]);
    }
}
