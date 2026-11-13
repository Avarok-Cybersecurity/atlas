// SPDX-License-Identifier: AGPL-3.0-only
// Portions derived from the ARC-1.0 release documentation and rayline-ai/vllm-arc (examples/readouts.py), Copyright
// 2026 Atlas Futures, Inc., licensed under the Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0). By
// Atlas contributors. See RAYLINE-ROUTING-ATTRIBUTION.md.
//! Responsibility: step 3 of the routing gate (B0): the encoder readouts of the prompts (`rows`). Each prompt is
//! serialized into role blocks, tokenized, run through the 12-layer CPU stack, and the final-norm hidden state at the
//! readout token of the last block is stored (BF16 weights, f32 activations). Resumable: a partial file is written
//! every 50 rows. Prompts over `--max-tokens` get a NaN row and `ok = false`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use serde_json::json;

use super::encoder::{Encoder, EncoderOptions};
use super::fit_eval::write_readouts;
use super::goldens::load_jsonl;
use super::serializer::{HfTokenizer, encode_blocks, render_blocks};
use super::tensorio::TensorFile;

/// Arguments of `rows`.
pub struct RowsArgs {
    pub release: PathBuf,
    pub prompts: PathBuf,
    pub out: PathBuf,
    pub limit: usize,
    pub block_end: String,
    pub shell_file: Option<PathBuf>,
    pub max_tokens: usize,
    pub fresh: bool,
    pub bf16: bool,
}

/// Rows already computed: `(row, n_tokens, ok)` by prompt id, with the width.
type Done = HashMap<String, (Vec<f32>, i64, bool)>;

fn read_partial(path: &Path) -> Result<Done> {
    let file = TensorFile::open(path)?;
    let meta = file.metadata();
    let ids: Vec<String> = serde_json::from_str(meta.get("ids_json").map_or("[]", String::as_str))?;
    let (shape, x) = file.read_f32("X")?;
    let d = shape.get(1).copied().unwrap_or(0);
    let n_tokens = file.read("n_tokens")?.data;
    let ok = file.read("ok")?.data;
    Ok(ids
        .into_iter()
        .enumerate()
        .map(|(k, id)| {
            (
                id,
                (
                    x[k * d..(k + 1) * d].to_vec(),
                    n_tokens[k] as i64,
                    ok[k] != 0.0,
                ),
            )
        })
        .collect())
}

/// `rows`: writes the readouts file. Returns the exit code.
pub fn cmd_rows(a: &RowsArgs) -> Result<i32> {
    if a.out.extension().is_none_or(|e| e != "safetensors") {
        eprintln!("route_gate rows: --out must end in .safetensors");
        return Ok(2);
    }
    let mut items = load_jsonl(&a.prompts)?;
    if a.limit > 0 {
        items.truncate(a.limit);
    }
    let shell = a
        .shell_file
        .as_ref()
        .map(|p| std::fs::read_to_string(p).with_context(|| format!("read {}", p.display())))
        .transpose()?;
    let tok = HfTokenizer::from_release(&a.release)?;
    let enc = Encoder::load(
        &a.release,
        EncoderOptions {
            bf16: a.bf16,
            ..EncoderOptions::default()
        },
    )?;
    let d = enc.cfg.hidden;
    let mut part = a.out.clone().into_os_string();
    part.push(".partial.safetensors");
    let part = PathBuf::from(part);
    let done: Done = if part.exists() && !a.fresh {
        read_partial(&part)?
    } else {
        HashMap::new()
    };
    if !done.is_empty() {
        println!(
            "route_gate rows: resuming with {} rows from {}",
            done.len(),
            part.display()
        );
    }
    let (mut ids, mut x, mut ntok, mut ok): (Vec<String>, Vec<f32>, Vec<i64>, Vec<bool>) =
        Default::default();
    let t0 = Instant::now();
    for (n, it) in items.iter().enumerate() {
        let id = it["id"].as_str().unwrap_or_default().to_string();
        let (row, nt, good) = match done.get(&id) {
            Some((row, nt, good)) => (row.clone(), *nt, *good),
            None => {
                let msgs = it["messages"].as_array().cloned().unwrap_or_default();
                let e = encode_blocks(&tok, &render_blocks(&msgs), shell.as_deref())?;
                let nt = e.ids.len();
                if nt <= a.max_tokens {
                    let h = enc.hidden(&e.ids)?;
                    let offs = e.pick_offsets(&a.block_end)?;
                    let Some(&last) = offs.last() else {
                        bail!("prompt {id} has no messages");
                    };
                    (h[last * d..(last + 1) * d].to_vec(), nt as i64, true)
                } else {
                    (vec![f32::NAN; d], nt as i64, false)
                }
            }
        };
        ids.push(id);
        x.extend_from_slice(&row);
        ntok.push(nt);
        ok.push(good);
        if (n + 1).is_multiple_of(50) {
            write_readouts(&part, &ids, &x, d, &ntok, &ok, &json!({}))?;
            println!(
                "route_gate rows: {}/{} rows, {:.0}s",
                n + 1,
                items.len(),
                t0.elapsed().as_secs_f64()
            );
        }
    }
    let meta = json!({
        "release": a.release.file_name().map(|n| n.to_string_lossy().into_owned()), "block_end": a.block_end,
        "shell": shell.is_some(), "max_tokens": a.max_tokens,
        "encoder": {"implementation": "route_gate rust cpu", "bf16_rounding": a.bf16, "layers": enc.layers.len(), "weights": "BF16"},
        "serializer": "stand-in role blocks (render_blocks)",
    });
    write_readouts(&a.out, &ids, &x, d, &ntok, &ok, &meta)?;
    if part.exists() {
        std::fs::remove_file(&part)?;
    }
    let bad = ok.iter().filter(|&&g| !g).count();
    println!(
        "route_gate rows: wrote {} rows to {} ({bad} over the {}-token cap, left as NaN)",
        ids.len(),
        a.out.display(),
        a.max_tokens
    );
    Ok(0)
}
