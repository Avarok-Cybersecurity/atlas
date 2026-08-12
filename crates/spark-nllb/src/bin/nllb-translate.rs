// SPDX-License-Identifier: AGPL-3.0-only

//! CPU NLLB-200 translation CLI.
//!
//! Usage:
//!
//! ```text
//! nllb-translate --model <dir> [--gguf <file.gguf>] [--lora <adapter_dir>] \
//!     --src eng_Latn --tgt fra_Latn "Hello, world."
//! ```
//!
//! `<dir>` is a safetensors NLLB checkpoint (e.g. a local clone of
//! `MonumentalSystems/nllb-200-3.3B`); it always supplies `config.json` +
//! `tokenizer.json`. Without `--gguf`, weights load from that directory's
//! safetensors. With `--gguf <file.gguf>` the weights instead come from an NLLB
//! GGUF (arch `nllb`, F16/F32) while config/tokenizer still come from `--model`.
//! `--lora <adapter_dir>` optionally applies a HuggingFace PEFT LoRA adapter
//! (`adapter_config.json` + `adapter_model.safetensors`) as a runtime delta
//! (safetensors weights only). Prints the translation to stdout.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use spark_nllb::NllbModel;
use tokenizers::Tokenizer;

struct Args {
    model: PathBuf,
    gguf: Option<PathBuf>,
    lora: Option<PathBuf>,
    src: String,
    tgt: String,
    text: String,
    max_new: usize,
    beams: usize,
}

fn parse_args() -> Result<Args> {
    let mut model = None;
    let mut gguf = None;
    let mut lora = None;
    let mut src = "eng_Latn".to_string();
    let mut tgt = "fra_Latn".to_string();
    let mut max_new = 128usize;
    let mut beams = 5usize; // NLLB default num_beams
    let mut text_parts = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model" | "-m" => {
                model = Some(PathBuf::from(it.next().context("--model needs a value")?))
            }
            "--gguf" => gguf = Some(PathBuf::from(it.next().context("--gguf needs a value")?)),
            "--lora" => lora = Some(PathBuf::from(it.next().context("--lora needs a value")?)),
            "--src" => src = it.next().context("--src needs a value")?,
            "--tgt" => tgt = it.next().context("--tgt needs a value")?,
            "--max-new" => max_new = it.next().context("--max-new needs a value")?.parse()?,
            "--beams" => beams = it.next().context("--beams needs a value")?.parse()?,
            other => text_parts.push(other.to_string()),
        }
    }
    let model = model.context("--model <dir> is required")?;
    if text_parts.is_empty() {
        bail!("no input text provided");
    }
    Ok(Args {
        model,
        gguf,
        lora,
        src,
        tgt,
        text: text_parts.join(" "),
        max_new,
        beams: beams.max(1),
    })
}

fn main() -> Result<()> {
    let args = parse_args()?;

    eprintln!("[nllb] loading {} ...", args.model.display());
    let t0 = std::time::Instant::now();
    let model = match (&args.gguf, &args.lora) {
        (Some(gguf), None) => NllbModel::load_gguf(&args.model, gguf)?,
        (Some(_), Some(_)) => {
            bail!("--gguf and --lora cannot be combined (LoRA path is safetensors-only)")
        }
        (None, Some(lora_dir)) => NllbModel::load_dir_with_lora(&args.model, lora_dir)?,
        (None, None) => NllbModel::load_dir(&args.model)?,
    };
    eprintln!(
        "[nllb] loaded in {:.1}s (d_model={}, enc={}, dec={}, lora_modules={})",
        t0.elapsed().as_secs_f32(),
        model.cfg.d_model,
        model.cfg.encoder_layers,
        model.cfg.decoder_layers,
        model.lora_modules(),
    );

    let tok = Tokenizer::from_file(args.model.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("loading tokenizer.json: {e}"))?;

    let src_id = tok
        .token_to_id(&args.src)
        .with_context(|| format!("unknown src lang token '{}'", args.src))?;
    let tgt_id = tok
        .token_to_id(&args.tgt)
        .with_context(|| format!("unknown tgt lang token '{}'", args.tgt))?;
    let eos = model.cfg.eos_token_id;

    // NLLB source format: [src_lang] + subwords + </s>.
    let enc = tok
        .encode(args.text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
    let mut input_ids = vec![src_id];
    input_ids.extend_from_slice(enc.get_ids());
    input_ids.push(eos);

    let t1 = std::time::Instant::now();
    // NLLB defaults: length_penalty=1.0, early_stopping=false.
    let out_ids = model.generate_beam(&input_ids, tgt_id, args.beams, args.max_new, 1.0, false);
    let dt = t1.elapsed().as_secs_f32();

    let text = tok
        .decode(&out_ids, true)
        .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
    println!("{text}");
    eprintln!(
        "[nllb] {} tokens in {:.1}s ({:.1} tok/s)",
        out_ids.len(),
        dt,
        out_ids.len() as f32 / dt
    );
    Ok(())
}
