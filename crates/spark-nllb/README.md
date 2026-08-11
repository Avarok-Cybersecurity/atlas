<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# spark-nllb

Self-contained **CPU** reference runtime for **NLLB-200 / M2M-100** — a
translation-focused *encoder-decoder* (seq2seq) transformer.

## Why a separate crate

Atlas's production engine is **GPU-only** (`GpuBackend` has no CPU
implementation), and its *generic* `TransformerLayer`/paged-KV/scheduler stack
is decoder-only — it assumes causal autoregressive generation. NLLB is a seq2seq
model (bidirectional encoder + decoder cross-attention + sinusoidal absolute
positions + ReLU FFN + biased LayerNorm), so the *generic* `spark-model` marker
loader for `model_type = "m2m_100" | "nllb"` deliberately fails fast.

NLLB **is** served on the GPU, but through a *dedicated* encoder-decoder runtime
(`spark-model` `model/nllb`, `NllbGpuModel`) that `build_model` selects before
the generic loader — including straight from an NLLB **GGUF** (arch `nllb`),
whose `enc/dec.blk.N.*` tensors the GPU loader remaps in
`spark-runtime` `weights::gguf::names` (the same map this crate applies on CPU).

This crate is the *"load it at least on CPU"* path: a dependency-light, fp32
port of HuggingFace `M2M100ForConditionalGeneration` that actually loads the
checkpoint and produces translations, with **no CUDA/Metal dependency**. It is
validated bit-faithfully against `transformers` (see `tests/reference.rs`).

## Weights

NLLB-200 ships as PyTorch `.bin` (pickle), which Atlas's safetensors-only
loader cannot read. A converted fp32 safetensors copy lives at:

- **`MonumentalSystems/nllb-200-3.3B`** (HuggingFace)

Download it (or any safetensors NLLB checkpoint) to a local directory.

## Usage

```bash
cargo run -p spark-nllb --release --bin nllb-translate -- \
    --model /path/to/nllb-200-3.3B-st \
    --src eng_Latn --tgt fra_Latn \
    "Hello, world. How are you today?"
# -> Bonjour, comment vous portez-vous aujourd'hui ?
```

`--src` / `--tgt` are NLLB FLORES-200 language codes (`eng_Latn`, `fra_Latn`,
`spa_Latn`, `deu_Latn`, …). `--beams N` sets the beam width (default `5`, the
NLLB default; `--beams 1` is greedy).

### GGUF weights

Weights can instead be read from an NLLB **GGUF** file (architecture `nllb`,
F16/F32 — e.g. `acceldium/nllb-200-3.3B-GGUF`) with `--gguf`:

```bash
nllb-translate --model /path/to/nllb-200-3.3B-st \
    --gguf /path/to/nllb-3.3B.gguf \
    --src eng_Latn --tgt fra_Latn "Hello, how are you?"
# -> Bonjour, comment allez vous ?   (byte-identical to the safetensors path)
```

`--model` still supplies `config.json` + `tokenizer.json` (the GGUF does not
carry the HF tokenizer this runtime needs). The GGUF is parsed by a small
self-contained reader (`src/gguf.rs`, F16→f32 / F32 only, no K-quant path); its
`enc/dec.blk.N.*` tensor names are remapped to the HuggingFace M2M-100 keys by
`weights::map_gguf_name`. The learned `position_embd.weight` is skipped — the
model regenerates sinusoidal positions. Because the shipped GGUF is F16, outputs
match the fp32 safetensors path (verified identical on the test sentences).

## Validation

```bash
NLLB_MODEL_DIR=/path/to/nllb-200-3.3B-st cargo test -p spark-nllb --release
```

Asserts the encoder hidden-state checksum and the exact greedy token sequence
against the HuggingFace reference. Skips silently when `NLLB_MODEL_DIR` is unset
(so CPU CI without the weights stays green).

## LoRA adapters

A HuggingFace PEFT LoRA adapter can be applied as a **runtime delta** (never
merged into the base weights), matching the GPU engine's philosophy:

```bash
nllb-translate --model /path/to/nllb-200-3.3B-st \
    --lora /path/to/adapter \
    --src eng_Latn --tgt fra_Latn "Hello, world."
```

`<adapter>` is a standard PEFT directory (`adapter_config.json` +
`adapter_model.safetensors`). For each adapted projection the output becomes
`y = x·Wᵀ + b + scale·(x·Aᵀ)·Bᵀ`, with `scale = alpha/r` (or `alpha/√r` under
rsLoRA). Every projection is covered: encoder + decoder self-attention
`q/k/v/out_proj`, decoder cross-attention `q/out_proj` + the cached cross
`k/v_proj`, and the FFN `fc1/fc2`. Modules the adapter does not target fall
through to the base weight unchanged; a `B == 0` (freshly-initialised) adapter
is a byte-exact no-op. See `src/lora.rs`.

## Status / next steps

- ✅ CPU fp32 encoder-decoder forward, greedy + beam search (NLLB defaults:
  `num_beams=5`, `length_penalty=1.0`, `early_stopping=false`), exact-match
  with HF `transformers` on both.
- ✅ PEFT LoRA adapters (runtime delta on every projection), validated
  end-to-end against the real 3.3B checkpoint (zero-B no-op + live-delta).
- ✅ GPU path (CUDA / GB10 + Metal): full encoder + decoder with cross-attention,
  KV cache, and beam search — bit-faithful to this CPU reference and exact-match
  to HF. Lives in `spark-model` (`examples/nllb_cuda_*`, `examples/nllb_metal_*`,
  `kernels/{gb10,metal}/nllb-200-3.3b` + `common/nllb_encoder.{cu,metal}`), with a
  bf16 tensor-core decode pipeline, M=1 GEMV decode, and request/beam batching.
  LoRA is not yet wired into the GPU path — this crate's CPU delta is the
  reference for that follow-up.
