<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# LFM3 on Atlas: Liquid AI's hybrid conv/attention model

_What Atlas needs to serve LFM3-1.7B, what was built for it, what the weights' license says, and what is still unmeasured._

**Status: written, not yet run.** The code is on the `port/consumer-opt` branch: the config parser, the weight loader with its two RoPE tables, the FP8 routes, the tool-call and reasoning parsers, the `kernels/ada/lfm3-1.7b` target and the bench tools. Everything marked "inferred" below is arithmetic from the model's shapes or a figure Liquid or a third party published; nothing here is a measurement on the RTX 4060 Ti. The GPU pass (plan queue item Q16) turns each into a number or a failure.

## License of the weights (read this first)

The LFM3 weights belong to Liquid AI and are licensed under the **LFM Open License v1.0**. The terms that decide who may run them:

- Definitions: "'Threshold' shall mean annual revenue of 10 million United States dollars ($10,000,000) or more."
- Section 5(a): "The rights granted under this License for Commercial Use are conditioned upon You or Your Legal Entity not exceeding the Threshold."
- Section 5(b): "Any Commercial Use of the Work or a Derivative Work by a Legal Entity that exceeds the Threshold is not licensed under this Agreement." Above the threshold a separate agreement with Liquid is needed.
- Section 5(c): the threshold does not apply to a qualified non-profit's non-commercial or research use.
- The revenue is that of the whole legal entity, controlled affiliates included. The same license (`license_name: lfm1.0`) covers every LFM3 checkpoint Atlas supports: `LFM3-1.7B`, `LFM3-1.7B-Base` and `LFM3-1.7B-FP8`.

What this means for Atlas, in plain terms (this is a reading of the license, not legal advice):

- **Atlas contains no LFM3 weights and no Liquid code.** The loader reads tensor names and shapes; the model code is translated from Apache-2.0 reference implementations (`LFM3-ATTRIBUTION.md`). An inference engine that loads weights is separable from the Work by the license's own definition of a Derivative Work, so Atlas stays AGPL-3.0-only.
- **The license follows the weights.** Whoever runs LFM3 through Atlas is bound by Liquid's terms, including the revenue threshold. Atlas will not bundle, mirror or download the weights for you: fetch them from Hugging Face under your own acceptance of the license.
- **Atlas says so.** `kernels/ada/lfm3-1.7b/MODEL.toml` carries the license in its header and in a `license` key, and the boot log prints one line naming the license whenever an LFM3 checkpoint loads.
- If benchmark results on LFM3 are published, name the license.

## The model

24 layers, hidden 2048, dense SwiGLU FFN 7680, vocabulary 128,000 (tied embeddings), context 262,144, RMSNorm eps 1e-5.

| Layer kind | Count | At layer indices | What it is |
|---|---|---|---|
| `conv` | 13 | 0, 1, 3, 5, ..., 23 | gated short convolution: `in_proj` to B, C, x (3 x 2048); `u = B * x`; depthwise causal conv, width 7, no activation; `y = C * conv(u)`; `out_proj`. State: the last 6 values of `u` per channel, `[2048, 6]` |
| `full_attention` | 6 | 2, 6, 10, 14, 18, 22 | GQA 16 query / 4 KV heads, head_dim 128, theta 1e6, half the head rotated |
| `sliding_attention` | 5 | 4, 8, 12, 16, 20 | the same, window 256, theta 1e4, the whole head rotated |

Every layer is `h = h + post_operator_norm(op(operator_norm(h)))` then `h = h + post_ffn_norm(ffn(ffn_norm(h)))` (sandwich norms), with a final `model.embedding_norm`. Attention has a per-head RMSNorm on Q and K before RoPE and a per-head sigmoid gate (`gate_proj [16, 2048]`) on the head outputs before `o_proj`. The FFN is `w2(silu(w1 x) * w3 x)`.

Two details that are easy to get wrong:

- **Partial RoPE is a table, not a smaller dimension.** HF keeps the frequency progression of the whole head (`1 / theta^(2j/128)`) for the first 32 of 64 pairs on the global layers and sets the other 32 frequencies to exactly zero. Atlas builds both inverse-frequency tables in f64 on the host and feeds them to the existing table RoPE kernel with `rotary_dim = 128`; a zero frequency is an exact identity rotation, so no new kernel exists for it.
- **The norm weights are plain** (`x * w / rms`), not Qwen's `(1 + w)`. `lfm3` is in `ships_vanilla_norm_weights`.

## What Atlas built

| Piece | Where | Notes |
|---|---|---|
| Config parser | `crates/avarok-core/src/config/parsers/lfm3.rs` | maps `conv` to the existing `LayerType::LinearAttention` (no enum variant added, so no match site elsewhere changes); the conv layer sets `uses_ssm_pool() == false`; refuses `conv_bias`, no sandwich norm, a conv width other than 7, any `rope_type` other than `default`, a sliding theta other than 1e4, a legacy `rope_scaling` |
| Loader | `crates/spark-model/src/weight_loader/lfm3/` | tensor map with shape and dtype validation (286 tensors), RoPE tables, attention on `Qwen3AttentionLayer` (per-head sigmoid gate, sandwich post norms, window), FFN on `DenseFfnLayer`, tied BF16 head |
| Conv layer and kernels | lane L38: `port/lfm3_conv/`, `kernels/ada/common/lfm3_shortconv_ada.cu` | the loader reaches it through one file, `conv_bridge.rs` |
| FP8 | `weight_loader/lfm3/fp8.rs`, row `lfm3_fp8_all` | see below |
| Tool calls and reasoning | `spark-server/src/tool_parser/lfm3.rs`, `reasoning_parser`, `tool_defaults.toml` | parser `lfm3` (qwen3_coder's XML), reasoning `lfm3` (`<think>`, prompt-opened) |
| Kernel target | `kernels/ada/lfm3-1.7b/MODEL.toml` | no symlinks; `kernel_source = "holo-3.1-0.8b"`; sampling 0.1 / top_k 50 / repetition 1.1 |
| Bench tools | `bench/lfm3_ref.py`, `bench/lfm3_qual.py`, `bench/pass.d/L37.sh` | HF reference dump, KL and template gates, 30-case tool gate |

## Serving it

```bash
# BF16, one stream (the weights must be on local disk under your own license acceptance)
spark serve --model-from-path ~/models/lfm3/LFM3-1.7B --model-name lfm3-1.7b \
    --kernel-target lfm3-1.7b --max-seq-len 8192 --max-num-seqs 1

# Liquid's FP8 checkpoint: E4M3 projections stay native, nothing to turn on
spark serve --model-from-path ~/models/lfm3/LFM3-1.7B-FP8 --kernel-target lfm3-1.7b ...

# BF16 checkpoint, every projection to FP8 at load (numerics change, off by default)
AVAROK_LFM3_FP8_ALL=1 spark serve --model-from-path ~/models/lfm3/LFM3-1.7B ...
```

Tensor parallelism, LoRA and speculative decoding are not supported: there is no MTP head, no published drafter, and its tokenizer matches nothing else Atlas serves. The conv layer declines rollback and multi-row verify.

### FP8

| Checkpoint | What happens |
|---|---|
| `LFM3-1.7B-FP8` (compressed-tensors `float-quantized`, 128x128 blocks) | FFN and attention projections load as native E4M3 with their `weight_scale` grids widened to FP32; the conv projections, head gate, norms, embedding and head stay BF16 |
| `LFM3-1.7B`, row off | everything BF16 |
| `LFM3-1.7B`, `lfm3_fp8_all` | FFN and attention projections quantized once at load to the same E4M3 block format (their BF16 is freed); the conv projections gain E4M3 decode copies and keep BF16 for prefill (+0.22 GB); the gate, norms, conv taps, embedding and head stay BF16. The existing `dense_fp8_from_bf16` and `attn_fp8_from_bf16` rows convert only their own class |

E4M3 keeps 3 mantissa bits to BF16's 7, so the row stays off until the receipt: mean top-20 KL against the BF16 serve below 0.003 (`lfm3_qual.py fp8-report`), which also compares Atlas's load-time quantizer with Liquid's own on the same prompts. The head stays BF16 by default (`--lm-head-dtype fp8` is the measured follow-up).

### Templates, tools, reasoning

The checkpoint's own `chat_template.jinja` renders the prompt: ChatML, `<think>` opened by the generation prompt (so output starts inside the reasoning block), earlier reasoning dropped unless `preserve_thinking`, tool schemas rendered by the template, deferred tools in `<additional_tools>`. The `lfm3` tool parser therefore adds no schema or instruction block of its own beyond a `tool_choice` constraint. Calls are XML:

```text
<tool_call>
<function=get_weather>
<parameter=city>
Paris
</parameter>
</function>
</tool_call>
```

Known limits: parameter values are whitespace-trimmed (the model card's own pattern strips only one newline on each side); `tool_reference` content parts (deferred-tool search results) are not part of the OpenAI request schema and are not carried; tool-call grammar is off until `lfm3_qual.py tools` has been run with it on and off. Whether the template renders under Atlas's Jinja environment (it may use string methods such as `startswith`, lane L39) is checked by `lfm3_qual.py templates`.

## Memory and speed (inferred)

Per decode step the model reads about 3.46 GB at BF16 (embedding and head 524 MB, FFN 2.27 GB, attention 231 MB, conv 437 MB) and about 2.0 to 2.3 GB with E4M3 projections. At 288 GB/s and the 73 to 85% of the byte ceiling Atlas reaches on the Holo models that is about 65 tok/s BF16 and 90 to 105 FP8, single stream. KV is 22.5 KB per token in BF16 across the 11 attention layers (5.9 GB at 256K, 2.9 GB with FP8 KV); only the 6 global layers need the full length, but Atlas keeps pages for the sliding layers too (a ring buffer for them is wave 2, row `kv_swa_ring`). The conv state is 13 x 2048 x 6 values per sequence (about 320 KB). The ModelConfig carries the conv geometry on the GDN schema (`ssm_qkvz_size() == 6144`), which sizes an SSM pool of about 0.6 MB per slot per layer that the conv layers do not use.

## What is still to be measured (queue item Q16)

`bench/pass.d/L37.sh` runs these once the weights are on disk and the two HF reference files exist (owner approvals D1, D12):

1. BF16 logits against HF (float32, CPU) on five prompts including one over 256 and one over 1,000 tokens: mean top-20 KL below 0.002, 32 identical greedy tokens.
2. The 30-case tool gate (12 simple, 6 parallel, 6 multiple, 6 irrelevance): every emitted call schema-valid, pass rate at least 0.75.
3. Chat-template prompt-token parity on ten conversations.
4. `bench5.py` and `conc.py` at 1, 2, 4 and 8 streams, BF16 and `lfm3_fp8_all`; Liquid's FP8 checkpoint beside them.
5. Needle in a haystack at 8K, 32K, 128K and 250K, BF16 KV and FP8 KV. Liquid's own RULER at 128K is 65.6 against Qwen3.5-4B's 84.6, so a weaker recall than Qwen is expected, not a bug.

Not covered: the d1 decision models (`lfm3_vl`), the NVFP4 checkpoint (Ada has no FP4 datapath), LFM3 as a draft model for another family (the tokenizers differ).
