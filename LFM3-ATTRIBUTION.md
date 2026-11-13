# LFM3 support: sources and attribution

Atlas runs Liquid AI's LFM3 text models (`LiquidAI/LFM3-1.7B`, `-Base`, `-FP8`). Support is
written for Atlas in Rust and CUDA from the published architecture; **no Liquid, HuggingFace or
vLLM source is copied**. The files below translate the architecture (tensor names, the block
structure, the RoPE rule, the config fields) from the Apache-2.0 reference implementations, so
each carries the translation header of master plan table 3.8 (T-HF). The model WEIGHTS are a
separate matter and are not part of Atlas: see "Weights license".

## Upstream sources

| Source | License | Revision used | What was read |
|---|---|---|---|
| HuggingFace transformers `models/lfm3`, as shipped in the model repo `LiquidAI/LFM3-1.7B` (`modeling_lfm3.py`, `configuration_lfm3.py`, `lfm3_conv.py`) | Apache-2.0, "Copyright 2026 The HuggingFace Team" (per-file header) | transformers 5.12.1 (the `transformers_version` of the checkpoint's `config.json`), read 2026-11-12 | the numerical reference |
| vLLM `vllm/model_executor/models/lfm3.py`, `vllm/model_executor/layers/mamba/short_conv.py` | Apache-2.0, "Copyright contributors to the vLLM project" | commit 8a36f08 | structure cross-check (attention, decoder layer, conv state shape; the conv as a recurrent layer) |
| llama.cpp `src/models/lfm3.cpp`, `conversion/lfm3.py` | MIT, "Copyright (c) 2023-2026 The ggml authors" | as read in research report 78 | tensor-name and zero-tail RoPE cross-check (idea only) |
| LiquidAI/LFM3-1.7B model card, `tokenizer_config.json` (`response_template`), `generation_config.json` | LFM Open License v1.0 for the repository; facts only | 2026-11-12 | tool-call wire format, sampling defaults, stop token |

HF `modeling_lfm3.py` sits in a repository whose `LICENSE` is the LFM Open License, while the
file's own header is Apache-2.0. Atlas copies none of it; it records facts and formulas.

## Files and the upstream they derive from

| Atlas file | Upstream (file and lines) | Change notes |
|---|---|---|
| `crates/avarok-core/src/config/parsers/lfm3.rs` | HF `configuration_lfm3.py:22-125` (`Lfm3Config` fields, defaults, `validate_architecture`) | Python dataclass to a Rust JSON parser onto `ModelConfig`; math-changing fields refused; GDN-schema sentinel for the conv layers |
| `crates/avarok-core/src/config/tests/lfm3.rs` | the real `LiquidAI/LFM3-1.7B` `config.json` | new tests |
| `crates/spark-model/src/weight_loader/lfm3/rope_tables.rs` | HF `modeling_lfm3.py:94-113` (`compute_default_rope_parameters`, the zero-tail partial RoPE) | torch f32 table to host f64 narrowed once; uploaded for the table RoPE kernel |
| `crates/spark-model/src/weight_loader/lfm3/names.rs` | HF `modeling_lfm3.py:149-157, 213-273, 332-499` (module and parameter names, shapes); safetensors header of the checkpoint | tensor map with shape and dtype validation |
| `crates/spark-model/src/weight_loader/lfm3/{attention,ffn,load_layers,mod}.rs` | HF `modeling_lfm3.py:213-273, 445-499`; vLLM `lfm3.py:112-266` | block structure mapped onto Atlas's `Qwen3AttentionLayer` and `DenseFfnLayer` |
| `crates/spark-model/src/weight_loader/lfm3/{conv_bridge,fp8}.rs` | HF `modeling_lfm3.py:332-443` (gated short conv tensors); Liquid's FP8 checkpoint layout | adapter to lane L38's conv layer; E4M3 block-scale routes |
| `crates/spark-server/src/tool_parser/lfm3.rs`, `reasoning_parser` entry | model card "Tool calling"; `tokenizer_config.json` `response_template` | wire format identical to qwen3_coder; parser adds only a name and a tool-choice-only prompt |
| `kernels/ada/lfm3-1.7b/MODEL.toml` | model card "Sampling parameters", `generation_config.json`, `config.json` | sampling presets, target selector |
| `kernels/ada/common/lfm3_shortconv_ada.cu` (lane L38) | HF `Lfm3ShortConv.forward` (`modeling_lfm3.py`) and the decode and packed-sequence prefill of `lfm3_conv.py`; vLLM `vllm/model_executor/layers/mamba/short_conv.py` | Triton/Python to CUDA; the gating (B * x, then C * v) fused around the 7-tap convolution; token-major input; per-sequence BF16 `[hidden, 6]` window; packed-sequence prefill |
| `crates/spark-model/src/port/lfm3_conv/{mod,init,trait_impl,decode,prefill,launch,state}.rs` (lane L38) | HF `Lfm3DecoderLayer.forward` (sandwich norms) and `Lfm3ShortConv.forward`; vLLM `lfm3.py` and `short_conv.py` (the conv layer as a recurrent layer with a `[hidden, 6]` state) | Python to Rust `TransformerLayer`; the layer owns its window (`uses_ssm_pool() == false`); FP8 projection tier for decode; no rollback or verify-multi |
| `crates/spark-model/src/port/lfm3_conv/{reference,tests,tests_mock}.rs` (lane L38) | HF `Lfm3ShortConv.forward` and `Lfm3DecoderLayer.forward` (the f32 oracle) | CPU reference written from the definition; new tests |

## Weights license (not part of Atlas)

The LFM3 weights are licensed by Liquid AI under the **LFM Open License v1.0**. Quoting its
definitions and section 5: "'Threshold' shall mean annual revenue of 10 million United States
dollars ($10,000,000) or more", and "Any Commercial Use of the Work or a Derivative Work by a
Legal Entity that exceeds the Threshold is not licensed under this Agreement". Atlas bundles no
LFM3 weights, mirrors none, and adds no Liquid code, so the AGPL-3.0 of Atlas is not affected
by the weights' terms; the weights' license follows the weights to whoever runs them. Anyone
serving LFM3 through Atlas must read the license. `docs/porting/lfm3.md` has the summary and the
boot log names the license whenever an LFM3 checkpoint loads.

## Notices

`NOTICE` carries one section for the HuggingFace and vLLM Apache-2.0 sources (fragment
`NOTICE.md` of lane L37); neither project ships a NOTICE file that must be reproduced beyond
that. `CITATIONS.md` carries the LFM3 entry. The Apache-2.0 text is `LICENSES/Apache-2.0.txt`.
