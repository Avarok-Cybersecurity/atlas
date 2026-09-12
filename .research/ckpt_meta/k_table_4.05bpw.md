# turboderp/Qwen3.8-Flash-Next-exl3 @ 4.05bpw_h6_ng6 — header-derived K table

Local copy: `gx10-9959:/tank/exl3-ckpt/qwen38-flash-next-4.05bpw` (downloaded 2026-09-01,
`hf download` via ~/.local/bin/hf, huggingface_hub 1.21.0, log `dl-4.05.log`, 15 min 24 s).
Every header below was parsed ON gx10 from the local files with `exl3_headers.py`
(stdlib only; K = trellis last dim / 16 from the tile layout `[in/16, out/16, 16K]`).

## Verification (against `api/models/.../tree/4.05bpw_h6_ng6`, `tree-4.05.json`)

- 27/27 files present, every size matches the API byte-for-byte (107,463,600,896 B = 107.46 GB).
- sha256 of ALL 14 LFS files matches `lfs.oid` (the 9 shards, `ngram_embedding.safetensors`,
  `vision_k6.safetensors`, `model.safetensors.index.json`, `quantization_config.json`,
  `tokenizer.json`); git blob sha1 of all 13 non-LFS files matches `oid`. `BAD COUNT 0`
  (`verify-4.05-full.log`).

## FACTS check — all CONFIRMED, with these refinements

| claim | verdict | header evidence |
|---|---|---|
| experts K=4 | CONFIRMED | 3 x 24,576 routed-expert trellis `[160,40,64]`/`[40,160,64]` -> K=4; mtp experts (3 x 512) also K=4 |
| GDN / attention / shared_expert K=6 | CONFIRMED | `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` (36 each), `self_attn.{q,k,v,o}_proj` (12 each), `mlp.shared_expert.*` (48 each) all last dim 96 -> K=6; the mtp layer's attention + shared_expert are K=6 too |
| lm_head K=6 | CONFIRMED | `lm_head.trellis` I16 `[160,15520,96]` -> `[248320,2560]`, K=6 (shard 9) |
| ngram K=6, monolithic `[320001536,61]` | CONFIRMED | ONE `...ngram_embedding.trellis` I16 `[320001536, 61]` (61 words = 1 + 160*6/16 -> K=6); file metadata `{"format":"exl3_ngram_trellis","version":"1","K":"6","codebook":"mul1","row_dim":"160","rows":"320001536"}`; NO `shard_rows` key (2.05's sharded file has one) |
| vision K=6 in a separate `vision_k6.safetensors`, NOT in the index | CONFIRMED | 987 tensors, 561,389,181 B; 0 `model.visual.*` tensors in any index-listed shard; index lists exactly the 9 `model-0000N-of-00009` files; all 164 vision trellis tensors have last dim 96 -> K=6 |
| vision_k6 dtypes F16 | CONFIRMED (all of them) | EVERY non-trellis vision tensor is F16, including `attn.qkv.weight [3456,1152]`, `attn.qkv.bias [3456]`, `pos_embed.weight [2304,1152]` (BF16 on 2.05), `patch_embed.proj.*`, all norms/biases. Trellis I16, `.suh/.svh` F16, `.mul1` I32 |
| fused `attn.qkv.weight` present for the loader | CONFIRMED | 27 x F16 `[3456,1152]` + bias, alongside the separate trellis `attn.{q,k,v}_proj` (K=6, each with its own `.bias [1152]` F16) |
| `mtp.hyper_connection_mixer` in shard 9, no patch file | CONFIRMED | `mtp.hyper_connection_mixer.{hc_norm,input_mix_weight_down,input_mix_weight_up}.weight` F16 in `model-00009-of-00009`; no `mtp_hyper_connection_mixer_patch.safetensors` in the tree |
| K=7 exists only on 5.05 | CONFIRMED for this branch | K values present on 4.05: {4, 5, 6} — no 2, 3, 7, 8. K=5 appears ONLY on `mtp.fc_embedding` / `mtp.fc_hidden` (`[2560,2560]`, shard 9) |
| packed resident ~68 GB | CONFIRMED | trellis+suh/svh packed = 65.20 GB (experts 62.14 incl. mtp experts, dense 2.72, vision 0.34) + 3.03 GB non-trellis 16-bit (excl. ngram) = **68.23 GB** |
| QSA indexer follows expert K | CONFIRMED | `self_attn.indexer.index_qk_proj` (12 + 1 mtp) K=4 |
| codebook | CONFIRMED mul1 everywhere | all 75,751 `.mul1` scalars = 0x83DCD12D (no mcg/3inst anywhere, incl. vision) |

Additional facts:
- `config.json quantization_config`: `{"quant_method":"exl3","version":"1.4.4","bits":4.05,"head_bits":6,"calibration":{"rows":250,"cols":2048},"out_scales":"always","codebook":"mul1","mtp_bits":4}` — no `vision_bits` key (2.05 map noted the same). `vision_config.intermediate_size` = 4304 while fc1/fc2 trellis pad to 4352 (same stride hazard as `.research/vision_exl3_map.md` section 1).
- `quantization_config.json` (96.6 MB) `tensor_storage` has 74,395 entries and **0 containing `visual`** — the tower was quantized out of band, consistent with it living in its own file.
- Unquantized (non-trellis) linears, as on 2.05: `embed_tokens` BF16 `[248320,2560]` (1.27 GB), all hyper-connection mixers F16, `mlp.gate` F16 `[512,2560]`, `shared_expert_gate` F16, `linear_attn.in_proj_a/b` F16 `[48,2560]`, PLE `key_proj`/`value_proj`/`conv1d` F16, norms BF16/F16 (PLE norms are F16 here, BF16 on 2.05).
- Tensor count 304,113 (2.05: 304,240 — the difference is 128 ngram shards -> 1 monolithic tensor, and the 3 mixer tensors moving into shard 9).

## What this means for the K gates (code as of `wip/exl3-kladder` HEAD)

| family on 4.05 | K | `exl3_native_supported` (K in {2,4}) | GEMV (`exl3_matmul.rs:316`, `exl3_dense.rs:84`: 2..=4) | GEMM envelope (`exl3_matmul.rs:107`: {2,3,4,5,6,8}) | fused `exl3_moe` (`moe_prefill.rs:203`: 2..=4) |
|---|---|---|---|---|---|
| routed experts, mtp experts, QSA indexer | 4 | pass | pass | pass | pass |
| GDN, attention, shared_expert, lm_head, mtp attention/shared, vision | 6 | **FAIL -> materialize to BF16 today** | no kernel | pass (K=6 sh1/sh2/sh3) | shared_expert K=6 would need k0=6 instances (trimmed) |
| `mtp.fc_embedding`, `mtp.fc_hidden` | 5 | FAIL | no kernel | pass | n/a |
| (excluded) K=7 | — | not on this branch | — | — | — |

So widening the gate to K in {2,3,4,5,6} and letting K>4 skip the GEMV tier covers 100% of the 4.05 trellis
families; nothing on this branch needs K=7 or K=8.

---

## Files

| file | size (bytes) | size (GB) | tensors | in index | metadata |
|---|---|---|---|---|---|
| model-00001-of-00009.safetensors | 8059362762 | 8.06 | 30933 | True |  |
| model-00002-of-00009.safetensors | 8067865392 | 8.07 | 37110 | True |  |
| model-00003-of-00009.safetensors | 8062251066 | 8.06 | 37116 | True |  |
| model-00004-of-00009.safetensors | 8067896318 | 8.07 | 37110 | True |  |
| model-00005-of-00009.safetensors | 8062251066 | 8.06 | 37116 | True |  |
| model-00006-of-00009.safetensors | 8067896318 | 8.07 | 37110 | True |  |
| model-00007-of-00009.safetensors | 8062251066 | 8.06 | 37116 | True |  |
| model-00008-of-00009.safetensors | 8067896318 | 8.07 | 37110 | True |  |
| model-00009-of-00009.safetensors | 3191543125 | 3.19 | 12400 | True |  |
| ngram_embedding.safetensors | 39040193720 | 39.04 | 5 | False | {"format": "exl3_ngram_trellis", "version": "1", "K": "6", "codebook": "mul1", "codebook_scale": "heuristic(gamma=3.0... |
| vision_k6.safetensors | 561389181 | 0.56 | 987 | False |  |

Total safetensors bytes: 107310796332 (107.31 GB); tensors: 304113; index lists 9 shard files.

## Trellis (EXL3 linear) families — K = trellis last dim / 16

| family | n | K | trellis dtype | logical [out,in] | codebook | suh/svh dtype | bias | also .weight | packed GB | files |
|---|---|---|---|---|---|---|---|---|---|---|
| `lm_head` | 1 | [6] | ['I16'] | [(248320, 2560)] | mul1 | ['F16'] | - | False | 0.477 | 9 |
| `model.language_model.layers.*.linear_attn.in_proj_qkv` | 36 | [6] | ['I16'] | [(10240, 2560)] | mul1 | ['F16'] | - | False | 0.709 | 1,2,3,4,5,6,7,8 |
| `model.language_model.layers.*.linear_attn.in_proj_z` | 36 | [6] | ['I16'] | [(6144, 2560)] | mul1 | ['F16'] | - | False | 0.425 | 1,2,3,4,5,6,7,8 |
| `model.language_model.layers.*.linear_attn.out_proj` | 36 | [6] | ['I16'] | [(2560, 6144)] | mul1 | ['F16'] | - | False | 0.425 | 1,2,3,4,5,6,7,8 |
| `model.language_model.layers.*.mlp.experts.*.down_proj` | 24576 | [4] | ['I16'] | [(2560, 640)] | mul1 | ['F16'] | - | False | 20.290 | 1,2,3,4,5,6,7,8,9 |
| `model.language_model.layers.*.mlp.experts.*.gate_proj` | 24576 | [4] | ['I16'] | [(640, 2560)] | mul1 | ['F16'] | - | False | 20.290 | 1,2,3,4,5,6,7,8,9 |
| `model.language_model.layers.*.mlp.experts.*.up_proj` | 24576 | [4] | ['I16'] | [(640, 2560)] | mul1 | ['F16'] | - | False | 20.290 | 1,2,3,4,5,6,7,8,9 |
| `model.language_model.layers.*.mlp.shared_expert.down_proj` | 48 | [6] | ['I16'] | [(2560, 640)] | mul1 | ['F16'] | - | False | 0.059 | 1,2,3,4,5,6,7,8,9 |
| `model.language_model.layers.*.mlp.shared_expert.gate_proj` | 48 | [6] | ['I16'] | [(640, 2560)] | mul1 | ['F16'] | - | False | 0.059 | 1,2,3,4,5,6,7,8,9 |
| `model.language_model.layers.*.mlp.shared_expert.up_proj` | 48 | [6] | ['I16'] | [(640, 2560)] | mul1 | ['F16'] | - | False | 0.059 | 1,2,3,4,5,6,7,8,9 |
| `model.language_model.layers.*.self_attn.indexer.index_qk_proj` | 12 | [4] | ['I16'] | [(640, 2560)] | mul1 | ['F16'] | - | False | 0.010 | 1,2,3,4,5,6,7,8,9 |
| `model.language_model.layers.*.self_attn.k_proj` | 12 | [6] | ['I16'] | [(512, 2560)] | mul1 | ['F16'] | - | False | 0.012 | 1,2,3,4,5,6,7,8,9 |
| `model.language_model.layers.*.self_attn.o_proj` | 12 | [6] | ['I16'] | [(2560, 6144)] | mul1 | ['F16'] | - | False | 0.142 | 1,2,3,4,5,6,7,8,9 |
| `model.language_model.layers.*.self_attn.q_proj` | 12 | [6] | ['I16'] | [(12288, 2560)] | mul1 | ['F16'] | - | False | 0.283 | 1,2,3,4,5,6,7,8,9 |
| `model.language_model.layers.*.self_attn.v_proj` | 12 | [6] | ['I16'] | [(512, 2560)] | mul1 | ['F16'] | - | False | 0.012 | 1,2,3,4,5,6,7,8,9 |
| `model.visual.blocks.*.attn.k_proj` | 27 | [6] | ['I16'] | [(1152, 1152)] | mul1 | ['F16'] | ['F16'] | False | 0.027 | vision_k6 |
| `model.visual.blocks.*.attn.proj` | 27 | [6] | ['I16'] | [(1152, 1152)] | mul1 | ['F16'] | ['F16'] | False | 0.027 | vision_k6 |
| `model.visual.blocks.*.attn.q_proj` | 27 | [6] | ['I16'] | [(1152, 1152)] | mul1 | ['F16'] | ['F16'] | False | 0.027 | vision_k6 |
| `model.visual.blocks.*.attn.v_proj` | 27 | [6] | ['I16'] | [(1152, 1152)] | mul1 | ['F16'] | ['F16'] | False | 0.027 | vision_k6 |
| `model.visual.blocks.*.mlp.linear_fc1` | 27 | [6] | ['I16'] | [(4352, 1152)] | mul1 | ['F16'] | ['F16'] | False | 0.102 | vision_k6 |
| `model.visual.blocks.*.mlp.linear_fc2` | 27 | [6] | ['I16'] | [(1152, 4352)] | mul1 | ['F16'] | ['F16'] | False | 0.102 | vision_k6 |
| `model.visual.merger.linear_fc1` | 1 | [6] | ['I16'] | [(4608, 4608)] | mul1 | ['F16'] | ['F16'] | False | 0.016 | vision_k6 |
| `model.visual.merger.linear_fc2` | 1 | [6] | ['I16'] | [(2560, 4608)] | mul1 | ['F16'] | ['F16'] | False | 0.009 | vision_k6 |
| `mtp.fc_embedding` | 1 | [5] | ['I16'] | [(2560, 2560)] | mul1 | ['F16'] | - | False | 0.004 | 9 |
| `mtp.fc_hidden` | 1 | [5] | ['I16'] | [(2560, 2560)] | mul1 | ['F16'] | - | False | 0.004 | 9 |
| `mtp.layers.*.mlp.experts.*.down_proj` | 512 | [4] | ['I16'] | [(2560, 640)] | mul1 | ['F16'] | - | False | 0.423 | 9 |
| `mtp.layers.*.mlp.experts.*.gate_proj` | 512 | [4] | ['I16'] | [(640, 2560)] | mul1 | ['F16'] | - | False | 0.423 | 9 |
| `mtp.layers.*.mlp.experts.*.up_proj` | 512 | [4] | ['I16'] | [(640, 2560)] | mul1 | ['F16'] | - | False | 0.423 | 9 |
| `mtp.layers.*.mlp.shared_expert.down_proj` | 1 | [6] | ['I16'] | [(2560, 640)] | mul1 | ['F16'] | - | False | 0.001 | 9 |
| `mtp.layers.*.mlp.shared_expert.gate_proj` | 1 | [6] | ['I16'] | [(640, 2560)] | mul1 | ['F16'] | - | False | 0.001 | 9 |
| `mtp.layers.*.mlp.shared_expert.up_proj` | 1 | [6] | ['I16'] | [(640, 2560)] | mul1 | ['F16'] | - | False | 0.001 | 9 |
| `mtp.layers.*.self_attn.indexer.index_qk_proj` | 1 | [4] | ['I16'] | [(640, 2560)] | mul1 | ['F16'] | - | False | 0.001 | 9 |
| `mtp.layers.*.self_attn.k_proj` | 1 | [6] | ['I16'] | [(512, 2560)] | mul1 | ['F16'] | - | False | 0.001 | 9 |
| `mtp.layers.*.self_attn.o_proj` | 1 | [6] | ['I16'] | [(2560, 6144)] | mul1 | ['F16'] | - | False | 0.012 | 9 |
| `mtp.layers.*.self_attn.q_proj` | 1 | [6] | ['I16'] | [(12288, 2560)] | mul1 | ['F16'] | - | False | 0.024 | 9 |
| `mtp.layers.*.self_attn.v_proj` | 1 | [6] | ['I16'] | [(512, 2560)] | mul1 | ['F16'] | - | False | 0.001 | 9 |

## Non-trellis tensor families

| family | n | dtypes | shapes (first 3) | bytes | files |
|---|---|---|---|---|---|
| `model.language_model.embed_tokens.weight` | 1 | ['BF16'] | [248320, 2560] | 1271398400 | 1 |
| `model.language_model.hyper_connection_mixer.hc_norm.weight` | 1 | ['F16'] | [10240] | 20480 | 9 |
| `model.language_model.hyper_connection_mixer.input_mix_weight_down.weight` | 1 | ['F16'] | [320, 10240] | 6553600 | 9 |
| `model.language_model.hyper_connection_mixer.input_mix_weight_up.weight` | 1 | ['F16'] | [10240, 320] | 6553600 | 9 |
| `model.language_model.layers.*.attn_hyper_connection.block_inject_weight.weight` | 48 | ['F16'] | [4, 10240] | 3932160 | 1,2,3,4,5,6,7,8,9 |
| `model.language_model.layers.*.attn_hyper_connection.hc_norm.weight` | 48 | ['F16'] | [10240] | 983040 | 1,2,3,4,5,6,7,8,9 |
| `model.language_model.layers.*.attn_hyper_connection.input_mix_weight_down.weight` | 48 | ['F16'] | [320, 10240] | 314572800 | 1,2,3,4,5,6,7,8,9 |
| `model.language_model.layers.*.attn_hyper_connection.input_mix_weight_up.weight` | 48 | ['F16'] | [10240, 320] | 314572800 | 1,2,3,4,5,6,7,8,9 |
| `model.language_model.layers.*.linear_attn.A_log` | 36 | ['BF16'] | [48] | 3456 | 1,2,3,4,5,6,7,8 |
| `model.language_model.layers.*.linear_attn.conv1d.weight` | 36 | ['BF16'] | [10240, 1, 4] | 2949120 | 1,2,3,4,5,6,7,8 |
| `model.language_model.layers.*.linear_attn.dt_bias` | 36 | ['BF16'] | [48] | 3456 | 1,2,3,4,5,6,7,8 |
| `model.language_model.layers.*.linear_attn.in_proj_a.weight` | 36 | ['F16'] | [48, 2560] | 8847360 | 1,2,3,4,5,6,7,8 |
| `model.language_model.layers.*.linear_attn.in_proj_b.weight` | 36 | ['F16'] | [48, 2560] | 8847360 | 1,2,3,4,5,6,7,8 |
| `model.language_model.layers.*.linear_attn.norm.weight` | 36 | ['BF16'] | [128] | 9216 | 1,2,3,4,5,6,7,8 |
| `model.language_model.layers.*.mlp.gate.weight` | 48 | ['F16'] | [512, 2560] | 125829120 | 1,2,3,4,5,6,7,8,9 |
| `model.language_model.layers.*.mlp.shared_expert_gate.weight` | 48 | ['F16'] | [1, 2560] | 245760 | 1,2,3,4,5,6,7,8,9 |
| `model.language_model.layers.*.mlp_hyper_connection.block_inject_weight.weight` | 48 | ['F16'] | [4, 10240] | 3932160 | 1,2,3,4,5,6,7,8,9 |
| `model.language_model.layers.*.mlp_hyper_connection.hc_norm.weight` | 48 | ['F16'] | [10240] | 983040 | 1,2,3,4,5,6,7,8,9 |
| `model.language_model.layers.*.mlp_hyper_connection.input_mix_weight_down.weight` | 48 | ['F16'] | [320, 10240] | 314572800 | 1,2,3,4,5,6,7,8,9 |
| `model.language_model.layers.*.mlp_hyper_connection.input_mix_weight_up.weight` | 48 | ['F16'] | [10240, 320] | 314572800 | 1,2,3,4,5,6,7,8,9 |
| `model.language_model.layers.*.ple.conv1d.weight` | 1 | ['F16'] | [10240, 1, 4] | 81920 | 1 |
| `model.language_model.layers.*.ple.key_proj.weight` | 1 | ['F16'] | [10240, 2560] | 52428800 | 1 |
| `model.language_model.layers.*.ple.norm_conv.weight` | 1 | ['F16'] | [10240] | 20480 | 1 |
| `model.language_model.layers.*.ple.norm_key.weight` | 1 | ['F16'] | [10240] | 20480 | 1 |
| `model.language_model.layers.*.ple.norm_query.weight` | 1 | ['F16'] | [10240] | 20480 | 1 |
| `model.language_model.layers.*.ple.ple_embedding.ngram_embedding.head_bias` | 1 | ['F16'] | [16, 160] | 5120 | ngram_embedding |
| `model.language_model.layers.*.ple.ple_embedding.ngram_embedding.head_offsets` | 1 | ['I64'] | [16] | 128 | ngram_embedding |
| `model.language_model.layers.*.ple.ple_embedding.ngram_embedding.head_vocab_sizes` | 1 | ['I64'] | [16] | 128 | ngram_embedding |
| `model.language_model.layers.*.ple.ple_embedding.ngram_embedding.layer_multipliers` | 1 | ['I64'] | [3] | 24 | ngram_embedding |
| `model.language_model.layers.*.ple.ple_embedding.ngram_embedding.trellis` | 1 | ['I16'] | [320001536, 61] | 39040187392 | ngram_embedding |
| `model.language_model.layers.*.ple.value_proj.weight` | 1 | ['F16'] | [2560, 2560] | 13107200 | 1 |
| `model.language_model.layers.*.self_attn.indexer.k_layernorm.weight` | 12 | ['BF16'] | [128] | 3072 | 1,2,3,4,5,6,7,8,9 |
| `model.language_model.layers.*.self_attn.indexer.q_layernorm.weight` | 12 | ['BF16'] | [128] | 3072 | 1,2,3,4,5,6,7,8,9 |
| `model.language_model.layers.*.self_attn.k_norm.weight` | 12 | ['BF16'] | [256] | 6144 | 1,2,3,4,5,6,7,8,9 |
| `model.language_model.layers.*.self_attn.q_norm.weight` | 12 | ['BF16'] | [256] | 6144 | 1,2,3,4,5,6,7,8,9 |
| `model.visual.blocks.*.attn.qkv.bias` | 27 | ['F16'] | [3456] | 186624 | vision_k6 |
| `model.visual.blocks.*.attn.qkv.weight` | 27 | ['F16'] | [3456, 1152] | 214990848 | vision_k6 |
| `model.visual.blocks.*.norm1.bias` | 27 | ['F16'] | [1152] | 62208 | vision_k6 |
| `model.visual.blocks.*.norm1.weight` | 27 | ['F16'] | [1152] | 62208 | vision_k6 |
| `model.visual.blocks.*.norm2.bias` | 27 | ['F16'] | [1152] | 62208 | vision_k6 |
| `model.visual.blocks.*.norm2.weight` | 27 | ['F16'] | [1152] | 62208 | vision_k6 |
| `model.visual.merger.norm.bias` | 1 | ['F16'] | [1152] | 2304 | vision_k6 |
| `model.visual.merger.norm.weight` | 1 | ['F16'] | [1152] | 2304 | vision_k6 |
| `model.visual.patch_embed.proj.bias` | 1 | ['F16'] | [1152] | 2304 | vision_k6 |
| `model.visual.patch_embed.proj.weight` | 1 | ['F16'] | [1152, 3, 2, 16, 16] | 3538944 | vision_k6 |
| `model.visual.pos_embed.weight` | 1 | ['F16'] | [2304, 1152] | 5308416 | vision_k6 |
| `mtp.hyper_connection_mixer.hc_norm.weight` | 1 | ['F16'] | [10240] | 20480 | 9 |
| `mtp.hyper_connection_mixer.input_mix_weight_down.weight` | 1 | ['F16'] | [320, 10240] | 6553600 | 9 |
| `mtp.hyper_connection_mixer.input_mix_weight_up.weight` | 1 | ['F16'] | [10240, 320] | 6553600 | 9 |
| `mtp.layers.*.attn_hyper_connection.block_inject_weight.weight` | 1 | ['F16'] | [4, 10240] | 81920 | 9 |
| `mtp.layers.*.attn_hyper_connection.hc_norm.weight` | 1 | ['F16'] | [10240] | 20480 | 9 |
| `mtp.layers.*.attn_hyper_connection.input_mix_weight_down.weight` | 1 | ['F16'] | [320, 10240] | 6553600 | 9 |
| `mtp.layers.*.attn_hyper_connection.input_mix_weight_up.weight` | 1 | ['F16'] | [10240, 320] | 6553600 | 9 |
| `mtp.layers.*.mlp.gate.weight` | 1 | ['F16'] | [512, 2560] | 2621440 | 9 |
| `mtp.layers.*.mlp.shared_expert_gate.weight` | 1 | ['F16'] | [1, 2560] | 5120 | 9 |
| `mtp.layers.*.mlp_hyper_connection.block_inject_weight.weight` | 1 | ['F16'] | [4, 10240] | 81920 | 9 |
| `mtp.layers.*.mlp_hyper_connection.hc_norm.weight` | 1 | ['F16'] | [10240] | 20480 | 9 |
| `mtp.layers.*.mlp_hyper_connection.input_mix_weight_down.weight` | 1 | ['F16'] | [320, 10240] | 6553600 | 9 |
| `mtp.layers.*.mlp_hyper_connection.input_mix_weight_up.weight` | 1 | ['F16'] | [10240, 320] | 6553600 | 9 |
| `mtp.layers.*.self_attn.indexer.k_layernorm.weight` | 1 | ['BF16'] | [128] | 256 | 9 |
| `mtp.layers.*.self_attn.indexer.q_layernorm.weight` | 1 | ['BF16'] | [128] | 256 | 9 |
| `mtp.layers.*.self_attn.k_norm.weight` | 1 | ['BF16'] | [256] | 512 | 9 |
| `mtp.layers.*.self_attn.q_norm.weight` | 1 | ['BF16'] | [256] | 512 | 9 |
| `mtp.pre_fc_norm_embedding.weight` | 1 | ['BF16'] | [2560] | 5120 | 9 |
| `mtp.pre_fc_norm_hidden.weight` | 1 | ['F16'] | [10240] | 20480 | 9 |

## K summary by coarse family

- **GDN (linear_attn)**: K = [6]
- **QSA indexer**: K = [4]
- **attention**: K = [6]
- **lm_head**: K = [6]
- **mtp: mtp.fc_embedding**: K = [5]
- **mtp: mtp.fc_hidden**: K = [5]
- **mtp: mtp.layers.*.mlp.experts.*.down_proj**: K = [4]
- **mtp: mtp.layers.*.mlp.experts.*.gate_proj**: K = [4]
- **mtp: mtp.layers.*.mlp.experts.*.up_proj**: K = [4]
- **mtp: mtp.layers.*.mlp.shared_expert.down_proj**: K = [6]
- **mtp: mtp.layers.*.mlp.shared_expert.gate_proj**: K = [6]
- **mtp: mtp.layers.*.mlp.shared_expert.up_proj**: K = [6]
- **routed experts**: K = [4]
- **shared_expert**: K = [6]
- **vision**: K = [6]

## ngram_embedding.safetensors

metadata: `{"format": "exl3_ngram_trellis", "version": "1", "K": "6", "codebook": "mul1", "codebook_scale": "heuristic(gamma=3.00, hi=0.92)", "row_dim": "160", "rows": "320001536", "source": "/mnt/str/models/qwen3.8-flash-next/hf"}`

- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.head_bias` F16 [16, 160]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.head_offsets` I64 [16]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.head_vocab_sizes` I64 [16]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.layer_multipliers` I64 [3]
- trellis tensors: 1; first: `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.trellis` I16 [320001536, 61]; words/row=61 -> K=(words-1)/10=6

## vision_k6.safetensors inventory (all tensors, grouped)

| family | n | dtypes | shapes |
|---|---|---|---|
| `model.visual.blocks.*.attn.k_proj.bias` | 27 | ['F16'] | [1152] |
| `model.visual.blocks.*.attn.k_proj.mul1` | 27 | ['I32'] | [] |
| `model.visual.blocks.*.attn.k_proj.suh` | 27 | ['F16'] | [1152] |
| `model.visual.blocks.*.attn.k_proj.svh` | 27 | ['F16'] | [1152] |
| `model.visual.blocks.*.attn.k_proj.trellis` | 27 | ['I16'] | [72, 72, 96] |
| `model.visual.blocks.*.attn.proj.bias` | 27 | ['F16'] | [1152] |
| `model.visual.blocks.*.attn.proj.mul1` | 27 | ['I32'] | [] |
| `model.visual.blocks.*.attn.proj.suh` | 27 | ['F16'] | [1152] |
| `model.visual.blocks.*.attn.proj.svh` | 27 | ['F16'] | [1152] |
| `model.visual.blocks.*.attn.proj.trellis` | 27 | ['I16'] | [72, 72, 96] |
| `model.visual.blocks.*.attn.q_proj.bias` | 27 | ['F16'] | [1152] |
| `model.visual.blocks.*.attn.q_proj.mul1` | 27 | ['I32'] | [] |
| `model.visual.blocks.*.attn.q_proj.suh` | 27 | ['F16'] | [1152] |
| `model.visual.blocks.*.attn.q_proj.svh` | 27 | ['F16'] | [1152] |
| `model.visual.blocks.*.attn.q_proj.trellis` | 27 | ['I16'] | [72, 72, 96] |
| `model.visual.blocks.*.attn.qkv.bias` | 27 | ['F16'] | [3456] |
| `model.visual.blocks.*.attn.qkv.weight` | 27 | ['F16'] | [3456, 1152] |
| `model.visual.blocks.*.attn.v_proj.bias` | 27 | ['F16'] | [1152] |
| `model.visual.blocks.*.attn.v_proj.mul1` | 27 | ['I32'] | [] |
| `model.visual.blocks.*.attn.v_proj.suh` | 27 | ['F16'] | [1152] |
| `model.visual.blocks.*.attn.v_proj.svh` | 27 | ['F16'] | [1152] |
| `model.visual.blocks.*.attn.v_proj.trellis` | 27 | ['I16'] | [72, 72, 96] |
| `model.visual.blocks.*.mlp.linear_fc1.bias` | 27 | ['F16'] | [4352] |
| `model.visual.blocks.*.mlp.linear_fc1.mul1` | 27 | ['I32'] | [] |
| `model.visual.blocks.*.mlp.linear_fc1.suh` | 27 | ['F16'] | [1152] |
| `model.visual.blocks.*.mlp.linear_fc1.svh` | 27 | ['F16'] | [4352] |
| `model.visual.blocks.*.mlp.linear_fc1.trellis` | 27 | ['I16'] | [72, 272, 96] |
| `model.visual.blocks.*.mlp.linear_fc2.bias` | 27 | ['F16'] | [1152] |
| `model.visual.blocks.*.mlp.linear_fc2.mul1` | 27 | ['I32'] | [] |
| `model.visual.blocks.*.mlp.linear_fc2.suh` | 27 | ['F16'] | [4352] |
| `model.visual.blocks.*.mlp.linear_fc2.svh` | 27 | ['F16'] | [1152] |
| `model.visual.blocks.*.mlp.linear_fc2.trellis` | 27 | ['I16'] | [272, 72, 96] |
| `model.visual.blocks.*.norm1.bias` | 27 | ['F16'] | [1152] |
| `model.visual.blocks.*.norm1.weight` | 27 | ['F16'] | [1152] |
| `model.visual.blocks.*.norm2.bias` | 27 | ['F16'] | [1152] |
| `model.visual.blocks.*.norm2.weight` | 27 | ['F16'] | [1152] |
| `model.visual.merger.linear_fc1.bias` | 1 | ['F16'] | [4608] |
| `model.visual.merger.linear_fc1.mul1` | 1 | ['I32'] | [] |
| `model.visual.merger.linear_fc1.suh` | 1 | ['F16'] | [4608] |
| `model.visual.merger.linear_fc1.svh` | 1 | ['F16'] | [4608] |
| `model.visual.merger.linear_fc1.trellis` | 1 | ['I16'] | [288, 288, 96] |
| `model.visual.merger.linear_fc2.bias` | 1 | ['F16'] | [2560] |
| `model.visual.merger.linear_fc2.mul1` | 1 | ['I32'] | [] |
| `model.visual.merger.linear_fc2.suh` | 1 | ['F16'] | [4608] |
| `model.visual.merger.linear_fc2.svh` | 1 | ['F16'] | [2560] |
| `model.visual.merger.linear_fc2.trellis` | 1 | ['I16'] | [288, 160, 96] |
| `model.visual.merger.norm.bias` | 1 | ['F16'] | [1152] |
| `model.visual.merger.norm.weight` | 1 | ['F16'] | [1152] |
| `model.visual.patch_embed.proj.bias` | 1 | ['F16'] | [1152] |
| `model.visual.patch_embed.proj.weight` | 1 | ['F16'] | [1152, 3, 2, 16, 16] |
| `model.visual.pos_embed.weight` | 1 | ['F16'] | [2304, 1152] |

model.visual.* tensors in index-listed shards: 0

---

# Extras (codebook scalars, quantization config, packed totals)

## codebook scalars (.mul1 I32; 0x83DCD12D = mul1, 0xCBAC1FED = mcg)
- `model.language_model.layers.0.mlp.experts.0.gate_proj.mul1` = 0x83DCD12D
- `model.language_model.layers.0.linear_attn.in_proj_qkv.mul1` = 0x83DCD12D
- `lm_head.mul1` = 0x83DCD12D
- `mtp.fc_hidden.mul1` = 0x83DCD12D
- `model.visual.blocks.0.mlp.linear_fc2.mul1` = 0x83DCD12D
- `model.visual.merger.linear_fc1.mul1` = 0x83DCD12D

## distinct .mul1 values across ALL files
- 0x83DCD12D: 75751 tensors

## config.json quantization_config
```json
{
 "quant_method": "exl3",
 "version": "1.4.4",
 "bits": 4.05,
 "head_bits": 6,
 "calibration": {
  "rows": 250,
  "cols": 2048
 },
 "out_scales": "always",
 "codebook": "mul1",
 "mtp_bits": 4
}
```
vision_config.intermediate_size=4304 hidden_size=1152 depth=27

## quantization_config.json (top-level)
- `quant_method`: "exl3"
- `version`: "1.4.4"
- `bits`: 4.05
- `head_bits`: 6
- `calibration`: dict len=2
- `out_scales`: "always"
- `codebook`: "mul1"
- `mtp_bits`: 4
- `tensor_storage`: dict len=74395
  - `tensor_storage[hc_expand]` = {"stored_tensors": {}}
  - `tensor_storage[lm_head]` = {"stored_tensors": {"lm_head.suh": {"shape": [2560], "n_bytes": 5120, "dtype": "torch.float16"}, "lm_head.svh": {"shape": [248320], "n_bytes": 496640, "dtype": "torch.float16"}, "lm_head.mul1": {"shape": [], "n_bytes": 4, "dtype": "torch.int32"}, "lm_head.trellis": {"shape": [160, 15520, 96], "n_byt
  - `tensor_storage[model.language_model.layers.0.mlp.experts.0.gate_proj]` = {"stored_tensors": {"model.language_model.layers.0.mlp.experts.0.gate_proj.suh": {"shape": [2560], "n_bytes": 5120, "dtype": "torch.float16"}, "model.language_model.layers.0.mlp.experts.0.gate_proj.svh": {"shape": [640], "n_bytes": 1280, "dtype": "torch.float16"}, "model.language_model.layers.0.mlp.
  - `tensor_storage[model.language_model.layers.0.linear_attn.in_proj_qkv]` = {"stored_tensors": {"model.language_model.layers.0.linear_attn.in_proj_qkv.suh": {"shape": [2560], "n_bytes": 5120, "dtype": "torch.float16"}, "model.language_model.layers.0.linear_attn.in_proj_qkv.svh": {"shape": [10240], "n_bytes": 20480, "dtype": "torch.float16"}, "model.language_model.layers.0.l
  - entries containing 'visual': 0

## packed-size totals (from headers; decimal GB)
- dense (GDN+attn+shared+lm_head+mtp+indexer): 2.72
- routed experts (incl. mtp experts): 62.14
- vision (trellis+suh/svh): 0.34
- all trellis linears packed: 65.20
- non-trellis tensors (excl. ngram file), on-disk bytes: 3.03 (of which vision_k6: 0.224)
- resident estimate if everything trellis is kept packed + non-trellis as 16-bit: 68.23 GB (ngram file NVMe-faulted, not resident)

---

# Verification log

```
OK  size         1696 exp         1696 .gitattributes  gitsha1 OK
OK  size         3235 exp         3235 LICENSE  gitsha1 OK
OK  size        65155 exp        65155 README.md  gitsha1 OK
OK  size         8952 exp         8952 chat_template.jinja  gitsha1 OK
OK  size         5055 exp         5055 config.json  gitsha1 OK
OK  size          202 exp          202 generation_config.json  gitsha1 OK
OK  size      3353259 exp      3353259 merges.txt  gitsha1 OK
OK  size   8059362762 exp   8059362762 model-00001-of-00009.safetensors  sha256 OK
OK  size   8067865392 exp   8067865392 model-00002-of-00009.safetensors  sha256 OK
OK  size   8062251066 exp   8062251066 model-00003-of-00009.safetensors  sha256 OK
OK  size   8067896318 exp   8067896318 model-00004-of-00009.safetensors  sha256 OK
OK  size   8062251066 exp   8062251066 model-00005-of-00009.safetensors  sha256 OK
OK  size   8067896318 exp   8067896318 model-00006-of-00009.safetensors  sha256 OK
OK  size   8062251066 exp   8062251066 model-00007-of-00009.safetensors  sha256 OK
OK  size   8067896318 exp   8067896318 model-00008-of-00009.safetensors  sha256 OK
OK  size   3191543125 exp   3191543125 model-00009-of-00009.safetensors  sha256 OK
OK  size     32677781 exp     32677781 model.safetensors.index.json  sha256 OK
OK  size  39040193720 exp  39040193720 ngram_embedding.safetensors  sha256 OK
OK  size          390 exp          390 preprocessor_config.json  gitsha1 OK
OK  size       324183 exp       324183 qbench_prompts.json  gitsha1 OK
OK  size       212719 exp       212719 qbench_prompts.md  gitsha1 OK
OK  size     96601545 exp     96601545 quantization_config.json  sha256 OK
OK  size     12809320 exp     12809320 tokenizer.json  sha256 OK
OK  size        17928 exp        17928 tokenizer_config.json  gitsha1 OK
OK  size          385 exp          385 video_preprocessor_config.json  gitsha1 OK
OK  size    561389181 exp    561389181 vision_k6.safetensors  sha256 OK
OK  size      6722759 exp      6722759 vocab.json  gitsha1 OK
BAD COUNT 0
```
