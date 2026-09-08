# turboderp/Qwen3.8-Flash-Next-exl3 @ 3.05bpw_h5_ng5 — header-derived K table

Local copy: `gx10-9959:/tank/exl3-ckpt/qwen38-flash-next-3.05bpw` (downloaded 2026-09-01 after the
4.05 branch verified; `hf download`, log `dl-3.05.log`, 12 min 11 s). Headers parsed ON gx10 with
`exl3_headers.py` (same method as `ck4_k_table.md`). /tank after both downloads: 274 GB free.

## Verification (against `tree-3.05.json`)

- 25/25 files present, all sizes match (85,139,442,313 B = 85.14 GB).
- Full sha256 of every LFS file (7 shards, ngram, mtp patch, index, quantization_config, tokenizer.json)
  and git blob sha1 of every non-LFS file: all OK, `BAD COUNT 0` (`verify-3.05-full.log`).

## K ladder — matches the `.research/vision_exl3_map.md` per-branch row exactly

| family | K | note |
|---|---|---|
| routed experts (3 x 24,576) + mtp experts (3 x 512) + QSA indexer (12 + 1) | **3** | `mtp_bits: 3` in config |
| GDN `linear_attn.*`, attention `self_attn.{q,k,v,o}_proj`, shared_expert, lm_head, mtp attention/shared | **5** | `head_bits: 5` |
| vision (164 trellis tensors; `attn.{q,k,v}_proj`, `attn.proj`, fc1/fc2, merger fc1/fc2) | **5** | `vision_bits: 5` IS present in `quantization_config` on this branch (absent on 4.05) |
| `mtp.fc_embedding`, `mtp.fc_hidden` | **4** | |
| ngram | **5** | SHARDED: 128 x `shard_{i}.trellis` I16 `[2500012, 51]` (51 words -> K=5), metadata has `shard_rows: 2500012` — the 2.05-style layout, NOT 4.05's monolithic one |

Layout differences vs 4.05 (all as predicted by the map):
- Vision lives in the index-listed LAST shard `model-00007-of-00007` (987 `model.visual.*` tensors) — no `vision_*.safetensors` sidecar; the fused `attn.qkv.weight` is **BF16** `[3456,1152]` (F16 on 4.05), `pos_embed.weight` BF16.
- `mtp_hyper_connection_mixer_patch.safetensors` (13.1 MB, 3 F16 tensors) is present and NOT index-listed; the mixer is not duplicated in the shards.
- Codebook: all 75,751 `.mul1` scalars = 0x83DCD12D (mul1).
- Tensor count 304,240 (identical to 2.05).
- Packed: trellis+aux 49.27 GB (experts 46.72, dense 2.27, vision 0.28) + 3.03 GB non-trellis 16-bit = **52.31 GB** resident estimate (ngram NVMe-faulted). Matches the map's 52.3.

## Gate implications

K values present: {3, 4, 5}. Experts at K=3 pass the GEMM envelope, the fused `exl3_moe` k0 switch (2..=4) and GEMV (2..=4) but FAIL `exl3_native_supported` (K in {2,4}) — so today every family on this branch (experts K=3 included) materializes to BF16/NVFP4. Dense/vision/ngram at K=5 need the GEMM-only path (no GEMV instance); `mtp.fc_*` at K=4 pass everything. Nothing on this branch needs K=7/8.

---

## Files

| file | size (bytes) | size (GB) | tensors | in index | metadata |
|---|---|---|---|---|---|
| model-00001-of-00007.safetensors | 8494539639 | 8.49 | 43301 | True |  |
| model-00002-of-00007.safetensors | 8175800810 | 8.18 | 49484 | True |  |
| model-00003-of-00007.safetensors | 8175819368 | 8.18 | 49484 | True |  |
| model-00004-of-00007.safetensors | 8175819368 | 8.18 | 49484 | True |  |
| model-00005-of-00007.safetensors | 8175819368 | 8.18 | 49484 | True |  |
| model-00006-of-00007.safetensors | 8175819368 | 8.18 | 49484 | True |  |
| model-00007-of-00007.safetensors | 2959581433 | 2.96 | 13384 | True |  |
| mtp_hyper_connection_mixer_patch.safetensors | 13128048 | 0.01 | 3 | False |  |
| ngram_embedding.safetensors | 32640183408 | 32.64 | 132 | False | {"format": "exl3_ngram_trellis", "version": "1", "K": "5", "codebook": "mul1", "codebook_scale": "heuristic(gamma=3.0... |

Total safetensors bytes: 84986510810 (84.99 GB); tensors: 304240; index lists 7 shard files.

## Trellis (EXL3 linear) families — K = trellis last dim / 16

| family | n | K | trellis dtype | logical [out,in] | codebook | suh/svh dtype | bias | also .weight | packed GB | files |
|---|---|---|---|---|---|---|---|---|---|---|
| `lm_head` | 1 | [5] | ['I16'] | [(248320, 2560)] | mul1 | ['F16'] | - | False | 0.398 | 7 |
| `model.language_model.layers.*.linear_attn.in_proj_qkv` | 36 | [5] | ['I16'] | [(10240, 2560)] | mul1 | ['F16'] | - | False | 0.591 | 1,2,3,4,5,6 |
| `model.language_model.layers.*.linear_attn.in_proj_z` | 36 | [5] | ['I16'] | [(6144, 2560)] | mul1 | ['F16'] | - | False | 0.355 | 1,2,3,4,5,6 |
| `model.language_model.layers.*.linear_attn.out_proj` | 36 | [5] | ['I16'] | [(2560, 6144)] | mul1 | ['F16'] | - | False | 0.355 | 1,2,3,4,5,6 |
| `model.language_model.layers.*.mlp.experts.*.down_proj` | 24576 | [3] | ['I16'] | [(2560, 640)] | mul1 | ['F16'] | - | False | 15.257 | 1,2,3,4,5,6,7 |
| `model.language_model.layers.*.mlp.experts.*.gate_proj` | 24576 | [3] | ['I16'] | [(640, 2560)] | mul1 | ['F16'] | - | False | 15.257 | 1,2,3,4,5,6,7 |
| `model.language_model.layers.*.mlp.experts.*.up_proj` | 24576 | [3] | ['I16'] | [(640, 2560)] | mul1 | ['F16'] | - | False | 15.257 | 1,2,3,4,5,6,7 |
| `model.language_model.layers.*.mlp.shared_expert.down_proj` | 48 | [5] | ['I16'] | [(2560, 640)] | mul1 | ['F16'] | - | False | 0.049 | 1,2,3,4,5,6,7 |
| `model.language_model.layers.*.mlp.shared_expert.gate_proj` | 48 | [5] | ['I16'] | [(640, 2560)] | mul1 | ['F16'] | - | False | 0.049 | 1,2,3,4,5,6,7 |
| `model.language_model.layers.*.mlp.shared_expert.up_proj` | 48 | [5] | ['I16'] | [(640, 2560)] | mul1 | ['F16'] | - | False | 0.049 | 1,2,3,4,5,6,7 |
| `model.language_model.layers.*.self_attn.indexer.index_qk_proj` | 12 | [3] | ['I16'] | [(640, 2560)] | mul1 | ['F16'] | - | False | 0.007 | 1,2,3,4,5,6,7 |
| `model.language_model.layers.*.self_attn.k_proj` | 12 | [5] | ['I16'] | [(512, 2560)] | mul1 | ['F16'] | - | False | 0.010 | 1,2,3,4,5,6,7 |
| `model.language_model.layers.*.self_attn.o_proj` | 12 | [5] | ['I16'] | [(2560, 6144)] | mul1 | ['F16'] | - | False | 0.118 | 1,2,3,4,5,6,7 |
| `model.language_model.layers.*.self_attn.q_proj` | 12 | [5] | ['I16'] | [(12288, 2560)] | mul1 | ['F16'] | - | False | 0.236 | 1,2,3,4,5,6,7 |
| `model.language_model.layers.*.self_attn.v_proj` | 12 | [5] | ['I16'] | [(512, 2560)] | mul1 | ['F16'] | - | False | 0.010 | 1,2,3,4,5,6,7 |
| `model.visual.blocks.*.attn.k_proj` | 27 | [5] | ['I16'] | [(1152, 1152)] | mul1 | ['F16'] | ['F16'] | False | 0.023 | 7 |
| `model.visual.blocks.*.attn.proj` | 27 | [5] | ['I16'] | [(1152, 1152)] | mul1 | ['F16'] | ['F16'] | False | 0.023 | 7 |
| `model.visual.blocks.*.attn.q_proj` | 27 | [5] | ['I16'] | [(1152, 1152)] | mul1 | ['F16'] | ['F16'] | False | 0.023 | 7 |
| `model.visual.blocks.*.attn.v_proj` | 27 | [5] | ['I16'] | [(1152, 1152)] | mul1 | ['F16'] | ['F16'] | False | 0.023 | 7 |
| `model.visual.blocks.*.mlp.linear_fc1` | 27 | [5] | ['I16'] | [(4352, 1152)] | mul1 | ['F16'] | ['F16'] | False | 0.085 | 7 |
| `model.visual.blocks.*.mlp.linear_fc2` | 27 | [5] | ['I16'] | [(1152, 4352)] | mul1 | ['F16'] | ['F16'] | False | 0.085 | 7 |
| `model.visual.merger.linear_fc1` | 1 | [5] | ['I16'] | [(4608, 4608)] | mul1 | ['F16'] | ['F16'] | False | 0.013 | 7 |
| `model.visual.merger.linear_fc2` | 1 | [5] | ['I16'] | [(2560, 4608)] | mul1 | ['F16'] | ['F16'] | False | 0.007 | 7 |
| `mtp.fc_embedding` | 1 | [4] | ['I16'] | [(2560, 2560)] | mul1 | ['F16'] | - | False | 0.003 | 7 |
| `mtp.fc_hidden` | 1 | [4] | ['I16'] | [(2560, 2560)] | mul1 | ['F16'] | - | False | 0.003 | 7 |
| `mtp.layers.*.mlp.experts.*.down_proj` | 512 | [3] | ['I16'] | [(2560, 640)] | mul1 | ['F16'] | - | False | 0.318 | 7 |
| `mtp.layers.*.mlp.experts.*.gate_proj` | 512 | [3] | ['I16'] | [(640, 2560)] | mul1 | ['F16'] | - | False | 0.318 | 7 |
| `mtp.layers.*.mlp.experts.*.up_proj` | 512 | [3] | ['I16'] | [(640, 2560)] | mul1 | ['F16'] | - | False | 0.318 | 7 |
| `mtp.layers.*.mlp.shared_expert.down_proj` | 1 | [5] | ['I16'] | [(2560, 640)] | mul1 | ['F16'] | - | False | 0.001 | 7 |
| `mtp.layers.*.mlp.shared_expert.gate_proj` | 1 | [5] | ['I16'] | [(640, 2560)] | mul1 | ['F16'] | - | False | 0.001 | 7 |
| `mtp.layers.*.mlp.shared_expert.up_proj` | 1 | [5] | ['I16'] | [(640, 2560)] | mul1 | ['F16'] | - | False | 0.001 | 7 |
| `mtp.layers.*.self_attn.indexer.index_qk_proj` | 1 | [3] | ['I16'] | [(640, 2560)] | mul1 | ['F16'] | - | False | 0.001 | 7 |
| `mtp.layers.*.self_attn.k_proj` | 1 | [5] | ['I16'] | [(512, 2560)] | mul1 | ['F16'] | - | False | 0.001 | 7 |
| `mtp.layers.*.self_attn.o_proj` | 1 | [5] | ['I16'] | [(2560, 6144)] | mul1 | ['F16'] | - | False | 0.010 | 7 |
| `mtp.layers.*.self_attn.q_proj` | 1 | [5] | ['I16'] | [(12288, 2560)] | mul1 | ['F16'] | - | False | 0.020 | 7 |
| `mtp.layers.*.self_attn.v_proj` | 1 | [5] | ['I16'] | [(512, 2560)] | mul1 | ['F16'] | - | False | 0.001 | 7 |

## Non-trellis tensor families

| family | n | dtypes | shapes (first 3) | bytes | files |
|---|---|---|---|---|---|
| `model.language_model.embed_tokens.weight` | 1 | ['BF16'] | [248320, 2560] | 1271398400 | 1 |
| `model.language_model.hyper_connection_mixer.hc_norm.weight` | 1 | ['F16'] | [10240] | 20480 | 7 |
| `model.language_model.hyper_connection_mixer.input_mix_weight_down.weight` | 1 | ['F16'] | [320, 10240] | 6553600 | 7 |
| `model.language_model.hyper_connection_mixer.input_mix_weight_up.weight` | 1 | ['F16'] | [10240, 320] | 6553600 | 7 |
| `model.language_model.layers.*.attn_hyper_connection.block_inject_weight.weight` | 48 | ['F16'] | [4, 10240] | 3932160 | 1,2,3,4,5,6,7 |
| `model.language_model.layers.*.attn_hyper_connection.hc_norm.weight` | 48 | ['F16'] | [10240] | 983040 | 1,2,3,4,5,6,7 |
| `model.language_model.layers.*.attn_hyper_connection.input_mix_weight_down.weight` | 48 | ['F16'] | [320, 10240] | 314572800 | 1,2,3,4,5,6,7 |
| `model.language_model.layers.*.attn_hyper_connection.input_mix_weight_up.weight` | 48 | ['F16'] | [10240, 320] | 314572800 | 1,2,3,4,5,6,7 |
| `model.language_model.layers.*.linear_attn.A_log` | 36 | ['BF16'] | [48] | 3456 | 1,2,3,4,5,6 |
| `model.language_model.layers.*.linear_attn.conv1d.weight` | 36 | ['BF16'] | [10240, 1, 4] | 2949120 | 1,2,3,4,5,6 |
| `model.language_model.layers.*.linear_attn.dt_bias` | 36 | ['BF16'] | [48] | 3456 | 1,2,3,4,5,6 |
| `model.language_model.layers.*.linear_attn.in_proj_a.weight` | 36 | ['F16'] | [48, 2560] | 8847360 | 1,2,3,4,5,6 |
| `model.language_model.layers.*.linear_attn.in_proj_b.weight` | 36 | ['F16'] | [48, 2560] | 8847360 | 1,2,3,4,5,6 |
| `model.language_model.layers.*.linear_attn.norm.weight` | 36 | ['BF16'] | [128] | 9216 | 1,2,3,4,5,6 |
| `model.language_model.layers.*.mlp.gate.weight` | 48 | ['F16'] | [512, 2560] | 125829120 | 1,2,3,4,5,6,7 |
| `model.language_model.layers.*.mlp.shared_expert_gate.weight` | 48 | ['F16'] | [1, 2560] | 245760 | 1,2,3,4,5,6,7 |
| `model.language_model.layers.*.mlp_hyper_connection.block_inject_weight.weight` | 48 | ['F16'] | [4, 10240] | 3932160 | 1,2,3,4,5,6,7 |
| `model.language_model.layers.*.mlp_hyper_connection.hc_norm.weight` | 48 | ['F16'] | [10240] | 983040 | 1,2,3,4,5,6,7 |
| `model.language_model.layers.*.mlp_hyper_connection.input_mix_weight_down.weight` | 48 | ['F16'] | [320, 10240] | 314572800 | 1,2,3,4,5,6,7 |
| `model.language_model.layers.*.mlp_hyper_connection.input_mix_weight_up.weight` | 48 | ['F16'] | [10240, 320] | 314572800 | 1,2,3,4,5,6,7 |
| `model.language_model.layers.*.ple.conv1d.weight` | 1 | ['F16'] | [10240, 1, 4] | 81920 | 1 |
| `model.language_model.layers.*.ple.key_proj.weight` | 1 | ['F16'] | [10240, 2560] | 52428800 | 1 |
| `model.language_model.layers.*.ple.norm_conv.weight` | 1 | ['BF16'] | [10240] | 20480 | 1 |
| `model.language_model.layers.*.ple.norm_key.weight` | 1 | ['BF16'] | [10240] | 20480 | 1 |
| `model.language_model.layers.*.ple.norm_query.weight` | 1 | ['BF16'] | [10240] | 20480 | 1 |
| `model.language_model.layers.*.ple.ple_embedding.ngram_embedding.head_bias` | 1 | ['F16'] | [16, 160] | 5120 | ngram_embedding |
| `model.language_model.layers.*.ple.ple_embedding.ngram_embedding.head_offsets` | 1 | ['I64'] | [16] | 128 | ngram_embedding |
| `model.language_model.layers.*.ple.ple_embedding.ngram_embedding.head_vocab_sizes` | 1 | ['I64'] | [16] | 128 | ngram_embedding |
| `model.language_model.layers.*.ple.ple_embedding.ngram_embedding.layer_multipliers` | 1 | ['I64'] | [3] | 24 | ngram_embedding |
| `model.language_model.layers.*.ple.ple_embedding.ngram_embedding.shard_*.trellis` | 128 | ['I16'] | [2500012, 51] | 32640156672 | ngram_embedding |
| `model.language_model.layers.*.ple.value_proj.weight` | 1 | ['F16'] | [2560, 2560] | 13107200 | 1 |
| `model.language_model.layers.*.self_attn.indexer.k_layernorm.weight` | 12 | ['BF16'] | [128] | 3072 | 1,2,3,4,5,6,7 |
| `model.language_model.layers.*.self_attn.indexer.q_layernorm.weight` | 12 | ['BF16'] | [128] | 3072 | 1,2,3,4,5,6,7 |
| `model.language_model.layers.*.self_attn.k_norm.weight` | 12 | ['BF16'] | [256] | 6144 | 1,2,3,4,5,6,7 |
| `model.language_model.layers.*.self_attn.q_norm.weight` | 12 | ['BF16'] | [256] | 6144 | 1,2,3,4,5,6,7 |
| `model.visual.blocks.*.attn.qkv.bias` | 27 | ['BF16'] | [3456] | 186624 | 7 |
| `model.visual.blocks.*.attn.qkv.weight` | 27 | ['BF16'] | [3456, 1152] | 214990848 | 7 |
| `model.visual.blocks.*.norm1.bias` | 27 | ['F16'] | [1152] | 62208 | 7 |
| `model.visual.blocks.*.norm1.weight` | 27 | ['F16'] | [1152] | 62208 | 7 |
| `model.visual.blocks.*.norm2.bias` | 27 | ['F16'] | [1152] | 62208 | 7 |
| `model.visual.blocks.*.norm2.weight` | 27 | ['F16'] | [1152] | 62208 | 7 |
| `model.visual.merger.norm.bias` | 1 | ['F16'] | [1152] | 2304 | 7 |
| `model.visual.merger.norm.weight` | 1 | ['F16'] | [1152] | 2304 | 7 |
| `model.visual.patch_embed.proj.bias` | 1 | ['F16'] | [1152] | 2304 | 7 |
| `model.visual.patch_embed.proj.weight` | 1 | ['F16'] | [1152, 3, 2, 16, 16] | 3538944 | 7 |
| `model.visual.pos_embed.weight` | 1 | ['BF16'] | [2304, 1152] | 5308416 | 7 |
| `mtp.hyper_connection_mixer.hc_norm.weight` | 1 | ['F16'] | [10240] | 20480 | mtp_hyper_connection_mixer_patch |
| `mtp.hyper_connection_mixer.input_mix_weight_down.weight` | 1 | ['F16'] | [320, 10240] | 6553600 | mtp_hyper_connection_mixer_patch |
| `mtp.hyper_connection_mixer.input_mix_weight_up.weight` | 1 | ['F16'] | [10240, 320] | 6553600 | mtp_hyper_connection_mixer_patch |
| `mtp.layers.*.attn_hyper_connection.block_inject_weight.weight` | 1 | ['F16'] | [4, 10240] | 81920 | 7 |
| `mtp.layers.*.attn_hyper_connection.hc_norm.weight` | 1 | ['F16'] | [10240] | 20480 | 7 |
| `mtp.layers.*.attn_hyper_connection.input_mix_weight_down.weight` | 1 | ['F16'] | [320, 10240] | 6553600 | 7 |
| `mtp.layers.*.attn_hyper_connection.input_mix_weight_up.weight` | 1 | ['F16'] | [10240, 320] | 6553600 | 7 |
| `mtp.layers.*.mlp.gate.weight` | 1 | ['F16'] | [512, 2560] | 2621440 | 7 |
| `mtp.layers.*.mlp.shared_expert_gate.weight` | 1 | ['F16'] | [1, 2560] | 5120 | 7 |
| `mtp.layers.*.mlp_hyper_connection.block_inject_weight.weight` | 1 | ['F16'] | [4, 10240] | 81920 | 7 |
| `mtp.layers.*.mlp_hyper_connection.hc_norm.weight` | 1 | ['F16'] | [10240] | 20480 | 7 |
| `mtp.layers.*.mlp_hyper_connection.input_mix_weight_down.weight` | 1 | ['F16'] | [320, 10240] | 6553600 | 7 |
| `mtp.layers.*.mlp_hyper_connection.input_mix_weight_up.weight` | 1 | ['F16'] | [10240, 320] | 6553600 | 7 |
| `mtp.layers.*.self_attn.indexer.k_layernorm.weight` | 1 | ['BF16'] | [128] | 256 | 7 |
| `mtp.layers.*.self_attn.indexer.q_layernorm.weight` | 1 | ['BF16'] | [128] | 256 | 7 |
| `mtp.layers.*.self_attn.k_norm.weight` | 1 | ['BF16'] | [256] | 512 | 7 |
| `mtp.layers.*.self_attn.q_norm.weight` | 1 | ['BF16'] | [256] | 512 | 7 |
| `mtp.pre_fc_norm_embedding.weight` | 1 | ['BF16'] | [2560] | 5120 | 7 |
| `mtp.pre_fc_norm_hidden.weight` | 1 | ['F16'] | [10240] | 20480 | 7 |

## K summary by coarse family

- **GDN (linear_attn)**: K = [5]
- **QSA indexer**: K = [3]
- **attention**: K = [5]
- **lm_head**: K = [5]
- **mtp: mtp.fc_embedding**: K = [4]
- **mtp: mtp.fc_hidden**: K = [4]
- **mtp: mtp.layers.*.mlp.experts.*.down_proj**: K = [3]
- **mtp: mtp.layers.*.mlp.experts.*.gate_proj**: K = [3]
- **mtp: mtp.layers.*.mlp.experts.*.up_proj**: K = [3]
- **mtp: mtp.layers.*.mlp.shared_expert.down_proj**: K = [5]
- **mtp: mtp.layers.*.mlp.shared_expert.gate_proj**: K = [5]
- **mtp: mtp.layers.*.mlp.shared_expert.up_proj**: K = [5]
- **routed experts**: K = [3]
- **shared_expert**: K = [5]
- **vision**: K = [5]

## ngram_embedding.safetensors

metadata: `{"format": "exl3_ngram_trellis", "version": "1", "K": "5", "codebook": "mul1", "codebook_scale": "heuristic(gamma=3.00, hi=0.95)", "row_dim": "160", "rows": "320001536", "shard_rows": "2500012", "source": "/mnt/str/models/qwen3.8-flash-next/hf"}`

- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.head_bias` F16 [16, 160]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.head_offsets` I64 [16]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.head_vocab_sizes` I64 [16]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.layer_multipliers` I64 [3]
- trellis tensors: 128; first: `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.trellis` I16 [2500012, 51]; last: `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_99.trellis` [2500012, 51]; words/row=51 -> K=(words-1)/10=5

## vision_k6.safetensors inventory (all tensors, grouped)

| family | n | dtypes | shapes |
|---|---|---|---|

model.visual.* tensors in index-listed shards: 987

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
 "bits": 3.05,
 "head_bits": 5,
 "calibration": {
  "rows": 250,
  "cols": 2048
 },
 "out_scales": "always",
 "codebook": "mul1",
 "vision_bits": 5,
 "mtp_bits": 3
}
```
vision_config.intermediate_size=4304 hidden_size=1152 depth=27

## quantization_config.json (top-level)
- `quant_method`: "exl3"
- `version`: "1.4.4"
- `bits`: 3.05
- `head_bits`: 5
- `calibration`: dict len=2
- `out_scales`: "always"
- `codebook`: "mul1"
- `vision_bits`: 5
- `mtp_bits`: 3
- `tensor_storage`: dict len=74398
  - `tensor_storage[hc_expand]` = {"stored_tensors": {}}
  - `tensor_storage[lm_head]` = {"stored_tensors": {"lm_head.suh": {"shape": [2560], "n_bytes": 5120, "dtype": "torch.float16"}, "lm_head.svh": {"shape": [248320], "n_bytes": 496640, "dtype": "torch.float16"}, "lm_head.mul1": {"shape": [], "n_bytes": 4, "dtype": "torch.int32"}, "lm_head.trellis": {"shape": [160, 15520, 80], "n_byt
  - `tensor_storage[model.language_model.layers.0.mlp.experts.0.gate_proj]` = {"stored_tensors": {"model.language_model.layers.0.mlp.experts.0.gate_proj.suh": {"shape": [2560], "n_bytes": 5120, "dtype": "torch.float16"}, "model.language_model.layers.0.mlp.experts.0.gate_proj.svh": {"shape": [640], "n_bytes": 1280, "dtype": "torch.float16"}, "model.language_model.layers.0.mlp.
  - `tensor_storage[model.language_model.layers.0.linear_attn.in_proj_qkv]` = {"stored_tensors": {"model.language_model.layers.0.linear_attn.in_proj_qkv.suh": {"shape": [2560], "n_bytes": 5120, "dtype": "torch.float16"}, "model.language_model.layers.0.linear_attn.in_proj_qkv.svh": {"shape": [10240], "n_bytes": 20480, "dtype": "torch.float16"}, "model.language_model.layers.0.l
  - entries containing 'visual': 0

## packed-size totals (from headers; decimal GB)
- dense (GDN+attn+shared+lm_head+mtp+indexer): 2.27
- routed experts (incl. mtp experts): 46.72
- vision (trellis+suh/svh): 0.28
- all trellis linears packed: 49.27
- non-trellis tensors (excl. ngram file), on-disk bytes: 3.03 (of which vision_k6: 0.000)
- resident estimate if everything trellis is kept packed + non-trellis as 16-bit: 52.31 GB (ngram file NVMe-faulted, not resident)

---

# Verification log

```
OK  size         1696 exp         1696 .gitattributes  gitsha1 OK
OK  size         3235 exp         3235 LICENSE  gitsha1 OK
OK  size        65155 exp        65155 README.md  gitsha1 OK
OK  size         8952 exp         8952 chat_template.jinja  gitsha1 OK
OK  size         5081 exp         5081 config.json  gitsha1 OK
OK  size          202 exp          202 generation_config.json  gitsha1 OK
OK  size      3353259 exp      3353259 merges.txt  gitsha1 OK
OK  size   8494539639 exp   8494539639 model-00001-of-00007.safetensors  sha256 OK
OK  size   8175800810 exp   8175800810 model-00002-of-00007.safetensors  sha256 OK
OK  size   8175819368 exp   8175819368 model-00003-of-00007.safetensors  sha256 OK
OK  size   8175819368 exp   8175819368 model-00004-of-00007.safetensors  sha256 OK
OK  size   8175819368 exp   8175819368 model-00005-of-00007.safetensors  sha256 OK
OK  size   8175819368 exp   8175819368 model-00006-of-00007.safetensors  sha256 OK
OK  size   2959581433 exp   2959581433 model-00007-of-00007.safetensors  sha256 OK
OK  size     32762977 exp     32762977 model.safetensors.index.json  sha256 OK
OK  size     13128048 exp     13128048 mtp_hyper_connection_mixer_patch.safetensors  sha256 OK
OK  size  32640183408 exp  32640183408 ngram_embedding.safetensors  sha256 OK
OK  size          390 exp          390 preprocessor_config.json  gitsha1 OK
OK  size       324183 exp       324183 qbench_prompts.json  gitsha1 OK
OK  size       212719 exp       212719 qbench_prompts.md  gitsha1 OK
OK  size     96643262 exp     96643262 quantization_config.json  sha256 OK
OK  size     12809320 exp     12809320 tokenizer.json  sha256 OK
OK  size        17928 exp        17928 tokenizer_config.json  gitsha1 OK
OK  size          385 exp          385 video_preprocessor_config.json  gitsha1 OK
OK  size      6722759 exp      6722759 vocab.json  gitsha1 OK
BAD COUNT 0
```
