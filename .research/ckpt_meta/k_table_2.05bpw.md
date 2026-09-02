# EXL3 checkpoint header table — `/tank/exl3-ckpt/qwen38-flash-next-2.05bpw`

## Files

| file | size (bytes) | size (GB) | tensors | in index | metadata |
|---|---|---|---|---|---|
| model-00001-of-00005.safetensors | 8336836652 | 8.34 | 61859 | True |  |
| model-00002-of-00005.safetensors | 8397310276 | 8.40 | 74226 | True |  |
| model-00003-of-00005.safetensors | 8397310276 | 8.40 | 74226 | True |  |
| model-00004-of-00005.safetensors | 8397310276 | 8.40 | 74226 | True |  |
| model-00005-of-00005.safetensors | 2880152226 | 2.88 | 19568 | True |  |
| mtp_hyper_connection_mixer_patch.safetensors | 13128048 | 0.01 | 3 | False |  |
| ngram_embedding.safetensors | 26240152672 | 26.24 | 132 | False | {"format": "exl3_ngram_trellis", "version": "1", "K": "4", "codebook": "mul1", "codebook_scale": "heuristic(gamma=3.0... |

Total safetensors bytes: 62662200426 (62.66 GB); tensors: 304240; index lists 5 shard files.

## Trellis (EXL3 linear) families — K = trellis last dim / 16

| family | n | K | trellis dtype | logical [out,in] | codebook | suh/svh dtype | bias | also .weight | packed GB | files |
|---|---|---|---|---|---|---|---|---|---|---|
| `lm_head` | 1 | [4] | ['I16'] | [(248320, 2560)] | mul1 | ['F16'] | - | False | 0.318 | 00005-of-00005 |
| `model.language_model.layers.*.linear_attn.in_proj_qkv` | 36 | [4] | ['I16'] | [(10240, 2560)] | mul1 | ['F16'] | - | False | 0.473 | 00001-of-00005,00002-of-00005,00003-of-00005,00004-of-00005,00005-of-00005 |
| `model.language_model.layers.*.linear_attn.in_proj_z` | 36 | [4] | ['I16'] | [(6144, 2560)] | mul1 | ['F16'] | - | False | 0.284 | 00001-of-00005,00002-of-00005,00003-of-00005,00004-of-00005,00005-of-00005 |
| `model.language_model.layers.*.linear_attn.out_proj` | 36 | [4] | ['I16'] | [(2560, 6144)] | mul1 | ['F16'] | - | False | 0.284 | 00001-of-00005,00002-of-00005,00003-of-00005,00004-of-00005,00005-of-00005 |
| `model.language_model.layers.*.mlp.experts.*.down_proj` | 24576 | [2] | ['I16'] | [(2560, 640)] | mul1 | ['F16'] | - | False | 10.224 | 00001-of-00005,00002-of-00005,00003-of-00005,00004-of-00005,00005-of-00005 |
| `model.language_model.layers.*.mlp.experts.*.gate_proj` | 24576 | [2] | ['I16'] | [(640, 2560)] | mul1 | ['F16'] | - | False | 10.224 | 00001-of-00005,00002-of-00005,00003-of-00005,00004-of-00005,00005-of-00005 |
| `model.language_model.layers.*.mlp.experts.*.up_proj` | 24576 | [2] | ['I16'] | [(640, 2560)] | mul1 | ['F16'] | - | False | 10.224 | 00001-of-00005,00002-of-00005,00003-of-00005,00004-of-00005,00005-of-00005 |
| `model.language_model.layers.*.mlp.shared_expert.down_proj` | 48 | [4] | ['I16'] | [(2560, 640)] | mul1 | ['F16'] | - | False | 0.040 | 00001-of-00005,00002-of-00005,00003-of-00005,00004-of-00005,00005-of-00005 |
| `model.language_model.layers.*.mlp.shared_expert.gate_proj` | 48 | [4] | ['I16'] | [(640, 2560)] | mul1 | ['F16'] | - | False | 0.040 | 00001-of-00005,00002-of-00005,00003-of-00005,00004-of-00005,00005-of-00005 |
| `model.language_model.layers.*.mlp.shared_expert.up_proj` | 48 | [4] | ['I16'] | [(640, 2560)] | mul1 | ['F16'] | - | False | 0.040 | 00001-of-00005,00002-of-00005,00003-of-00005,00004-of-00005,00005-of-00005 |
| `model.language_model.layers.*.ple.ple_embedding.ngram_embedding.shard_*` | 128 | [2] FRAC! | ['I16'] | [(656, 40000192)] | ? | None | - | False | 26.240 | ngram_embedding |
| `model.language_model.layers.*.self_attn.indexer.index_qk_proj` | 12 | [2] | ['I16'] | [(640, 2560)] | mul1 | ['F16'] | - | False | 0.005 | 00001-of-00005,00002-of-00005,00003-of-00005,00004-of-00005,00005-of-00005 |
| `model.language_model.layers.*.self_attn.k_proj` | 12 | [4] | ['I16'] | [(512, 2560)] | mul1 | ['F16'] | - | False | 0.008 | 00001-of-00005,00002-of-00005,00003-of-00005,00004-of-00005,00005-of-00005 |
| `model.language_model.layers.*.self_attn.o_proj` | 12 | [4] | ['I16'] | [(2560, 6144)] | mul1 | ['F16'] | - | False | 0.095 | 00001-of-00005,00002-of-00005,00003-of-00005,00004-of-00005,00005-of-00005 |
| `model.language_model.layers.*.self_attn.q_proj` | 12 | [4] | ['I16'] | [(12288, 2560)] | mul1 | ['F16'] | - | False | 0.189 | 00001-of-00005,00002-of-00005,00003-of-00005,00004-of-00005,00005-of-00005 |
| `model.language_model.layers.*.self_attn.v_proj` | 12 | [4] | ['I16'] | [(512, 2560)] | mul1 | ['F16'] | - | False | 0.008 | 00001-of-00005,00002-of-00005,00003-of-00005,00004-of-00005,00005-of-00005 |
| `model.visual.blocks.*.attn.k_proj` | 27 | [4] | ['I16'] | [(1152, 1152)] | mul1 | ['F16'] | ['F16'] | False | 0.018 | 00005-of-00005 |
| `model.visual.blocks.*.attn.proj` | 27 | [4] | ['I16'] | [(1152, 1152)] | mul1 | ['F16'] | ['F16'] | False | 0.018 | 00005-of-00005 |
| `model.visual.blocks.*.attn.q_proj` | 27 | [4] | ['I16'] | [(1152, 1152)] | mul1 | ['F16'] | ['F16'] | False | 0.018 | 00005-of-00005 |
| `model.visual.blocks.*.attn.v_proj` | 27 | [4] | ['I16'] | [(1152, 1152)] | mul1 | ['F16'] | ['F16'] | False | 0.018 | 00005-of-00005 |
| `model.visual.blocks.*.mlp.linear_fc1` | 27 | [4] | ['I16'] | [(4352, 1152)] | mul1 | ['F16'] | ['F16'] | False | 0.068 | 00005-of-00005 |
| `model.visual.blocks.*.mlp.linear_fc2` | 27 | [4] | ['I16'] | [(1152, 4352)] | mul1 | ['F16'] | ['F16'] | False | 0.068 | 00005-of-00005 |
| `model.visual.merger.linear_fc1` | 1 | [4] | ['I16'] | [(4608, 4608)] | mul1 | ['F16'] | ['F16'] | False | 0.011 | 00005-of-00005 |
| `model.visual.merger.linear_fc2` | 1 | [4] | ['I16'] | [(2560, 4608)] | mul1 | ['F16'] | ['F16'] | False | 0.006 | 00005-of-00005 |
| `mtp.fc_embedding` | 1 | [3] | ['I16'] | [(2560, 2560)] | mul1 | ['F16'] | - | False | 0.002 | 00005-of-00005 |
| `mtp.fc_hidden` | 1 | [3] | ['I16'] | [(2560, 2560)] | mul1 | ['F16'] | - | False | 0.002 | 00005-of-00005 |
| `mtp.layers.*.mlp.experts.*.down_proj` | 512 | [2] | ['I16'] | [(2560, 640)] | mul1 | ['F16'] | - | False | 0.213 | 00005-of-00005 |
| `mtp.layers.*.mlp.experts.*.gate_proj` | 512 | [2] | ['I16'] | [(640, 2560)] | mul1 | ['F16'] | - | False | 0.213 | 00005-of-00005 |
| `mtp.layers.*.mlp.experts.*.up_proj` | 512 | [2] | ['I16'] | [(640, 2560)] | mul1 | ['F16'] | - | False | 0.213 | 00005-of-00005 |
| `mtp.layers.*.mlp.shared_expert.down_proj` | 1 | [4] | ['I16'] | [(2560, 640)] | mul1 | ['F16'] | - | False | 0.001 | 00005-of-00005 |
| `mtp.layers.*.mlp.shared_expert.gate_proj` | 1 | [4] | ['I16'] | [(640, 2560)] | mul1 | ['F16'] | - | False | 0.001 | 00005-of-00005 |
| `mtp.layers.*.mlp.shared_expert.up_proj` | 1 | [4] | ['I16'] | [(640, 2560)] | mul1 | ['F16'] | - | False | 0.001 | 00005-of-00005 |
| `mtp.layers.*.self_attn.indexer.index_qk_proj` | 1 | [2] | ['I16'] | [(640, 2560)] | mul1 | ['F16'] | - | False | 0.000 | 00005-of-00005 |
| `mtp.layers.*.self_attn.k_proj` | 1 | [4] | ['I16'] | [(512, 2560)] | mul1 | ['F16'] | - | False | 0.001 | 00005-of-00005 |
| `mtp.layers.*.self_attn.o_proj` | 1 | [4] | ['I16'] | [(2560, 6144)] | mul1 | ['F16'] | - | False | 0.008 | 00005-of-00005 |
| `mtp.layers.*.self_attn.q_proj` | 1 | [4] | ['I16'] | [(12288, 2560)] | mul1 | ['F16'] | - | False | 0.016 | 00005-of-00005 |
| `mtp.layers.*.self_attn.v_proj` | 1 | [4] | ['I16'] | [(512, 2560)] | mul1 | ['F16'] | - | False | 0.001 | 00005-of-00005 |

## Non-trellis tensor families

| family | n | dtypes | shapes (first 3) | bytes | files |
|---|---|---|---|---|---|
| `model.language_model.embed_tokens.weight` | 1 | ['BF16'] | [248320, 2560] | 1271398400 | 00001-of-00005 |
| `model.language_model.hyper_connection_mixer.hc_norm.weight` | 1 | ['F16'] | [10240] | 20480 | 00005-of-00005 |
| `model.language_model.hyper_connection_mixer.input_mix_weight_down.weight` | 1 | ['F16'] | [320, 10240] | 6553600 | 00005-of-00005 |
| `model.language_model.hyper_connection_mixer.input_mix_weight_up.weight` | 1 | ['F16'] | [10240, 320] | 6553600 | 00005-of-00005 |
| `model.language_model.layers.*.attn_hyper_connection.block_inject_weight.weight` | 48 | ['F16'] | [4, 10240] | 3932160 | 00001-of-00005,00002-of-00005,00003-of-00005,00004-of-00005,00005-of-00005 |
| `model.language_model.layers.*.attn_hyper_connection.hc_norm.weight` | 48 | ['F16'] | [10240] | 983040 | 00001-of-00005,00002-of-00005,00003-of-00005,00004-of-00005,00005-of-00005 |
| `model.language_model.layers.*.attn_hyper_connection.input_mix_weight_down.weight` | 48 | ['F16'] | [320, 10240] | 314572800 | 00001-of-00005,00002-of-00005,00003-of-00005,00004-of-00005,00005-of-00005 |
| `model.language_model.layers.*.attn_hyper_connection.input_mix_weight_up.weight` | 48 | ['F16'] | [10240, 320] | 314572800 | 00001-of-00005,00002-of-00005,00003-of-00005,00004-of-00005,00005-of-00005 |
| `model.language_model.layers.*.linear_attn.A_log` | 36 | ['BF16'] | [48] | 3456 | 00001-of-00005,00002-of-00005,00003-of-00005,00004-of-00005,00005-of-00005 |
| `model.language_model.layers.*.linear_attn.conv1d.weight` | 36 | ['BF16'] | [10240, 1, 4] | 2949120 | 00001-of-00005,00002-of-00005,00003-of-00005,00004-of-00005,00005-of-00005 |
| `model.language_model.layers.*.linear_attn.dt_bias` | 36 | ['BF16'] | [48] | 3456 | 00001-of-00005,00002-of-00005,00003-of-00005,00004-of-00005,00005-of-00005 |
| `model.language_model.layers.*.linear_attn.in_proj_a.weight` | 36 | ['F16'] | [48, 2560] | 8847360 | 00001-of-00005,00002-of-00005,00003-of-00005,00004-of-00005,00005-of-00005 |
| `model.language_model.layers.*.linear_attn.in_proj_b.weight` | 36 | ['F16'] | [48, 2560] | 8847360 | 00001-of-00005,00002-of-00005,00003-of-00005,00004-of-00005,00005-of-00005 |
| `model.language_model.layers.*.linear_attn.norm.weight` | 36 | ['BF16'] | [128] | 9216 | 00001-of-00005,00002-of-00005,00003-of-00005,00004-of-00005,00005-of-00005 |
| `model.language_model.layers.*.mlp.gate.weight` | 48 | ['F16'] | [512, 2560] | 125829120 | 00001-of-00005,00002-of-00005,00003-of-00005,00004-of-00005,00005-of-00005 |
| `model.language_model.layers.*.mlp.shared_expert_gate.weight` | 48 | ['F16'] | [1, 2560] | 245760 | 00001-of-00005,00002-of-00005,00003-of-00005,00004-of-00005,00005-of-00005 |
| `model.language_model.layers.*.mlp_hyper_connection.block_inject_weight.weight` | 48 | ['F16'] | [4, 10240] | 3932160 | 00001-of-00005,00002-of-00005,00003-of-00005,00004-of-00005,00005-of-00005 |
| `model.language_model.layers.*.mlp_hyper_connection.hc_norm.weight` | 48 | ['F16'] | [10240] | 983040 | 00001-of-00005,00002-of-00005,00003-of-00005,00004-of-00005,00005-of-00005 |
| `model.language_model.layers.*.mlp_hyper_connection.input_mix_weight_down.weight` | 48 | ['F16'] | [320, 10240] | 314572800 | 00001-of-00005,00002-of-00005,00003-of-00005,00004-of-00005,00005-of-00005 |
| `model.language_model.layers.*.mlp_hyper_connection.input_mix_weight_up.weight` | 48 | ['F16'] | [10240, 320] | 314572800 | 00001-of-00005,00002-of-00005,00003-of-00005,00004-of-00005,00005-of-00005 |
| `model.language_model.layers.*.ple.conv1d.weight` | 1 | ['F16'] | [10240, 1, 4] | 81920 | 00001-of-00005 |
| `model.language_model.layers.*.ple.key_proj.weight` | 1 | ['F16'] | [10240, 2560] | 52428800 | 00001-of-00005 |
| `model.language_model.layers.*.ple.norm_conv.weight` | 1 | ['BF16'] | [10240] | 20480 | 00001-of-00005 |
| `model.language_model.layers.*.ple.norm_key.weight` | 1 | ['BF16'] | [10240] | 20480 | 00001-of-00005 |
| `model.language_model.layers.*.ple.norm_query.weight` | 1 | ['BF16'] | [10240] | 20480 | 00001-of-00005 |
| `model.language_model.layers.*.ple.ple_embedding.ngram_embedding.head_bias` | 1 | ['F16'] | [16, 160] | 5120 | ngram_embedding |
| `model.language_model.layers.*.ple.ple_embedding.ngram_embedding.head_offsets` | 1 | ['I64'] | [16] | 128 | ngram_embedding |
| `model.language_model.layers.*.ple.ple_embedding.ngram_embedding.head_vocab_sizes` | 1 | ['I64'] | [16] | 128 | ngram_embedding |
| `model.language_model.layers.*.ple.ple_embedding.ngram_embedding.layer_multipliers` | 1 | ['I64'] | [3] | 24 | ngram_embedding |
| `model.language_model.layers.*.ple.value_proj.weight` | 1 | ['F16'] | [2560, 2560] | 13107200 | 00001-of-00005 |
| `model.language_model.layers.*.self_attn.indexer.k_layernorm.weight` | 12 | ['BF16'] | [128] | 3072 | 00001-of-00005,00002-of-00005,00003-of-00005,00004-of-00005,00005-of-00005 |
| `model.language_model.layers.*.self_attn.indexer.q_layernorm.weight` | 12 | ['BF16'] | [128] | 3072 | 00001-of-00005,00002-of-00005,00003-of-00005,00004-of-00005,00005-of-00005 |
| `model.language_model.layers.*.self_attn.k_norm.weight` | 12 | ['BF16'] | [256] | 6144 | 00001-of-00005,00002-of-00005,00003-of-00005,00004-of-00005,00005-of-00005 |
| `model.language_model.layers.*.self_attn.q_norm.weight` | 12 | ['BF16'] | [256] | 6144 | 00001-of-00005,00002-of-00005,00003-of-00005,00004-of-00005,00005-of-00005 |
| `model.visual.blocks.*.attn.qkv.bias` | 27 | ['BF16'] | [3456] | 186624 | 00005-of-00005 |
| `model.visual.blocks.*.attn.qkv.weight` | 27 | ['BF16'] | [3456, 1152] | 214990848 | 00005-of-00005 |
| `model.visual.blocks.*.norm1.bias` | 27 | ['F16'] | [1152] | 62208 | 00005-of-00005 |
| `model.visual.blocks.*.norm1.weight` | 27 | ['F16'] | [1152] | 62208 | 00005-of-00005 |
| `model.visual.blocks.*.norm2.bias` | 27 | ['F16'] | [1152] | 62208 | 00005-of-00005 |
| `model.visual.blocks.*.norm2.weight` | 27 | ['F16'] | [1152] | 62208 | 00005-of-00005 |
| `model.visual.merger.norm.bias` | 1 | ['F16'] | [1152] | 2304 | 00005-of-00005 |
| `model.visual.merger.norm.weight` | 1 | ['F16'] | [1152] | 2304 | 00005-of-00005 |
| `model.visual.patch_embed.proj.bias` | 1 | ['F16'] | [1152] | 2304 | 00005-of-00005 |
| `model.visual.patch_embed.proj.weight` | 1 | ['F16'] | [1152, 3, 2, 16, 16] | 3538944 | 00005-of-00005 |
| `model.visual.pos_embed.weight` | 1 | ['BF16'] | [2304, 1152] | 5308416 | 00005-of-00005 |
| `mtp.hyper_connection_mixer.hc_norm.weight` | 1 | ['F16'] | [10240] | 20480 | mtp_hyper_connection_mixer_patch |
| `mtp.hyper_connection_mixer.input_mix_weight_down.weight` | 1 | ['F16'] | [320, 10240] | 6553600 | mtp_hyper_connection_mixer_patch |
| `mtp.hyper_connection_mixer.input_mix_weight_up.weight` | 1 | ['F16'] | [10240, 320] | 6553600 | mtp_hyper_connection_mixer_patch |
| `mtp.layers.*.attn_hyper_connection.block_inject_weight.weight` | 1 | ['F16'] | [4, 10240] | 81920 | 00005-of-00005 |
| `mtp.layers.*.attn_hyper_connection.hc_norm.weight` | 1 | ['F16'] | [10240] | 20480 | 00005-of-00005 |
| `mtp.layers.*.attn_hyper_connection.input_mix_weight_down.weight` | 1 | ['F16'] | [320, 10240] | 6553600 | 00005-of-00005 |
| `mtp.layers.*.attn_hyper_connection.input_mix_weight_up.weight` | 1 | ['F16'] | [10240, 320] | 6553600 | 00005-of-00005 |
| `mtp.layers.*.mlp.gate.weight` | 1 | ['F16'] | [512, 2560] | 2621440 | 00005-of-00005 |
| `mtp.layers.*.mlp.shared_expert_gate.weight` | 1 | ['F16'] | [1, 2560] | 5120 | 00005-of-00005 |
| `mtp.layers.*.mlp_hyper_connection.block_inject_weight.weight` | 1 | ['F16'] | [4, 10240] | 81920 | 00005-of-00005 |
| `mtp.layers.*.mlp_hyper_connection.hc_norm.weight` | 1 | ['F16'] | [10240] | 20480 | 00005-of-00005 |
| `mtp.layers.*.mlp_hyper_connection.input_mix_weight_down.weight` | 1 | ['F16'] | [320, 10240] | 6553600 | 00005-of-00005 |
| `mtp.layers.*.mlp_hyper_connection.input_mix_weight_up.weight` | 1 | ['F16'] | [10240, 320] | 6553600 | 00005-of-00005 |
| `mtp.layers.*.self_attn.indexer.k_layernorm.weight` | 1 | ['BF16'] | [128] | 256 | 00005-of-00005 |
| `mtp.layers.*.self_attn.indexer.q_layernorm.weight` | 1 | ['BF16'] | [128] | 256 | 00005-of-00005 |
| `mtp.layers.*.self_attn.k_norm.weight` | 1 | ['BF16'] | [256] | 512 | 00005-of-00005 |
| `mtp.layers.*.self_attn.q_norm.weight` | 1 | ['BF16'] | [256] | 512 | 00005-of-00005 |
| `mtp.pre_fc_norm_embedding.weight` | 1 | ['BF16'] | [2560] | 5120 | 00005-of-00005 |
| `mtp.pre_fc_norm_hidden.weight` | 1 | ['F16'] | [10240] | 20480 | 00005-of-00005 |

## K summary by coarse family

- **GDN (linear_attn)**: K = [4]
- **QSA indexer**: K = [2]
- **attention**: K = [4]
- **lm_head**: K = [4]
- **mtp: mtp.fc_embedding**: K = [3]
- **mtp: mtp.fc_hidden**: K = [3]
- **mtp: mtp.layers.*.mlp.experts.*.down_proj**: K = [2]
- **mtp: mtp.layers.*.mlp.experts.*.gate_proj**: K = [2]
- **mtp: mtp.layers.*.mlp.experts.*.up_proj**: K = [2]
- **mtp: mtp.layers.*.mlp.shared_expert.down_proj**: K = [4]
- **mtp: mtp.layers.*.mlp.shared_expert.gate_proj**: K = [4]
- **mtp: mtp.layers.*.mlp.shared_expert.up_proj**: K = [4]
- **other: model.language_model.layers.*.ple.ple_embedding.ngram_embedding.shard_***: K = [2]
- **routed experts**: K = [2]
- **shared_expert**: K = [4]
- **vision**: K = [4]

## ngram_embedding.safetensors

- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.head_bias` F16 [16, 160]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.head_offsets` I64 [16]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.head_vocab_sizes` I64 [16]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.layer_multipliers` I64 [3]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_1.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_10.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_100.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_101.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_102.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_103.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_104.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_105.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_106.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_107.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_108.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_109.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_11.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_110.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_111.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_112.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_113.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_114.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_115.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_116.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_117.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_118.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_119.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_12.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_120.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_121.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_122.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_123.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_124.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_125.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_126.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_127.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_13.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_14.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_15.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_16.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_17.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_18.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_19.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_2.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_20.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_21.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_22.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_23.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_24.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_25.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_26.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_27.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_28.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_29.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_3.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_30.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_31.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_32.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_33.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_34.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_35.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_36.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_37.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_38.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_39.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_4.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_40.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_41.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_42.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_43.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_44.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_45.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_46.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_47.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_48.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_49.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_5.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_50.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_51.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_52.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_53.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_54.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_55.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_56.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_57.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_58.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_59.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_6.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_60.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_61.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_62.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_63.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_64.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_65.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_66.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_67.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_68.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_69.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_7.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_70.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_71.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_72.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_73.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_74.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_75.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_76.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_77.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_78.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_79.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_8.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_80.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_81.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_82.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_83.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_84.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_85.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_86.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_87.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_88.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_89.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_9.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_90.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_91.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_92.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_93.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_94.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_95.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_96.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_97.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_98.trellis` I16 [2500012, 41]
- `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_99.trellis` I16 [2500012, 41]

## vision_k6.safetensors inventory (all tensors, grouped)

| family | n | dtypes | shapes |
|---|---|---|---|

model.visual.* tensors in index-listed shards: 987
