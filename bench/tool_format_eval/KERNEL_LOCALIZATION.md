# Kernel-level localization: Atlas-FP8 vs vLLM-FP8 forward-pass divergence

Goal: name the Atlas kernel responsible for the tool-calling forward-pass
divergence (engine half of the defect; see README.md). Method: dump per-layer
residual + per-op hidden states from Atlas (`ATLAS_OP_DUMP`/`ATLAS_GDN_DUMP`/
`ATLAS_NEMO_DUMP`, local `spark` binary) and from vLLM-FP8 (in-process hooks,
`bench/fp8_dgx2_drift/vllm_layer_dump.py` adapted), SAME tokens, last-token
slice, f32. Compare with `cosine_atlas_vllm.py`.

## Result (149-token probe, current single-GPU build, Qwen3.6-35B-A3B-FP8)

- Per-LAYER residual cosine: all 40 layers cos > 0.997, |vllm|≈|atlas| (magnitudes
  match → directional drift). Mild at this short prompt; SSM/GDN layers track vLLM.
- Per-OP `o_proj` (attention output) cosine — divergence concentrated and DEPTH-GROWING:
    L3 0.9983 / L15 0.9993 / L19 0.9968 / L23 0.9960 / **L27 0.9907 (rel_l2 .139)** /
    **L31 0.9925 (rel_l2 .124)** / L35 0.9968 / L39 0.9992.
  The attention OUTPUT carries 2-3x the relative error of the residual stream.
- Prior 10382-token study (bench/fp8_dgx2_drift/FP8_COSINE_RESULTS_2026_05_29.txt):
  clean per-layer onset at **L3 (first full-attention layer)**, monotonic collapse
  to L38 (cos 0.9865). The amplified long-context version of the same pattern.

## Conclusion

The divergent kernel is Atlas's **full-attention path** (attention compute +
`o_proj` FP8 GEMM at sm_121) — NOT the SSM/GatedDeltaNet, MoE, or norm kernels,
which track vLLM. Error concentrates in the attention output and accumulates with
depth AND context length, explaining why tool-calling degrades with more tools /
longer context / thinking (more tokens through the attention).

## Next refinement (not yet run)
- Longer prompt (~2-8K tok) to reproduce the dramatic late-layer collapse on this build.
- Finer per-op: Atlas dumps q_proj/q_post_rope/k_proj/k_post_norm/k_post_rope/
  v_proj/attn_out_pre_gate/attn_out_post_gate. vLLM fuses qkv, so use the HF
  reference (needs the transformers finegrained-fp8 `kernels` pkg pinned to a
  compatible version, or a manual fused-MoE dequant) to split q vs k vs attn-compute.
