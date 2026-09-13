# RST session — S1 GPU decode (CPU fallback wrapper)

CHARTER
-----------------------------------------------
Find whether `K3BoundLayer::decode` can serve a token through the 0.40B graph without CUDA KDA, by copying hidden D2H, running the atlas-core mixer+MLP+AttnRes, and copying H2D — and whether a planted AttnRes mix=0 mutant still moves tokens.

#AREAS
S1
S3
S6
K3BoundLayer
CPU-fallback-GPU-wrapper

START
2026-09-12

ORACLE
Self-consistency: `forward_one_layer` loop == `forward_token` (identity copy-out).
C3: `LayerCache::to_bytes` / `from_bytes` after a prefix matches in-place cache.
Claims: decode must not bail `K3 GPU forward is not this slice`.
Known-bad: mix=0 greedy tokens diverge from mix=1 (`cpu_fallback_mix0_changes_greedy_tokens`).
Product: `spark serve` of `inference-optimization/Kimi-K3-0.40B` is the S6 check (spark1; not this Mac session).

KNOWN-BAD (instrument)
`cpu_fallback_mix0_changes_greedy_tokens` — mix=0 (identity skip) changes greedy tokens vs mix=1.
On spark serve, plant the same mutant with `K3_ATTNRES_MIX=0` (no Ablation CLI). Drop-weight / `K3_FORCE_EXPERT=0` is the C6 lever.

TEST NOTES
- Default path is still the **CPU fallback GPU wrapper**. `K3_CUDA_KDA=1` swaps only the KDA mixer core onto `kda_decode.cu`; MLP+AttnRes stay CPU. `decode_graph_unsupported` stays true.
- Mac: `ATLAS_SKIP_BUILD=1 cargo test -p atlas-core --lib kimi_k3`. spark-model may not compile on Mac.
- Prefix-cache: Marconi `snapshot_aux` / `restore_aux` serializes per-layer `LayerCache` (KDA conv/recurrent + MLA KV). Same C3 semantics as the CPU tests.
- AttnRes is per-token across layers; the wrapper keys the host stream by the residual `DevicePtr` so the model's layer-outer prefill still matches CPU token-outer order.
- Last layer writes `AttnRes(output_res_*)` back to GPU hidden; the model loop still applies `final_norm` + lm_head on device.

BUGS
#N/A this slice for the host math. spark1 `spark serve` of the 0.40B twin is the remaining product check (common kernels + this wrapper). Official 1.56 TB is out of scope.

STOP
Charter complete for the wrapper math + known-bad. CUDA KDA parked. spark1 serve parked for the box that holds the twin.
