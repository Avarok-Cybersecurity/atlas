# RST session — S5 MXFP4 experts reuse DSV4 E8M0 grouped GEMM

CHARTER
-----------------------------------------------
Find whether K3 official packed experts bind to the existing DeepSeek-V4 E8M0 grouped GEMM (`moe_w4a16_grouped_gemm_ptrtable_e8m0`) without a second MXFP4 stack or a GDN copy, and whether a flipped E8M0 scale is still caught.

#AREAS
S5
MXFP4
extra_cu
KimiK3WeightLoader

ORACLE
`cargo test -p atlas-core --lib kimi_k3 --no-default-features`
`ATLAS_SKIP_BUILD=1 cargo test -p spark-model --lib weight_loader::kimi_k3`
`ATLAS_SKIP_BUILD=1 cargo test -p atlas-kernels --test target_resolution kimi_k3_extra_cu`
Known-bad (instrument failed first):
- Packed tensor with a flipped E8M0 scale byte diverges from the BF16 reference (`flipped_e8m0_scale_diverges_from_bf16_reference`)
- `weight_packed` without `K3_ALLOW_MXFP4=1` → `S5 MXFP4 not this slice`
- KERNEL.toml extra_cu path missing or DSV4 source without `moe_w4a16_grouped_gemm_ptrtable_e8m0` fails `kimi_k3_extra_cu_reuses_dsv4_e8m0_grouped_gemm`

#NOTES
- extra_cu: `kernels/gb10/kimi-k3/{mxfp4,nvfp4}/KERNEL.toml` → `../../deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu`
- Module `moe_w4a16`. Expected handle `moe_w4a16_grouped_gemm_ptrtable_e8m0` (plus `_t` / `_t_k64` / fused_gate_up E8M0 twins). Not `[expected_absent]`.
- Unique KDA stem stays: `bf16/kda_decode.cu`; nvfp4 + mxfp4 symlink it.
- GPU lander: `quantized_k3_mxfp4_e8m0` → `quantized_mxfp4_e8m0_pair` when `K3_ALLOW_MXFP4=1`.
- PTX compile of extra_cu needs spark2 nvcc (`ATLAS_SKIP_BUILD` unset). This sheet's handle-present check is the source lookup name.
- Do not download official 1.56 TB.

#BUGS
#N/A synthetic RST. Dispatch of the GEMM from K3BoundLayer LatentMoE is not this slice (host MLP still).

STOP
Charter complete for extra_cu reuse + packed lander + flipped-scale known-bad. Spark2 nvcc compile of the extra_cu module parked until a CUDA rebuild.
