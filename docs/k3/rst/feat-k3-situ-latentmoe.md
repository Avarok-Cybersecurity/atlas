# RST session — feat/k3-situ-latentmoe (extracted)

CHARTER
-----------------------------------------------
Find whether LatentMoE frozen-gate top-k + SiTU-GLU mix is the CPU ref, and whether packed experts look up DSV4 `moe_w4a16_grouped_gemm_ptrtable_e8m0` instead of silent host F32.

AREAS
feat/k3-situ-latentmoe
C6 (twin mix lives on umbrella)

ORACLE
- `force_expert_zero_mutant_diverges` on CPU mix.
- `K3MoeGemmKernels::resolve` looks up `moe_w4a16::moe_w4a16_grouped_gemm_ptrtable_e8m0`.
- extra_cu path is DSV4 `moe_w4a16_grouped_gemm.cu` (not a copy).

KNOWN-BAD
- Force expert 0 vs frozen top-1 changes mix.
- `deny_kernel_resolve_bails_not_silent_cpu` — missing PTX contains `cannot silently run host F32`.

TEST NOTES
`cargo test -p avarok-core --lib -- kimi_k3::situ kimi_k3::latent_moe`
`cargo test -p spark-model --lib -- deny_kernel_resolve_bails_not_silent_cpu` (Linux)

BUGS
#N/A this slice.

TEST NOTES (review 1080)
Host unpack SSOT is `avarok_core::mxfp4_e8m0` (moved from `spark_model::weight_map::fp8_lut`). Same LUT, nibble order, exp=0/255 → 0.0. Gate tip: this commit on `feat/k3-situ-latentmoe`. GPU GEMM remains DSV4 extra_cu, not a second stack.

STOP
Charter complete for SiTU + LatentMoE + DSV4 E8M0 launch. mmap expert backend is the next slice.
