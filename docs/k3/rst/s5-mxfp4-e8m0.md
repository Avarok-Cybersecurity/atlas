# RST session — S5 MXFP4 reuses DSV4 E8M0

CHARTER
-----------------------------------------------
Find whether K3 official packed experts (`w1/w2/w3.weight_packed` + `weight_scale`) map onto the existing DeepSeek-V4 E8M0 unpack, without a second math stack, and whether a flipped scale is caught.

ORACLE
`cargo test -p atlas-core --lib kimi_k3 --no-default-features`
Known-bad (instrument failed first):
- Packed tensor with a flipped E8M0 scale byte diverges from the BF16 reference produced by `dequant_nvfp4_e8m0_to_bf16`
- `weight_packed` without `K3_ALLOW_MXFP4=1` → `S5 MXFP4 not this slice` (not a silent BF16 bind)

#NOTES
- Host SSOT: `atlas_core::mxfp4_e8m0` (extracted from DSV4 `dequant_nvfp4_e8m0_to_bf16`).
- K3 name map: `.weight_packed` / `.weight_scale` → DSV4 `.weight` / `.scale`.
- GPU lander wrapper: `quantized_k3_mxfp4_e8m0` → `quantized_mxfp4_e8m0_pair`. GEMM not this slice.
- Do not download official 1.56 TB.

#BUGS
#N/A this slice for synthetic RST.

STOP
Charter complete for S5 host map + DSV4 unpack reuse. GPU expert GEMM parked.
