# RST session — S1 CUDA KDA decode vs CPU ref

CHARTER
-----------------------------------------------
Find whether a **new** one-token CUDA KDA decode (conv-4 + L2 q/k + sigmoid(beta) + delta-rule) matches the atlas-core CPU oracle on a tiny synthetic, and whether planted known-bads (zero conv, skip sigmoid) still move the output. Do not copy GDN / Mamba / GLM `kda_*.cu`.

#AREAS
S1
KDA
kda_decode.cu
kda_cuda

START
2026-09-12

ORACLE
Self-consistency: CUDA V-outer nest (`recurrent_cuda_order`) == `kda_recurrent_step` on twin geometry (H=8, D=32, K=4).
Source: `kernels/gb10/kimi-k3/bf16/kda_decode.cu` declares `k3_kda_conv_update_f32` + `k3_kda_recurrent_step_f32`, applies `k3_sigmoid(beta[h])`, does **not** contain `KDA_REC_BODY` / `gated_delta_rule` / mamba2.
Host: `launch_k3_kda_decode_token` looks up module `kda_decode` and launches conv then recurrent (mock contract; real PTX is spark2 nvcc).
Claims: unique stem — not a shadow of `common/kda_recurrent.cu`.
Known-bad: zero conv weights, and skip `sigmoid` on `beta=0`, must diverge.

KNOWN-BAD (instrument)
`zero_conv_diverges` — conv_w = 0 vs 0.2 changes the core vector.
`skip_sigmoid_diverges` — raw `beta=0` (delta scale 0) vs `sigmoid(0)=0.5` changes the core vector.
Source-contract `kda_decode_cu_is_k3_not_gdn_shadow` would fail if the file were a GDN/GLM paste or dropped in-kernel sigmoid.

TEST NOTES
- Mac: `ATLAS_SKIP_BUILD=1 cargo test -p atlas-core --lib kimi_k3`. No nvcc. Numeric compare is the host nest that mirrors the `.cu` loops vs `kda_decode_token` / `kda_recurrent_step`.
- Serve of 0.40B stays on the CPU-fallback wrapper this slice. CUDA KDA is the kernel + launch harness, not `K3BoundLayer::decode`.
- Twin omit `gate_lower_bound` (FLA unbounded). Gate is an input; this kernel exponentiates log-decay like the CPU ref.
- spark2 still must `nvcc` `kda_decode.cu` into the kimi-k3/bf16 target and run `launch_k3_kda_decode_token` against the CPU oracle on-device.

BUGS
#N/A this slice for the host oracle + source contract. Device numeric vs CPU is parked on spark2 compile.

STOP
Charter complete for the unique kernel + CPU-parity instrument + known-bads. spark2 nvcc / on-device compare parked. Official 1.56 TB out of scope.
