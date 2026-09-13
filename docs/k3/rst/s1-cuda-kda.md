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
`K3BoundLayer::decode` LinearAttention default is CUDA `launch_k3_kda_decode_token`. `K3_CUDA_KDA=0` keeps the CPU mixer. MLP+AttnRes stay CPU either way.
Claims: unique stem — not a shadow of `common/kda_recurrent.cu`.
Known-bad: zero conv weights, skip `sigmoid` on `beta=0`, AttnRes mix=0, and a zero injected KDA core, must diverge.

KNOWN-BAD (instrument)
`zero_conv_diverges` — conv_w = 0 vs 0.2 changes the core vector.
`skip_sigmoid_diverges` — raw `beta=0` (delta scale 0) vs `sigmoid(0)=0.5` changes the core vector.
`cpu_fallback_mix0_changes_greedy_tokens` — mix=0 still moves greedy vs mix=1 on the default CPU BoundLayer path (`K3_ATTNRES_MIX=0` on serve).
`injected_zero_kda_core_diverges` — a zero conv+recurrent core moves the layer output (proves the mixer injection site).
Source-contract `kda_decode_cu_is_k3_not_gdn_shadow` would fail if the file were a GDN/GLM paste or dropped in-kernel sigmoid.

TEST NOTES
- Mac: `ATLAS_SKIP_BUILD=1 cargo test -p atlas-core --lib kimi_k3`. No nvcc. Numeric compare is the host nest that mirrors the `.cu` loops vs `kda_decode_token` / `kda_recurrent_step`.
- **Default LinearAttention decode is CUDA.** Mock GPU does not run the `.cu`, so this slice cannot prove CUDA==CPU on C1 aviation greedy. `K3_CUDA_KDA=0` is the CPU escape (live spark1 serve is still the previous CPU binary until rebuild).
- BoundLayer mock: KDA + flag launches conv then recurrent; KDA without flag does not look up `kda_decode`; MLA ignores the flag.
- Twin omit `gate_lower_bound` (FLA unbounded). Gate is an input; this kernel exponentiates log-decay like the CPU ref.
- spark2: `kda_decode.cu` already nvcc'd. Need a **Rust rebuild** of spark-model/server + recopy `spark-k3` before serve uses CUDA KDA. Then aviation greedy + mix=0.

BUGS
#N/A this slice for the host oracle + source contract + BoundLayer CUDA default. Device numeric vs CPU is parked on spark2 rebuild.

STOP
Charter complete for unique kernel + CPU-parity instrument + known-bads + BoundLayer LinearAttention CUDA default. On-device CUDA==CPU and aviation-after-rebuild parked. Official 1.56 TB out of scope.
