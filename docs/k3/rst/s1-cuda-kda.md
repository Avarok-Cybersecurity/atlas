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
- **Default LinearAttention decode is CUDA.** Mock GPU does not run the `.cu`, so this slice cannot prove CUDA==CPU on C1 aviation greedy. `K3_CUDA_KDA=0` is the CPU escape.
- BoundLayer mock: KDA + flag launches conv then recurrent; KDA without flag does not look up `kda_decode`; MLA ignores the flag.
- Twin omit `gate_lower_bound` (FLA unbounded). Gate is an input; this kernel exponentiates log-decay like the CPU ref.

TEST NOTES (7661a9a94, spark1 serve)
- spark2 `cargo build --release -p spark-server --features cuda,nccl` at `7661a9a94`. Recopied `/home/pidtom/k3-lab/bin/spark-k3`. SHA sidecar matches. No `--dangerously-allow-unresolved-kernel-lookups`.
- CUDA default (unset `K3_CUDA_KDA`): boot live, then first aviation prefill **aborts**. `kernel lookup kda_decode::k3_kda_conv_update_f32 ... failed AFTER the boot audit sealed`. Empty HTTP reply. Process dead.
- Selected target was `(sm_121, kimi-k3, nvfp4)` (178 modules). Stem lives only at `kernels/gb10/kimi-k3/bf16/kda_decode.cu`. nvfp4 bundle does not ship it.
- `K3_CUDA_KDA=0`: `/v1/completions` T=0 max_tokens=16 → `'there is no way a bee should be able to fly. Its wings are too'`. Prefill first token **1459**. C1 prefix holds.
- Known-bad `K3_ATTNRES_MIX=0` (still CPU escape): `'to to to to to to to to to to to to to to to to'`, first token **308**. Mix lever still moves tokens.

TEST NOTES (7238f64bf, nvfp4 stem)
- `kernels/gb10/kimi-k3/nvfp4/{kda_decode.cu,KERNEL.toml}` symlink the bf16 unique stem. spark2 CUDA rebuild: kimi-k3 nvfp4 **179** modules (was 178), 1 model-specific override. BUILD_EXIT:0.
- CUDA default (unset `K3_CUDA_KDA`, no unresolved-lookup): selected `(sm_121, kimi-k3, nvfp4)` (179 modules). Log: `K3 LinearAttention decode via CUDA kda_decode`.
- Aviation T=0 max_tokens=16 → `'there is no way a bee should be able to fly. Its wings are too'`. Prefill first token **1459**. C1 prefix holds on the CUDA mixer.
- Known-bad `K3_ATTNRES_MIX=0` (still CUDA KDA): `'to to to…'`, first token **308**. Mix lever still moves tokens.
- Live spark1 :8888 left on CUDA default (no `K3_CUDA_KDA`, no mix0).

BUGS
#N/A CUDA-default aviation C1 + mix=0 known-bad on spark1 nvfp4 serve after `7238f64bf`.
#N/A this slice for the host oracle + source contract + BoundLayer CUDA default.

STOP
On-device CUDA default **matches** C1 aviation greedy on spark1 nvfp4 serve. Mix=0 still moves tokens. Do not book: projections / AttnRes / MLP still host; MXFP4 grouped GEMM and official 1.56 TB still out of scope.
