# RST session — S1 CUDA gated-NoPE MLA decode vs CPU ref

CHARTER
-----------------------------------------------
Find whether a **new** one-token CUDA gated-NoPE MLA decode (rope slots allocated but not rotated; sigmoid output gate) matches the atlas-core CPU oracle on a tiny synthetic, and whether planted known-bads (skip output-gate, apply RoPE when NoPE) still move the output. Do not copy qwen3_attention / GLM DSA MLA / DeepSeek `mla_paged_decode`.

#AREAS
S1
MLA
mla_decode.cu
mla_cuda

START
2026-09-13

ORACLE
Self-consistency: CUDA nest (`sdpa_gate_cuda_order`) == `sdpa_one` + `apply_output_gate` on twin geometry (H=8, nope=64, rope=32, dv=64).
Source: `kernels/gb10/kimi-k3/bf16/mla_decode.cu` declares `k3_mla_maybe_rope_f32` + `k3_mla_sdpa_gate_f32`, early-returns on `use_nope`, applies `k3_sigmoid` when `use_gate`. Does **not** contain `qwen3_attention` / `glm5next_dsa` / `mla_paged_decode` / `gated_delta_rule`.
Host: `launch_k3_mla_decode_token` looks up module `mla_decode` and launches rope then sdpa_gate (mock contract; real PTX is spark2 nvcc).
`K3BoundLayer::decode` FullAttention default is CPU `mla_decode_token`. `K3_CUDA_MLA=1` is the CUDA opt-in. KDA default is unchanged (`K3_CUDA_KDA` not-0). Projections / AttnRes / MLP stay CPU either way.
Claims: unique stem — not a shadow of `common/` or qwen3_attention. nvfp4 serve bundle **symlinks** the stem (same lesson as `kda_decode`).
Known-bad: skip output-gate, apply RoPE when NoPE, must diverge.

KNOWN-BAD (instrument)
`skip_output_gate_diverges` — `g=0` with gate on is `sigmoid(0)=0.5`; skip-gate is identity 1.0. Output moves.
`apply_rope_when_nope_diverges` — rotating the packed rope slice at pos=3 moves q/k vs NoPE leave-alone.
Source-contract `mla_decode_cu_is_k3_not_qwen_shadow` would fail if the file were a qwen3/GLM/DeepSeek paste or dropped in-kernel gate/NoPE flags.

TEST NOTES
- Mac: `ATLAS_SKIP_BUILD=1 cargo test -p atlas-core --lib kimi_k3`. No nvcc. Numeric compare is the host nest that mirrors the `.cu` loops vs `mla_decode_token` / `sdpa_one`.
- **Default FullAttention decode is CPU.** Mock GPU does not run the `.cu`, so this slice cannot prove CUDA==CPU on C1 aviation greedy. That is why CUDA is opt-in (`K3_CUDA_MLA=1`), not default-on like KDA.
- BoundLayer mock: MLA + flag launches rope then sdpa; MLA without flag does not look up `mla_decode`; KDA ignores the MLA flag.
- Serve selects `(sm_121, kimi-k3, nvfp4)`. Stem is symlinked into nvfp4/ so opt-in lookup will not abort the way KDA did before `7238f64bf`. **spark2 CUDA rebuild is still required** before `K3_CUDA_MLA=1` can run; default CPU aviation does not need it.

BUGS
#N/A this slice for the host oracle + source contract + BoundLayer opt-in. On-device C1 with `K3_CUDA_MLA=1` is **not** claimed.

STOP
Charter complete for the CPU nest + known-bads + unique stem + BoundLayer opt-in. Do not book: CUDA MLA C1 unproven on spark1 (needs spark2 nvcc); projections / AttnRes / MLP still host; extra_cu GEMM is not dispatched from LatentMoE; official 1.56 TB not downloaded.
