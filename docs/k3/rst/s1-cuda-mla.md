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
`K3BoundLayer::decode` FullAttention default is CUDA `launch_k3_mla_decode_token`. `K3_CUDA_MLA=0` keeps the CPU mixer. Same polarity as KDA. Projections / AttnRes / router stay CPU either way.
Claims: unique stem — not a shadow of `common/` or qwen3_attention. nvfp4 serve bundle **symlinks** the stem (same lesson as `kda_decode`).
Known-bad: skip output-gate, apply RoPE when NoPE, must diverge.

KNOWN-BAD (instrument)
`skip_output_gate_diverges` — `g=0` with gate on is `sigmoid(0)=0.5`; skip-gate is identity 1.0. Output moves.
`apply_rope_when_nope_diverges` — rotating the packed rope slice at pos=3 moves q/k vs NoPE leave-alone.
Source-contract `mla_decode_cu_is_k3_not_qwen_shadow` would fail if the file were a qwen3/GLM/DeepSeek paste or dropped in-kernel gate/NoPE flags.

TEST NOTES
- Mac: `ATLAS_SKIP_BUILD=1 cargo test -p atlas-core --lib kimi_k3`. No nvcc. Numeric compare is the host nest that mirrors the `.cu` loops vs `mla_decode_token` / `sdpa_one`.
- **Default FullAttention decode is CUDA.** Mock GPU does not run the `.cu`. `K3_CUDA_MLA=0` is the CPU escape. Serve C1 with `K3_CUDA_MLA=1` already logged `mla_decode` + bee-fly at `0898345`; this slice matches KDA polarity.
- BoundLayer mock: MLA + flag launches rope then sdpa; MLA without flag does not look up `mla_decode`; KDA ignores the MLA flag.
- Serve selects `(sm_121, kimi-k3, nvfp4)`. Stem is already symlinked into nvfp4/.

TEST NOTES (9d6c273, CUDA MLA default)
- spark1 spark-k3 SHA `9d6c273`. Serve **without** `K3_CUDA_MLA` (unset, not `=1`). Log: `K3 FullAttention decode via CUDA mla_decode` and `K3 LinearAttention decode via CUDA kda_decode`.
- Aviation T=0 max_tokens=16 → `'there is no way a bee should be able to fly. Its wings are too'`. Prefill first token **1459**. C1 prefix holds on the default CUDA mixer.
- Known-bad `K3_ATTNRES_MIX=0` (still CUDA MLA+KDA): `'to to to to to to to to to to to to to to to to'`, first token **308**. Mix lever still moves tokens.
- Live spark1 :8888 left on CUDA default (no `K3_CUDA_MLA`, no mix0).

BUGS
#N/A host oracle + source contract + BoundLayer default-on.
#N/A CUDA-default aviation C1 + mix=0 known-bad on spark1 nvfp4 serve after `9d6c273`.

STOP
On-device CUDA MLA default **matches** C1 aviation greedy on spark1 nvfp4 serve. Mix=0 still moves tokens. Do not book: projections / AttnRes / router still host; official 1.56 TB not downloaded. Rental soak is CR1–CR3.
