# Paper Summary

The Atlas team maintains a technical paper in `paper/atlas.tex` — a two-column LaTeX document titled:

> **"Atlas: A Custom CUDA Inference Engine for Hybrid Mamba/Attention MoE Models on NVIDIA Blackwell GB10"**

The ArXiv version is the academic-facing companion to this book. Where the book is a guide for operators and contributors, the paper is the reference you cite from another piece of research.

## Abstract (paraphrased)

Atlas is a pure-Rust LLM inference engine targeting a single `(Hardware, Model, Quantization)` tuple at a time and hyperoptimizing each tuple independently. On NVIDIA's GB10 Grace-Blackwell Superchip (SM121), running Qwen3.5-35B-A3B in NVFP4 with MTP speculative decoding, Atlas reaches 131 tokens/second — 3.6× NVIDIA's vLLM on the same hardware and model. The paper describes the kernel registry mechanism, the SBIO-based Rust trait layer that enables testing without a GPU, the NVFP4 software E2M1 conversion that works around SM121's missing native FP4 MMA, and the Marconi SSM snapshot cache that makes prefix caching correct on hybrid SSM+attention models.

## Key claims the paper makes

- **Specialization scales.** Per-`(H, M_q)` kernel sets, combined with vendor-agnostic runtime traits, scale to many targets without regressing existing ones.
- **Software E2M1 on SM121 is viable.** Branchless FP32 → E2M1 conversion in 7 ALU ops closes the gap left by the missing hardware instruction; Atlas's NVFP4 throughput on GB10 is the silicon ceiling.
- **MTP + constrained decoding is a throughput multiplier on agent workloads.** XGrammar-masked MTP drafts achieve ~95% acceptance inside tool calls, yielding +37% throughput on agentic traces.
- **Hybrid SSM+attention prefix caching requires state snapshots.** Marconi (the SSM snapshot cache) produces byte-identical warm-cache output; without it, prefix caching would silently diverge on hybrid models.

## How the book and the paper relate

- The **book** covers operations + architecture + contribution workflow. If you want to run or extend Atlas, start here.
- The **paper** covers the research claims + benchmark methodology + comparisons to contemporaneous work (vLLM NVFP4, TRT-LLM NVFP4, SGLang, FlashInfer). If you're writing a related paper or a systems course, cite it.

Both share kernel benchmark numbers, the supported-model matrix, and the architectural rationale. The book is the more expansive document; the paper is the tighter academic framing.

## References

- **Paper source**: [`paper/atlas.tex`](https://github.com/Avarok-Cybersecurity/atlas/blob/main/paper/atlas.tex) (build with `pdflatex`).
- **Citations the paper relies on** (also in the README's Citations section):
  - FlashAttention-2 (ICLR 2024) — tiled online softmax
  - FlashAttention-4 (2025) — software polynomial exp, conditional softmax rescaling
  - FlashInfer (MLSys 2025) — block-sparse paged KV, gather-SMEM-MMA
  - SageAttention 3 (NeurIPS 2025) — native FP4 attention on newer Blackwell
  - LeanAttention (2024) — stream-K tile scheduling for decode
  - XGrammar (2024) — token-bitmap automaton for constrained decoding
