# Further Reading

Curated references that informed Atlas's design. Not exhaustive — just the papers, articles, and prior-art projects worth reading if you want to understand why Atlas is shaped the way it is.

## Kernel engineering

- **FlashAttention-2** — Tri Dao (ICLR 2024). [arXiv:2307.08691](https://arxiv.org/abs/2307.08691). Foundation of Atlas's prefill kernels — tiled online softmax, Q/K/V tiling with shared memory, causal masking.
- **FlashAttention-4** — Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao (2025). [arXiv:2603.05451](https://arxiv.org/abs/2603.05451). Conditional softmax rescaling (skip ~90% of rescale ops), software polynomial exponential (`sw_exp`, avoids SFU bottleneck).
- **FlashInfer** — Ye et al. (MLSys 2025 Best Paper). [arXiv:2501.01005](https://arxiv.org/abs/2501.01005). Block-sparse paged KV, gather-SMEM-MMA pattern. Informed Atlas's paged decode kernel.
- **SageAttention 3** — Zhang et al. (NeurIPS 2025). [arXiv:2505.11594](https://arxiv.org/abs/2505.11594). Native FP4 attention on newer Blackwell. Planned direction when SM12x+ silicon lands.
- **LeanAttention** — Roy, Vassilieva, Willke, Mendis (2024). [arXiv:2405.10480](https://arxiv.org/abs/2405.10480). Stream-K tile scheduling for decode attention. Planned for SM occupancy improvements.
- **CUTLASS** documentation and examples. The BF16 MMA fragment shapes and `cp.async` pipelining patterns in Atlas's kernels follow CUTLASS conventions.

## State-space models

- **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** — Gu, Dao (2023). [arXiv:2312.00752](https://arxiv.org/abs/2312.00752). The original selective SSM.
- **Mamba-2** — Dao, Gu (ICML 2024). [arXiv:2405.21060](https://arxiv.org/abs/2405.21060). The variant Nemotron-H uses.
- **Gated Delta Networks** — Yang, Dao, et al. (2024). Closer to Qwen3.5's GDN formulation.
- **GDN register-tile results** — Atlas's internal experiments in `gdn_regtile_results.md` track tile-shape tradeoffs on GB10.

## Quantization

- **NVFP4 / FP4 microscaling** — NVIDIA's blog posts on Blackwell FP4. The public docs for SM120 coverage are thin; much of Atlas's SM121 workaround has no upstream equivalent yet.
- **SmoothQuant** — Xiao, Lin, et al. (ICML 2023). Scale factor calibration ideas used indirectly in Atlas's FP8 KV calibration.
- **Compressed-Tensors format** — the HF `compressed-tensors` library's on-disk FP8 block-scaled layout.
- **TurboQuant** (Atlas internal) — `docs/design/turboquant-nightjob-2026-03-31.md`. WHT + Lloyd-Max 4-bit KV cache with ~2× lower MSE than NVFP4 at the same bit rate.

## Speculative decoding

- **Fast Inference from Transformers via Speculative Decoding** — Leviathan, Kalman, Matias (ICML 2023). The foundational paper.
- **Medusa: Multi-Token Prediction Heads** — Cai et al. (2024). MTP-style draft heads.
- **Self-speculative Decoding** — Layer-skipping drafter. Atlas implements a variant.

## Constrained decoding

- **XGrammar** — Li, Chen, Chen, et al. (2024). [arXiv:2411.15100](https://arxiv.org/abs/2411.15100). The token-bitmap automaton approach Atlas uses.
- **SGLang** — Zheng et al. (NeurIPS 2024). Broader exploration of structured-output techniques; Atlas shares design elements with SGLang's `regex_fsm`.

## Inference systems

- **vLLM** — Kwon et al. (SOSP 2023). The PagedAttention paper. Atlas's paged KV cache follows the vLLM model with Atlas-specific kernel work below it.
- **TensorRT-LLM** documentation and source. Atlas's TRT-LLM benchmark comparisons in `docs/history/` are informed by reading the TRT-LLM codebase; the 29.6 tok/s ceiling for NVFP4 on SM121 TRT-LLM is documented in `TRTLLM_OPTIMIZATION_RESULTS.md`.
- **SGLang** — structured-generation inference framework.
- **Triton** — inference server. Peripheral; Atlas does not use it but the operational patterns are informative.

## MoE routing

- **Mixture-of-Experts with Sigmoid Routing** — various 2023–2024 papers. MiniMax-M2.7's 256-expert sigmoid-routed MoE is an unusual design Atlas had to support natively.
- **Switch Transformer** — Fedus, Zoph, Shazeer (2021). The top-1 routing baseline.

## Adjacent projects worth studying

- **`scitix/InstantTensor`** — the fast safetensors loader Atlas's `O_DIRECT` + pipelined reader is modeled on.
- **`huggingface/tokenizers`** — the tokenizer library Atlas wraps.
- **`huggingface/safetensors`** — format spec + Rust impl.
- **`PyKeOps` / symbolic autograd** — not used, but the philosophy (compile once per shape, amortise forever) resonates with AI Kernel HyperCompiling.

## Atlas-internal references

Inside the repo, the canonical long-form references are in [`docs/design/`](https://github.com/Avarok-Cybersecurity/atlas/tree/main/docs/design). Notable:

- `NVFP4_COHERENCE.md` — why `--kv-high-precision-layers` exists.
- `fp8-native-design.md` — end-to-end FP8 serving design.
- `ep2-token-dispatch-design.md` — EP=2 MoE dispatch.
- `turboquant-nightjob-2026-03-31.md` — TurboQuant KV.
- `xgrammar-integration-plan.md`, `xgrammar2-upgrade-plan.md` — constrained decoding history.
- `tool-calling-gap-analysis.md` — the broader tool-call reliability story.
- `mixkvq-design.md` — mixed-precision KV per layer.
- `qwen35-ttft-roadmap.md` — TTFT improvements on Qwen3.5.
- `single-kernel-prefill-proposal.md` — the theory behind prefill v47.
- `nvfp4-quantizer-plan.md` — the quantizer's numeric path.
- `agentic-quality-research-synthesis.md` — agent-workload quality research.

For the broader research context that informed Atlas's direction, see `docs/atlas-spark-research-articles.md` in the repo — a rolling curated list that's longer and more current than this page.
