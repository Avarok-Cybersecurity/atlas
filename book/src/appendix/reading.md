# Further Reading

Curated references that informed Avarok's design. Not exhaustive — just the papers, articles, and prior-art projects worth reading if you want to understand why Avarok is shaped the way it is.

## Kernel engineering

- **FlashAttention-2** — Tri Dao (ICLR 2024). [arXiv:2307.08691](https://arxiv.org/abs/2307.08691). Foundation of Avarok's prefill kernels — tiled online softmax, Q/K/V tiling with shared memory, causal masking.
- **FlashAttention-4** — Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao (2025). [arXiv:2603.05451](https://arxiv.org/abs/2603.05451). Conditional softmax rescaling (skip ~90% of rescale ops), software polynomial exponential (`sw_exp`, avoids SFU bottleneck).
- **FlashInfer** — Ye et al. (MLSys 2025 Best Paper). [arXiv:2501.01005](https://arxiv.org/abs/2501.01005). Block-sparse paged KV, gather-SMEM-MMA pattern. Informed Avarok's paged decode kernel.
- **SageAttention 3** — Zhang et al. (NeurIPS 2025). [arXiv:2505.11594](https://arxiv.org/abs/2505.11594). Native FP4 attention on newer Blackwell. Planned direction when SM12x+ silicon lands.
- **LeanAttention** — Roy, Vassilieva, Willke, Mendis (2024). [arXiv:2405.10480](https://arxiv.org/abs/2405.10480). Stream-K tile scheduling for decode attention. Planned for SM occupancy improvements.
- **CUTLASS** documentation and examples. The BF16 MMA fragment shapes and `cp.async` pipelining patterns in Avarok's kernels follow CUTLASS conventions.

## State-space models

- **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** — Gu, Dao (2023). [arXiv:2312.00752](https://arxiv.org/abs/2312.00752). The original selective SSM.
- **Mamba-2** — Dao, Gu (ICML 2024). [arXiv:2405.21060](https://arxiv.org/abs/2405.21060). The variant Nemotron-H uses.
- **Gated Delta Networks** — Yang, Dao, et al. (2024). Closer to Qwen3.5's GDN formulation.
- **GDN register-tile results** — Avarok's internal experiments in `gdn_regtile_results.md` track tile-shape tradeoffs on GB10.

## Quantization

- **NVFP4 / FP4 microscaling** — NVIDIA's blog posts on Blackwell FP4. The public docs for SM120 coverage are thin; much of Avarok's SM121 workaround has no upstream equivalent yet.
- **SmoothQuant** — Xiao, Lin, et al. (ICML 2023). Scale factor calibration ideas used indirectly in Avarok's FP8 KV calibration.
- **Compressed-Tensors format** — the HF `compressed-tensors` library's on-disk FP8 block-scaled layout.
- **TurboQuant** (Avarok internal) — `docs/turboquant-plus.md`. WHT + Lloyd-Max 4-bit KV cache with ~2× lower MSE than NVFP4 at the same bit rate.

## Speculative decoding

- **Fast Inference from Transformers via Speculative Decoding** — Leviathan, Kalman, Matias (ICML 2023). The foundational paper.
- **Medusa: Multi-Token Prediction Heads** — Cai et al. (2024). MTP-style draft heads.
- **Self-speculative Decoding** — Layer-skipping drafter. Avarok implements a variant.

## Constrained decoding

- **XGrammar** — Li, Chen, Chen, et al. (2024). [arXiv:2411.15100](https://arxiv.org/abs/2411.15100). The token-bitmap automaton approach Avarok uses.
- **SGLang** — Zheng et al. (NeurIPS 2024). Broader exploration of structured-output techniques; Avarok shares design elements with SGLang's `regex_fsm`.

## Inference systems

- **vLLM** — Kwon et al. (SOSP 2023). The PagedAttention paper. Avarok's paged KV cache follows the vLLM model with Avarok-specific kernel work below it.
- **TensorRT-LLM** documentation and source. Avarok's TRT-LLM benchmark comparisons in `docs/AVAROK_SPARK_JOURNEY.md` are informed by reading the TRT-LLM codebase; that journey records the 29.6 tok/s ceiling for NVFP4 on SM121 TRT-LLM.
- **SGLang** — structured-generation inference framework.
- **Triton** — inference server. Peripheral; Avarok does not use it but the operational patterns are informative.

## MoE routing

- **Mixture-of-Experts with Sigmoid Routing** — various 2023–2024 papers. MiniMax-M2.7's 256-expert sigmoid-routed MoE is an unusual design Avarok had to support natively.
- **Switch Transformer** — Fedus, Zoph, Shazeer (2021). The top-1 routing baseline.

## Adjacent projects worth studying

- **`scitix/InstantTensor`** — the fast safetensors loader Avarok's `O_DIRECT` + pipelined reader is modeled on.
- **`huggingface/tokenizers`** — the tokenizer library Avarok wraps.
- **`huggingface/safetensors`** — format spec + Rust impl.
- **`PyKeOps` / symbolic autograd** — not used, but the philosophy (compile once per shape, amortise forever) resonates with AI Kernel HyperCompiling.

## Avarok-internal references

Inside the repo, the canonical long-form references are the architecture decision records in [`docs/adr/`](https://github.com/Avarok-Cybersecurity/avarok/tree/main/docs/adr), plus the top-level notes alongside them. Notable:

- `docs/adr/0004-nvfp4-fp8-quantization.md` — NVFP4/FP8 quantization, including why `--kv-high-precision-layers` exists.
- `docs/adr/0003-hybrid-ssm-attention.md` — hybrid SSM/attention design and chunked SSM prefill.
- `docs/adr/0007-tp-ep-composition.md`, `docs/adr/0011-ep-batched-decode-optimization.md` — EP=2 MoE dispatch and batched decode.
- `docs/adr/0010-vendor-xgrammar.md` — constrained decoding via vendored XGrammar.
- `docs/turboquant-plus.md` — TurboQuant KV.
- `docs/ARCHITECTURE.md`, `docs/AVAROK_KERNELS.md`, `docs/HARDWARE.md` — the system, kernel, and hardware overviews.

For the broader research context that informed Avarok's direction, see `docs/avarok-spark-research-articles.md` in the repo — a rolling curated list that's longer and more current than this page.
