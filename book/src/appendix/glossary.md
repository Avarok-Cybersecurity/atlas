# Glossary

Short definitions for the acronyms and names that recur in this book and the Avarok codebase.

| Term | Definition |
|---|---|
| **AGPL-3.0** | GNU Affero General Public License, v3. Avarok's community-edition license. Copyleft; network use counts as distribution. |
| **axum** | Rust async web framework (built on tokio + tower). Avarok's HTTP layer. |
| **BF16** | Brain Floating-Point 16. 1 sign + 8 exponent + 7 mantissa. Standard precision for Avarok activations and residual streams. |
| **CLA** | Contributor License Agreement. Required before Avarok PR merge; see `CLA.md`. |
| **CommBackend** | Avarok's trait for collective ops (all-reduce, broadcast, send/recv). NCCL-backed in production; no-op in single-GPU. |
| **ComputeTarget** | Avarok's build-time trait for vendor-specific compilers (`nvcc`, `xcrun metal`, `hipcc`, `icpx`). |
| **conv1d** | 1D convolution, typically causal with small kernel width (3–4). Used in Mamba-style SSMs. |
| **CUTLASS** | NVIDIA's open-source CUDA template library for GEMM and related ops. Avarok uses it for certain NVFP4 paths. |
| **cp.async** | CUDA instruction for asynchronous global-to-shared-memory copies. Key to pipelining on SM80+ architectures. |
| **DGX Spark** | NVIDIA's GB10-based workstation. Avarok's initial hardware target. |
| **DType** | Data type enum in `avarok-core` — E2M1, FP8E4M3, FP8E5M2, BF16, FP16, FP32. |
| **E2M1** | 4-bit float format: 1 sign + 2 exponent + 1 mantissa. Values: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}. The storage format of NVFP4 weights. |
| **E4M3** | 8-bit float format: 1 sign + 4 exponent + 3 mantissa. The standard FP8 format. |
| **EP=2** | Expert Parallelism across 2 nodes. Avarok's multi-GPU shape — experts split across ranks, other layers replicated. |
| **Flash Attention** | Tiled online-softmax attention kernel family. Avarok's prefill kernel builds on FA-2 + FA-4. |
| **FP8** | 8-bit floating-point. In Avarok context, usually E4M3. |
| **GB10** | NVIDIA Grace-Blackwell GB10 Superchip. SM121. 119.7 GB unified memory. |
| **GDN** | Gated Delta Rule. Qwen3.5's variant of the delta-net SSM. |
| **GDR** | Gated Delta Rule (see above) / also NCCL's GPUDirect RDMA level. Context-dependent. |
| **GEMM** | General matrix-matrix multiplication. The workhorse tensor-core op. |
| **GeGLU** | Gated GELU. Gemma-4's activation. |
| **GpuBackend** | Avarok's runtime trait for GPU ops (memory, launch, streams, graphs). CUDA-backed in production; mockable for tests. |
| **Grace** | The ARM CPU half of GB10. Used for CPU-side NEON SIMD precomputation (e.g. RoPE tables). |
| **HARDWARE.toml** | Per-hardware metadata in `kernels/<hw>/`. Vendor, arch, memory specs. |
| **HF** | HuggingFace. Avarok loads HF-format checkpoints via `safetensors`. |
| **HyperCompiling** | "AI Kernel HyperCompiling" — Avarok's philosophy. Specialize per `(H, M_q)` target; abstractions stay above the kernel layer. |
| **IORouter** | The SBIO pattern name for an I/O-side trait (`GpuBackend`, `CommBackend`, `WeightStore`). |
| **KernelTarget** | The `(arch, model, quant)` dispatch key. `avarok-core::target::KernelTarget`. |
| **KV cache** | Cached key and value tensors from attention. Paged in Avarok. |
| **LPDDR5X** | The memory technology GB10 uses. Unified with CPU; 273 GB/s peak bandwidth. |
| **Mamba / Mamba-2** | Selective state-space models. Mamba-2 is the variant used by Nemotron-H. |
| **Marconi** | Avarok's SSM snapshot cache. Extension of RadixAttention to hybrid models. |
| **MMA** | Matrix Multiply-Accumulate. NVIDIA's tensor-core operation (`mma.sync.aligned.m16n8k16.*`). |
| **MoE** | Mixture of Experts. FFN split into N experts with per-token top-k routing. |
| **MODEL.toml** | Per-model metadata in `kernels/<hw>/<model>/`. Sampling presets, thinking budget, behavior defaults. |
| **MRoPE** | Multi-RoPE. Variant of RoPE that splits head dim into spatial (H, W) and temporal (T) segments. Used by vision models. |
| **MTP** | Multi-Token Prediction. Avarok's speculative-decoding mechanism using a model-native draft head. |
| **NCCL** | NVIDIA's collective-ops library for multi-GPU / multi-node. |
| **NVFP4** | 4-bit E2M1 weights + FP8 E4M3 per-block scales (block=16). Avarok's flagship quant format on GB10. |
| **O_DIRECT** | Linux open flag that bypasses the page cache. Used by Avarok's fast safetensors loader. |
| **PCND** | "Prefer Config / No Defaults" — a user-instruction principle: no implicit defaults in production paths. |
| **PTX** | Parallel Thread Execution. NVIDIA's virtual ISA; Avarok's compiled kernels ship as PTX. |
| **RadixAttention** | Prefix-caching mechanism built on a radix tree over token sequences. |
| **RMSNorm** | Root-mean-square normalization. The norm used in every modern transformer Avarok supports. |
| **RoCE** | RDMA over Converged Ethernet. Avarok's multi-node transport. |
| **RoPE** | Rotary Position Embedding. Position encoding used by every transformer in the support matrix. |
| **Rust** | The language Avarok is written in (stable, edition 2024). |
| **SafeTensors** | Format for HF model checkpoints. Avarok's loader reads it directly. |
| **SBIO** | Separation of Business logic and I/O. The architectural pattern that keeps Avarok ~80% unit-testable without a GPU. |
| **SDD** | Structure-Driven Development. User-instruction principle for abstraction design. |
| **SLAI** | SLO-Aware Inference. Avarok's TBT-deadline-aware scheduling policy. |
| **SM100 / SM101 / SM120 / SM121** | NVIDIA Streaming Multiprocessor architecture identifiers. GB10 is SM121. |
| **SSM** | State-Space Model. Mamba / delta-net style layer. |
| **SSOT** | Single Source of Truth. User-instruction principle. |
| **TBT** | Time Between Tokens. The decode-step latency SLAI optimises. |
| **TTFT** | Time To First Token. The prefill-stage latency. |
| **Tensor core** | Dedicated MMA hardware on NVIDIA GPUs; Avarok targets the BF16 and E4M3 tensor cores on SM121. |
| **TurboQuant** | Avarok's WHT + Lloyd-Max 4/3/8-bit KV-cache quant format. Lower MSE than NVFP4 at the same bit rate. |
| **vLLM** | Popular open-source LLM inference framework. Avarok's primary throughput baseline. |
| **WHT** | Walsh-Hadamard Transform. Used in TurboQuant to flatten outliers before quantization. |
| **XGrammar** | Token-bitmap automaton for constrained decoding. Avarok's tool-call + structured-output enforcement substrate. |
