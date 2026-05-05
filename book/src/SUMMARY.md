# Summary

[Introduction](./introduction.md)

---

# Part I — Getting Started

- [Philosophy](./getting-started/philosophy.md)
- [Installation](./getting-started/installation.md)
- [Quickstart](./getting-started/quickstart.md)
- [Supported Models](./getting-started/models.md)

# Part II — Architecture

- [Philosophy: AI Kernel HyperCompiling](./architecture/philosophy.md)
- [Workspace Layout](./architecture/workspace.md)
- [Kernel Dispatch Pipeline](./architecture/dispatch.md)
- [SBIO: Business Logic vs I/O](./architecture/sbio.md)

# Part III — The Crates

- [atlas-core](./crates/atlas-core.md)
- [atlas-quant](./crates/atlas-quant.md)
- [Primitives: norm / activation / embed / reduce](./crates/atlas-primitives.md)
- [atlas-kernels](./crates/atlas-kernels.md)
- [spark-runtime](./crates/spark-runtime.md)
- [spark-comm](./crates/spark-comm.md)
- [spark-model](./crates/spark-model.md)
- [spark-server](./crates/spark-server.md)
- [atlas-spark-bench](./crates/atlas-spark-bench.md)

# Part IV — Engineering Deep Dives

- [CUDA Kernel Engineering](./deep-dives/kernels.md)
- [NVFP4 Quantization](./deep-dives/nvfp4.md)
- [FP8 Native Serving](./deep-dives/fp8.md)
- [Attention & Paged KV Cache](./deep-dives/attention.md)
- [MoE Routing & Experts](./deep-dives/moe.md)
- [SSM / Mamba / GDN Layers](./deep-dives/ssm.md)
- [Speculative Decoding (MTP)](./deep-dives/mtp.md)
- [Constrained Decoding (XGrammar)](./deep-dives/xgrammar.md)

# Part V — Operating Atlas

- [OpenAI-Compatible Server](./operations/server.md)
- [Tool Calling & Streaming](./operations/tools.md)
- [Multi-GPU & EP=2](./operations/multi-gpu.md)
- [Benchmarking](./operations/benchmarks.md)

# Part VI — The Project

- [Contributing](./project/contributing.md)
- [Security Policy](./project/security.md)
- [Release Notes](./project/releases.md)

---

# Appendix

- [Paper Summary](./appendix/paper.md)
- [A Category-Theoretic Perspective](./appendix/category-theory.md)
- [Glossary](./appendix/glossary.md)
- [Further Reading](./appendix/reading.md)

---

[API Reference ↗](./api-redirect.md)
