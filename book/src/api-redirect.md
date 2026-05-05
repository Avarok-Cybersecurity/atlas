# API Reference

<script>window.location.replace("/api/");</script>
<noscript>
<meta http-equiv="refresh" content="0; url=/api/">
</noscript>

If you are not redirected automatically, follow this link to the [Rust API reference](/api/).

The API reference is generated from the crate source with `cargo doc --workspace --no-deps` on every merge to `main`. Top-level crates:

- [`atlas_core`](/api/atlas_core/) — target abstractions, tensor, dtype, kernel registry
- [`atlas_quant`](/api/atlas_quant/) — NVFP4 / FP8 quantization kernels
- [`atlas_norm`](/api/atlas_norm/) / [`atlas_activation`](/api/atlas_activation/) / [`atlas_embed`](/api/atlas_embed/) / [`atlas_reduce`](/api/atlas_reduce/) — primitive op traits
- [`atlas_kernels`](/api/atlas_kernels/) — embedded PTX registry
- [`spark_runtime`](/api/spark_runtime/) — GPU backend, KV cache, sampler
- [`spark_comm`](/api/spark_comm/) — collective-op trait + NCCL impl
- [`spark_model`](/api/spark_model/) — layer assembly, weight loaders, engine
- [`spark_server`](/api/spark_server/) — HTTP server, tool parsing
- [`atlas_spark_bench`](/api/atlas_spark_bench/) — benchmark client
