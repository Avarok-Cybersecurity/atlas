# Kernel Dispatch Pipeline

One Atlas binary contains kernels for every `(Hardware, Model, Quantization)` target it was built for. This chapter traces a single chat completion from the moment the HTTP request arrives to the moment a kernel launches on the GPU, so you know exactly where each piece of dispatch lives.

## The high-level flow

```
             1. HTTP request                                    7. Kernel launch
┌──────────────┐   ┌──────────┐   ┌───────────┐   ┌───────────┐   ┌──────────┐
│ OpenAI       │──►│ axum     │──►│ scheduler │──►│ engine    │──►│ PTX on   │
│ client       │   │ (server) │   │ (server)  │   │ (model)   │   │ GPU      │
└──────────────┘   └──────────┘   └───────────┘   └───────────┘   └──────────┘
                                                        │               ▲
                                                        ▼               │
                                                 ┌──────────────────────┴───┐
                                                 │ KernelTarget → PtxModule │
                                                 │ (atlas-kernels)          │
                                                 └──────────────────────────┘
```

The dispatch decisions happen in two distinct phases:

- **Build time** — `atlas-kernels/build.rs` decides *which PTX to embed*.
- **Startup** — `spark-server::main` decides *which embedded PTX to upload to the GPU* based on the model being served.

After startup, the fast path is deterministic: a layer's `forward(ctx)` always calls the same `KernelHandle`s, always on the same GPU stream, always in the same order. There is no per-request dispatch decision. This is the payoff of the specialization thesis — no branching, no polymorphism across kernel variants, no cache miss.

## Phase 1 — build time: which PTX gets embedded

`atlas-kernels/build.rs` runs during `cargo build`. Its job:

1. Read the three wildcards:
   - `ATLAS_TARGET_HW` (default `gb10`; accepts `*`)
   - `ATLAS_TARGET_MODEL` (default `*` — all)
   - `ATLAS_TARGET_QUANT` (default `*` — all)
2. Enumerate `kernels/<hw>/<model>/<quant>/` leaves matching the wildcards.
3. For each leaf, read `HARDWARE.toml` to learn the vendor, and call `resolve_compute_target(vendor)` to get a `Box<dyn ComputeTarget>`:
   - `Vendor::Nvidia` → `NvidiaTarget { nvcc }`
   - `Vendor::Apple` → `AppleTarget { xcrun }` (planned)
   - `Vendor::Amd` → `AmdTarget { hipcc }` (planned)
4. Call `compute_target.compile(source, out, arch, extra_flags)` on every `.cu` / `.metal` / `.hip` file in the leaf.
5. Parse `KERNEL.toml` for module-name overrides (some kernels are compiled from `e2m1_branchless.cu` but exposed at runtime as the `e2m1` module).
6. Emit an auto-generated Rust file, `$OUT_DIR/target_ptx.rs`, that `include!()`'s back into `atlas-kernels/src/lib.rs`. The generated file contains one `pub static PTX_<TARGET>: &[PtxModule]` per target plus an `all_ptx_sets()` function that returns the whole set.

The output is one single PTX set per target, embedded in the final `spark-server` binary as a byte slice. This is why "one Docker image, one binary, zero runtime compilation" is true.

`ATLAS_SKIP_BUILD=1` short-circuits the whole phase: `build.rs` emits a stub `target_ptx.rs` with empty constants so that `clippy`, `fmt`, and non-GPU tests can compile on a Linux host with no `nvcc`. The CI in `.github/workflows/ci.yml` uses this.

## Phase 2 — startup: which embedded PTX gets uploaded

When the user runs `spark serve <model-id>`, `spark-server/src/main.rs` does the following, roughly in order:

1. **Parse the model config.** `atlas_core::config::ModelConfig::from_hf(&model_path)` reads `config.json` and its nested text/vision configs.
2. **Canonicalize `model_type`.** Lowercase, replace `-` and `.` with `_`. `"Qwen3.5_NextForCausalLM"` becomes `"qwen3_5_next_for_causal_lm"`. This is the key we dispatch on.
3. **Resolve the KernelTarget.** Given `model_type` and the selected quantization (from config or `--kv-cache-dtype` when overriding), `atlas-kernels::select_target(hw, model, quant)` looks up the matching `KernelTarget`. Fail fast with a clear error if there's no match.
4. **Instantiate the GpuBackend.** `AtlasCudaBackend::new(gpu_ordinal, &ptx_set.modules)` uploads every embedded PTX module for the chosen target to the GPU, via `cuModuleLoadData`. Kernel handles are cached per `(module_name, function_name)` pair.
5. **Instantiate the ModelWeightLoader.** `spark_model::factory::loader_for_config(&config)` matches on the canonical `model_type` and returns `Box<dyn ModelWeightLoader>`.
6. **Load weights.** The loader translates HF weight names (`model.layers.0.self_attn.q_proj.weight`) into Atlas layer types (`Qwen3AttentionLayer`), going through `WeightStore` (the `O_DIRECT` fast path) and the quantization helpers in `spark_model::weight_map`.
7. **Build layer trait objects.** Each loaded layer becomes a `Box<dyn TransformerLayer>` stored in the `InferenceEngine`.
8. **Capture CUDA graphs.** For each supported batch size, `engine.capture_graph(bs)` runs a single decode step inside a graph region. Subsequent decodes replay the graph — one GPU launch for the whole forward pass.
9. **Bind the HTTP endpoint.** `axum::Router::new()...serve(&addr)` starts listening.

At this point dispatch is frozen. Every request goes through the same kernels, the same graph, the same streams.

## Phase 3 — per-request path

```
POST /v1/chat/completions
 │
 ▼
spark_server::api::chat_completions    (axum handler)
 │
 ▼  1. Apply jinja chat template
 │  2. Tokenize
 │  3. Enqueue Request {prompt_ids, sampling, stream?, tools?}
 │
 ▼
spark_server::scheduler                (SLAI or FIFO)
 │
 ▼  1. Allocate KV pages for prefix
 │  2. Chunked prefill through InferenceEngine
 │  3. Enter the decode loop
 │
 ▼
spark_model::engine::InferenceEngine::decode_step
 │
 ▼  for layer in layers:
 │      layer.forward(&ctx)      ← dyn dispatch, one per layer
 │          └─ calls into ops.rs kernel launches
 │              └─ GpuBackend::launch(KernelHandle, grid, block, args, stream)
 │                  └─ CUDA cuLaunchKernel    (PTX on GPU)
 │
 ▼
Sampler                              (argmax / top-p / top-n-sigma / min-p)
 │
 ▼
Detokenize → stream chunk → HTTP response
```

Two dynamic-dispatch points:

- **`dyn TransformerLayer`** — one virtual call per layer per step. Layer types (`Qwen3AttentionLayer`, `MoeLayer`, `Qwen3SsmLayer`, `NemotronMamba2Layer`, `VisionEncoder`) hold their own pre-resolved `KernelHandle`s for the ops they need. The virtual call is cheap — typically ~ns — against a forward pass that takes ~0.1–1 ms per token.
- **`&dyn GpuBackend`** — one virtual call per kernel launch. Same argument; the overhead is negligible compared to the kernel itself.

Both virtual calls are unavoidable consequences of the specialization thesis: we want `spark-server` to not know what `GpuBackend` it's talking to, and we want `InferenceEngine` to not know what layer shape it's running. That's how new hardware and new models plug in.

With CUDA graphs enabled (the default in production), steps 5–6 collapse to a single `cuGraphLaunch` — the dynamic dispatch cost disappears into the graph capture phase.

## Where to look in the code

| Question | File |
|---|---|
| "How is `KernelTarget` resolved at startup?" | `crates/atlas-kernels/src/lib.rs`, look for `select_target()` + `include!(target_ptx.rs)` |
| "How does a kernel get compiled at build time?" | `crates/atlas-kernels/build.rs`, `crates/atlas-core/src/compute.rs` |
| "How does a layer launch a kernel?" | `crates/spark-model/src/layers/ops.rs`, look for `KernelLaunch::new(gpu, kernel).grid(...).arg_ptr(...).launch(stream)` |
| "How does the engine loop over layers?" | `crates/spark-model/src/engine.rs` |
| "How does the factory pick a `ModelWeightLoader`?" | `crates/spark-model/src/factory.rs` — `loader_for_config()` |
| "How is the HTTP request parsed into a scheduler job?" | `crates/spark-server/src/api.rs`, `crates/spark-server/src/scheduler.rs` |

The [spark-runtime chapter](../crates/spark-runtime.md) expands on `GpuBackend`; [spark-model](../crates/spark-model.md) on the layer/factory split; [atlas-kernels](../crates/atlas-kernels.md) on the build-time codegen. The [SBIO chapter](./sbio.md) explains why every arrow in the diagram above goes through a trait.
