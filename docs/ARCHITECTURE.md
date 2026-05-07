# Atlas Architecture

A 5-minute tour of the crate graph, the build pipeline, and the request lifecycle.

## Crate graph

Atlas is a single Cargo workspace with 15 crates organized into four layers:

```
                         ┌─────────────────────────┐
   HTTP / scheduling     │      spark-server       │
                         │  (OpenAI + Anthropic    │
                         │  API, scheduler, tool   │
                         │  parsing, grammar)      │
                         └──────────┬──────────────┘
                                    │
                         ┌──────────┴──────────────┐
   model + layers        │       spark-model       │
                         │  (TransformerModel,     │
                         │  per-arch layers, MoE,  │
                         │  SSM, attention, MTP,   │
                         │  weight loading)        │
                         └──────────┬──────────────┘
                                    │
            ┌───────────────────────┼─────────────────────┐
            │                       │                     │
   GPU runtime          inter-rank comm           NVMe-backed swap
   ┌────────┴────────┐ ┌────────────┴───────┐  ┌──────────┴────────┐
   │  spark-runtime  │ │    spark-comm      │  │  spark-storage    │
   │  (PagedKvCache, │ │  (NCCL backend +   │  │  (HSS, io_uring)  │
   │  buffers,       │ │  RDMA, all_reduce, │  └───────────────────┘
   │  cuda_backend,  │ │  broadcast)        │
   │  sampler,       │ └────────────────────┘
   │  weights)       │
   └────────┬────────┘
            │
   ┌────────┴────────┐
   │  atlas-kernels  │   PTX bytes embedded at compile time
   │  (build.rs:     │   per (hardware, model, quant) tuple
   │  compile *.cu   │   from kernels/<hw>/<model>/<quant>/
   │  → PTX → embed) │
   └────────┬────────┘
            │
   ┌────────┴────────┐    Stable types + traits shared across the whole tree
   │  atlas-core     │    — ModelConfig, registry, hardware capability
   │                 │    enums, error types.
   └─────────────────┘

   Placeholder crates (not yet wired into the hot path):
   atlas-activation · atlas-norm · atlas-quant · atlas-embed · atlas-reduce
   These exist as future homes for host-side primitives and are
   currently thin scaffolding so the dependency graph is stable.

   Test/bench crates:
   atlas-spark-bench  (Criterion microbenchmarks per kernel category)
   cufile-sys         (FFI for NVIDIA cuFile / GDS — currently dormant on GB10)
```

## Layer-by-layer

### `atlas-core`
The shared "vocabulary" crate. Defines:
- `ModelConfig` — every supported model's HuggingFace `config.json` parsed into a normalized struct, plus per-family overrides (Qwen3.5 / Qwen3.6 / Nemotron / Gemma-4 / Mistral / MiniMax).
- `LayerType` — `FullAttention | LinearAttention | Mlp | Moe | Dense`.
- `KernelTarget` / `TargetPtxSet` — what kernels the binary was compiled with.
- Public error / result types.

Every other crate depends on this. **Keep it stable.** Breaking changes here ripple everywhere.

### `atlas-kernels`
A pure-build-script crate whose `build.rs` (~915 LoC) does the heavy lifting:

1. Reads `kernels/<hw>/HARDWARE.toml` for arch, sm, fp32-residual flag.
2. Reads `kernels/<hw>/<model>/MODEL.toml` for model-specific kernel-target metadata, sampling presets, behavior knobs.
3. For each `(hardware, model, quant)` tuple selected via `ATLAS_TARGET_*` env vars (or `*` wildcard), compiles every `*.cu` under `kernels/<hw>/<model>/<quant>/` and `kernels/<hw>/<quant>/` (shared) to PTX via `nvcc`, deduplicating model-specific overrides over the shared pool.
4. Emits a generated `target_ptx.rs` containing `static PTX_BYTES: &[(&str, &[u8])]` arrays — the kernel modules as PTX strings linked into the binary.
5. Marks `cargo:rerun-if-changed=` on every `.cu` and `.toml` it touched so incremental builds work.

Skip mode (`ATLAS_SKIP_BUILD=1`): emits a stub registry. Used by CI for GPU-free `cargo check` / `clippy` runs.

### `spark-runtime`
GPU-side runtime primitives:
- `cuda_backend.rs` — `GpuBackend` trait + `AtlasCudaBackend` (cudarc-based) impl. Wraps device pointers, kernel handles, streams, allocator.
- `buffers.rs` — pinned-host + device scratch arenas sized per-batch.
- `kv_cache.rs` — paged KV cache with FP8 / NVFP4 / BF16 dtype variants.
- `sampler.rs` — host-side sampling (top-k, top-p, temperature, repetition penalty, DRY, Lloyd–Max). The GPU produces logits; host samples.
- `weights.rs` — safetensors loader with shard streaming + OOM guard.

Hardware-agnostic: the `GpuBackend` trait could be implemented by another vendor backend. Today only `AtlasCudaBackend` exists.

### `spark-comm`
NCCL-backed multi-rank communication:
- `nccl_backend.rs` — collective ops (`all_reduce`, `broadcast`, `all_gather`) with RoCEv2 wiring.
- Used by EP/TP layers to exchange expert outputs / sharded attention outputs.

### `spark-storage`
NVMe-backed KV cache swap (high-speed swap). When the on-device KV cache fills, blocks evict to disk via `io_uring` and stream back on demand. GB10 lacks GDS support, so the path uses pinned-host bounce buffers.

### `spark-model`
The bulk of the inference logic:
- `model.rs` (split into `model/`) — `TransformerModel` orchestrator: per-layer forward dispatch, prefill/decode/verify path selection, SSM state management, KV cache binding, EP/TP composition.
- `layers/` — per-architecture layer impls:
  - `qwen3_attention/` — full-attention with KV cache (paged or MLA); decode + prefill + verify trait impls.
  - `qwen3_ssm/` — gated delta net (Qwen3-Next/3.5/3.6 hybrid models).
  - `nemotron_mamba2/` — Mamba2 SSM variant.
  - `moe/` — sparse mixture-of-experts (routed + shared expert).
  - `dense_ffn/` — non-MoE FFN.
  - `vision_encoder/` — Qwen3-VL ViT.
  - `mtp_head.rs` / `dflash_head/` — speculative drafter heads.
- `weight_loader/` — per-architecture weight name → typed-struct mapping; one variant per family (`qwen3.rs`, `qwen35.rs`, `gemma4.rs`, `nemotron.rs`, `mistral_loader.rs`, `minimax.rs`, `qwen3_vl.rs`, `dflash_loader.rs`).
- `weight_map/` — quant-format detection + dequant utilities (NVFP4 E2M1 + FP8 group scales, FP8 block-scaled, BF16 raw).
- `traits.rs` — `Model` trait that the scheduler talks to. The trait is split across multiple `impl Model for TransformerModel { ... }` blocks under `model/trait_impl/` (see `docs/adr/0006-multi-file-module-idiom.md`).
- `factory.rs` — given a `ModelConfig`, picks the right `WeightLoader` and constructs a `TransformerModel`.

### `spark-server`
HTTP + scheduling:
- `main_modules/serve.rs` — boot sequence (12 phases under `serve_phases/`).
- `scheduler/` — main scheduler loop (`run()`), per-step phases (decode, verify K=2/3/γ, MTP, NGram, prefill, sample, emit, lifecycle).
- `api/` — OpenAI-compat handlers (`chat_completions`, `responses`, `completions`, etc.).
- `anthropic/` — Anthropic API translation layer.
- `openai/` — OpenAI request/response types.
- `tool_parser/` — per-model tool-call format parsers (Hermes JSON, Qwen3-coder XML, Gemma-4 native, MiniMax XML, Mistral native, bare JSON, etc.).
- `grammar/` — XGrammar-backed constrained decoding (tool schemas, JSON mode).
- `tokenizer.rs` — chat-template rendering via Jinja, streaming decode.

## Request lifecycle (decode path)

1. **HTTP** — Axum receives `POST /v1/chat/completions`, body parsed in `api/chat/mod.rs`.
2. **Pre-process** — `chat/msg_entry::build_msg_entries` extracts messages + tools; `chat/template::render_template` renders the chat template; `chat/sampling_setup::build_sampling` resolves sampling preset, stops, grammar.
3. **Submit** — request enqueued onto the scheduler's pending queue.
4. **Scheduler tick** (`scheduler/mod.rs::run` loop):
   - Promote prefill-completed sequences to active.
   - Sample drafts (MTP / NGram / DFlash if enabled).
   - Verify drafts via the model's verify path (`decode_verify_k2/k3/k4/kgamma`).
   - Emit accepted tokens via SSE or buffer for blocking response.
5. **Per-token forward**:
   - For each layer: `TransformerLayer::decode` — `qwen3_attention`, `qwen3_ssm`, `moe`, etc.
   - GPU kernels launched via `gpu.kernel(module, fn)` lookups against the embedded `TargetPtxSet`.
   - KV cache append, paged decode attention, SSM state update, MoE routing + expert GEMV.
6. **Sample** — host pulls logits, applies sampling, emits token.

## Build pipeline

```
$ ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.6-35b-a3b ATLAS_TARGET_QUANT=nvfp4 \
    cargo build --release -p spark-server
```

What happens:
1. Cargo invokes `atlas-kernels/build.rs`.
2. `build.rs` walks `kernels/gb10/qwen3.6-35b-a3b/nvfp4/*.cu` + `kernels/gb10/common/*.cu` (shared), invokes `nvcc -arch=sm_121 --ptx` on each, and emits `OUT_DIR/target_ptx.rs` with the resulting PTX bytes.
3. The Rust crates compile, linking the generated module.
4. At runtime, `spark-runtime::gpu` loads the PTX into the CUDA driver and exposes kernel handles.

For a multi-target binary (sweep mode): `ATLAS_TARGET_MODEL='*' ATLAS_TARGET_QUANT='*'` compiles every kernel target. The binary picks the right target at startup based on the loaded model's `model_type` and `hidden_size`.

## See also

- `docs/HARDWARE.md` — adding a new SM target / model family.
- `docs/DEPLOYMENT.md` — Docker, multi-rank, NVMe swap.
- `docs/adr/` — Architecture Decision Records explaining major design choices.
