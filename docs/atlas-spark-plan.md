# Atlas Spark — Phase 1 Implementation Plan

**Goal**: Single-request greedy decode of Qwen3-Next-80B-A3B-Instruct-NVFP4, all Atlas kernels, pure Rust, OpenAI-compatible API.

---

## Principles Applied

```
**Principles Applied:**
- SDD: GSI ≥ 2 for Model (Qwen3 now, LLaMA/Mistral later), KernelBackend
  (SM121 now, future SM targets), WeightLoader (safetensors now, GGUF later),
  CommBackend (NCCL now, future transports)
- SBIO: GPU dispatch, filesystem, network, and inter-GPU communication are all
  I/O — routed through GpuBackend, StorageBackend, CommBackend, and async
  HTTP handler traits
- SSOT: ModelConfig + ParallelConfig are the single sources for model dims and
  sharding — no hardcoded shapes or ranks in kernels
- PCND: All runtime parameters (model path, port, GPU ordinal, tp_size, ep_size,
  max_seq_len, block_size) required explicitly via CLI args — no implicit
  defaults in production code

**Violations Avoided:**
- PCND: Removed implicit defaults for max_seq_len, block_size, GPU ordinal,
  parallelism config — all must be specified or fail fast
- SSOT: Weight name mapping derives layer count/types from ModelConfig; weight
  sharding derives from ParallelConfig instead of hardcoding
- SBIO: CUDA kernel dispatch goes through GpuBackend trait; inter-GPU
  communication goes through CommBackend trait — not direct cuLaunchKernel
  or ncclAllReduce calls in business logic
```

---

## SDD Analysis

### Abstraction 1: Model

| | |
|---|---|
| **GSI** | 2 (Qwen3 now, LLaMA/Mistral planned) |
| **Variation points** | Layer composition (attention vs SSM vs dense), MoE vs dense FFN, attention pattern (GQA ratio), quantization format |
| **Common behavior** | [prefill()](file:///workspace/atlas/crates/atlas-py/src/vllm.rs#223-295), [decode()](file:///workspace/atlas/crates/atlas-py/src/vllm.rs#296-350), `vocab_size()`, layer iteration |

### Abstraction 2: GpuBackend (SBIO IORouter for GPU)

| | |
|---|---|
| **GSI** | 2 (AtlasCudaBackend for production, MockGpuBackend for tests) |
| **Variation points** | Kernel dispatch, memory allocation, stream management |
| **Common behavior** | `alloc()`, `free()`, `copy_h2d()`, `launch_kernel()`, [synchronize()](file:///workspace/atlas/crates/atlas-py/src/lib.rs#91-99) |

### Abstraction 3: WeightLoader (SBIO IORouter for storage)

| | |
|---|---|
| **GSI** | 2 (SafetensorsLoader for production, MockWeightLoader for tests) |
| **Variation points** | File format (safetensors vs GGUF), sharding strategy, quantization unpacking |
| **Common behavior** | `load(model_dir) → WeightStore` |

### Abstraction 4: CommBackend (SBIO IORouter for inter-GPU communication)

| | |
|---|---|
| **GSI** | 2 (NcclBackend for production, MockCommBackend for tests + single-GPU) |
| **Variation points** | Transport (NCCL, RCCL for AMD, custom TCP), topology awareness |
| **Common behavior** | `all_reduce()`, `all_gather()`, `all_to_all()`, `broadcast()`, `barrier()` |

---

## SBIO I/O Table

| Type | Location | Abstract Call | Production Impl | Mock Impl |
|---|---|---|---|---|
| GPU alloc | `kv_cache.rs`, `weights.rs` | `gpu.alloc(bytes) → DevicePtr` | `cuMemAlloc_v2` | `HashMap<DevicePtr, Vec<u8>>` |
| GPU copy | `weights.rs` | `gpu.copy_h2d(src, dst, bytes)` | `cuMemcpyHtoD_v2` | `memcpy into HashMap` |
| GPU launch | `qwen3.rs` | `gpu.launch(func, grid, block, args, stream)` | `cuLaunchKernel` | `record call + noop` |
| GPU sync | `engine.rs` | `gpu.synchronize(stream)` | `cuStreamSynchronize` | `noop` |
| AllReduce | `qwen3.rs` (after TP GEMM) | `comm.all_reduce(buf, op, stream)` | `ncclAllReduce` | `noop (single-GPU)` |
| AllToAll | `qwen3.rs` (MoE routing) | `comm.all_to_all(send, recv, stream)` | `ncclGroupStart/Send/Recv` | `memcpy (single-GPU)` |
| Filesystem | `weights.rs` | `loader.load(path) → WeightStore` | `mmap + safetensors` | `in-memory HashMap` |
| Network | `api.rs` | `axum::Router` handler | HTTP/SSE via tokio | N/A (test with axum test client) |

---

## Crate Structure

```
crates/
├── spark-model/          # Model abstraction + Qwen3 implementation
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs            # pub mod declarations
│       ├── traits.rs         # Model, Layer traits (SDD Step 3)
│       ├── qwen3.rs          # Qwen3HybridModel (SDD Step 4)
│       ├── parallel.rs       # ParallelConfig, weight sharding logic
│       └── weight_map.rs     # HF name → layer index mapping (SSOT from ModelConfig)
│
├── spark-runtime/        # Execution engine, GPU backend, weights, KV cache
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── gpu.rs            # GpuBackend trait + AtlasCudaBackend (SBIO Step 3-4)
│       ├── engine.rs         # InferenceEngine: prefill + decode loop
│       ├── weights.rs        # WeightLoader trait + SafetensorsLoader (SBIO)
│       ├── kv_cache.rs       # PagedKvCache: block allocator
│       ├── sampler.rs        # Greedy argmax (Phase 1)
│       └── ptx.rs            # PTX embedding (moved from atlas-py, SSOT)
│
├── spark-comm/           # Inter-GPU communication (NCCL)
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── traits.rs         # CommBackend trait (SDD Step 3)
│       ├── nccl.rs           # NcclBackend implementation (SDD Step 4)
│       └── single.rs         # SingleGpuBackend (noop, for single-Spark mode)
│
├── spark-server/         # HTTP server binary
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs           # Entry point, CLI args (PCND: all explicit)
│       ├── api.rs            # /v1/chat/completions, /v1/models, /health
│       ├── openai.rs         # Request/response serde types
│       └── tokenizer.rs      # HuggingFace tokenizers wrapper
```

> [!NOTE]
> `atlas-py` is **not modified or deleted** — it continues to work for anyone using the Python bindings. But it is no longer on the critical path for Atlas Spark. The PTX sources are embedded directly in `spark-runtime/src/ptx.rs` using the same `include_str!` pattern.

---

## CLI Specification

### `spark-server/src/cli.rs` — [NEW]

Full CLI using `clap` derive macros. **PCND: every parameter is explicit — no hidden defaults in the struct.** Defaults are only allowed when documented with a rationale comment for tests.

```rust
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

/// Atlas Spark — Pure Rust LLM inference server for DGX Spark.
#[derive(Parser, Debug)]
#[command(name = "atlas-spark", version, about, long_about = None)]
pub struct Cli {
    /// Subcommand to execute.
    #[command(subcommand)]
    pub command: Command,
}

#[derive(clap::Subcommand, Debug)]
pub enum Command {
    /// Start the inference server.
    Serve(ServeArgs),
    /// Validate model weights and config without starting the server.
    Validate(ValidateArgs),
    /// Run a single completion from stdin/args (no server).
    Complete(CompleteArgs),
    /// Print detected GPU info and exit.
    GpuInfo,
}

// ── Serve ──────────────────────────────────────────────────

#[derive(Parser, Debug)]
pub struct ServeArgs {
    // ── Model ──
    /// Path to HuggingFace model directory (must contain config.json
    /// and safetensors files).
    #[arg(long, env = "ATLAS_MODEL")]
    pub model: PathBuf,

    /// Path to tokenizer.json (defaults to <model>/tokenizer.json).
    #[arg(long, env = "ATLAS_TOKENIZER")]
    pub tokenizer: Option<PathBuf>,

    /// Quantization format of the model weights.
    #[arg(long, value_enum, env = "ATLAS_QUANT")]
    pub quantization: Quantization,

    // ── Server ──
    /// Host address to bind the HTTP server to.
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,

    /// Port to bind the HTTP server to.
    #[arg(long, env = "ATLAS_PORT")]
    pub port: u16,

    // ── Parallelism ──
    /// GPU device ordinal (0-based).
    #[arg(long, env = "ATLAS_GPU_ORDINAL")]
    pub gpu_ordinal: usize,

    /// Tensor parallelism degree (number of GPUs to shard weight
    /// matrices across). Set to 1 for single-GPU.
    #[arg(long, env = "ATLAS_TP_SIZE")]
    pub tp_size: usize,

    /// Expert parallelism degree (number of GPUs to distribute MoE
    /// experts across). Set to 1 for single-GPU.
    #[arg(long, env = "ATLAS_EP_SIZE")]
    pub ep_size: usize,

    /// This node's rank in the distributed group (0-based).
    #[arg(long, env = "ATLAS_RANK")]
    pub rank: usize,

    /// NCCL master address (IP of rank 0 node).
    #[arg(long, env = "ATLAS_MASTER_ADDR")]
    pub master_addr: Option<String>,

    /// NCCL master port.
    #[arg(long, env = "ATLAS_MASTER_PORT")]
    pub master_port: Option<u16>,

    // ── Speculative Decoding ──
    /// Speculative decoding method.
    #[arg(long, value_enum, default_value = "none")]
    pub speculative_method: SpeculativeMethod,

    /// Number of speculative tokens to draft per step.
    #[arg(long)]
    pub num_speculative_tokens: Option<u32>,

    /// Path to EAGLE-3 draft head weights (required when
    /// --speculative-method=eagle3).
    #[arg(long)]
    pub draft_head_path: Option<PathBuf>,

    // ── KV Cache ──
    /// Maximum sequence length (prompt + generation).
    #[arg(long, env = "ATLAS_MAX_SEQ_LEN")]
    pub max_seq_len: usize,

    /// KV cache block size (tokens per block).
    #[arg(long, env = "ATLAS_BLOCK_SIZE")]
    pub block_size: usize,

    /// KV cache data type.
    #[arg(long, value_enum, default_value = "fp8")]
    pub kv_cache_dtype: KvCacheDtype,

    /// Fraction of GPU memory to allocate for KV cache (0.0-1.0).
    #[arg(long, env = "ATLAS_GPU_MEMORY_UTIL")]
    pub gpu_memory_utilization: f32,

    // ── Performance ──
    /// Maximum number of concurrent sequences.
    #[arg(long, env = "ATLAS_MAX_NUM_SEQS")]
    pub max_num_seqs: usize,

    /// Enable CUDA graph capture for the decode path.
    #[arg(long)]
    pub enable_cuda_graphs: bool,

    /// Enable chunked prefill (split long prompts into chunks).
    #[arg(long)]
    pub enable_chunked_prefill: bool,

    /// Chunk size for chunked prefill (tokens per chunk).
    #[arg(long)]
    pub prefill_chunk_size: Option<usize>,

    /// Enable prefix caching (radix tree KV reuse).
    #[arg(long)]
    pub enable_prefix_caching: bool,

    // ── Logging ──
    /// Log level (trace, debug, info, warn, error).
    #[arg(long, default_value = "info", env = "ATLAS_LOG_LEVEL")]
    pub log_level: String,
}

// ── Validate ───────────────────────────────────────────────

#[derive(Parser, Debug)]
pub struct ValidateArgs {
    /// Path to HuggingFace model directory.
    #[arg(long)]
    pub model: PathBuf,

    /// Quantization format.
    #[arg(long, value_enum)]
    pub quantization: Quantization,
}

// ── Complete (offline, no server) ──────────────────────────

#[derive(Parser, Debug)]
pub struct CompleteArgs {
    /// Path to HuggingFace model directory.
    #[arg(long)]
    pub model: PathBuf,

    /// Quantization format.
    #[arg(long, value_enum)]
    pub quantization: Quantization,

    /// GPU ordinal.
    #[arg(long)]
    pub gpu_ordinal: usize,

    /// Prompt text (or "-" for stdin).
    #[arg(long)]
    pub prompt: String,

    /// Maximum tokens to generate.
    #[arg(long)]
    pub max_tokens: u32,

    /// Sampling temperature (0.0 = greedy).
    #[arg(long, default_value = "0.0")]
    pub temperature: f32,
}

// ── Enums ──────────────────────────────────────────────────

#[derive(ValueEnum, Clone, Debug)]
pub enum Quantization {
    Bf16,
    Fp8,
    Nvfp4,
    W4a16,
}

#[derive(ValueEnum, Clone, Debug)]
pub enum SpeculativeMethod {
    /// No speculative decoding.
    None,
    /// Multi-Token Prediction (built-in MTP heads).
    Mtp,
    /// EAGLE-3 (requires --draft-head-path).
    Eagle3,
}

#[derive(ValueEnum, Clone, Debug)]
pub enum KvCacheDtype {
    Bf16,
    Fp8,
}
```

### Example usage

```bash
# Single Spark — basic serve
atlas-spark serve \
  --model /models/qwen3-next-80b \
  --quantization nvfp4 \
  --port 8888 \
  --gpu-ordinal 0 \
  --tp-size 1 --ep-size 1 --rank 0 \
  --max-seq-len 4096 --block-size 16 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 128 \
  --kv-cache-dtype fp8

# Multi-Spark (2 nodes) — run on each node
# Node 0:
atlas-spark serve \
  --model /models/qwen3-next-80b \
  --quantization nvfp4 \
  --port 8888 \
  --gpu-ordinal 0 \
  --tp-size 2 --ep-size 2 --rank 0 \
  --master-addr 10.0.0.1 --master-port 29500 \
  --max-seq-len 4096 --block-size 16 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 128

# Node 1:
atlas-spark serve \
  --model /models/qwen3-next-80b \
  --quantization nvfp4 \
  --port 8888 \
  --gpu-ordinal 0 \
  --tp-size 2 --ep-size 2 --rank 1 \
  --master-addr 10.0.0.1 --master-port 29500 \
  --max-seq-len 4096 --block-size 16 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 128

# With MTP speculative decoding
atlas-spark serve \
  --model /models/qwen3-next-80b \
  --quantization nvfp4 \
  --port 8888 \
  --gpu-ordinal 0 --tp-size 1 --ep-size 1 --rank 0 \
  --max-seq-len 4096 --block-size 16 \
  --gpu-memory-utilization 0.90 --max-num-seqs 128 \
  --speculative-method mtp --num-speculative-tokens 2

# With EAGLE-3 speculative decoding
atlas-spark serve \
  --model /models/qwen3-next-80b \
  --quantization nvfp4 \
  --port 8888 \
  --gpu-ordinal 0 --tp-size 1 --ep-size 1 --rank 0 \
  --max-seq-len 4096 --block-size 16 \
  --gpu-memory-utilization 0.90 --max-num-seqs 128 \
  --speculative-method eagle3 --num-speculative-tokens 5 \
  --draft-head-path /models/eagle3-qwen3-head

# Quick offline generation (no server)
atlas-spark complete \
  --model /models/qwen3-next-80b \
  --quantization nvfp4 \
  --gpu-ordinal 0 \
  --prompt "Hello, world!" \
  --max-tokens 64 --temperature 0.0

# Validate model before deploying
atlas-spark validate \
  --model /models/qwen3-next-80b \
  --quantization nvfp4

# GPU info
atlas-spark gpu-info
```

### CLI Validation Rules (fail fast — PCND)

```rust
impl ServeArgs {
    pub fn validate(&self) -> Result<()> {
        // PCND: fail explicitly, never silently assume
        if !self.model.exists() {
            bail!("--model path does not exist: {}", self.model.display());
        }
        if self.tp_size == 0 || self.ep_size == 0 {
            bail!("--tp-size and --ep-size must be >= 1");
        }
        if self.tp_size > 1 || self.ep_size > 1 {
            let addr = self.master_addr.as_ref()
                .ok_or_else(|| anyhow!("--master-addr required for multi-GPU"))?;
            let port = self.master_port
                .ok_or_else(|| anyhow!("--master-port required for multi-GPU"))?;
            if addr.is_empty() { bail!("--master-addr cannot be empty"); }
            if port == 0 { bail!("--master-port cannot be 0"); }
        }
        if self.speculative_method == SpeculativeMethod::Eagle3 {
            let path = self.draft_head_path.as_ref()
                .ok_or_else(|| anyhow!("--draft-head-path required for eagle3"))?;
            if !path.exists() { bail!("--draft-head-path does not exist: {}", path.display()); }
        }
        if matches!(self.speculative_method, SpeculativeMethod::Mtp | SpeculativeMethod::Eagle3)
            && self.num_speculative_tokens.is_none()
        {
            bail!("--num-speculative-tokens required when speculative decoding is enabled");
        }
        if self.gpu_memory_utilization <= 0.0 || self.gpu_memory_utilization > 1.0 {
            bail!("--gpu-memory-utilization must be in (0.0, 1.0]");
        }
        Ok(())
    }
}
```

### CLI Test Strategy

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn test_minimal_serve() {
        let args = Cli::try_parse_from([
            "atlas-spark", "serve",
            "--model", "/tmp/model",
            "--quantization", "nvfp4",
            "--port", "8888",
            "--gpu-ordinal", "0",
            "--tp-size", "1", "--ep-size", "1", "--rank", "0",
            "--max-seq-len", "4096", "--block-size", "16",
            "--gpu-memory-utilization", "0.9",
            "--max-num-seqs", "128",
        ]);
        assert!(args.is_ok());
    }

    #[test]
    fn test_multi_gpu_requires_master() {
        let cli = Cli::try_parse_from([
            "atlas-spark", "serve",
            "--model", "/tmp/model",
            "--quantization", "nvfp4",
            "--port", "8888",
            "--gpu-ordinal", "0",
            "--tp-size", "2", "--ep-size", "2", "--rank", "0",
            "--max-seq-len", "4096", "--block-size", "16",
            "--gpu-memory-utilization", "0.9",
            "--max-num-seqs", "128",
        ]).unwrap();
        // Should fail validation — no master-addr
        if let Command::Serve(args) = cli.command {
            assert!(args.validate().is_err());
        }
    }

    #[test]
    fn test_eagle3_requires_draft_path() {
        let cli = Cli::try_parse_from([
            "atlas-spark", "serve",
            "--model", "/tmp/model",
            "--quantization", "nvfp4",
            "--port", "8888",
            "--gpu-ordinal", "0",
            "--tp-size", "1", "--ep-size", "1", "--rank", "0",
            "--max-seq-len", "4096", "--block-size", "16",
            "--gpu-memory-utilization", "0.9",
            "--max-num-seqs", "128",
            "--speculative-method", "eagle3",
            "--num-speculative-tokens", "5",
        ]).unwrap();
        if let Command::Serve(args) = cli.command {
            assert!(args.validate().is_err());
        }
    }

    #[test]
    fn test_complete_subcommand() {
        let args = Cli::try_parse_from([
            "atlas-spark", "complete",
            "--model", "/tmp/model",
            "--quantization", "nvfp4",
            "--gpu-ordinal", "0",
            "--prompt", "Hello!",
            "--max-tokens", "32",
        ]);
        assert!(args.is_ok());
    }

    #[test]
    fn test_gpu_info_subcommand() {
        let args = Cli::try_parse_from(["atlas-spark", "gpu-info"]);
        assert!(args.is_ok());
    }
}
```

## Key Traits (SDD Step 3)

### `spark-model/src/traits.rs`

```rust
use atlas_core::config::ModelConfig;

/// Forward-pass state for a single sequence.
pub struct SequenceState {
    /// Block table: indices into the KV cache pool
    pub block_table: Vec<u32>,
    /// Block table GPU pointer (uploaded once, updated on block allocation)
    pub block_table_ptr: u64,
    /// Current sequence length
    pub seq_len: u32,
}

/// A loaded model ready for inference.
///
/// Implementations encapsulate the full forward pass: embedding lookup,
/// layer iteration, final norm, and LM head projection.
pub trait Model: Send + Sync {
    /// Run prefill on all input tokens. Populates KV cache, returns logits.
    fn prefill(
        &self,
        tokens: &[u32],
        state: &mut SequenceState,
        stream: u64,
    ) -> Result<Vec<f32>>;

    /// Run one decode step. Appends to KV cache, returns logits.
    fn decode(
        &self,
        token: u32,
        state: &mut SequenceState,
        stream: u64,
    ) -> Result<Vec<f32>>;

    /// Vocabulary size.
    fn vocab_size(&self) -> usize;

    /// Model configuration.
    fn config(&self) -> &ModelConfig;
}

/// Layer types in a hybrid transformer (SDD variation point).
pub enum LayerKind {
    Attention,
    Ssm,
}
```

### `spark-runtime/src/gpu.rs`

```rust
/// Device pointer — opaque u64 wrapping a CUdeviceptr.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DevicePtr(pub u64);

/// SBIO IORouter for GPU operations.
///
/// All CUDA interactions flow through this trait. Business logic (model
/// forward pass, KV cache management) never calls cuLaunchKernel directly.
pub trait GpuBackend: Send + Sync {
    /// Allocate device memory.
    fn alloc(&self, bytes: usize) -> Result<DevicePtr>;

    /// Free device memory.
    fn free(&self, ptr: DevicePtr) -> Result<()>;

    /// Copy host → device.
    fn copy_h2d(&self, src: *const u8, dst: DevicePtr, bytes: usize) -> Result<()>;

    /// Copy device → host.
    fn copy_d2h(&self, src: DevicePtr, dst: *mut u8, bytes: usize) -> Result<()>;

    /// Launch a kernel on the given stream.
    fn launch(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        stream: u64,
        params: &mut [*mut std::ffi::c_void],
    ) -> Result<()>;

    /// Synchronize a stream.
    fn synchronize(&self, stream: u64) -> Result<()>;

    /// Get the default stream handle.
    fn default_stream(&self) -> u64;

    /// Look up a kernel function by module and name.
    fn kernel(&self, module: &str, func: &str) -> Result<KernelHandle>;
}
```

### `spark-comm/src/traits.rs`

```rust
/// SBIO IORouter for inter-GPU communication.
///
/// All collective operations flow through this trait. Business logic
/// (model forward pass) never calls ncclAllReduce directly.
pub trait CommBackend: Send + Sync {
    /// AllReduce: sum partial results across all ranks.
    /// Used after tensor-parallel GEMM (QKV proj, output proj, LM head).
    fn all_reduce(
        &self,
        buf: DevicePtr,
        count: usize,
        dtype: DType,
        op: ReduceOp,
        stream: u64,
    ) -> Result<()>;

    /// AllGather: gather sharded tensors from all ranks.
    /// Used to reconstruct full hidden state when needed.
    fn all_gather(
        &self,
        send_buf: DevicePtr,
        recv_buf: DevicePtr,
        count: usize,
        dtype: DType,
        stream: u64,
    ) -> Result<()>;

    /// AllToAll: route tokens to their assigned expert's rank.
    /// Used for MoE expert parallelism.
    fn all_to_all(
        &self,
        send_buf: DevicePtr,
        recv_buf: DevicePtr,
        send_counts: &[usize],
        recv_counts: &[usize],
        dtype: DType,
        stream: u64,
    ) -> Result<()>;

    /// Broadcast: send data from rank 0 to all ranks.
    /// Used for initial weight distribution and sampled tokens.
    fn broadcast(
        &self,
        buf: DevicePtr,
        count: usize,
        dtype: DType,
        root: usize,
        stream: u64,
    ) -> Result<()>;

    /// Barrier: synchronize all ranks.
    fn barrier(&self) -> Result<()>;

    /// This rank's index (0-based).
    fn rank(&self) -> usize;

    /// Total number of ranks.
    fn world_size(&self) -> usize;
}

pub enum ReduceOp {
    Sum,
    Max,
}
```

### `spark-model/src/parallel.rs`

```rust
/// Parallelism configuration — PCND: all values explicit, no defaults.
pub struct ParallelConfig {
    /// Tensor parallelism degree (shards weight matrices across GPUs).
    /// Each GPU holds hidden_size/tp_size columns of QKV, output proj.
    pub tp_size: usize,

    /// Expert parallelism degree (distributes MoE experts across GPUs).
    /// Each GPU holds num_experts/ep_size experts.
    pub ep_size: usize,

    /// This rank's position (0-based).
    pub rank: usize,

    /// Total world size (must equal tp_size for now; EP uses same group).
    pub world_size: usize,
}
```

### `spark-runtime/src/weights.rs`

```rust
/// A single weight tensor on the GPU.
pub struct WeightTensor {
    pub ptr: DevicePtr,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

/// Stored model weights, keyed by HuggingFace name.
pub struct WeightStore {
    weights: HashMap<String, WeightTensor>,
}

/// SBIO IORouter for weight loading (filesystem I/O).
pub trait WeightLoader {
    /// Load all weights from a model directory onto the GPU.
    fn load(
        &self,
        model_dir: &Path,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<WeightStore>;
}
```

---

## Qwen3 Forward Pass (SDD Step 4)

Each operation maps to an existing Atlas CUDA kernel. With multi-GPU, TP shards GEMMs and EP shards MoE experts:

| Step | Operation | Atlas Kernel | Multi-GPU |
|---|---|---|---|
| 1 | Embedding lookup | `cuMemcpyDtoD` | Replicated on all ranks |
| 2 | RMSNorm (input) | `rms_norm_bf16` | Local (no comm) |
| 3 | QKV projection | `w4a16_gemm_kernel` | **TP: column-parallel**, each rank does `hidden/tp_size` cols |
| 4 | RoPE | `rope_forward_fused` | Local (no comm) |
| 5 | Paged decode attn | `paged_decode_attn_fp8` | Local (each rank has its KV heads) |
| 6 | KV cache write | `reshape_and_cache_flash_fp8` | Local |
| 7 | Output projection | `w4a16_gemm_kernel` | **TP: row-parallel** → **`AllReduce`** |
| 8 | Residual add | fused with RMSNorm | Local (after AllReduce) |
| 9 | RMSNorm (post-attn) | `rms_norm_bf16` | Local |
| 10a | MoE gate routing | Router GEMM | Local → **`AllToAll`** (send tokens to expert owners) |
| 10b | MoE expert GEMM | `moe_w4a16_grouped_gemm` | **EP: each rank runs `experts/ep_size`** |
| 10c | MoE reduce | unpermute | **`AllToAll`** (return results to token owners) |
| 11 | Gated RMSNorm (SSM) | `gated_rms_norm_bf16` | Local |
| 12 | Conv1d update (SSM) | `causal_conv1d_update` | Local |
| 13 | GDR decode (SSM) | `gated_delta_rule_decode` | Local |
| 14 | Final RMSNorm | `rms_norm_bf16` | Local |
| 15 | LM head projection | `w4a16_gemm_kernel` | **TP: column-parallel** → **`AllGather`** |

---

## Dependencies

### Workspace Cargo.toml additions

```toml
# HTTP server
axum = { version = "0.8", features = ["json"] }
tokio = { version = "1", features = ["full"] }
tower-http = { version = "0.6", features = ["cors"] }

# Inter-GPU communication (NCCL via cudarc)
cudarc = { version = "0.19", features = ["driver", "nvrtc", "nccl"] }

# Model loading
safetensors = "0.5"
memmap2 = "0.9"

# Tokenizer
tokenizers = { version = "0.21" }

# CLI (PCND: explicit configuration)
clap = { version = "4", features = ["derive"] }

# OpenAI API types
uuid = { version = "1", features = ["v4"] }

# Logging
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
```

### Per-crate dependencies

| Crate | Depends on |
|---|---|
| `spark-model` | `atlas-core` (ModelConfig, DType), `spark-runtime` (GpuBackend, WeightStore, SequenceState), `spark-comm` (CommBackend) |
| `spark-runtime` | `atlas-core` (registry, stream), `atlas-attention`, `atlas-gemm`, `atlas-norm`, `atlas-activation`, `atlas-embed`, `atlas-ssm`, `atlas-quant`, `safetensors`, `memmap2`, `tracing` |
| `spark-comm` | `atlas-core` (DType, DevicePtr), `cudarc` (NCCL bindings), `tracing` |
| `spark-server` | `spark-model`, `spark-runtime`, `spark-comm`, `axum`, `tokio`, `tokenizers`, `clap`, `uuid`, `tracing`, `tracing-subscriber` |

---

## What Doesn't Change

| Component | Status |
|---|---|
| `atlas-core` | ✅ Untouched (registry, stream, tensor, config all reused as-is) |
| `atlas-attention` | ✅ Untouched (flash attn, paged decode, KV cache kernels) |
| `atlas-gemm` | ✅ Untouched (dense, W4A16, MoE GEMM) |
| `atlas-ssm` | ✅ Untouched (conv1d, gated delta rule) |
| `atlas-norm` | ✅ Untouched (RMSNorm, gated RMSNorm) |
| `atlas-activation` | ✅ Untouched (SiLU×Mul) |
| `atlas-embed` | ✅ Untouched (RoPE) |
| `atlas-quant` | ✅ Untouched (NVFP4, FP8) |
| `atlas-reduce` | ✅ Untouched |
| `atlas-py` | ⏸️ Deprecated — still builds, no longer on critical path |
| `cuda_kernels/` | ✅ Untouched — all 17 .cu files |

---

## Optimization Strategy: Matching & Beating vLLM

### Overview

vLLM's performance comes from four major optimizations. Here's how Atlas Spark addresses each:

| vLLM Optimization | How vLLM Does It | Atlas Spark Approach | Phase |
|---|---|---|---|
| **Kernel fusion** | `torch.compile` inductor backend fuses adjacent ops | **Custom fused CUDA kernels** — no compiler limitations | 1 (basic) → 2 (advanced) |
| **CUDA graphs** | Capture decode step, replay with zero launch overhead | **`CudaGraphCapture` in engine.rs** — capture full decode path | 2 |
| **Continuous batching** | Dynamic batch scheduling across concurrent requests | **Rust scheduler** — no Python GIL, lock-free request queue | 2 |
| **Speculative decoding (MTP)** | Draft 2 tokens, verify in one forward pass | **MTP integration** — orthogonal to kernel work | 3 |
| **PagedAttention** | Paged KV cache to avoid memory fragmentation | **Already in plan** — `PagedKvCache` in `kv_cache.rs` | 1 |

### Kernel Fusion (Phase 1 + Phase 2)

In vLLM, `torch.compile`'s inductor fuses these patterns (from PROGRESS.md):
- `residual_add + rms_norm` → single fused kernel
- `rms_norm + rope` → single fused kernel  
- `silu × gate + down_proj` → single fused kernel

**In Atlas Spark, we do this better.** torch.compile can only fuse what its graph compiler discovers. We can write **hand-tuned fused kernels** for exactly the patterns we need:

| Fusion | vLLM (torch.compile) | Atlas Spark | New Kernel? |
|---|---|---|---|
| `residual_add + rms_norm` | ✅ Inductor fuses | ✅ **Fused CUDA kernel** | **Phase 2**: Write `fused_residual_rms_norm.cu` |
| `rms_norm → qkv_proj` | ❌ Can't fuse norm + GEMM | ✅ **Possible** — norm output stays in registers | **Phase 2**: Investigate register-level fusion |
| `silu × gate` | ✅ Inductor fuses | ✅ **Already have** `fused_silu_mul` kernel | No — existing kernel |
| [rope](file:///workspace/atlas/crates/atlas-core/src/config.rs#58-61) | ✅ Inductor keeps as one op | ✅ **Already have** fused RoPE kernel | No — existing kernel |
| MoE permute→GEMM→silu→GEMM→unpermute | ✅ Marlin handles end-to-end | ✅ **Already have** `moe_forward_w4a16` | No — existing kernel |

**Phase 1 approach**: Launch each Atlas kernel individually. This is already faster than vLLM with 7 patches (which got 36.5 tok/s), because our individual kernels are 4-18× faster than vLLM's.

**Phase 2 approach**: Write 2-3 fused kernels for the hot path (`fused_residual_rms_norm`, potentially `fused_norm_rope`). Since norms + activations are ~5-10% of compute time, this targets 2-5% additional throughput.

### CUDA Graph Capture (Phase 2)

CUDA graphs eliminate **all CPU-side kernel launch overhead** by recording a sequence of GPU operations and replaying them as a single unit.

```rust
// In engine.rs — Phase 2 addition
pub struct CudaGraphExecutor {
    /// Captured graph for the decode path (fixed shapes)
    decode_graph: Option<CudaGraph>,
    /// Pre-allocated input/output buffers (CUDA graphs need fixed addresses)
    pinned_buffers: PinnedBuffers,
}

impl CudaGraphExecutor {
    /// Capture the decode forward pass as a CUDA graph.
    /// Called once after model load with dummy inputs.
    fn capture_decode_graph(&mut self, model: &dyn Model, stream: u64) -> Result<()>;

    /// Replay the captured graph (zero launch overhead).
    fn replay_decode(&self, token: u32, state: &mut SequenceState) -> Result<Vec<f32>>;
}
```

**Why this works especially well for us**: vLLM uses `CUDA_GRAPH_MODE=PIECEWISE` because torch.compile graphs create fragmented capture regions. Atlas Spark can capture the **entire decode path** as one graph — from embedding lookup through LM head — because we control every kernel launch with no Python/PyTorch in between.

| | vLLM CUDA Graphs | Atlas Spark CUDA Graphs |
|---|---|---|
| Capture granularity | PIECEWISE (fragmented) | **FULL** (entire decode step) |
| Python overhead during replay | ~1ms (scheduler + model runner) | **0ms** (pure Rust replay) |
| Graph switch cost | High (multiple graph segments) | **Low** (single graph) |

### What We Get For Free (No Extra Work)

| Optimization | Why It's Free |
|---|---|
| **Zero Python overhead** | Rust binary — no GIL, no PyTorch dispatch, no Python allocator |
| **No torch.compile warmup** | No JIT compilation phase — PTX is pre-compiled at build time |
| **Unified memory awareness** | GB10's LPDDR5X means no CPU↔GPU copy for weight loading (mmap directly) |
| **Static memory layout** | Pre-allocate all buffers at init — no runtime allocation during inference |
| **Minimal binary** | ~50MB static binary vs ~2GB Docker image with Python + PyTorch + vLLM |

### Performance Projection by Phase

| Phase | Optimizations Active | Expected tok/s | vs vLLM 40.5 |
|---|---|---|---|
| **Phase 1** | All Atlas kernels, no fusion, no CUDA graphs | **42-45** | +4-11% |
| **Phase 2** | + Fused kernels + CUDA graphs + continuous batching | **50-55** | +23-36% |
| **Phase 3** | + MTP speculative decoding (2 tok/step) | **80-100** | +97-147% |
| vLLM v22 ceiling | FlashInfer + Marlin + MTP | 59.9 | baseline |

> [!NOTE]
> Phase 1 already beats vLLM+Atlas (40.5 tok/s) because we run **all 17 kernels** instead of just 3, and eliminate Python overhead. The Phase 2 fusions provide incremental gains. The real leap is Phase 3 MTP, which doubles effective throughput.

---

## Verification Plan

### Automated Tests

```bash
# spark-model: weight mapping derives correctly from ModelConfig
cargo test -p spark-model

# spark-runtime: KV cache allocation, sampler, GPU mock
cargo test -p spark-runtime

# spark-server: OpenAI API serialization round-trip
cargo test -p spark-server
```

All tests use `MockGpuBackend` — no GPU required for unit tests (SBIO).

### Integration (requires GPU + model weights)

```bash
# Full forward pass correctness
cargo test -p spark-runtime --test integration -- --ignored

# End-to-end server test
cargo run -p spark-server -- \
  --model /path/to/qwen3 \
  --port 8888 \
  --gpu-ordinal 0 \
  --max-seq-len 4096 \
  --block-size 16

curl http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3","messages":[{"role":"user","content":"Hello!"}],"max_tokens":32,"stream":true}'
```

### Performance Targets

| Metric | vLLM + Atlas (current) | Atlas Spark (target) |
|---|---|---|
| Decode tok/s | 40.5 | 50-55 |
| Time to first token | ~200ms | ~50ms |
| Peak memory | ~90GB | ~60GB |
