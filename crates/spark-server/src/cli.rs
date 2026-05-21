// SPDX-License-Identifier: AGPL-3.0-only

//! CLI argument parsing.

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "spark", about = "Atlas Spark — pure Rust LLM inference server")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(clap::Subcommand, Debug)]
pub enum Command {
    /// Start the inference server.
    Serve(ServeArgs),
}

/// Arguments for the `serve` subcommand.
#[derive(Parser, Debug)]
pub struct ServeArgs {
    /// HuggingFace model ID (e.g. "nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4")
    /// or a local directory path containing config.json.
    #[arg(value_name = "MODEL", required_unless_present = "model_from_path")]
    pub model: Option<String>,

    /// Load model directly from this filesystem path (skips HF cache resolution).
    #[arg(long, value_name = "PATH")]
    pub model_from_path: Option<PathBuf>,

    /// Override model name shown in /v1/models and API responses.
    /// Defaults to the positional MODEL argument, then config.json _name_or_path.
    #[arg(long, alias = "served-model-name", value_name = "NAME")]
    pub model_name: Option<String>,

    /// Override HuggingFace cache directory
    /// (default: $HF_HUB_CACHE, $HF_HOME/hub, or ~/.cache/huggingface/hub).
    #[arg(long, value_name = "DIR")]
    pub cache_dir: Option<PathBuf>,

    /// HTTP port.
    #[arg(long, default_value_t = 8888)]
    pub port: u16,

    /// GPU ordinal.
    #[arg(long, default_value_t = 0)]
    pub gpu_ordinal: usize,

    /// Maximum sequence length.
    #[arg(long, default_value_t = 32768)]
    pub max_seq_len: usize,

    /// KV cache block size (tokens per block).
    #[arg(long, default_value_t = 16)]
    pub block_size: usize,

    /// KV cache dtype (fp8, bf16, or nvfp4).
    /// Default: fp8. NVFP4 uses less memory but may lose coherence at long context
    /// without --kv-high-precision-layers. FP8 is the safe default.
    #[arg(long, default_value = "fp8")]
    pub kv_cache_dtype: String,

    /// Boundary attention layers to keep at BF16 KV cache precision (first N + last N).
    /// Protects attention sink tokens (early layers) and output quality (final layers)
    /// from quantization error while saving memory on middle layers.
    /// Accepts: number, "auto" (=2, recommended), "max"/"all" (all BF16).
    /// Default: 0 (all layers use --kv-cache-dtype).
    #[arg(long, default_value = "0")]
    pub kv_high_precision_layers: String,

    /// GPU memory utilization (0.0-1.0).
    #[arg(long, default_value_t = 0.90)]
    pub gpu_memory_utilization: f64,

    /// Maximum concurrent sequences.
    #[arg(long, default_value_t = 128)]
    pub max_num_seqs: usize,

    /// Global kill-switch for chain-of-thought / reasoning output.
    /// When set, the server forces thinking OFF regardless of what the
    /// client requests (reasoning_effort, thinking.budget_tokens, etc.)
    /// or what MODEL.toml declares as the default. Precedence (highest
    /// wins): this flag → request body → MODEL.toml `[behavior]`.thinking_default.
    ///
    /// Harry Potter alias: `--stupify` (stuns the model's inner monologue).
    #[arg(long, visible_alias = "stupify", default_value_t = false)]
    pub disable_thinking: bool,

    /// Override MODEL.toml's `[behavior].max_thinking_budget` (tokens).
    /// Sets the per-request ceiling for thinking-block length. Per-request
    /// `thinking.budget_tokens` (or `reasoning_effort`) still wins below
    /// this ceiling; the (max_tokens * 9 / 10) safety cap is always enforced.
    #[arg(long)]
    pub max_thinking_budget: Option<u32>,

    /// Default chat template kwargs applied when the client sends no
    /// thinking parameters (no `reasoning.effort`, `chat_template_kwargs`,
    /// or `enable_thinking` in the request body). A JSON object with
    /// optional keys: `enable_thinking` (bool), `thinking_budget` (u32).
    ///
    /// Precedence (highest wins): request body → this flag → MODEL.toml.
    /// Example: `--default-chat-template-kwargs '{"enable_thinking":true}'`
    #[arg(long, value_name = "JSON")]
    pub default_chat_template_kwargs: Option<String>,

    /// Currently slower than regular decode for hybrid SSM models.
    #[arg(long, default_value_t = false)]
    pub speculative: bool,

    /// Enable self-speculative decoding: draft via layer-skipping (no MTP weights needed).
    /// Skips SSM layers during drafting for cheap predictions, then verifies with full model.
    #[arg(long, default_value_t = false)]
    pub self_speculative: bool,

    /// Enable N-gram speculative decoding: CPU-side pattern matching proposer
    /// with CUDA-graphed K=2 verification. No extra weights needed.
    #[arg(long, default_value_t = false)]
    pub ngram_speculative: bool,

    /// Enable DFlash block-diffusion speculative decoding (Z Lab,
    /// arXiv 2602.06036). Pairs the target with a small Qwen3-architecture
    /// drafter (e.g. `z-lab/Qwen3.6-35B-A3B-DFlash`) that emits γ tokens
    /// per step via bidirectional in-block attention conditioned on captured
    /// target hidden states. Mutually exclusive with `--speculative`.
    #[arg(long, default_value_t = false, conflicts_with = "speculative")]
    pub dflash: bool,

    /// HuggingFace id (or local path) of the DFlash drafter checkpoint.
    /// When `--dflash` is set without `--draft-model`, the value falls
    /// through from the target's MODEL.toml `[dflash].draft_model` field.
    #[arg(long)]
    pub draft_model: Option<String>,

    /// DFlash block size γ (parallel draft tokens per step). Defaults to
    /// the drafter's `block_size` from `config.json` (16 for the published
    /// Qwen3.6-DFlash drafters); override only for ablation. Higher γ
    /// increases per-step verify cost but raises peak speedup.
    #[arg(long, default_value_t = 16)]
    pub dflash_gamma: usize,

    /// DFlash drafter sliding-window size for long context. The drafter
    /// runs full-prefix attention by default; at Atlas's typical 16K
    /// `--max-seq-len`, drafter attention dominates per-step cost. The
    /// upstream sglang / vLLM default is 4096. Set to 0 to disable
    /// (full attention).
    #[arg(long, default_value_t = 4096)]
    pub dflash_window_size: usize,

    /// Number of draft tokens per speculative step (1=K=2, 2=K=3, 3=K=4 verify).
    /// Higher K verifies more drafts per step. Uses WY-chunkwise GDN kernels.
    #[arg(long, default_value_t = 1)]
    pub num_drafts: usize,

    /// Maximum concurrent sequences batched into one GPU decode step.
    #[arg(long, default_value_t = 8)]
    pub max_batch_size: usize,

    /// MTP head weight precision: bf16 (highest accuracy, most memory —
    /// the default), fp8 (balanced but slower due to D2H sync in MoE),
    /// nvfp4 (fastest — uses fused device-side expert dispatch; opt in
    /// explicitly when throughput matters more than accuracy).
    #[arg(long, default_value = "bf16")]
    pub mtp_quantization: String,

    /// MTP draft vocabulary size. Limits the LM head GEMV to the first N
    /// token IDs, reducing propose latency. BPE tokenizers place frequent
    /// tokens at low IDs — 100K covers >99% of English outputs while
    /// cutting propose time by 37% (2.15ms → 1.35ms) with zero acceptance
    /// loss. Set to 0 to use full vocabulary.
    #[arg(long, default_value_t = 100000)]
    pub mtp_vocab: u32,

    /// Enable prefix caching via radix tree (RadixAttention).
    /// Caches KV blocks for recurring prompt prefixes. For SSM models,
    /// KV is recomputed when no SSM snapshot exists (safe but no TTFT speedup
    /// without Marconi snapshots). Block table reuse still avoids allocation.
    #[arg(long, default_value_t = false, num_args = 0..=1, default_missing_value = "true")]
    pub enable_prefix_caching: bool,

    /// Dump every /v1/chat/completions, /v1/responses, and
    /// /v1/messages (Anthropic) request — plus the corresponding
    /// response (non-streaming) or aggregated stream — as JSONL to a
    /// file. Intended for extracting the exact system prompt and tool
    /// schema a client (opencode, Claude Code, etc.) is sending, and
    /// for replaying failure cases in fixtures.
    ///
    /// With no value: a temp file is created under $TMPDIR and its
    /// path is logged at INFO on startup. With a PATH: appends (never
    /// truncates) to that file. Each line is one JSON object:
    ///   `{ "ts": "<iso8601>", "endpoint": "...", "kind": "request"|"response",`
    ///     "seq": N, "body": { ... } }
    /// so entries can be grouped by `seq` to reconstruct pairs.
    #[arg(long, num_args = 0..=1, default_missing_value = "<auto>", value_name = "PATH")]
    pub dump: Option<String>,

    /// Scheduling policy: fifo (default) or slai (SLO-aware).
    /// SLAI prioritizes decode for sequences nearing TBT deadline
    /// and orders prefills shortest-prompt-first.
    #[arg(long, default_value = "fifo")]
    pub scheduling_policy: String,

    /// TBT deadline in milliseconds for SLAI scheduling policy.
    /// Sequences approaching this deadline trigger decode-first priority.
    #[arg(long, default_value_t = 100)]
    pub tbt_deadline_ms: u64,

    /// Maximum tokens to prefill per scheduler iteration (chunked prefill).
    /// Long prompts are split into chunks of this size, interleaved with
    /// decode steps for active sequences. Set to 0 to disable chunking
    /// (process entire prompt in one shot, legacy behavior).
    /// Chunked prefill: split long prompts into chunks, interleaved with
    /// decode steps. 8192 default halves chunk count vs 4096, giving ~11%
    /// TTFT improvement at 32K with no decode regression on DGX Spark.
    /// Set to 0 to disable (process entire prompt at once).
    #[arg(long, default_value_t = 8192)]
    pub max_prefill_tokens: usize,

    /// Minimum free GPU memory (in MB) to keep as a safety margin during
    /// model loading. If free memory drops below this threshold after any
    /// shard, loading is aborted to prevent system OOM. Default 4096 MB
    /// accounts for CUDA context, NCCL buffers, and allocator overhead.
    #[arg(long, default_value_t = 4096)]
    pub oom_guard_mb: usize,

    // ── Parallelism ──
    /// Global rank (0=head, 1=worker, …). Only used when --world-size > 1.
    #[arg(long, default_value_t = 0)]
    pub rank: usize,

    /// Total physical ranks across all parallelism dims. Set to 2 for two-node
    /// deployment. Must satisfy `world_size == tp_size × ep_size` (orthogonal
    /// mesh) or `world_size == tp_size == ep_size` (overlapping groups on the
    /// same physical ranks — used for 2-GPU TP+EP composition).
    #[arg(long, default_value_t = 1)]
    pub world_size: usize,

    /// Tensor-parallel dimension. Splits attention/MLP weights column- and
    /// row-parallel across `tp_size` ranks. 1 = no TP. Composes with EP:
    /// MoE expert weights stay EP-sharded; attention/MLP get TP-sharded.
    #[arg(long, default_value_t = 1)]
    pub tp_size: usize,

    /// Expert-parallel dimension. Splits MoE expert weights across `ep_size`
    /// ranks. 1 = no EP. Default of 1 keeps single-rank semantics.
    #[arg(long, default_value_t = 1)]
    pub ep_size: usize,

    /// NCCL bootstrap address (IP of rank 0 node).
    #[arg(long, default_value = "127.0.0.1")]
    pub master_addr: String,

    /// NCCL bootstrap port.
    #[arg(long, default_value_t = 29500)]
    pub master_port: u16,

    /// Tool call parser format. Enables OpenAI-compatible tool calling.
    /// Supported: "hermes" (Qwen3/3.5 JSON format), "qwen3_coder" (Nemotron-H XML format).
    /// When set, tool definitions in requests are injected into the system
    /// prompt and model output is parsed for tool_call tags.
    #[arg(long, value_name = "FORMAT")]
    pub tool_call_parser: Option<String>,

    /// Maximum output tokens per tool-calling request. Caps max_tokens from the
    /// client when tools are active to prevent unbounded generation if the model
    /// doesn't emit a </tool_call> stop token. Must be high enough for Write
    /// tool calls with large file content. Default 8192.
    #[arg(long, default_value_t = 8192)]
    pub tool_max_tokens: usize,

    /// Number of SSM state snapshot slots for Marconi prefix caching.
    /// Each slot stores SSM h_state + conv_state for all SSM layers,
    /// enabling full prefix skip (KV + SSM) on cache hits.
    /// 0 = disabled. 16 = recommended for repeated-prefix and multi-turn workloads.
    /// Intermediate checkpoints (--ssm-checkpoint-interval) require extra slots:
    /// ~(max_context / checkpoint_interval_tokens) per cached sequence.
    #[arg(long, default_value_t = 16)]
    pub ssm_cache_slots: usize,

    /// Save SSM state snapshots at regular block boundaries during prefill.
    /// When set to N > 0, a snapshot is saved every N blocks during chunked
    /// prefill. On future prefix cache hits, the deepest intermediate snapshot
    /// is restored, reducing SSM recomputation from the full prefix to just
    /// the tokens between the checkpoint and the match point.
    /// 0 = disabled (leaf-only snapshots). 256 = every 4096 tokens (block_size=16).
    #[arg(long, default_value_t = 256)]
    pub ssm_checkpoint_interval: usize,

    /// Enable automatic context compaction for long conversations.
    /// **DISABLED BY DEFAULT** (2026-04-25): the auto-compactor has
    /// historically been a source of agent loops — synthesised
    /// continuation messages and middle-of-history truncation
    /// themselves trigger drift (cf. opencode issues #15533, #17169,
    /// #19339). Oversize requests get a clean 400 error
    /// (`Prompt too long`) rather than a silently-rewritten context.
    ///
    /// Only pass `--auto-compact[=THRESHOLD]` if you have explicitly
    /// validated that compaction is safe for your model + workload.
    /// Without a value: threshold=0.75 (compact at 75% of max_seq_len).
    /// With a value: compact at that fraction (e.g., 0.80 = 80%).
    ///
    /// Method: Active Context Compression (arXiv:2601.07190) — the
    /// server uses the model itself to summarize older conversation
    /// turns into a condensed knowledge block.
    #[arg(long, value_name = "THRESHOLD", num_args = 0..=1, default_missing_value = "0.75")]
    pub auto_compact: Option<f32>,

    /// Default top-n-sigma for sampling (filter tokens by logit z-score).
    /// 0.0 = disabled. Recommended: 1.0 for NVFP4 models AND for agent
    /// workloads — top-n-σ is temperature-invariant (Tang et al.,
    /// arXiv:2411.07641) so it is more robust than top-p across the
    /// per-phase temperature drift agentic loops induce.
    #[arg(long, default_value_t = 1.0)]
    pub default_top_n_sigma: f32,

    /// Default min-p for sampling (keep tokens with prob >= min_p * max_prob).
    /// 0.0 = disabled. Recommended: 0.05-0.1.
    #[arg(long, default_value_t = 0.0)]
    pub default_min_p: f32,

    /// Swap space in GB for KV cache overflow to disk. When GPU blocks are
    /// exhausted, sequences are swapped to disk and resumed later.
    /// 0 = disabled. Swap files stored in /tmp/atlas-swap/.
    #[arg(long, default_value_t = 3)]
    pub swap_space_gb: usize,

    // ── --high-speed-swap (lossless block-level KV streaming) ──
    // Coexists with --swap-space-gb: the existing flag handles
    // sequence-level admission control (whole-sequence evict/restore),
    // --high-speed-swap handles intra-sequence block-level streaming via
    // io_uring + a predictor-driven scratch pool. See spark-storage crate
    // and the plan at .claude/plans/i-want-to-ensure-valiant-bunny.md.
    // Disabled by default; enabling requires the four flags below.
    #[arg(long, default_value_t = false)]
    pub high_speed_swap: bool,

    /// Directory for the per-layer NVMe-backed KV files. Required when
    /// --high-speed-swap is set; must be on a different mount than
    /// --swap-space-gb's /tmp/atlas-swap to avoid file collisions.
    #[arg(long)]
    pub high_speed_swap_dir: Option<std::path::PathBuf>,

    /// Total disk budget for --high-speed-swap, in GiB.
    #[arg(long)]
    pub high_speed_swap_gb: Option<u64>,

    /// HBM scratch slot count (number of resident blocks).
    #[arg(long)]
    pub high_speed_swap_resident_blocks: Option<u32>,

    /// Predictor low-rank dimension (Phase 1 ships at r=32).
    #[arg(long, default_value_t = 32)]
    pub high_speed_swap_rank: u32,

    /// io_uring submission queue depth (Phase 3 shows QD=8 reaches
    /// 3.4 GB/s on this DGX Spark image).
    #[arg(long, default_value_t = 8)]
    pub high_speed_swap_qd: u32,

    /// Capture the per-layer body in a CUDA graph and replay (Phase 4).
    /// Defaults to mirror --high-speed-swap.
    #[arg(long)]
    pub high_speed_swap_graph: Option<bool>,

    /// Per-sequence HBM cache cap for `--high-speed-swap` (Phase 6.1).
    /// When set together with --high-speed-swap, each sequence is limited
    /// to N HBM-resident KV blocks; older blocks are evicted to disk and
    /// streamed back via the orchestrator on demand. The KV cache total
    /// allocation shrinks to roughly `max_batch_size × N` blocks. Default
    /// 64 (= 1024 tokens HBM-resident at block_size=16). Set to
    /// max_seq_len/block_size to disable HBM-shrink (no eviction; useful
    /// for diff-against-no-swap correctness checks).
    #[arg(long, default_value_t = 64)]
    pub high_speed_swap_cache_blocks_per_seq: u32,

    /// Default request timeout in seconds. 0 = no timeout.
    #[arg(long, default_value_t = 300)]
    pub request_timeout: u32,

    /// Enable per-kernel profiling: sync + time each operation within layers.
    /// Disables CUDA graphs for accurate per-op timing. Adds ~10% overhead.
    #[arg(long, default_value_t = false)]
    pub profile: bool,

    /// Number of warmup tokens for online FP8 KV cache scale calibration.
    /// During the first N tokens, tracks max |K| and max |V| values across
    /// all attention layers. After N tokens, computes per-tensor scales as
    /// max/448 (mapping the observed range to FP8 E4M3 [-448, 448]).
    /// 0 = disabled (use static scales from checkpoint, or uncalibrated 1.0).
    /// Only applies when --kv-cache-dtype is fp8.
    #[arg(long, default_value_t = 0)]
    pub fp8_kv_calibration_tokens: usize,

    /// Path to a warmup prompt file (JSON messages or plain text).
    /// At startup, the server tokenizes and prefills this prompt, inserting the
    /// resulting KV cache + SSM snapshot into the prefix cache. This eliminates
    /// the cold-start TTFT penalty (~196ms) on the first real request.
    #[arg(long)]
    pub warmup_prompt: Option<std::path::PathBuf>,

    /// Enable adaptive sampling (entropy-based greedy gating, zone detection).
    /// Computes Shannon entropy over the full vocabulary per token to dynamically
    /// switch between greedy and sampled decoding. Improves quality for mixed
    /// content (code + prose) at the cost of ~2-3x decode throughput reduction.
    /// Off by default for maximum throughput.
    #[arg(long, default_value_t = false)]
    pub adaptive_sampling: bool,

    /// Disable the InstantTensor-style fast weight loader and use the mmap
    /// loader instead. The fast loader (O_DIRECT + pipelined reader/copier,
    /// with a per-shard heuristic that picks between O_DIRECT and buffered
    /// reads) is on by default — this flag is an escape hatch for rare
    /// filesystems that misbehave with O_DIRECT or for A/B debugging.
    /// Setting `ATLAS_FAST_LOAD=0` has the same effect.
    #[arg(long, default_value_t = false)]
    pub no_fast_load: bool,

    /// Address to bind the HTTP listener to. Defaults to `127.0.0.1` so a
    /// fresh install is reachable only from the local machine; pass
    /// `0.0.0.0` to expose on all interfaces (the server logs a warning
    /// when it does, since combined with the permissive default CORS this
    /// makes the API reachable to anything on the LAN).
    #[arg(long, alias = "host", default_value = "127.0.0.1", value_name = "ADDR")]
    pub bind: String,

    /// Require an `Authorization: Bearer <token>` header on `/v1/*`,
    /// `/tokenize`, and `/detokenize`. The token must match one loaded
    /// via `--auth-tokens-file` or `--auth-token`. `/health`, `/health/live`,
    /// and `/metrics` stay open as scrape targets.
    ///
    /// Defaults to off — Atlas is local-by-default, so most users can
    /// skip this. Turn on whenever the server is reachable from anywhere
    /// other than `localhost` (i.e. whenever you've passed `--bind 0.0.0.0`
    /// or are running behind an exposed port-forward).
    #[arg(long, default_value_t = false)]
    pub require_auth: bool,

    /// Path to a file containing valid bearer tokens, one per line. Blank
    /// lines and lines starting with `#` are ignored. Permissions should
    /// be `0600`. The file is read once at startup; SIGHUP reloading is
    /// not supported (restart the server to rotate keys).
    #[arg(long, value_name = "PATH", conflicts_with = "auth_token")]
    pub auth_tokens_file: Option<std::path::PathBuf>,

    /// A single inline bearer token. Convenient for quick starts; not
    /// recommended for production because the token is visible in
    /// `ps`/`/proc/<pid>/cmdline`. Use `--auth-tokens-file` instead.
    #[arg(long, value_name = "TOKEN", conflicts_with = "auth_tokens_file")]
    pub auth_token: Option<String>,
}
