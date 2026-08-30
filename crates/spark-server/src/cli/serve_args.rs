// SPDX-License-Identifier: AGPL-3.0-only

//! `serve` subcommand arguments.
//! Split out of `cli.rs` to keep each file under the 500-LoC cap; the
//! struct is re-exported as `cli::ServeArgs` so call sites are unchanged.
use clap::Parser;
use std::path::PathBuf;

/// Default for `--request-timeout`, in seconds. Named rather than inlined
/// in the clap attribute because it is a production behavior with a
/// user-visible consequence (a cut response), not an anonymous literal:
/// PCND requires the default be documented and overridable, and this is
/// the single place it is defined.
pub const DEFAULT_REQUEST_TIMEOUT_SECS: u32 = 300;

/// Engine fallback for `--num-drafts` when neither the flag nor MODEL.toml
/// `[behavior].default_num_drafts` provides a value. Named per PCND: the
/// CLI → MODEL.toml → engine precedence is resolved explicitly in
/// `serve_phases::config::resolve_num_drafts`, and this is the single place
/// the engine fallback is defined.
pub const DEFAULT_NUM_DRAFTS: usize = 1;

/// Engine fallback for `--kv-cache-dtype` when neither the flag nor MODEL.toml
/// `[behavior].default_kv_dtype` provides a value. Named per PCND; resolved
/// explicitly in `serve_phases::kv_cache::resolve_kv_dtype_str`.
pub const DEFAULT_KV_CACHE_DTYPE: &str = "fp8";

/// Arguments for the `serve` subcommand.
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct ServeArgs {
    /// HuggingFace model ID (e.g. "nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4")
    /// or a local directory path containing config.json.
    /// Optional: omit both this and `--model-from-path` to boot into the
    /// Library and pick a model there. That is a TTY-only affordance — plain
    /// mode has no Library to fall back to, so `validate_serve_args` rejects
    /// the combination rather than starting a server that can serve nothing.
    #[arg(value_name = "MODEL")]
    pub model: Option<String>,

    /// Load a different model when a request names one, ollama-style.
    ///
    /// OFF by default, and the default is the point: even narrowed to models
    /// with a known recipe, one stray request becomes a multi-minute outage for
    /// every other client on the box, and a benchmark sweep naming a sibling
    /// checkpoint would swap mid-run. Only a request whose `model` resolves to
    /// a DIFFERENT known recipe acts — absent, unknown, or already-live names
    /// are served by the current model exactly as they are today.
    #[arg(long)]
    pub auto_swap: bool,

    /// Forbid request-triggered model loading outright.
    ///
    /// For deployments where the served model is part of the contract: a
    /// client must never be able to change what the endpoint is running, no
    /// matter what else is on the command line.
    ///
    /// **This WINS over `--auto-swap`** rather than conflicting with it. A
    /// conflict error would be the wrong behaviour here: the enabling flag
    /// typically comes from a shared base config or an image's default command,
    /// and the operator locking the deployment down appends theirs. Refusing to
    /// start would punish exactly the person doing the safe thing, and — worse
    /// — the natural workaround is to delete the deny flag.
    ///
    /// Only affects REQUEST-triggered swaps. An operator at the dashboard can
    /// still change the model; this is about what a client can cause.
    #[arg(long)]
    pub no_auto_swap: bool,

    /// Serve even when kernel lookups this model's dispatch issued did not
    /// resolve.
    ///
    /// A lookup that returns handle 0 does not error — the caller takes a
    /// slower (or disabled) dispatch path and nothing says so. That is how the
    /// 27B ran with concurrent decode pinned to its per-sequence fallback while
    /// every gate stayed green, so the default is to REFUSE to start and print
    /// the full list with each dispatch site.
    ///
    /// Passing this does not suppress the report: the same enumerated list is
    /// logged at `warn!` on every boot. A flag that muted the warning would
    /// recreate the bug it exists to catch.
    ///
    /// This replaces `ATLAS_ALLOW_SHADOWED_KERNELS`, which covered only the
    /// shadow-dropped subset — one switch, and a CLI flag rather than an
    /// environment variable so it is visible in the command that started the
    /// process.
    #[arg(long, default_value_t = false)]
    pub dangerously_allow_unresolved_kernel_lookups: bool,

    /// Resolve all kernels for the model and exit, reporting any that did not
    /// resolve. Does not start the server. The EXIT CODE IS THE NUMBER of
    /// unresolved kernels — 0 means every lookup resolved.
    ///
    /// A dry run that stops immediately after the kernel audit: config, GPU
    /// init, weight load and model construction all run (every lookup lives in
    /// a layer constructor, so a check that skipped them would resolve a
    /// DIFFERENT set than a real serve), then the report is printed and the
    /// process exits. The scheduler is never started and no port is bound.
    ///
    /// A POSIX status is 8 bits, so the code is CLAMPED at 255 — 256 would be
    /// reported as 0, i.e. a broken model reading as a pass. Whenever the clamp
    /// bites, the true count is printed alongside it and carried in the JSON.
    ///
    /// The exit code IGNORES
    /// `--dangerously-allow-unresolved-kernel-lookups`: a check whose answer
    /// another flag can silence is worth nothing. Passing both still prints the
    /// full list and still exits with the count.
    ///
    /// A one-line JSON object (`{"atlas_kernel_check": …}`) is printed on
    /// stdout after the human report, so a sweep over every target can
    /// aggregate without parsing prose.
    #[arg(long, default_value_t = false)]
    pub check_kernels: bool,

    /// Load model directly from this filesystem path (skips HF cache resolution).
    #[arg(long, value_name = "PATH")]
    pub model_from_path: Option<PathBuf>,

    /// Override model name shown in /v1/models and API responses.
    /// Defaults to the positional MODEL argument, then config.json _name_or_path.
    #[arg(long, alias = "served-model-name", value_name = "NAME")]
    pub model_name: Option<String>,

    /// Pin kernel-target resolution to this compiled target directory name
    /// (e.g. "qwen3.8-27b").
    ///
    /// Normally resolution selects on the checkpoint's `(model_type,
    /// hidden_size)`, breaking ties between config-identical checkpoints
    /// (Qwen3.6-27B vs Qwen3.8-27B) by matching each target's declared
    /// `match_names` against the model id/path. When the reference carries
    /// no identity — `--model-from-path /model` — that tie cannot break and
    /// startup refuses rather than guessing; this flag is the explicit
    /// answer. The pinned target must still declare support for the
    /// checkpoint's `(model_type, hidden_size)`: a pin can choose between
    /// compatible targets, never force an incompatible one.
    #[arg(long, value_name = "TARGET")]
    pub kernel_target: Option<String>,

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
    /// Precedence (highest wins): this flag → MODEL.toml
    /// `[behavior].default_kv_dtype` → fp8 (the safe engine default,
    /// `DEFAULT_KV_CACHE_DTYPE`). An explicitly passed value always wins,
    /// including `fp8` itself. NVFP4 uses less memory but may lose coherence
    /// at long context without --kv-high-precision-layers.
    ///
    /// The `turbo2`/`turbo3`/`turbo4`/`turbo8` variants (and the asymmetric
    /// `*k_*v` pairs built from them) are EXPERIMENTAL: they are not built for
    /// every kernel target, and a target that lacks them fails the kv-cache
    /// kernel preflight at startup rather than serving on a fallback.
    #[arg(long)]
    pub kv_cache_dtype: Option<String>,

    // ── GDN / SSM decode path ──
    //
    // These four were `ATLAS_*` environment variables. They are CONFIGURATION,
    // not diagnostics: the enterprise-concurrency campaign's best recipe needs
    // three of them, and a recipe that has to carry a ten-line env block is a
    // recipe nobody can read or audit. A CLI flag satisfies PCND exactly as an
    // env var does — the value is explicit, never implicit — while also being
    // discoverable in `--help` and visible in `ps`. The environment variables
    // remain honoured as a fallback so existing scripts keep working; the flag
    // WINS when both are given.
    /// Storage dtype for the GDN decode h-state: `f32` (default), `f16`, or
    /// `f16-pool`.
    ///
    /// The decode scan is pure state traffic — it runs at ~90% of GB10's
    /// row-strided ceiling — so halving the state footprint halves its time:
    /// +13.25% at C=16, +19.67% at C=32, and 1.286x WALL at C=64.
    ///
    /// f32 stays the default deliberately. `f16` changes the h-state numerics,
    /// which shifts drafter accept patterns and therefore the emitted
    /// trajectory: a +3.17% token tax was measured in the speculation-ON
    /// regime (C=1/8/32, p=0.030) and 0.23% with speculation off. The tax
    /// vanishes at C=64. Choose it per workload — which is precisely why it is
    /// a flag and not a default.
    ///
    /// `f16-pool` is `f16` PLUS h pools that are SIZED at 2 bytes/element
    /// rather than merely holding FP16 in the first half of an FP32-sized
    /// slot. That is where the memory win is — the h-state is ~95% of a
    /// 151.5 MiB per-sequence state blob, so the bs=128 SSM reserve on the
    /// 27B drops from 36.7 GiB to ~25.6 GiB and the per-sequence per-step
    /// state traffic that sets the marginal cost of a concurrent sequence
    /// halves with it. Prefill keeps its unchanged FP32 kernels: they run
    /// over a per-slot FP32 staging blob (`max_batch_size × one layer's h
    /// blob`, ~256 MB at bs=128 on the 27B) which the layer widens into and
    /// narrows back, so the pool holds FP16 at every moment outside a layer's
    /// prefill call.
    ///
    /// NOT compatible with `--speculative`: the MTP verify arms still stride
    /// and byte-copy the h intermediate/checkpoint pools at the FP32 width,
    /// which over a narrowed pool overruns into the neighbouring slot instead
    /// of failing. Refused at parse time.
    ///
    /// ★ NUMERICS. `f16-pool` puts the FP16 state on the PREFILL recurrence's
    /// chunk boundaries as well as on decode, and a reduced-precision
    /// recurrence accumulates rounding COHERENTLY (vLLM's fp16 mamba cache
    /// needed stochastic rounding for exactly this; there is no published
    /// fp8-recurrent-state accuracy study). Rounding here is
    /// round-to-nearest-even. Default OFF, in no gate recipe, and no number
    /// measured under it may be published without passing
    /// `ssm-state-poisoning-gate`, `decode-floor`, `bfcl-subset` and the
    /// agentic gate.
    ///
    /// Legacy: `ATLAS_SSM_H_FP16` (presence) selects f16 — never f16-pool,
    /// which has no environment spelling — when NONE of the three GDN flags
    /// is given. `GdnFlags` is published as one cell, so any of them takes
    /// the whole decision away from the environment; `warn_shadowed_env`
    /// says so when that happens rather than leaving it to be discovered in a
    /// benchmark number.
    #[arg(long)]
    pub ssm_h_dtype: Option<String>,

    /// EXPERIMENTAL — SSM verify-rollback mode: `snapshot` (default) or
    /// `replay`.
    ///
    /// `snapshot` is the wired production path: every verify writes per-token
    /// h/conv state snapshots and a partial accept restores from them.
    /// `replay` is a capacity SCAFFOLD: it keeps only the pre-verify
    /// checkpoint blob per verify slot plus a small verify-window input ring
    /// (-18.8 GiB total reserve at bs=128/K=4 on the 27B vs the wave-47
    /// reference), and reconstructs partial accepts by replaying accepted
    /// tokens from the checkpoint. Its device wiring (input capture + replay)
    /// is NOT implemented yet: a replay serve boots — the preflight reserve
    /// shows the win — but every speculative verify step refuses loudly.
    /// The default is explicit (PCND): published on every serve.
    #[arg(long, default_value = "snapshot")]
    pub ssm_rollback_mode: String,

    /// Fused GDN output-norm kernel on the decode path (default: off).
    ///
    /// Required by `--ssm-h-dtype f16`: the FP16 h-state twins live on the
    /// fused-norm arm, and the unfused arm is FP32-only. Left opt-in because
    /// its bitwise gate could not certify output-equivalence — the CONTROL leg
    /// (two identical serves) itself differed on 7 of 42 completions at C=4/16,
    /// so the flag legs' 5-6 differences are not separable from run-to-run
    /// nondeterminism. Under PCND an unproven numerics change is explicit
    /// configuration, not a default.
    ///
    /// Legacy: `ATLAS_GDN_FUSED_NORM=1`, on the same terms as `--ssm-h-dtype`.
    ///
    /// `Option` so that ABSENT is distinguishable from `false`: publishing the
    /// clap default sealed the flags cell on every boot, which made the legacy
    /// variable inert while `--help` still documented it. Bare
    /// `--gdn-fused-norm` still means on; `--gdn-fused-norm false` is the
    /// explicit off.
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    pub gdn_fused_norm: Option<bool>,

    /// Batched multi-sequence GDN recurrent decode kernel (default: off).
    ///
    /// One strided launch across the batch instead of one per sequence.
    /// Same bitwise-certification gap as `--gdn-fused-norm`; see that flag.
    ///
    /// Legacy: `ATLAS_SSM_BATCHED_RECURRENT=1`, on the same terms as
    /// `--gdn-fused-norm`, and `Option` for the same reason.
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    pub ssm_batched_recurrent: Option<bool>,

    /// VARLEN (ragged) batched prefill — OPT-IN (default: off).
    ///
    /// Concatenates concurrently queued prompts of DIFFERENT lengths into one
    /// forward per wave (cu_seqlens geometry): every projection/FFN GEMM
    /// launches once at M = Σ tokens instead of once per request at its own
    /// small M, and the scheduler defers chunk-0 so late arrivals join the
    /// next wave. Waves are capped at the `--max-prefill-tokens` budget and
    /// iterate until every queued stream has advanced.
    ///
    /// Same certification caveat as `--ssm-batched-recurrent`: batching
    /// changes GEMM row counts, and kernels selected on row count round
    /// differently, so per-request outputs are not bitwise-identical to the
    /// serial path. Gate recipes stay on the default (off) until certified.
    ///
    /// Legacy: `ATLAS_PREFILL_VARLEN=1`, on the same terms as
    /// `--gdn-fused-norm`, and `Option` for the same reason.
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    pub prefill_varlen_batch: Option<bool>,

    /// Sequential-decode-exact GDN/SSM verify chain — OPT-IN (default: off).
    ///
    /// SCOPE, and it is narrower than this flag once claimed: it makes the
    /// GDN/SSM verify chain exact. It does NOT make speculative output
    /// bitwise-equal to non-speculative output end to end, and setting it
    /// will not give you a reproducible spec-on serve.
    ///
    /// Why not (measured on GB10, issue #459): every FFN and attention
    /// projection is computed by a kernel selected on ROW COUNT. A token
    /// through K=4 verify takes `w4a16_gemv_batch4`; the same token through
    /// sequential decode takes `w4a16_gemv`. Those are separate
    /// implementations and they round differently — ~5e-5 of output lanes
    /// differ by exactly 1 ULP in BF16, on every projection shape measured
    /// (qkv/o, ffn gate/up, ffn down). Across 64 layers that is ample to flip
    /// a thin top-2 argmax, and it is entirely outside the chain this flag
    /// makes exact, so no amount of SSM exactness closes it.
    ///
    /// FUTURE WORK: route every verify token through the same single-row
    /// kernels sequential decode uses — FFN gate/up/down, attention QKV and
    /// out_proj — which is what the exact arm already does for the GDN chain,
    /// extended to the rest of the forward. That, not this flag alone, is
    /// what end-to-end spec-on == spec-off would require.
    ///
    /// The default verify pass runs the WY-chunkwise / fused BF16-conv arms:
    /// fast, but their BF16-output conv (h-state relL2 ~8.6e-4 per K=4
    /// window, committed into persistent SSM state) plus a ~3.4e-8 chunkwise
    /// reordering term diverge from the sequential-decode reference — an
    /// argmax flip only needs a per-logit error above a thin top-2 margin.
    /// With `--exact-verify` the verify pass instead runs, per token, exactly
    /// the GDN/SSM kernel chain sequential decode runs (measured h relL2 =
    /// 0.0), at a measured decode-step cost of ~+35% at the n=8/K=4 verify
    /// rung, ~+22% at n=16/K=2 and ~+36% at n=32/K=2 (GDN phase
    /// +116%/+63%/+69%). Those two divergence terms are what this closes —
    /// the projection-kernel term described above is not among them.
    ///
    /// Opt-in follows every surveyed production engine — vLLM
    /// (`VLLM_BATCH_INVARIANT=1`), SGLang (`--enable-deterministic-inference`,
    /// ~34% avg slowdown), TensorRT-LLM and TGI (no exact mode at all) — none
    /// pays for exactness by default.
    ///
    /// Rejected beside `--ssm-h-dtype f16`: the exact arm's kernels are FP32
    /// readers and must never read the FP16 h-state pool.
    ///
    /// Replaces `--verify-wy` (removed, never in a release), which was the
    /// opt-OUT back when exact was briefly the default.
    ///
    /// No legacy environment variable — new configuration is CLI-only.
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    pub exact_verify: Option<bool>,

    /// Mid-chunk SSM tail capture on the prefill path (default: on).
    ///
    /// Captures GDN recurrent + conv state in-pass at the block-floored
    /// matched-prefix boundary, removing the ~868 ms extra forward pass the
    /// clamp-based tail-checkpoint path costs on a warm turn. Off is
    /// byte-identical to the pre-2026-07-19 baseline.
    ///
    /// Legacy: `ATLAS_SSM_TAIL_MIDCHUNK=0` disables when this flag is ABSENT.
    ///
    /// Absent is not the same as `--ssm-tail-midchunk true`, which is why there
    /// is no clap default here: publishing a default sealed the runtime's cell
    /// on every boot and made that documented opt-out a silent no-op. Give the
    /// flag to decide, omit it to let the environment decide, and with neither
    /// it is on.
    #[arg(long, action = clap::ArgAction::Set)]
    pub ssm_tail_midchunk: Option<bool>,

    /// MTP throughput gate: `auto` (default) or `force`.
    ///
    /// `auto` arms the arbiter, which measures whether speculative verify is
    /// net-positive in the current regime and stops paying for it when it is
    /// not. `force` DISARMS the arbiter so verify steps keep flowing even
    /// where it would measure them net-negative — a diagnostic, needed to
    /// collect verify samples for attribution, and never a production setting:
    /// if forcing wins, the GATE is miscalibrated and that is the fix, not
    /// this flag. To run without speculation at all, omit `--speculative`.
    ///
    /// Legacy: `ATLAS_MTP_GATE_FORCE=1` selects `force` when this flag is
    /// ABSENT. As with `--ssm-tail-midchunk`, there is no clap default: a
    /// published default would seal the scheduler's cell to `auto` on every
    /// boot and silently ignore the variable it documents.
    #[arg(long)]
    pub mtp_gate: Option<String>,

    /// LM-head precision: `default` (the clap default — no override, the model config
    /// decides), `bf16` (final vocab projection in BF16 — the SAFE CHOICE, and what to pass
    /// when the config picks something lower; matches vLLM checkpoint precision), `nvfp4` (force the
    /// model's NVFP4-packed lm_head), or `fp8` (runtime-quantize the lm_head to FP8 E4M3
    /// per-row, w8a16_gemv decode). The big vocab projection (~1.78 GB/token BF16 at a
    /// 248K vocab) is the single largest per-token weight read; `fp8` halves it and `nvfp4`
    /// quarters it for a real decode speedup (measured Qwen3.6-35B-A3B: bf16 72 → fp8 86
    /// (+20%) → nvfp4 97 (+35%) tok/s).
    ///
    /// ⚠️  WARNING — reducing lm-head precision below BF16 can be CATASTROPHIC on some
    /// models. Low-margin argmax flips in the final vocab projection COMPOUND over LONG
    /// structured generation and derail the output entirely, even when SHORT responses look
    /// perfectly clean. MEASURED on Qwen3.6-35B-A3B (webserver_ok agentic harness, N=10):
    /// bf16 = 10/10; BOTH fp8 and nvfp4 COLLAPSE (cargo-invalid output, 360s timeouts)
    /// despite clean short-prompt coherence. ALWAYS quality-gate per-model with a
    /// long-generation harness before enabling nvfp4/fp8 — do not trust a short smoke.
    /// (An FP32-accumulate logits path would cut the flips but forces host-side sampling
    /// → ~6 tok/s; making nvfp4/fp8 both fast AND safe needs a GPU-side FP32 sampler.)
    /// Replaces the former ATLAS_LMHEAD_BF16 env var.
    #[arg(long, default_value = "default")]
    pub lm_head_dtype: String,

    /// Boundary attention layers to keep at BF16 KV cache precision (first N + last N).
    /// Protects attention sink tokens (early layers) and output quality (final layers)
    /// from quantization error while saving memory on middle layers.
    /// Accepts: number, "auto" (=2, recommended), "max"/"all" (all BF16).
    /// Default: 0 (all layers use --kv-cache-dtype).
    #[arg(long, default_value = "0")]
    pub kv_high_precision_layers: String,

    /// Fraction of total GPU memory this process may consume (0.0-1.0).
    /// Weights, buffers, KV cache, and reserves all count against this cap.
    /// Matches vLLM / sparkrun semantics: 0.50 on a 120 GB device means
    /// Atlas will use at most ~60 GB in total.
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
    /// Sets the per-request ceiling for thinking-block length, and anchors
    /// the client `reasoning_effort` ladder (minimal/low/medium/high/xhigh
    /// = 1/4x, 1/2x, 1x, 2x, 4x of this value). An explicit client token
    /// budget (`thinking.budget_tokens`, `thinking_token_budget`) still
    /// wins outright; the (max_tokens * 9 / 10) safety cap is enforced
    /// unless MODEL.toml sets `cap_thinking_at_max_tokens = false`.
    #[arg(long)]
    pub max_thinking_budget: Option<u32>,

    /// Override MODEL.toml's `[behavior].max_inter_tool_prose` (tokens):
    /// the cap on free-prose tokens between successive tool calls on a
    /// tool-armed request, after which the scheduler ends the response
    /// with finish_reason "length" (#328). 0 disables the guard entirely.
    /// Precedence (highest wins): this flag → ATLAS_MAX_INTER_TOOL_PROSE
    /// → MODEL.toml → built-in default (3072).
    #[arg(long)]
    pub max_inter_tool_prose: Option<u32>,

    /// Arm or disarm the content-loop watchdog, overriding MODEL.toml's
    /// `[behavior].enable_loop_watchdog`. The watchdog ends (or rolls back)
    /// a response whose tail is a short-period token repeat; its built-in
    /// threshold (3 end-anchored repeats of a period-2..64 pattern) can
    /// false-positive on legitimately repetitive output such as code.
    /// Precedence (highest wins): this flag → ATLAS_CONTENT_LOOP_WATCHDOG
    /// → MODEL.toml. Runtime-toggleable from the TUI via `/watchdog on|off`.
    #[arg(long)]
    pub content_loop_watchdog: Option<bool>,

    /// Override the content-loop watchdog's repeat threshold (end-anchored
    /// consecutive repeats that constitute a loop; built-in default 3).
    /// Raise it for models whose legitimate output is short-period
    /// repetitive (code, tables). A per-request `repetition_detection`
    /// object still outranks this. Precedence: this flag →
    /// ATLAS_CONTENT_LOOP_MIN_REPEATS → built-in default.
    #[arg(long)]
    pub content_loop_min_repeats: Option<u32>,

    /// Override MODEL.toml's `[behavior].disable_tool_grammar`.
    /// When true, the server skips XGrammar structural-tag enforcement on
    /// `tool_choice="auto"` requests; tools are still parsed from output
    /// post-hoc by the tool_call_parser. Matches vLLM's default behaviour
    /// in auto mode (vLLM only grammar-constrains when tool_choice="required").
    #[arg(long)]
    pub disable_tool_grammar: Option<bool>,

    /// Default chat template kwargs applied when the client sends no
    /// thinking parameters (no `reasoning.effort`, `chat_template_kwargs`,
    /// or `enable_thinking` in the request body). A JSON object with
    /// optional keys: `enable_thinking` (bool), `thinking_budget` (u32),
    /// `reasoning_effort` (none|minimal|low|medium|high|xhigh|max — the
    /// served default effort tier when clients are silent; unset = the
    /// neutral "medium"), `preserve_thinking` (bool). Unknown keys or an
    /// unknown `reasoning_effort` value abort startup (fail-fast; they
    /// were silently ignored before 2026-08-15).
    ///
    /// Precedence (highest wins): request body → this flag → MODEL.toml.
    /// Example: `--default-chat-template-kwargs '{"reasoning_effort":"xhigh"}'`
    #[arg(long, value_name = "JSON")]
    pub default_chat_template_kwargs: Option<String>,

    /// Ignore the `jinja-templates/` override directory and render every
    /// model off its OWN chat template (`chat_template.jinja` /
    /// `tokenizer_config.json`), relying on the Rust message-preprocessing
    /// (`tokenizer/message_preprocess.rs`) for Atlas's cross-cutting chat
    /// behaviors. Default off: an override file's presence is the opt-in
    /// signal that a model needs a template fix Rust preprocessing can't
    /// express (see `jinja-templates/README.md`).
    #[arg(long, default_value_t = false)]
    pub disable_template_overrides: bool,

    /// Enable MTP speculative decoding. The scheduler then MEASURES the
    /// verify-step cost over the first decode steps of serving and auto-disables
    /// MTP if it is provably net-negative (verify multiplier ≥ 1 + num_drafts),
    /// so this flag never regresses decode on configs where MTP does not pay
    /// off. See `scheduler::mtp_gate`.
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

    /// DFlash block size γ (parallel draft tokens per step). Default 16 —
    /// the `block_size` of every published Qwen3.6-DFlash drafter; override
    /// only for ablation. Higher γ increases per-step verify cost but
    /// raises peak speedup.
    ///
    /// Unset = the drafter's trained block size (`dflash_config.block_size`,
    /// via `effective_block_size()`), which is the only correct value for a
    /// block-diffusion drafter: serving Qwen3.8-27B-DFlash2 (block 8) at the
    /// old clap default of 16 corrupted every draft row (bidirectional block
    /// attention shares softmax with the 8 phantom rows) — accept measured
    /// 1.1% at 16 vs 3.2%+ at 8. An explicit value remains an override for
    /// ablation only.
    #[arg(long)]
    pub dflash_gamma: Option<usize>,

    /// DFlash drafter sliding-window size for long context. The drafter
    /// runs full-prefix attention by default; at Atlas's typical 16K
    /// `--max-seq-len`, drafter attention dominates per-step cost. The
    /// upstream sglang / vLLM default is 4096. Set to 0 to disable
    /// (full attention).
    #[arg(long, default_value_t = 4096)]
    pub dflash_window_size: usize,

    /// Number of draft tokens per speculative step (1=K=2, 2=K=3, 3=K=4 verify).
    /// Higher K verifies more drafts per step. Uses WY-chunkwise GDN kernels.
    /// Precedence (highest wins): this flag → MODEL.toml
    /// `[behavior].default_num_drafts` → 1 (`DEFAULT_NUM_DRAFTS`). An
    /// explicitly passed value always wins, including `--num-drafts 1` on a
    /// model whose MODEL.toml defaults higher.
    #[arg(long)]
    pub num_drafts: Option<usize>,

    /// Maximum concurrent sequences batched into one GPU decode step.
    #[arg(long, default_value_t = 8)]
    pub max_batch_size: usize,

    /// MTP head weight precision: bf16 (default, highest acceptance rate
    /// = highest end-to-end throughput; the MTP head is small so the memory
    /// cost is modest), fp8 (1 byte/weight, balanced; slower draft due to
    /// a D2H sync in MoE dispatch), nvfp4 (0.5 byte/weight, fastest draft
    /// forward but lossier projections → lower acceptance rate, so end-to-
    /// end throughput is usually worse than bf16).
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
    /// Supported: "hermes" (JSON `<tool_call>`, Qwen3-VL / Qwen3-Next),
    /// "qwen3_coder" (Qwen3-Coder XML `<function=...>`, Qwen3.5 family +
    /// Nemotron-H), "qwen3_xml", "gemma4", "mistral", "minimax_xml",
    /// "bare_json", "poolside_v1". See the `FromStr for ToolCallFormat` in
    /// tool_parser.rs.
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
    /// When set to N > 0, a snapshot is saved at every chunked-prefill chunk
    /// boundary whose block index is a multiple of N. On future prefix cache
    /// hits, the deepest intermediate snapshot is restored, reducing SSM
    /// recomputation to the tokens between the checkpoint and the match point.
    /// Independent of this interval, a tail checkpoint is always saved at the
    /// prompt's last full-block boundary (plus a leaf snapshot at prompt end);
    /// warm multi-turn restores hit the tail checkpoint. Chunk size is never
    /// reduced to serve this interval.
    /// 0 = tail + leaf snapshots only. 256 = every 4096 tokens (block_size=16).
    #[arg(long, default_value_t = 256)]
    pub ssm_checkpoint_interval: usize,

    /// Record which MoE experts each request routes to, so a request that
    /// asks (`"report_expert_metadata": true` in the request body) gets the
    /// per-layer expert set back on `usage.expert_activation`.
    ///
    /// A BOOT flag, not a per-request one, because the device-side copies
    /// that collect the routing have to be recorded into the decode CUDA
    /// graph: a request cannot add work to a graph that was already
    /// captured. Which requests REPORT is still per-request; this only
    /// decides whether the serve can answer at all.
    ///
    /// Costs one staging buffer sized (layers x max batch tokens x top-k)
    /// x 8 bytes, and one device-to-device copy per MoE layer per pass.
    /// Ignored on a dense (non-MoE) checkpoint, which has no router to
    /// observe; a request that asks for experts there is refused by name.
    ///
    /// Feeds the `expert-categories` benchmark, which maps prompt
    /// categories to the experts they need.
    #[arg(long, default_value_t = false)]
    pub expert_telemetry: bool,

    /// Load ONLY the MoE experts this prompt category routes to, from the
    /// model's MODEL.toml `[expert_categories]` table.
    ///
    /// The table is produced by the `expert-categories` benchmark and read at
    /// BUILD time, so a newly measured category needs a rebuild before this
    /// flag can name it. An unknown name lists what the model declares.
    ///
    /// Experts outside the category are not loaded at all, and the router is
    /// masked so it cannot select one — the serve answers from a re-weighted
    /// blend of the experts it holds. Expect a quality cost that grows as
    /// coverage falls, and traffic outside the category's register to pay the
    /// most.
    ///
    /// Refused rather than silently degraded on configurations whose routing
    /// path is not masked (expert parallelism, zero-computation-expert
    /// routers), and on dense checkpoints, which have no experts to select.
    #[arg(long)]
    pub expert_category: Option<String>,

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
    ///
    /// A5 (2026-05-26): default raised from 0.0 → 0.08 per the
    /// `research3_quantization_sampler.md` recommendation #3. On FP8
    /// models the long noisy tail of the logit distribution gets
    /// truncated, preventing low-probability tokens from winning under
    /// FP8 quantization noise. BF16 models pay essentially nothing for
    /// the floor (their distributions are already concentrated).
    /// Override via `--default-min-p 0.0` if a deployment specifically
    /// needs to disable.
    #[arg(long, default_value_t = 0.08)]
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

    /// Server-side deadline for a single request, in seconds. A request
    /// that exceeds it is CUT and the response is reported with
    /// `finish_reason="timeout"` (never "length") plus a WARN log naming
    /// the slot, elapsed time and tokens emitted — a truncation must never
    /// look like a normal completion. `0` disables the deadline entirely.
    /// Overridable per request via the OpenAI `timeout` field.
    #[arg(long, default_value_t = DEFAULT_REQUEST_TIMEOUT_SECS)]
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
    /// Precedence (highest wins): this flag → MODEL.toml
    /// `[behavior].fp8_kv_calibration_tokens` → 0. An explicit value always
    /// wins — passing 0 force-disables calibration even on a model whose
    /// MODEL.toml enables it.
    #[arg(long)]
    pub fp8_kv_calibration_tokens: Option<usize>,

    /// Headroom multiplier applied to the first-observe absmax when the online
    /// FP8 KV scale freezes (calibration freezes on the FIRST observe so the
    /// write scale always equals the read scale). The first observe sees only
    /// the first prefill chunk, so the frozen scale covers headroom× its
    /// observed max — later tokens whose magnitude grows don't clip, at a cost
    /// of <1 bit of precision. Must be ≥ 1.0 (below 1.0 guarantees clipping;
    /// rejected at startup). Replaces `ATLAS_FP8_KV_HEADROOM`.
    #[arg(long, default_value_t = 2.0)]
    pub fp8_kv_headroom: f32,

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

    /// Disable the interactive TUI dashboard even on a TTY, keeping the plain
    /// log stream. The TUI also auto-disables when stdout/stdin is not an
    /// interactive terminal (pipes, `docker -d`, CI) or `ATLAS_NO_TUI=1`.
    #[arg(long, default_value_t = false)]
    pub no_tui: bool,

    /// Ask the fast loader to prefetch each buffered shard before per-tensor
    /// reads. Useful on NFS-backed model stores with many small tensors per
    /// shard, where normal kernel readahead may not keep up. Also enabled by
    /// `ATLAS_FAST_LOAD_PREFETCH_SHARDS=1`.
    #[arg(long, default_value_t = false)]
    pub fast_load_prefetch_shards: bool,

    /// Vision input AREA bound in pixels, applied before patching. Overrides
    /// the checkpoint in BOTH directions — it may raise the bound as well as
    /// lower it.
    ///
    /// 0 (the default) means "use the checkpoint's own bound", read from
    /// `preprocessor_config.json` (`size.longest_edge`, or `max_pixels`;
    /// despite the name both are pixel COUNTS). When the checkpoint declares
    /// none, the preprocessor falls back to clamping the long side to 1280px.
    ///
    /// Until 2026-08-14 this flag could only ever LOWER the resolution: the
    /// 1280px clamp was unconditional and the checkpoint's own bound was
    /// never read, so a model built for 4096² was served at roughly a tenth
    /// of its permitted area with nothing logged. Raising this raises the
    /// vision token count per image quadratically — a 4096² image is ~16k
    /// merged tokens — so it is charged against the context budget.
    ///
    /// Also settable with `ATLAS_VISION_MAX_PIXELS`.
    #[arg(long, default_value_t = 0)]
    pub vision_max_pixels: usize,

    /// Fetch `image_url` parts that carry an http(s) URL, instead of
    /// rejecting them.
    ///
    /// OFF by default, and deliberately not default-ON-with-a-kill-switch
    /// like most Atlas features. Enabling it makes the inference server issue
    /// outbound HTTP to addresses chosen by anyone who can send it a chat
    /// request — a server-side request forgery primitive. A deployment that
    /// never wanted that must not acquire it by upgrading. With the flag off,
    /// a URL is refused with a 400 naming this flag, so the capability is
    /// discoverable rather than silently missing; clients that cannot be
    /// changed send a base64 `data:` URI instead.
    ///
    /// Switched on, the fetch is still bounded: loopback/private/link-local
    /// destinations are refused (including across redirects), the body is
    /// capped while being read rather than by trusting `Content-Length`, the
    /// response must declare an image content type, and the whole request is
    /// time-limited.
    #[arg(long, default_value_t = false)]
    pub vision_allow_remote_images: bool,

    /// Cap, in MiB, on a single fetched remote image. Enforced against bytes
    /// actually read, so a remote understating its `Content-Length` cannot
    /// exceed it. No effect unless `--vision-allow-remote-images` is set.
    #[arg(long, default_value_t = 20)]
    pub vision_remote_image_max_mb: usize,

    /// Wall-clock budget, in seconds, for fetching one remote image. Bounds a
    /// slow-loris response, which would otherwise hold a thread from the
    /// prepare pool for as long as the remote cared to keep the socket open.
    /// No effect unless `--vision-allow-remote-images` is set.
    #[arg(long, default_value_t = 10)]
    pub vision_remote_image_timeout_s: u64,

    /// Also permit remote images on loopback, private and link-local
    /// addresses.
    ///
    /// A SECOND grant on top of `--vision-allow-remote-images`, because
    /// "fetch from the public internet" and "fetch from inside my network"
    /// have different blast radii. Only set this where the image host is
    /// genuinely internal: it re-opens the path to link-local cloud instance
    /// metadata (169.254.169.254), where a successful fetch returns
    /// credentials as an ordinary HTTP body.
    #[arg(long, default_value_t = false)]
    pub vision_remote_image_allow_private: bool,

    /// Decode video content parts with ffmpeg.
    ///
    /// ★ VIDEO SUPPORT REQUIRES FFMPEG ON THE HOST for every container except
    /// animated GIF. Atlas does not bundle a video decoder: GIF is decoded
    /// in-process in pure Rust, and MP4/MOV, WebM/Matroska and AVI — that is,
    /// H.264, H.265, VP9 and AV1 — are decoded by running `ffmpeg`. Without
    /// this flag a video part is refused with a 400 naming the flag; with it
    /// but no ffmpeg on PATH, the server WARNS loudly at startup and each
    /// video request fails naming the binary.
    ///
    /// Install it with `apt install ffmpeg` (Debian/Ubuntu) or
    /// `dnf install ffmpeg` (Fedora/RHEL).
    ///
    /// Off by default because it makes the server execute another program
    /// per video request. The decode is bounded on every axis the caller
    /// controls — no shell, no temp file, capped frames, capped output,
    /// capped wall clock — but a deployment that does not want subprocess
    /// execution must not acquire it by upgrading.
    #[arg(long, default_value_t = false)]
    pub video_allow_ffmpeg: bool,

    /// Path to the ffmpeg binary. A bare name is resolved on PATH; an
    /// absolute path is used as given, so a deployment can pin a known build
    /// instead of inheriting whatever PATH offers. No effect unless
    /// `--video-allow-ffmpeg` is set.
    #[arg(long, default_value = "ffmpeg")]
    pub video_ffmpeg_path: String,

    /// Frames per second to sample a video at. The checkpoints' own
    /// `video_processor` blocks declare 2, and the sampling is done BY the
    /// decoder rather than by decoding everything and discarding most of it.
    /// Raising this multiplies the vision tokens a clip costs.
    #[arg(long, default_value_t = 2.0)]
    pub video_fps: f32,

    /// Hard cap on frames taken from one video, before temporal grouping.
    /// Matches the checkpoints' `max_frames`. At the default 2 fps this is
    /// just over six minutes of clip.
    #[arg(long, default_value_t = 768)]
    pub video_max_frames: usize,

    /// Wall-clock budget for decoding one video. Bounds a decoder that hangs
    /// on a malformed container; the child is killed when it expires.
    #[arg(long, default_value_t = 120)]
    pub video_decode_timeout_s: u64,

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

    /// LoRA adapter to serve, as NAME=PATH_OR_HF_ID (e.g.
    /// `holo-sft=/data/adapters/holo-sft` or `holo-sft=org/holo-31-08b-lora`).
    /// Repeatable: each adapter loads into its own pool slot at startup and is
    /// advertised by GET /v1/models; requests route per-slot via the `adapter`
    /// field (unset = the installed active adapter). Runtime BF16 delta —
    /// never merged into the base weights.
    #[arg(long, value_name = "NAME=PATH_OR_HF_ID", value_parser = parse_lora_adapter_spec)]
    pub lora_adapter: Vec<(String, String)>,

    /// NLLB/M2M-100 ONLY: source-language token for translation (e.g.
    /// `eng_Latn`). Prepended to the encoder input. Required when serving an
    /// `m2m_100`/`nllb` checkpoint; ignored otherwise.
    #[arg(long)]
    pub src_lang: Option<String>,

    /// NLLB/M2M-100 ONLY: target-language token (`forced_bos`, e.g. `fra_Latn`,
    /// `gvn_Latn`). Forced as the first decoded token. Required when serving an
    /// `m2m_100`/`nllb` checkpoint; ignored otherwise.
    #[arg(long)]
    pub tgt_lang: Option<String>,

    /// Maximum LoRA adapter rank. The A/B slot pool and delta scratch buffers
    /// are allocated rank-padded to this value at startup (frozen v1 layout
    /// contract); an adapter whose `r` exceeds it is rejected at load.
    ///
    /// UNSET (default) derives it from the adapters actually configured, which
    /// is almost always what you want: BOTH delta stages contract at the padded
    /// rank, and the B operand is `[n_out, max_rank]`, so padding an r=8 adapter
    /// to 64 moves 8x the bytes for the same math. Measured on qwen3.8-27B with
    /// an r=8 adapter: pool 5392 -> 674 MiB and prefill 608 -> 730 tok/s just
    /// from not padding.
    ///
    /// Set it explicitly only to reserve headroom for a LARGER adapter staged
    /// in later — the pool layout is frozen at startup, so a stage-in above the
    /// pool's rank is a named reject.
    #[arg(long)]
    pub max_lora_rank: Option<usize>,

    /// Maximum number of LoRA adapter slots in the rank-padded pool. Slots
    /// beyond the startup-resident adapters are cache headroom for demand
    /// promotion (`--lora-stageable`).
    #[arg(long, default_value_t = 8)]
    pub max_loras: usize,

    /// Task #27: a STAGEABLE (promotable-but-not-resident) LoRA adapter, as
    /// `NAME=PEER_STAGE_ID=CONFIG_DIR` (repeatable). NAME is what a request's
    /// `adapter` field asks for; PEER_STAGE_ID is the adapter's id on the
    /// `$ATLAS_LORA_PEER` weight peer; CONFIG_DIR is a local dir with
    /// `adapter_config.json` (the peer manifest carries no r/alpha, so the peft
    /// scaling is read from here at startup). A request naming a stageable
    /// adapter triggers an on-miss RDMA promotion into a cache pool slot instead
    /// of a 404. Requires `$ATLAS_LORA_PEER`. Empty = today's resident-only
    /// behaviour, byte-identical.
    #[arg(long, value_name = "NAME=PEER_ID=DIR", value_parser = parse_lora_stageable_spec)]
    pub lora_stageable: Vec<(String, String, String)>,

    /// A DISK-stageable (promotable-but-not-resident, NO peer) LoRA adapter, as
    /// `NAME=PATH_OR_HF_ID` (repeatable). A request naming NAME triggers an
    /// on-miss DISK fault-in into a cache pool slot (LRU-evicted) instead of a
    /// 404 — the no-RDMA sibling of `--lora-stageable`. Needs
    /// `ATLAS_LORA_ROTATE=1` (so decode runs eager and the disk swap can
    /// re-point a cache slot) and `--max-loras > resident count` for cache
    /// headroom. Empty = today's behaviour, byte-identical.
    #[arg(long, value_name = "NAME=PATH_OR_HF_ID", value_parser = parse_lora_adapter_spec)]
    pub lora_stageable_disk: Vec<(String, String)>,
}

impl ServeArgs {
    /// Effective speculative draft count. Valid only AFTER
    /// `serve_phases::apply_model_default_num_drafts` has resolved the
    /// CLI → MODEL.toml → engine-default precedence into `num_drafts` (it
    /// runs before any GPU/pool sizing in `serve_load`). Reading it earlier
    /// is a startup-ordering bug: fail fast rather than size a pool off a
    /// guessed value. Pre-resolution readers (CLI validation, TUI badges)
    /// must match on the `Option` directly instead.
    /// The served DFlash γ: explicit flag wins; otherwise the drafter's
    /// trained block size (caller passes it once parsed); otherwise the
    /// legacy 16 (every pre-DFlash2 published drafter).
    pub fn resolved_dflash_gamma(&self, drafter_block_size: Option<usize>) -> usize {
        self.dflash_gamma.or(drafter_block_size).unwrap_or(16)
    }

    pub fn resolved_num_drafts(&self) -> usize {
        self.num_drafts
            .expect("num_drafts read before apply_model_default_num_drafts resolved it")
    }
}

/// Value parser for `--lora-adapter NAME=PATH_OR_HF_ID`.
fn parse_lora_adapter_spec(s: &str) -> Result<(String, String), String> {
    let (name, spec) = s
        .split_once('=')
        .ok_or_else(|| format!("--lora-adapter must be NAME=PATH_OR_HF_ID, got '{s}'"))?;
    if name.is_empty() || spec.is_empty() {
        return Err(format!("--lora-adapter: empty name or path in '{s}'"));
    }
    Ok((name.to_string(), spec.to_string()))
}

/// Value parser for `--lora-stageable NAME=PEER_ID=DIR` (Task #27). Splits into
/// exactly three non-empty parts on the first two `=` (a filesystem DIR may
/// itself contain no `=`; peer ids and names never do). All three parts are
/// required — a missing DIR would leave the promoted adapter with no peft
/// scaling source.
fn parse_lora_stageable_spec(s: &str) -> Result<(String, String, String), String> {
    let mut parts = s.splitn(3, '=');
    let name = parts.next().unwrap_or("");
    let peer_id = parts.next().unwrap_or("");
    let dir = parts.next().unwrap_or("");
    if name.is_empty() || peer_id.is_empty() || dir.is_empty() {
        return Err(format!(
            "--lora-stageable must be NAME=PEER_ID=DIR (all three non-empty), got '{s}'"
        ));
    }
    Ok((name.to_string(), peer_id.to_string(), dir.to_string()))
}

#[cfg(test)]
mod stageable_spec_tests {
    use super::{parse_lora_adapter_spec, parse_lora_stageable_spec};

    #[test]
    fn parses_disk_stageable() {
        // `--lora-stageable-disk NAME=PATH_OR_HF_ID` reuses parse_lora_adapter_spec.
        assert_eq!(
            parse_lora_adapter_spec("cold-a=/data/adapters/cold-a"),
            Ok(("cold-a".to_string(), "/data/adapters/cold-a".to_string()))
        );
        assert_eq!(
            parse_lora_adapter_spec("cold-b=org/cold-b-lora"),
            Ok(("cold-b".to_string(), "org/cold-b-lora".to_string()))
        );
        assert!(parse_lora_adapter_spec("cold-a").is_err());
        assert!(parse_lora_adapter_spec("=/data/adapters/cold-a").is_err());
        assert!(parse_lora_adapter_spec("cold-a=").is_err());
    }

    #[test]
    fn parses_three_parts() {
        assert_eq!(
            parse_lora_stageable_spec("sparky=stage-7=/data/adapters/sparky"),
            Ok((
                "sparky".to_string(),
                "stage-7".to_string(),
                "/data/adapters/sparky".to_string()
            ))
        );
    }

    #[test]
    fn rejects_missing_parts() {
        assert!(parse_lora_stageable_spec("sparky").is_err());
        assert!(parse_lora_stageable_spec("sparky=stage-7").is_err());
        assert!(parse_lora_stageable_spec("=stage-7=/dir").is_err());
        assert!(parse_lora_stageable_spec("sparky==/dir").is_err());
        assert!(parse_lora_stageable_spec("sparky=stage-7=").is_err());
    }

    #[test]
    fn dir_may_contain_equals_after_first_two() {
        // splitn(3) keeps everything after the 2nd '=' as the DIR.
        assert_eq!(
            parse_lora_stageable_spec("n=p=/weird/dir=x"),
            Ok(("n".to_string(), "p".to_string(), "/weird/dir=x".to_string()))
        );
    }
}
