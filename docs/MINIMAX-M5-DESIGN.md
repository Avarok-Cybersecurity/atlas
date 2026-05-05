# MiniMax M2 — M5 Design: MTP K=3 with 3 Draft Modules

**Status**: Implementation-ready. Existing MTP infrastructure does most of
the work; main change is loading + dispatching three sets of MTP weights
instead of reusing one.
**Acceptance (from gameplan)**: same-output test vs K=1 baseline (must be
identical modulo temperature), ≥50 tok/s decode batch=1, ≥60% draft
acceptance rate on standard chat.

## The two flavors of MTP

Atlas today implements **autoregressive single-module MTP**, used by
Qwen3.5-35B / 122B:

- One `MtpHead` (one transformer layer + one LM head) is loaded once.
- `propose(num_drafts=K)` runs the MTP module K times sequentially.
- Each iteration feeds the *previous draft's hidden state* into the
  same module to predict the next token.

MiniMax M2 (and DeepSeek-V3 before it) uses **multi-module MTP**:

- Three (`num_mtp_modules: 3`) separate transformer layers (each with
  `mtp_transformer_layers: 1`), each with its own weights.
- Module 0 predicts position `t+1` from the target's hidden at `t`.
- Module 1 predicts position `t+2` from module 0's hidden at `t+1`.
- Module 2 predicts position `t+3` from module 1's hidden at `t+2`.
- Weights are loaded under separate prefixes (typical naming:
  `model.layers.{N + i}` where N = `num_hidden_layers` and i ∈ [0,
  num_mtp_modules) — the MTP modules are appended to the layer list
  in the checkpoint).

The verify side is unchanged: target model runs once over `K` draft
tokens, returns logits for each; accept the longest matching prefix.

## Atlas infrastructure that already covers K=3

- `DraftProposer` trait (`crates/spark-model/src/speculative.rs`) is
  K-agnostic. `propose()` takes a `num_drafts: usize` parameter.
- `MtpHead::propose` already loops `for i in 0..num_drafts`. Comment
  on `after_verify` (line 885) explicitly handles "K=3: drafted 2,
  accepted 0 → trim 2…".
- Verify loop in `spark-runtime` accepts arbitrary K. The Atlas K=2
  spec-decode commit added bench infra at K up to 3.

The trait API is the right shape; **only the proposer impl changes**.

## What M5 ships

### 1. New `MultiModuleMtpProposer` (replaces single MtpHead for MiniMax)

New struct in `crates/spark-model/src/layers/mtp_head.rs` (or new
`mtp_multi.rs` if prefer file split):

```rust
pub struct MultiModuleMtpHead {
    /// One MTP module per draft slot. `modules[i]` is invoked when
    /// `propose()` produces draft #i. Length = num_mtp_modules
    /// (3 for MiniMax M2).
    modules: Vec<MtpHead>,
}

impl DraftProposer for MultiModuleMtpHead {
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        // Per-module ProposerState — each module has its own KV cache.
        let states = self.modules.iter()
            .map(|m| m.alloc_state(gpu))
            .collect::<Result<Vec<_>>>()?;
        Ok(Box::new(MultiModuleMtpState { per_module: states, last_num_drafted: 0 }))
    }

    fn propose(
        &self,
        last_token: u32,
        target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
        draft_embed_target: Option<DevicePtr>,
    ) -> Result<Vec<u32>> {
        // Cap drafts at num_modules — caller can ask for fewer (e.g.
        // K=1 for non-spec baseline tests).
        let k = num_drafts.min(self.modules.len());
        // For each module, call its forward_one with the previous step's
        // hidden state. Identical loop shape to MtpHead::propose, but
        // dispatching to a different `MtpHead` each iteration.
        // ... (~30 lines, mirrors lines 831-864 of mtp_head.rs)
    }

    fn after_verify(...) {
        // Trim each module's KV cache by the same amount (drafted - accepted).
        // Modules don't share KV; each tracks its own seq_len.
    }
}
```

### 2. Loader change in `weight_loader/minimax.rs`

`load_mtp_weights` returns a `MtpWeights` enum variant carrying three
sets of MTP weights. Existing `MtpWeights` is a single set; need to
extend to enum:

```rust
pub enum MtpWeights {
    Single(SingleMtpWeights),                 // existing Qwen3.5 path
    MultiModule(Vec<SingleMtpWeights>),       // new MiniMax path
}
```

The factory then constructs either `MtpHead` or `MultiModuleMtpHead`
based on `MtpWeights` variant. Existing call sites pattern-match.

Weight prefix mapping (from inspecting MiniMax checkpoints; verify on
the tiny-random model where `num_mtp_modules=3, num_hidden_layers=2`):

```
model.layers.2.{...}    → MTP module 0
model.layers.3.{...}    → MTP module 1
model.layers.4.{...}    → MTP module 2
```

(MTP modules are appended to the layer list at indices `[N, N+M)` where
N = num_hidden_layers, M = num_mtp_modules. Modeling code in
`modeling_minimax_m2.MiniMaxM2ForCausalLM` confirms this layout.)

Each module includes:
- `embed_tokens.weight` — shared with main model? **OPEN QUESTION 1**
  in gameplan §7. Inspect tiny checkpoint to confirm.
- `enorm.weight` (RMSNorm on the embedding before concat)
- `hnorm.weight` (RMSNorm on the previous hidden before concat)
- `eh_proj.weight` (projection from `[2 * hidden_size]` concat down to
  `[hidden_size]`)
- `self_attn.{q_proj, k_proj, v_proj, o_proj, q_norm, k_norm}` —
  same shape as a regular attention layer
- `block_sparse_moe.{gate, e_score_correction_bias, experts}` —
  same shape as a regular MoE layer
- `input_layernorm.weight`, `post_attention_layernorm.weight`
- `norm.weight` (final RMSNorm before the LM head)

LM head: shared with main model (MiniMax `tie_word_embeddings: false`,
so there's a separate `lm_head.weight` in the checkpoint, but it's
shared across all MTP modules — they just pick from the same vocab).

### 3. `MODEL.toml` declaration

Add to `kernels/gb10/minimax-m2-229b/MODEL.toml`:

```toml
mtp_layers = 3
mtp_transformer_layers = 1
mtp_type = "multi_module"   # vs "single" for existing Qwen3.5 MTP
```

The `mtp_type` field would gate proposer construction in the factory.

### 4. CLI / runtime hookup

`spark-server` `--num-drafts` flag already takes any K. Default for
MiniMax should be 3 (matches `num_mtp_modules`). Wire from MODEL.toml
`[behavior].default_num_drafts = 3`.

## Why this matters for throughput

MiniMax targets ~50 tok/s decode at batch=1 per the gameplan. Without
MTP at K=1, expect ~28-30 tok/s (similar to Qwen3.5-122B no-MTP). With
3-module MTP at 60% accept:
- Effective tokens per step = 1 + 0.6 × 2 = 2.2 (one verified, plus
  expected accepted drafts)
- 30 tok/s × 2.2 / 1.0 = 66 tok/s

That comfortably beats the 50 tok/s acceptance bar. 60% accept is
plausible — DeepSeek-V3 reports ~80% on chat; MiniMax should be
similar since the same architecture pattern.

## Validation plan

1. **Unit — three modules constructable**: Load the tiny-random
   checkpoint, build three `MtpHead` instances from the per-module
   weights, assert `modules.len() == 3` and each module's vocab matches
   `config.vocab_size`.

2. **Same-output as K=1 baseline**: Run `serve` with `--num-drafts 0`
   (no spec) and capture greedy output for 5 prompts. Run with
   `--num-drafts 3` and capture greedy output. They MUST be identical
   modulo temperature (proves MTP isn't drifting from target).

3. **Acceptance-rate metric**: Add a counter that increments per
   accepted draft and per draft attempted. Log accept ratio every 100
   decode steps. Acceptance criteria: ≥60% on standard chat prompts.

4. **End-to-end tok/s**: Reuse `bench_isl_osl.py prefill_1k` config.
   Target ≥50 tok/s decode at batch=1.

## Estimated effort

- `MultiModuleMtpHead` impl: 0.5 day (mostly delegation to existing
  `MtpHead::forward_one`).
- Weight loader + `MtpWeights` enum extension: 0.5 day.
- Factory dispatch + MODEL.toml field: 0.25 day.
- Tests + acceptance bench: 0.5 day.
- **Total: 1.5 days.**

## Dependencies

- **M1**: landed.
- **M2**: needed (each MTP module has its own attention with partial
  RoPE + per-layer qk_norm — same wiring as the main 62 layers).
- **M3**: needed (each MTP module has its own MoE layer with sigmoid
  routing).
- **M4**: needed for full FP8 path. M5 can ship with BF16 first if
  validating against tiny-random.

## Files touched (estimate)

- `crates/spark-model/src/layers/mtp_head.rs` (+150 lines, new struct
  + impl, plus factor `forward_one` helper if not already separable)
- `crates/spark-model/src/weight_map.rs` (+30 lines, MtpWeights enum
  variant + load helper)
- `crates/spark-model/src/weight_loader/minimax.rs` (+50 lines for
  `load_mtp_weights`)
- `crates/spark-model/src/factory.rs` (+10 lines for proposer
  construction switch)
- `kernels/gb10/minimax-m2-229b/MODEL.toml` (already has `mtp_layers
  = 3`; add `mtp_type` field)
- Tests: `tests/integration/minimax_mtp.rs` (+150 lines)

Net additive. Single existing call site needs to switch from the
direct `MtpHead` constructor to a factory function that returns the
correct proposer type — surgical change.

## Open questions — with findings

### OPEN QUESTION 1 (from gameplan §7): do MTP modules share the embedding?

**Finding from tiny-random checkpoint inspection (2026-04-13):**

The `yujiepan/minimax-m2.7-tiny-random` checkpoint **ships no MTP
module weights at all** despite `use_mtp: true, num_mtp_modules: 3,
mtp_transformer_layers: 1` in its config. Only layers 0 and 1 are
present (matching `num_hidden_layers: 2`), then straight to
`model.norm.weight` and `lm_head.weight`. Expected MTP layer indices
2, 3, 4 are absent.

Full weight key list on tiny-random:
```
lm_head.weight
model.embed_tokens.weight
model.layers.0.{block_sparse_moe.{e_score_correction_bias, gate.weight, experts.0..255.{w1,w2,w3}.weight},
                input_layernorm.weight, post_attention_layernorm.weight,
                self_attn.{q_proj, k_proj, v_proj, o_proj, q_norm, k_norm}.weight}
model.layers.1.{... same structure ...}
model.norm.weight
```

Implication: the tiny-random is useful for M1-M4 validation (config
dispatch, attention shape, MoE routing, FP8) but **cannot validate
M5**. We need either:
1. The full 229B checkpoint (500GB+, impractical for this env).
2. A different tiny variant that ships MTP weights — request from the
   community or generate ourselves using a reduced-experts
   MiniMaxM2ForCausalLM init with MTP enabled.
3. Unit-level validation only: construct three `MtpHead` instances
   from randomly-initialized weights, verify the dispatcher loop
   wires correctly without regression. Defer end-to-end byte-exact
   validation until actual weights are stageable.

**Recommended path**: do option 3 for the code change, document that
M5 is structurally complete but lacks weight-level validation, and
flag to the team that full-weight validation needs a GPU session
when 229B is downloaded. This matches the same honesty constraint
discussed for M2.

### OPEN QUESTION 2 (from gameplan §7): correction bias static or dynamic?

**Finding:** the tiny-random has
`model.layers.{i}.block_sparse_moe.e_score_correction_bias` as a
**loaded tensor** of shape `[num_experts]`. No training-time update
hook in `modeling_minimax_m2.py` touches it at forward time. Confirms
it's static — just a weight tensor, same as DeepSeek-V3's
loss-free-balancing bias. M3 loader can treat it as a plain
`[num_experts]` f32 (or bf16 → f32 cast) tensor per layer.
