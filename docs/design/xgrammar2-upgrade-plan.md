# XGrammar2 Upgrade Plan for Atlas

Date: 2026-04-01
Status: Research complete, ready for implementation decision

## Executive Summary

Atlas currently uses xgrammar-rs v0.1.31 (vendored), wrapping the xgrammar C++ library via
autocxx FFI. "XGrammar 2" is not a separate library -- it refers to architectural improvements
(paper: arXiv 2601.04426) that shipped incrementally in xgrammar v0.1.30-v0.1.33. The upgrade
path is: bump the vendored xgrammar-rs to v0.1.32+ (which tracks xgrammar C++ v0.1.33), gaining
6x faster compilation, cross-grammar caching, and Earley-based parsing. A pure Rust rewrite is
feasible but high-effort and not recommended as the primary path.

---

## 1. What Is "XGrammar 2"?

XGrammar 2 is **not a new library** -- it is a set of improvements to the existing mlc-ai/xgrammar
C++ library, described in the paper "XGrammar 2: Dynamic and Efficient Structured Generation
Engine for Agentic LLMs" (arXiv 2601.04426, March 2026). These improvements landed across
releases v0.1.30 through v0.1.33:

| Version | Key XGrammar2 Features |
|---------|----------------------|
| v0.1.30 | Cross-grammar cache (initial), TagDispatch, excluded strings for any_text |
| v0.1.31 | Reverted cross-grammar cache (Windows compat), bitmask fix -- **this is what Atlas has** |
| v0.1.32 | Cross-grammar cache restored, kRepeat moved from AST to FSM, AST fallback paths removed from Earley parser, builtin structural tag formats |
| v0.1.33 | Token-level grammar (Token/ExcludeToken/TokenTagDispatch edges), RepeatFormat, DispatchFormat, TokenDispatchFormat, structural tag-level cache, Fork() for GrammarMatcher, BatchRollback, IsCompleted |

### Key Technical Improvements

1. **TagDispatch**: First-class support for tag-triggered structure switching. Uses Aho-Corasick
   automata for efficient trigger matching. Atlas already uses structural tags with triggers; this
   makes them faster and more expressive.

2. **Cross-Grammar Cache**: Substructure-level cache reuse across different grammars. When
   different tool definitions share common JSON schema substructures, masks are reused. Provides
   50.7% reuse in dynamic tool-calling scenarios. Compilation drops from >1000ms to ~10ms.

3. **Earley-Based Parser**: Replaces v1's PDA (pushdown automaton) with Earley parsing. Handles
   non-deterministic grammars without exponential state growth. Caches only scannable states,
   reducing memory.

4. **JIT Compilation**: Defers mask cache generation to runtime on first state visit. Overlaps
   preprocessing with LLM prefilling. Eliminates upfront compilation stall.

5. **Repetition State Compression**: Bounds grammar size for large repetition ranges (MinLength,
   MaxLength, MinItems, MaxItems), preventing compilation blowup on complex schemas.

### Performance Numbers (from paper)

| Metric | XGrammar v1 | XGrammar 2 |
|--------|-------------|------------|
| Compilation time | >1000ms | ~10ms (100x reduction) |
| Per-token mask generation | ~45-126 us | ~45-126 us (same) |
| End-to-end overhead | ~30-40% | <6% |
| Token throughput (Llama 3.1-8B, batch 128) | 738 tok/s | 1938 tok/s |

---

## 2. Current Atlas XGrammar Integration

### What Atlas Uses (grammar.rs)

Atlas's `grammar.rs` uses these xgrammar-rs APIs:

**Types imported:**
- `CompiledGrammar`, `Grammar`, `GrammarCompiler`, `GrammarMatcher`
- `StructuralTagItem`, `TokenizerInfo`, `VocabType`
- `DLTensor`, `DLDataType`, `DLDataTypeCode`, `DLDevice`, `DLDeviceType`
- `allocate_token_bitmask`, `get_bitmask_shape`, `reset_token_bitmask`

**GrammarEngine (server-wide, initialized once):**
- `TokenizerInfo::new(vocab, VocabType::RAW, stop_tokens, false)`
- `GrammarCompiler::new(&tokenizer_info, 1, true, -1)` -- 1 thread, cache on, no limit
- `Grammar::from_structural_tag(&json_string)` -- for tool call grammars
- `compiler.compile_grammar(&grammar)` -- compile structural tag grammars
- `compiler.compile_json_schema(schema, ...)` -- JSON schema enforcement
- `compiler.compile_builtin_json_grammar()` -- any-JSON mode
- `compiler.compile_grammar_from_ebnf(ebnf, root)` -- EBNF mode

**GrammarState (per-request):**
- `GrammarMatcher::new(compiled, None, false, -1)` -- unlimited rollback
- `matcher.fill_next_token_bitmask(&mut tensor, 0, false)` -- per-token mask
- `matcher.accept_token(id)` -- advance state
- `matcher.is_terminated()` -- check completion
- `matcher.rollback(n)` -- MTP speculative decode rollback
- `matcher.reset()` -- reset to initial

**Bitmask handling (manual DLTensor construction):**
- `allocate_token_bitmask(1, vocab_size)` -> `Box<[i32]>`
- Manual `DLTensor` struct construction with shape/stride arrays
- `reset_token_bitmask(&mut data)` -- fill with -1 (all bits set)
- Manual bit-level bitmask application to f32 logits

**Schema sanitization (pure Rust, ~200 lines):**
- `sanitize_schema_for_grammar()` -- prevents C++ crashes from edge-case schemas
- `enforce_min_length_on_required_strings()` -- prevents empty string parameters
- `resolve_local_ref()` -- $ref resolution
- Handles: empty enum, empty anyOf/oneOf, allOf merging, empty object properties

### C++ FFI Layer (vendor/xgrammar-rs)

- **Build system**: 1130-line `build.rs` using cmake + autocxx
- **C++ wrappers**: ~1100 lines of `.hpp` headers wrapping xgrammar C++ with try-catch
- **Rust wrappers**: ~500 lines of Rust FFI code
- **Upstream C++**: ~37 source files (earley_parser.cc, grammar_matcher.cc, json_schema_converter.cc, etc.)
- **Dependencies**: autocxx 0.30, cxx 1.0, cmake

### Known FFI Pain Points

1. **Process termination on C++ exceptions**: xgrammar's `LogFatalError` can terminate the process
   through FFI. The `sanitize_schema_for_grammar()` function exists specifically to prevent schemas
   that trigger these crashes.
2. **`unsafe impl Send`**: GrammarEngine has a manual `unsafe impl Send` because C++ raw pointers
   don't auto-impl Send.
3. **DLTensor ceremony**: Manual construction of DLTensor with raw pointers for shape/stride arrays
   that must outlive the tensor.
4. **Build complexity**: 1130-line build.rs, cmake dependency, C++ compiler requirement.
5. **No error recovery**: C++ exceptions in `fill_next_token_bitmask` or `accept_token` can panic
   the Rust side.

---

## 3. Upgrade Path: xgrammar-rs v0.1.31 -> v0.1.32+

### What Changes in the C++ API (v0.1.31 -> v0.1.33)

**New methods (non-breaking additions):**
- `GrammarMatcher::Fork()` -- clone matcher state (useful for speculative decode)
- `GrammarMatcher::IsCompleted()` -- distinguish "completed grammar" from "terminated with stop token"
- `BatchGrammarMatcher::BatchRollback()` -- batch rollback support
- `GrammarCompiler::CompileStructuralTag()` -- direct structural tag compilation (no Grammar intermediate)
- `GrammarCompiler::CompileRegex()` -- regex compilation
- `GrammarCompiler::ClearCache()`, `GetCacheSizeBytes()`, `CacheLimitBytes()` -- cache management

**Internal changes (no API breakage):**
- Earley parser replaces PDA internals (transparent to API users)
- Cross-grammar cache operates automatically inside GrammarCompiler
- JIT compilation is automatic when cache is enabled
- Repetition compression is automatic for schema compilation

**Structural tag format changes:**
- New tag types: `PlusFormat`, `OptionalFormat`, `StarFormat`
- New dispatch: `TokenFormat`, `ExcludeTokenFormat`, `AnyTokensFormat`, `TokenTriggeredTagsFormat`
- Simplified TagDispatch: removed `stop_eos` and `stop_str` parameters
- These are JSON-level changes to the structural tag specification, not C++ API changes

**Breaking changes:**
- `TagDispatch` simplified by removing `stop_eos`/`stop_str` -- Atlas does not use these
- `kRepeat` moved from AST to FSM -- internal, no API impact
- AST fallback paths removed from Earley parser -- internal, no API impact

### Changes Needed in Atlas

#### 3.1 Vendor Update

Update `vendor/xgrammar-rs` to the latest version from trymirai/xgrammar-rs (currently v0.1.32
on GitHub, which wraps xgrammar C++ v0.1.33 based on the git ref in build.rs).

**Steps:**
1. Clone the latest trymirai/xgrammar-rs
2. Replace `vendor/xgrammar-rs/` contents
3. Update git submodule ref in build.rs (currently `19a6893f...`)
4. Verify Atlas build succeeds

**Risk:** Low. The Rust API surface that Atlas uses is backward-compatible. The xgrammar-rs
maintainer (Eugene Bokhan, who also contributed to upstream xgrammar v0.1.32) tracks upstream
closely.

#### 3.2 grammar.rs Changes

**No changes required for basic upgrade.** All APIs Atlas uses exist in v0.1.33 with the same
signatures. The benefits (faster compilation, cross-grammar cache, Earley parser) are automatic.

**Optional improvements to leverage new features:**

1. **Use `Fork()` for MTP speculative decode** instead of rollback:
   ```rust
   // Before (current): rollback on rejection
   state.rollback(n);
   
   // After (optional): fork before speculation, discard on rejection
   let checkpoint = state.fork();
   // ... accept draft tokens ...
   // On rejection: restore from checkpoint
   ```

2. **Use `IsCompleted()`** to distinguish grammar completion from stop-token termination.

3. **Expose cache management** via `ClearCache()` / `GetCacheSizeBytes()` for monitoring.

4. **Use `CompileStructuralTag()`** on GrammarCompiler directly instead of going through
   `Grammar::from_structural_tag()` then `compiler.compile_grammar()`. Saves one indirection.

5. **Remove some sanitization workarounds** if upstream fixes handle the edge cases. Test whether
   empty-enum, empty-anyOf, etc. still crash in v0.1.33's Earley parser.

#### 3.3 Build System

The build.rs cmake approach remains the same. The only change is the upstream xgrammar git ref.
Build time may change slightly due to the larger Earley parser implementation.

---

## 4. Pure Rust Rewrite Evaluation

### Scope Assessment

The core xgrammar C++ implementation consists of ~37 source files covering:

| Component | Estimated Rust LOC | Complexity |
|-----------|--------------------|------------|
| Earley parser + state management | 3000-5000 | Very high |
| Grammar builder + EBNF parser | 2000-3000 | High |
| JSON schema -> EBNF converter | 2000-3000 | Medium-high |
| FSM builder + optimizer | 1500-2500 | High |
| Structural tag handler | 1000-1500 | Medium |
| Token bitmask generation + cache | 1500-2000 | High |
| Regex -> EBNF converter | 800-1200 | Medium |
| TokenizerInfo + vocab processing | 500-800 | Low-medium |
| Cross-grammar cache | 1000-1500 | High |
| **Total** | **~13000-20000** | **Very high** |

### What Atlas Actually Needs (Minimal Viable Subset)

Atlas uses only a fraction of xgrammar's capabilities:

1. **JSON schema -> token bitmask**: Convert a JSON schema to a grammar, compile it against a
   vocabulary, produce per-token bitmasks during decode.
2. **Structural tags with triggers**: Hermes `<tool_call>` and Qwen3 `<function=...>` formats.
3. **Accept/rollback**: Token acceptance with rollback for MTP speculative decode.
4. **EBNF compilation**: Raw EBNF string compilation (used minimally).

A minimal pure Rust implementation for just these features would still require:
- Earley parser or equivalent CFG parser (~3000 lines)
- JSON schema to grammar converter (~2000 lines)
- Token bitmask generation with vocabulary trie (~2000 lines)
- Structural tag dispatch with Aho-Corasick triggers (~1000 lines)
- Grammar representation + builder (~1500 lines)

**Estimated total: ~10000-12000 lines of Rust.**

### Existing Pure Rust Alternatives

**1. llguidance (guidance-ai/llguidance)**
- Pure Rust, v1.0.0 released June 2025, actively maintained (1525 commits, 724 stars)
- Earley parser + derivatives-based lexer
- ~50us per-token mask generation (comparable to xgrammar)
- JSON schema support, regex support, Lark-format grammars
- Integrated with vLLM and SGLang
- **Missing**: No structural tag support, no tool-calling format support, no Hermes/Qwen XML
  format. Would need Atlas-side grammar construction.
- **Missing**: No documented rollback/speculative decode API.
- **License**: Apache-2.0

**2. KBNF (Dan-wanna-M/kbnf)**
- Pure Rust, Earley recognizer with Leo optimization
- O(m*n) for LR(k) grammars
- **Missing**: No JSON schema support, no structural tags, no tool calling, no rollback
- Low adoption (58 stars, 2 forks), appears dormant
- **Not viable** for Atlas.

### Pure Rust: Pros and Cons

**Pros:**
- Eliminate C++ build dependency (cmake, C++ compiler, 1130-line build.rs)
- Proper Rust error handling (no process termination from C++ exceptions)
- `Send + Sync` naturally (no `unsafe impl Send`)
- No DLTensor ceremony (use native Rust slices)
- Eliminates schema sanitization workarounds (Rust panics are catchable)
- Faster iteration on grammar bugs (no C++/Rust boundary debugging)
- Smaller Docker images (no C++ toolchain needed at build time)

**Cons:**
- **Massive effort**: 10000-12000 lines of complex parser/automata code
- **Correctness risk**: Earley parser + bitmask generation is subtle; xgrammar has years of
  testing and production use in vLLM/SGLang/TensorRT-LLM
- **Maintenance burden**: Must track upstream xgrammar improvements manually
- **Performance risk**: xgrammar's C++ is heavily optimized; naive Rust may be slower
- **Timeline**: 4-8 weeks for a senior engineer, vs. 1-2 days for vendor bump

### Hybrid Approach: llguidance + Custom Structural Tags

Use llguidance (pure Rust) for JSON schema enforcement, and build a thin Rust layer for structural
tag dispatch (Hermes/Qwen format) on top:

**Effort**: ~2000-3000 lines of Rust (structural tag dispatch + format-specific grammar builders)
**Risk**: Medium -- llguidance API is stable but structural tag integration is novel
**Timeline**: 1-2 weeks

---

## 5. Recommended Approach

### Phase 1: Vendor Bump (immediate, 1-2 days)

**Action**: Update vendored xgrammar-rs from v0.1.31 to v0.1.32 (which wraps xgrammar C++ v0.1.33).

**Benefits gained automatically:**
- 6x faster grammar compilation via cross-grammar cache
- Earley parser (handles edge-case grammars that crashed the PDA)
- JIT compilation (eliminates compilation stalls on first request)
- Repetition state compression (handles complex schemas without blowup)
- `Fork()` for cleaner MTP speculative decode checkpointing
- `IsCompleted()` for better termination detection
- Structural tag-level caching

**Changes to grammar.rs**: None required. All existing APIs are backward-compatible.

**Optional grammar.rs improvements:**
- Replace `rollback()` with `Fork()` for MTP
- Add `IsCompleted()` alongside `is_terminated()`
- Expose cache size monitoring
- Test whether schema sanitization workarounds can be simplified

### Phase 2: Evaluate llguidance for Pure Rust Path (medium-term, 1-2 weeks research)

**Action**: Prototype llguidance integration as an alternative backend behind a trait.

```rust
pub trait GrammarBackend: Send {
    fn compile_json_schema(&mut self, schema: &str) -> Result<CompiledGrammarHandle, GrammarError>;
    fn compile_structural_tag(&mut self, spec: &str) -> Result<CompiledGrammarHandle, GrammarError>;
    fn new_matcher(&self, compiled: &CompiledGrammarHandle) -> Result<Box<dyn GrammarMatcherTrait>, GrammarError>;
}

pub trait GrammarMatcherTrait: Send {
    fn fill_bitmask(&mut self, bitmask: &mut [i32]) -> bool;
    fn accept_token(&mut self, token_id: u32) -> bool;
    fn is_terminated(&self) -> bool;
    fn rollback(&mut self, n: usize);
    fn reset(&mut self);
}
```

This allows switching between xgrammar (C++ FFI) and llguidance (pure Rust) backends. If
llguidance proves viable for Atlas's structural tag needs, it becomes the long-term replacement.

### Phase 3: Long-Term Pure Rust (deferred, only if Phase 2 succeeds)

If llguidance handles Atlas's needs well, migrate fully to pure Rust and drop the C++ dependency.
If not, stay on xgrammar-rs with periodic vendor bumps.

### NOT Recommended

- **Full pure Rust rewrite from scratch**: Too much effort (10K+ lines), too much correctness risk,
  for a component that works well via FFI.
- **KBNF**: Insufficient feature set, low adoption, appears unmaintained.
- **Staying on v0.1.31**: Leaves significant performance on the table (6x compilation speedup)
  and misses Earley parser fixes for edge-case grammars.

---

## 6. Detailed File Change Map

### Phase 1 (Vendor Bump)

| File | Change | Risk |
|------|--------|------|
| `vendor/xgrammar-rs/` | Replace with xgrammar-rs v0.1.32 from trymirai/xgrammar-rs | Low |
| `Cargo.toml` (workspace) | Update path dependency if needed | None |
| `Cargo.lock` | Regenerated automatically | None |
| `crates/spark-server/src/grammar.rs` | No changes required | None |

### Phase 1 Optional Improvements

| File | Change | Risk |
|------|--------|------|
| `crates/spark-server/src/grammar.rs` | Add `fork()` to GrammarState for MTP | Low |
| `crates/spark-server/src/grammar.rs` | Add `is_completed()` method | Low |
| `crates/spark-server/src/grammar.rs` | Test removal of some sanitization workarounds | Medium |
| `crates/spark-server/src/grammar.rs` | Add cache monitoring (size, clear) | Low |

### Phase 2 (Backend Trait)

| File | Change | Risk |
|------|--------|------|
| `crates/spark-server/src/grammar.rs` | Extract trait, wrap xgrammar as one impl | Medium |
| `crates/spark-server/src/grammar_llguidance.rs` | New file: llguidance backend | Medium |
| `crates/spark-server/Cargo.toml` | Add llguidance dependency (feature-gated) | Low |
| `Cargo.toml` (workspace) | Add llguidance to workspace deps | Low |

---

## 7. Error Handling Improvements (Pure Rust vs C++ FFI)

### Current C++ FFI Error Model

```
C++ exception thrown (e.g., LogFatalError)
  -> Caught by try-catch in cxx_utils/*.hpp
  -> Propagated as error string to Rust
  -> Returned as Result<T, String>

BUT: Some C++ paths call abort()/terminate() directly
  -> Kills the entire Atlas process
  -> No recovery possible
  -> This is why sanitize_schema_for_grammar() exists
```

### What Pure Rust Would Give

```
Rust panic (catchable with catch_unwind)
  -> Or better: proper Result<T, E> error types
  -> Graceful error propagation
  -> Per-request failure, not process death
  -> No need for defensive schema sanitization
```

### Specific Improvements

1. **Schema validation errors**: Return `Err(GrammarError::InvalidSchema(...))` instead of
   process termination.
2. **Grammar compilation failures**: Return typed errors with context (which rule failed, why).
3. **Bitmask generation**: Return `Result<bool, GrammarError>` instead of potentially panicking
   through FFI.
4. **Accept token**: Return detailed rejection reason (grammar state, expected tokens) instead
   of just `false`.
5. **Thread safety**: Natural `Send + Sync` without `unsafe impl Send`.

---

## 8. Performance Implications

### Vendor Bump (Phase 1)

| Operation | v0.1.31 (current) | v0.1.33 (after bump) |
|-----------|-------------------|---------------------|
| First compilation | >1000ms | ~10ms (100x faster with cross-grammar cache) |
| Subsequent compilations (same schema) | Cached | Cached (+ substructure reuse) |
| Per-token mask | ~45-126 us | ~45-126 us (same) |
| Memory per grammar | Proportional to schema | Bounded (repetition compression) |
| Startup latency | Full compilation upfront | JIT deferred (overlaps with prefill) |

### Pure Rust (Phase 3, if pursued)

| Concern | Expected Impact |
|---------|-----------------|
| Per-token mask latency | Comparable (~50us, see llguidance benchmarks) |
| Compilation time | Potentially slower without cross-grammar cache |
| Memory | Potentially lower (no C++ allocator overhead) |
| Build time | Significantly faster (no cmake, no C++ compilation) |
| Docker image size | Smaller (no C++ toolchain layer) |

---

## References

- XGrammar 2 paper: https://arxiv.org/abs/2601.04426
- XGrammar GitHub: https://github.com/mlc-ai/xgrammar
- xgrammar-rs (Rust bindings): https://github.com/trymirai/xgrammar-rs
- llguidance (pure Rust alternative): https://github.com/guidance-ai/llguidance
- KBNF (pure Rust, limited): https://github.com/Dan-wanna-M/kbnf
- xgrammar v0.1.33 release: https://github.com/mlc-ai/xgrammar/releases/tag/v0.1.33
- xgrammar v0.1.32 release: https://github.com/mlc-ai/xgrammar/releases/tag/v0.1.32
- Atlas existing integration plan: /workspace/atlas/tasks/xgrammar-integration-plan.md
- Atlas grammar.rs: /workspace/atlas/crates/spark-server/src/grammar.rs
- Atlas xgrammar-rs vendor: /workspace/atlas/vendor/xgrammar-rs/
