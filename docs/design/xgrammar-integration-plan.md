# XGrammar Integration Plan for Atlas

Date: 2026-03-16
Status: Design phase

## Problem Statement

Atlas currently enforces `tool_choice="required"` by suppressing EOS tokens until a `<tool_call>` token is generated (single-token detection). This has two critical failures:

1. **Silent fallback**: If `<tool_call>` is not a single token in the tokenizer vocabulary (e.g., it's tokenized as `<tool` + `_call` + `>`), the suppression never activates and the model generates plain text instead of tool calls.
2. **No structural enforcement**: Even when the `<tool_call>` token IS generated, there's no guarantee the content between `<tool_call>` and `</tool_call>` is valid JSON/XML. The model can produce malformed tool calls that fail at parse time.

XGrammar-based constrained decoding solves both: it builds a token bitmask at each decode step that physically prevents the model from generating invalid tokens, guaranteeing structurally valid output.

## How vLLM Does It (Reference Architecture)

### Core Components (from vLLM v0.8.5+)

1. **StructuredOutputManager** (`vllm/v1/structured_output/__init__.py`)
   - Per-request grammar state management
   - Handles `response_format` (JSON mode) and `tool_choice` (tool call mode)
   - Manages grammar caching and compilation

2. **XGrammarBackend** (`vllm/v1/structured_output/backend_xgrammar.py`)
   - Wraps XGrammar's `GrammarCompiler` and `GrammarMatcher`
   - Compiles grammar once per unique schema (cached)
   - Per-token: `fill_bitmask()` → apply to logits → `accept_tokens()`

3. **GPU-side bitmask application** (`vllm/v1/worker/gpu/structured_outputs.py`)
   - `apply_token_bitmask_inplace()` CUDA kernel
   - Fused with logit post-processing, negligible overhead (<0.05ms)

### vLLM Tool Calling Flow

```
Request arrives with tool_choice="required" + tool definitions
    ↓
StructuredOutputManager compiles grammar for tool call format:
  - Hermes format: <tool_call>{"name":"fn","arguments":{...}}</tool_call>
  - Qwen format: <tool_call><function=fn>...</function></tool_call>
  Grammar includes JSON schema for arguments from tool definitions
    ↓
Each decode step:
  1. GrammarMatcher.fill_bitmask(bitmask, batch_idx)
     - Sets invalid tokens to 0 (masked)
     - EOS masked until grammar reaches "accepted" state
  2. apply_token_bitmask_inplace(logits, bitmask)
     - GPU kernel: masked tokens → -inf
  3. Normal sampling on masked logits
  4. GrammarMatcher.accept_tokens(request_id, [sampled_token])
     - Advances PDA state
  5. If grammar.is_terminated() → allow EOS
    ↓
Result: Structurally guaranteed valid tool call
```

### vLLM Performance

- Grammar compilation: 100-500ms (cached per unique schema)
- Per-token bitmask generation: 0.1-1ms (CPU PDA traversal)
- GPU kernel: <0.05ms (fused with logits)
- Total overhead: <1% of inference time
- Speculative decode: supported via `max_rollback_tokens` parameter

## How SGLang Does It

SGLang uses a similar XGrammar approach but with:
- **EBNF Composer** for tool calling (generates EBNF grammars from JSON schema)
- **RadixAttention-based caching** (prefix tree for compiled grammars)
- **Overlapped execution**: CPU grammar work happens while GPU runs the next token
- Supports Hermes, JSON, Pythonic, and XML tool call formats

## Atlas Integration Design

### Architecture

**Use `xgrammar-rs` (crates.io v0.1.31)** — pure Rust bindings to XGrammar. No FFI wrapper needed.

Source: https://github.com/trymirai/xgrammar-rs
Crate: `xgrammar-rs = "0.1.31"` (with `hf` feature for HuggingFace tokenizer)

Key API:
- `Grammar::from_json_schema(schema, ...) → Grammar`
- `Grammar::from_ebnf(ebnf, "root") → Grammar`
- `GrammarCompiler::new(tokenizer_info, max_threads) → GrammarCompiler`
- `compiler.compile_grammar(&grammar) → CompiledGrammar`
- `GrammarMatcher::new(&compiled, ...) → GrammarMatcher`
- `matcher.fill_next_token_bitmask(&mut bitmask)` — get allowed token bitmask
- `matcher.accept_token(token_id) → bool` — advance state
- `matcher.is_terminated() → bool` — grammar complete
- `matcher.rollback(n)` — rewind for speculative decode
- `TokenizerInfo::from_huggingface(&tokenizer, None, None)` — HF tokenizer integration

### Integration Points in Atlas

1. **Grammar compilation** (at request parse time in `api.rs`):
   - When `tool_choice="required"` or `response_format.type="json_schema"`:
     - Build grammar from tool definitions + JSON schemas
     - Cache by (tool_set_hash, format_type)
   - Pass compiled grammar handle to `InferenceRequest`

2. **Bitmask generation** (in `scheduler.rs` decode loop):
   - After sampling logits are on host (D2H already happens):
     - Call `GrammarMatcher::FillNextTokenBitmask(bitmask)`
     - Apply bitmask to FP32 logits (set masked → -inf)
     - Then proceed with normal sampling
   - After token is sampled:
     - Call `GrammarMatcher::AcceptToken(sampled_token_id)`

3. **MTP speculative decode integration** (in `model.rs` verify path):
   - Before accepting verified tokens:
     - Check each token against grammar state
     - If grammar rejects a verified token, truncate the batch there
   - On rejection, `GrammarMatcher::Rollback(rejected_count)`

4. **EOS handling** (replaces current `require_tool_call` flag):
   - When grammar is active, EOS suppression is handled BY the grammar
   - No more single-token `<tool_call>` detection
   - Grammar naturally masks EOS until the full structure is complete
   - Remove the `require_tool_call` and `tool_call_start_token` fields

### File Touch Map

| File | Change |
|------|--------|
| `Cargo.toml` (workspace) | Add `xgrammar-rs = { version = "0.1", features = ["hf"] }` |
| `crates/spark-server/Cargo.toml` | Add xgrammar-rs dependency |
| `crates/spark-server/src/api.rs` | Grammar compilation at request parse time |
| `crates/spark-server/src/scheduler.rs` | Bitmask application in decode loop |
| `crates/spark-model/src/model.rs` | MTP verify integration |
| `crates/spark-runtime/src/sampler.rs` | Bitmask application before sampling |

### Implementation Order

**Phase 1: Basic JSON schema enforcement (no tool calling)**
1. Add `xgrammar-rs` dependency to `spark-server`
2. Initialize `TokenizerInfo::from_huggingface()` at server startup
3. Add grammar compilation for `response_format.type="json_schema"`
4. Apply bitmask in sampler (CPU-side, after D2H)
5. Test with simple JSON schemas

**Phase 2: Tool call enforcement**
1. Build grammar compiler for Hermes `<tool_call>` format
2. Build grammar compiler for Qwen3 `<tool_call><function=...>` format
3. Replace `require_tool_call` with grammar-based EOS suppression
4. Test with all tool calling test cases

**Phase 3: MTP integration**
1. Add grammar state checkpointing for speculative decode
2. Verify grammar rollback works with MTP K=2/3/4
3. Benchmark overhead per token

**Phase 4: Performance optimization**
1. Cache compiled grammars by tool set hash
2. Overlap CPU bitmask generation with GPU compute
3. Consider GPU-side bitmask application kernel

### Grammar Definitions

**Hermes tool call format** (JSON):
```ebnf
root ::= "<tool_call>" ws tool_call_body ws "</tool_call>"
tool_call_body ::= "{" ws "\"name\"" ws ":" ws string ws "," ws "\"arguments\"" ws ":" ws arguments_object ws "}"
arguments_object ::= <compiled from tool's JSON schema>
```

**Qwen3 Coder tool call format** (XML):
```ebnf
root ::= "<tool_call>" function_call "</tool_call>"
function_call ::= "<function=" function_name ">" parameters "</function>"
function_name ::= <one of: allowed function names>
parameters ::= (parameter)*
parameter ::= "<parameter=" param_name ">" param_value "</parameter>"
```

### Dependencies

- **XGrammar library**: `xgrammar` Python package includes `libxgrammar.a` (C++ static lib)
  - Located at: `/workspace/.local/lib/python3.12/site-packages/xgrammar/lib/libxgrammar.a`
  - C++ headers needed for FFI binding generation
- **Tokenizer info**: XGrammar needs the tokenizer's vocabulary to build token-level bitmasks
  - Atlas already has the HuggingFace tokenizer loaded in `tokenizer.rs`
  - XGrammar's `TokenizerInfo::FromHuggingFace(vocab, stop_tokens)` accepts the vocabulary

### Risk Assessment

| Risk | Mitigation |
|------|------------|
| FFI complexity | Start with minimal C API surface (4 functions) |
| Tokenizer mismatch | Use same HF tokenizer Atlas already loads |
| Speculative decode conflict | Implement rollback before enabling with MTP |
| Performance overhead | CPU bitmask is <1ms/token per vLLM measurements |
| Grammar compilation latency | Cache by tool set hash (same tools = same grammar) |

### References

- XGrammar paper: https://arxiv.org/abs/2411.15100
- XGrammar structural tags: https://xgrammar.mlc.ai/docs/tutorials/structural_tag.html
- vLLM structured outputs: https://docs.vllm.ai/en/latest/features/structured_outputs/
- vLLM tool calling: https://docs.vllm.ai/en/latest/features/tool_calling/
- SGLang function calling: https://docs.sglang.ai/advanced_features/function_calling.html
- SGLang EBNF composer: `sglang/srt/function_call/ebnf_composer.py`
- vLLM XGrammar backend: `vllm/v1/structured_output/backend_xgrammar.py`

### Key Insight from Research

The XGrammar library already exists on this machine at `/workspace/.local/lib/python3.12/site-packages/xgrammar/`. It includes:
- `libxgrammar.a` (50MB static library)
- Python bindings showing the exact C++ API surface
- CUDA kernel for GPU-side bitmask application

Atlas can FFI into this directly rather than reimplementing. The API surface is small: compile, fill_bitmask, accept_token, rollback, is_terminated.
