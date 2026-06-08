# Research C1/agent4 — Whitespace-Mask Scan for Qwen3.6 FP8 Drift

**Date:** 2026-05-26
**Tokenizer:** `Qwen/Qwen3.6-35B-A3B` (snapshot `995ad96eacd98c81ed38be0c5b274b04031597b0`)
**Vocab size:** 248,070 (incl. added/special tokens)
**JSON output:** `qwen36_whitespace_tokens.json`
**Scan script:** `scan_ws_tokens.py`

---

## 1. Token-Group Summary

| Group | Definition | Count |
|------:|------------|------:|
| **G1** | Pure-whitespace tokens (only ` `, `\t`, `\n`, `\r`, `\x0b`, `\x0c`, U+00A0, U+3000) | **425** |
| **G2** | Short (`len ≤ 4`) leading-whitespace tokens with a **non-alphabetic** tail (punct / digit / bracket / quote) — i.e. tokens that survive `.trim()` and produce visible content | **1161** |
| **G3** | Single-digit-with-leading-space tokens (` 0`..` 9`) | **0** |

### G3 finding (negative result, important)

Qwen3.6 does **NOT** have atomic ` 0`..` 9` tokens. Every ` 0`..` 9` string tokenizes as `[220, digit]`:

```
' 0' -> [220, 15]
' 1' -> [220, 16]
...
' 9' -> [220, 24]
```

This explains the "split-digit" Tier A drift (`0.1 .0`): FP8 logit-margin noise on `220` (the bare space) flips an intended digit-continuation into a spurious space-then-digit sequence. **Token 220 alone is the entire G3 problem.**

---

## 2. Top 15 Most-Likely Drift Culprits Beyond the Existing 5-Token Mask

The current hardcoded mask is `{220, 198, 197, 256, 271}` (` `, `\n`, `\t`, `  `, `\n\n`). The next-tier culprits to add — ranked by how likely the sampler picks them in a low-margin parameter-body distribution under FP8 drift:

### Pure-whitespace neighbours (G1) — extend the param-body trim escape

| id | repr | rationale |
|---:|------|-----------|
| 199 | `'\x0b'` | vertical-tab; stripped by `.trim()` |
| 200 | `'\x0c'` | form-feed; stripped by `.trim()` |
| 201 | `'\r'` | carriage-return; stripped by `.trim()` |
| 257 | `'    '` (4 sp) | very common indent; sampler picks under FP8 noise |
| 262 | `'   '` (3 sp) | indent variant |
| 285 | `'       '` (7 sp) | indent variant |
| 297 | `'\t\t'` | double-tab indent |
| 317 | `'\r\n'` | windows newline; `.trim()` strips |
| 695 | `' \n'` | the "soft return" pair; both ws |
| 845 | `'\r\n\r\n'` | windows double-newline |
| 987 | `'\n\n\n\n'` | quad-newline |
| 1358 | `'\n\n\n'` | triple-newline |
| 1517 | `'\t\n'` | mixed ws |

### Short leading-ws non-alpha (G2) — `.trim()` survivors that emit ONE punct char

These don't get stripped, but they emit a non-content first char (often opening punct that confuses the parameter parser's expectation of meaningful first byte):

| id | repr | rationale |
|---:|------|-----------|
| 641 | `' .'` | **fired the `0.1 .0` Tier A drift** |
| 535 | `' :'` | colon-after-space confuses key/value boundary |
| 1116 | `' ,'` | comma-after-space ends args prematurely |
| 328 | `' "'` | quote injection inside an already-quoted body |
| 318 | `' ('` | spurious paren-open |
| 313 | `' {'` | spurious brace-open |

(Full list of all 1161 G2 ids is in `qwen36_whitespace_tokens.json` → `group2_short_leading_ws_nonalpha`.)

---

## 3. Atlas Hardcoded-Mask Sites (Both Locations)

There are exactly **two** call sites to update:

### Site A — logit-bias mask
**File:** `crates/spark-server/src/scheduler/decode_logits_seq.rs:454-463`
```rust
if a.inside_parameter_body && a.param_body_chars_emitted == 0 {
    // Close-tag opener `</`
    logit_bias_local.push((510u32, -8.0f32));
    // Common whitespace tokens
    logit_bias_local.push((220u32, -8.0f32)); // ` `
    logit_bias_local.push((198u32, -8.0f32)); // `\n`
    logit_bias_local.push((197u32, -8.0f32)); // `\t`
    logit_bias_local.push((256u32, -8.0f32)); // `  `
    logit_bias_local.push((271u32, -8.0f32)); // `\n\n`
}
```
**What it does:** While inside `<parameter=KEY>…</parameter>` body AND zero content tokens have been emitted, applies a `-8.0` logit bias to the close-tag-opener `</` and the 5 most common ws tokens. Forces the model to emit a non-ws content byte first.

The comment on lines 443-447 already acknowledges the gap:
> "The Qwen3.6 vocab has many multi-byte whitespace tokens beyond these 5, so this is not bulletproof — but it covers the empirically-most-likely tokens… A future Tier 1b would do a full vocab scan for whitespace-only tokens at boot."

### Site B — char-counter skip
**File:** `crates/spark-server/src/scheduler/emit_step.rs:163-166`
```rust
let is_whitespace_token = matches!(
    tok,
    220 | 198 | 197 | 256 | 271
);
if !is_whitespace_token {
    a.param_body_chars_emitted =
        a.param_body_chars_emitted.saturating_add(1);
}
```
**What it does:** When the model emits a token while inside the parameter body, increments `param_body_chars_emitted` ONLY IF the token isn't whitespace-only. The downstream parser's `.trim()` (in `tool_parser/parse_single_b.rs:105`) would strip pure-ws bytes to empty, so without this exclusion the model can "satisfy" the gate with whitespace then immediately emit `</parameter>`, yielding empty args.

**Confirmation:** No other matches for the `220, 198, 197, 256, 271` literal in `crates/`.

---

## 4. Tokenizer Load Path & Recommended Boot-Scan Site

Atlas already has a clean **OnceLock<Arc<[bool]>>** SSOT pattern for tokenizer-derived per-id masks:

- `crates/spark-server/src/scheduler/helpers.rs:323` — `NUMERIC_TOKEN_MASK`
- `crates/spark-server/src/scheduler/helpers.rs:343` — `BOUNDARY_TOKEN_MASK`
- `crates/spark-server/src/scheduler/helpers.rs:372` — `MID_WORD_TOKEN_MASK`

All three are populated by a single boot-time scan in:
**`crates/spark-server/src/main_modules/serve_phases/tokenizer_runtime.rs:102-187`**

That function is called from `serve.rs` after the tokenizer is loaded by `ChatTokenizer::from_model_dir` (`crates/spark-server/src/tokenizer/chat_impl.rs:18-66`). It loops over `0..vocab_size`, calls `tokenizer.decode_with_special(&[id])`, classifies, and stores `Arc<[bool]>` masks.

### Cleanest extension

Add a **fourth mask** in the same loop body — already iterating every id, already calling `decode_with_special` per id. Zero-extra-cost incremental work:

```rust
// In tokenizer_runtime.rs, alongside boundary_count / mid_word_count loop:
let mut ws_only_count = 0usize;
let mut ws_only_mask: Vec<bool> = vec![false; vocab_size];
let mut leading_ws_nonalpha_mask: Vec<bool> = vec![false; vocab_size];
for (id, _) in mask.iter().enumerate() {
    if let Ok(s) = tokenizer.decode_with_special(&[id as u32])
        && !s.is_empty()
    {
        if s.chars().all(|c| c.is_whitespace()) {
            ws_only_mask[id] = true;
            ws_only_count += 1;
        } else if s.chars().count() <= 4
            && s.starts_with(char::is_whitespace)
        {
            let tail = s.trim_start();
            if let Some(first) = tail.chars().next()
                && !first.is_alphabetic()
            {
                leading_ws_nonalpha_mask[id] = true;
            }
        }
    }
}
crate::scheduler::set_ws_only_token_mask(Arc::from(ws_only_mask));
crate::scheduler::set_leading_ws_nonalpha_mask(Arc::from(leading_ws_nonalpha_mask));
tracing::info!(
    "Whitespace-only mask: {ws_only_count}/{vocab_size} ids; \
     leading-ws-nonalpha: <separate count>; param-body guard active"
);
```

### Recommended public API

In `crates/spark-server/src/scheduler/helpers.rs` (alongside the existing three masks):

```rust
static WS_ONLY_TOKEN_MASK: OnceLock<Arc<[bool]>> = OnceLock::new();
static LEADING_WS_NONALPHA_MASK: OnceLock<Arc<[bool]>> = OnceLock::new();

pub fn set_ws_only_token_mask(m: Arc<[bool]>)        { let _ = WS_ONLY_TOKEN_MASK.set(m); }
pub fn set_leading_ws_nonalpha_mask(m: Arc<[bool]>)  { let _ = LEADING_WS_NONALPHA_MASK.set(m); }
pub fn ws_only_token_mask() -> Option<Arc<[bool]>>       { WS_ONLY_TOKEN_MASK.get().cloned() }
pub fn leading_ws_nonalpha_mask() -> Option<Arc<[bool]>> { LEADING_WS_NONALPHA_MASK.get().cloned() }
```

Re-export from `scheduler/mod.rs` next to the existing `set_mid_word_token_mask` line (`mod.rs:55-56`).

### Call-site rewrites

**Site A (decode_logits_seq.rs:454-463)** — replace the 5 hardcoded pushes with a mask-driven loop. To avoid bloating `logit_bias_local` with 425+ entries every token, prefer applying the bias **directly to the logits tensor** in the existing logits-processing pipeline (the same way mid-word `</think>` suppression already works via `mid_word_token_mask`). The mask is `Arc<[bool]>` so it can be moved into the per-step env cheaply.

**Site B (emit_step.rs:163-170)** — replace `matches!(tok, 220 | 198 | 197 | 256 | 271)` with:
```rust
let is_whitespace_token = crate::scheduler::ws_only_token_mask()
    .map(|m| m.get(tok as usize).copied().unwrap_or(false))
    .unwrap_or(false);
```
Fail-open if the mask hasn't been initialized (boot race / unit-test path).

### Why this is the right shape

- **SSOT:** the mask is derived from the tokenizer, not hardcoded vocab IDs. Works for ANY model loaded by Atlas (not just Qwen3.6), so the same code keeps working for Qwen-Coder, MiniMax M2.7, Mistral 4, etc.
- **PCND:** fail-open is explicit (`Option::None ⇒ skip suppression`) — no implicit default vocab IDs.
- **SBIO:** zero I/O in the hot path; `Arc<[bool]>` is cloned by ref-count, indexed by `id as usize` in O(1).
- **Cost:** scan adds ~248k decode calls at boot (= a few ms — already paid 3× for the existing masks).

---

## 5. Open Question — Group 2 Scope

The G2 set is large (1161 ids). Two reasonable choices:

1. **Tight (safe):** Mask only G1 (425 ids). Catches every `.trim()` escape. The G2 tokens emit visible non-ws content so the parameter parser doesn't strip them — but they DO emit punct-first which can still confuse JSON-shape parsing under XGrammar.
2. **Broad (aggressive):** Mask G1 + G2-non-alpha-tail-of-length-1 (e.g. ` .`, ` :`, ` ,`, ` "`, ` (`, ` {`, ` <`, ` =`, ` *`, ` /`, ` @`, ` #`, ` !`, ` ?`, ` -`, ` +`, ` &`, ` |`, ` ~`, ` $`, ` %`, ` \`, ` _`, ` `, ` ;`, ` >`, ` )`, ` ]`, ` }` — ~30 ids). Still fail-open, but catches the specific Tier A drift fire (`641 ' .'`).

**Recommendation:** ship G1 first (425-id mask, drop-in replacement for the 5-token mask). Add a separate, narrower G2-singleton-punct mask (the ~30 above) in a second pass with its own opt-in env flag, since some legitimate JSON args genuinely start with ` "` or ` {`.

---

## 6. Deliverables Index

| Artifact | Path |
|----------|------|
| JSON token tables | `/workspace/atlas-mtp/bench/fp8_dgx2_drift/qwen36_whitespace_tokens.json` |
| Scan script (reproducible) | `/workspace/atlas-mtp/bench/fp8_dgx2_drift/scan_ws_tokens.py` |
| This report | `/workspace/atlas-mtp/bench/fp8_dgx2_drift/research_C1agent4_ws_mask_scan.md` |
| Site A (logit bias) | `/workspace/atlas-mtp/crates/spark-server/src/scheduler/decode_logits_seq.rs:454-463` |
| Site B (char counter) | `/workspace/atlas-mtp/crates/spark-server/src/scheduler/emit_step.rs:163-170` |
| Mask SSOT (helpers) | `/workspace/atlas-mtp/crates/spark-server/src/scheduler/helpers.rs:323-385` |
| Boot scan (extend here) | `/workspace/atlas-mtp/crates/spark-server/src/main_modules/serve_phases/tokenizer_runtime.rs:102-187` |
| Tokenizer load | `/workspace/atlas-mtp/crates/spark-server/src/tokenizer/chat_impl.rs:18-66` |
