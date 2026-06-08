# Forensic Analysis — `atlas_turn11_probe.json`

**Probe**: `/workspace/atlas-dumps/numdrift/atlas_turn11_probe.json`
**Model**: `Qwen/Qwen3.6-35B-A3B-FP8` (Atlas FP8 native serving)
**Sampling**: `temperature=0.3, top_p=1, max_tokens=4` (note: 4 is the probe replay
budget, not the original generation cap), `tool_choice=auto`
**Turn count**: 14 assistant turns over 30 messages
**User task**: "create a pure rust Axum project inside `./test-rust-axum-v3`,
ping/pong endpoint, tests pass, curl proves it works, tear down server."

This report walks every assistant message, identifies drift points, classifies
the failure mode, then ranks observed patterns against `MISSION_PROGRESS.md`.

---

## Per-turn walkthrough

### Turn [2] — Healthy parallel-probe opening
- **Reasoning (233 ch)**: "create a Rust Axum project with a ping/pong endpoint…
  Let me check if Rust is installed and create the…" *(truncated)*
- **Action**: 5 parallel `bash` calls — `which cargo`, `ls
  /home/nologik/test-rust-axum-v3`, `which curl`, `which jq`, `ss -tlnp | grep
  3000`. **Path is correct.** Description fields are sensible. This is the only
  fully healthy turn in the probe.

### Turn [8] — First near-duplicate symptom, no failure yet
- Reasoning: `"Port 3000 is in use.Let me useport 3001 instead. Let mecreate the"`
  *(truncated; note missing spaces — a known FP8 tokenizer-side artifact, not the
  drift bug)*.
- Action: `bash mkdir -p /home/nologik/test-rust-axum-v3/src`. **Path still
  correct.** Tool returns "(no output)" — success.

### Turn [10] — REPEAT of turn [8]; first tool-arg malformation
- Reasoning: `"Port 3000 is in use,soI'll use port 3001 for the server. Let
  mecreate the Axum"` — paraphrased near-duplicate of [8].
- Content: `"\n\nPort 3000 is occupied, so I'll use port 3001. Let me create
  the Axum v3 project files.\n\n"` — paraphrase of [8]'s content.
- **Critical**: the `write` tool's `filePath` argument contains an embedded
  XML closer + the actual Cargo.toml body:

  ```json
  "filePath":"/home/nologik/test-rust-axum-v3/Cargo.toml\n</parametercontent>\n[package]\nname = \"test-raxum-v3\"\n…\n</parametercontent>"
  ```

  `content` is empty (`""`). The model emitted the body **into the `filePath`
  field** and added two fake `</parametercontent>` sentinels around it.
- **Tool result [11]**: `"Unknown: FileSystem.makeDirectory
  (/home/nologik/test-rust-axum-v3/Cargo.toml\n</parametercontent>\n…"` — i.e.
  the entire malformed string was interpreted as a path and the FS layer tried
  to **mkdir** it, creating a directory literally named `Cargo.toml` plus
  weird siblings.

**This is the canonical "qwen_xml leaking into JSON" symptom** described in
`MISSION_PROGRESS.md` Epoch 3. The model is emitting the qwen_xml tool format
(`<parameter=name>value</parameter>`) and the harness is wrapping that in JSON
without splitting the params, so `</parametercontent>` (the leaked qwen_xml
closer text the model intended) ends up inside the JSON string value.
Reasoning is *coherent for this position* (it correctly identifies the goal:
"create the Cargo.toml") but the action stream is structurally broken.

### Turn [12] — Reasoning correctly diagnoses [10], action introduces NEW drift
- Reasoning: `"The write tool failed because the content was malformed - I had
  '<parametercontent>' instead of proper formatting. Let me fix this and write
  the"` *(truncated)*.
- The reasoning is **correct and self-aware** — the model knows what went wrong.
- Action: `write` with proper `content` field this time, but `filePath` is
  `/home/nologik/test-rustaxum-v3/Cargo.toml` — **the hyphen between `rust`
  and `axum` was dropped**. The path-token drift is the actual numerical bug:
  this is the FP8 MoE routing / KV-decode-side error that
  `MISSION_PROGRESS.md` epoch 1-4 chronicles — a single byte deleted from a
  freshly-decoded path string.
- Tool [13] returns `"Wrote file successfully."` — the wrong-path write
  silently succeeds. **The model has now scattered files across two
  directories** and has no way to detect this from the tool result alone.

This is the first **reasoning-correct, action-wrong** disconnect.

### Turn [14] — Reasoning catches the hyphen drift
- Reasoning: `"The file was written to /home/nologik/test-rustaxum-v3/Cargo.toml
  instead of /home/nologik/test-rust-axum-v3/Cargo.toml. Let me check what
  hap"` *(truncated)*. The model **did** notice the missing hyphen.
- Action: `bash ls /home/nologik/test-rust-axum-v3/ && ls
  /home/nologik/test-rustaxum-v3/ 2>/dev/null || echo "test-rustaxum-v3 not
  found"`. Reasonable.
- Tool [15] returns:
  ```
  Cargo.toml
  <
  src
  Cargo.toml
  ```
  i.e. **both** directories contain `Cargo.toml` artifacts. The standalone
  `<` line is a clue that the original [10] mkdir created a literal `<`
  filename inside `test-rust-axum-v3/`. The model's diagnosis of "broken
  Cargo.toml" is correct; the underlying bug is that [10] mkdir'd a path
  containing `\n</parametercontent>\n…`, splatting many artifact paths.

### Turn [16] — Reasoning correct; rm/mv strategy works in shell terms
- Reasoning: identifies missing hyphen as the issue.
- Action: `mv /home/nologik/test-rustaxum-v3/Cargo.toml
  /home/nologik/test-rust-axum-v3/Cargo.toml && rm -rf
  /home/nologik/test-rustaxum-v3`. Reasonable but **fails to address the
  artefact tree from turn [10]**. Tool [17] returns `"(no output)"`.

### Turn [18] — NEW path drift; reasoning vague
- Reasoning: `"Now let me verify the directory structure and create"`
  *(truncated, no path mentioned)*.
- Action: `read filePath="/home/nologik/test/test-rust-axum-v3"` — **a `test/`
  subdir hallucinated out of nothing**. This path was never mentioned by the
  user, the tool history, or any prior reasoning. The model is now drifting
  into a different namespace. Tool [19] returns "File not found".

This matches MISSION_PROGRESS epoch 4 v58 ("Model hallucinated entirely wrong
paths") — passes weak validation but is semantically wrong. Reasoning here is
*absent* (it's truncated and never names a path), so the disconnect is
"reasoning empty / action confabulated".

### Turn [20] — Recovery probe, reasoning stub
- Reasoning: `"Let me check whatd"` — 18 chars, truncated mid-token.
  Reasoning is essentially **destroyed** at this point.
- Action: `bash ls /home/nologik/` — actually a sensible probe.
- Tool [21] returns a 1497-char listing of home dir containing
  `test-rust-axum-v3` plus dozens of other things. The model has now seen the
  correct path again.

### Turn [22] — Recovery
- Reasoning correctly says the directory exists; action is `ls -la
  /home/nologik/test-rust-axum-v3/` (path correct).
- Tool [23] shows the smoking gun:
  ```
  -rw-rw-r--+  Cargo.toml         (the file from [16] mv)
  drwxrwxr-x+ Cargo.toml          (the directory created by [10] mkdir error)
  <                               (literal '<' file)
  drwxrwxr-x+ src                 (from [8] mkdir)
  ```
  i.e. **a file and a directory both named `Cargo.toml` co-exist**, plus a
  literal `<` artifact.

### Turn [24] — Reasoning correct, **action drifts back to phantom path**
- Reasoning: `"It looks like there's a Cargo.toml directory which is wrong -
  it seems my earlier write command created a directoryinstead of a file.
  Let me check what's in"` — **perfectly correct diagnosis**.
- Action: `bash ls -la /home/nologik/test/test-rust-axum-v3/` — **drifts back
  to the phantom `test/` path from turn [18]**, instead of inspecting the
  broken Cargo.toml in the real path that its own reasoning just identified.
- Tool [25] returns "No such file or directory".

**This is the smoking-gun reasoning↔action disconnect.** Reasoning says
"check what's in [the broken Cargo.toml directory]"; action lists a phantom
sibling directory. The model is being driven by **stale tokens** from turn
[18] re-surfacing in the action stream while reasoning has moved on.

### Turn [26] — Reasoning completely empty; action sensible
- Reasoning: `""` (empty).
- Action: `bash ls -la /home/nologik/test-rust-axum-v3/src/` — correct path.
- Tool [27]: empty directory (correctly so).

The reasoning channel has degraded to zero here. The action chose the correct
path almost as if **the action stream is now ignoring reasoning entirely**.

### Turn [28] — Last turn; reasoning correct, action correct (recovery)
- Reasoning: `"The directory exists but has a broken Cargo.toml (it's both a
  file and a directory). Let me fix this and create the"` *(truncated)*.
- Action: `bash rm -rf /home/nologik/test-rust-axum-v3 && mkdir -p
  /home/nologik/test-rust-axum-v3/src`. Correct recovery move.

The probe ends here — there is no next assistant turn, so we never see whether
the model proceeds to actually write the files.

---

## Cross-reference vs `MISSION_PROGRESS.md`

| Observed in probe | MISSION_PROGRESS attribution |
|---|---|
| [10] qwen_xml closer leaking into JSON `filePath` | **Already attributed** — Epoch 1/3 ("empty `<parameter=>` slipped through grammar", xgrammar ε-edge bugs). The probe shows the worst-case form: the whole body in `filePath`. |
| [12] one-byte hyphen drop `rustaxum` vs `rust-axum` | **Partly attributed** — Epoch 2c v56 ("character drift `test-rust-xam-v56`"), Epoch 4 v58 ("/test-tu-au-u8"). This probe gives a more diagnostic example: drift happened on a *fresh write*, not multi-turn drift. |
| [18] phantom `/test/` subdir | **Partly attributed** — Epoch 4 v58 "hallucinated entirely wrong paths that PASS validation". |
| [24] reasoning↔action disconnect | **NOT in MISSION_PROGRESS** — the doc focuses on grammar enforcement and FP8 precision, not on reasoning↔action mis-routing. This is a new finding. |
| [20]/[26] degraded reasoning channel (`"Let me check whatd"`, `""`) | **NOT in MISSION_PROGRESS** — the doc measures cosine drift on hidden states but doesn't connect it to truncated/empty reasoning_content. This may be a `preserve_thinking=true` symptom (flagged as A4 future work). |
| [8]/[10] near-duplicate assistant turns around a `(no output)` tool | **NOT in MISSION_PROGRESS**. Looks like a loop-attractor seeded by an uninformative tool response. |

---

## System prompt assessment

The system prompt is **short (1350 ch) and reasonable**: tools listed, env
block correct, instruction "Think briefly, then act. Do NOT describe tool calls
in your thinking" is sensible. **No "try multiple approaches" anti-pattern.**

What it does NOT include and arguably should:
- The user's target path `./test-rust-axum-v3` is only in the user message,
  not echoed into the system block. The model is forced to rely on
  long-distance attention to a turn-1 token that gets further away each turn.
- No explicit "if a tool returns '(no output)', that's success; do not
  re-describe what you just did" — this would short-circuit the [8]→[10]
  near-duplicate loop.

The **bash tool description** is healthy (workdir parameter, mkdir
verification step). The **write tool description** is also healthy. Opencode
is *not* the culprit for the qwen_xml leak — the model is emitting qwen_xml
inside a JSON string field because Atlas's xgrammar JSON-schema mode lowers
the qwen_xml tool format incorrectly (Epoch 1 finding).

---

## Top-5 ranked failure patterns and interventions

### 1. **qwen_xml closer leaks into JSON `filePath`** (turn [10])
- **What**: model puts file body inside `filePath`, with `</parametercontent>`
  sentinels.
- **Frequency in probe**: 1 instance, but catastrophic (creates
  permanently-broken directory).
- **Intervention**: ship Tier-0 EBNF grammar (MISSION_PROGRESS Epoch 3 already
  proved EBNF rule inlining structurally enforces non-empty parameters; need
  to extend to **forbid `<` byte inside string-valued params** via EBNF
  `value ::= [^<\n]+`). This is the existing roadmap — keep pushing.

### 2. **One-byte tokenizer drift on path strings** (turn [12]: `rustaxum`)
- **What**: the hyphen between `rust` and `axum` is dropped during fresh
  decode. The model recognises the drift in turn [14] but the wrong-path
  write already succeeded silently.
- **Intervention**: tool-side **echo-canonical-path back to the model**.
  Today the `write` tool returns `"Wrote file successfully."` with no path.
  Have it return `"Wrote file successfully: /actual/path/written"`. The model
  will then see the drift in the tool result loop and self-correct without
  needing intermediate `ls` probes. This is an opencode-side fix, no Atlas
  change.

### 3. **Reasoning↔action disconnect** (turn [24])
- **What**: reasoning says "check the broken Cargo.toml directory"; action
  lists `/home/nologik/test/test-rust-axum-v3/` (phantom sub-dir from turn
  [18]). The action stream is being driven by tokens that the reasoning
  channel has already moved past.
- **Hypothesis**: the chat-template's `<think>...</think>` block is being
  KV-cached *separately* from the action tokens, and during decode the
  attention into the action region is over-weighting tokens from a
  several-turns-ago user-visible context (turn [18]'s phantom path stayed in
  visible content). This matches the **FlowKV per-turn KV isolation**
  hypothesis (A4/B9) in MISSION_PROGRESS.
- **Intervention**: implement per-turn KV isolation OR
  `preserve_thinking=true` chat-template fix so the action tokens don't
  attend stale path tokens from prior turns' content.

### 4. **Reasoning channel degrading to empty / truncated** (turns [20],
   [26], [28])
- **What**: across the 14 assistant turns, reasoning_content lengths drop
  from 233 → 65 → 78 → 143 → 142 → 163 → 52 → 18 → 91 → 158 → 0 → 115.
  Notably **turn [26] emits an empty reasoning channel**, and several
  earlier turns are mid-token truncated (`"Let me check whatd"`, `"Let me
  fix this and create the"`).
- **Hypothesis**: the model is hitting an internal `</think>` terminator
  prematurely under FP8 logit drift. This is consistent with the
  MISSION_PROGRESS Phase 2b/2c note that softmax & FP8 KV drift cause
  late-layer regressions (Bug #2 in `project_qwen36_phase2b_softmax_expf`).
  When reasoning collapses to "" (turn [26]), the action channel is forced
  to drive end-to-end from short-distance attention, **and it actually does
  pick the correct path** — suggesting reasoning collapse may not be a
  proximate cause of action drift, but it removes a verification surface.
- **Intervention**: bump `POST_THINK_MIN_CONTENT` floor for reasoning_content
  same way `project_qwen36_fp8_post_think_eos.md` did for assistant content
  (require ≥N chars before honouring `</think>`). Side benefit: easier to
  detect drift from outside Atlas.

### 5. **Near-duplicate turn loop-attractor on uninformative tool returns**
   (turns [8]→[9: `(no output)`]→[10])
- **What**: a `(no output)` tool result is followed by an assistant turn that
  is a paraphrase of the previous assistant turn (both reasoning AND
  content). This is the "loop-attractor" pattern.
- **Hypothesis**: with no new content in the prompt, the model's next-token
  distribution is dominated by recency-bias over its own previous turn,
  producing paraphrase. The xgrammar/EBNF enforcement does nothing here
  because the *structural* form is fine; the *content* is repetitive.
- **Intervention** (opencode-side): when a tool returns `(no output)`,
  augment the tool result with `"(command completed successfully, no
  stdout)"`. Empty payloads invite paraphrase. This is a one-line opencode
  patch with no Atlas-side cost.

---

## Summary

The probe shows Atlas FP8 on Qwen3.6 producing **mostly-correct reasoning
with intermittently-corrupt actions**. The action corruptions fall into
three classes:

- (a) **Structural** (qwen_xml in JSON, turn [10]): solved by the existing
  EBNF roadmap.
- (b) **Token-level** (one-byte path drops, turn [12]; phantom subdirs,
  turn [18]): NOT solved by Atlas-side fixes alone; the loop has to be
  closed by **opencode echoing canonical paths back to the model**.
- (c) **Cross-turn** (turn [24]'s reasoning↔action disconnect): driven by
  multi-turn KV pollution; requires the FlowKV or `preserve_thinking=true`
  intervention listed as future work in MISSION_PROGRESS.

Mission progress's existing roadmap (EBNF, sampler bias, validators,
FlowKV-as-future-work) covers (a) and (c). What is **missing** from the
roadmap and demonstrated in this probe:

1. **Opencode-side tool-result hardening** (echo path back; never return
   "(no output)") — cheap fixes that would close (b) and the loop-attractor
   without any Atlas change.
2. **Reasoning-channel minimum-length floor** — Atlas-side parallel to the
   already-shipped `POST_THINK_MIN_CONTENT` for content.

Word count: ~2350.
