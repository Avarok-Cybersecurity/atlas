# Agent A — Tool-Call Malformation Findings

## TL;DR (hypothesis)

**The content-loop watchdog is amputating tool-call argument JSON mid-emission.** When it fires inside an XGrammar-constrained tool body, the response ends with `finish_reason=length` and whatever fragment has streamed becomes the final tool-call JSON. That's why opencode sees empty `{}`, half-formed paths like `test-rust-ax{\"v19/src/main.rs\"}`, and `"command":""` skeletons. There is a near-perfect 1:1 correspondence between watchdog/salvage fires in the server log and malformations in the dump.

## Per-malformation table (10 malformed of 17 tool_calls in the final request)

| # | dump seq | msg_idx | tool | args (verbatim, truncated) | preceded by |
|---|----------|---------|------|----------------------------|-------------|
| 1 | 4 | 8 | `write` | `{"content":"…\n#[tok::main]\n…","filePath":"test-rust-axum-v19/src/main.rs"}` | full JSON, but token-level corruption: `tok::main` vs `tokio::main`, unused `reqwest` import — model-side, not watchdog |
| 2 | 6 | 10 | `write` | rewritten — `#[tokio::main]` correct now | tool-result feedback |
| 3 | 11 | 18 | `edit` | `{"filePath":"test-rust-ax{\"v19/src/main.rs\"}","newString":"let app = app();","oldString":"let mut app = app();"}` | **Content-loop watchdog fired, ending response 140 tok `length`** (server.log:2699) — cut JSON mid-stream at `test-rust-ax`, leaving a stray `{` inside the string |
| 4 | 12 | 20 | `edit` | identical garbled path | tool-result "File not found" → model re-emitted the same string |
| 5 | 13 | 22 | `bash` | `{"command":"","description":"Fix mut warnings with sed{\"command\":\"sed -i 's/let mut app = app();/…\\",\\"description\\":\\"Fix mut warnings with sed\\"}"}` | model nested a *second* call inside the `description` value of the first — watchdog cut between |
| 6 | 14 | 24 | `write` | `{}` | **Content-loop watchdog 184 tok `length`** (2973) — truncated before any field key |
| 7 | 15 | 26 | `bash` | `{"command":"","description":"Run: "}` | **Content-loop watchdog 88 tok `length`** (3005) |
| 8 | 17 (intermediate, no msg) | — | `description` (synthetic name) | rejected by validator | **`tool-call salvage fired pass=param_as_function_salvage`** (3106) — parser took a *param key* as the *function name* |
| 9 | 18 | 31 | `bash` | `{"command":"","description":"Run: "}` | **Content-loop watchdog 78 tok `length`** (3130) |
| 10 | 19 | 33 | `edit` | `{}` | **Content-loop watchdog 86 tok `length`** (3162) |
| (11) | 20 intermediate | — | `description` | salvage path | **salvage** (3184) |
| (12) | 20 | 36 | `edit` | `{}` | salvage + `[atlas] Tool call rejected: Unknown tool 'description'` injected into content |

## Conversation-depth turning point

| seq | prompt_tok | comp_tok | reasoning_tok | finish | malformed |
|----:|-----------:|---------:|--------------:|--------|-----------|
| 4 | 7,740 | 467 | 300 | tool_calls | yes (`#[tok::main]` typo only) |
| 5 | 7,920 | 893 | 9 | tool_calls | yes (rewrite) |
| 6–10 | 8.8k–13.0k | 90–140 | 17–66 | tool_calls | no — clean structured JSON |
| **11** | **13,152** | **94** | **12** | **tool_calls** | **first watchdog amputation** |
| 12–20 | 13.3k–14.0k | 66–184 | 17–36 | mostly `length` | every single one malformed |

The **turning point is `prompt_tokens ≈ 13k`**. Below 13k the model emits full JSON. Above 13k the watchdog fires on virtually every response and trims output to 78–184 tokens with `finish_reason=length`.

`reasoning_tokens` is **not** the discriminator — by seq 11 it's already 12 (far under the 768 budget). Thinking is not being burned; emit is being amputated.

## Why the watchdog fires so much

MTP K2 acceptance over the whole session: **8 ACCEPT / 2,489 REJECT = 0.32%**. Atlas is effectively running greedy decode plus K2 verify overhead. Inside grammar-constrained JSON (escape runs, repeated indentation, period attractors after tool-error feedback) the verify path keeps proposing the same token (e.g. `prev_draft=1152` reappearing 4× in a row at log:2695–2698). That looks identical to a content-loop period-N attractor to the watchdog, which then ends the response with `length` before XGrammar has closed the JSON.

## Concrete log references

- Watchdog → truncated tool-call pairs: `server.log` lines **2699+2701, 2787+2789, 2973, 3005+3007, 3130+3132, 3162**.
- Salvage path inventing `description` as a function name: `server.log` **3106, 3184** (`pass=param_as_function_salvage`).
- MTP K2 rejection storm immediately before watchdog: `server.log` **2695–2698**, **2871–2972**.
- First malformation at `prompt_tokens=13,152` (`dump.jsonl` response seq=11).

## Recommended fix direction

Suppress the content-loop watchdog while XGrammar reports it is **inside a tool_call body** (between `<tool_call>` start token 248058 and `</tool_call>` end token 248059, or while the grammar state is in a JSON value). The repetition the watchdog is matching is a property of constrained JSON, not a real attractor. The grammar already guarantees structural termination; only sample under it without secondary heuristic interference. Secondary item: investigate the `param_as_function_salvage` path — it converts a key like `"description":` into a tool name when an earlier `<tool_call>` body was truncated, surfacing a phantom tool to opencode rather than failing silently.

Model-level token corruption (`#[tok::main]`, `withcurl`, `axa`) is a separate concern (likely FP8 KV drift at long context), but it is **not** the cause of the empty `{}` malformations — those are pure server-side amputation.
