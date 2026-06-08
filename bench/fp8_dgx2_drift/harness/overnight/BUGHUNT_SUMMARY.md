# Post-overnight Atlas bug hunt — summary (2026-06-08)

Image under test: **atlas-gb10:cc-toolbody** (CC6 fix), atlas-camp @ :8888,
Qwen3.6-35B-A3B-FP8, bf16 head / 32k / bf16 KV / MTP K1 / slai.
Client: **claude-code** (Atlas `/v1/messages`). opencode unavailable (see below).

## Headline
The overnight ladder found + fixed **CC6** (large single-shot file-write
truncation — the night's real Atlas defect; committed 247ba88, pushed). This
follow-up phase stressed the agentic paths CC5/CC6 did **not** cover — the
**Edit tool** and **multi-turn** sessions — and found **no new Atlas bug**.

## Coverage (all clean on Atlas)
| Path | Probe | Result |
|------|-------|--------|
| In-place stub edit (Edit tool) | edit_probe | ✅ 3 stubs edited in place, 3/3 |
| Long-context edit | edit_probe2 longctx | ✅ buried stub in 309-line file, neighbors intact |
| Deep-context edit (~1000L/~13k tok) | edit_probe3 bigedit | ✅ buried scale_123 edited; **FP8 deep drift did NOT corrupt the Edit old_string match** |
| Multi-file rename (2 files + call site) | edit_probe2 multifile | ✅ doubler→times_two everywhere |
| Large single Write (295 lines) | edit_probe3 bigwrite | ✅ complete, ends clean, **0 envelope cuts — CC6 holds at scale** |
| Sustained multi-turn (8 stubs, ~16 round-trips) | edit_probe4 | ✅ no corruption; 6 sequential edits all correct |

## De-confounded as NON-Atlas
- **opencode is down** — known stale-cache hang (124 timeout, 0 bytes, no Atlas
  request even after cache-aside). Recovery = power-cycle + reinstall (plumbing).
- **Model temp-variance no-ops**: claude-code occasionally returns in 15–50s
  having done little/none of an agentic task, then **succeeds on re-roll**
  (multifile, bigedit, edit_probe4 run2). Stochastic model behavior — NOT a
  deterministic Atlas cut (a real Atlas bug would fail at the same point).
- **Model coding bugs**: macro-defined-after-use (bigwrite), E0382 borrow / axum
  API confusion (overnight L4-L8). Coherent, complete code with real Rust bugs.
- **FP8 floor (#211)**: single dropped operator deep in context (overnight L8
  `(a-b).abs() EPSILON`). Multi-week W8A8+FP32-epilogue work; did **not** manifest
  as Edit-tool corruption even at ~13k-token context.

## One mild signal to watch (not a bug today)
On the sustained, naturally-repetitive multi-turn task (implement-stub →
run-test → repeat), the **SimHash + thinking-loop watchdogs fired** (ring_len
2–3; thinking forced-closed at 48 tok). They **recovered** — the session went on
to complete 6 correct stubs — so no harm here. But loop watchdogs firing on
legitimately-repetitive agentic work is the SPINFIX concern (task #234); on a
longer/more-repetitive session it could contribute to early termination. Worth a
future tuning pass (gate these watchdogs harder when tools are active and edits
are progressing), but no clear failure to fix tonight.

## Harness lessons (folded into the probes)
- Run **N>1 per scenario** — single runs conflate model temp-variance with real
  failures (edit_probe4 does 2×; a no-op that succeeds on re-roll = model, not Atlas).
- Completion checks must verify the **task**, not just `cargo test` — a
  behavior-preserving rename passes tests even if not performed (multifile gap).
- `still_has_todo`-style substring checks must exclude doc-comments (false-positive).

## Conclusion
The claude-code agentic path (Read / Edit / Bash / Write, across trivial,
long-, deep-context, multi-file, and multi-turn) is **robust on Atlas
post-CC6**. Hunt converged. Next real lever for agentic quality remains the FP8
precision floor (#211, multi-week) and an optional watchdog-on-repetitive-work
tuning pass — neither a tonight-fixable defect.
