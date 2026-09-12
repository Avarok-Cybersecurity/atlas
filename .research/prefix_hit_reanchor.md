# Full-prompt prefix-cache hit falls to a cold prefill — repro, root cause, fix, GPU recipe

Branch: `wip/exl3-upstream` (worktree `/home/ms/atlas/.claude/worktrees/exl3-upstream`), forked from
`wip/exl3-research @ da09c7bcf`. Pre-existing engine behaviour — measured on this branch with AND
without the EXL3 gates, so it is not an EXL3 regression. No GPU was used for this task; the fix is
CPU-built and unit-tested, the validation recipe at the end is for the operator.

Unified diff of the fix: `/home/ms/.claude/jobs/5a7bd33d/tmp/upstream/prefix_hit_fix.diff` (8 files, +494 lines incl. tests/docs).

## 1. Symptom (exact log lines)

A request whose prompt is ENTIRELY covered by the radix cache (`cached_tokens == prompt_tokens`, 3286
tokens / 206 blocks at 16 tokens per block) pays the full cold prefill — 6-7 s at 3.3K — instead of the
~0.2 s an intermediate (multi-turn, `matched < total`) hit gets. `RUST_LOG=info` shows, in order:

```
# request 1 (cold) — end of prefill, finish leaf saved at the full prompt:
Saved SSM snapshot <id> for 3286 tokens (206 blocks) [chunk]
# ... plus an intermediate checkpoint at the block-aligned depth 3264 (= 204 blocks) for the same prefix,
#     written seconds earlier by whichever writer ran on this serve (the line is one of
#     "Intermediate SSM checkpoint saved at token 3264 (snapshot_id N, block 204)"   [prefill_b_save_checkpoint]
#     "midchunk EARLY tail SSM capture at token 3264 (snap N)"                        [SCALE builds only]
#     — the fix does not care which; any snapshot strictly below the prompt qualifies).

# request 2 (identical prompt, everything cached):
Prefix cache hit: 3286 tokens (206 blocks) but no SSM snapshot — recomputing all KV
```

The second line is the misleading one: the snapshot index DID return a snapshot for this prefix (the
exact leaf at 3286), and a usable one at 3264 also exists. Note also that the info line
`exact-leaf snapshot shortcut bypassed (default; ATLAS_MARCONI_EXACT=1 re-enables) ...` does NOT
print on the default path — it sits behind `if skip && ...` and `skip` is already false by then
(see below) — so the only visible trace is the "no SSM snapshot" line. `usage.prompt_tokens_details.
cached_tokens` still reports 3286, because the KV half of the match (block refs) is taken regardless.

## 2. Code path (before the fix)

`crates/spark-model/src/model/trait_impl/prefill_b/prefix_lookup.rs::prefill_b_prefix_lookup`, chunk 0:

1. `self.prefix_cache.lookup(tokens, bs, session_hash, adapter_id)` → `RadixTree::lookup`
   (`crates/spark-runtime/src/radix_tree.rs`). Phase 1 walks the radix (matched = 3286, refs taken).
   Phase 2 calls `SsmSnapshotIndex::lookup_tiered(tokens, matched_tokens=3286, ...)`
   (`radix_tree/snapshot_tier.rs`), which returns the single DEEPEST entry with
   `token_count <= matched_tokens` and a matching prefix hash: the exact leaf at 3286. The 3264
   checkpoint is a valid candidate but loses the max — the index never reports a runner-up.
2. `eff_ssm_snapshot` (`trait_impl/ssm_fault_in.rs`) folds resident/tier → `(Some(leaf), 3286)`.
3. In `prefix_lookup.rs` the restore condition computes
   `bypass_exact = snap_tok == matched && matched == total && ATLAS_MARCONI_EXACT != "1"` — true by
   default: the exact full-prompt shortcut is deliberately bypassed as unsound (the snapshot holds
   state@N, the last token needs state@(N-1), the recurrence is not invertible; the shortcut's
   double-advance also poisons the KV of position N-1 in a block shared with the cache). With
   `ATLAS_MARCONI_EXACT=1` the sibling guard `exact_without_hidden` still declines a leaf that has no
   stashed hidden. Either way the `if` fails → `skip = false`.
4. There is no second attempt: control falls to
   `if matched > 0 && !skip && has_ssm { info!("Prefix cache hit: ... but no SSM snapshot — recomputing all KV") }`
   and `skip_tokens = 0`. Full recompute of KV + SSM over all 3286 tokens.

Hypothesis from the task — "the lookup selects the LARGEST snapshot <= matched (the exact leaf), the
bypass then forces skip=false and falls to FULL recompute instead of retrying with the next-lower
non-exact snapshot" — is CONFIRMED by reading `lookup_tiered` (single best, `token_count > matched →
skip`, else `token_count > best_depth → best`) and the `prefix_lookup.rs` control flow above. The
3264 anchor is sound to use: it is the established intermediate-hit path with `snap_tok < matched ==
total` (the "leaf evicted" case the file's own CRITICAL comment describes) — restore state@3264,
`marconi_skip_to = 3264`, the suffix prefill replays `[3264, 3286)` through SSM (22 tokens) and
produces the last token's logits normally; KV writes below `cached_prefix_tokens` stay floored
(`layer_kv_write_start`) so the shared cache blocks are never rewritten; `marconi_exact_snap` stays
`None` so no exact-hit fixup runs.

`prefill_a.rs` / `prefill_c.rs` have no exact-leaf bypass (they restore the exact leaf), so only the
prefill_b path was affected and only it is changed.

## 3. The fix (selection only — the exact-leaf shortcut and its bypass are untouched)

### spark-runtime

* `crates/spark-runtime/src/prefix_cache/ssm_anchor.rs` (NEW, 132 LoC): `SsmAnchor` — the `ssm_*` half
  of a `PrefixMatch` (`snapshot`, `snapshot_tokens`, `tier_key`, `tier_tokens`, `is_tail`), with
  `depth()` folding resident-vs-tier exactly like `eff_ssm_snapshot`, and
  `PrefixMatch::{ssm_anchor, set_ssm_anchor}` (KV half untouched). Two unit tests.
* `crates/spark-runtime/src/prefix_cache.rs`: new trait method with a default
  `fn lookup_ssm_anchor(&self, tokens, max_tokens, session_hash, adapter_id) -> SsmAnchor { NONE }` —
  the deepest snapshot with depth `<= max_tokens`, WITHOUT walking the radix or taking block refs.
  `NoPrefixCaching` (and any mock) inherits the default.
* `crates/spark-runtime/src/radix_tree.rs`: `RadixTree::lookup_ssm_anchor` = the existing
  `lookup_tiered` conversion, factored out of `lookup` (which now calls it with
  `max_tokens = matched_tokens`, byte-identical result: same filter — prefix hash, tail
  session-gate, adapter key, resident-or-spilled).

### spark-model

* `crates/spark-model/src/model/trait_impl/prefill_b/prefix_reanchor.rs` (NEW, 201 LoC):
  * pure `exact_leaf_declined(depth, matched, total, exact_enabled, has_hidden)` — mirrors
    `bypass_exact` / `exact_without_hidden`: `depth > 0 && depth == matched && matched == total &&
    (!exact_enabled || !has_hidden)`; intermediate anchors and warm turns (`matched < total`) never.
  * pure `reanchor_cap(total) = total - 1` (the cap is inclusive → strictly below the prompt).
  * `TransformerModel::prefill_b_resolve_ssm_anchor(tokens, &mut prefix_match, total, session_hash,
    adapter_id, stream) -> (Option<usize>, usize)`:
    1. BEFORE any tier fault-in, if the raw anchor is an exact leaf declined by the default bypass →
       `prefix_cache.lookup_ssm_anchor(tokens, total - 1, ...)`; if found, `set_ssm_anchor` on the
       match and log `Marconi exact leaf at token {total} declined (...); re-anchored on the deepest
       snapshot below the prompt: token {d} ({n} SSM tokens to replay, tail=.., resident=..)`. Doing
       this first means a spilled exact leaf is never faulted in just to be declined.
    2. `eff_ssm_snapshot` as before.
    3. AFTER the leaf is resident (only reachable with `ATLAS_MARCONI_EXACT=1`): a hidden-less finish
       leaf is declined too → same re-anchor, then `eff_ssm_snapshot` again.
    If nothing qualifies below the prompt the match is left untouched and control falls through to
    the pre-existing full-recompute path unchanged (debug log).
  * Four unit tests on the pure functions (the repro numbers 3286 / 3264).
* `prefix_lookup.rs`: the single `eff_ssm_snapshot` call is replaced by
  `prefill_b_resolve_ssm_anchor(...)`; nothing else changes. Because the re-anchored depth is
  `< matched == total`, every downstream guard runs unchanged on the NEW anchor:
  `marconi_min_tokens()` (256), `snap_tok > 0`, `matched <= total`, `exact_without_hidden` /
  `bypass_exact` (now false), the tail session gate (`ssm_snapshot_is_tail` is carried by the
  anchor; `lookup_tiered` already applied it too), and `requires_aux_state() → aux(snap).is_some()`
  for PLE/QSA models (an aux-less 3264 slot is still declined → old behaviour, never stale state).
  The later `if skip && prefix_match.ssm_snapshot_tokens == matched && matched == total` block reads
  the rewritten field and stays inert. `ATLAS_MARCONI_EXACT=1` semantics are unchanged: a leaf WITH
  hidden is restored via the shortcut exactly as before; only the hidden-less case now re-anchors.
* `prefill_b.rs`: `mod prefix_reanchor;`.

Side effects worth knowing: the re-anchor is a second `lookup_tiered` call, so the ATLAS_SSM_SNAP_STATS
Phase-0 counters count one extra lookup+hit on such a request and the intermediate's LRU recency is
bumped (desired — it is the anchor actually used).

LoC (500 cap, `.github/workflows/file-size-cap.yml`, `crates/**/*.rs`): prefix_lookup.rs 481 → 492,
prefix_cache.rs 425 → 449, radix_tree.rs 298 → 310, radix_tree/tests/snapshot.rs 347 → 429,
prefill_b.rs 380 → 381; new files 132 / 201. No allow-list change.

### Gates (CPU only, logs in `/home/ms/.claude/jobs/5a7bd33d/tmp/upstream/`)

* `cargo test --release -p spark-model --lib` — 687 passed / 0 failed / 11 ignored (683 + 4 new) — `fix_test_model.log`
* `cargo test --release -p spark-runtime --lib` — 295 passed / 0 failed / 12 ignored (291 + 4 new) — `fix_test_runtime.log`
* `cargo test --release -p spark-server --lib` — 104 passed / 0 failed — `fix_test_server.log`
* `cargo build --release -p spark-server --bin spark` — EXIT=0, no warnings from our crates — `fix_build.log`
* `cargo clippy --release -p spark-runtime -p spark-model -p spark-server --all-targets` — EXIT=0 (only the two atlas-kernels notes) — `fix_clippy.log`
* `rustfmt --edition 2024` on every touched file.

## 4. GPU validation recipe (operator; NOT run here)

Boot the EXL3 serve from the exl3-upstream binary (the canonical boot script hardcodes the
exl3-research binary — point it at the rebuilt one):

```bash
# from a box where both GPUs are free; check `free -g` first (util 0.6, one Atlas instance)
W=/home/ms/atlas/.claude/worktrees/exl3-upstream
sed "s#exl3-research/target/release/spark#exl3-upstream/target/release/spark#g" \
    $W/.research/boot/boot_native_dense.sh > /tmp/boot_upstream.sh
CTX=8192 PREFILL=8192 SEQS=1 LOG=/home/ms/.claude/jobs/5a7bd33d/tmp/upstream/gpu_prefix_hit.log \
    bash /tmp/boot_upstream.sh &          # RUST_LOG=info, --enable-prefix-caching, --ssm-cache-slots 48, port 8890
until curl -sf http://127.0.0.1:8890/v1/models >/dev/null; do sleep 5; done
```

Send the SAME ~3.3K-token prompt three times, temperature 0, measuring wall-to-first-token:

```bash
cat > /tmp/prefix_hit_probe.py <<'EOF'
import json, time, urllib.request
URL = "http://127.0.0.1:8890/v1/chat/completions"
# ~3.3K tokens: 60 numbered paragraphs of filler + a question (salt it if the serve already saw it).
body = "\n\n".join(f"Paragraph {i}: " + ("The quick brown fox jumps over the lazy dog near the riverbank while the miller counts sacks of grain. " * 4) for i in range(60))
prompt = body + "\n\nIn one sentence: how many paragraphs are above, and what animal is mentioned?"
req = {"model": "qwen4exp-exl3", "messages": [{"role": "user", "content": prompt}],
       "temperature": 0, "max_tokens": 64, "stream": True,
       "stream_options": {"include_usage": True}}
answers = []
for i in range(3):
    t0 = time.perf_counter(); ttft = None; text = ""; usage = None
    r = urllib.request.urlopen(urllib.request.Request(URL, data=json.dumps(req).encode(),
        headers={"Content-Type": "application/json"}), timeout=600)
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]": continue
        ev = json.loads(line[5:])
        if ev.get("usage"): usage = ev["usage"]
        for ch in ev.get("choices", []):
            d = ch.get("delta", {}).get("content")
            if d:
                if ttft is None: ttft = time.perf_counter() - t0
                text += d
    answers.append(text)
    cached = (usage or {}).get("prompt_tokens_details", {}).get("cached_tokens")
    print(f"req{i+1}: ttft={ttft*1000:.0f} ms prompt_tokens={(usage or {}).get('prompt_tokens')} cached_tokens={cached}\n  {text!r}")
print("IDENTICAL" if len(set(answers)) == 1 else "DIVERGED")
EOF
python3 -u /tmp/prefix_hit_probe.py 2>&1 | tee /home/ms/.claude/jobs/5a7bd33d/tmp/upstream/gpu_prefix_hit_probe.txt
```

Expected:

* req1: cold, `cached_tokens` 0 (or None), TTFT ≈ 6-7 s at 3.3K on GB10 (the baseline for this shape).
* req2 AND req3: `cached_tokens == prompt_tokens`, TTFT ≈ 200 ms (a 22-token SSM replay + attention
  over cached KV), and the three answers IDENTICAL (temp 0). If the text differs only in the tail of
  a long generation it is the ordinary warm-turn fp-ordering effect shared with the existing
  `matched < total` path; a wrong/garbled FIRST token would indicate a real state misalignment — file it.
* Serve log (`grep -n "Marconi\|Prefix cache hit\|Saved SSM snapshot\|checkpoint saved" $LOG`):
  ```
  # req1
  Saved SSM snapshot <a> for <N> tokens (<B> blocks) [chunk]
  # req2, req3 — the new line, then the ordinary intermediate-hit line, and NO "no SSM snapshot" line:
  Marconi exact leaf at token <N> declined (bypassed by default; ATLAS_MARCONI_EXACT=1 re-enables); re-anchored on the deepest snapshot below the prompt: token <d> (<N-d> SSM tokens to replay, tail=false, resident=true)
  Marconi intermediate hit: restored from checkpoint at token <d> (skipping <d> tokens, replaying <N-d> SSM tokens to reach <N>; <N-d> of those are the anchor->match gap to <N>)
  ```
  `<d>` is whatever snapshot below the prompt exists on this serve (3264 in the measured run). If
  instead you see `Marconi exact leaf at token <N> declined ...; no snapshot below the prompt —
  falling through to full recompute` at `RUST_LOG=debug`, no intermediate anchor was ever saved for
  this prompt on this serve (single-chunk prefill with no checkpoint writer) — that is the
  pre-existing behaviour, not the bug this fixes; confirm which checkpoint writers are active
  (`--ssm-checkpoint-interval`, chunking) before reading it as a failure.

Controls (optional):

* `ATLAS_MARCONI_EXACT=1` on the serve → req2/req3 take the unchanged exact-leaf shortcut
  (`Marconi SSM cache hit: <N> tokens skipped ...`); this is the A/B arm the bypass exists for and is
  NOT the fix's path.
* Old behaviour for comparison: the exl3-research binary (`/home/ms/atlas/.claude/worktrees/exl3-research/target/release/spark`)
  reproduces the 6-7 s req2/req3 with the `... but no SSM snapshot — recomputing all KV` line.
* PLE/QSA aux gate: if req2 prints the re-anchor line but then still `... but no SSM snapshot`, the
  chosen slot had no aux state (`requires_aux_state()` models decline it) — expected-safe, and means
  the intermediate writer on this serve does not attach aux; check `collect_aux_states` at that writer.
