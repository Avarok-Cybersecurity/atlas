# CLAUDE HANDOFF — DFlash verify accept-decay: clean gated fix (Option B)

**Branch to work on:** `fix/dflash-verify-argmax-gate` (off clean base `8bed827`).
**DO NOT build on** `bisect/no-verify-pipeline` / commit `ef8092d` — that is the
throwaway diagnostic ("do not merge"); it strips the pipeline unconditionally and
breaks MTP. It only existed to prove the root cause. Ignore it for the fix.

**Build target (Ronald runs builds, NOT you):**
`ATLAS_TARGET_MODEL=qwen3.6-27b cargo build --release -p spark-server`

---

## ROOT CAUSE (settled, bench-proven — do not re-derive)

The merge routed DFlash verify token-selection through
`verify_pick_all_with_pipeline`, which applies `rep_pen=1.1` + `dry_mult=0.5`.
The DFlash drafter PROPOSES on raw argmax (no penalties). Verifier and drafter
therefore judge on DIFFERENT bases. As context grows, more tokens accrue
penalties, the verifier's penalized pick diverges from the drafter's argmax pick,
and K3 accept decays monotonically: 0.55 -> 0.12 -> 0.00, 8.8 tok/s.

Proven by the bisect (`ef8092d`): bypass the pipeline in verify_k3 -> raw argmax
(GOLD basis, matches the drafter) -> accept holds 0.4-0.85, NO decay, 13.9 tok/s.

DEAD THEORY (do not revisit): the SSM rollback ring is NOT involved — it is dead
code in this path (`emit_token` never calls `snapshot_boundary_if_ssm` /
`rollback_to_boundary`; ring stays empty). Ruled out by static read + a
disable-rollback bench, both flat.

---

## WHY THE BISECT IS NOT THE FIX

`step_verify_k3` is SHARED. Its only caller is the dispatch in
`crates/spark-server/src/scheduler/mtp_step.rs:163-171`, which routes purely by
`drafts.len()`:
- `>= 4`           -> `step_verify_dflash`
- `>=3 && num>=3`  -> `step_verify_k4`
- `>=2 && num>=2`  -> `step_verify_k3`   <-- DFlash γ=3 (num_drafts=2) AND MTP num_drafts=2 BOTH land here
- else             -> `step_verify_k2`

So unconditionally removing the pipeline in verify_k3 (the bisect) also strips
rep_pen/DRY from nologik's MTP K3 path. We must gate on "is this process running
the DFlash drafter," not rip the pipeline out for everyone.

There is NO per-sequence dflash flag, and NO `is_dflash_active()` method (it does
not exist — earlier proposal referenced a phantom symbol). DFlash is a
PROCESS-WIDE launch mode: `args.dflash` (set by `spark serve … --dflash`). MTP
mode (no `--dflash`) never coexists in the same process. So the gate is a single
bool threaded from `args.dflash` into the verify steps.

---

## THE CHANGE (minimal, additive)

Thread a `dflash_verify_raw_argmax: bool` (value = `args.dflash`) from the
scheduler entry down to the verify steps, and gate the pipeline call.

1. **`crates/spark-server/src/main_modules/serve.rs`** — at the scheduler
   `run(...)` call site (the spawn that ultimately reaches
   `scheduler/mod.rs::run`), pass `args.dflash` in as a new bool param.

2. **`crates/spark-server/src/scheduler/mod.rs`** — add the bool to `run(...)`'s
   signature (next to `use_speculative`), and forward it into the
   `step_mtp(&*model, &mut active, num_drafts, &verify_ctx)` call at ~line 333.

3. **`crates/spark-server/src/scheduler/mtp_step.rs`** — add the bool param to
   `step_mtp(...)` and forward it into `step_verify_k3/k4/k2` calls
   (lines 166/168/170). (`step_verify_dflash` at 164 already uses GPU argmax and
   does NOT call the pipeline — verify this; if true, leave it untouched.)

4. **`crates/spark-server/src/scheduler/verify_k3_step.rs`** (and k4, k2) — add
   the bool param. Gate the `verify_pick_all_with_pipeline` block:

   ```rust
   let (v0, v1, v2) = if dflash_verify_raw_argmax {
       // DFlash drafter proposes on raw argmax; verify on the SAME (GOLD)
       // basis so verifier/drafter judge identically. No rep_pen/DRY here.
       (v0_argmax, v1_argmax, v2_argmax)
   } else {
       // MTP path: full pre-sample pipeline (rep_pen + DRY) unchanged.
       let processed = crate::scheduler::verify_pipeline_helper::verify_pick_all_with_pipeline(
           model, &[v0_argmax, v1_argmax, v2_argmax], a, verify_ctx);
       (processed.first().copied().unwrap_or(v0_argmax),
        processed.get(1).copied().unwrap_or(v1_argmax),
        processed.get(2).copied().unwrap_or(v2_argmax))
   };
   ```

   Apply the identical gate shape in `verify_k4_step.rs` and `verify_k2_step.rs`
   (DFlash γ=4 and γ=2 have the SAME basis mismatch the bisect missed — fix all
   three for consistency).

## GRAMMAR — already safe, do not add masking to the verify pick

Grammar boundary truncation runs UPSTREAM at `mtp_step.rs:148`
(`truncate_drafts_at_grammar_boundary`) BEFORE dispatch. The grammar mask is not
part of the verify token pick, so raw argmax in verify does NOT break structured
output. Do not add grammar masking inside the verify steps.

## SCOPE BOUNDS (hard)
- Do NOT refactor `verify_pipeline_helper` or nologik's MTP verify logic.
- Do NOT author kernels. This is pure Rust control-flow threading + one gate.
- Do NOT touch `step_verify_dflash` unless it is confirmed to call the pipeline.
- Smallest diff that threads one bool and gates three call sites.

## DONE = 
- Builds clean with the target flag above.
- Ronald benches DFlash γ=3: accept 0.4-0.85, no decay, ~13.9 tok/s.
- Ronald sanity-benches MTP (no `--dflash`): pipeline still fires, no regression.
- Friday verifies the diff on disk vs SHA, squashes to ONE commit naming root
  cause + the global-flag gate, pushes the merge.
