---
name: oracle_pre_commit_cross_hardware_check
description: "O.R.A.C.L.E::pre_commit_cross_hardware_check — run BEFORE pushing any commit that touches kernels/ (especially kernels/gb10/common/), a __SCALE__ / __HIP / __CUDA_ARCH__ arm, cfg!(atlas_scale) or cfg(atlas_hip) host dispatch, a KERNEL.toml / MODEL.toml / HARDWARE.toml, or a symlink under kernels/. Detects cross-hardware kernel interference (CHKI): a change meant for one hardware (gb10, strix, strix-hip, metal) that alters what another hardware compiles through symlinks, common/ fan-out, or kernel_source redirects — strix is 7 real files and 105 symlinks into gb10, so a gb10 edit silently changes what AMD builds. Runs the static reach check, gathers verbatim evidence, launches the oracle for the remedy verdict (benign / parameterize in HARDWARE.toml with a reader / separate kernel with no symlink), and writes the Hardware: and CHKI-Verdict: trailers the CI job cross-hardware-reach requires. Also use when that CI job is red, when a strix or strix-hip compile leg fails on a gb10 edit, or when tempted to add a __SCALE__ guard to a gb10 file. Born from d584c0c50 (a gb10 edit broke hipcc, caught only by a compile leg) and 17fe989ec (a Strix workaround in gb10 common code invalidated 27 targets' records)."
argument-hint: "[<base-ref>=origin/main] [--staged | --range A..B | --paths p1 p2 ...]"
allowed-tools: Bash, Read, Grep, Glob, Agent
---

# /oracle_pre_commit_cross_hardware_check — CHKI before push

One question, answered before `git push`: **did this change alter another hardware's compiled
inputs, and what is the remedy?** The static half is provable and runs in seconds; the
judgement half goes to the `opus` agent `.claude/agents/oracle_chki.md`. The CI job
`cross-hardware-reach` runs the same script and will fail the PR on anything this procedure
would have caught — so run it here first.

Why it exists, in two lines: `d584c0c50` (a gb10 edit broke hipcc, caught **only** because a
compile leg failed) and `17fe989ec` (a Strix workaround in gb10 common code invalidated 27
benchmark records). A performance-only leak has never been visible at all.

## Step 1 — the static check (seconds, no GPU, no cargo)

```bash
python3 scripts/check_cross_hardware.py --base "${1:-origin/main}" --worktree
```

Modes: `--staged`, `--range A..B`, `--paths ...` (attribution only, never fails).

Read the table. Every row saying `CHKI` or `DECLARED`, and every `structural:` finding, is an
input to Step 3. **If the verdict is `OK` and no row is `DECLARED`, you are done — do not
invent work.**

`S1` (new cross-hardware symlink) and `S4` (dangling symlink) cannot be declared away. Replace
the symlink with a real file, or restore the target, then re-run.

Exit codes: `0` clean · `1` violation · `2` **cannot see the inputs**, which is as red as `1`
— the same doctrine as `campaign-guard.sh`'s exit 2.

## Step 2 — gather the verbatim inputs

The oracle refuses summaries. Collect all of it:

```bash
python3 scripts/check_cross_hardware.py --base "$BASE" --worktree --json
git diff "$BASE"...HEAD -- kernels/
git log --format=%B "$BASE"..HEAD | grep -E '^(Hardware|CHKI-Verdict):' || true
git diff --name-only "$BASE"...HEAD -- crates/ | xargs -r grep -ln 'atlas_scale\|atlas_hip' || true
```

For every file whose `reach_hw` has more than one entry, per reached AMD hardware:

```bash
git show "$BASE:$F" > /tmp/before.cu
unifdef -k -D__SCALE__=1 /tmp/before.cu > /tmp/b.amd; unifdef -k -D__SCALE__=1 "$F" > /tmp/a.amd
diff -u /tmp/b.amd /tmp/a.amd | wc -l      # 0 = the text AMD sees is unchanged
unifdef -k -U__SCALE__  /tmp/before.cu > /tmp/b.nv;  unifdef -k -U__SCALE__  "$F" > /tmp/a.nv
diff -u /tmp/b.nv /tmp/a.nv | wc -l        # 0 = the text NVIDIA sees is unchanged
```

If `unifdef` is absent, say so in the prompt — the oracle will classify arms by hand and hold
you to a stricter proof. If `nvcc` is present, add a PTX sha256 before/after for the gb10 side.

## Step 3 — launch the oracle

Launch `oracle_chki` with everything from Step 2 pasted verbatim, plus one sentence naming the
intended hardware and the evidence for it. Wait for exactly one verdict line. **Anything hedged
is `CHKI-FAIL`.** On `CHKI-FAIL`, fix what it names and return to Step 1 — do not re-prompt for
a softer answer.

## Step 4 — apply the remedy

**benign** — nothing changes in `kernels/`. Go to Step 5.

**parameterize** — in the SAME commit: add the key to `kernels/<hw>/HARDWARE.toml`; add the
reader in `crates/atlas-kernels/build.rs` beside the `arch`/`vendor` reads; replace the
`__SCALE__` branch with `#if KEY == VALUE`. ★ Only `vendor` and `arch` are read today — a key
with no reader is decoration, and `kernels/strix/HARDWARE.toml` says so in writing.

**separate** — never a symlink: the intended hardware gets the changed kernel as a REAL file;
every other hardware whose symlink pointed at it gets a real copy of the BASE bytes, proven by
`sha256sum` on both sides. Then run `python3 scripts/check_kernel_shadows.py` — RULE 1 still
applies within the hardware you just wrote into.

## Step 5 — trailers, then re-run

```
Hardware: <every hardware in the reach table, comma-separated>
CHKI-Verdict: <benign|parameterized|de-shared> — <the oracle's proof line>
```

Then `python3 scripts/check_cross_hardware.py --base "$BASE" --worktree` must print
`verdict: OK`. Paste the reach table into the PR description; if the PR will be squash-merged,
copy both trailers into the PR body too.

## Do not

- **Do not add a `__SCALE__` / `__HIP*` guard to a gb10-owned file as the remedy.** That is the
  state this check exists to unwind; it passes only as `DECLARED`, and the oracle will send it
  to (b) or (c).
- **Do not create a symlink under `kernels/` that resolves into another hardware's tree.** S1
  fails it with no override.
- **Do not write `Hardware:` from memory.** Copy the reach column; R1 fails on any hardware you
  omit, including the one that owns the bytes.
- **Do not call a change benign for strix because "nobody can measure strix".** Unmeasured is
  `CHKI-FAIL`.
