# references/pr-and-ci.md — sync-pr automation, CI-green preflight, CLA/CI cleanup

## Remotes (know which is which)
```
origin      https://github.com/Avarok-Cybersecurity/atlas.git   # the fork we ship (maintainer pushes here)
monumental  https://github.com/MonumentalSystems/atlas.git       # upstream
```
Sync direction: **upstream `monumental/main` → fork `origin/main`.** Never the
reverse without intent. `sparkrun` recipes and `avarok/atlas-gb10` are built from
`origin`.

## `/atlas-release sync-pr` — the loop
1. **Detect:** `git fetch monumental && git log --oneline origin/main..monumental/main`.
   If empty, nothing to do.
2. **Sync (fast-forward only):** advance `origin/main` to `monumental/main`. If it
   won't fast-forward, **stop and report** — a diverged fork is a human decision,
   not an auto-merge.
3. **Build + verify:** `/atlas-release build *` → `/atlas-release verify`. A sync
   that fails the serve matrix is a blocker to surface, not to ship past.
4. **Enablement PR (only if warranted):** if the merge added a new
   `kernels/gb10/<model>/MODEL.toml` or a new loader that needs a
   `MODEL.toml`/`factory.rs`/docker stanza to be *deployable*, open **that**
   enablement PR on `origin`. Otherwise there is no PR — a plain engine sync just
   feeds the next image cut.
5. **Preflight (below) must be green before submit.** AI-attributed, CLA-clean.

**No-takeover rule:** sync-pr opens *additive* enablement PRs or pushes a trivial
fix to an existing branch. It never opens a parallel reimplementation of someone's
in-flight work — reference their branch and comment instead. Contribute to the
author's work, don't fork around it.

**Attribution:** commit as the contributing maintainer; state AI authorship in the
PR body per `CONTRIBUTING.md` §Authorship (this is an AI-first repo — AI authorship
is the norm), not as a tool/bot signature trailer in the commit.

## CI-green preflight — run locally before every PR
CI is **entirely GPU-free** (stubs `libcuda`/`libnccl` so no-GPU runners link).
The kernel work happens on your box; CI only proves the Rust compiles + hygiene.
Run exactly what CI runs, in this order — a red preflight is a red PR:

```bash
# 1. fmt          (ci.yml: fmt)
cargo fmt --all -- --check
# 2. clippy       (ci.yml: clippy — deny warnings comes from [workspace.lints], so no -Dwarnings needed)
ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo clippy --workspace --tests
# 3. license hdrs (SPDX AGPL-3.0-only line 1 of every new .rs/.cu/.cuh/.h/.hpp/.cpp;
#    wraps the same apache/skywalking-eyes engine ci.yml runs against .licenserc.yaml)
bash scripts/check-license-headers.sh
# 4. typos        (ci.yml: typos)
typos                                   # cargo install typos-cli
# 5. unit tests   (ci.yml: test — non-#[ignore] only, GPU-free)
ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo test --workspace --locked
# 6. file-size    (file-size-cap.yml: ≤500 LoC per crates/**/*.rs, allow-list aside)
find crates -name '*.rs' -not -name '*.bak' -not -path '*/target/*' | xargs wc -l | awk '$1>500 && $2!="total"'
# 7. deny         (security.yml: cargo-deny)
cargo deny check advisories licenses sources bans
```

CI job → check map (so a red check tells you which command to rerun):
`fmt` · `clippy` · `license-headers` (skywalking-eyes, `.licenserc.yaml`) ·
`typos` · `test` · `test-macos-metal` (asserts the metal binary links **neither**
libcuda nor libnccl) · `File size cap` · `Security Audit` (weekly + on Cargo.*
changes) · `Docs` · `Site`.

⚠️ **CI-green ≠ shippable.** None of the above boots a model. Only
`/atlas-release verify` (the serve matrix) proves the image serves. Keep the two
gates distinct in your head and in PR descriptions.

## CLA cleanup — "the CLA check isn't ideal, clean this up"

### What it does today (`cla.yml`)
`contributor-assistant/github-action@v2.6.1` posts a comment on any PR whose
author is neither an existing signer (`signatures/version1/cla.json` on the
`cla-signatures` branch) nor in the `allowlist`, and sets a **failing status
check that blocks merge** until they reply with the exact phrase
`I have read the CLA Document and I hereby sign the CLA`. A custom "Prepare Dynamic
Signature Match" pre-step scrapes PR comment history via `gh api` to re-inject the
signature on `synchronize`/push so re-pushes don't wipe it.

### Why it's friction for an AI-first repo
- The repo philosophy is "**all PRs are AI-generated**" — yet the CLA gates every
  PR whose author isn't allowlisted, adding a manual comment ceremony to
  automation the maintainer owns.
- The `allowlist` originally covered only **bots** (`dependabot[bot]`,
  `renovate[bot]`, `google-labs-jules[bot]`, `claude[bot]`, plus bare
  `google-labs-jules`/`claude`). The maintainer's own human accounts (`AzeezIsh`,
  `tbraun96`) were **not** on it — so their AI PRs depended on the mutable
  signature-state file rather than a stable rule.
- The dynamic-signature bolt-on is a maintenance smell: it papers over the
  ceremony not persisting across re-pushes, depends on comment order (`.[-1]`) and
  heredoc delimiters, and `cla.json` was hand-seeded with synthetic timestamps.

### The cleanup
1. **✅ Applied — allowlist the maintainer-controlled identities.** This pass added
   `AzeezIsh,tbraun96` to `cla.yml`'s `allowlist` so their (AI-generated) PRs never
   touch the CLA path. Rationale: the maintainers already own the copyright and have
   signed — the gate adds nothing on their own PRs, and the allowlist is a stable
   rule where the `cla.json` state is a mutable file on a side branch. (Left in the
   working tree for maintainer review before commit — `cla.yml` is a
   `pull_request_target` security surface, so it does not auto-push.)
2. **Optional — make the CLA informational, not required.** Keep the bot's comment
   but drop the CLA status from the required-checks set, so it never *blocks* an AI
   PR — external human contributors still get asked, and the record is still written.
3. **Optional — retire the dynamic-signature bolt-on.** Now that maintainer
   identities are allowlisted they no longer hit that gated path; it remains only to
   survive re-pushes for external first-time human contributors. Keep it (with the
   note added to `cla.yml`) or remove it if (2) lands.

Whichever path: keep external first-time human contributors covered (the CLA is a
real legal instrument — `CLA.md` grants the Enterprise-Edition re-license right,
and the signing ceremony is the only recorded consent). The goal is to stop
gating the maintainer's own automation, not to remove the CLA.

## CI convergence — the drift to reconcile
The maintainer flagged "our PRs and CI are all over the place." The concrete,
low-risk reconciliations (the convergence pass applies the unambiguous ones to the
working tree, unpushed, for review):

| Drift | SSOT / fix |
|-------|-----------|
| **File-size cap: 250 vs 500.** `AGENTS.md` + the global guide say "≤250 lines"; `CONTRIBUTING.md` + `file-size-cap.yml` enforce **≤500 LoC**. | CI is SSOT → **500**. Fix `AGENTS.md` to `≤500` and cross-reference the workflow. (Applied.) |
| **Dead links.** `docs/EP2-TROUBLESHOOTING.md` (never created) was referenced by `docs/DEPLOYMENT.md`, `book/src/operations/multi-gpu.md`, and `docs/adr/0007-tp-ep-composition.md`. | All three repointed to `docs/GB10_DEPLOYMENT_GUIDE.md` §7. (Applied.) |
| **Dead script.** `AGENTS.md`, `.github/pull_request_template.md`, and `book/src/project/contributing.md` all told contributors to run `scripts/check-license-headers.sh`, which didn't exist — CI uses `apache/skywalking-eyes`. | Created `scripts/check-license-headers.sh` as a thin wrapper over that same engine, so all three references now resolve. (Applied.) |
| **Image provenance.** `release.sh`/`release.yml` pass `--build-arg ATLAS_GIT_SHA` but the Dockerfile had no `ARG`, so it was dropped. | `ARG`+`LABEL org.opencontainers.image.revision` added to `docker/gb10/Dockerfile`. (Applied.) |
| **Action-version drift.** `release.yml` uses `checkout@v4` + `upload-artifact@v4`; the rest use `checkout@v6` / `upload-artifact@v7` / `download-artifact@v8`. | Pin all workflows to one major per action. (Recommend — touch when next editing release.yml.) |
| **`cap` is eroding.** The allow-list has grown to ~40 files, several far over 500 (metal_backend 734), which only warn. | Machine-enforce the "rationale + tracking issue per addition" policy, or schedule splits. (Recommend.) |
| **No image in CI.** `release.yml` ships tarballs only; the image is manual. | The Stage-1→3 self-hosted workflow in `references/pipeline.md` §Closing the gap. |
| **Required-check trap.** `security.yml`/`docs.yml` deliberately drop the PR path filter so kernel-only PRs aren't left permanently "expected"; `site.yml` still has one. | If `Site / Build` is ever made required, give it the same no-path-filter treatment. (Recommend.) |
| **Version scheme ×3.** crate `0.1.0` vs image `alpha-2.x`/`3.0.0` vs distro `vX.Y.Z`. | Anchor the image to the git SHA tag (`references/pipeline.md` §Image); let moving tags point at it. |
