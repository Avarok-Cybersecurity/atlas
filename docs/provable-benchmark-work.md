# Proving Benchmark Work to a GPU-less Verifier
## A layered design for the Atlas gate record pipeline

---

## 1. The Honest Threat Model

The verifier is `check.rs` running on `ubuntu-latest`. It has no GPU, no GB10, no trusted clock on the bench box, and no independent party anywhere in the loop. Every design decision below follows from that.

**(a) Accidental inclusion of wrong records — SOLVABLE, and the actual job.**
Stale records, records for the wrong benchmark directory, records taken on a dirty tree, records whose sha doesn't cover head, records missing a pinned serve override. The current scheme already catches most of this; the layers below make accidents *nearly impossible*: a record that wasn't produced by the CLI, after the PR opened, from a real transcript tree, will not assemble the required fields by accident.

**(b) A careless operator taking a shortcut — MOSTLY SOLVABLE.**
"Re-run just the failing leg and paste the numbers in," "reuse last week's record with a new sha," "run the 4096-max-seq variant that drops 20 hard samples." These are defeated by binding the record to a per-PR nonce, to a committed transcript tree with N and an ordered-sample-id SHA, and to arithmetic consistency between per-sample data and the aggregates. Shortcuts stop being *easier* than the honest run — which is the correct bar for this attacker.

**(c) A determined insider with the hardware — NOT SOLVABLE, and we must say so.**
The adversarial exercise broke every scheme evaluated, including the full layered stack, with the same move: *replay-and-relabel*. Take genuine transcripts from a prior honest run (temp 0.0 / seed 42 makes them valid forever), rewrite only the timing fields to be arithmetically consistent by construction, receive the nonce, sign on dgx1, co-sign on dgx2 — both keys are the same operator's. Cost: ~1–2 CPU-hours, ~$0 GPU. Every check the GPU-less verifier can run passes; every check a future GB10 auditor can run *also* passes, because only the timing — the one quantity with no possible cryptographic witness — was forged. No zkML proof fixes this (a zk proof shows *what* was computed, never *how fast*), no TEE on this silicon exists, and the "second party" is us. Timing floors are, and will remain, self-reported numbers whose honesty rests on attribution, auditability, and independent reproduction.

**The strongest honest claim this design can make:** it makes accidents nearly impossible, makes shortcuts more expensive than honest runs, makes accuracy metrics audit-hard forever, and makes any forgery premeditated, multi-step, externally timestamped, and non-repudiably attributable to a named key — but it does not make forgery impossible, and it cannot, on any hardware we own, prove timing to anyone.

---

## 2. What the Current Scheme Actually Proves

The README claim — *"Records are Ed25519-signed against the commit that produced them, so one cannot be edited or re-pointed at another commit."* — overstates in three ways:

1. **"the commit that produced them"** implies the signature witnesses production. It does not. The signer freely chooses both the record bytes and the `git_sha`; the signature proves only that a key-holder vouched for that pairing. Nothing binds the sha to any execution — any commit whose tree diff vs. head avoids `PERF_PATHS` satisfies the check, and it is fabricable in minutes.
2. **"one cannot be edited"** is true only for *third parties*. The key-holder can edit and re-sign at will; keys are self-registered TOFU (`.github/record-signers/<fp>.pub` lands in the same PR), so "key-holder" means "whoever opened the PR."
3. It omits the exemption: any record with `recorded_at < 1_788_268_400` verifies as `Verified::Exempt` with **no signature at all**, and `check.rs` never cross-checks `recorded_at` against the filename date. A backdated forgery currently needs no key whatsoever.

**Exact replacement wording:**

> Records are Ed25519-signed over the record bytes and the named commit. This guarantees that no one *other than a registered key-holder* can alter a record or re-point it at a different commit after signing, and that every record is attributable to a specific key in `.github/record-signers/`. It does not prove that the named commit produced the record, that the benchmark ran, or that the metrics were measured — the signer attests those, and the signature makes that attestation non-repudiable.

---

## 3. Is ZKP the Answer?

**No. Rejected on arithmetic, not on taste.**

- zkLLM (CCS'24, best published system) proves a **13B** forward pass on an A100 in ~1,789 s (986 s commit + 803 s prove). Linear extrapolation to 27B: **~3,716 s per forward pass**.
- The agentic run is 1007 samples × ~200 output tokens ≈ **208,000 GPU-hours ≈ 23.7 GPU-years** to prove a **1.79-hour** benchmark — a ~116,000× overhead. General zkVMs (SP1, RISC Zero, Jolt) are 10⁵–10⁶× and worse.
- No zkML framework has circuits for NVFP4 (E2M1 + E4M3 block scales), the qwen3.8 GDN/hybrid spine, or sm_121 kernels. Building them is a research program, not an integration.
- GB10's 273 GB/s LPDDR5X would inflate the bandwidth-bound commitment phase a further ~3–5× vs. A100 HBM.
- **Decisive even in the counterfactual:** a zk proof attests the function, never the wall clock. The most-gamed metric class — decode floors, TTFT, wall — is outside the reach of *all* verifiable-computation schemes. The same is true of TEEs, which are moot anyway: GB10 has no CC mode (`nvidia-smi conf-compute` unrecognized on 580.126.09), the Cortex-X925/A725 CPU has no Arm CCA, and `/sys/class/tpm` is empty. Both angles should be closed with a documented negative in the design doc so no future session re-litigates them.

---

## 4. The Layered Design (ordered by value per unit effort)

### Layer 0 — Close the backdating exemption and filename cross-check. **DO FIRST.**
- **Proves:** nothing new; removes the *no-signature-needed* forgery path (`Verified::Exempt`) and the filename/`recorded_at` mismatch gap.
- **Costs:** ~20 lines in `check.rs`. An afternoon.
- **Doesn't stop:** anything signed. This is a bug fix, not a feature.
- **Verdict: mandatory.** The current cheapest attack ("type any f64, date it last month, skip the .sig") dies here.

### Layer 1 — Per-PR challenge nonce, commit-reveal (schema v2).
- **Proves:** the record was *assembled after the PR opened*. Kills pre-baked and backdated records structurally, not just by cutover date.
- **Costs:** one small GHA job minting 32 random bytes into a check-run output; `transcript_hmac = HMAC-SHA256(nonce, merkle_root)` via `ring::hmac` (already in the tree); a schema-v2 check in `check.rs`. Days.
- **Doesn't stop:** replaying old *transcripts* into a fresh assembly; proves assembly time, not execution time.
- **Verdict: do it.** Highest value/effort of the new layers.

### Layer 2 — Transcript Merkle root + arithmetic consistency.
- **Proves:** the aggregates are consistent with 1007 committed per-sample records: Σ per-sample latencies ≈ wall (±ε), Σ tokens / Σ decode-time ≈ `server_decode_tok_s`, N and the ordered-`sample_id` SHA match the pinned draw. This mechanizes what CLAUDE.md already demands manually, and it would have made the max-seq-len=4096 sample-dropping (the inflated BFCL 87.24) visible in the committed tree.
- **Costs:** harness writes per-sample JSON; SHA-256 Merkle over ~1007 leaves (<1 s CPU); tree as CI artifact, 32-byte root in the signed record; ~150 lines of recomputation in `check.rs`.
- **Doesn't stop:** a forger who fabricates 1007 *mutually consistent* transcripts — internal coherence is under the forger's control. It raises effort from "one f64" to "a consistent fake corpus," which is the honest description.
- **Verdict: do it.** Collapses the forgeable surface from every field to the transcript tree itself.

### Layer 3 — Rekor external timestamp on the record hash.
- **Proves:** a third party observed the record hash at a specific time (append-only public log). Kills clock forgery independently of GitHub.
- **Costs:** one HTTPS POST (`hashedrekord`) from the CLI after signing, log index in the `.sig` sidecar; one GET + signature check in CI (~2 s). No heavy crate.
- **Doesn't stop:** anything about execution; timestamps assembly only. **Do not** migrate to full Sigstore keyless (Fulcio/DSSE) — the adversarial pass showed author-then-sign forges it identically, and the Ed25519 chain already provides attribution. Take the timestamp, skip the identity migration.
- **Verdict: do it — the timestamp only.**

### Layer 4 — TOPLOC-style activation commitments.
- **Proves:** to any *future GB10 owner* (not to CI): the committed outputs are genuine outputs of the committed checkpoint on the committed prompts. Top-128 last-hidden-state values per 32 decode tokens (~258 B/32 tok, 1–5 MB/run, <1% overhead) in the same Merkle tree; prefill validation is ~100× faster than generation. Makes accuracy metrics (IoU, BFCL) permanently audit-hard and exposes swapped-model or patched-CLI runs.
- **Costs:** a serve-path hook on the last hidden state (materialized every decode step anyway), plus one afternoon calibrating exponent/mantissa tolerances for NVFP4 on known-good dgx1-vs-dgx2 replays.
- **Doesn't stop:** timing forgery at all — a replayed genuine run's TOPLOC blobs validate forever. Proves "this model produced these tokens," never "on this hardware at this speed."
- **Verdict: do it,** and label it honestly as *auditability*, not verification.

### Layer 5 — Second-box co-signature spot check (PoSP-lite).
- **Proves:** a second registered key re-ran k=16 nonce-selected samples and matched. ~50 lines in `gate/signing.rs`, one `check.rs` rule, 2–6 GPU-minutes on the idle box.
- **Doesn't stop:** us. Both keys belong to one operator; this is *not* independence and must never be described as such. It raises forgery to a two-key, premeditated act and catches one box silently misconfigured — real but modest value.
- **Verdict: do it, cheaply, with honest labeling.** If a genuinely independent GB10 owner ever joins, this layer's plumbing becomes the real thing.

### Recommended AGAINST
- **zkSNARK/zkVM proofs** — §3. Commit the negative to the design doc.
- **TEE/hardware attestation** — impossible on GB10 (no CC, no CCA, no TPM); even on H100-class it attests firmware, not workloads, and would attest the wrong silicon. Commit the negative with the on-box evidence.
- **Sigstore keyless identity migration** — forged identically by author-then-sign; keep Ed25519 + Rekor timestamp.
- **Fabricated-plausibility checks on `hardware_state`** — CI cannot verify a GPU it doesn't have; adding fake rigor invites false confidence. Leave it disclosure-only, documented as such.

---

## 5. Concrete Implementation Plan for This Repo

**Phase 0 (bug fixes, `crates/atlas-plugin/src/gate/check.rs`):**
1. Reject `Verified::Exempt` for any record *added in the PR diff* regardless of `recorded_at` (grandfather only pre-existing committed records).
2. Cross-check filename date (`YYYY-MM-DD-<sha>.json`) against `recorded_at` (±48 h).
3. Change `record_is_for` mismatch from *skipped* to *failed* for records added in the diff.

**Phase 1 (nonce, schema v2):**
- `.github/workflows/bench-nonce.yml`: on PR labeled `bench`, mint 32 random bytes, post as check-run output with `issued_at`.
- `record.rs`: add `challenge_nonce: [u8;32]`, `transcript_root: [u8;32]`, `transcript_hmac` (`ring::hmac::HMAC_SHA256`), bump `schema` to 2.
- `check.rs`: for schema ≥ 2, fetch the check-run, verify nonce match, `recorded_at ≥ issued_at`, and the HMAC.
- CI "One PR, one commit, one signer" shell step: extend to "…one nonce."

**Phase 2 (transcripts):**
- Bench harness: write per-sample `{prompt_sha256, response_token_ids, ttft_ms, latency_ms, n_tokens}`; Merkle-hash (SHA-256); upload tree as a CI artifact keyed by `transcript_root`.
- `check.rs`: download the artifact in the certification job; recompute Σ-latency vs. wall, token math vs. `server_decode_tok_s` and TPOT, N and ordered-`sample_id` SHA vs. the BENCH.toml pin. Tolerances (`ε`) go in `BENCH.toml`, not code defaults (per the BENCH.toml-thresholds lesson).

**Phase 3 (Rekor):**
- CLI post-sign step: POST `hashedrekord` of SHA-256(record) to `rekor.sigstore.dev`; store `log_index` + inclusion proof in the `.sig` sidecar. `check.rs`: verify inclusion, require `integratedTime ≥ nonce.issued_at`.

**Phase 4 (TOPLOC):**
- Serve-path hook: top-128 of last hidden state per 32 decode tokens → per-sample blob → same Merkle tree. Calibration run: dgx1 vs. dgx2 replay of a known-good record to set NVFP4 tolerances; commit thresholds to `BENCH.toml`.
- Ship `atlas bench audit <record>`: any GB10 replays k samples via prefill and checks commitments.

**Phase 5 (co-sign):**
- `gate/signing.rs`: accept a second detached signature over identical record bytes; peer derives k=16 indices as `HMAC(nonce, "audit") mod N`, re-runs, checks TOPLOC agreement, signs. `check.rs`: require two distinct registered fingerprints for records after a new cutover constant.

Total: days of engineering, <1 minute of added CI time, <1% bench overhead.

---

## 6. What to Say in the README Instead

> **What the gate verifies — and what it cannot.**
>
> Benchmark records are Ed25519-signed over the record bytes and the commit they name. No one other than a registered key-holder can alter a signed record or re-point it at another commit; every record is non-repudiably attributable to a key in `.github/record-signers/`.
>
> Records additionally commit to: a per-PR challenge nonce (so a record cannot predate the PR that lands it), a public Rekor timestamp (so its creation time is third-party-attested), and a Merkle root over per-sample transcripts whose token counts, latencies, and sample draw CI recomputes against the claimed aggregates. Activation commitments in the transcript tree let anyone with a GB10 audit any sample, indefinitely, at ~100× less than the cost of the original run; a second box on our bench co-signs a nonce-selected spot check.
>
> Be clear about the limits. CI runs without a GPU: it verifies signatures, freshness, and internal consistency — it does not witness execution. Accuracy metrics (IoU, BFCL) are audit-hard: fabricating them requires fabricating model outputs that hardware audits expose. Timing metrics (wall, TTFT, tok/s) have no possible cryptographic witness on any hardware we own — no zero-knowledge proof or trusted-execution scheme attests a wall clock, and GB10 has no attestation hardware at all. Timing numbers are therefore signed, timestamped, internally consistent *claims*, checkable only by independent re-runs of the public benchmark on independent hardware — which we invite.
>
> In one sentence: this pipeline makes accidental inclusion of wrong records nearly impossible, makes shortcuts costlier than honest runs, and makes forgery a premeditated, externally-logged, attributable act — but it does not make forgery impossible, and we do not claim otherwise.