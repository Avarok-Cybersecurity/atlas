# WIP: Kimi K3 architecture bring-up (do not merge)

> **This PR is not expected to merge.** It is a historical working log. The branch will stay messy on purpose — dead-ends, WIP commits, lab comments, half-written modules — until we cut clean topic PRs *off of it*. Do not `/stamp`, `/seal`, squash-merge, or review it as a landing candidate. Reviewers: ignore mergeability; use the conversation as the notebook.

Working log for Atlas K3. **Draft only.** Winning slices are extracted later into atomic certified PRs. See `docs/k3/PRD.md`.

## Summary

Working log for Kimi K3 architecture bring-up on lab hardware (spark1 / spark2 / 5090). **Not a merge candidate. Expected to be historical and messy** until new PRs are opened from this branch. Architecture is completed at home; rental silicon is a soak after C0–C7.

Closes #

## Status

| Field | Value |
| --- | --- |
| Phase | S0 — harness / fabric / C0 |
| Last host | workstation (Mac) + spark1 + spark2 + train (5090) |
| Last session | 2026-09-11 |
| Official weights downloaded in lab | **no** |
| Rental booked | **no** |

## Rental gate

S7 (rental soak) is forbidden until every box is green.

- [ ] C0 config / factory / weight-name map dry-run (no shard download)
- [ ] C1 `Kimi-K3-0.40B` greedy 128 tok × 8 prompts, exact vs HF
- [ ] C2 prefill-then-decode vs full-prefill logits (atol/rtol in test file)
- [ ] C3 prefix-cache hit == no-cache decode
- [ ] C4 MLA KV and KDA state advance on the same positions (incl. after prefix hit)
- [ ] C5 AttnRes fixture max-abs-err bound
- [ ] C6 LatentMoE top-k + mix vs frozen gates
- [ ] C7 production-width dummy TP=2 on spark1+spark2 == spark1 single-GPU tokens

49M `smol-kimi-k3` is shape-only. It does not satisfy C1.

## Lab hosts

Named in the PRD. Iface / `NCCL_IB_HCA` live in `docs/k3/LAB.md`. **Do not put lab IPs in this PR.**

| Host | Role |
| --- | --- |
| spark1 | head / rank 0 / `:8888` / NCCL master |
| spark2 | worker / rank 1 |
| train / 5090 | correctness GPU only; SM120 launch test **failed** (see LAB.md) — PyTorch/shape-debug only |

Constraint: spark1+spark2 ≈ 240 GB UMA. Official `moonshotai/Kimi-K3` MXFP4 ≈ 1.561 TB. Two Sparks cannot load official K3.

Fabric pin (live): `NCCL_SOCKET_IFNAME=enp1s0f1np1` `NCCL_IB_HCA=rocep1s0f1`. `enp1s0f0np0` is Down. Passwordless SSH spark1 ↔ spark2: yes. RoCE ICMP 0% loss.

## File map

| Path | Purpose |
| --- | --- |
| `docs/k3/PRD.md` | Binding PRD |
| `docs/k3/UMBRELLA.md` | This file / PR body |
| `docs/k3/LAB.md` | Host inventory, NIC pins, NCCL proof |
| `docs/k3/BAKEOFF.md` | Harness schema |
| `docs/k3/RENTAL.md` | Filled only after C0–C7 |
| `kernels/gb10/kimi-k3/MODEL.toml` | Target skeleton |
| `kernels/gb10/kimi-k3/bf16/` | Twin + dummy (empty in this commit) |
| `kernels/gb10/kimi-k3/mxfp4/` | Fixtures only in lab (empty in this commit) |
| `crates/spark-model/src/weight_loader/kimi_k3.rs` | Loader (not yet) |
| `crates/spark-model/src/kimi_k3/` | Graph: layer, kda, mla, attnres, latent_moe, situ, cache (not yet) |

## Decision log

- K3-DECISION: KDA is a new backend, not a GDN/Mamba reuse.
- K3-DECISION: C1 token-exact reference is `inference-optimization/Kimi-K3-0.40B`. 49M is shape-only.
- K3-DECISION: Reuse DeepSeek-V4 MXFP4 E8M0 path; do not invent a second stack.
- K3-DECISION: Umbrella never merges. It is historical and messy by design. Extract `feat/k3-*` slices; do not tidy this branch for landing.
- K3-DECISION: GLM-5.3-Flash KDA geometry *matches* K3's `validate()` numbers (see the 2026-09-10 gap analysis) but that is not permission to copy GDN/Mamba kernels into `kernels/gb10/kimi-k3/`. Adapt only behind goldens.
- K3-LAB: pin `enp1s0f1np1` / `rocep1s0f1`. Do not copy `start-ep2.sh`'s `enp1s0f0np0` comment.

## Dead-ends

_None yet. Record them here so they are not retried on rented silicon._

## Extraction plan

| Future PR | Slice | Gate | Status |
| --- | --- | --- | --- |
| `feat/k3-config-loader` | MODEL.toml + config + safetensors map | C0 | not started |
| `feat/k3-kda` | KDA module + state | C4 | not started |
| `feat/k3-mla-gated` | gated MLA + NoPE | C1/C4 | not started |
| `feat/k3-attnres` | block residual mix | C5 | not started |
| `feat/k3-situ-latentmoe` | SiTU-GLU + latent down/experts/up | C6 | not started |
| `feat/k3-hybrid-cache` | paged MLA KV + KDA state + prefix | C3 | not started |
| `feat/k3-mxfp4` | packed expert loader + one-shard fixture | S5 | not started |
| `feat/k3-dual-spark` | TP/EP path | C7 | not started |
| `docs/k3-bakeoff-rental` | BAKEOFF + RENTAL | after C0–C7 | not started |

## Adjacent Hopper work (not this PR)

Today's Hopper / native-FP8 topic PRs stay on their own branches. Do not dump them onto `wip/k3-bringup`. Index (TheTom, open as of 2026-09-11, Avarok-Cybersecurity/atlas):

- #996 `pr/hopper-b200-targets` — Hopper sm_90a + B200 sm_100a targets (supersedes draft #895)
- #1013 `pr/hopper-decode-splitk`
- #1016 `pr/hopper-decode-kernels`
- #1017 `pr/hopper-prefill-kernels`
- #1018 `pr/hopper-fp8-act-quant`
- plus the native-FP8 stack (#984–#1019) and tbraun96's campaign PR #1012

## Test plan

Lab gates C0–C7. Workspace tests stay green via feature-flag / ignore on **new** tests only.

- [ ] `cargo fmt --all -- --check` (docs + MODEL.toml only in this commit)
- [ ] `ATLAS_SKIP_BUILD=1 cargo clippy --workspace --tests --all-features -- -Dwarnings`
- [ ] `bash scripts/check-license-headers.sh`
- [ ] `python3 scripts/check_kernel_shadows.py`
- [ ] `python3 scripts/check_cross_hardware.py --base avarok/main --worktree`
- [ ] Tested against a real model / hardware if the change affects runtime behaviour — N/A this commit
- [ ] Added or updated tests where applicable — N/A this commit (C0 skeleton is S1)

## Notes for reviewers

**Do not merge this PR.** Do not `/stamp` or `/seal`. Do not expect a clean history, a 500-LoC-clean tree, or a squash. This is a lab notebook that will accumulate messy commits until we open new PRs off of it. Use the conversation as the log. Extract atomic certified PRs later; those are the ones that land.

## Authorship / CLA

AI-authored working log, per Atlas default. CLA will be checked on extracted topic PRs. **Will not `/stamp` or `/seal` or merge this PR.**

- [x] I have read and agree to the [Contributor License Agreement](../CLA.md).
