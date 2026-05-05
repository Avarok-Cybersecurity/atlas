# Release Notes

Atlas's release notes are per-version markdown files in [`docs/releases/`](https://github.com/Avarok-Cybersecurity/atlas/tree/main/docs/releases). This chapter links to them and summarises the big themes across recent alphas. For the latest release, check the repo — this page is a stable pointer, not a ticker.

## Release naming

`alpha-<major>.<minor><letter>` — e.g. `alpha-2.43`, `alpha-2.44`, `alpha-2.14c`. Minor versions bump on any meaningful feature or fix; letters (`a`, `b`, `c`) are patch-level iterations on the same minor.

Since Atlas is pre-1.0 and under aggressive development, semantic versioning does not apply. Any release can break API or CLI compatibility — the per-release notes document what.

## Where to read them

| Source | URL |
|---|---|
| Release notes folder | [`docs/releases/`](https://github.com/Avarok-Cybersecurity/atlas/tree/main/docs/releases) |
| GitHub Releases | `https://github.com/Avarok-Cybersecurity/atlas/releases` (if tagged) |
| Docker Hub | `https://hub.docker.com/r/avarok/atlas-gb10/tags` |

The multi-model Docker image always tracks the latest alpha at `avarok/atlas-gb10:latest`. Specific versions are tagged as `avarok/atlas-gb10:alpha-2.44` etc.

## Recent themes

Rather than duplicate every release note, here's the shape of recent work. Each theme maps to a set of design docs under `docs/design/` and bug-sweep notes in history.

### alpha-2.0 → alpha-2.20 — coherence and the model matrix

The long effort to get the full 12-model matrix to pass an end-to-end coherence suite. Highlights:

- Fast safetensors loader (`InstantTensor`-style, `O_DIRECT` + pipelined).
- MTP speculative decoding landed for Qwen3.5-35B, Qwen3-Next-80B, Qwen3.5-122B.
- RadixAttention prefix caching + Marconi SSM snapshot caching.
- Qwen3-VL vision tower integration (ViT block + merger layer + MRoPE image position IDs).
- Nemotron-H Mamba-2 integration.
- Chunked SSM prefill (saves 7–9 GB for long-context prefill).

### alpha-2.20 → alpha-2.35 — MiniMax and the 256-expert problem

Getting MiniMax-M2 and M2.7 to pass:

- 256-expert sigmoid MoE routing (distinct from softmax-topk).
- EP=2 token dispatch kernel for >256 experts.
- `rms_norm` placement fix in the MiniMax MoE path.
- `norm_topk_prob` semantics (sum-normalised, not softmax).
- FP8-free enforcement on NVFP4 shared-expert path.
- Template-forced thinking detection that distinguishes MiniMax-style `<think>` seeding from Qwen-style.

alpha-2.35 was the first release where M2.7-NVFP4 EP=2 passed the full coherence + tool-call + TPS suite.

### alpha-2.35 → alpha-2.44 — OSS prep + bug sweeps

Thirteen waves of systematic audit-framework bug sweeps (`project_bug_sweep_wave1_2026_04_22.md` through `wave13`). Net effect:

- **Wave 1** — attention/SSM sibling stride bugs (K≠2 paths), slot-keyed CUDA graph caches, compact_sequence pointer leak.
- **Wave 5** — Responses flat-form tools, vision prefix-cache contamination skip, EP=2 MTP guard.
- **Wave 6** — NVFP4 MTP loader force-BF16 when `ignore_modules` lists `mtp.*`, FP8 prefill shared-experts allreduce reorder.
- **Wave 7** — SSM dummy slot defensive fix, qwen3 parser literal-`</tool_call>` + missing-`</parameter>` recovery.
- **Wave 8** — Sampler `temperature=0 && rep_penalty=0` div-by-zero guards, longest-first stop-sequence matching.
- **Wave 9** — Rate-limiter `MAX_KEYS` DoS guard, body-size env-configurable.
- **Wave 10** — Responses function_call(_output) items + instructions stacking, multi-block reasoning extractor, MoE topk bounds, weight loader scale=0 guard.
- **Wave 11** — Streaming Responses store tool_calls, balanced markdown URL parens (Wikipedia URLs), self-spec rollback fail-fast on SSM.
- **Wave 12** — Anthropic streaming `stop_sequence` populated, `/tokenize`+`/detokenize` gated under `ATLAS_REQUIRE_AUTH`.

Vision fixes (7 ViT + MRoPE image position IDs) landed all four vision models passing the Mona-Lisa test.

### Pass-N passes

Alongside the bug sweeps, "Pass-N" work is the systematic model-matrix regression suite. Each Pass runs the full 171-test suite across all models and all flag combinations. Milestones:

- **Pass-14** — 152/171 (88.9%), 10/19 perfect. 14 fixes delivered.
- **Pass-16** — 233/247 (94.3%), 15/19 perfect. Fixed the 80B-MTP `seq_len += k-1` off-by-one bootstrap.
- **Pass-21** — 233/247, 14/19. Tool-parser +4; 122B-nvfp4 LC regressed from minimax-m2 branch commits.
- **Pass-22** — 237/247 (96.0%), 15/19 perfect. HARDWARE.toml SSOT + workspace lints + Cluster B error propagation.

## What's next

OSS release prep (alpha-2.43-share) was the major non-code milestone: archive tags, `docs/design/` cleanup, `atlas-internal/` separation for proprietary artefacts. See `project_oss_prep.md` in history.

For the current roadmap, check the repo's pinned issues and the authoritative design docs at [`docs/design/`](https://github.com/Avarok-Cybersecurity/atlas/tree/main/docs/design).
