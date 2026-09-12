# RST at every K3 phase

Rapid Software Testing (Bach/Bolton). **A C-gate or S-rung is not green without a session sheet.** Checks (unit tests, JSONL, NCCL benches) are instruments. They are not the testing.

Private lab oracles and session notes may cite the vault (`RST/`). This file is the in-tree contract so extracted PRs inherit it. Do not paste courseware.

## Five rules (override everything)

1. **Name the oracle — including the instrument.** A "clean" run from an unvalidated harness is not evidence. Hit a known-bad case first.
2. **Observe, do not infer.** Cite the line, log, HTTP status, or tensor name.
3. **Coverage is relative to a model** (SFDIPOT / C-gates), never a test count.
4. **Stopping is a decision.** Name the heuristic on the sheet.
5. **Evidence beats instruction** — including this PRD.

## Required per phase (S0–S7 and each C0–C7)

Before claiming the phase done, land `docs/k3/rst/<phase>-<slug>.md` with:

| Field | Required |
| --- | --- |
| Charter | 1–3 sentences. Mission, not a script. |
| Areas | Product elements / C-gate / host |
| Oracle | How a problem would be recognized. At least one of: HF twin, self-consistency, comparable engine (vLLM), claims (PRD/MODEL.toml), product (other Atlas models), known-bad mutant |
| Known-bad | The instrument was shown to fail on a planted defect *before* a clean result is trusted |
| Notes | Observed, not inferred |
| Bugs / issues | `#BUG` or `#ISSUE` or `#N/A` |
| Stop | Named heuristic (no more time / no new news / risk parked / charter complete) |

Run **independent charters in parallel** when they do not contend for the same GPU. Typical split: spark1 serve / spark2 vLLM or NCCL worker / workstation parser tests / 5090 PyTorch-only.

## Phase charters (start here; rewrite if the product changes)

| Phase | Charter | Primary oracle | Known-bad |
| --- | --- | --- | --- |
| S0 fabric | Find whether dual-Spark RoCE actually carries NCCL, and what happens if the pin is wrong | Transport string `NET/IB` + nonzero algbw | Unpinned / Down HCA (`enp1s0f0np0`) must *not* silently look like success |
| S0 5090 | Find whether SM121 images run on the 5090 | CUDA launch error string | Control: `sm_120` cubin **must** launch |
| S0 bake-off | Find whether Atlas and vLLM both speak OpenAI on a shipped MoE; do not treat tok/s as K3 news | HTTP + greedy text; comparable-product (vLLM) | Wrong model id, empty messages, `max_tokens=0` |
| C0 | Find whether official `config.json` + 96-shard class map are actually parsed and refused when broken | Vendored official JSON; TSV class list | Drop `text_config`; omit one required tensor class; 95 shards |
| C1 | Token-exact vs HF on 0.40B | HF greedy 128 tok × 8 prompts | Prompt that HF answers and a mutated graph that must diverge |
| C2 | Prefill-then-decode vs full-prefill logits | atol/rtol in the test file | Skip a layer on one path only |
| C3 | Prefix-cache hit == no-cache decode | Bit/token identity | Stale block after a cache write |
| C4 | MLA KV and KDA state advance on the same positions | Fixture + HF | Prefix-hit then wrong state slot |
| C5 | AttnRes mix bound | Frozen residual fixture | Zero the mix weights |
| C6 | LatentMoE top-k + mix vs frozen gates | Frozen gate vector | Force expert 0 |
| C7 | Dummy TP=2 == single-GPU tokens | spark1 vs spark1+spark2 | Kill rank 1 mid-decode |
| S7 rental | Soak only. RST here is "is the rental the product we already tested" | C0–C7 receipts | Do not debug architecture on the rental |

## What RST is not

- Not permission to skip `cargo test`
- Not a tok/s number
- Not `/stamp`
- Not comparing spark dummy tok/s to GB300-NVL72 marketing

## Sheets

| Sheet | Status |
| --- | --- |
| `docs/k3/rst/s0-fabric-nccl.md` | done 2026-09-11 |
| `docs/k3/rst/s0-5090-sm121.md` | done 2026-09-11 |
| `docs/k3/rst/s0-atlas-serve.md` | done 2026-09-11 |
| `docs/k3/rst/s0-vllm-bakeoff.md` | done 2026-09-11 (TRITON_ATTN, think-off JSONL) |
| `docs/k3/rst/c0-config-loader.md` | **green** 2026-09-12 spark2 9/9 |
| `docs/k3/rst/s1-graph.md` | in flight |
