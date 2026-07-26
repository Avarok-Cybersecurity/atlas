# GB10 concurrency campaign — STATE (durable; survives session crashes)

**Goal:** beat vLLM at C=[1,2,4,8,16] — aggregate tok/s at every C AND TTFT/TPOT p50/p99 not losing.
**Plan:** `/workspace/.claude/plans/validated-zooming-bentley.md` (approved 2026-07-25).
**Runtime dir:** `/workspace/.wt-golden/conc_sweep/` (driver logs + per-leg results json).

## How to resume after a session crash
1. Read this file — the Log below is appended by the DRIVER after every leg (not by the session).
2. `tail /workspace/.wt-golden/conc_sweep/phaseA.log` — look for `LEG_DONE <name>` / `PHASEA_DONE` /
   `SERVE_DIED`.
3. Drivers are resumable: re-running `bench/phaseA_c_sweep.sh` skips any leg whose
   `conc_sweep/results/<leg>.json` already exists.
4. Re-arm the convenience monitor on `conc_sweep/*.log` (filter:
   `LEG_DONE|_DONE|SERVE_DIED|Traceback|CUDA error|out of memory`).

## Configuration of record
- Box: dgx1. Model: 27B (Atlas: `centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf`; vLLM: same checkpoint if it
  loads, else `nvidia/Qwen3.6-27B-NVFP4` — the leg json records which).
- Atlas serve: golden flags + `--max-batch-size 16`, fifo scheduling (SLAI starves prefill at load),
  env incl. `ATLAS_MTP_GATE_FORCE=1`. Binary: pushed tip of PR #369.
- vLLM serve: `sparkrun-eugr-vllm:latest` (vLLM 0.23.1rc1.dev207), `--max-num-seqs 128`,
  `--max-model-len 32768`, util 0.85.
- Synthetic scoreboard: `bench/bench-atlas-concurrency.py`, C=[1,2,4,8,16], default 4 ISL/OSL
  regimes (≤4096); agentic-harness `target_concurrency` sweep follows as a second driver.

## Log (appended by drivers)
5048fa13d69c2870420a8b9050f54221  conc_sweep/spark_phaseA_baseline
- 2026-07-25T18:13:50Z LEG atlas_synth SERVE_DIED
- CONFIG CHANGE after first atlas_synth SERVE_DIED: --max-batch-size 16 + slots 128 + nd=3 needs
  ~52G of SSM reservations (seq-state 14.2G + rollback ring 18.9G + Marconi 18.9G) + 17.5G weights
  before ANY KV — preflight refusal territory at util 0.70. New atlas C-config: --max-batch-size 20
  (headroom over C=16; pool-boundary exhaustion KILLS requests) + --ssm-cache-slots 32 (synthetic
  sweep has no multi-turn reuse; 4.7G) = 46.2G SSM. Driver now captures a deathlog on serve failure.
- 2026-07-25T19:15:28Z LEG vllm_synth DONE on centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf -> results/vllm_synth.json
- 2026-07-25T19:15:34Z PHASEA_DONE
- 2026-07-25T19:17:44Z LEG atlas_synth SERVE_DIED (deathlog: conc_sweep/atlas_synth.deathlog)
- 2026-07-25T19:17:47Z PHASEA_DONE
- SECOND atlas_synth death, deathlog decisive: "39.7 GB consumed + 50.5 GB inference reserve =
  90.2 GB committed" vs 85.2 budget at bs=20/nd=3. Fix: bs=16 (saves ~8.3G: seq-state 3.6 + ring
  4.7) AND --max-seq-len 4096 (the sweep's regimes cap at ISL+OSL=2048; 32768 was inflating
  max_blocks_per_seq metadata and KV expectations for no benefit). nd=3 kept for C=1 K=4 fairness.
51fc31d43a6e59aec8e9eaced56a02b2  conc_sweep/spark_phaseB
5048fa13d69c2870420a8b9050f54221  conc_sweep/spark_phaseA_baseline
- 2026-07-25T22:10:42Z LEG atlas_synth DONE -> results/atlas_synth.json
- 2026-07-25T22:10:46Z PHASE A compare written -> results/compare.txt
- 2026-07-25T22:10:46Z PHASEA_DONE
- 2026-07-26T00:40:49Z LEG atlasB_nographs DONE -> results/atlasB_nographs.json
- 2026-07-26T02:57:30Z LEG atlasB_graphs DONE -> results/atlasB_graphs.json
- 2026-07-26T02:57:34Z PHASEB_DONE
