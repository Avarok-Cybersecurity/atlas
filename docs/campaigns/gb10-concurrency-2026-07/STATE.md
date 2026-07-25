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
