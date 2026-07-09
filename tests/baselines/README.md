# tps baselines

One `<label>.json` per serve-matrix spec (e.g. `35B-nvfp4.json`), each holding
`{"tps": <blessed tokens/sec>}`. `tests/gate_results.py` fails a model whose
measured tps drops more than `TPS_TOLERANCE` (10%) below its baseline — the
regression check that `tps > 0` alone can't give.

Regenerate deliberately, never implicitly:

```bash
python3 tests/run_all_models.py                    # measure on real GB10 hardware
python3 tests/gate_results.py --update-baselines    # write {"tps": avg} per label
git diff tests/baselines                            # review, then commit
```

Labels match the manifest roster in `tests/run_all_models.py`. A model with no
baseline is checked for liveness only (with a note) until one is blessed; pass
`--require-baselines` to make a missing baseline a hard fail for a milestone.

Credit: baseline-snapshot approach proposed by @Sujimoshi in #253.
