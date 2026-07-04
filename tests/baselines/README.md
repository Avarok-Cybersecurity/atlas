# TPS baselines

This directory holds one `<label>.json` file per release-matrix spec (e.g.
`27B-dense-nvfp4.json`), each containing `{"tps": <measured tokens/sec>}`.
Files are produced by running `pytest tests/test_release_matrix.py
--update-baselines` on real GB10 hardware, then reviewed and committed like
any other diff — regenerating a baseline is a deliberate, reviewed action,
not something that happens implicitly during a normal test run.
