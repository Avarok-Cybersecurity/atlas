# Bounded rental rehearsal tools

These standard-library Python tools validate a supplied Atlas binary and model.
They do not add model support, GPU targets, scheduler features, or certification.
The binary must already support the selected device, model and launch options.
No checkpoint download or credentials are required by the tools.

- [`launch.py`](LAUNCH.md) admits a complete single-host rank map, checks binary
  SHA256, architecture and device UUIDs, refuses occupied GPUs/ports, runs an
  actual generation canary, and shuts down only its owned process groups.
- `probe.py` checks completion content, counts, stop reasons and finite values.
  Integer token-array requests and exact stop sequences are supported.
- `compare.py` runs explicit cases and compares full outputs and counts against
  an independently obtained receipt. `--help` describes its bounded CLI.
- [`soak.py`](SOAK.md) checks repeat/stream agreement, cancellation recovery and
  concurrent clients against serial responses, with per-request deadlines.
- `smoke-cases.json` supplies bounded development prompts; it is not a quality
  benchmark or proof of semantic model correctness.

Every run writes to a new output directory and retains failure evidence.
Record the supplied binary/model hashes and launch settings with results.
A healthy endpoint alone is insufficient, and matching concurrent responses
are not proof of GPU batching. Multi-host launch is deliberately unsupported.
Never send credentials in the manifest environment: configuration is recorded.

Run local protocol/process tests without a GPU:

```sh
python3 -m unittest discover -s scripts/k3 -p 'test_*.py'
```

The harness was extracted from development rehearsal work in #1150. Historical
Spark/B200 results refer to the integration binaries identified there, not to a
fresh build of this tools-only PR. The documentation is self-contained and does
not depend on those private machines or their local files.
