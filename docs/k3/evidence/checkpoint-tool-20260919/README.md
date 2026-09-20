# Live checkpoint tool smoke test — 2026-09-19

Passed on Spark2 in an isolated `checkpoint-tool-test` directory, using the
existing rental worktree Python environment and `huggingface_hub` 1.32.0.
No GPU work or existing model directories were involved.

The public repository was `hf-internal-testing/tiny-random-gpt2`, pinned by
the Hugging Face API to revision `71034c5d8bde858ff824298bdedc65515b97d2b9`.
The actual metadata manifest admitted 10 files totaling 12,512,611 bytes
against a hard 100,000,000-byte ceiling before downloading any weights.

The production `checkpoint.py` download and verify commands completed these
checks within a cumulative five-minute transfer budget (8.71 seconds used):

1. Initial download and full hash verification succeeded.
2. A second download with `HF_HUB_OFFLINE=1` and an unreachable endpoint
   succeeded with no output and unchanged file modification times.
3. Flipping one byte of `config.json` made verification exit 1 and identify
   exactly that file as corrupt.
4. Download repaired the file; verification succeeded and its bytes matched
   the original file exactly.

`receipt.json` records the tested script SHA256, pinned identity, sizes and
phase results. Phase elapsed times are cumulative from the start of the test.
`manifest.json` and command standard output provide the small supporting
evidence. This tests real Hugging Face metadata, download, skip and repair
paths; it does not establish throughput or memory behavior for the full K3
checkpoint, interrupted partial-file resumption, or GPU compatibility.
