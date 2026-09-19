# Pinned checkpoint staging

This harness does not load a GPU, rent hardware, or start Atlas. Run its metadata
step before the rental, then stage bytes onto the rental's local storage. Use an
isolated Python environment with `huggingface_hub` installed and record that
environment's package versions. Authentication uses the normal HF login or
environment mechanism; never put a token in commands, manifests, or logs.

Choose and record the official model's immutable 40-character revision. Mutable
branches and tags are refused. The following placeholders must be replaced:

```bash
python3 scripts/k3/checkpoint.py manifest \
  --repo moonshotai/Kimi-K3 --revision <40-character-commit> > /path/k3-manifest.json
```

This contacts the model-info API for file metadata, not weight bytes. The
manifest includes all repository files, exact sizes, LFS SHA256 identities, and
Git blob SHA1 identities for small non-LFS files. It includes tokenizer/code
files rather than assuming that JSON plus safetensors is a complete snapshot.
Inspect the manifest and sum `files[].size` before booking storage.

On the Linux rental, choose a dedicated snapshot directory and explicit budgets:

```bash
python3 scripts/k3/checkpoint.py download \
  --manifest /path/k3-manifest.json --root /models/k3-<revision> \
  --reserve-bytes 500000000000 --attempts 2 \
  --file-timeout 1800 --total-timeout 14400 \
  --progress-log /path/evidence/k3-download-attempt-1.jsonl
python3 scripts/k3/checkpoint.py verify \
  --manifest /path/k3-manifest.json --root /models/k3-<revision>
```

Those budgets are example operating choices, not K3 requirements. Reserve space
for builds, logs, and staging. Admission conservatively requires all unverified
file sizes plus the reserve; it does not credit resumable partial bytes. This
can refuse a nearly complete download on a full disk. Free additional space
instead of reducing the model manifest. No model files are deleted by the tool.

HF manages resumable partial downloads within the snapshot. Rerunning the same
command hashes existing files, skips verified files, and forces replacement of
present but corrupt files. A pin marker refuses a different model/revision in
the same directory. An advisory process lock refuses concurrent harness runs in
that directory. Do not modify the snapshot with another downloader during use.

Each file transfer runs in its own process group. Attempts and transfer time
are bounded; timeout/interruption terminates that owned group. The total deadline
is checked between files/attempts, and limits each child; disk hashing is not
interruptible by that deadline. Allow additional time for full verification of
terabytes. A failed transfer, insufficient space, absent file, wrong size, or
wrong hash returns nonzero. Keep that exit status when piping output to logs.

Optional `--progress-log` creates a new JSONL receipt **outside** the snapshot.
Existing log paths are refused; use a new path when resuming. Each event is
flushed with a UTC timestamp, elapsed time, checkpoint pin, total bytes,
previously verified bytes, newly verified bytes, and remaining unverified bytes.
Events distinguish existing-file verification, transfer attempts, successful
file verification, failed attempts, completion, and terminal failure. Failure
receipts include the exception type; inspect ordinary stderr for the reason.

`verified_bytes_per_second` measures newly verified whole-file bytes divided by
time since the first transfer attempt, including retries and verification.
It is **completion-based goodput**, not network bandwidth: no partial transfer
bytes are observed, and reused files are excluded. `estimated_remaining_seconds`
is an estimate from that rate; it is null until a file completes and zero when
all bytes verify. Events occur at file/attempt boundaries, so a large shard can
leave the log unchanged during transfer or hashing. HF's own progress display
can provide additional transfer information. A killed process or unavailable
filesystem may leave the final event incomplete; absence of `complete` is not
success. Preserve the command's exit status as well as the log.

`verify` hashes every listed file and emits a JSON result with model pin, total
bytes, and missing/corrupt paths. It does not infer rank residency, loader
support, successful inference, or numerical equivalence. Preserve the manifest
and verifier output with the rental evidence. The manifest is the trusted
identity input: obtain it directly from the reviewed HF repository, not from an
untrusted checkpoint directory.

Offline regression checks (no HF dependency or downloads):

```bash
python3 -m unittest discover -s scripts/k3 -p test_checkpoint.py
```
