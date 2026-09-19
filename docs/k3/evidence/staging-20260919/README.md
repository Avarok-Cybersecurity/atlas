# Serving staging smoke test

Spark2 CPU-only development fixture. One actual packed twin shard was
hardlinked into a new source directory, with the twin model config and a
566-tensor index generated from its actual safetensors header. Official tokenizer
assets were copied into that isolated development source; the original model
and tokenizer assets were not changed. This mixed fixture is **staging-only**,
not an official full-model compatibility or inference result.

The staging tool verified every development-manifest hash and derived tokenizer
hash, then produced a separate serving directory. All weight inodes matched the
source; index/shard canonical paths stayed inside the serving root; before/after
hashes of all original inputs matched. Eight regression tests additionally cover
missing/corrupt inputs, escaped symlinks/index paths, already existing outputs,
cross-filesystem link refusal without copying weights, changed tokenizer data,
duplicate index keys, and verified nested shard layouts refused by Atlas.

The complete-model staging command reads every source byte for verification.
Budget this disk I/O once before billed GPU work; no separate redundant full
hash pass is required. Hardlinked weights consume no second full weight copy.
The serving directory and original snapshot must both remain read-only to avoid
changing shared weight inodes.
