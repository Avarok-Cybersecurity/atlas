# Offline K3 checkpoint and tokenizer preparation

These tools prepare pinned assets without GPUs. They do not add a model loader,
chat API, or inference support. See CHECKPOINT.md for bounded downloads, disk
admission, hash verification, and resume behavior.

## Tokenizer conversion

Use an isolated environment with `tokenizers==0.23.2` and `tiktoken==0.9.0`.
At official revision `f831ab66814297da540d832a5235f8e904f29d06`, retain
`tiktoken.model`, `tokenization_kimi.py`, and `tokenizer_config.json`.

```sh
python scripts/k3/tokenizer.py --source /models/k3-source --output /models/k3-derived
```

The output must be new. Conversion parses the tokenizer pattern as AST data;
it does not execute the tokenizer source. It verifies token ranks, reserved
IDs and special-token flags against tiktoken, including a reload of its output.
Keep the derived receipt separate from the official checkpoint manifest.
The model EOS is 163586; the tokenizer's named EOS is 163585. They differ.

## Serving asset directory

After downloading and verifying the complete checkpoint:

```sh
python scripts/k3/stage_serving.py --source /models/k3-source \
  --manifest /models/k3-manifest.json --derived /models/k3-derived \
  --output /models/k3-serving
```

Staging verifies source hashes and the index, copies metadata and hardlinks
weights into a new directory on the same filesystem. It refuses symlinks,
escaping paths, changed assets, existing outputs and cross-filesystem links.
It never silently copies the checkpoint. Hardlinked weights share inodes;
keep the source and staged weights immutable. This directory alone does not
establish that the current Atlas loader supports the checkpoint.

## Segmented XTML prompt preparation

Also retain `encoding_k3.py` from the pinned revision. Review that source before
use: the offline reference tool executes only its exact allowlisted SHA256,
and rejects other versions. This differs from the AST-only tokenizer converter.

```sh
python scripts/k3/prepare_prompt.py --source /models/k3-source \
  --messages /data/messages.json --model kimi-k3 --max-tokens 32 \
  --thinking off --output /data/request.json
```

Messages use the official encoder schema. Structural markers receive special
encoding; user and tool content receive ordinary encoding. Flattening segments
before encoding changes this contract. Outputs are a token-array completion
request and a provenance receipt. A compatible K3 server is a separate
prerequisite; these tools do not validate chat, tool execution, full-model
quality, or throughput.

## Local checks and evidence boundary

```sh
python3 -m unittest discover -s scripts/k3 -p 'test_*.py'
```

The extraction's 27 offline checks cover corruption, unsafe paths, retries,
resume accounting, untrusted encoder source, token-rank errors and segmented
marker handling. Original production-asset differential results remain in
integration PR #1150. They are historical evidence, not a rerun on this branch.
No network download, rental, or full checkpoint test is required by this suite.
