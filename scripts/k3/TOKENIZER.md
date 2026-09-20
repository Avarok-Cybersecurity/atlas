# Official K3 tokenizer preparation (CPU only)

The pinned `moonshotai/Kimi-K3` checkpoint supplies `tiktoken.model`, not
`tokenizer.json`. Atlas requires the latter. Prepare this **before renting**;
no model weights or GPU are needed. The source assets remain immutable.

Install `tokenizers==0.23.2` and `tiktoken==0.9.0` in a preparation virtualenv.
Download these three assets at revision
`f831ab66814297da540d832a5235f8e904f29d06` using the pinned checkpoint tooling:
`tiktoken.model`, `tokenization_kimi.py`, and `tokenizer_config.json`.
Also retain the pinned model `config.json` for the serving-loader check.
Then run:

```bash
python scripts/k3/tokenizer.py --source /data/k3-official --output /data/k3-derived
mkdir /data/k3-tokenizer-check
cp /data/k3-derived/tokenizer.json /data/k3-tokenizer-check/tokenizer.json
cp /data/k3-official/tokenizer_config.json /data/k3-tokenizer-check/tokenizer_config.json
cp /data/k3-official/config.json /data/k3-tokenizer-check/config.json
AVAROK_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo run -p spark-server \
  --example k3_tokenizer_check -- /data/k3-derived /data/k3-tokenizer-check
```

The output directory must not exist. Conversion reads the regex as Python AST
literal data; it does not import or execute checkpoint code. It preserves byte
ranks, all 256 reserved token IDs, explicit special-token flags, and the original
Unicode regex (including Han exclusions). The generated tokenizer is tested
against tiktoken, then reloaded and tested again. The Rust example checks the
same oracle with Atlas's locked tokenizer dependency and Oniguruma feature,
and with the actual `ChatTokenizer::from_model_dir` encode/decode path. It also
asserts that parsing the real model config and loading the tokenizer preserves
the model's EOS rather than substituting the tokenizer's named EOS.

Keep `derived-tokenizer.json` (input hashes, output hash, versions, and coverage)
as a **separate derived-asset receipt**. Do not add its hash to, or rewrite, the
official download manifest. Keep `tokenizer-oracle.json` outside Git as a test
artifact. The small directory above is a CPU tokenizer check only, without weights.

After the complete official download, create the actual serving directory:

```bash
python scripts/k3/stage_serving.py --source /data/k3-official \
  --manifest /data/k3-manifest.json --derived /data/k3-derived \
  --output /data/k3-serving
```

The tool independently verifies every original manifest hash before staging,
checks every indexed shard is in that manifest, and checks the converter's
source/output hashes. It hardlinks safetensors weights and copies metadata plus
the derived tokenizer. The output must be new, outside the source/derived roots,
and on the **same filesystem as all weights**. It refuses symlinks (including
symlink ancestry: use canonical absolute paths), cross-filesystem links, and
existing outputs; it never silently copies terabytes of weights. Failure removes
only the new staging directory. `serving-stage.json` records official manifest
identity and derived provenance separately. No original files are modified.

**Do not use symlinked weights or index files:** Atlas canonicalizes these paths
and correctly rejects references escaping the serving root. Hardlinks remain
inside that root but share source inodes, so treat both weight copies as read-only;
never modify them in place. The tool does not change permissions on shared
inodes. Keep the source snapshot quiescent during verification/staging.

The original `tokenizer_config.json` and model config are copied unchanged;
their identity selects Atlas's explicit unsupported-chat guard.

The model config's `eos_token_id` is **163586 (`<|end_of_msg|>`)**, while the
tokenizer config names `[EOS]` at 163585. These are distinct; do not replace the
model's stop ID with the tokenizer's named EOS. `<|open|>`, `<|close|>`, and
`<|sep|>` are IDs 163587–163589 and **are not skipped by special-token decoding**.
Reserved tokens lacking an explicit special flag likewise remain visible.

This asset enables raw text/token-ID **completions**, not full chat support.
Official XTML encodes structural markers specially but user/tool strings as
ordinary text in separate segments; flattening the rendered prompt before BPE
can change token IDs and enable marker injection. Use the offline segmented
prompt preparation tool for bring-up requests. Do not describe this conversion
as validating chat, tools, reasoning channels, or full-model output quality.

Spark CPU validation: 2,265 encode/roundtrip cases (multilingual text, combining
marks, emoji, code, whitespace, control bytes, every reserved marker, and 2,000
seeded fuzz cases), all 163,584 base-token individual decodes, plus exact
skip-special decoding against the source flags. This is differential coverage,
not a mathematical proof for every possible input or arbitrarily long input.
The official Transformers wrapper separately chunks pathological long strings;
this converter checks the underlying tiktoken encoder used by inference engines.
