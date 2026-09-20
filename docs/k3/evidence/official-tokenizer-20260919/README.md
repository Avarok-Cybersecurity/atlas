# Official tokenizer CPU validation

Source: `moonshotai/Kimi-K3` revision
`f831ab66814297da540d832a5235f8e904f29d06`. Only tokenizer assets downloaded;
no full weights. Runbook: [`TOKENIZER.md`](../../../../scripts/k3/TOKENIZER.md).

Python conversion and save/reload: 2,265 exact encode/decode/skip-special cases;
163,584 individual base-token decodes. Atlas's actual Rust dependency
`tokenizers 0.23.2` with `onig` also passed all 2,265 oracle cases on Spark2 CPU through both the raw tokenizer and actual
`ChatTokenizer::from_model_dir` encode/decode path. Model config parsing
retained EOS 163586, and named EOS 163585 remained distinct:

```text
PASS: 2265 official K3 tiktoken cases match Atlas Rust and ChatTokenizer; model EOS preserved
```

`receipt.json` hashes immutable inputs and the derived output separately.
Generated tokenizer (~12 MB) and full oracle are external artifacts, not tracked
source or benchmark certification evidence. Output is suitable for completion
bring-up; native XTML chat/tool/streaming semantics are a separate integration.
