# Sharp chat template (lane L39)

Atlas ships one third-party chat template inside the binary: the "Sharp" template for Qwen3.5, 3.6
and 3.8 models. It is a verbatim copy of an Apache-2.0 file plus one leading Jinja comment. It is
off by default: it is used only when the operator passes `--chat-template sharp` or
`--chat-template sharp-no-terse` (or sets `AVAROK_CHAT_TEMPLATE`). Without the flag every prompt is
byte-identical to what the model's own template produces.

## Upstream chain

| Link | Source | License | Holder |
|---|---|---|---|
| Sharp | https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates, file `chat_template.jinja`, `template_version = "qwen3.8-froggeric-v22.5.0"` (repository last modified 2026-10-10) | Apache-2.0 (`license: apache-2.0` in its README front matter) | Saga Ishtardottir (peculiar-ragdoll), per the repository's citation block |
| Fixed templates | https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates, v22.5 (the Sharp repository records it as its base) | Apache-2.0 | froggeric |
| Original | The Qwen3.5 chat template shipped with the Qwen3.5 checkpoints | Apache-2.0 (the checkpoints' `LICENSE`) | Alibaba Cloud / Qwen team |

Obtained 2026-11-12 as text over HTTPS. SHA-256 of the downloaded `chat_template.jinja`:
`cdff39fb26b60dc90faa292e726655c6b21f62db497846e02e4c4bbab942a84a` (30,411 bytes, 487 lines, LF line
endings, no trailing newline). The upstream git revision was not recorded: only file contents were
read, not commit metadata. None of the upstream repository's scripts or images were copied or run.
The upstream repository carries no NOTICE file, so there is no NOTICE text to reproduce beyond the
notices in this file.

## What Atlas changed

- `crates/spark-server/assets/chat_templates/qwen-sharp-v22.5.0.jinja` is the upstream file with
  exactly one change: a Jinja comment as its new first line, naming the chain above. The remaining
  487 lines are byte-identical (checked with `diff` against the download).
- `sharp-no-terse` is not a second file. At load Atlas prepends the single line
  `{%- set terse = false %}` to the same text, which the template honors through its own `terse`
  switch (upstream line 181). The variant keeps the retention of history reasoning and the
  tool-format fixes and drops the terseness system prompt.
- Atlas adds nothing to the template's text at render time. It supplies the template's inputs the
  way it does for every Qwen template (`tokenizer/chat_render.rs`), plus the optional `terse`
  variable from `chat_template_kwargs.terse`.

## Where it is used

| Atlas file | What |
|---|---|
| `crates/spark-server/assets/chat_templates/qwen-sharp-v22.5.0.jinja` | the embedded template (derived file) |
| `crates/spark-server/src/tokenizer/template_select.rs` | embeds it with `include_str!`, selects it, builds `sharp-no-terse` |
| `crates/spark-server/test_data/chat_templates/sharp/{cases.json,goldens.json}` | conversations rendered by the template, and what Python Jinja2 produces for them (test fixtures) |
| `tools/chat_template_goldens/gen_sharp_goldens.py` | dev-only generator of those fixtures; Atlas-written, reads the embedded template |
| `crates/spark-server/test_data/chat_templates/sharp/qwen3.5-4b-stock.jinja`, `qwen3.5-0.8b-stock.jinja` | byte copies of the `chat_template.jinja` shipped with Qwen3.5-4B and Qwen3.5-0.8B (Apache-2.0, Alibaba Cloud / Qwen team), used only as the control for "no flag means unchanged prompts" |

## License text

Apache License, Version 2.0: `LICENSES/Apache-2.0.txt`. The AGPL-3.0-only license of the rest of
Atlas does not change the license of the copied template: it stays Apache-2.0, as section 4 of the
Apache license requires for the retained notices. The NOTICE entry is in `NOTICE`; the citation is in
`CITATIONS.md`.

## What the template does and does not claim

The upstream project reports fewer thinking tokens per correct answer on 27B-class models and no
measurement on 4B or 0.8B models. Atlas treats it as an opt-in answer-style feature and measures it
separately (`bench/sharp_eval.py`); it is never used in kernel, KL, prefill or decode benchmarks, because it
adds roughly 150 prompt tokens to every request.
