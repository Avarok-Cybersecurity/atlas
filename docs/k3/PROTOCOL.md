# K3 protocol readiness and the initial rental boundary

The official model and the 0.40B twin have different tokenizer/chat contracts.
Passing the twin's chat endpoint is not evidence that official K3 chat works.
The current rental path supports **raw completions**, including independently
prepared integer token arrays. Official K3 chat requests deliberately fail
before inference; enabling streaming or tools does not remove that guard.

## What the pinned checkpoint actually ships

Audited revision: `f831ab66814297da540d832a5235f8e904f29d06` of
[moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06).

- No `tokenizer.json` or Jinja chat template. The assets are `tiktoken.model`,
  `tokenization_kimi.py`, `tokenizer_config.json`, and `encoding_k3.py`.
- The Python encoder renders XTML structural tokens `<|open|>`, `<|close|>`,
  `<|sep|>` and message/channel names, rather than ChatML or K2 tool markers.
- Structural segments use reserved token IDs; literal markers in user text,
  tool arguments/results and attributes must remain ordinary BPE text.
  Concatenating the segments then using a normal tokenizer violates that rule.
- Generation starts inside the assistant `think` channel when thinking is on,
  or `response` when off. An assistant history entry contains response and
  optional thinking/tools channels. Tool calls contain typed argument blocks.
- Tool results are reordered by opaque call ID; matching call names override
  stale result names. JSON-string arguments preserve number/array literal
  bytes. Malformed argument JSON has a separate raw-JSON encoding.
- The model config's EOS is **163586 (`<|end_of_msg|>`)**. Tokenizer config's
  `[EOS]` is 163585; do not substitute it for the model's configured stop ID.

The ten cases in [official-xtml.json](fixtures/official-xtml.json) were generated
by the actual pinned encoder, recording every segment and its special-token
permission. They are independent reference fixtures for future implementation,
**not tests claiming Atlas can parse XTML output**. They cover plain/thinking
prompts, literal markers, attribute escaping, Unicode, typed calls, reversed
results, tool declarations, invalid argument JSON and required-tool prompting.

## Prepare a valid prompt before starting the rental

First stage and verify the small pinned tokenizer assets, and derive
`tokenizer.json` with `scripts/k3/tokenizer.py`. Keep the verified snapshot
immutable; place derived files in a separate model staging directory. Retain
`tokenizer_config.json` so the native contract remains identifiable.

For an official chat-shaped numerical canary, prepare its token IDs offline:

```bash
# source contains the four pinned assets listed above; the script verifies
# their exact hashes and only executes the reviewed encoder with that hash.
printf '%s\n' '[{"role":"user","content":"What is 2 + 2?"}]' > messages.json
python3 scripts/k3/prepare_prompt.py \
  --source /models/k3-tokenizer-assets --messages messages.json \
  --model kimi-k3 --max-tokens 64 --thinking off --output request.json
curl --fail-with-body -H 'Content-Type: application/json' \
  --data-binary @request.json http://127.0.0.1:8000/v1/completions
```

This uses Python `tiktoken` only for offline preparation, never in the shipped
engine. The payload is an integer array, so Atlas does not retokenize it. A
sidecar records source hashes, prompt identity, thinking state and EOS ID.
The output remains **raw XTML**, not an OpenAI chat response. Do not execute
model-emitted tools; inspect/save the raw output and compare token IDs to an
independent reference. Streaming raw completions likewise does not perform
XTML channel/tool demultiplexing. Max-token termination can leave incomplete
XTML: preserve that evidence instead of treating it as a valid tool call.

## Work required before enabling official chat

1. Add native segmented message encoding, using the independent fixtures above
   and token-ID comparisons against the official encoder. Preserve ordinary
   encoding for marker literals, tool-result ordering, typed argument values,
   requested thinking mode and actual EOS. Reject unsupported modalities.
2. Add one XTML state machine shared by blocking and streaming responses.
   Distinguish seeded thinking/response state, message termination, response
   text, tools and typed argument blocks. Refuse malformed/truncated tool calls.
3. Test every UTF-8 boundary around structural markers, partial headers,
   multi-call and empty-call cases, truncated arguments, and marker-like user
   content. Require streaming reassembly to agree with blocking parse results.
4. Wire explicit model behavior/parser selection for OpenAI and Anthropic
   surfaces; prevent generic tool instruction injection. Test tool choices,
   tool history and reasoning visibility at the HTTP boundary.
5. On the real model, verify useful answers and actual learned tool semantics.
   The tiny twin can test transport/lifecycle only; synthetic fixtures prove
   protocol implementation, not model competence.

The current guard and its CPU tests prevent false chat compatibility. This is
sufficient for a bounded raw-inference/kernel bring-up session, not a declaration
that an official K3 chat or agent deployment is ready.
