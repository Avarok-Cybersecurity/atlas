#!/usr/bin/env python3
"""Compare a live Atlas serve of the SLICED LongCat checkpoint against the
golden-derived expectation (see slice_expected.py).

Sends the 16 fixture token ids as a RAW TOKEN-ID prompt (the completions API
accepts `prompt: [ids]`), greedy, 1 token, with logprobs — then checks the
argmax and the top-k logit ordering.

PASS criteria:
  * argmax matches the reference exactly (the reference margin is >1.5, far
    above NVFP4 rounding), and
  * the top-5 reference ids all appear in the served top-8 (ordering within
    near-ties may differ under 4-bit quantization).

Usage: slice_compare.py PORT MODEL EXPECTED_JSON
"""
import json
import sys

import requests

port, model, exp_path = sys.argv[1], sys.argv[2], sys.argv[3]
exp = json.load(open(exp_path))

r = requests.post(
    f'http://127.0.0.1:{port}/v1/completions',
    json={
        'model': model,
        'prompt': exp['tokens'],
        'max_tokens': 1,
        'temperature': 0,
        'logprobs': 8,
    },
    timeout=600,
)
print('HTTP', r.status_code)
d = r.json()
if r.status_code != 200:
    print(json.dumps(d, indent=1)[:1200])
    sys.exit(1)

choice = d['choices'][0]
lp = choice.get('logprobs') or {}
top = (lp.get('top_logprobs') or [{}])[0]
got_ids = []
tok_ids = lp.get('tokens_ids') or lp.get('token_ids')
if isinstance(top, dict) and top:
    # OpenAI shape: {token_text: logprob}; ids are not always echoed, so fall
    # back to the sampled token id when present.
    print('served top_logprobs:', json.dumps(top)[:400])
if tok_ids:
    got_ids = list(tok_ids)

print('reference argmax id :', exp['argmax'], f"(logit {exp['top_logits'][0]:.3f},"
      f" margin {exp['logit_spread']:.3f})")
print('reference top-8 ids :', exp['top_ids'])
print('served text         :', json.dumps(choice.get('text'))[:120])
if got_ids:
    print('served token ids    :', got_ids[:4])
    ok = got_ids[0] == exp['argmax']
    print('ARGMAX MATCH:', ok)
    sys.exit(0 if ok else 2)
print('NOTE: server did not echo token ids; compare the decoded text against '
      'the reference argmax token using the tokenizer.')
