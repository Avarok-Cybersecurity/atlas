# SPDX-License-Identifier: AGPL-3.0-only

# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///

"""Measure a verbosity control vector by its effect on completion length.

    uv run scripts/measure_verbosity_vector.py --vector verbosity [--scales -0.15,0,0.15]

This is why verbosity is a good worked example: the effect is a NUMBER, so the
demo is falsifiable. A style vector you judge by reading prose can only be
argued about; a length vector either moves `completion_tokens` or it does not.

Held constant across arms: prompt set, temperature 0, max_tokens, thinking off,
and the serve itself. The ONLY thing that varies is the steering, which is what
makes this a comparison rather than two observations.

`add` mode with a signed scale is the interesting arm: projection REMOVES the
verbosity axis (-> neutral), addition DISPLACES along it, so a signed scale
gives a dial in both directions. Report the median, and report the spread —
a delta inside the spread is noise, not a result.
"""

import argparse
import json
import statistics
import sys
import urllib.error
import urllib.request

PROMPTS = [
    "How do I reverse a linked list?",
    "What does a hash map do when two keys collide?",
    "How do I find which commit introduced a bug?",
    "What is the difference between a process and a thread?",
    "How do I read a file line by line in Python?",
    "What does a load balancer do?",
    "How does binary search work?",
    "What is a race condition?",
]


def ask(base, model, prompt, max_tokens, vector, scale_note):
    body = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': max_tokens,
        'temperature': 0,
        'chat_template_kwargs': {'reasoning_effort': 'none'},
    }
    if vector:
        body['control_vector'] = vector
    req = urllib.request.Request(
        f'{base}/v1/chat/completions', data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        raise SystemExit(f'request failed ({scale_note}): HTTP {e.code} '
                         f'{e.read().decode("utf8", "replace")[:200]}')
    u = d['usage']
    det = u.get('completion_tokens_details', {}) or {}
    if det.get('reasoning_tokens'):
        raise SystemExit('thinking leaked into the completion — the token count '
                         'would measure reasoning, not verbosity')
    return u['completion_tokens'], d['choices'][0]['finish_reason']


def arm(base, model, vector, label, max_tokens):
    cts, truncated = [], 0
    for p in PROMPTS:
        ct, finish = ask(base, model, p, max_tokens, vector, label)
        cts.append(ct)
        if finish == 'length':
            truncated += 1
    med = statistics.median(cts)
    print(f'  {label:<24} median={med:7.1f}  mean={statistics.mean(cts):7.1f}  '
          f'min={min(cts)}  max={max(cts)}  truncated={truncated}/{len(cts)}')
    if truncated:
        print(f'     NOTE: {truncated} hit max_tokens — the ceiling is clipping '
              f'the effect, raise --max-tokens before quoting this')
    return med, cts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='http://127.0.0.1:8899')
    ap.add_argument('--model', default='qwen3.8-flash-next-nvfp4-tp2-ep2')
    ap.add_argument('--vector', default='verbosity',
                    help='registered control-vector NAME')
    ap.add_argument('--max-tokens', type=int, default=2000)
    args = ap.parse_args()

    print(f'{len(PROMPTS)} prompts, temperature 0, thinking off, '
          f'max_tokens={args.max_tokens}')
    print('completion_tokens per arm:')
    base_med, base_all = arm(args.base, args.model, None, 'no steering', args.max_tokens)
    steer_med, steer_all = arm(args.base, args.model, args.vector,
                               f'control_vector={args.vector}', args.max_tokens)

    delta = 100.0 * (steer_med - base_med) / base_med if base_med else 0.0
    spread = (max(base_all) - min(base_all)) / base_med * 100.0 if base_med else 0.0
    print()
    print(f'median completion_tokens: {base_med:.1f} -> {steer_med:.1f} '
          f'({delta:+.1f}%)')
    print(f'within-arm spread across prompts: {spread:.0f}% of the median')
    print()
    if abs(delta) <= 10:
        print('VERDICT: no clear effect. Prompts differ from each other far more '
              'than the arms differ, so this needs per-prompt pairing (same '
              'prompt, both arms, paired delta) before any claim.')
    else:
        print('VERDICT: the arms differ. Single-harness, one prompt set — quote '
              'as "observed", and repeat before treating the size as settled.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
