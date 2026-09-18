# SPDX-License-Identifier: AGPL-3.0-only

# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///

"""Drive a two-pass control-vector derivation against a running serve.

    uv run scripts/run_control_vector_capture.py \
        --positive examples/control-vectors/verbosity/positive.txt \
        --negative examples/control-vectors/verbosity/negative.txt \
        --out-dir /tmp/verbosity

The serve must have been started with `--control-vector-capture`.

Each prompt is sent with `max_tokens: 1` — the capture hook accumulates during
PREFILL, so generating tokens would add decode activations to the mean and cost
minutes for nothing. Prompts are sent SEQUENTIALLY: the accumulator is global,
so concurrency would interleave two corpora into one sum with no way to tell
afterwards.

Order is reset -> positive -> dump -> reset -> negative -> dump. The leading
reset matters: an accumulator left warm from an earlier run silently averages
into this one and looks like a weak direction rather than a mistake.
"""

import argparse
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request


def post(base, path, payload, timeout=300):
    req = urllib.request.Request(
        f'{base}{path}', data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf8', 'replace')[:300]
        raise SystemExit(f'{path} failed: HTTP {e.code} {body}')


def capture(base, action, path=None):
    payload = {'action': action}
    if path:
        payload['path'] = str(path)
    return post(base, '/v1/control_vector/capture', payload)


def run_corpus(base, model, prompts, label):
    t0 = time.time()
    for i, p in enumerate(prompts, 1):
        post(base, '/v1/chat/completions', {
            'model': model,
            'messages': [{'role': 'user', 'content': p}],
            # Prefill is where capture happens; one token is enough to make the
            # request legal.
            'max_tokens': 1,
            'temperature': 0,
            'chat_template_kwargs': {'reasoning_effort': 'none'},
        }, timeout=900)
        if i % 25 == 0 or i == len(prompts):
            print(f'  {label}: {i}/{len(prompts)} ({time.time() - t0:.0f}s)',
                  flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='http://127.0.0.1:8899')
    ap.add_argument('--model', default='qwen3.8-flash-next-nvfp4-tp2-ep2')
    ap.add_argument('--positive', required=True)
    ap.add_argument('--negative', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    pos = [l.strip() for l in open(args.positive) if l.strip()]
    neg = [l.strip() for l in open(args.negative) if l.strip()]
    if len(pos) != len(neg):
        raise SystemExit(
            f'corpora must be paired line-by-line: {len(pos)} positive vs '
            f'{len(neg)} negative. Anything that differs between a pair other '
            f'than the property under test contaminates the direction.')
    print(f'{len(pos)} paired prompts')

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pos_dump, neg_dump = out / 'positive.bin', out / 'negative.bin'

    capture(args.base, 'reset')
    run_corpus(args.base, args.model, pos, 'positive')
    r = capture(args.base, 'dump', pos_dump)
    print(f'positive dump: {pos_dump} ({r.get("tokens")} tokens)')

    capture(args.base, 'reset')
    run_corpus(args.base, args.model, neg, 'negative')
    r = capture(args.base, 'dump', neg_dump)
    print(f'negative dump: {neg_dump} ({r.get("tokens")} tokens)')

    print('\nnext:')
    print(f'  uv run scripts/derive_control_vector.py \\')
    print(f'      {pos_dump} {neg_dump} {out / "vector.gguf"} --layers 4-44')
    return 0


if __name__ == '__main__':
    sys.exit(main())
