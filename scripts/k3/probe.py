# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded real-generation canary; HTTP health alone is not inference evidence."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import urllib.request


def validate(response, expected_prefix=None):
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('non-finite numeric response')
        if isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, list):
            for item in value:
                finite(item)
    if not isinstance(response, dict):
        raise ValueError('response must be an object')
    finite(response)
    choices = response.get('choices')
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError('expected exactly one completion')
    if not isinstance(choices[0], dict):
        raise ValueError('completion must be an object')
    text = choices[0].get('text')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('empty completion')
    if choices[0].get('finish_reason') not in ('stop', 'length'):
        raise ValueError('missing or unexpected finish reason')
    if expected_prefix is not None and not text.startswith(expected_prefix):
        raise ValueError('completion differs from expected prefix')
    usage = response.get('usage', {})
    if not isinstance(usage, dict):
        raise ValueError('usage must be an object')
    count = usage.get('completion_tokens')
    if type(count) is not int or count <= 0:
        raise ValueError('completion token count must be positive')
    return text


def request(endpoint, payload, timeout):
    req = urllib.request.Request(endpoint.rstrip('/') + '/v1/completions',
                                 data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        # Bound a broken endpoint's response; parent deadline bounds trickle reads.
        raw = response.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError('response exceeded 4 MiB')
    return raw.decode('utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--endpoint', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--prompt', required=True)
    p.add_argument('--expected-prefix')
    p.add_argument('--max-tokens', type=int, default=32)
    p.add_argument('--deadline', type=float, default=120)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.max_tokens <= 0 or not math.isfinite(args.deadline) or args.deadline <= 0:
        p.error('max-tokens and deadline must be positive and finite')
    payload = dict(model=args.model, prompt=args.prompt, max_tokens=args.max_tokens,
                   temperature=0, stream=False)
    # Exclusive creation prevents replacing an earlier receipt, including on failure.
    with args.output.open('x') as receipt:
        record = dict(schema=1, request=payload, status='failed',
                      check='expected-prefix' if args.expected_prefix is not None else 'nonempty-generation')
        started = time.monotonic()
        try:
            child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '_request'],
                                   input=json.dumps([args.endpoint, payload, args.deadline]),
                                   text=True, capture_output=True, timeout=args.deadline)
            record['raw_response'] = child.stdout
            if child.returncode:
                raise ValueError(child.stderr.strip() or 'request failed')
            response = json.loads(child.stdout)
            validate(response, args.expected_prefix)
            record['response'] = response
            record['status'] = 'passed'
        except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
            record['error'] = str(exc)
        record['elapsed_seconds'] = time.monotonic() - started
        json.dump(record, receipt, indent=2, allow_nan=False)
        receipt.write('\n')
    print(record['status'])
    return 0 if record['status'] == 'passed' else 1


if __name__ == '__main__':
    if sys.argv[1:] == ['_request']:
        try:
            print(request(*json.load(sys.stdin)))
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
    else:
        sys.exit(main())
