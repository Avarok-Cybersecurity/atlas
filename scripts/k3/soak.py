# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded completion lifecycle soak; development evidence, not certification."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request

from probe import validate

LIMIT = 4 * 1024 * 1024


def read_sse(response, model, cancel=False):
    text, finish, size, fields = '', None, 0, []
    while True:
        line = response.readline(LIMIT + 1)
        size += len(line)
        if size > LIMIT:
            raise ValueError('SSE response exceeded 4 MiB')
        if not line:
            raise ValueError('SSE ended without [DONE]')
        line = line.decode('utf-8').rstrip('\r\n')
        if line.startswith('data:'):
            fields.append(line[5:].lstrip(' '))
        elif line == '' and fields:
            data, fields = '\n'.join(fields), []
            if data == '[DONE]':
                if finish not in ('stop', 'length') or not text.strip():
                    raise ValueError('SSE missing nonempty text or finish reason')
                return dict(status='completed', text=text, finish_reason=finish)
            chunk = json.loads(data)
            if not isinstance(chunk, dict) or 'error' in chunk:
                raise ValueError('SSE error or invalid event')
            if chunk.get('model') != model:
                raise ValueError('SSE model mismatch')
            choices = chunk.get('choices')
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError('SSE expected exactly one choice')
            choice = choices[0]
            if not isinstance(choice, dict) or choice.get('index') != 0:
                raise ValueError('SSE invalid choice index')
            delta = choice.get('text', '')
            if not isinstance(delta, str) or finish is not None:
                raise ValueError('SSE invalid text or event after terminal choice')
            text += delta
            if cancel and delta:
                if choice.get('finish_reason') is not None:
                    raise ValueError('generation already terminal before cancellation')
                # Returning closes the HTTP response immediately in request().
                return dict(status='client_cancelled', text=text)
            finish = choice.get('finish_reason')
            if finish is not None and finish not in ('stop', 'length'):
                raise ValueError('SSE invalid finish reason')


def request(endpoint, payload, mode, timeout):
    payload = dict(payload, stream=mode != 'plain')
    req = urllib.request.Request(endpoint.rstrip('/') + '/v1/completions',
                                 data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        if mode != 'plain':
            if response.headers.get_content_type() != 'text/event-stream':
                raise ValueError('stream response is not text/event-stream')
            return read_sse(response, payload['model'], mode == 'cancel')
        raw = response.read(LIMIT + 1)
        if len(raw) > LIMIT:
            raise ValueError('JSON response exceeded 4 MiB')
        body = json.loads(raw)
        text = validate(body, request=payload)
        return dict(status='completed', text=text,
                    finish_reason=body['choices'][0]['finish_reason'], usage=body['usage'])


def bounded_request(endpoint, payload, mode, deadline):
    if deadline <= 0:
        raise TimeoutError('total soak deadline exceeded')
    try:
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '_request'],
                               input=json.dumps([endpoint, payload, mode, deadline]),
                               text=True, capture_output=True, timeout=deadline)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError('request wall-clock deadline exceeded') from exc
    if child.returncode:
        raise ValueError(child.stderr.strip() or 'request process failed')
    return json.loads(child.stdout)


def match(actual, expected):
    for key in ('text', 'finish_reason'):
        if actual.get(key) != expected.get(key):
            raise ValueError('completion differs from isolated baseline: ' + key)
    if 'usage' in actual:
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            expected_count = expected.get('usage', expected).get(key)
            if expected_count is not None and actual['usage'].get(key) != expected_count:
                raise ValueError('token count differs from baseline: ' + key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--cases', type=Path, default=Path(__file__).with_name('smoke-cases.json'))
    parser.add_argument('--baseline-receipt', type=Path,
                        help='generation.json with baseline_cases text/finish expectations')
    parser.add_argument('--cancel-max-tokens', type=int, default=128)
    parser.add_argument('--cycles', type=int, default=10)
    parser.add_argument('--concurrency', type=int, default=2)
    parser.add_argument('--request-deadline', type=float, default=60)
    parser.add_argument('--deadline', type=float, default=1800)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.cancel_max_tokens <= 4096:
        parser.error('cancel-max-tokens must be 1..4096')
    if args.cycles < 1 or not 2 <= args.concurrency <= 16:
        parser.error('positive cycles and concurrency 2..16 required')
    if any(not math.isfinite(x) or x <= 0 for x in (args.deadline, args.request_deadline)):
        parser.error('deadlines must be positive and finite')
    cases = json.loads(args.cases.read_text())
    if not isinstance(cases, list) or not cases:
        parser.error('cases must be a nonempty list')
    for case in cases:
        if (not isinstance(case, dict) or not isinstance(case.get('id'), str)
                or not isinstance(case.get('prompt'), str) or not case['prompt']
                or type(case.get('max_tokens')) is not int or not 1 <= case['max_tokens'] <= 4096):
            parser.error('each case needs id, nonempty prompt, max_tokens 1..4096')
    if len({case['id'] for case in cases}) != len(cases):
        parser.error('case ids must be unique')
    trusted = None
    if args.baseline_receipt:
        trusted = {row['id']: row for row in json.loads(args.baseline_receipt.read_text())['baseline_cases']}
        if any(case['id'] not in trusted for case in cases):
            parser.error('baseline receipt must cover every case id')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    end = started + args.deadline
    summary = dict(schema=1, status='failed', model=args.model, cases=cases,
                   cycles_requested=args.cycles, cycles_completed=0, requests=0,
                   concurrency=args.concurrency, cancel_max_tokens=args.cancel_max_tokens,
                   trusted_baseline_checked=trusted is not None, deadline_seconds=args.deadline,
                   request_deadline_seconds=args.request_deadline,
                   limitations=['client cancellation does not prove immediate GPU cancellation',
                                'concurrent client launches do not prove server batching',
                                'no memory measurements or leak claims'])
    lock = threading.Lock()
    with (args.output / 'receipts.jsonl').open('x') as receipts:
        def run(case, phase, mode='plain', expected=None, barrier=None):
            payload = dict(model=args.model, prompt=case['prompt'],
                           max_tokens=args.cancel_max_tokens if mode == 'cancel' else case['max_tokens'],
                           temperature=0)
            before = time.monotonic()
            record = dict(case=case['id'], phase=phase, mode=mode, status='failed',
                          started_seconds=before-started)
            try:
                if barrier is not None:
                    barrier.wait(timeout=max(.001, end-time.monotonic()))
                result = bounded_request(args.endpoint, payload, mode,
                                         min(args.request_deadline, end-time.monotonic()))
                record['result'] = result
                if expected is not None:
                    match(result, expected)
                if mode == 'cancel' and result['status'] != 'client_cancelled':
                    raise ValueError('request did not cancel after content')
                record['status'] = 'passed'
                return result
            except Exception as exc:
                record['error'] = str(exc)
                raise
            finally:
                record['elapsed_seconds'] = time.monotonic()-before
                with lock:
                    summary['requests'] += 1
                    receipts.write(json.dumps(record, allow_nan=False) + '\n')
                    receipts.flush()
        try:
            baselines = [run(case, 'baseline', expected=trusted[case['id']] if trusted else None)
                         for case in cases]
            for cycle in range(args.cycles):
                for case, baseline in zip(cases, baselines):
                    run(case, f'cycle-{cycle}:repeat', expected=baseline)
                index = cycle % len(cases)
                run(cases[index], f'cycle-{cycle}:stream', 'stream', baselines[index])
                run(cases[index], f'cycle-{cycle}:cancel', 'cancel')
                run(cases[0], f'cycle-{cycle}:after-cancel', expected=baselines[0])
                barrier = threading.Barrier(args.concurrency)
                with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                    jobs = []
                    for offset in range(args.concurrency):
                        idx = (cycle+offset) % len(cases)
                        jobs.append(pool.submit(run, cases[idx], f'cycle-{cycle}:concurrent',
                                                'plain', baselines[idx], barrier))
                    for job in jobs:
                        job.result()
                summary['cycles_completed'] += 1
            summary['status'] = 'passed'
        except Exception as exc:
            summary['error'] = str(exc)
        finally:
            summary['elapsed_seconds'] = time.monotonic()-started
            (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, allow_nan=False))
    return 0 if summary['status'] == 'passed' else 1


if __name__ == '__main__':
    if sys.argv[1:] == ['_request']:
        try:
            print(json.dumps(request(*json.load(sys.stdin)), allow_nan=False))
        except Exception as error:
            print(str(error), file=sys.stderr)
            sys.exit(1)
    else:
        sys.exit(main())
