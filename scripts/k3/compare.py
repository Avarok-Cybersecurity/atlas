# SPDX-License-Identifier: AGPL-3.0-only
"""Record or compare a sequential completion suite (TP1 versus TP2, or cold versus warm)."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

from probe import validate as validate_response


def validate_cases(cases):
    if not isinstance(cases, list) or not cases:
        raise ValueError('suite must be a nonempty list')
    seen = set()
    for case in cases:
        if set(case) != {'id', 'prompt', 'max_tokens'}:
            raise ValueError('case requires id, prompt, max_tokens')
        if not isinstance(case['id'], str) or not case['id'].isascii() or not case['id'].replace('-', '').replace('_', '').isalnum() or case['id'] in seen:
            raise ValueError('case IDs must be unique safe filenames')
        seen.add(case['id'])
        if not isinstance(case['prompt'], str) or type(case['max_tokens']) is not int or case['max_tokens'] <= 0:
            raise ValueError('invalid prompt/token cap')
    return cases


def compare(reference, current):
    for record in (reference, current):
        if not isinstance(record, dict) or not {'request', 'status', 'response'} <= record.keys():
            raise ValueError('malformed generation receipt')
        request = record['request']
        if (not isinstance(request, dict) or not isinstance(request.get('model'), str)
                or type(request.get('max_tokens')) is not int or request['max_tokens'] <= 0):
            raise ValueError('malformed request in generation receipt')
        validate_response(record['response'], request=request)
    if reference['request'] != current['request']:
        raise ValueError('request differs from reference')
    if reference['status'] != 'passed' or current['status'] != 'passed':
        raise ValueError('both generations must pass before comparison')
    a, b = reference['response'], current['response']
    if a['choices'][0]['text'] != b['choices'][0]['text']:
        raise ValueError('completion text mismatch')
    if a['choices'][0]['finish_reason'] != b['choices'][0]['finish_reason']:
        raise ValueError('finish reason mismatch')
    for key in ('prompt_tokens', 'completion_tokens'):
        if a['usage'][key] != b['usage'][key]:
            raise ValueError(f'{key} mismatch')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--endpoint', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--deadline', type=float, required=True, help='per-request deadline in seconds')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference', type=Path)
    args = p.parse_args()
    if not math.isfinite(args.deadline) or args.deadline <= 0:
        p.error("deadline must be positive and finite")
    cases = validate_cases(json.loads(args.cases.read_text()))
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for case in cases:
        path = args.output / (case['id'] + '.json')
        command = [sys.executable, str(Path(__file__).with_name('probe.py')),
                   '--endpoint', args.endpoint, '--model', args.model,
                   '--prompt', case['prompt'], '--max-tokens', str(case['max_tokens']),
                   '--deadline', str(args.deadline), '--output', str(path)]
        row = {'id': case['id'], 'passed': False}
        try:
            subprocess.run(command, check=True, timeout=args.deadline + 5, capture_output=True)
            current = json.loads(path.read_text())
            if args.reference:
                compare(json.loads((args.reference/path.name).read_text()), current)
            row['passed'] = True
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            row['error'] = str(exc)
        results.append(row)
        # Preserve partial progress even if a later case fails or the runner is stopped.
        (args.output/'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results))
    return 0 if all(r['passed'] for r in results) else 1


if __name__ == '__main__':
    sys.exit(main())
