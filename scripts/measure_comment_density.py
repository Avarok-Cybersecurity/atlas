# SPDX-License-Identifier: AGPL-3.0-only

# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///

"""Measure a comment-density control vector by a LENGTH-INVARIANT ratio.

    uv run scripts/measure_comment_density.py --arms terse,base,verbose

The verbosity example is scored on `completion_tokens`, and that metric has a
known weakness: almost any sufficiently strong steering lengthens output. An
unrelated refusal vector moved length +8.6% on this model. So a vector that
merely perturbs the residual stream scores as a weak success there.

This scores:

    comment lines / total lines, inside fenced code blocks

which is a RATIO. Rambling adds comment lines and code lines together and moves
it very little, so the metric cannot be reached by accident. That is the whole
reason this example exists.

Reported alongside, because they are how the result gets falsified:

  - `lines`, so a ratio shift can be checked against whether the model simply
    stopped writing code.
  - `fences`, because a response with no code block has no ratio at all, and
    averaging an absent value as zero would manufacture an effect.
  - completion_tokens, so a "density" change that is really just length is
    visible as such.
"""

import argparse
import json
import re
import statistics
import sys
import urllib.error
import urllib.request

# Held-out tasks: none appears in the derivation corpus. A measured effect on
# a training line could be the vector recognising that line.
PROMPTS = [
    "Write a Python function that parses a duration string like '1h30m' into seconds.",
    "Write a Rust function that splits a string on commas and trims each field.",
    "Write a bash script that prints the git branch of every repo under a directory.",
    "Write a CUDA kernel that computes the elementwise maximum of two arrays.",
    "Write a JavaScript function that formats a number with thousands separators.",
    "Write a Go function that reads newline-delimited JSON from a reader.",
    "Write a SQL query that finds duplicate email addresses in a users table.",
    "Write a C function that reverses the bytes of a 32-bit integer.",
]

FENCE = re.compile(r'^\s*```')
# Line-comment openers per language. Block comments are deliberately not
# handled: they are rare in generated snippets, and a half-correct block-comment
# parser would silently miscount rather than obviously fail.
COMMENT_PREFIXES = ('#', '//', '--', ';', '%')


def code_lines(text):
    """Return (comment_lines, total_code_lines, fence_count) over fenced blocks."""
    inside, comments, total, fences = False, 0, 0, 0
    for line in text.splitlines():
        if FENCE.match(line):
            inside = not inside
            fences += 1
            continue
        if not inside:
            continue
        s = line.strip()
        if not s:
            continue
        total += 1
        if s.startswith(COMMENT_PREFIXES):
            comments += 1
        elif '#' in s or '//' in s:
            # Trailing comment on a code line counts too — "x = 1  # why".
            # Crude: a '#' or '//' inside a string literal is miscounted. It is
            # applied identically to every arm, so it biases the absolute ratio
            # and not the comparison between arms.
            comments += 1
    return comments, total, fences


def ask(base, model, prompt, max_tokens, vector):
    body = {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': max_tokens, 'temperature': 0,
            'chat_template_kwargs': {'reasoning_effort': 'none'}}
    if vector:
        body['control_vector'] = vector
    req = urllib.request.Request(f'{base}/v1/chat/completions',
                                 data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        raise SystemExit(f'HTTP {e.code}: {e.read().decode("utf8", "replace")[:300]}')
    return (d['usage']['completion_tokens'], d['choices'][0]['finish_reason'],
            d['choices'][0]['message']['content'])


def run_arm(base, model, arm, max_tokens, fh):
    vector = None if arm == 'base' else arm
    ratios, toks, nocode, trunc = [], [], 0, 0
    for p in PROMPTS:
        ct, fin, txt = ask(base, model, p, max_tokens, vector)
        c, t, f = code_lines(txt)
        toks.append(ct)
        trunc += fin == 'length'
        if t == 0 or f < 2:
            nocode += 1
        else:
            ratios.append(c / t)
        if fh:
            fh.write(f'===== {arm} | {p}\n[{ct} tok, {fin}, '
                     f'{c}/{t} comment lines]\n{txt}\n\n')
    if not ratios:
        print(f'  {arm:<12} NO CODE BLOCKS in any response — no ratio exists',
              flush=True)
        return None
    med = statistics.median(ratios)
    # An arm that stopped writing code in half its responses is not measuring
    # comment density any more, and its surviving ratio is drawn from whichever
    # few answers still had a code block. Report the number, but disqualify it
    # from the dose-response — otherwise one broken arm at the end of the ladder
    # reads as "not ordered by dose" and impeaches the arms that are fine.
    disqualified = nocode * 2 >= len(PROMPTS)
    print(f'  {arm:<12} comment ratio {med:5.3f}   '
          f'median {statistics.median(toks):6.0f} tok   '
          f'n={len(ratios)}/{len(PROMPTS)}'
          f'{f"   NO-CODE {nocode}" if nocode else ""}'
          f'{f"   TRUNC {trunc}" if trunc else ""}'
          f'{"   DISQUALIFIED: stopped writing code" if disqualified else ""}',
          flush=True)
    return None if disqualified else med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='http://127.0.0.1:8899')
    ap.add_argument('--model', default='qwen3.8-flash-next-nvfp4-tp2-ep2')
    ap.add_argument('--arms', required=True,
                    help="comma-separated registered names in dose order; "
                         "use 'base' for the unsteered arm")
    ap.add_argument('--max-tokens', type=int, default=1200)
    ap.add_argument('--samples', default='')
    a = ap.parse_args()

    arms = [s.strip() for s in a.arms.split(',') if s.strip()]
    print(f'{len(PROMPTS)} held-out tasks, temp 0, thinking off, '
          f'max_tokens={a.max_tokens}')
    print('metric: comment lines / total lines inside fenced code blocks\n')

    fh = open(a.samples, 'w') if a.samples else None
    try:
        meds = [(arm, run_arm(a.base, a.model, arm, a.max_tokens, fh))
                for arm in arms]
    finally:
        if fh:
            fh.close()

    clean = [(arm, m) for arm, m in meds if m is not None]
    print('\n=== dose-response ===')
    for arm, m in clean:
        print(f'  {arm:<12} {m:5.3f} {"#" * int(m * 60)}')
    if len(clean) >= 3:
        v = [m for _, m in clean]
        inc = all(v[i] <= v[i + 1] for i in range(len(v) - 1))
        dec = all(v[i] >= v[i + 1] for i in range(len(v) - 1))
        print(f'\nmonotone across arms: '
              f'{"YES" if inc or dec else "NO — not ordered by dose"}')
    if fh:
        print(f'full texts -> {a.samples}')
    print('\nA ratio shift with no change in `lines` is the result. A ratio '
          'shift that only appears\nbecause the model stopped writing code is '
          'not — check the NO-CODE column and read the text.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
