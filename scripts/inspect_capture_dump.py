# SPDX-License-Identifier: AGPL-3.0-only

# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///

"""Inspect capture dumps and size an `add`-mode scale from measured magnitudes.

    uv run scripts/inspect_capture_dump.py positive.bin negative.bin

Sits between capture and derive. `inspect_control_vector.py` reads the finished
GGUF; this reads the raw FP64 accumulators the serve wrote, which still carry
the one thing the GGUF can lose: how big the residual stream actually is.

# Why this is not optional before using `add`

The two operators consume magnitude differently:

    project   h -= s * (h.v) v      direction only; `ControlVector::load`
                                    folds a stored row's norm into the scale,
                                    so magnitude cancels and any scaling of
                                    the file behaves identically.
    add       h += s * row          the row is applied VERBATIM under one
                                    GLOBAL scalar.

So for `add` the question "what is s?" has no answer that does not reference the
stream it is being added to. Guessing invites a specific, quiet failure: pick s
three orders of magnitude too small and the model is untouched, which reads in
the results table as "add mode does nothing" — a dose of zero reported as a
null result. Too large and the output degenerates, which reads as "the vector
is harmful". Neither is a fact about the vector.

# The divisor

The device accumulator is a SUM over every token seen, with the token count in
the header. A verbose corpus is longer than its terse pair by construction, so
subtracting the raw sums subtracts one corpus's worth of tokens from another's
and calls the residue a direction. Always divide first.

# What the numbers mean

`|mean|` is the norm of the per-token MEAN activation. Averaging across tokens
cancels whatever varies between them, so it is a LOWER BOUND on a typical
token's norm, not an estimate of it. Treat it as a floor: scales far below it
are certainly gentle, scales far above it are certainly aggressive, and the
middle deserves a measured dose-response rather than a claim.

`|diff| / |mean|` is the fraction of the stream that lies along the contrast. It
also bounds what `project` can do — projection can only remove what is there, so
a tiny ratio means a null result describes the axis rather than the operator.
"""

import argparse
import pathlib
import struct
import sys

import numpy as np

DUMP_MAGIC = 0x4356_4341  # "ACVC"
HEADER = 32


def load(path):
    """Return `(n_layer, hidden, per-token mean)` for one dump."""
    raw = pathlib.Path(path).read_bytes()
    magic, n_layer, hidden = struct.unpack_from('<Iii', raw, 0)
    if magic != DUMP_MAGIC:
        raise SystemExit(f'{path}: bad magic 0x{magic:08x} — not a capture dump')
    (tokens,) = struct.unpack_from('<Q', raw, 16)
    if tokens == 0:
        raise SystemExit(f'{path}: header says zero tokens')
    need = HEADER + n_layer * hidden * 8
    if len(raw) < need:
        raise SystemExit(f'{path}: truncated ({len(raw)} bytes, need {need})')
    data = np.frombuffer(raw, dtype='<f8', count=n_layer * hidden,
                         offset=HEADER).reshape(n_layer, hidden) / float(tokens)
    print(f'{path}: {tokens} tokens, [{n_layer}, {hidden}]')
    return n_layer, hidden, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('positive')
    ap.add_argument('negative')
    ap.add_argument('--layers', default=None,
                    help='A-B inclusive, to match what derive will keep')
    a = ap.parse_args()

    nl, h, pos = load(a.positive)
    nl2, h2, neg = load(a.negative)
    if (nl, h) != (nl2, h2):
        raise SystemExit(f'geometry mismatch: [{nl},{h}] vs [{nl2},{h2}]')

    lo, hi = 1, nl - 1
    if a.layers:
        x, y = a.layers.split('-')
        lo, hi = max(1, int(x)), min(nl - 1, int(y))

    act = (np.linalg.norm(pos, axis=1) + np.linalg.norm(neg, axis=1)) / 2.0
    dn = np.linalg.norm(pos - neg, axis=1)

    print(f'\n{"layer":>5} {"|mean_act|":>11} {"|diff|":>10} {"diff/act":>9}')
    step = max(1, (hi - lo + 1) // 12)
    for i in range(lo, hi + 1, step):
        r = dn[i] / act[i] if act[i] else 0.0
        print(f'{i:5} {act[i]:11.4f} {dn[i]:10.4f} {r:9.4f}')

    sl = slice(lo, hi + 1)
    live = act[sl] > 0
    a_l, d_l = act[sl][live], dn[sl][live]
    ratio = d_l / a_l
    print(f'\nover layers {lo}..{hi}:')
    print(f'  |mean_act| : median {np.median(a_l):.4f}  '
          f'min {a_l.min():.4f}  max {a_l.max():.4f}')
    print(f'  |diff|     : median {np.median(d_l):.4f}  '
          f'min {d_l.min():.4f}  max {d_l.max():.4f}   '
          f'(spans {d_l.max() / d_l.min():.0f}x across depth)')
    print(f'  diff/act   : median {np.median(ratio):.4f}  '
          f'min {ratio.min():.4f}  max {ratio.max():.4f}')

    print('\nadd-mode scales, against a RAW-magnitude file '
          '(derive --magnitude raw):')
    print('  scale 1.0 = one unit of the measured contrast, already weighted '
          'correctly per layer.')
    print('  Sweep 0.5 / 1 / 2 / 4 and read the curve; a real axis orders '
          'itself by dose.')
    print('\nadd-mode scales, against a UNIT file (the project-mode default):')
    print('  There is no single right answer, which is the point. One global '
          'scalar must cover a')
    print(f'  {d_l.max() / d_l.min():.0f}x magnitude range across layers, so it '
          'is wrong at nearly every layer.')
    print(f'  If you must: s ~ {np.median(d_l):.3f} reproduces the median '
          'layer and mis-doses the rest.')
    if np.median(ratio) < 0.05:
        print('\nWARNING: the contrast is a very small fraction of the stream. '
              'A weak result here is a statement about the CORPUS, not about '
              'the operators — revisit the pairing before sweeping scales.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
