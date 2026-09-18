# SPDX-License-Identifier: AGPL-3.0-only

# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///

"""Turn two capture dumps into a llama.cpp-compatible control-vector GGUF.

    uv run scripts/derive_control_vector.py positive.bin negative.bin out.gguf \
        [--model-hint qwen4exp] [--layers 4-44]

A control vector is a contrastive MEAN DIFFERENCE — no gradients, no training.
Each dump is a per-layer running SUM of stream-mean activations plus the token
count it represents; this divides to get means, subtracts negative from
positive, and normalises each layer to unit length.

Unit-normalising is what makes the result drop into the existing apply path:
`ControlVector::load` in project mode folds a stored vector's norm into the
per-layer scale, so unit rows at user scale 1.0 give s = 1.0 — the same
convention as the published artifact.

The output is validated by round-tripping through
`scripts/inspect_control_vector.py`, which is the same reader the engine uses.
"""

import argparse
import struct
import sys

import numpy as np

DUMP_MAGIC = 0x43564341  # "ACVC"
GGUF_MAGIC = 0x46554747  # "GGUF"


def read_dump(path):
    """(sums [n_layer, hidden] float64, tokens) from a capture dump."""
    raw = open(path, 'rb').read()
    if len(raw) < 32:
        raise SystemExit(f'{path}: too short to be a capture dump')
    magic, n_layer, hidden, _pad, tokens, _res = struct.unpack_from('<IIIIQQ', raw, 0)
    if magic != DUMP_MAGIC:
        raise SystemExit(f'{path}: bad magic {magic:#x}, expected {DUMP_MAGIC:#x}')
    want = 32 + n_layer * hidden * 8
    if len(raw) != want:
        raise SystemExit(f'{path}: expected {want} bytes for [{n_layer}, {hidden}], got {len(raw)}')
    if tokens == 0:
        raise SystemExit(f'{path}: zero tokens — nothing was captured')
    sums = np.frombuffer(raw, dtype='<f8', count=n_layer * hidden, offset=32)
    return sums.reshape(n_layer, hidden).copy(), tokens


def write_gguf(path, directions, model_hint):
    """Write `direction.N` F32 tensors, matching llama.cpp's control-vector layout."""
    n_layer, hidden = directions.shape
    # Layer 0 never carries a direction (llama.cpp has none and the engine
    # rejects one), so tensors run 1..n_layer-1.
    idx = [il for il in range(1, n_layer) if np.any(directions[il])]

    def s(x):
        b = x.encode()
        return struct.pack('<Q', len(b)) + b

    out = bytearray()
    out += struct.pack('<I', GGUF_MAGIC)
    out += struct.pack('<I', 3)                      # version
    out += struct.pack('<Q', len(idx))               # tensor count
    out += struct.pack('<Q', 3)                      # kv count
    out += s('general.architecture') + struct.pack('<I', 8) + s('controlvector')
    out += s('controlvector.model_hint') + struct.pack('<I', 8) + s(model_hint)
    out += s('controlvector.layer_count') + struct.pack('<I', 5) + struct.pack('<i', max(idx))

    off = 0
    for il in idx:
        out += s(f'direction.{il}')
        out += struct.pack('<I', 1)                  # n_dims
        out += struct.pack('<Q', hidden)
        out += struct.pack('<I', 0)                  # ggml F32
        out += struct.pack('<Q', off)
        off += hidden * 4
    while len(out) % 32:                             # general.alignment default
        out += b'\0'
    for il in idx:
        out += directions[il].astype('<f4').tobytes()
    open(path, 'wb').write(bytes(out))
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('positive', help='capture dump for the POSITIVE corpus')
    ap.add_argument('negative', help='capture dump for the NEGATIVE corpus')
    ap.add_argument('out', help='output .gguf')
    ap.add_argument('--model-hint', default='qwen4exp')
    ap.add_argument('--layers', default=None,
                    help='A-B inclusive; layers outside are zeroed and omitted')
    ap.add_argument('--magnitude', choices=('unit', 'raw'), default='unit',
                    help='unit: normalise each layer to length 1 — for PROJECT '
                         'mode, which uses direction only (default). '
                         'raw: keep |mean-diff| per layer — for ADD mode, '
                         'which applies the row verbatim under one global '
                         'scale and so needs the per-layer magnitude.')
    args = ap.parse_args()

    pos, pos_tok = read_dump(args.positive)
    neg, neg_tok = read_dump(args.negative)
    if pos.shape != neg.shape:
        raise SystemExit(f'geometry mismatch: {pos.shape} vs {neg.shape}')
    n_layer, hidden = pos.shape
    print(f'positive: {pos_tok} tokens   negative: {neg_tok} tokens   '
          f'[{n_layer}, {hidden}]')
    # An extreme imbalance is not fatal but it skews the contrast, and it is
    # usually a sign one pass was cut short rather than a deliberate choice.
    ratio = max(pos_tok, neg_tok) / min(pos_tok, neg_tok)
    if ratio > 1.5:
        print(f'WARNING: corpora differ {ratio:.2f}x in tokens — the means are '
              f'still valid but the sets are not matched; consider rebalancing.')

    diff = (pos / pos_tok) - (neg / neg_tok)

    lo, hi = 1, n_layer - 1
    if args.layers:
        a, b = args.layers.split('-')
        lo, hi = max(1, int(a)), min(n_layer - 1, int(b))
    keep = np.zeros_like(diff)
    keep[lo:hi + 1] = diff[lo:hi + 1]

    norms = np.linalg.norm(keep, axis=1)
    live = norms > 0
    unit = keep.copy()
    unit[live] /= norms[live][:, None]

    # Which magnitude to WRITE depends on the operator the file is for, and
    # getting it wrong is silent rather than loud.
    #
    #   project (h -= s*(h.v)v) only uses the DIRECTION: `ControlVector::load`
    #   folds a stored row's norm into the per-layer scale, so unit rows at
    #   user scale 1.0 reproduce llama.cpp exactly. Magnitude is redundant.
    #
    #   add (h += s*row) uses the row VERBATIM with one GLOBAL scalar. Unit
    #   rows therefore ask a single number to serve every layer at once — and
    #   |mean-diff| spans ~11x across a typical active range (0.20 to 2.13 over
    #   layers 4..44 for the shipped verbosity example). Any global scalar is
    #   then simultaneously too large somewhere and too small somewhere else,
    #   and the usual outcome is a dose of nearly zero reported as "add mode
    #   does nothing".
    #
    # Keeping the raw magnitudes makes scale 1.0 mean "one unit of the measured
    # contrast, correctly weighted at every layer", so 2.0 and 4.0 are honest
    # multiples and a dose-response curve is interpretable.
    out_rows = unit if args.magnitude == 'unit' else keep
    idx = write_gguf(args.out, out_rows, args.model_hint)
    print(f'wrote {args.out}: {len(idx)} directions, layers {min(idx)}..{max(idx)}'
          f'  [magnitude={args.magnitude}'
          f'{" - for project mode" if args.magnitude == "unit" else " - for add mode"}]')
    print(f'raw |mean-diff| per layer: min={norms[live].min():.6f} '
          f'max={norms[live].max():.6f} median={np.median(norms[live]):.6f}')
    # Cross-layer coherence: one feature carried across depth, or noise? Random
    # directions in `hidden` dims would sit near zero.
    a = unit[lo:hi + 1][norms[lo:hi + 1] > 0]
    if len(a) > 1:
        cos = a @ a.T
        n = cos.shape[0]
        print(f'adjacent-layer cosine mean={np.mean(np.diag(cos, 1)):.4f}   '
              f'all-pairs off-diagonal mean={np.mean(cos[np.triu_indices(n, 1)]):.4f}')
        print('(low values here mean the contrast did not find a consistent '
              'direction — check the corpus before trusting the vector)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
