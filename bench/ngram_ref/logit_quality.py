"""Score one LongCat precision arm against the reference logits.

The per-sublayer cosine in `full_forward_cmp.py` says the port is wired
correctly. It does NOT say what the output distribution costs, because a
0.995 hidden-state cosine can still reorder the top of a 131072-way softmax.
This script closes that: it compares Atlas's FINAL logits against the HF
reference's, which is what actually decides the emitted token.

Reference: `longcat_forward_golden.npz['logits_last']`, f32, full vocab.
Atlas:     `$ATLAS_NEMO_DUMP/atlas_logits.bin`, f32, full vocab, written by
           prefill_b/finalize_last.rs step 7.

Why the dump and not the API: `/v1/completions` logprobs are keyed by decoded
STRING and capped at 20, so two ids that decode alike collide and the tail is
invisible. The dump is the raw vector.

Usage:
    ATLAS_NEMO_DUMP=<dir> ./serve_longcat_tui.sh          # arm under test
    logit_quality.py <port> <dumpdir> [label]
"""
import json
import os
import sys

import numpy as np
import requests

PORT = sys.argv[1] if len(sys.argv) > 1 else '8888'
DUMP = sys.argv[2] if len(sys.argv) > 2 else '/home/ms/.claude/jobs/5a7bd33d/tmp/nemodump'
LABEL = sys.argv[3] if len(sys.argv) > 3 else 'arm'

HERE = os.path.dirname(os.path.abspath(__file__))
gold = np.load(f'{HERE}/longcat_forward_golden.npz')
tokens = [int(t) for t in gold['tokens']]
ref = gold['logits_last'].astype(np.float64)

path = os.path.join(DUMP, 'atlas_logits.bin')
if os.path.exists(path):
    os.remove(path)

r = requests.post(f'http://127.0.0.1:{PORT}/v1/completions',
                  json={'model': 'longcat-full', 'prompt': tokens,
                        'max_tokens': 1, 'temperature': 0}, timeout=1800).json()
if 'choices' not in r:
    print('server error:', json.dumps(r)[:400])
    sys.exit(1)

if not os.path.exists(path):
    print(f'no dump at {path} — is ATLAS_NEMO_DUMP set on the SERVER process?')
    sys.exit(1)

got = np.fromfile(path, dtype=np.float32).astype(np.float64)
n = min(len(got), len(ref))
got, ref = got[:n], ref[:n]


def logsoftmax(x):
    m = x.max()
    e = np.exp(x - m)
    return (x - m) - np.log(e.sum())


lp_ref, lp_got = logsoftmax(ref), logsoftmax(got)
p_ref = np.exp(lp_ref)

# Rank metrics: what the sampler actually sees.
o_ref = np.argsort(-ref)
o_got = np.argsort(-got)
top1 = int(o_ref[0] == o_got[0])
ov = {k: len(set(o_ref[:k].tolist()) & set(o_got[:k].tolist())) for k in (1, 5, 10, 20, 100)}

cos = float(ref @ got / (np.linalg.norm(ref) * np.linalg.norm(got)))
# Full-vocab KL(ref || atlas) in nats — the honest distribution distance.
kl = float((p_ref * (lp_ref - lp_got)).sum())
# How much probability mass the reference's own top-1 keeps under Atlas.
p_ref_top1 = float(p_ref[o_ref[0]])
p_got_at_ref_top1 = float(np.exp(lp_got[o_ref[0]]))

print(f'=== {LABEL} ===')
print(f'served text        : {r["choices"][0]["text"]!r}')
print(f'ref  top-1 id      : {int(o_ref[0])}  (p={p_ref_top1:.4f})')
print(f'atlas top-1 id     : {int(o_got[0])}  (p at ref top-1 = {p_got_at_ref_top1:.4f})')
print(f'top-1 agree        : {"YES" if top1 else "NO"}')
for k in (5, 10, 20, 100):
    print(f'top-{k:<3d} overlap    : {ov[k]}/{k}')
print(f'logit cosine       : {cos:.6f}')
print(f'KL(ref||atlas)     : {kl:.6f} nats')

out = os.path.join(DUMP, f'quality_{LABEL}.json')
json.dump({'label': LABEL, 'top1_agree': top1, 'overlap': ov, 'cos': cos,
           'kl': kl, 'p_ref_top1': p_ref_top1,
           'p_got_at_ref_top1': p_got_at_ref_top1,
           'ref_top1_id': int(o_ref[0]), 'atlas_top1_id': int(o_got[0]),
           'text': r['choices'][0]['text']}, open(out, 'w'), indent=2)
print(f'\nwrote {out}')
