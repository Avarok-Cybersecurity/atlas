"""Authoritative per-family footprint, summed from the safetensors index.

My earlier Hub tree-walk gave 126.0 GB but the index says total_size =
135.2 GB, so the walk undercounted (pagination). This sums real file sizes:
local blobs where present, Hub Content-Length for the few still downloading.
The resident-vs-deferred split is the number the single-Spark decision rests
on, so it should not come from a paginated listing.
"""
import json
import os
import urllib.request
from collections import defaultdict

GB = 1 << 30
SNAP = ('/tank/hf/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/'
        'snapshots/7b719225242aacd3dbd3f9407468c2ee9a9d2594')
REPO = 'RadixArk/Qwen3.8-Flash-Next-NVFP4'

idx = json.load(open(os.path.join(SNAP, 'model.safetensors.index.json')))
files = sorted(set(idx['weight_map'].values()))


def remote_size(name):
    url = f'https://huggingface.co/{REPO}/resolve/main/{name}'
    req = urllib.request.Request(url, method='HEAD',
                                 headers={'User-Agent': 'atlas-sizing'})
    r = urllib.request.urlopen(req, timeout=120)
    n = r.headers.get('X-Linked-Size') or r.headers.get('Content-Length')
    return int(n)


def fam(f):
    if f.startswith('layer-'):
        return 'routed experts (NVFP4)'
    if f.startswith('model-bf16-'):
        return 'backbone (BF16)'
    if f.startswith('model-plefp8-'):
        return 'PLE n-gram tables (FP8)'
    return 'other'


agg, cnt, remote = defaultdict(int), defaultdict(int), 0
for f in files:
    p = os.path.join(SNAP, f)
    if os.path.exists(p):
        sz = os.path.getsize(os.path.realpath(p))
    else:
        sz = remote_size(f)
        remote += 1
    agg[fam(f)] += sz
    cnt[fam(f)] += 1

tot = sum(agg.values())
print(f'({remote} file(s) sized from the Hub — still downloading)\n')
print(f'{"GB":>9}  {"files":>5}  family')
for k in sorted(agg, key=lambda x: -agg[x]):
    print(f'{agg[k]/GB:9.2f}  {cnt[k]:5d}  {k}')
print(f'{tot/GB:9.2f}  {sum(cnt.values()):5d}  TOTAL')
print(f'index total_size: {idx["metadata"]["total_size"]/GB:.2f} GB (tensor bytes, excl. headers)')

ple = agg['PLE n-gram tables (FP8)']
mtp = 4.7  # mtp.layers.0 experts, BF16, from the header scan
print(f'\nresident if PLE deferred to NVMe: {(tot-ple)/GB:.2f} GB')
print(f'  ... and MTP dropped for v1:      {(tot-ple)/GB - mtp:.2f} GB')
print(f'budget: 121.6 GB box x 0.80 util = 97.3 GB')
print(f'headroom for KV: {97.3 - ((tot-ple)/GB - mtp):.2f} GB (MTP dropped)')
