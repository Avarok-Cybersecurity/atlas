"""Why does 74.82 GB of tensors occupy 86.16 GB of GPU memory?

302,488 allocations, and this checkpoint's NVFP4 layout is scalar-heavy:
every expert projection carries `input_scale` and `weight_scale_2` as
0-dim F32 tensors. If each allocation is rounded up to a granularity, tiny
tensors dominate the waste — and 11.3 GB is the difference between fitting
on one Spark and not.
"""
import json
import os
from collections import Counter

SNAP = ('/tank/hf/hub/models--Inferact--Qwen3.8-Flash-Next-NVFP4/'
        'snapshots/129972269565f7f4f664fdf8dd42268d3bbda9fd')
GB = 1 << 30
DT = {'BF16': 2, 'F16': 2, 'F32': 4, 'F8_E4M3': 1, 'F8_E5M2': 1,
      'I8': 1, 'U8': 1, 'I32': 4, 'I64': 8}

import struct
idx = json.load(open(os.path.join(SNAP, 'model.safetensors.index.json')))
wm = idx['weight_map']

# Read headers for every file once.
sizes = []
for f in sorted(set(wm.values())):
    p = os.path.join(SNAP, f)
    with open(p, 'rb') as fh:
        n = struct.unpack('<Q', fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    for k, v in hdr.items():
        if k == '__metadata__':
            continue
        if 'ngram_embedding.shard_' in k:
            continue  # deferred, never uploaded
        nel = 1
        for d in v['shape']:
            nel *= d
        sizes.append(nel * DT.get(v['dtype'], 2))

tot = sum(sizes)
print(f'uploaded tensors : {len(sizes)}')
print(f'raw bytes        : {tot/GB:.2f} GiB')

buckets = Counter()
for s in sizes:
    if s <= 4:
        buckets['<=4 B (scalars)'] += 1
    elif s <= 1024:
        buckets['<=1 KB'] += 1
    elif s <= 65536:
        buckets['<=64 KB'] += 1
    else:
        buckets['>64 KB'] += 1
print('\nsize distribution:')
for k, v in buckets.most_common():
    print(f'  {v:7d}  {k}')

print('\nfootprint if every allocation rounds up to G:')
for g in (4096, 65536, 131072, 2 << 20):
    padded = sum((s + g - 1) // g * g for s in sizes)
    label = f'{g//1024} KB' if g < (1 << 20) else f'{g//(1<<20)} MB'
    print(f'  granularity {label:>6}: {padded/GB:7.2f} GiB  '
          f'(+{(padded-tot)/GB:6.2f} GiB overhead)')
print('\nmeasured on GPU: 86.16 GiB used after upload (+11.34 over raw)')
