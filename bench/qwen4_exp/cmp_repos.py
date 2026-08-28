"""Side-by-side of the two NVFP4 releases of Qwen3.8-Flash-Next.

Same base model, same architecture, different packaging. The question is
what a loader has to do differently, so this reports per-family footprint
AND the tensor-name shapes that decide loader work.
"""
import json
import re
import urllib.request
from collections import defaultdict

GB = 1 << 30
REPOS = {
    'RadixArk': 'RadixArk/Qwen3.8-Flash-Next-NVFP4',
    'Inferact': 'Inferact/Qwen3.8-Flash-Next-NVFP4',
}


def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'atlas-cmp'})
    return json.load(urllib.request.urlopen(req, timeout=300))


def fam(k):
    if '.ple.' in k or 'ngram_embedding' in k:
        return 'PLE n-gram tables'
    if k.startswith('mtp.') or '.mtp.' in k:
        return 'MTP'
    if 'visual' in k:
        return 'vision'
    if re.search(r'\.mlp\.experts\.|experts\.gate_up_proj|experts\.down_proj', k):
        return 'routed experts'
    return 'backbone'


for label, repo in REPOS.items():
    idx = get(f'https://huggingface.co/{repo}/resolve/main/model.safetensors.index.json')
    wm = idx['weight_map']
    tot = idx['metadata']['total_size']

    # Per-family tensor counts (bytes need headers; totals come from the index).
    n = defaultdict(int)
    for k in wm:
        n[fam(k)] += 1

    # Distinct file-name shapes, so the packaging difference is visible.
    shapes = sorted({re.sub(r'\d+', 'N', f) for f in set(wm.values())})

    print(f'=== {label} ({repo}) ===')
    print(f'  index total_size: {tot/GB:.2f} GiB   files: {len(set(wm.values()))}   '
          f'tensors: {len(wm)}')
    print(f'  {"tensors":>9}  family')
    for k in sorted(n, key=lambda x: -n[x]):
        print(f'  {n[k]:9d}  {k}')
    print('  file-name shapes:')
    for s in shapes:
        print(f'    {s}')
    print()
