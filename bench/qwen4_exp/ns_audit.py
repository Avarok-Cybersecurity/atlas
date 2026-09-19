"""Every tensor family in the checkpoint, and where a loader must put it.

296,475 tensors collapse to ~111 families. A loader that silently skips a
family produces a model that runs and is wrong, so the useful artifact is
the COMPLETE list with an explicit destination for each — including the ones
we intend to drop.
"""
import json
import os
import re
from collections import defaultdict

SNAP = ('/tank/hf/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/'
        'snapshots/7b719225242aacd3dbd3f9407468c2ee9a9d2594')

idx = json.load(open(os.path.join(SNAP, 'model.safetensors.index.json')))
wm = idx['weight_map']

fam = defaultdict(int)
for k in wm:
    n = re.sub(r'\.layers\.\d+\.', '.layers.N.', k)
    n = re.sub(r'\.blocks\.\d+\.', '.blocks.N.', n)
    n = re.sub(r'\.experts\.\d+\.', '.experts.N.', n)
    n = re.sub(r'shard_\d+', 'shard_N', n)
    fam[n] += 1


def dest(k):
    if '.ple.' in k or 'ngram_embedding' in k:
        return 'C  PLE n-gram'
    if k.startswith('mtp.'):
        return 'I  MTP (drop v1)'
    if 'visual' in k:
        return 'H  vision (have)'
    if 'hyper_connection' in k:
        return 'B  mHC'
    if '.indexer.' in k:
        return 'D  QSA indexer'
    if '.linear_attn.' in k:
        return 'F  GDN'
    if '.mlp.experts.' in k:
        return 'G  MoE experts'
    if '.mlp.' in k or '.mlp_' in k:
        return 'G  MoE shared/router'
    if '.self_attn.' in k:
        return 'E  attention'
    if 'embed_tokens' in k or 'lm_head' in k or k.endswith('norm.weight'):
        return 'A  embed/head/norm'
    return '?? UNCLASSIFIED'


groups = defaultdict(lambda: [0, 0])  # families, tensors
for f, c in fam.items():
    g = groups[dest(f)]
    g[0] += 1
    g[1] += c

print(f'{"families":>9} {"tensors":>9}  destination (issue #753 item)')
for k in sorted(groups):
    fams, cnt = groups[k]
    print(f'{fams:9d} {cnt:9d}  {k}')
print(f'{len(fam):9d} {sum(fam.values()):9d}  TOTAL')

unc = sorted(f for f in fam if dest(f).startswith('??'))
if unc:
    print(f'\nUNCLASSIFIED ({len(unc)}):')
    for f in unc:
        print(f'  {f}')
else:
    print('\nno unclassified families')

print('\n--- per-destination families ---')
for want in sorted(groups):
    if want.startswith(('H', 'I')):
        continue  # have / dropping; not this phase's checklist
    print(f'\n{want}:')
    for f in sorted(f for f in fam if dest(f) == want):
        print(f'    {f}')
