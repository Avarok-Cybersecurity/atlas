"""Which kernel entry points does the qwen3.6-35b-a3b shadow add over common/?

Iterating one missing symbol per 4-minute model load is the slow way to
discover the shadow set. This lists, per overridden file stem, the
`extern "C" __global__` names the shadow defines that common/ does not.
"""
import os
import re

ROOT = '/home/ms/atlas/.claude/worktrees/nemo-behavior/kernels/gb10'
SHADOW = os.path.join(ROOT, 'qwen3.6-35b-a3b/nvfp4')
COMMON = os.path.join(ROOT, 'common')

# `__launch_bounds__(...)` and other attributes can sit between `__global__`
# and the name, so skip any non-`(` tokens before the identifier that is
# followed by the parameter list. Getting this wrong under-reports a
# shadow's entry points, which is how the GDN split kernels were missed.
PAT = re.compile(r'extern\s+"C"\s+__global__\s+(?:[\w()\s]*?\s)??void\s+(\w+)\s*\(')


def entries(path):
    try:
        return set(PAT.findall(open(path, errors='replace').read()))
    except FileNotFoundError:
        return set()


for f in sorted(os.listdir(SHADOW)):
    if not f.endswith('.cu'):
        continue
    s = entries(os.path.join(SHADOW, f))
    c = entries(os.path.join(COMMON, f))
    only = sorted(s - c)
    print(f'{f}: shadow {len(s)}, common {len(c)}, shadow-only {len(only)}')
    for n in only:
        print(f'    + {n}')
