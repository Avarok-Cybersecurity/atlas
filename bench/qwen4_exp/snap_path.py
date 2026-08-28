# SPDX-License-Identifier: AGPL-3.0-only
"""Where the Qwen3.8-Flash-Next snapshot is, on whichever box this is.

Every reference script here hardcoded one absolute path under `/tank/hf`, which
is a directory on one particular machine. On any other box — including a DGX
with the model sitting in the ordinary HuggingFace cache — the scripts failed
with a path nobody could act on, and the goldens they generate are exactly what
you need when you are debugging a checkpoint that will not load.

Resolution order, first hit wins:

1. `QWEN4EXP_PATH`, if set — an explicit choice always beats a search.
2. The caller's own default, if it exists. That keeps the `/tank/hf` path
   working where it is real, so nothing changes on the machine it was written
   for.
3. The HuggingFace cache: `models--<org>--<repo>/snapshots/<rev>`, newest
   revision. Both published conversions are looked for, `Inferact` FIRST
   because that is what these references were calibrated against — the golden
   generators read PLE rows without applying `weight_scale`, which is right for
   Inferact's BF16 table and wrong for RadixArk's FP8 one.

Raises with all three candidates named, rather than returning a path that is
not there.
"""

import glob
import os

_CACHE = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
# Order matters; see the module docstring.
_REPOS = (
    "models--Inferact--Qwen3.8-Flash-Next-NVFP4",
    "models--RadixArk--Qwen3.8-Flash-Next-NVFP4",
)


def _from_cache() -> str | None:
    for repo in _REPOS:
        snaps = sorted(
            glob.glob(os.path.join(_CACHE, "hub", repo, "snapshots", "*")),
            key=os.path.getmtime,
            reverse=True,
        )
        for s in snaps:
            # A snapshot dir with no config.json is a partial download.
            if os.path.exists(os.path.join(s, "config.json")):
                return s
    return None


def resolve_snapshot(default: str | None = None) -> str:
    """The snapshot directory to read, or raise saying where it looked."""
    env = os.environ.get("QWEN4EXP_PATH")
    if env:
        if not os.path.isdir(env):
            raise SystemExit(f"QWEN4EXP_PATH={env} is not a directory")
        return env
    if default and os.path.isdir(default):
        return default
    found = _from_cache()
    if found:
        return found
    raise SystemExit(
        "cannot find a Qwen3.8-Flash-Next snapshot. Looked at:\n"
        f"  $QWEN4EXP_PATH   (unset)\n"
        f"  {default or '(no default)'}\n"
        f"  {os.path.join(_CACHE, 'hub')}/{{{','.join(_REPOS)}}}/snapshots/*\n"
        "Set QWEN4EXP_PATH, or `hf download Inferact/Qwen3.8-Flash-Next-NVFP4`."
    )
