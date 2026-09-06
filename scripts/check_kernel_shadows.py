#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Kernel Structure Enforcer: validate the kernels/{hw} shadowing layout.

The build (`crates/atlas-kernels/build.rs::collect_cu_files`) resolves each
model's kernel set by file stem: a file in `kernels/{hw}/{model}/{quant}/`
shadows its same-stem namesake in `kernels/{hw}/common/`. This script guards
that mechanism against the two defects that silently corrupt a build:

  RULE 1 (shadow == common): a shadow whose resolved content is byte-identical
    to its common namesake. It overrides nothing useful while masking future
    common/ improvements (shadowing is whole-file, not per-symbol — the
    shadow-drift failure class documented in build.rs). Delete it instead.

  RULE 2 (cross-model duplicate): two or more REGULAR (non-symlink) files
    with the same stem and identical content in different model dirs of one
    hardware set. Divergence-prone copies; the sanctioned sharing mechanism
    is a relative symlink to one canonical file (see
    kernels/gb10/holo-3.1-4b/nvfp4/).

Unique shadows (no matching regular file elsewhere) are valid. Symlinks are
valid regardless of what they point to (they are the sharing mechanism).

NOT CHECKED HERE — dropped entry points. A shadow that keeps its namesake's
name but declares FEWER kernels is the third defect of this family, and the one
that actually shipped (the 27B's four multi-sequence GDN decode kernels, gone
until 2026-07-26). Deciding it needs the entry points a source declares, which
means resolving `#define KERNEL_NAME` + `#include` + token-paste macros, and
then filtering by the per-target `[shadow_exempt]` tables. That resolver is
`crates/atlas-kernels/build_shadow.rs`, and it is enforced by
`crates/atlas-kernels/tests/kernel_shadow_detector.rs` in the same CI run as
this script. Reimplementing it here in Python would be a second, silently
diverging copy of the rule — this note exists so the gap in THIS file reads as
a decision rather than an oversight.

Exit 0 when clean; exit 1 and list every violation otherwise.

Usage: scripts/check_kernel_shadows.py [kernels_root]
"""

import hashlib
import os
import sys
from collections import defaultdict
from pathlib import Path

# Hardware set -> kernel source extension (must mirror
# build_target.rs `source_extension()` per vendor).
HW_SOURCE_EXT = {
    "b200": "cu",
    "gb10": "cu",
    "hopper": "cu",
    "metal": "metal",
    "strix": "cu",
    "strix-hip": "cu",
}


def content_hash(path: Path) -> str:
    """SHA-256 of the symlink-resolved file content."""
    return hashlib.sha256(Path(os.path.realpath(path)).read_bytes()).hexdigest()


def collect_hw(hw_dir: Path, ext: str):
    """Return (common_by_stem, shadows) for one hardware set.

    common_by_stem: stem -> content hash.
    shadows: (stem, hash) -> list of (path, is_symlink).
    """
    common_by_stem = {}
    common_dir = hw_dir / "common"
    if common_dir.is_dir():
        for f in sorted(common_dir.glob(f"*.{ext}")):
            common_by_stem[f.stem] = content_hash(f)

    shadows = defaultdict(list)
    for model_dir in sorted(hw_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "common":
            continue
        for quant_dir in sorted(model_dir.iterdir()):
            if not quant_dir.is_dir():
                continue
            for f in sorted(quant_dir.glob(f"*.{ext}")):
                shadows[(f.stem, content_hash(f))].append((f, f.is_symlink()))
    return common_by_stem, shadows


def check_hw(hw_name: str, hw_dir: Path, ext: str) -> list[str]:
    violations = []
    common_by_stem, shadows = collect_hw(hw_dir, ext)

    for (stem, digest), entries in sorted(shadows.items()):
        rels = sorted(str(p.relative_to(hw_dir.parent)) for p, _ in entries)

        # RULE 1: shadow identical to its common namesake.
        if stem in common_by_stem and digest == common_by_stem[stem]:
            violations.append(
                f"RULE1 {hw_name}: shadow {stem} is byte-identical to "
                f"common/{stem}.{ext} (dead override) at {', '.join(rels)}"
            )

        # RULE 2: multiple REGULAR files with identical (stem, content).
        regulars = [p for p, is_link in entries if not is_link]
        if len(regulars) > 1:
            violations.append(
                f"RULE2 {hw_name}: {len(regulars)} identical regular copies of "
                f"{stem}.{ext} — keep one canonical file, symlink the rest:\n    "
                + "\n    ".join(str(p.relative_to(hw_dir.parent)) for p in regulars)
            )
    return violations


def main() -> int:
    kernels_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("kernels")
    if not kernels_root.is_dir():
        print(f"error: kernels root not found: {kernels_root}", file=sys.stderr)
        return 1

    violations = []
    for hw_name, ext in sorted(HW_SOURCE_EXT.items()):
        hw_dir = kernels_root / hw_name
        if hw_dir.is_dir():
            violations.extend(check_hw(hw_name, hw_dir, ext))

    if violations:
        print(f"kernel shadow structure: {len(violations)} violation(s)")
        for v in violations:
            print(f"  {v}")
        return 1

    print("kernel shadow structure: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
