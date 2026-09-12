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

  RULE 3 (undeclared common/ override): a REGULAR file in an INHERITING
    target's `common/` -- kernels/hopper and kernels/b200, whose entries are
    otherwise relative symlinks into kernels/gb10/common -- that the target's
    HARDWARE.toml does not list in `[kernels] overrides`.

    Maintainer rule, 2026-09-11 (tbraun96): "symlinks are fine provided the
    pointed-to gb10 file is not edited when iterating on Hopper; Hopper-tuned
    kernels must be real files under kernels/hopper/." A real file there is
    therefore CORRECT and expected -- but only when it is declared. An
    undeclared one is indistinguishable from a silent fork of a shared kernel,
    which is the defect this whole script exists to catch, and a fork of
    common/ is worse than a fork of a model shadow: it diverges for every
    model on the target at once. A declared override that has VANISHED is the
    mirror-image fault and is reported too.

    This script reports every target's override list on a clean run, so the
    answer to "which kernels does this target tune for itself" is one command
    and not a `find`.

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
import tomllib
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


# Hardware trees whose `common/` MIRRORS another tree's: {mirror: origin}.
# They compile the origin's kernels through relative symlinks, so a regular
# file in their common/ is an override and must be declared. SSOT shared with
# crates/atlas-kernels/tests/support/inherited.rs `INHERITED`; adding a target
# means adding it in both, which is the moment to decide what it inherits.
MIRRORED_COMMON = {
    "b200": "gb10",
    "hopper": "gb10",
}


def declared_overrides(hw_dir: Path) -> set[str]:
    """`[kernels] overrides` from one HARDWARE.toml, as file names."""
    path = hw_dir / "HARDWARE.toml"
    if not path.is_file():
        return set()
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return set(data.get("kernels", {}).get("overrides", []))


def check_common_overrides(hw_name: str, hw_dir: Path) -> tuple[list[str], list[str]]:
    """RULE 3 for one mirrored tree. Returns (violations, override names)."""
    common = hw_dir / "common"
    if not common.is_dir():
        return ([f"RULE3 {hw_name}: no common/ directory to check"], [])
    declared = declared_overrides(hw_dir)
    violations = []
    real = {f.name for f in sorted(common.iterdir()) if not f.is_symlink() and f.is_file()}
    for undeclared in sorted(real - declared):
        violations.append(
            f"RULE3 {hw_name}: common/{undeclared} is a regular file but is not "
            f"listed in kernels/{hw_name}/HARDWARE.toml [kernels] overrides. A "
            f"real file here is how a target owns a tuned kernel -- declare it, "
            f"or make it a relative symlink into the tree it inherits."
        )
    reported = []
    for name in sorted(declared):
        path = common / name
        # `exists()` follows symlinks, so this catches both "deleted" and
        # "declared but dangling" -- a link whose target was renamed away is
        # invisible to `ls` and to git, and surfaces first as an nvcc error.
        if not path.exists():
            violations.append(
                f"RULE3 {hw_name}: common/{name} is declared in [kernels] "
                f"overrides but is missing or does not resolve"
            )
            continue
        # How the target HOLDS it is the interesting half: a real file means
        # this target owns and tunes the source, a symlink means it shares
        # another target's tuning. Both are legitimate; conflating them in the
        # report would hide which tree an edit lands in.
        #
        # And whether the ORIGIN has the same name is the other half. A
        # declared name the origin also carries REPLACES it (the origin keeps
        # its own file, untouched, for the targets that inherit it); a name the
        # origin does not have is an ADDITION. Saying which is what tells a
        # reader whether editing the origin's file would reach this target.
        origin_has = (hw_dir.parent / MIRRORED_COMMON[hw_name] / "common" / name).exists()
        shape = "replaces" if origin_has else "adds"
        if path.is_symlink():
            reported.append(f"{name} ({shape}) -> {os.readlink(path)}")
        else:
            reported.append(f"{name} ({shape}, own source)")
    return (violations, reported)


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

    # ★ Every hardware tree in the map must EXIST. The loop below skips a
    # missing one, so a rename or a move of `kernels/gb10` left this required
    # check scanning nothing and printing "kernel shadow structure: OK" --
    # verified by running it against an empty `kernels/`: rc=0. A gate that
    # cannot see its inputs must refuse, not congratulate. HW_SOURCE_EXT is the
    # SSOT for which trees are covered; adding or removing hardware means
    # editing it in the same commit, which is exactly the moment to notice.
    missing = [hw for hw in sorted(HW_SOURCE_EXT) if not (kernels_root / hw).is_dir()]
    if missing:
        print(
            f"error: {kernels_root}/ is missing hardware tree(s): {', '.join(missing)}.\n"
            f"       They are listed in HW_SOURCE_EXT, so this check believes it covers\n"
            f"       them -- and it silently scanned nothing instead. If a tree moved or\n"
            f"       was retired, update HW_SOURCE_EXT in the same commit.",
            file=sys.stderr,
        )
        return 1

    violations = []
    overrides_by_hw: dict[str, list[str]] = {}
    for hw_name, ext in sorted(HW_SOURCE_EXT.items()):
        hw_dir = kernels_root / hw_name
        violations.extend(check_hw(hw_name, hw_dir, ext))
        if hw_name in MIRRORED_COMMON:
            hw_violations, overrides = check_common_overrides(hw_name, hw_dir)
            violations.extend(hw_violations)
            overrides_by_hw[hw_name] = overrides

    if violations:
        print(f"kernel shadow structure: {len(violations)} violation(s)")
        for v in violations:
            print(f"  {v}")
        return 1

    print(
        "kernel shadow structure: OK "
        f"({len(HW_SOURCE_EXT)} hardware trees scanned: {', '.join(sorted(HW_SOURCE_EXT))})"
    )
    # Which kernels each inheriting target owns, on a clean run. The point of
    # declaring overrides is that this question has an answer; printing it is
    # what makes the answer reachable without reading the tree.
    for hw_name in sorted(overrides_by_hw):
        overrides = overrides_by_hw[hw_name]
        origin = MIRRORED_COMMON[hw_name]
        if overrides:
            print(
                f"  {hw_name}/common declares {len(overrides)} override(s) of "
                f"{origin}/common — `replaces` = {origin} has the same name and "
                f"keeps its own copy, `adds` = {origin} does not have it at all:"
            )
            for entry in overrides:
                print(f"      {entry}")
        else:
            print(f"  {hw_name}/common inherits every kernel from {origin}/common")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
