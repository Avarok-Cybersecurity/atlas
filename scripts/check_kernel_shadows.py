#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Kernel Structure Enforcer: validate the kernels/{hw} shadowing layout.

The build (`crates/avarok-kernels/build.rs::collect_cu_files`) resolves each
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

  RULE 3 (undeclared common/ override, undeclared common/ OMISSION): a
    REGULAR file in an INHERITING target's `common/` -- kernels/hopper,
    kernels/b200 and kernels/r9700, whose entries are otherwise relative
    symlinks into kernels/gb10/common -- that the target's HARDWARE.toml does
    not list in `[kernels] overrides`; and, the mirror image, an entry the
    origin has that the mirror does NOT carry and that the same HARDWARE.toml
    does not list in `[kernels] absent`.

    The omission half is what kernels/r9700 is for. Its mirror is SUBTRACTIVE:
    ten .cu and one header of kernels/gb10/common do not compile with SCALE
    1.7.1 for gfx1201, so they are not linked, and each is declared
    `[expected_absent]` in the MODEL.tomls so the boot audit accepts the gap.
    Silence is not available as a third option: the curated 99-entry tree that
    preceded the mirror was missing dense_gemv_bf16_batch2 by nothing more
    than not having been updated when gb10 gained it, and the only signal was
    a model build that died at "Module 'dense_gemv_bf16_batch2' not loaded".
    An undeclared omission now fails here instead.

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
`crates/avarok-kernels/build_shadow.rs`, and it is enforced by
`crates/avarok-kernels/tests/kernel_shadow_detector.rs` in the same CI run as
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
    "r9700": "cu",
    "strix": "cu",
    "strix-hip": "cu",
}


# Hardware trees whose `common/` MIRRORS another tree's: {mirror: origin}.
# They compile the origin's kernels through relative symlinks, so a regular
# file in their common/ is an override and must be declared, and an origin
# entry they do not carry is an omission and must be declared too.
#
# crates/avarok-kernels/tests/support/inherited.rs `INHERITED` is the Rust-side
# list and it is a SUBSET of this one, deliberately: it asserts vendor =
# "nvidia" and drives the Hopper/B200 campaign's HARDWARE.toml and MODEL.toml
# parity checks, none of which describe an AMD/SCALE target. r9700 mirrors the
# same origin and gets the same structural guard here; adding a target to
# either list is the moment to decide whether it belongs in the other.
MIRRORED_COMMON = {
    "b200": "gb10",
    "hopper": "gb10",
    "r9700": "gb10",
}


def declared_kernels(hw_dir: Path, key: str) -> set[str]:
    """`[kernels] <key>` from one HARDWARE.toml, as file names."""
    path = hw_dir / "HARDWARE.toml"
    if not path.is_file():
        return set()
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return set(data.get("kernels", {}).get(key, []))


def check_common_overrides(
    hw_name: str, hw_dir: Path
) -> tuple[list[str], list[str], list[str]]:
    """RULE 3 for one mirrored tree.

    Returns (violations, override names, declared omissions).
    """
    common = hw_dir / "common"
    if not common.is_dir():
        return ([f"RULE3 {hw_name}: no common/ directory to check"], [], [])
    declared = declared_kernels(hw_dir, "overrides")
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

    # The OMISSION half. `origin - mirror` is what this target does not
    # compile; `[kernels] absent` is what it says it does not compile. They
    # must be the same set. An entry only in the first is the silent-shrink
    # defect (a gb10 kernel that never got a link); an entry only in the
    # second is a declaration that has outlived its reason -- the file came
    # back, or the origin dropped it -- and both are reported.
    origin_dir = hw_dir.parent / MIRRORED_COMMON[hw_name] / "common"
    origin_names = {f.name for f in origin_dir.iterdir()} if origin_dir.is_dir() else set()
    mirrored = {f.name for f in common.iterdir()}
    missing = origin_names - mirrored
    absent = declared_kernels(hw_dir, "absent")
    for undeclared in sorted(missing - absent):
        violations.append(
            f"RULE3 {hw_name}: {MIRRORED_COMMON[hw_name]}/common/{undeclared} has "
            f"no counterpart in kernels/{hw_name}/common and is not listed in "
            f"kernels/{hw_name}/HARDWARE.toml [kernels] absent. A mirror that "
            f"silently carries fewer kernels than its origin is the defect that "
            f"took down the r9700 qwen3.8-27b model build -- link it, or declare "
            f"it absent with the compiler error that justifies the omission."
        )
    for stale in sorted(absent - missing):
        violations.append(
            f"RULE3 {hw_name}: common/{stale} is listed in [kernels] absent but "
            f"is not missing from this mirror (either it is linked here now, or "
            f"{MIRRORED_COMMON[hw_name]}/common no longer has it) -- drop the "
            f"declaration"
        )
    return (violations, reported, sorted(missing))


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
    omissions_by_hw: dict[str, list[str]] = {}
    for hw_name, ext in sorted(HW_SOURCE_EXT.items()):
        hw_dir = kernels_root / hw_name
        violations.extend(check_hw(hw_name, hw_dir, ext))
        if hw_name in MIRRORED_COMMON:
            hw_violations, overrides, omissions = check_common_overrides(hw_name, hw_dir)
            violations.extend(hw_violations)
            overrides_by_hw[hw_name] = overrides
            omissions_by_hw[hw_name] = omissions

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
            print(f"  {hw_name}/common declares no override of {origin}/common")
    # And which entries it does NOT carry. Printed even when the list is empty,
    # because "this mirror is complete" is the claim a reader most wants
    # confirmed and an absent line reads as an unasked question.
    for hw_name in sorted(omissions_by_hw):
        omissions = omissions_by_hw[hw_name]
        origin = MIRRORED_COMMON[hw_name]
        if omissions:
            print(
                f"  {hw_name}/common omits {len(omissions)} {origin}/common "
                f"entr{'y' if len(omissions) == 1 else 'ies'}, each declared in "
                f"kernels/{hw_name}/HARDWARE.toml [kernels] absent:"
            )
            for name in omissions:
                print(f"      {name}")
        else:
            print(f"  {hw_name}/common carries every {origin}/common entry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
