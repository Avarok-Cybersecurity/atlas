#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Cross-hardware kernel interference (CHKI): who else compiles the bytes you changed?

`kernels/<hw>/` LOOKS like one tree per hardware. It is not. strix is 7 real files and
105 symlinks, 97 of them into `kernels/gb10/common/`; strix-hip is 22 real and 87
symlinks. So an edit to a gb10 file silently changes what AMD compiles, and the only
signal has ever been a compile failure:

  d584c0c50  `__syncwarp()` added to gb10/common/w4a16_gemv.cu, which is symlinked into
             strix and strix-hip; hipcc rejects it. Caught ONLY because the windows-hip
             release leg failed. A performance-only effect would have been invisible.
  17fe989ec  A Strix workaround left inside gb10 common device code invalidated 27
             targets' benchmark records and forced a full GPU re-measurement.

`gate/taxon.rs::hardware_of()` cannot see any of this — it is a path prefix. This script
resolves the three real sharing mechanisms (symlinks, `common/` fan-out, `[model]
kernel_source` redirects, plus symlinked KERNEL.toml/MODEL.toml) and says which hardware
actually consumes each changed path.

WHY IT IS FAIL-CLOSED, unlike classify-diff.sh: that script decides what to SKIP and must
fail open. This one decides a VERDICT. "Cannot see the inputs" (rc 2) is as red as a
violation (rc 1) — the same doctrine as campaign-guard.sh's exit 2.

WHAT IT CANNOT DECIDE, and hands to the oracle
(.claude/skills/oracle_pre_commit_cross_hardware_check): whether a hunk sits inside a
`#if defined(__SCALE__)` arm the other hardware never compiles; whether a `cfg!(atlas_scale)`
host site pairs with a device constant you moved; and whether a textually-shared change is
performance-relevant to a hardware with no benchmark record and no CI box. Reach is provable;
harm is not.
"""
import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path

HW_SOURCE_EXT = {"gb10": ".cu", "metal": ".metal", "strix": ".cu", "strix-hip": ".cu"}
CONFIG_NAMES = {"HARDWARE.toml", "KERNEL.toml", "MODEL.toml"}
AMD_TOKENS = ("__SCALE__", "__HIPCC__", "__HIP_DEVICE_COMPILE__", "__HIP_PLATFORM")


def git(*a, cwd=None, check=True):
    r = subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(a)}: {r.stderr.strip()}")
    return r.stdout


def hardware_of(path):
    p = Path(path).parts
    return p[1] if len(p) > 1 and p[0] == "kernels" else None


def kernel_source(model_dir):
    """`[model] kernel_source = "other"` redirects where a model's quant dirs are read from."""
    mt = model_dir / "MODEL.toml"
    if not mt.exists():
        return None
    for line in mt.read_text(errors="replace").splitlines():
        s = line.split("#", 1)[0].strip()
        if s.startswith("kernel_source"):
            v = s.split("=", 1)[1].strip().strip('"').strip("'")
            return v or None
    return None


def quoted_includes(path):
    """Quoted #includes, resolved LEXICALLY against the file's own directory.

    Lexical, not canonical, because nvcc is handed the SYMLINK path (build.rs passes the
    read_dir entry and sets no -I), so `#include "x.cuh"` inside a symlinked .cu resolves
    against the LINKING hardware's dir. That is how strix-hip's real prefill_paged_compute.cuh
    (BR64 32) shadows gb10's (BR64 64) even though the .cu including it is a symlink.
    """
    out = []
    try:
        txt = Path(path).read_text(errors="replace")
    except OSError:
        return out
    for line in txt.splitlines():
        s = line.strip()
        if s.startswith("//") or not s.startswith("#include"):
            continue
        rest = s[len("#include"):].strip()
        if not rest.startswith('"'):
            continue
        end = rest.find('"', 1)
        if end > 1:
            out.append(rest[1:end])
    return out


def build_index(root):
    """realpath -> set of (hw, model, quant) targets that consume it."""
    kroot = root / "kernels"
    missing = [h for h in HW_SOURCE_EXT if not (kroot / h).is_dir()]
    if missing:
        raise RuntimeError(f"hardware tree(s) absent from {kroot}: {', '.join(missing)} "
                           f"— refusing to scan a tree I cannot index")
    consumers, targets = {}, []
    for hw, ext in HW_SOURCE_EXT.items():
        hwdir = kroot / hw
        if not (hwdir / "HARDWARE.toml").exists():
            continue
        common = hwdir / "common"
        for model_dir in sorted(p for p in hwdir.iterdir() if p.is_dir() and p.name != "common"):
            if not (model_dir / "MODEL.toml").exists():
                continue
            src_dir = model_dir
            ks = kernel_source(model_dir)
            if ks:
                cand = hwdir / ks
                if not (cand / "MODEL.toml").exists():
                    # unresolved redirect: fail closed, this model reaches everything on hw
                    src_dir = hwdir
                else:
                    src_dir = cand
            for quant_dir in sorted(p for p in src_dir.iterdir() if p.is_dir()):
                t = (hw, model_dir.name, quant_dir.name)
                targets.append(t)
                # stem-keyed merge: common first, the model's quant dir SHADOWS it
                merged = {}
                for d in (common, quant_dir):
                    if d.is_dir():
                        for f in sorted(d.glob(f"*{ext}")):
                            merged[f.stem] = f
                seen = set()
                for f in list(merged.values()):
                    stack = [f]
                    while stack:
                        cur = stack.pop()
                        rp = os.path.realpath(cur)
                        if rp in seen:
                            continue
                        seen.add(rp)
                        consumers.setdefault(rp, set()).add(t)
                        for inc in quoted_includes(cur):
                            cand = Path(cur).parent / inc
                            if cand.exists():
                                stack.append(cand)
                for cfg in (hwdir / "HARDWARE.toml", common / "KERNEL.toml",
                            model_dir / "MODEL.toml", quant_dir / "KERNEL.toml"):
                    if cfg.exists():
                        consumers.setdefault(os.path.realpath(cfg), set()).add(t)
    return consumers, targets


def changed_paths(root, base, head, three_dot, worktree):
    if worktree:
        spec = [base]
    else:
        spec = [f"{base}...{head}"] if three_dot else [f"{base}..{head}"] if head else [base]
    out = git("diff", "--name-status", "--no-renames", *spec, "--", "kernels/", cwd=root)
    rows = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            rows.append((parts[0][0], parts[-1]))
    return rows


def symlink_via(root, realpath_target):
    """Every symlink under kernels/ that resolves to this file, for the operator."""
    via = []
    kroot = root / "kernels"
    for dirpath, dirnames, filenames in os.walk(kroot):
        for n in list(dirnames) + filenames:
            p = Path(dirpath) / n
            if p.is_symlink() and os.path.realpath(p) == realpath_target:
                via.append((str(p.relative_to(root)), os.readlink(p)))
    return via


def trailers(root, base, head):
    """`Hardware:` and `CHKI-Verdict:` over BASE..HEAD, plus PR_BODY if present."""
    hw, verdict = set(), ""
    try:
        body = git("log", "--format=%B", f"{base}..{head or 'HEAD'}", cwd=root, check=False)
    except Exception:
        body = ""
    body += "\n" + os.environ.get("PR_BODY", "")
    for line in body.splitlines():
        s = line.strip()
        if s.lower().startswith("hardware:"):
            hw |= {t.strip() for t in s.split(":", 1)[1].split(",") if t.strip()}
        elif s.lower().startswith("chki-verdict:"):
            verdict = s.split(":", 1)[1].strip()
    return hw, verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--base")
    ap.add_argument("--head")
    ap.add_argument("--merge-base", action="store_true", help="three-dot diff (pull_request)")
    ap.add_argument("--worktree", action="store_true", help="include uncommitted changes")
    ap.add_argument("--paths", nargs="*", help="attribution only; never fails on R1")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-selftest", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    try:
        if not args.no_selftest:
            selftest()
        consumers, targets = build_index(root)
    except Exception as e:
        print(f"cross-hardware: CANNOT SEE INPUTS — {e}", file=sys.stderr)
        return 2

    attribution_only = args.paths is not None
    if attribution_only:
        rows = [("M", p) for p in args.paths]
    else:
        if not args.base:
            print("cross-hardware: no base given", file=sys.stderr)
            return 2
        try:
            rows = changed_paths(root, args.base, args.head, args.merge_base, args.worktree)
        except Exception as e:
            print(f"cross-hardware: CANNOT SEE INPUTS — {e}", file=sys.stderr)
            return 2

    declared, verdict_trailer = (set(), "") if attribution_only else trailers(root, args.base, args.head)
    results, structural = [], []

    for status, rel in rows:
        owner = hardware_of(rel)
        if owner is None:
            continue
        ap_ = root / rel
        rp = os.path.realpath(ap_) if ap_.exists() else str(ap_)
        reach = consumers.get(rp, set())
        reach_hw = sorted({t[0] for t in reach})
        leak = sorted(set(reach_hw) - {owner})

        # S1: a NEW symlink that crosses hardware is never allowed.
        if status in ("A", "T") and ap_.is_symlink():
            tgt_hw = hardware_of(os.path.relpath(os.path.realpath(ap_), root))
            if tgt_hw and tgt_hw != owner:
                structural.append(("S1", rel, f"new cross-hardware symlink -> {tgt_hw}"))

        if attribution_only:
            v = "ATTRIBUTION"
        elif not leak:
            v = "OK"
        elif not declared:
            v = f"CHKI (undeclared: {', '.join(leak)})"
        elif not (set(reach_hw) | {owner}) <= declared:
            miss = sorted((set(reach_hw) | {owner}) - declared)
            v = f"CHKI (unnamed: {', '.join(miss)})"
        else:
            v = f"DECLARED ({', '.join(leak)})"
        results.append({"path": rel, "status": status, "owner": owner,
                        "reach_hw": reach_hw, "targets": len(reach),
                        "via": [f"{a} -> {b}" for a, b in symlink_via(root, rp)] if leak else [],
                        "verdict": v})

    # S4: dangling symlinks anywhere under kernels/
    for dirpath, dirnames, filenames in os.walk(root / "kernels"):
        for n in list(dirnames) + filenames:
            p = Path(dirpath) / n
            if p.is_symlink() and not p.exists():
                structural.append(("S4", str(p.relative_to(root)), f"dangling -> {os.readlink(p)}"))

    bad = [r for r in results if r["verdict"].startswith("CHKI")]
    declared_rows = [r for r in results if r["verdict"].startswith("DECLARED")]
    needs_verdict = bool(declared_rows) and not verdict_trailer

    if args.json:
        print(json.dumps({"base": args.base, "head": args.head,
                          "trailers": {"hardware": sorted(declared), "verdict": verdict_trailer},
                          "paths": results,
                          "structural": [{"rule": a, "path": b, "detail": c} for a, b, c in structural],
                          "verdict": "OK" if not (bad or structural or needs_verdict) else "CHKI"},
                         indent=2))
    else:
        print(f"cross-hardware reach: {len(results)} changed kernel path(s)")
        for r in results:
            print(f"  {r['path']:<58} {r['owner']:<10} {','.join(r['reach_hw']) or '-':<22} "
                  f"{r['targets']:>4}  {r['verdict']}")
            for v in r["via"]:
                print(f"      via: {v}")
        for a, b, c in structural:
            print(f"  {a} {b}: {c}")
        print(f"Hardware trailer: {', '.join(sorted(declared)) or '(none)'}    "
              f"CHKI-Verdict: {verdict_trailer or '(none)'}")
        if attribution_only:
            print("verdict: ATTRIBUTION (no pass/fail in --paths mode)")
        elif bad or structural or needs_verdict:
            print("verdict: CHKI")
            print("next: /oracle_pre_commit_cross_hardware_check, then ONE of")
            print("  (a) Hardware: <every reached hw>  +  CHKI-Verdict: benign -- <proof>")
            print("  (b) parameterize in kernels/<hw>/HARDWARE.toml AND add the reader in")
            print("      crates/atlas-kernels/build.rs in the SAME change (only `vendor` and")
            print("      `arch` are read today; a key with no reader is decoration)")
            print("  (c) separate kernel in the intended tree, pinned to base bytes, NO symlink")
        else:
            print("verdict: OK")

    if attribution_only:
        return 0
    return 1 if (bad or structural or needs_verdict) else 0


def selftest():
    """RED/GREEN fixtures. A rule that cannot fire is a rule that is not a check."""
    with tempfile.TemporaryDirectory() as td:
        r = Path(td); k = r / "kernels"
        for hw, vendor in (("gb10", "nvidia"), ("strix", "amd")):
            (k / hw / "common").mkdir(parents=True)
            (k / hw / "HARDWARE.toml").write_text(f'[hardware]\nname="{hw}"\nvendor="{vendor}"\narch="x"\n')
            (k / hw / "m1" / "q").mkdir(parents=True)
            (k / hw / "m1" / "MODEL.toml").write_text("[model]\n")
        (k / "gb10" / "common" / "shared.cu").write_text("// shared\n")
        (k / "strix" / "common" / "shared.cu").symlink_to("../../gb10/common/shared.cu")
        (k / "gb10" / "common" / "own.cu").write_text("// gb10 only\n")
        for hw in ("metal", "strix-hip"):
            (k / hw / "common").mkdir(parents=True)
            (k / hw / "HARDWARE.toml").write_text(f'[hardware]\nname="{hw}"\nvendor="x"\narch="y"\n')
        cons, _ = build_index(r)
        shared_rp = os.path.realpath(k / "gb10" / "common" / "shared.cu")
        hw_reach = {t[0] for t in cons.get(shared_rp, set())}
        assert hw_reach == {"gb10", "strix"}, f"RED fixture: symlink reach was {hw_reach}"
        own_rp = os.path.realpath(k / "gb10" / "common" / "own.cu")
        assert {t[0] for t in cons.get(own_rp, set())} == {"gb10"}, "GREEN fixture: own.cu leaked"


if __name__ == "__main__":
    sys.exit(main())
