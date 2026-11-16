#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Compile-only portability probe: does a kernel tree build for another arch?

Every `.cu` file the build would compile for one hardware target is compiled
to PTX with the build's own flags (`nvcc --ptx`), and the PTX is then run
through `ptxas` for the arch under test. Nothing is loaded and no GPU is
needed, so the probe answers "would this tree even assemble for sm_XX?" on
any machine that has a CUDA toolkit. It is a necessary condition for a port
(an `mma.sync ... block_scale` instruction or a sm_121-only intrinsic fails
here, loudly, with the file and line), not a sufficient one: it says nothing
about launch bounds, numerics or speed.

    # Which GB10 kernels survive an Ada (sm_89) compile?
    PATH=$HOME/cuda/bin:$PATH scripts/check_arch_portability.py --hw gb10 --arch sm_89

    # Default: the `ada` hardware target if kernels/ada exists, else gb10,
    # always at sm_89.
    scripts/check_arch_portability.py

Flags are read from the three layers `crates/avarok-kernels/build.rs` reads,
least specific first, merged and de-duplicated by the same rule as
`build_flags::merge_extra_flags`:

    kernels/<hw>/HARDWARE.toml           [build] extra_nvcc_flags
    kernels/<hw>/common/KERNEL.toml      [build] extra_nvcc_flags
    kernels/<hw>/<model>/<quant>/KERNEL.toml   [build] extra_nvcc_flags

and the nvcc command line is the one `build_target.rs` builds:
`--ptx -arch=<arch> -O3 <flags> --Werror all-warnings <src> -o <out>`
(plus `AVAROK_EXTRA_NVCC_FLAGS` and `AVAROK_KERNEL_NO_STRICT`, honored the
same way).

Simplifications against the real build, all of them deliberate:

* NVIDIA-vendor hardware only. `hip`, `amd` and `apple` targets use other
  compilers and are refused.
* The arch under test replaces HARDWARE.toml's `arch` (`--arch native` uses
  the declared one, which reproduces the real build's compile line).
* `-std=c++17` is added by the build only on a Windows host; this probe
  targets Linux hosts and never adds it.
* Model targets are enumerated the way the build's `resolve_targets` does
  (models with a MODEL.toml, `[model] kernel_source` redirection, the quant
  directory under the source model, `common/` as the base layer that model
  files shadow by file stem), but MODEL.toml is not parsed beyond
  `kernel_source`, and the build's collision checks are not repeated.
* Compile jobs are de-duplicated by (source path as written, sorted flags),
  exactly as the build does, so a symlinked mirror counts as its own source.
* `ptxas` runs with `-arch=<arch>` only. It catches unknown instructions and
  resource overflows; it does not run `cuobjdump` or inspect the SASS.

Requires Python 3.11+ (tomllib) and `nvcc` and `ptxas` on PATH (or in the
directory given by `--cuda-bin`).

Exit status: 0 when every file passes both stages, 1 when any fails, 2 when
the probe itself cannot run (missing tool, unknown hardware, no sources).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KERNELS = REPO / "kernels"

DEFAULT_ARCH = "sm_89"
DEFAULT_QUANT = "nvfp4"

# Mirrors `KERNEL_STRICT_DIAG_SUPPRESS` in crates/avarok-kernels/build_target.rs.
# That list is empty today ("every kernel warning is fixed at source"); if the
# build grows an entry, add it here so the probe judges the same warnings.
STRICT_DIAG_SUPPRESS: list[int] = []


class ProbeError(Exception):
    """The probe could not run (distinct from a kernel failing to compile)."""


@dataclasses.dataclass(frozen=True)
class Job:
    """One unique compile: a source file and the exact flags it gets."""

    source: Path
    flags: tuple[str, ...]  # sorted, the dedup key (the build sorts them too)
    ordered_flags: tuple[str, ...]  # merge order, what nvcc is actually given
    targets: tuple[str, ...]  # "model/quant" labels that compile it

    @property
    def rel(self) -> str:
        try:
            return str(self.source.relative_to(KERNELS))
        except ValueError:
            return str(self.source)


@dataclasses.dataclass
class Result:
    job: Job
    nvcc_ok: bool = False
    ptxas_ok: bool | None = None  # None: not run because nvcc failed
    error: str = ""


# ── Target enumeration (mirrors build.rs `resolve_targets`) ───────────────────


def load_toml(path: Path) -> dict:
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        raise ProbeError(f"missing {path}") from None
    except tomllib.TOMLDecodeError as e:
        raise ProbeError(f"bad TOML in {path}: {e}") from None


def extra_flags(toml_path: Path, key: str = "extra_nvcc_flags") -> list[str]:
    """`[build] <key>` of a TOML file, or [] when the file or key is absent."""
    if not toml_path.exists():
        return []
    flags = load_toml(toml_path).get("build", {}).get(key, [])
    if not all(isinstance(f, str) for f in flags):
        raise ProbeError(f"{toml_path}: [build] {key} entries must be strings")
    return list(flags)


def merge_flags(*layers: list[str]) -> list[str]:
    """Least specific layer first; a flag repeated in a later layer is dropped.

    Same rule as `build_flags::merge_extra_flags`.
    """
    out: list[str] = []
    for layer in layers:
        for flag in layer:
            if flag not in out:
                out.append(flag)
    return out


def list_subdirs(path: Path) -> list[str]:
    if not path.is_dir():
        return []
    return sorted(p.name for p in path.iterdir() if p.is_dir())


def kernel_source_of(model_dir: Path) -> str | None:
    """`[model] kernel_source` from MODEL.toml, if the model redirects."""
    model_toml = model_dir / "MODEL.toml"
    if not model_toml.exists():
        return None
    src = load_toml(model_toml).get("model", {}).get("kernel_source")
    return src if isinstance(src, str) and src.strip() else None


def cu_files(common: Path | None, model_dir: Path) -> list[Path]:
    """Common files, shadowed by same-stem files in the model dir.

    Same layering as `collect_cu_files` in build.rs.
    """
    by_stem: dict[str, Path] = {}
    for d in (common, model_dir):
        if d is None or not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix == ".cu":
                by_stem[f.stem] = f
    return sorted(by_stem.values())


def enumerate_jobs(
    hw: str, models: str, quant: str
) -> tuple[list[Job], dict]:
    """All unique compile jobs for `hw`, and the hardware's TOML."""
    hw_dir = KERNELS / hw
    if not hw_dir.is_dir():
        known = ", ".join(list_subdirs(KERNELS)) or "(none)"
        raise ProbeError(f"no kernels/{hw}; hardware targets here: {known}")
    hw_toml = load_toml(hw_dir / "HARDWARE.toml")
    vendor = hw_toml.get("hardware", {}).get("vendor", "nvidia")
    if vendor not in ("nvidia", ""):
        raise ProbeError(
            f"kernels/{hw} is vendor '{vendor}'; this probe drives nvcc and "
            "ptxas and only handles NVIDIA hardware targets"
        )
    hw_flags = extra_flags(hw_dir / "HARDWARE.toml")
    common_dir = hw_dir / "common"
    common_flags = extra_flags(common_dir / "KERNEL.toml")

    if models == "*":
        names = [
            d for d in list_subdirs(hw_dir) if (hw_dir / d / "MODEL.toml").exists()
        ]
    else:
        names = [m.strip() for m in models.split(",") if m.strip()]
    if not names:
        raise ProbeError(f"no model targets under kernels/{hw} (looked for MODEL.toml)")

    # key -> (source, ordered flags, targets)
    jobs: dict[tuple[Path, tuple[str, ...]], tuple[Path, tuple[str, ...], list[str]]] = {}
    for model in names:
        model_dir = hw_dir / model
        if not model_dir.is_dir():
            raise ProbeError(f"no model kernels/{hw}/{model}")
        redirect = kernel_source_of(model_dir)
        src_dir = hw_dir / redirect if redirect else model_dir
        if redirect and not (src_dir / "MODEL.toml").exists():
            raise ProbeError(
                f"{model}/MODEL.toml: kernel_source = \"{redirect}\" names no target"
            )
        quants = list_subdirs(src_dir) if quant == "*" else [quant]
        for q in quants:
            model_kernel_dir = src_dir / q
            if not model_kernel_dir.is_dir() and not common_dir.is_dir():
                raise ProbeError(f"no kernel directory for ({model}, {q})")
            model_flags = extra_flags(model_kernel_dir / "KERNEL.toml")
            flags = merge_flags(hw_flags, common_flags, model_flags)
            label = f"{model}/{q}"
            for src in cu_files(common_dir, model_kernel_dir):
                key = (src, tuple(sorted(flags)))
                if key in jobs:
                    jobs[key][2].append(label)
                else:
                    jobs[key] = (src, tuple(flags), [label])

    out = [
        Job(src, key_flags, ordered, tuple(targets))
        for (src, key_flags), (_, ordered, targets) in sorted(
            jobs.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])
        )
    ]
    if not out:
        raise ProbeError(f"no .cu files found for kernels/{hw}")
    return out, hw_toml


# ── Compilation ───────────────────────────────────────────────────────────────


def nvcc_args(job: Job, arch: str, out: Path) -> list[str]:
    """The command line `NvidiaTarget::compile` builds, minus the executable."""
    args = ["--ptx", f"-arch={arch}", "-O3", *job.ordered_flags]
    if os.environ.get("AVAROK_KERNEL_NO_STRICT") != "1":
        args += ["--Werror", "all-warnings"]
        for num in STRICT_DIAG_SUPPRESS:
            args += ["-Xcudafe", f"--diag_suppress={num}"]
    args += os.environ.get("AVAROK_EXTRA_NVCC_FLAGS", "").split()
    args += [str(job.source), "-o", str(out)]
    return args


_ERROR_RE = re.compile(r"\b(error|fatal)\b", re.IGNORECASE)


def first_error_line(text: str) -> str:
    """The first line that looks like a diagnostic, else the first non-blank."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        if _ERROR_RE.search(ln):
            return ln
    return lines[0] if lines else "(no output)"


def run_job(
    job: Job, arch: str, nvcc: str, ptxas: str, workdir: Path, timeout: int
) -> Result:
    res = Result(job)
    ptx = workdir / (re.sub(r"[^A-Za-z0-9_.-]", "_", job.rel) + f".{abs(hash(job.flags)) % 10**8}.ptx")
    try:
        p = subprocess.run(
            [nvcc, *nvcc_args(job, arch, ptx)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        res.error = f"nvcc timed out after {timeout}s"
        return res
    if p.returncode != 0:
        res.error = first_error_line(p.stderr + "\n" + p.stdout)
        return res
    res.nvcc_ok = True
    try:
        q = subprocess.run(
            [ptxas, f"-arch={arch}", str(ptx), "-o", os.devnull],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        res.ptxas_ok = False
        res.error = f"ptxas timed out after {timeout}s"
        return res
    res.ptxas_ok = q.returncode == 0
    if not res.ptxas_ok:
        res.error = first_error_line(q.stderr + "\n" + q.stdout)
    ptx.unlink(missing_ok=True)
    return res


# ── Reporting ─────────────────────────────────────────────────────────────────


def stage(ok: bool | None) -> str:
    return "-" if ok is None else ("PASS" if ok else "FAIL")


def table(results: list[Result]) -> str:
    rows = [("file", "nvcc", "ptxas", "first error")]
    # Flags every job shares are noise; show only what distinguishes a job, so
    # two compiles of one source (e.g. with and without -DHDIM=128) read apart.
    shared = set.intersection(*(set(r.job.flags) for r in results)) if results else set()
    for r in results:
        label = r.job.rel
        extra = [f for f in r.job.ordered_flags if f not in shared]
        if extra:
            label += f" [{' '.join(extra)}]"
        if len(r.job.targets) > 1:
            label += f" (x{len(r.job.targets)})"
        rows.append((label, stage(r.nvcc_ok), stage(r.ptxas_ok), r.error))
    w = [max(len(row[i]) for row in rows) for i in range(3)]
    lines = []
    for i, row in enumerate(rows):
        lines.append(
            f"{row[0]:<{w[0]}}  {row[1]:<{w[1]}}  {row[2]:<{w[2]}}  {row[3]}".rstrip()
        )
        if i == 0:
            lines.append("-" * min(len(lines[0]) + 20, 160))
    return "\n".join(lines)


def summary(results: list[Result], hw: str, arch: str) -> str:
    n = len(results)
    nvcc_fail = sum(not r.nvcc_ok for r in results)
    ptxas_fail = sum(r.nvcc_ok and not r.ptxas_ok for r in results)
    ok = n - nvcc_fail - ptxas_fail
    return (
        f"hw={hw} arch={arch}: {n} unique compiles, {ok} pass, "
        f"{nvcc_fail} fail in nvcc, {ptxas_fail} fail in ptxas"
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compile every .cu a hardware target builds to PTX with nvcc, "
        "assemble it with ptxas for another arch, and print a pass/fail table. "
        "No GPU needed."
    )
    default_hw = "ada" if (KERNELS / "ada").is_dir() else "gb10"
    ap.add_argument(
        "--hw",
        default=default_hw,
        help=f"hardware target under kernels/ (default: {default_hw}; 'ada' if "
        "that tree exists, else 'gb10')",
    )
    ap.add_argument(
        "--arch",
        default=DEFAULT_ARCH,
        help=f"arch to compile for (default: {DEFAULT_ARCH}); 'native' uses the "
        "arch HARDWARE.toml declares",
    )
    ap.add_argument(
        "--model",
        default="*",
        help="comma-separated model targets, or '*' for all (the build's default)",
    )
    ap.add_argument(
        "--quant",
        default=DEFAULT_QUANT,
        help=f"quant directory, or '*' for all (default: {DEFAULT_QUANT}, the "
        "build's default)",
    )
    ap.add_argument(
        "--cuda-bin",
        type=Path,
        help="directory holding nvcc and ptxas (prepended to PATH)",
    )
    ap.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=2,
        help="parallel compiles (default 2; each nvcc is single-threaded)",
    )
    ap.add_argument("--nice", type=int, default=10, help="niceness for children")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per tool run")
    ap.add_argument(
        "--only",
        help="regex; probe only files whose path under kernels/ matches",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="print the compile jobs and their flags, then exit",
    )
    ap.add_argument(
        "--failures-only",
        action="store_true",
        help="print only failing rows in the table",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        jobs, hw_toml = enumerate_jobs(args.hw, args.model, args.quant)
        arch = args.arch
        if arch == "native":
            arch = hw_toml.get("hardware", {}).get("arch", "")
            if not arch:
                raise ProbeError(f"kernels/{args.hw}/HARDWARE.toml declares no arch")
        if args.only:
            rx = re.compile(args.only)
            jobs = [j for j in jobs if rx.search(j.rel)]
            if not jobs:
                raise ProbeError(f"--only {args.only!r} matches no source")
        if args.list:
            for j in jobs:
                print(f"{j.rel}\t{' '.join(j.ordered_flags)}\t{len(j.targets)} target(s)")
            print(f"{len(jobs)} unique compile jobs", file=sys.stderr)
            return 0

        if args.cuda_bin:
            os.environ["PATH"] = f"{args.cuda_bin}{os.pathsep}{os.environ['PATH']}"
        nvcc, ptxas = shutil.which("nvcc"), shutil.which("ptxas")
        if not nvcc or not ptxas:
            raise ProbeError(
                "nvcc and ptxas must be on PATH (or pass --cuda-bin DIR); "
                f"found nvcc={nvcc} ptxas={ptxas}"
            )
    except ProbeError as e:
        print(f"check_arch_portability: {e}", file=sys.stderr)
        return 2

    if args.nice:
        os.nice(args.nice)
    ver = subprocess.run([nvcc, "--version"], capture_output=True, text=True)
    tail = ver.stdout.strip().splitlines()[-1] if ver.stdout.strip() else "?"
    print(f"nvcc: {nvcc} ({tail})")
    print(f"probing kernels/{args.hw} for {arch}: {len(jobs)} unique compiles, -j{args.jobs}")

    results: list[Result] = []
    with tempfile.TemporaryDirectory(prefix="arch-portability-") as tmp:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            futures = [
                pool.submit(run_job, j, arch, nvcc, ptxas, Path(tmp), args.timeout)
                for j in jobs
            ]
            for fut in futures:
                results.append(fut.result())

    shown = [r for r in results if not (args.failures_only and r.nvcc_ok and r.ptxas_ok)]
    print()
    print(table(shown))
    print()
    print(summary(results, args.hw, arch))
    return 0 if all(r.nvcc_ok and r.ptxas_ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
