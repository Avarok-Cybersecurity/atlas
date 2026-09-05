#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Does every Atlas kernel compile for a given SM architecture? Answered with
# nvcc alone — no H100, no B200, no GPU of any kind.
#
# Named for the campaign it was written for; it is not Hopper-specific and
# `--hw` takes any set under `kernels/`. (The name is left alone because CI and
# docs reference it, and a rename buys nothing a comment does not.)
#
# `nvcc --ptx` is a cross-compile: it emits device assembly for `-arch=sm_90a`
# on a host with no NVIDIA hardware present. `ptxas -v` then assembles that PTX
# for the same arch and reports register and spill pressure. Between them they
# answer "can this kernel exist on this architecture", which is the first
# question of a new hardware target and the only one that can be answered
# before the hardware arrives. They do NOT answer whether it is correct or
# fast; that needs the device.
#
# Usage:
#   scripts/hopper_ptx_gate.sh [--hw hopper] [--model NAME|all] [--jobs N]
#                              [--arch sm_90a] [--nvcc PATH]
#                              [--out ledger.json] [--selftest]
#
#   --hw       hardware set under kernels/ (default: hopper)
#   --model    one model directory, or `all` for every model in the set
#   --jobs     parallel compiles (default: 4)
#   --arch     override the arch; default is HARDWARE.toml's `arch`
#   --nvcc     path to nvcc; else $NVCC_BIN, $CUDA_HOME/bin/nvcc, PATH,
#              /usr/local/cuda/bin/nvcc
#   --out      ledger path (default: hopper_ptx_gate.json). A markdown summary
#              is written alongside it with the same basename and `.md`.
#   --strict   add `--Werror all-warnings`, matching what build.rs does. Use it
#              to predict whether the real kernel build would go green; leave it
#              off to ask only whether the instructions exist on this arch.
#   --selftest run ONLY the self-test and exit
#
# The self-test runs first, always. It compiles two fixtures under
# scripts/fixtures/hopper_gate/ — one that must pass for any arch, one that
# must fail for this one — and if either verdict is wrong the gate refuses to
# report results at all. A gate whose failure path has never executed is not
# evidence.
#
# The NEGATIVE fixture is chosen per arch, because no one instruction is absent
# from every architecture Atlas targets. `redux.sync.max.abs.f32` is absent
# everywhere EXCEPT sm_100a; the warp-level `mma ... kind::mxf4nvf4
# .block_scale` is absent on sm_90a and sm_100a and present on sm_120/sm_121.
# An arch with no registered negative fixture is REFUSED, not waved through:
# running the gate without a working failure path is the thing this section
# exists to prevent.
#
# Exit status: 0 only if the self-test held AND every kernel compiled.
#
# `--Werror all-warnings`, which `build.rs` always adds ("clippy for kernels"),
# is OFF by default here and available as `--strict`. The default answers the
# narrower question — does the instruction selection exist on this
# architecture — where a promoted warning would make an unrelated diagnostic
# look like an ISA gap. `--strict` answers the other one: would the real build
# go green.

set -euo pipefail

# ── Internal parallel worker (re-entrant; see the xargs call below) ──
if [ "${1:-}" = "--compile-one" ]; then
  IFS=$'\t' read -r key src incdirs flags <<<"$2"
  work="$3"
  nvcc="$4"
  ptxas="$5"
  arch="$6"
  ptx="$work/ptx/$key.ptx"
  strict=()
  if [ "${ATLAS_GATE_STRICT:-0}" = 1 ]; then
    strict=(--Werror all-warnings)
  fi
  inc=()
  for d in $incdirs; do inc+=("-I$d"); done
  # `$flags` is deliberately word-split: it is a flag list, not one argument.
  # shellcheck disable=SC2086
  if nvcc_err=$("$nvcc" --ptx "-arch=$arch" -O3 $flags "${strict[@]}" "${inc[@]}" -o "$ptx" "$src" 2>&1); then
    ptx_ok=1
  else
    ptx_ok=0
  fi
  ptxas_ok=0
  ptxas_out=""
  if [ "$ptx_ok" = 1 ]; then
    if ptxas_out=$("$ptxas" "-arch=$arch" -v -o /dev/null "$ptx" 2>&1); then
      ptxas_ok=1
    fi
  fi
  rm -f "$ptx"
  printf '%s\n' "$nvcc_err" >"$work/err/$key.txt"
  printf '%s\n' "$ptxas_out" >"$work/vrb/$key.txt"
  printf '%s\t%s\t%s\n' "$key" "$ptx_ok" "$ptxas_ok" >"$work/res/$key.tsv"
  exit 0
fi

HW=hopper
MODEL=all
JOBS=4
ARCH=""
NVCC=""
OUT="hopper_ptx_gate.json"
SELFTEST_ONLY=0
STRICT=0

while [ $# -gt 0 ]; do
  case "$1" in
    --hw) HW="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --arch) ARCH="$2"; shift 2 ;;
    --nvcc) NVCC="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --strict) STRICT=1; shift ;;
    --selftest) SELFTEST_ONLY=1; shift ;;
    -h|--help) sed -n '4,54p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
FIXTURES="$SCRIPT_DIR/fixtures/hopper_gate"
HW_DIR="$ROOT/kernels/$HW"

[ -d "$HW_DIR" ] || { echo "no such hardware set: $HW_DIR" >&2; exit 2; }

# ── Toolchain ──
# CUDA is routinely installed without being on PATH (it is not on PATH on the
# DGX Spark boxes), so PATH is the third place looked, not the first.
if [ -z "$NVCC" ]; then
  if [ -n "${NVCC_BIN:-}" ]; then NVCC="$NVCC_BIN"
  elif [ -n "${CUDA_HOME:-}" ] && [ -x "$CUDA_HOME/bin/nvcc" ]; then NVCC="$CUDA_HOME/bin/nvcc"
  elif command -v nvcc >/dev/null 2>&1; then NVCC="$(command -v nvcc)"
  elif [ -x /usr/local/cuda/bin/nvcc ]; then NVCC=/usr/local/cuda/bin/nvcc
  else echo "nvcc not found — pass --nvcc PATH or set CUDA_HOME" >&2; exit 2; fi
fi
[ -x "$NVCC" ] || { echo "not executable: $NVCC" >&2; exit 2; }
PTXAS="$(dirname "$NVCC")/ptxas"
[ -x "$PTXAS" ] || { echo "ptxas not found next to nvcc: $PTXAS" >&2; exit 2; }

if [ -z "$ARCH" ]; then
  ARCH="$(sed -n 's/^[[:space:]]*arch[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' \
            "$HW_DIR/HARDWARE.toml" | head -1)"
fi
[ -n "$ARCH" ] || { echo "no arch in $HW_DIR/HARDWARE.toml and no --arch" >&2; exit 2; }

NVCC_VERSION="$("$NVCC" --version | tail -1 | tr -s ' ')"

# ── Which negative fixture is valid for THIS arch ──
# See the fixtures' own measured tables. Neither Blackwell architecture is a
# superset of the other, so the two fixtures are mirror images and the choice
# cannot be avoided by picking a "more portable" instruction.
case "$ARCH" in
  sm_100*)          BAD_FIXTURE=known_bad_post_blackwell_dc.cu ;;
  sm_9*|sm_12*)     BAD_FIXTURE=known_bad_post_hopper.cu ;;
  *)                BAD_FIXTURE="" ;;
esac
if [ -z "$BAD_FIXTURE" ]; then
  echo "no negative self-test fixture is registered for $ARCH." >&2
  echo "  A gate with no failure path proves nothing, so this refuses to run." >&2
  echo "  Add a fixture under $FIXTURES that ptxas rejects for $ARCH and" >&2
  echo "  accepts elsewhere, record its measured table in the file, and add" >&2
  echo "  the arm to the case statement in $(basename "${BASH_SOURCE[0]}")." >&2
  exit 2
fi
[ -f "$FIXTURES/$BAD_FIXTURE" ] || {
  echo "negative fixture missing: $FIXTURES/$BAD_FIXTURE" >&2; exit 2; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/ptx" "$WORK/err" "$WORK/vrb" "$WORK/res"

# ── Self-test ──
# Both fixtures go through the same two-stage pipeline the real kernels do.
selftest() {
  local good bad
  if "$NVCC" --ptx "-arch=$ARCH" -O3 -o "$WORK/good.ptx" "$FIXTURES/known_good.cu" >/dev/null 2>&1 \
     && "$PTXAS" "-arch=$ARCH" -o /dev/null "$WORK/good.ptx" >/dev/null 2>&1; then
    good=true
  else
    good=false
  fi
  if "$NVCC" --ptx "-arch=$ARCH" -O3 -o "$WORK/bad.ptx" "$FIXTURES/$BAD_FIXTURE" >/dev/null 2>&1 \
     && "$PTXAS" "-arch=$ARCH" -o /dev/null "$WORK/bad.ptx" >/dev/null 2>&1; then
    bad=false   # it compiled; the negative fixture did NOT fail
  else
    bad=true
  fi
  printf '%s %s\n' "$good" "$bad"
}

read -r SELF_GOOD SELF_BAD <<<"$(selftest)"
echo "self-test @ $ARCH: known_good passed=$SELF_GOOD  $BAD_FIXTURE failed=$SELF_BAD"
SELF_OK=0
[ "$SELF_GOOD" = true ] && [ "$SELF_BAD" = true ] && SELF_OK=1

if [ "$SELF_OK" != 1 ]; then
  echo "SELF-TEST FAILED — refusing to report kernel results." >&2
  if [ "$SELF_BAD" != true ]; then
    echo "  $BAD_FIXTURE COMPILED for $ARCH, so the instruction it relies on is" >&2
    echo "  present here (see the fixture's own measured table). The gate has no" >&2
    echo "  working failure path at this arch: the FIXTURE is what needs fixing," >&2
    echo "  not the kernels." >&2
  fi
  if [ "$SELF_GOOD" != true ]; then
    echo "  known_good.cu FAILED for $ARCH — the toolchain or the arch string is wrong." >&2
    "$NVCC" --ptx "-arch=$ARCH" -O3 -o /dev/null "$FIXTURES/known_good.cu" || true
  fi
  exit 1
fi
if [ "$SELFTEST_ONLY" = 1 ]; then
  echo "self-test only: OK"
  exit 0
fi

# ── Which models ──
if [ "$MODEL" = all ]; then
  MODELS=()
  for d in "$HW_DIR"/*/; do
    [ -f "$d/MODEL.toml" ] || continue
    MODELS+=("$(basename "$d")")
  done
else
  MODELS=("$MODEL")
fi
[ "${#MODELS[@]}" -gt 0 ] || { echo "no model directories under $HW_DIR" >&2; exit 2; }

# ── Task list ──
# Mirrors build.rs: the per-quant file set is common/ overridden by the model
# dir BY FILE STEM, and the flag list has THREE layers, merged
# least-specific-first and deduped — HARDWARE.toml's [build] extra_nvcc_flags,
# then common/KERNEL.toml's, then the model quant dir's
# (build_flags.rs `merge_extra_flags` + build_parse.rs `parse_kernel_toml`).
#
# The hardware layer is what makes this gate answer the question it claims to.
# kernels/hopper and kernels/b200 define -DATLAS_NO_WARP_BLOCKSCALE_MMA there,
# compiling out a W4A4 region neither ISA can assemble; a gate that ignored
# that layer would keep reporting a failure the real build does not have.
#
# Identical (source, flags) pairs are compiled ONCE. Several gb10 models share
# one physical kernel through symlinks — deepseek-v4-flash's w4a16_gemm.cu is
# three models' — and build.rs deduplicates the same way.
python3 - "$HW_DIR" "$WORK" "${MODELS[@]}" <<'PY'
import hashlib, os, sys, tomllib
hw_dir, work = sys.argv[1], sys.argv[2]
models = sys.argv[3:]
common = os.path.join(hw_dir, "common")

def build_flags(path):
    """[build] extra_nvcc_flags from one TOML file; [] if it has none."""
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        t = tomllib.load(f)
    return list(t.get("build", {}).get("extra_nvcc_flags", []))

def flags(d):
    return build_flags(os.path.join(d, "KERNEL.toml"))

def cu(d):
    if not os.path.isdir(d):
        return {}
    return {f[:-3]: os.path.join(d, f) for f in os.listdir(d) if f.endswith(".cu")}

hw_flags = build_flags(os.path.join(hw_dir, "HARDWARE.toml"))
base_flags = flags(common)
tasks, mapping = {}, []
for model in models:
    mdir = os.path.join(hw_dir, model, "nvfp4")
    merged = []
    for f in hw_flags + base_flags + flags(mdir):
        if f not in merged:
            merged.append(f)
    files = cu(common)
    files.update(cu(mdir))
    incdirs = f"{mdir} {common}"
    for stem, src in sorted(files.items()):
        real = os.path.realpath(src)
        key = hashlib.sha256(
            (real + "\0" + "\0".join(merged) + "\0" + incdirs).encode()
        ).hexdigest()[:16]
        tasks[key] = (key, src, incdirs, " ".join(merged))
        mapping.append((model, stem, key, os.path.relpath(real, os.path.dirname(hw_dir))))

with open(os.path.join(work, "tasks.tsv"), "w") as f:
    for t in tasks.values():
        f.write("\t".join(t) + "\n")
with open(os.path.join(work, "map.tsv"), "w") as f:
    for m in mapping:
        f.write("\t".join(m) + "\n")
# The hardware flag layer, for the ledger: a receipt that does not say the
# compile line cannot be checked against the build it claims to predict.
with open(os.path.join(work, "hw_flags.txt"), "w") as f:
    f.write("\n".join(hw_flags))
print(f"{len(mapping)} kernel(s) across {len(models)} model(s); "
      f"{len(tasks)} unique compile(s) after dedup"
      + (f"; hardware flags: {' '.join(hw_flags)}" if hw_flags else ""))
PY

# ── Compile ──
echo "compiling for $ARCH with $JOBS job(s), strict=$STRICT — $NVCC_VERSION"
ATLAS_GATE_STRICT="$STRICT" \
xargs -a "$WORK/tasks.tsv" -d '\n' -P "$JOBS" -I LINE \
  bash "${BASH_SOURCE[0]}" --compile-one LINE "$WORK" "$NVCC" "$PTXAS" "$ARCH"

# ── Ledger ──
python3 - "$WORK" "$OUT" "$ARCH" "$HW" "$NVCC_VERSION" "$SELF_GOOD" "$SELF_BAD" "$STRICT" "$BAD_FIXTURE" <<'PY'
import json, os, re, socket, sys, datetime
work, out, arch, hw, nvcc_version, self_good, self_bad, strict, bad_fixture = sys.argv[1:10]

res = {}
for name in os.listdir(os.path.join(work, "res")):
    key, ptx_ok, ptxas_ok = open(os.path.join(work, "res", name)).read().split()
    res[key] = (ptx_ok == "1", ptxas_ok == "1")

def head(path):
    if not os.path.exists(path):
        return ""
    lines = [l.rstrip() for l in open(path, errors="replace") if l.strip()]
    for l in lines:
        if "error" in l.lower():
            return l[:400]
    return lines[0][:400] if lines else ""

# ptxas -v reports per entry function; the target-wide numbers that matter are
# the worst ones — the kernel that spills is the kernel that is slow.
REG = re.compile(r"Used (\d+) registers")
SPILL = re.compile(r"(\d+) bytes spill stores")

def pressure(path):
    if not os.path.exists(path):
        return None, None
    text = open(path, errors="replace").read()
    regs = [int(m) for m in REG.findall(text)]
    spills = [int(m) for m in SPILL.findall(text)]
    return (max(regs) if regs else None, max(spills) if spills else None)

results = []
for line in open(os.path.join(work, "map.tsv")):
    model, stem, key, src = line.rstrip("\n").split("\t")
    ptx_ok, ptxas_ok = res.get(key, (False, False))
    regs, spill = pressure(os.path.join(work, "vrb", key + ".txt"))
    results.append({
        "file": src, "stem": stem, "model": model,
        "ptx_ok": ptx_ok, "ptxas_ok": ptxas_ok,
        "registers_max": regs, "spill_bytes": spill,
        "error_head": "" if (ptx_ok and ptxas_ok)
                      else head(os.path.join(work, "err", key + ".txt"))
                           or head(os.path.join(work, "vrb", key + ".txt")),
    })
results.sort(key=lambda r: (r["model"], r["stem"]))

by_model = {}
for r in results:
    b = by_model.setdefault(r["model"], {"total": 0, "pass": 0, "fail": 0})
    b["total"] += 1
    b["pass" if (r["ptx_ok"] and r["ptxas_ok"]) else "fail"] += 1
failures = [r for r in results if not (r["ptx_ok"] and r["ptxas_ok"])]

hw_flags_path = os.path.join(work, "hw_flags.txt")
hw_flags = [l for l in open(hw_flags_path).read().split("\n") if l] \
    if os.path.exists(hw_flags_path) else []

ledger = {
    "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "host": socket.gethostname(),
    "hw": hw, "arch": arch, "nvcc": nvcc_version,
    # HARDWARE.toml [build] extra_nvcc_flags — the layer under common/ and the
    # model's. On hopper/b200 this is what compiles out the W4A4 region.
    "hardware_flags": hw_flags,
    "strict": strict == "1",
    "selftest": {
        "bad_failed": self_bad == "true",
        "good_passed": self_good == "true",
        # WHICH negative fixture. Per-arch, so a receipt that does not name it
        # cannot be checked against the fixture's measured table.
        "bad_fixture": bad_fixture,
    },
    "summary": {
        "total": len(results),
        "pass": len(results) - len(failures),
        "fail": len(failures),
        "by_model": by_model,
    },
    "results": results,
}
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
with open(out, "w") as f:
    json.dump(ledger, f, indent=2)
    f.write("\n")

md = os.path.splitext(out)[0] + ".md"
L = []
L.append(f"# Atlas PTX gate — `{hw}` @ `{arch}`\n")
L.append(f"* generated: {ledger['generated_utc']} on `{ledger['host']}`")
L.append(f"* toolchain: {nvcc_version}")
L.append(f"* strict (`--Werror all-warnings`, as build.rs): {ledger['strict']}")
if hw_flags:
    L.append("* HARDWARE.toml `[build] extra_nvcc_flags`: `"
             + " ".join(hw_flags) + "`")
L.append(f"* self-test: known_good passed={ledger['selftest']['good_passed']}, "
         f"`{bad_fixture}` failed={ledger['selftest']['bad_failed']}")
L.append(f"* **{ledger['summary']['pass']}/{ledger['summary']['total']} kernels compiled**"
         f" ({ledger['summary']['fail']} failed)\n")
L.append("| model | kernels | pass | fail |")
L.append("|---|---:|---:|---:|")
for m, b in sorted(by_model.items()):
    L.append(f"| {m} | {b['total']} | {b['pass']} | {b['fail']} |")
L.append("")
if failures:
    L.append("## Failures\n")
    L.append("| model | kernel | stage | first error |")
    L.append("|---|---|---|---|")
    for r in failures:
        stage = "nvcc --ptx" if not r["ptx_ok"] else "ptxas"
        err = r["error_head"].replace("|", "\\|")
        L.append(f"| {r['model']} | `{r['stem']}` | {stage} | `{err}` |")
    L.append("")
else:
    L.append("No failures: every kernel in this hardware set emitted PTX and "
             "assembled for the target architecture.\n")
top = sorted((r for r in results if r["registers_max"]),
             key=lambda r: -r["registers_max"])[:10]
if top:
    L.append("## Highest register pressure\n")
    L.append("| model | kernel | max registers | spill bytes |")
    L.append("|---|---|---:|---:|")
    for r in top:
        L.append(f"| {r['model']} | `{r['stem']}` | {r['registers_max']} | "
                 f"{r['spill_bytes'] if r['spill_bytes'] is not None else '-'} |")
    L.append("")
L.append(f"Compilation is not correctness. Nothing here has run on {hw} "
         "silicon; these kernels are known to EXIST for the architecture, not "
         "to produce the right numbers or to be fast.\n")
open(md, "w").write("\n".join(L))

print(f"{ledger['summary']['pass']}/{ledger['summary']['total']} kernels compiled "
      f"for {arch}; {ledger['summary']['fail']} failed")
print(f"ledger:  {out}")
print(f"summary: {md}")
sys.exit(1 if failures else 0)
PY
