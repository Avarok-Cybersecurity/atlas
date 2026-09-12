#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Tests for scripts/hopper_ptx_gate.sh. No GPU anywhere: cases (a)-(d) go
# through `--list-tasks`, which resolves the source set and prints it without
# compiling anything, and case (e) cross-compiles one fixture with nvcc/ptxas
# on the host CPU. (e) is SKIPPED, not failed, where no nvcc+ptxas pair is
# installed.
#
# What each case is actually defending:
#
#   (a) `[model] kernel_source` is followed the way the real build follows it.
#       kernels/gb10/qwen3.8-27b/MODEL.toml redirects to qwen3.6-27b, so the
#       compiled set is common/ overridden BY FILE STEM from
#       kernels/gb10/qwen3.6-27b/nvfp4/ -- 181 files, of which 14 are the
#       redirect target's and 4 of those shadow a common kernel (w4a16_gemm,
#       moe_w4a16_grouped_gemm, gated_delta_rule,
#       inferspark_prefill_paged_indirect). A gate that ignored the redirect
#       selected 171 common tasks and reported a green receipt for sources the
#       build never compiles. The oracle is crates/atlas-kernels/build.rs
#       (`kernel_src_dir`, ~line 1065) and build_parse.rs::parse_kernel_source.
#   (b) the redirect is an ALIAS, not a variant: qwen3.8-27b and qwen3.6-27b
#       must select the same source paths. If they ever diverge, one of the two
#       receipts is describing a build that does not happen.
#   (c) qwen3.8-27b's own directory is never a source. It ships MODEL.toml and
#       BENCH.toml and no .cu tree at all; a task pointing into it would be a
#       file that does not exist.
#   (d) a nonexistent model is REFUSED (exit 2), naming what it looked for.
#       Before, an unknown name simply produced the common kernels and
#       attributed them to it -- a receipt for a model that is not in the tree.
#   (e) the ledger records EVERY ptxas error for a file, not just the first.
#       ptxas emits one `error   :` line per rejected entry function, so a file
#       with several rejected entries used to appear in the Failures table as
#       one -- 22 files standing in for 42 rejected entries. The fixture
#       known_bad_two_entries.cu has two oversized-shared entries and one valid
#       one, so `error_count`, `len(errors)` and `summary.rejected_entries` all
#       have a known answer: 2.
#   (f) shellcheck on the gate and on this file.
#
# Usage: bash scripts/hopper_ptx_gate_test.sh
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
GATE="$HERE/hopper_ptx_gate.sh"
ROOT="$(dirname "$HERE")"

asserts=0
fail() { echo "ASSERT FAILED [$1]: $2" >&2; exit 1; }
ok() { asserts=$((asserts + 1)); echo "  ok [$1] $2"; }

# ── (a) qwen3.8-27b resolves through kernel_source to qwen3.6-27b ────────────
out38="$(bash "$GATE" --hw gb10 --model qwen3.8-27b --list-tasks 2>&1)"; rc=$?
[ $rc -eq 0 ] || fail a "--list-tasks exited $rc:
$out38"

tasks38="$(grep -c '^qwen3\.8-27b	' <<<"$out38")"
[ "$tasks38" -eq 181 ] || fail a "expected 181 tasks for qwen3.8-27b, got $tasks38:
$out38"
ok a "qwen3.8-27b lists 181 tasks"

model_files="$(awk -F'\t' '$3 !~ /^kernels\/gb10\/common\// {print $3}' <<<"$out38" | sort)"
n_model="$(grep -c . <<<"$model_files")"
[ "$n_model" -eq 14 ] || fail a "expected 14 non-common sources, got $n_model:
$model_files"
stray="$(grep -v '^kernels/gb10/qwen3\.6-27b/nvfp4/' <<<"$model_files")"
[ -z "$stray" ] || fail a "every non-common source must come from the redirect target:
$stray"
ok a "the 14 model kernels all come from kernels/gb10/qwen3.6-27b/nvfp4/"

for stem in w4a16_gemm moe_w4a16_grouped_gemm gated_delta_rule \
            inferspark_prefill_paged_indirect; do
  line="$(awk -F'\t' -v s="$stem" '$2 == s {print $3}' <<<"$out38")"
  [ "$line" = "kernels/gb10/qwen3.6-27b/nvfp4/$stem.cu" ] \
    || fail a "$stem must shadow the common kernel, resolved to '$line'"
done
ok a "the 4 shadowing stems resolve to the redirect target, not common/"

# ── (b) the redirect is an alias: same sources as the target itself ──────────
out36="$(bash "$GATE" --hw gb10 --model qwen3.6-27b --list-tasks 2>&1)"; rc=$?
[ $rc -eq 0 ] || fail b "--list-tasks for qwen3.6-27b exited $rc:
$out36"
diff <(cut -f2,3 <<<"$out38" | sort) <(cut -f2,3 <<<"$out36" | sort) >/dev/null \
  || fail b "qwen3.8-27b and qwen3.6-27b must select the same (stem, source) set:
$(diff <(cut -f2,3 <<<"$out38" | sort) <(cut -f2,3 <<<"$out36" | sort))"
ok b "qwen3.8-27b and qwen3.6-27b select an identical (stem, source) set"

# ── (c) the redirecting model's own directory is never compiled ──────────────
own="$(grep -F 'kernels/gb10/qwen3.8-27b/' <<<"$out38" || true)"
[ -z "$own" ] || fail c "qwen3.8-27b's own directory must supply no kernel source:
$own"
[ -d "$ROOT/kernels/gb10/qwen3.8-27b/nvfp4" ] \
  && fail c "this test assumes qwen3.8-27b ships no nvfp4 tree; it now does"
ok c "no task points into kernels/gb10/qwen3.8-27b/"

# ── (d) a nonexistent model is refused ──────────────────────────────────────
out_bad="$(bash "$GATE" --hw gb10 --model does-not-exist --list-tasks 2>&1)"; rc=$?
[ $rc -eq 2 ] || fail d "an unknown model must exit 2, got $rc:
$out_bad"
grep -Fq -- "does-not-exist" <<<"$out_bad" \
  || fail d "the refusal must name the model it looked for: $out_bad"
grep -Fq -- "qwen3.6-27b" <<<"$out_bad" \
  || fail d "the refusal must list the models that do exist: $out_bad"
ok d "an unknown model exits 2 and names what it looked for"

# ── (e) every ptxas error is recorded, not just the first ───────────────────
# The one case that needs a toolchain. It cross-compiles for sm_90a, so it
# needs no NVIDIA hardware and touches no GPU if one is present.
#
# The gate compiles what is under `kernels/<hw>/`, so the fixture is handed a
# throwaway hardware set in a sandbox tree: a copy of scripts/ (the gate finds
# its fixtures relative to itself and derives the repo root from its own path)
# plus a one-model kernels/ tree holding the fixture. Nothing is written into
# the real tree; the sandbox is a mktemp -d the trap removes.
find_toolchain() {
  local n
  for n in "${NVCC_BIN:-}" "${CUDA_HOME:-}/bin/nvcc" \
           "$(command -v nvcc 2>/dev/null)" /usr/local/cuda/bin/nvcc; do
    # ptxas is what rejects this fixture, so nvcc on its own is not enough.
    if [ -n "$n" ] && [ -x "$n" ] && [ -x "$(dirname "$n")/ptxas" ]; then
      printf '%s\n' "$n"
      return 0
    fi
  done
  return 1
}

NVCC="$(find_toolchain)" || NVCC=""
if [ -z "$NVCC" ]; then
  echo "  -- [e] no nvcc+ptxas pair found, skipped"
else
  SB="$(mktemp -d)"
  trap 'rm -rf "$SB"' EXIT
  mkdir -p "$SB/scripts" "$SB/kernels/gatefx/common" \
           "$SB/kernels/gatefx/twoentries/nvfp4"
  cp "$GATE" "$SB/scripts/"
  cp -R "$HERE/fixtures" "$SB/scripts/"
  cp "$HERE/fixtures/hopper_gate/known_bad_two_entries.cu" \
     "$SB/kernels/gatefx/twoentries/nvfp4/"
  # sm_90a because that is an arch whose registered negative self-test fixture
  # (known_bad_post_hopper.cu) is measured to fail there: the gate's own
  # self-test keeps a working failure path inside the sandbox.
  cat >"$SB/kernels/gatefx/HARDWARE.toml" <<'TOML'
[hardware]
name = "gatefx"
vendor = "nvidia"
arch = "sm_90a"
TOML
  cat >"$SB/kernels/gatefx/twoentries/MODEL.toml" <<'TOML'
[model]
name = "twoentries"
TOML
  out_e="$(bash "$SB/scripts/hopper_ptx_gate.sh" --hw gatefx --model twoentries \
             --jobs 1 --nvcc "$NVCC" --out "$SB/ledger.json" 2>&1)"; rc=$?
  # 1, not 0: the fixture is there to be rejected. A 2 is the gate refusing to
  # run at all (no toolchain, no arch, no negative fixture) and is a test bug.
  [ $rc -eq 1 ] || fail e "the gate must exit 1 on the two-entry fixture, got $rc:
$out_e"
  python3 - "$SB/ledger.json" <<'PY' || fail e "the ledger must record BOTH ptxas errors:
$out_e"
import json
import sys

led = json.load(open(sys.argv[1]))
bad = []
rows = [r for r in led["results"] if r["stem"] == "known_bad_two_entries"]
if len(rows) != 1:
    bad.append(f"expected exactly one result row, got {len(rows)}")
else:
    r = rows[0]
    # nvcc emits the PTX; the 48 KiB static shared limit is ptxas's to enforce.
    if (r["ptx_ok"], r["ptxas_ok"]) != (True, False):
        bad.append("expected ptx_ok=True, ptxas_ok=False; got "
                   f"{r['ptx_ok']}, {r['ptxas_ok']}")
    if r.get("error_count") != 2:
        bad.append(f"error_count: expected 2, got {r.get('error_count')!r}")
    errors = r.get("errors")
    if not isinstance(errors, list) or len(errors) != 2:
        bad.append(f"errors: expected a list of 2, got {errors!r}")
    elif r.get("error_head") != errors[0]:
        bad.append("error_head must stay the first of errors: "
                   f"{r.get('error_head')!r} vs {errors[0]!r}")
if led["summary"].get("rejected_entries") != 2:
    bad.append("summary.rejected_entries: expected 2, got "
               f"{led['summary'].get('rejected_entries')!r}")
if bad:
    print("\n".join("    " + b for b in bad), file=sys.stderr)
    sys.exit(1)
PY
  ok e "both ptxas errors land in the ledger (error_count=2, rejected_entries=2)"
fi

# ── (f) lints ───────────────────────────────────────────────────────────────
if command -v shellcheck >/dev/null 2>&1; then
  shellcheck "$GATE" || fail f "shellcheck failed on $GATE"
  shellcheck "${BASH_SOURCE[0]}" || fail f "shellcheck failed on this test"
  ok f "shellcheck clean on the gate and on this test"
else
  echo "  -- [f] shellcheck not installed, skipped"
fi

echo ""
echo "ALL $asserts assertions passed."
