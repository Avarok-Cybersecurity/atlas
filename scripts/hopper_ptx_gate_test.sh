#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Tests for scripts/hopper_ptx_gate.sh. No CUDA toolchain, no GPU: every case
# goes through `--list-tasks`, which resolves the source set and prints it
# without compiling anything.
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
#   (e) shellcheck on the gate and on this file.
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

# ── (e) lints ───────────────────────────────────────────────────────────────
if command -v shellcheck >/dev/null 2>&1; then
  shellcheck "$GATE" || fail e "shellcheck failed on $GATE"
  shellcheck "${BASH_SOURCE[0]}" || fail e "shellcheck failed on this test"
  ok e "shellcheck clean on the gate and on this test"
else
  echo "  -- [e] shellcheck not installed, skipped"
fi

echo ""
echo "ALL $asserts assertions passed."
