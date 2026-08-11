#!/usr/bin/env bash
# rebuild.sh — reproduce libatlasgdn.so + gdn_holo.so from pinned sources.
#
# This is the ONE recipe for the binaries pinned in PINS.sha256. It is the
# union of the two places the recipe actually lives:
#   * export + gdn_holo.so link:  STATUS.md (Route-A history, 2026-06-30)
#   * libatlasgdn.so link:        docker/gb10/Dockerfile.builder (RUN cd 3rdparty_patches/gdn_aot ...)
# Do not add flags here without changing those in the same commit.
#
# Pipeline:
#   1. FlashInfer checkout at the pinned rev (or $FLASHINFER_HOME as-is, warned if drifted)
#   2. apply delta_rule_sm120_aot_export.patch   (enables compiled_fn.export_to_c)
#   3. gdn_export.py  -> gdn_holo_0.{h,o}        (NEEDS: GB10-class GPU + torch cu13
#                                                 + nvidia-cutlass-dsl[cu13]==4.5.0)
#   4. g++ -shared gdn_holo_0.o                  -> gdn_holo.so
#   5. nvcc gdn_transpose.cu + g++ shim link     -> libatlasgdn.so
#   6. sha256sum of everything produced, compared against PINS.sha256
#
# Env knobs (all optional):
#   FLASHINFER_HOME  existing FlashInfer checkout; unset => clone the pinned rev
#   GDN_PYTHON       python with torch+cutlass-dsl+flashinfer deps (default: python3)
#   GDN_HOLO_O       pre-exported gdn_holo_0.o — SKIPS steps 1-3 (link-only rebuild)
#   GDN_REBUILD_OUT  output dir (default: /tmp/gdn_rebuild)
#   CUTE_DSL_LIB     dir containing libcute_dsl_runtime.so (default: aot_config --libdir)
#   GDN_RPATH        -Wl,-rpath value for libatlasgdn.so (default: $CUTE_DSL_LIB).
#                    NOTE: the rpath string is part of the bytes; the committed
#                    artifact carries /tmp/gdn-bench/.../nvidia_cutlass_dsl/lib.
#   CUDA_HOME        CUDA toolkit (default: /usr/local/cuda)
set -euo pipefail

# ── Pins (keep in sync with PINS.sha256 + docker/gb10/Dockerfile.builder) ─────
FLASHINFER_GIT_URL=${FLASHINFER_GIT_URL:-https://github.com/flashinfer-ai/flashinfer.git}
FLASHINFER_SHA=a671c02ee2fbcdde7cc991f5a01c7cf5eb4a8972
CUTLASS_DSL_VER=4.5.0
SM_ARCH=sm_121a

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${GDN_REBUILD_OUT:-/tmp/gdn_rebuild}
PY=${GDN_PYTHON:-python3}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$OUT"

die() { echo "rebuild.sh: ERROR: $*" >&2; exit 1; }
note() { echo "rebuild.sh: $*"; }

# ── Steps 1-3: AOT export (skipped when GDN_HOLO_O points at a pre-built .o) ──
if [ -n "${GDN_HOLO_O:-}" ]; then
    [ -f "$GDN_HOLO_O" ] || die "GDN_HOLO_O=$GDN_HOLO_O does not exist"
    note "using pre-exported AOT object: $GDN_HOLO_O (export steps 1-3 SKIPPED)"
    cp "$GDN_HOLO_O" "$OUT/gdn_holo_0.o"
else
    # 1. FlashInfer checkout at the pinned rev.
    if [ -z "${FLASHINFER_HOME:-}" ]; then
        FLASHINFER_HOME="$OUT/flashinfer"
        if [ ! -d "$FLASHINFER_HOME/.git" ]; then
            note "cloning FlashInfer @ $FLASHINFER_SHA -> $FLASHINFER_HOME"
            git clone --filter=blob:none "$FLASHINFER_GIT_URL" "$FLASHINFER_HOME"
        fi
        git -C "$FLASHINFER_HOME" checkout --quiet "$FLASHINFER_SHA"
    fi
    [ -f "$FLASHINFER_HOME/flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_sm120.py" ] \
        || die "FLASHINFER_HOME=$FLASHINFER_HOME has no gdn_kernels/delta_rule_dsl — wrong dir or too-old rev"
    ACTUAL_SHA=$(git -C "$FLASHINFER_HOME" rev-parse HEAD 2>/dev/null || echo unknown)
    if [ "$ACTUAL_SHA" != "$FLASHINFER_SHA" ]; then
        note "WARNING: FLASHINFER_HOME is at $ACTUAL_SHA, pin is $FLASHINFER_SHA — bytes will not be comparable"
    fi

    # 2. Apply the export-enabling patch (idempotent). The patch's ---/+++ headers
    #    carry absolute /home/ms/flashinfer paths, so name the target explicitly.
    TARGET_PY="$FLASHINFER_HOME/flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_sm120.py"
    PATCH_FILE="$HERE/delta_rule_sm120_aot_export.patch"
    if patch --dry-run --silent --reverse --force "$TARGET_PY" < "$PATCH_FILE" >/dev/null 2>&1; then
        note "patch already applied to $TARGET_PY"
    elif patch --dry-run --silent --forward --force "$TARGET_PY" < "$PATCH_FILE" >/dev/null 2>&1; then
        patch --forward --backup --suffix=.bak_aotexport "$TARGET_PY" < "$PATCH_FILE"
        note "applied delta_rule_sm120_aot_export.patch (backup: .bak_aotexport)"
    else
        die "delta_rule_sm120_aot_export.patch applies to neither direction of $TARGET_PY — FlashInfer rev drifted from pin $FLASHINFER_SHA"
    fi

    # 3. Run the export. This step is GPU- and toolchain-bound: CuTe DSL JIT-compiles
    #    the kernel for the LOCAL GPU before export_to_c can capture it.
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 \
        || die "no NVIDIA GPU visible. The AOT export must run on a GB10-class (sm_121a) box \
— gx10-9959 is the only VERIFIED export environment (dgx-00 reproducibly fails the export; \
the Dockerfile.builder image lacks torch and cannot rebuild gdn_holo_0.o from a clean checkout). \
Link-only rebuilds can pass GDN_HOLO_O=<pre-exported gdn_holo_0.o> instead."
    "$PY" -c 'import torch' 2>/dev/null \
        || die "$PY has no torch. Need torch (cu13 build) — e.g. uv pip install torch --index-url https://download.pytorch.org/whl/cu130"
    "$PY" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' \
        || die "torch sees no CUDA device (driver/compat mismatch?). Need a working cu13 torch on a GB10-class box."
    "$PY" -c 'import cutlass' 2>/dev/null \
        || die "$PY has no CuTe DSL. Need: uv pip install 'nvidia-cutlass-dsl[cu13]==$CUTLASS_DSL_VER' (the version the pin was built with)"
    PYTHONPATH="$FLASHINFER_HOME" "$PY" -c 'import flashinfer' 2>/dev/null \
        || die "flashinfer not importable from $FLASHINFER_HOME (missing deps? pip install -r $FLASHINFER_HOME/requirements.txt, torch excluded)"

    # The CuTe-DSL JIT/export engine needs the cuda-13.2 compat userspace on
    # GB10 boxes whose system driver is older (historical recipe, STATUS.md:
    # LD_LIBRARY_PATH=/usr/local/cuda-13.2/compat:...). Honor it when present.
    COMPAT_DIR=${GDN_COMPAT_DIR:-/usr/local/cuda-13.2/compat}
    # VERIFIED gx10 2026-08-10: the JIT session resolves its 8 trampoline
    # symbols (_cuKernelGetAttribute, _cudaLaunchKernelEx, ...) from
    # libcute_dsl_runtime.so, which must be BOTH on the loader path and
    # LD_PRELOADed — compat alone still fails with the same Symbols-not-found.
    CUTE_RT_LIB=$("$PY" - <<'PYEOF'
import glob, sys, sysconfig
cands = glob.glob(sysconfig.get_paths()["purelib"] + "/nvidia_cutlass_dsl/lib")
print(cands[0] if cands else "")
PYEOF
)
    [ -n "$CUTE_RT_LIB" ] || die "nvidia_cutlass_dsl/lib not found in this python env (pip install 'nvidia-cutlass-dsl[cu13]==$CUTLASS_DSL_VER')"
    EXPORT_LD_PATH="$CUTE_RT_LIB:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    EXPORT_PRELOAD="$CUTE_RT_LIB/libcute_dsl_runtime.so"
    if [ -d "$COMPAT_DIR" ]; then
        EXPORT_LD_PATH="$COMPAT_DIR:$EXPORT_LD_PATH"
        note "using CUDA compat driver: $COMPAT_DIR"
    else
        note "WARNING: no cuda-13.2 compat dir at $COMPAT_DIR — on boxes with a 13.0-era system driver (e.g. dgx-00, 580.173.02) the CuTe-DSL engine fails with 'JIT session error: Symbols not found: [cuKernelGetAttribute, cudaLaunchKernelEx, ...]' and export_to_c fails with 'Failed to dump object file with PIC relocation'"
    fi
    note "exporting AOT kernel (CuTe DSL JIT + export_to_c, ~1-3 min on GB10)..."
    rm -rf /tmp/gdn_aot   # gdn_export.py hardcodes this output dir
    ( cd "$OUT" && LD_LIBRARY_PATH="$EXPORT_LD_PATH" LD_PRELOAD="$EXPORT_PRELOAD" PYTHONPATH="$FLASHINFER_HOME" CUTE_DSL_ARCH=$SM_ARCH "$PY" "$HERE/gdn_export.py" ) \
        || true   # gdn_export.py prints per-kernel EXPORT FAIL details; verdict is the .o below
    [ -f /tmp/gdn_aot/gdn_holo_0.o ] || die "export did not produce /tmp/gdn_aot/gdn_holo_0.o (see output above). \
Known-good environment: a GB10 box WITH the cuda-13.2 compat driver stack (gx10-9959, or the \
gx10-9959 with the cuda-13.2 compat stack) + nvidia-cutlass-dsl[cu13]==$CUTLASS_DSL_VER. \
NOTE: the Dockerfile.builder image is NOT a verified export env (no torch; and its relink step \
references a gitignored gdn_holo_0.o, so it cannot build from a clean checkout — pre-existing bug). \
dgx-00 (driver 580.173.02, no /usr/local/cuda-13.2/compat) reproducibly FAILS here — verified 2026-08-10."
    cp /tmp/gdn_aot/gdn_holo_0.o /tmp/gdn_aot/gdn_holo_0.h "$OUT/"
    if ! cmp -s "$OUT/gdn_holo_0.h" "$HERE/gdn_holo_0.h"; then
        note "WARNING: exported gdn_holo_0.h differs from the committed header — C ABI drift; do NOT ship without revalidating gdn_shim.cpp"
    fi
fi

# ── Locate the CuTe DSL runtime lib (needed by both links) ────────────────────
if [ -z "${CUTE_DSL_LIB:-}" ]; then
    CUTE_DSL_LIB=$("$PY" -m cutlass.cute.export.aot_config --libdir 2>/dev/null || true)
fi
[ -n "$CUTE_DSL_LIB" ] && [ -f "$CUTE_DSL_LIB/libcute_dsl_runtime.so" ] \
    || die "libcute_dsl_runtime.so not found. Set CUTE_DSL_LIB=<dir> or install 'nvidia-cutlass-dsl[cu13]==$CUTLASS_DSL_VER' into \$GDN_PYTHON"
note "CuTe DSL runtime: $CUTE_DSL_LIB"

command -v g++ >/dev/null 2>&1 || die "g++ not found (pin was linked with g++ 13.3.0, Ubuntu 24.04 aarch64)"
[ -x "$CUDA_HOME/bin/nvcc" ] || command -v nvcc >/dev/null 2>&1 \
    || die "nvcc not found — need the CUDA 13.x toolkit for gdn_transpose.cu (-arch=$SM_ARCH); set CUDA_HOME"
NVCC=${NVCC:-$CUDA_HOME/bin/nvcc}; command -v "$NVCC" >/dev/null 2>&1 || NVCC=nvcc

cd "$OUT"

# ── Step 4: gdn_holo.so (STATUS.md: g++ -shared + aot_config --ldflags --libs) ─
LDCFG=$("$PY" -m cutlass.cute.export.aot_config --ldflags --libs 2>/dev/null | tr '\n' ' ' \
        || echo "-L$CUTE_DSL_LIB -lcute_dsl_runtime")
# shellcheck disable=SC2086  # LDCFG is intentionally word-split (-L... -l...)
g++ -shared gdn_holo_0.o -o gdn_holo.so $LDCFG
note "linked gdn_holo.so"

# ── Step 5: libatlasgdn.so (docker/gb10/Dockerfile.builder recipe, verbatim
#    plus -L$CUDA_HOME/lib64 which the docker image supplies via ldconfig) ─────
"$NVCC" -arch=$SM_ARCH -Xcompiler -fPIC -c "$HERE/gdn_transpose.cu" -o gdn_transpose.o
# -Wl,--build-id=none: the ONLY nondeterminism left in the pipeline (verified
# gx10 2026-08-10: .o and gdn_holo.so bit-identical across runs; libatlasgdn.so
# differed run-to-run solely from the linker build-id).
g++ -O2 -fPIC -shared -Wl,--build-id=none "$HERE/gdn_shim.cpp" gdn_transpose.o gdn_holo_0.o \
    -o libatlasgdn.so \
    -I"$HERE" -I"$CUDA_HOME/include" -L"$CUDA_HOME/lib64" -lcudart \
    -L"$CUTE_DSL_LIB" -lcute_dsl_runtime -Wl,-rpath,"${GDN_RPATH:-$CUTE_DSL_LIB}"
note "linked libatlasgdn.so"

# ── Step 6: hashes + verdict vs PINS.sha256 ───────────────────────────────────
echo
echo "== rebuilt artifact hashes ($OUT) =="
sha256sum gdn_holo_0.o gdn_holo.so libatlasgdn.so
echo
echo "== committed pins ($HERE/PINS.sha256) =="
grep -E '^[0-9a-f]{64}' "$HERE/PINS.sha256" || true
echo
verdict=0
for so in libatlasgdn.so gdn_holo.so; do
    want=$(grep -E "^[0-9a-f]{64}  $so\$" "$HERE/PINS.sha256" | cut -d' ' -f1 || true)
    got=$(sha256sum "$so" | cut -d' ' -f1)
    if [ "$want" = "$got" ]; then
        echo "MATCH  $so — rebuild is bit-identical to the committed pin"
    else
        echo "DRIFT  $so — rebuilt $got != pinned ${want:-<no pin>}"
        verdict=1
    fi
done
if [ "$verdict" -ne 0 ]; then
    cat <<'EOF'

Rebuilt bytes differ from the pins. This is EXPECTED unless the exporting
toolchain matches the original exactly (CuTe DSL 4.5.0 JIT output, FlashInfer
rev+patch, gcc 13.3.0, nvcc 13.x, and the identical -Wl,-rpath string — see
PINS.sha256 header). Record BOTH hashes and the toolchain versions in the
commit message if you intend to re-pin: update PINS.sha256 in the same commit
that swaps the .so files, or CI (gdn-so-pin.yml) will fail.
EOF
fi
exit "$verdict"
