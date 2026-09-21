#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Verify the committed GDN binaries match PINS.sha256.
#
# Extracted verbatim from gdn-so-pin.yml so the standalone job and the batched
# `cheap checks` job run THE SAME CODE during the transition. If these two
# ever disagree, the batch is not a faithful merge and the transition is not
# safe to complete.
set -euo pipefail
cd 3rdparty_patches/gdn_aot
if sha256sum --check --strict PINS.sha256; then
  echo "All pinned GDN artifacts (libatlasgdn.so, gdn_holo.so, export patch) match PINS.sha256."
  exit 0
fi
echo ""
echo "::error file=3rdparty_patches/gdn_aot/PINS.sha256::A pinned GDN binary or the export patch changed without updating PINS.sha256."
cat <<'EOF'
The committed GDN AOT binaries no longer match their recorded pins.

If the change is INTENTIONAL (a deliberate re-export/relink):
  1. Rebuild reproducibly:  3rdparty_patches/gdn_aot/rebuild.sh
     (GB10 box with the cuda-13.2 compat stack; see the script header
     and STATUS.md "PROVENANCE" for the toolchain requirements).
  2. Update PINS.sha256 with the new sha256sums IN THE SAME COMMIT,
     and record in the commit message which toolchain versions
     (nvidia-cutlass-dsl, FlashInfer rev, gcc, nvcc) produced them.

If you did NOT mean to change these binaries: someone or something
swapped a .so blob — do not merge; restore the pinned bytes.
EOF
exit 1

