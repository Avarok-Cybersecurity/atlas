#!/bin/bash
# E1 — the 128-token warm-restore anomaly (Marconi floor). Thin wrapper: the arms,
# cells and pass rules live in ab_concurrent-prefill-ttft.sh under EXPERIMENT=E1.
#   ARM=A   ./ab_marconi_floor.sh                              # control (floor 256 [default])
#   ARM=B0  ATLAS_MARCONI_MIN_TOKENS=0  ./ab_marconi_floor.sh
#   ARM=B64 ATLAS_MARCONI_MIN_TOKENS=64 ./ab_marconi_floor.sh  # shipping candidate
#   ./ab_marconi_floor.sh all                                  # every E1 arm, control re-run last
exec env EXPERIMENT=E1 "$(dirname "$0")/ab_concurrent-prefill-ttft.sh" "$@"
