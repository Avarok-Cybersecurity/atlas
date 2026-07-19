# Lever-verify methodology (Tier A / Tier B)

Two cheap verify tiers to fold inference levers WITHOUT running a full 2.5h e2e per variant.

- **Tier A — equivalence-preserving changes** (e.g. in-place GDN K=4 verify, midchunk GDN
  tail capture): `verify_win.sh` runs a byte-identical output check (identical outputs ⇒
  identical BFCL score ⇒ accuracy provably preserved) + `probe_profile.py` /
  `probe_continuation.py` for the speed delta. Fold if byte-identical AND faster.
- **Tier B — numerics-changing changes** (e.g. fp8 KV): BFCL subset A/B
  (`category_sample_pct` cut + `--accuracy-only`) for the accuracy delta + speed probe.
  Fold only if accuracy ≥ baseline.

Scripts are box-specific (paths under `/workspace`, endpoint on localhost:8888 / 10.10.10.2:8888).
See `docs/lever-folding/LEVER_FOLDING_CONTEXT_2026_07_19.md` for the full run state, lever
inventory, gotchas, and next steps.
