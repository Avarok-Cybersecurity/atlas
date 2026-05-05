<!--
Thanks for contributing to Atlas! A few reminders:

- Read CONTRIBUTING.md if you haven't yet.
- CI runs cargo fmt, cargo clippy -Dwarnings, the SPDX-header check, typos,
  and cargo-deny (via the security workflow). Please run `cargo fmt --all`
  and `bash scripts/check-license-headers.sh` locally before pushing.
- If you add a new source file, it needs `// SPDX-License-Identifier: AGPL-3.0-only`
  on line 1.
-->

## Summary

<!-- 1-3 sentences: what changes and why. Reference the driving issue if any. -->

Closes #

## Test plan

<!-- Concrete checklist of how you verified this. Logs/benchmarks welcome. -->

- [ ] `cargo fmt --all -- --check`
- [ ] `ATLAS_SKIP_BUILD=1 cargo clippy --workspace --tests --all-features -- -Dwarnings`
- [ ] `bash scripts/check-license-headers.sh`
- [ ] Tested against a real model / hardware if the change affects runtime behaviour
- [ ] Added or updated tests where applicable

## Notes for reviewers

<!-- Design rationale, trade-offs, follow-ups you deferred, things you want a second opinion on. -->

## CLA

- [ ] I have read and agree to the [Contributor License Agreement](../CLA.md).
