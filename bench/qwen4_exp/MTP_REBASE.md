# PR834 rebase onto main

- Original tip: `60a3d5a47`.
- Base: `avarok/main` at `8682329cce0dd8bec1d3775704e978533b00bf7a`.
- New tip: `0d099342e977544acdf096e1c6d22d87a6d1f952`.
- Branch: `integration/pr834-main-rebase`.
- Worktree: `/home/ms/atlas/.claude/worktrees/pr834-main-rebase`.
- Command: `git rebase --rebase-merges avarok/main`.

## Conflict resolution

Only recreated historical merge `0d64bcb83` conflicted, in `crates/spark-model/src/model/trait_impl/verify_hc.rs` and `crates/spark-model/src/model/types.rs`. Reapplied the exact original merge resolution for those files: retain both the per-row `VerifyAuxRows` snapshot state and the batched `pending_verify_span` state; preserve batched early return and the separate per-row auxiliary restore path. No newer main change touched either file. All later fixes replayed cleanly.

## Verification

- Worktree clean; root checkout untouched.
- New main is ancestor of rebased tip (`git merge-base HEAD avarok/main` equals new main).
- Final tree `dbc90b6d289e670c44abbc93f824c7109775249e` is exactly the tree independently produced by `git merge-tree --write-tree 60a3d5a47 avarok/main`.
- No changes between old/rebased tips in `crates/`, `kernels/`, `Cargo.toml`, or `Cargo.lock`.
- Differences from old tip are exactly main's 39 CI/certification/documentation/site files.
- No compile run during the root GPU sweep; runtime and dependency trees are unchanged.
- Nothing pushed; root branch remains at its original tip.

Root checkout now uses the rebased tip. Release build succeeded with unchanged runtime/dependency sources.
