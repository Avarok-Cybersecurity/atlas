#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Release helpers used by .github/workflows/release.yml.
#
# `bump <version>` rewrites `[workspace.package].version` in the root
# Cargo.toml and refreshes Cargo.lock. It is deliberately the ONLY place that
# knows how to edit the version: the workflow, a local release and any future
# scripts/release.sh all go through here, so the Cargo version can never drift
# from the git tag.
#
# `preview <version>` prints the release notes GitHub would generate, without
# creating anything — used to make a dry run show its work.
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
usage:
  release-notes.sh bump <version>       rewrite workspace version + Cargo.lock
  release-notes.sh preview <version>    print the release notes GitHub would generate
EOF
    exit 2
}

repo_root() { git rev-parse --show-toplevel; }

# Rewrite only the `version` key inside the `[workspace.package]` table.
# A blind `s/version = ...//` would also rewrite `rust-version`, every
# dependency pin, and the `[package]` version of any crate in the file.
bump() {
    local version="$1" root manifest
    [ -n "$version" ] || usage
    printf '%s' "$version" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$' || {
        echo "error: version must be bare semver (1.2.3), got '$version'" >&2
        exit 1
    }
    root="$(repo_root)"
    manifest="$root/Cargo.toml"

    awk -v ver="$version" '
        /^\[/ { in_wp = ($0 == "[workspace.package]") }
        in_wp && /^version[[:space:]]*=/ && !done { print "version = \"" ver "\""; done = 1; next }
        { print }
        END { if (!done) { print "error: [workspace.package].version not found" > "/dev/stderr"; exit 1 } }
    ' "$manifest" > "$manifest.tmp"
    mv "$manifest.tmp" "$manifest"

    # Keep Cargo.lock in step; --offline so a release build never silently
    # picks up a newer transitive dependency than the one that was tested.
    (cd "$root" && cargo update --workspace --offline 2>/dev/null || cargo update --workspace)

    echo "workspace version -> $version"
    grep -A5 '^\[workspace.package\]' "$manifest" | grep '^version'
}

# Ask GitHub for the notes it would attach to this tag. Same generator the
# real publish uses, so a dry run previews the actual output rather than an
# approximation of it.
preview() {
    local version="$1" repo prev
    [ -n "$version" ] || usage
    repo="${GITHUB_REPOSITORY:-$(gh repo view --json nameWithOwner -q .nameWithOwner)}"
    prev="$(gh api "repos/$repo/releases/latest" -q .tag_name 2>/dev/null || true)"

    local args=(-f tag_name="v$version")
    [ -n "$prev" ] && args+=(-f previous_tag_name="$prev")

    echo "# Release notes preview for v$version (previous tag: ${prev:-<none, will cover full history>})"
    gh api --method POST "repos/$repo/releases/generate-notes" "${args[@]}" -q .body
}

case "${1:-}" in
    bump)    shift; bump "${1:-}" ;;
    preview) shift; preview "${1:-}" ;;
    *)       usage ;;
esac
