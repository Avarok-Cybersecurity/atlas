#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
#
# Tombstone. This URL used to install `sparkrun`; it no longer does.
#
# It is kept because the old one-liner is in blog posts, chat history, and
# people's notes. A 404 there would be silent; this is not. It deliberately does
# not install anything on your behalf — being redirected to software you did not
# ask for is the reason the launcher changed.
set -eu

printf '\033[1;33m[atlas]\033[0m %s\n' "quickstart.sh has been replaced." >&2
cat >&2 <<'MSG'

  Atlas is now launched with `atlasctl`, which replaces sparkrun.

  Install it with:

      curl -fsSL https://atlasinference.io/install.sh | sh

  Why the change: sparkrun redirects the Atlas recipe registry to a repository
  Atlas does not control, and marks it trusted — which lets recipe-supplied
  shell commands run on your host. If you have sparkrun installed, run
  `atlasctl doctor` after installing, or see:

      https://github.com/Avarok-Cybersecurity/atlas-recipes/blob/main/SECURITY.md

MSG
exit 1
