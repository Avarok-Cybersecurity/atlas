#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Validate campaign request-mode eligibility before rendering or launching."""

import argparse
import json
import pathlib
import sys

POLICY_PATH = pathlib.Path(__file__).with_name("campaign_policy.json")
E_THINK_POLICY = 9


def refusal(doc, model, think):
    """Return a refusal reason, or None for an explicitly permitted mode."""
    entry = doc.get("models", {}).get(model)
    if not isinstance(entry, dict):
        return f"REFUSED: {model} --think {think}: no campaign thinking policy"
    if entry.get("blocked_reason"):
        return f"REFUSED: {model} --think {think}: {entry['blocked_reason']}"
    allowed = entry.get("allowed")
    if (not isinstance(allowed, list) or not allowed
            or any(mode not in ("on", "off") for mode in allowed)):
        return f"REFUSED: {model} --think {think}: invalid campaign thinking policy"
    if think not in allowed:
        reason = entry.get("reason", "mode is excluded by the campaign policy")
        return f"REFUSED: {model} --think {think}: {reason}; allowed: {', '.join(allowed)}"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--think", choices=["on", "off"], required=True)
    args = ap.parse_args()
    error = refusal(json.loads(POLICY_PATH.read_text()), args.model, args.think)
    if error:
        print(error, file=sys.stderr)
        return E_THINK_POLICY
    return 0


if __name__ == "__main__":
    sys.exit(main())
