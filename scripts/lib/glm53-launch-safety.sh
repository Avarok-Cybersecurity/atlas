#!/bin/bash
# Validate the launcher's documented space-separated EXTRA_ARGS contract, then emit
# the operator-provided serve tail followed by the fixed GLM safety pin.
#
# The pin is defense in depth, NOT the correctness boundary: `spark serve` refuses
# a KV-only swap-out for any model whose prefill builds state outside KV
# (`ModelConfig::kv_only_swap_out_is_safe` -> `resolve_swap_space_gb`), however it
# is invoked. What this file actually buys is the EXTRA_ARGS contract itself —
# these tokens are interpolated into a REMOTE shell command, and validating them
# is worth doing whatever the flag under discussion happens to be.
glm53_safe_serve_tail() {
  local extra_args="${1:-}"
  local arg

  # These tokens are interpolated into a remote shell command by the existing launcher.
  # Validate the complete string first: word splitting would otherwise hide newlines, and
  # printing the original value would put that hidden control syntax back into the command.
  # [:blank:] admits only the documented space/tab separators, never CR/LF.
  if [[ ! "$extra_args" =~ ^[[:alnum:]_./,:+=%@[:blank:]-]*$ ]]; then
    return 2
  fi

  for arg in $extra_args; do
    case "$arg" in
      --swap-space-gb|--swap-space-gb=*) return 2 ;;
    esac
  done

  # Appending this last is defense in depth for clap-style duplicate resolution; the
  # explicit duplicate rejection above is the fail-closed guarantee for supported input.
  if [[ -n "$extra_args" ]]; then
    printf '%s ' "$extra_args"
  fi
  printf '%s' '--swap-space-gb 0'
}
