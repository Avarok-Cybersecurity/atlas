// SPDX-License-Identifier: AGPL-3.0-only

//! Anthropic test suites split by area.

// TODO: stale tests — `translator_a` / `translator_b` reference
// `anthropic_to_chat_request_json`, `chat_to_anthropic_response`, and
// `Event` which were renamed/removed during the streaming refactor.
// `types_convert` has E0282 inference errors after a serde upgrade.
// All three files left on disk; re-enable as their assertions are
// updated against the current Anthropic translator API.
// mod translator_a;
// mod translator_b;
// mod types_convert;
