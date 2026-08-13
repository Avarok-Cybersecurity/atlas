// SPDX-License-Identifier: AGPL-3.0-only

//! The GitHub protocol parsers and the body assembly, pinned against the
//! response shapes the docs describe — the shapes marked unverified in the
//! design are exactly why every parse here has a malformed-input twin.

use super::*;

// ── Target resolution ──

#[test]
fn compiled_in_identity_is_the_default() {
    let t = target_from(None, None).expect("official identity");
    assert_eq!(t.client_id, OFFICIAL_CLIENT_ID);
    assert_eq!(t.repo, OFFICIAL_REPO);
}

#[test]
fn overrides_replace_the_identity() {
    let t = target_from(Some("Iv1fork".into()), Some("fork/atlas".into())).expect("override");
    assert_eq!(t.client_id, "Iv1fork");
    assert_eq!(t.repo, "fork/atlas");
}

#[test]
fn set_but_empty_override_fails_loudly() {
    // An empty client_id would produce a GitHub error page the user cannot
    // act on; refusing with the env-var names is the actionable failure.
    assert_eq!(target_from(Some("".into()), None), Err(NOT_CONFIGURED));
    assert_eq!(target_from(None, Some("  ".into())), Err(NOT_CONFIGURED));
}

impl PartialEq for Target {
    fn eq(&self, other: &Self) -> bool {
        self.client_id == other.client_id && self.repo == other.repo
    }
}
impl std::fmt::Debug for Target {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Target({}, {})", self.client_id, self.repo)
    }
}

// ── Device-code parsing ──

const GRANT: &str = r#"{"device_code":"3584d83530557fdd1f46af8289938c8ef79f9dc5","user_code":"WDJB-MJHT","verification_uri":"https://github.com/login/device","expires_in":899,"interval":5}"#;

#[test]
fn device_code_response_parses() {
    let g = parse_device_code(200, GRANT).expect("grant");
    assert_eq!(g.user_code, "WDJB-MJHT");
    assert_eq!(g.verification_uri, "https://github.com/login/device");
    assert_eq!(g.expires_in.as_secs(), 899);
    assert_eq!(g.interval.as_secs(), 5);
}

#[test]
fn non_github_verification_uri_is_refused() {
    // Displaying an attacker-influenceable URL under our chrome is a phishing
    // primitive; anything but https://github.com/… ends the flow.
    for uri in [
        "http://github.com/login/device",
        "https://github.evil.com/x",
    ] {
        let body = GRANT.replace("https://github.com/login/device", uri);
        let err = parse_device_code(200, &body).err().expect("must refuse");
        assert!(err.contains("refusing"), "{uri}: {err}");
    }
}

#[test]
fn device_flow_disabled_is_named_for_the_maintainer() {
    let err = parse_device_code(200, r#"{"error":"device_flow_disabled"}"#)
        .err()
        .expect("err");
    assert!(err.contains("device flow disabled"), "{err}");
}

#[test]
fn missing_fields_and_non_json_fail_with_the_status() {
    // The response shapes are GitHub's to change; a missing `interval` must
    // not invent a poll cadence (PCND), it must fail.
    assert!(parse_device_code(200, r#"{"user_code":"X"}"#).is_err());
    let err = parse_device_code(502, "<html>bad gateway</html>")
        .err()
        .expect("502 fails");
    assert!(err.contains("502"));
}

// ── Poll parsing ──

#[test]
fn poll_outcomes_map_one_to_one() {
    assert!(matches!(
        parse_poll(200, r#"{"error":"authorization_pending"}"#),
        PollOutcome::Pending
    ));
    assert!(matches!(
        parse_poll(200, r#"{"error":"slow_down","interval":10}"#),
        PollOutcome::SlowDown
    ));
    assert!(matches!(
        parse_poll(200, r#"{"error":"expired_token"}"#),
        PollOutcome::Expired
    ));
    assert!(matches!(
        parse_poll(200, r#"{"error":"access_denied"}"#),
        PollOutcome::Denied
    ));
    assert!(matches!(parse_poll(200, "not json"), PollOutcome::Fatal(_)));
    assert!(matches!(
        parse_poll(200, r#"{"ok":true}"#),
        PollOutcome::Fatal(_)
    ));
}

#[test]
fn authorized_poll_carries_both_tokens() {
    let out = parse_poll(
        200,
        r#"{"access_token":"ghu_AAA","expires_in":28800,"refresh_token":"ghr_BBB","token_type":"bearer"}"#,
    );
    let PollOutcome::Authorized { access, refresh } = out else {
        panic!("expected Authorized");
    };
    assert_eq!(access.expose(), "ghu_AAA");
    assert_eq!(refresh.expect("refresh").expose(), "ghr_BBB");
}

#[test]
fn authorized_poll_without_refresh_still_authorizes() {
    // An app with token expiration opted out issues no refresh token; the
    // submit path treats that as "401 is terminal" rather than an error here.
    let out = parse_poll(200, r#"{"access_token":"ghu_AAA","token_type":"bearer"}"#);
    let PollOutcome::Authorized { refresh, .. } = out else {
        panic!("expected Authorized");
    };
    assert!(refresh.is_none());
}

#[test]
fn refresh_parse_only_accepts_a_token() {
    assert!(parse_refresh(200, r#"{"access_token":"ghu_C"}"#).is_some());
    assert!(parse_refresh(200, r#"{"error":"bad_refresh_token"}"#).is_none());
    assert!(parse_refresh(401, "").is_none());
}

// ── Issue response parsing + the failure table ──

#[test]
fn only_a_201_is_created() {
    let ok = parse_issue_created(
        201,
        r#"{"number":214,"html_url":"https://github.com/o/r/issues/214"}"#,
    );
    let (n, url) = ok.expect("created");
    assert_eq!(n, 214);
    assert_eq!(url, "https://github.com/o/r/issues/214");
    // A 200 is NOT creation — the draft is cleared only on this parse, so
    // leniency here is what would lose a report.
    assert!(parse_issue_created(200, r#"{"number":1}"#).is_err());
}

#[test]
fn a_201_with_an_unreadable_body_still_counts_as_created() {
    // Claiming failure for an issue that exists invites a duplicate post.
    let (n, url) = parse_issue_created(201, "garbage").expect("created");
    assert_eq!(n, 0);
    assert!(url.is_empty());
}

fn failure(status: u16, body: &str) -> IssueFailure {
    match parse_issue_created(status, body) {
        Err(f) => f,
        Ok(_) => panic!("HTTP {status} must not parse as created"),
    }
}

#[test]
fn the_failure_table_row_by_row() {
    let repo = "owner/atlas";
    let msg = |s, b, ra| describe_issue_failure(&failure(s, b), repo, ra);
    assert_eq!(
        msg(403, r#"{"message":"API rate limit exceeded"}"#, Some(30)),
        "GitHub is rate-limiting — retry in 30s"
    );
    assert_eq!(
        msg(429, "{}", Some(7)),
        "GitHub is rate-limiting — retry in 7s"
    );
    assert_eq!(
        msg(403, r#"{"message":"secondary rate limit hit"}"#, None),
        "GitHub is rate-limiting — retry shortly"
    );
    assert!(msg(403, r#"{"message":"Resource not accessible"}"#, None).contains("403"));
    assert!(msg(404, "{}", None).contains("not installed"));
    assert!(msg(410, "{}", None).contains("archived or disabled"));
    assert_eq!(
        msg(
            422,
            r#"{"message":"Validation Failed","errors":[{"message":"body is too long"}]}"#,
            None
        ),
        "GitHub rejected the issue: Validation Failed: body is too long"
    );
    assert_eq!(
        msg(503, "", None),
        "GitHub returned 503 — try again shortly"
    );
    assert!(msg(418, "{}", None).contains("418"));
}

// ── Body assembly ──

#[test]
fn composed_body_carries_text_env_marker_in_order() {
    let c = compose_body("it broke", "Atlas 1.0 · c0ffee · linux/aarch64", None, None)
        .expect("compose");
    let text = c.body.find("it broke").expect("user text");
    let env = c.body.find("## Environment").expect("env");
    let marker = c.body.find(MARKER).expect("marker");
    assert!(text < env && env < marker, "{}", c.body);
    assert_eq!(c.logs_total, 0);
    assert_eq!(c.chars, c.body.chars().count());
}

#[test]
fn logs_ride_inside_a_wide_enough_fence() {
    let logs = vec![
        "clean line".to_string(),
        "evil ``` fence escape".to_string(),
    ];
    let c = compose_body("t", "env", Some(&logs), Some("/tee")).expect("compose");
    // The content's ``` run forces a 4-backtick fence, so the log cannot
    // break out and render as markdown in the public issue.
    assert!(c.body.contains("````text"), "{}", c.body);
    assert!(c.body.contains("evil ``` fence escape"));
    assert_eq!(c.logs_included, 2);
    assert_eq!(c.logs_total, 2);
}

#[test]
fn oversize_logs_trim_but_oversize_user_text_refuses() {
    let logs: Vec<String> = (0..5000)
        .map(|i| format!("{i} {}", "y".repeat(40)))
        .collect();
    let c = compose_body("short", "env", Some(&logs), Some("/tee")).expect("compose");
    assert!(c.logs_included < c.logs_total);
    assert!(
        c.body.contains("earlier lines omitted"),
        "the trim must be visible"
    );
    assert!(c.chars <= super::super::redact::BODY_BUDGET);

    // The log tail is ours to trim; the user's words are not.
    let huge = "w".repeat(super::super::redact::BODY_BUDGET + 1);
    let err = compose_body(&huge, "env", None, None).expect_err("refuse");
    assert!(err.contains("trim"), "{err}");
}

#[test]
fn assembled_body_always_fits_github() {
    // Property over sizes near the budget edge: whatever the log volume, the
    // composed body must clear GitHub's hard limit.
    for n in [0usize, 1, 100, 2_000, 20_000] {
        let logs: Vec<String> = (0..n)
            .map(|i| format!("line {i} {}", "z".repeat(i % 97)))
            .collect();
        let c = compose_body("report", "env", Some(&logs), None).expect("compose");
        assert!(
            c.chars <= super::super::redact::GITHUB_BODY_LIMIT,
            "n={n}: {}",
            c.chars
        );
    }
}

// ── SecretString ──

#[test]
fn secret_is_reachable_only_through_expose() {
    let s = SecretString::new("ghu_TOP".into());
    assert_eq!(s.expose(), "ghu_TOP");
    let c = s.clone();
    assert_eq!(c.expose(), "ghu_TOP");
    // No Debug/Display/Serialize impls exist — `format!("{:?}", s)` fails to
    // compile, which is the guard (checked here by the code not existing to
    // write, not by a runtime assertion).
}

#[test]
fn env_line_names_the_unknowns_explicitly() {
    let line = env_line("", false);
    assert!(line.contains("model: none"), "{line}");
    assert!(line.contains("engine ready: false"), "{line}");
    let line = env_line("Qwen3.6-27B", true);
    assert!(line.contains("model: Qwen3.6-27B"), "{line}");
}
