// SPDX-License-Identifier: AGPL-3.0-only

//! Issue reporting: the GitHub protocol as PURE functions — response parsing,
//! failure mapping, body assembly — with no I/O in this file. The two HTTPS
//! calls live behind [`report_http`](super::report_http)'s transport seam, so
//! every row of the failure table is unit-testable without a network.
//!
//! # Auth model (why device flow, and why nothing is stored)
//!
//! The TUI is a public client: anything compiled into it is readable by every
//! user, so no client secret can exist here. GitHub's device flow is the OAuth
//! grant built for that — `client_id` only, the user types a short code into
//! github.com, and the resulting user token acts only on repos the GitHub App
//! is installed on (`Issues: write` + `Metadata: read`, nothing else).
//!
//! The access and refresh tokens live in [`SecretString`]s in process memory
//! for the process lifetime and nowhere else — never written to disk, never
//! logged, never in the issue body (CWE-522/CWE-256). These are shared boxes:
//! a token file would be readable by every session running as this user, now
//! and from backups. The cost is one browser authorization per server process
//! that reports an issue; the 8-hour token expiry plus the in-memory refresh
//! token covers the rest of the process's life without another round-trip.

use std::time::Duration;

/// The official build's identity. PUBLIC by design — the client_id travels in
/// every device-flow POST — but still identity: forks override with
/// `ATLAS_REPORT_CLIENT_ID` / `ATLAS_REPORT_REPO` rather than shipping issues
/// into the upstream tracker.
pub const OFFICIAL_CLIENT_ID: &str = "Iv23liAv6nlb4RaYaJSp";
pub const OFFICIAL_REPO: &str = "Avarok-Cybersecurity/atlas";

/// Hidden marker a repo Action keys the `tui-report` label on. The API
/// silently DROPS `labels` sent by users without push access, so the app
/// cannot set the label itself — the body carries the signal instead.
pub const MARKER: &str = "<!-- atlas-tui-report -->";

pub const DEVICE_CODE_URL: &str = "https://github.com/login/device/code";
pub const TOKEN_URL: &str = "https://github.com/login/oauth/access_token";

pub fn issues_url(repo: &str) -> String {
    format!("https://api.github.com/repos/{repo}/issues")
}

/// What a failed submit tells the user when the transport (DNS, TLS, refused
/// connection) is what failed. One constant so the auth and submit paths
/// cannot drift apart in wording.
pub const NETWORK_FAILED: &str = "could not reach github.com — check network and retry (s)";

pub const NOT_CONFIGURED: &str = "issue reporting is not configured for this build (set ATLAS_REPORT_CLIENT_ID / ATLAS_REPORT_REPO)";

// ── Secrets ──

/// An in-memory secret. Deliberately implements neither `Debug` nor `Display`
/// nor `Serialize`: a `{:?}` on it fails to COMPILE, which is the cheapest
/// possible guard against the token reaching a `tracing` event or a format
/// string (CWE-532). Zeroed on drop as scrub hygiene — best-effort, since
/// moves can leave earlier copies, but strictly better than leaving the bytes.
pub struct SecretString(String);

impl SecretString {
    pub fn new(s: String) -> Self {
        Self(s)
    }
    /// The one way to read the value. Named so call sites are greppable.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl Clone for SecretString {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl Drop for SecretString {
    fn drop(&mut self) {
        // NUL bytes are valid UTF-8, so overwriting in place is safe.
        unsafe { self.0.as_mut_vec() }.fill(0);
    }
}

// ── Target resolution ──

pub struct Target {
    pub client_id: String,
    pub repo: String,
}

/// Where reports go for THIS process: env overrides, else the compiled-in
/// identity.
pub fn target() -> Result<Target, &'static str> {
    target_from(
        std::env::var("ATLAS_REPORT_CLIENT_ID").ok(),
        std::env::var("ATLAS_REPORT_REPO").ok(),
    )
}

/// Set-but-empty overrides — and a fork that blanked the constants — fail
/// loudly with [`NOT_CONFIGURED`] rather than half-working: a device-flow POST
/// with an empty client_id would produce a GitHub error page the user cannot
/// act on.
pub fn target_from(id: Option<String>, repo: Option<String>) -> Result<Target, &'static str> {
    let client_id = id.unwrap_or_else(|| OFFICIAL_CLIENT_ID.to_string());
    let repo = repo.unwrap_or_else(|| OFFICIAL_REPO.to_string());
    if client_id.trim().is_empty() || repo.trim().is_empty() {
        return Err(NOT_CONFIGURED);
    }
    Ok(Target { client_id, repo })
}

// ── Events crossing back to the render thread ──

/// What the worker threads send home. No variant carries the device code —
/// it is the secret half of the grant while pending and stays inside the
/// worker (the user code, which IS meant to be displayed, is a different
/// string).
pub enum ReportEvent {
    CodeReady {
        user_code: String,
        verification_uri: String,
        expires_in: Duration,
    },
    Authorized {
        access: SecretString,
        refresh: Option<SecretString>,
    },
    AuthFailed {
        message: String,
    },
    Created {
        number: u64,
        url: String,
    },
    SubmitFailed {
        message: String,
        /// The stored tokens are known-bad (401 that a refresh could not
        /// cure); the state drops them so the next attempt re-authorizes
        /// instead of replaying a dead token.
        drop_auth: bool,
    },
}

// ── Response parsing (pure) ──

/// The device-code grant. `device_code` never leaves this struct's owners.
pub struct DeviceGrant {
    pub device_code: String,
    pub user_code: String,
    pub verification_uri: String,
    pub expires_in: Duration,
    pub interval: Duration,
}

fn json(body: &str) -> Option<serde_json::Value> {
    serde_json::from_str(body).ok()
}

fn str_field(v: &serde_json::Value, k: &str) -> Option<String> {
    v.get(k)?.as_str().map(str::to_string)
}

/// Parse `POST /login/device/code`. Every field is required — a response
/// missing `interval` would otherwise poll at some invented cadence GitHub
/// never agreed to (PCND: no implicit defaults).
pub fn parse_device_code(status: u16, body: &str) -> Result<DeviceGrant, String> {
    let Some(v) = json(body) else {
        return Err(format!(
            "unexpected response from github.com (HTTP {status})"
        ));
    };
    if let Some(err) = str_field(&v, "error") {
        return Err(map_oauth_error(&err, &v));
    }
    let grant = (|| {
        Some(DeviceGrant {
            device_code: str_field(&v, "device_code")?,
            user_code: str_field(&v, "user_code")?,
            verification_uri: str_field(&v, "verification_uri")?,
            expires_in: Duration::from_secs(v.get("expires_in")?.as_u64()?),
            interval: Duration::from_secs(v.get("interval")?.as_u64()?),
        })
    })();
    let Some(grant) = grant else {
        return Err(format!(
            "github.com answered without the device-code fields (HTTP {status})"
        ));
    };
    // The URI is about to be shown to a human as "go here and type the code".
    // Anything but https on github.com is refused outright — displaying an
    // attacker-influenceable URL under our chrome is a phishing primitive.
    if !grant.verification_uri.starts_with("https://github.com/") {
        return Err("unexpected verification URL in GitHub's response — refusing".to_string());
    }
    Ok(grant)
}

/// One poll of `POST /login/oauth/access_token`.
pub enum PollOutcome {
    Authorized {
        access: SecretString,
        refresh: Option<SecretString>,
    },
    /// Keep polling at the agreed interval.
    Pending,
    /// Keep polling, 5 seconds slower — GitHub's documented backoff signal.
    SlowDown,
    /// The 15-minute code lifetime ran out.
    Expired,
    /// The user clicked cancel on github.com.
    Denied,
    /// Anything that ends the flow with a message.
    Fatal(String),
}

pub fn parse_poll(status: u16, body: &str) -> PollOutcome {
    let Some(v) = json(body) else {
        return PollOutcome::Fatal(format!(
            "unexpected response from github.com while polling (HTTP {status})"
        ));
    };
    if let Some(access) = str_field(&v, "access_token") {
        return PollOutcome::Authorized {
            access: SecretString::new(access),
            refresh: str_field(&v, "refresh_token").map(SecretString::new),
        };
    }
    match str_field(&v, "error").as_deref() {
        Some("authorization_pending") => PollOutcome::Pending,
        Some("slow_down") => PollOutcome::SlowDown,
        Some("expired_token") => PollOutcome::Expired,
        Some("access_denied") => PollOutcome::Denied,
        Some(err) => PollOutcome::Fatal(map_oauth_error(err, &v)),
        None => PollOutcome::Fatal(format!(
            "github.com answered the poll without a token or an error (HTTP {status})"
        )),
    }
}

/// The refresh grant answers in the same shape as the poll; only the terminal
/// outcomes are meaningful — `None` is "the refresh token is dead", whatever
/// the wording.
pub fn parse_refresh(status: u16, body: &str) -> Option<(SecretString, Option<SecretString>)> {
    match parse_poll(status, body) {
        PollOutcome::Authorized { access, refresh } => Some((access, refresh)),
        _ => None,
    }
}

fn map_oauth_error(err: &str, v: &serde_json::Value) -> String {
    match err {
        // The single App-settings toggle this whole feature depends on.
        "device_flow_disabled" => {
            "this build's GitHub App is misconfigured (device flow disabled) — report to the maintainers"
                .to_string()
        }
        "unsupported_grant_type" | "incorrect_client_credentials" | "incorrect_device_code" => {
            format!("GitHub refused the authorization request ({err})")
        }
        other => {
            let detail = str_field(v, "error_description").unwrap_or_default();
            if detail.is_empty() {
                format!("GitHub authorization failed ({other})")
            } else {
                format!("GitHub authorization failed: {detail}")
            }
        }
    }
}

/// Parse `POST /repos/{owner}/{repo}/issues`. `Ok` is a 201 and nothing else —
/// the composer state is cleared only on this value, so a lenient parse here
/// would be what loses a user's draft.
pub fn parse_issue_created(status: u16, body: &str) -> Result<(u64, String), IssueFailure> {
    if status == 201 {
        let v = json(body);
        let number = v.as_ref().and_then(|v| v.get("number")?.as_u64());
        let url = v.as_ref().and_then(|v| str_field(v, "html_url"));
        return match (number, url) {
            (Some(n), Some(u)) => Ok((n, u)),
            // A 201 whose body we cannot read still created the issue;
            // claiming failure would invite a duplicate submission.
            _ => Ok((0, String::new())),
        };
    }
    Err(IssueFailure {
        status,
        message: github_message(body),
    })
}

#[derive(Debug)]
pub struct IssueFailure {
    pub status: u16,
    pub message: String,
}

fn github_message(body: &str) -> String {
    json(body)
        .and_then(|v| {
            let msg = str_field(&v, "message")?;
            let detail = v
                .get("errors")
                .and_then(|e| e.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|e| e.get("message").and_then(|m| m.as_str()))
                        .collect::<Vec<_>>()
                        .join("; ")
                })
                .unwrap_or_default();
            Some(if detail.is_empty() {
                msg
            } else {
                format!("{msg}: {detail}")
            })
        })
        .unwrap_or_default()
}

/// Every non-201 outcome as words the user can act on. 401 is NOT mapped here:
/// the submit worker owns it, because the right response is a silent refresh,
/// not a message.
pub fn describe_issue_failure(f: &IssueFailure, repo: &str, retry_after: Option<u64>) -> String {
    match f.status {
        403 | 429 => match retry_after {
            Some(n) => format!("GitHub is rate-limiting — retry in {n}s"),
            None if f.message.to_lowercase().contains("rate limit") => {
                "GitHub is rate-limiting — retry shortly".to_string()
            }
            None => format!(
                "GitHub refused the request (403): {}",
                or_unstated(&f.message)
            ),
        },
        404 => format!(
            "GitHub answered 404 for {repo} — the repository may not exist, or the reporter app is not installed on it"
        ),
        410 => {
            format!("the issue tracker at {repo} is archived or disabled — it cannot accept issues")
        }
        422 => format!("GitHub rejected the issue: {}", or_unstated(&f.message)),
        s if (500..600).contains(&s) => format!("GitHub returned {s} — try again shortly"),
        s => format!(
            "GitHub returned an unexpected {s}: {}",
            or_unstated(&f.message)
        ),
    }
}

fn or_unstated(msg: &str) -> &str {
    if msg.is_empty() {
        "(no detail given)"
    } else {
        msg
    }
}

// ── Body assembly (pure) ──

/// The final issue body, plus the numbers the preview states about it.
#[derive(Clone, Debug)]
pub struct Composed {
    pub body: String,
    pub chars: usize,
    pub logs_included: usize,
    pub logs_total: usize,
}

/// The `## Environment` footer line. `commit unknown` is explicit rather than
/// omitted: a maintainer reading the issue needs to see that the build carried
/// no commit stamp, not wonder whether the reporter deleted the line.
pub fn env_line(model: &str, engine_ready: bool) -> String {
    format!(
        "Atlas {} · {} · {}/{} · model: {} · engine ready: {engine_ready}",
        crate::cli::ATLAS_VERSION,
        option_env!("ATLAS_BUILD_COMMIT").unwrap_or("commit unknown"),
        std::env::consts::OS,
        std::env::consts::ARCH,
        if model.is_empty() { "none" } else { model },
    )
}

/// Assemble the exact body that will be posted — the same function feeds the
/// preview and the POST, so the preview cannot show one thing and ship
/// another. `logs` must already be redacted; this function only budgets.
///
/// Refuses (never truncates) when the user's own text cannot fit: the log
/// tail is ours to trim, the user's words are not.
pub fn compose_body(
    user_text: &str,
    env: &str,
    logs: Option<&[String]>,
    tee_path: Option<&str>,
) -> Result<Composed, String> {
    use super::redact::{BODY_BUDGET, GITHUB_BODY_LIMIT, fence_for, trim_to_budget};
    let head = format!("{}\n\n## Environment\n\n{env}\n", user_text.trim_end());
    let tail = format!("\n{MARKER}\n");
    // Flat reserve for the log section's own chrome (heading + fences). 200
    // covers a heading and two fences a dozen backticks wide; the final check
    // below still refuses if some pathological content beats it.
    const SECTION_RESERVE: usize = 200;
    let fixed = head.chars().count() + tail.chars().count();
    if fixed + if logs.is_some() { SECTION_RESERVE } else { 0 } > BODY_BUDGET {
        return Err(format!(
            "the report text is {fixed} characters; GitHub's limit is {GITHUB_BODY_LIMIT} and the dashboard reserves headroom — trim it below {BODY_BUDGET}",
        ));
    }
    let (section, logs_included, logs_total) = match logs {
        None => (String::new(), 0, 0),
        Some(lines) => {
            let trimmed = trim_to_budget(lines, BODY_BUDGET - fixed - SECTION_RESERVE, tee_path);
            let fence = fence_for(&trimmed.text);
            let section = format!(
                "\n## Server log (last {} of {} lines, redacted best-effort)\n\n{fence}text\n{}\n{fence}\n",
                trimmed.included, trimmed.total, trimmed.text
            );
            (section, trimmed.included, trimmed.total)
        }
    };
    let body = format!("{head}{section}{tail}");
    let chars = body.chars().count();
    if chars > GITHUB_BODY_LIMIT {
        // Unreachable by construction; refusing loudly beats a 422 after the
        // user already pressed send.
        return Err(format!(
            "assembled body is {chars} characters — over GitHub's limit"
        ));
    }
    Ok(Composed {
        body,
        chars,
        logs_included,
        logs_total,
    })
}

#[cfg(test)]
#[path = "report_tests.rs"]
mod tests;
