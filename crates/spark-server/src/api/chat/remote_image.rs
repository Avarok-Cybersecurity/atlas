// SPDX-License-Identifier: AGPL-3.0-only

//! Server-side fetching of `image_url` content parts, off by default.
//!
//! # Why this is opt-in when the house rule says default-ON with a kill-switch
//!
//! That rule exists so a feature is not left dark behind a flag nobody sets.
//! It does not apply to acquiring a capability the server did not previously
//! have. Turning this on makes the inference server issue outbound HTTP to
//! addresses chosen by whoever can send it a chat request — a server-side
//! request forgery primitive. A deployment that never wanted that must not
//! acquire it by upgrading, so the default stays REJECT and the operator opts
//! in per deployment. The rejection is a clear 400 naming the flag, not a
//! silent drop, so the capability is discoverable rather than dark.
//!
//! # What it defends against
//!
//! Even switched on, the fetch is bounded on every axis an attacker controls:
//!
//! - **Address**: link-local, loopback, private and unique-local destinations
//!   are refused. `169.254.169.254` is the cloud instance-metadata endpoint,
//!   and reaching it from inside the request path is the canonical SSRF
//!   escalation — credentials come back as a perfectly ordinary HTTP body.
//!   `--vision-remote-image-allow-private` re-permits them for deployments
//!   whose image host genuinely is internal.
//! - **Size**: capped, and enforced while READING rather than by trusting
//!   `Content-Length`, which the remote controls and can understate.
//! - **Time**: capped, so a slow-loris response cannot pin a blocking thread
//!   from the prepare pool for the lifetime of the process.
//! - **Redirects**: followed a bounded number of times, and every hop is
//!   re-checked — a public URL that 302s to `127.0.0.1` is the standard way
//!   to walk past an address check applied only to the first request.
//! - **Type**: the response must declare an image content type.
//!
//! # Where it runs
//!
//! Inside `prepare_chat_prompt`, which the chat handler already calls under
//! `tokio::task::spawn_blocking` (see `api/chat/mod.rs`), so a blocking client
//! is the correct choice and no async plumbing is involved. `ureq` is already
//! a dependency and already used this way by `model_download/hf.rs`.

use std::io::Read;
use std::net::IpAddr;

/// Operator policy for remote image fetching. `enabled: false` is the default
/// and the only state in which no outbound request can originate here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RemoteImagePolicy {
    pub enabled: bool,
    pub max_bytes: usize,
    pub timeout_secs: u64,
    /// Permit loopback/private/link-local destinations. Separate from
    /// `enabled` because "fetch from the public internet" and "fetch from
    /// inside my network" are different grants with different blast radii.
    pub allow_private: bool,
}

impl Default for RemoteImagePolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            max_bytes: 20 * 1024 * 1024,
            timeout_secs: 10,
            allow_private: false,
        }
    }
}

/// Hops followed before giving up. Enough for the CDN shortener chains real
/// image URLs use, few enough that a redirect loop terminates quickly.
const MAX_REDIRECTS: u32 = 4;

/// Is this address one that a request from the public internet should never be
/// able to make us reach?
///
/// Deliberately covers more than "private": loopback reaches services bound to
/// localhost that assume they are unreachable, and link-local reaches cloud
/// instance metadata.
pub fn is_blocked_address(ip: IpAddr) -> bool {
    // Normalise IPv4-mapped IPv6 FIRST. `::ffff:127.0.0.1` is loopback wearing
    // a v6 hat, and checking it as a v6 address walks straight past every
    // v4 predicate below.
    let ip = match ip {
        IpAddr::V6(v6) => v6.to_ipv4_mapped().map_or(ip, IpAddr::V4),
        v4 => v4,
    };
    match ip {
        IpAddr::V4(v4) => {
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_broadcast()
                || v4.is_documentation()
                || v4.is_unspecified()
                // 100.64.0.0/10, carrier-grade NAT — routable-looking but not
                // public, and `is_private` does not cover it.
                || (v4.octets()[0] == 100 && (64..128).contains(&v4.octets()[1]))
        }
        IpAddr::V6(v6) => {
            v6.is_loopback()
                || v6.is_unspecified()
                // fc00::/7 unique-local and fe80::/10 link-local. Neither has
                // a stable predicate on stable Rust, so the prefixes are
                // matched directly.
                || (v6.segments()[0] & 0xfe00) == 0xfc00
                || (v6.segments()[0] & 0xffc0) == 0xfe80
        }
    }
}

/// Resolve `host` and refuse if ANY resolved address is blocked.
///
/// Any, not all: a name that resolves to both a public and a loopback address
/// is a DNS-rebinding shape, and picking the public one would be trusting the
/// resolver to hand back the same answer twice.
fn check_host(host: &str, port: u16, policy: &RemoteImagePolicy) -> Result<(), String> {
    if policy.allow_private {
        return Ok(());
    }
    // A literal IP in the URL never reaches the resolver, so parse first.
    if let Ok(ip) = host.parse::<IpAddr>() {
        return if is_blocked_address(ip) {
            Err(format!("{ip} is a loopback/private/link-local address"))
        } else {
            Ok(())
        };
    }
    use std::net::ToSocketAddrs;
    let addrs = (host, port)
        .to_socket_addrs()
        .map_err(|e| format!("could not resolve {host}: {e}"))?;
    let mut saw = false;
    for a in addrs {
        saw = true;
        if is_blocked_address(a.ip()) {
            return Err(format!(
                "{host} resolves to {}, a loopback/private/link-local address",
                a.ip()
            ));
        }
    }
    if saw {
        Ok(())
    } else {
        Err(format!("{host} resolved to no addresses"))
    }
}

/// Split a URL into (scheme, host, port) without pulling in a URL crate.
fn split_url(url: &str) -> Result<(&str, &str, u16), String> {
    let (scheme, rest) = url
        .split_once("://")
        .ok_or_else(|| "not an absolute http(s) URL".to_string())?;
    if scheme != "http" && scheme != "https" {
        return Err(format!("scheme {scheme:?} is not http or https"));
    }
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    // Strip userinfo: `http://metadata@evil/` points at `evil`, and a check
    // that read the part before `@` as the host would check the wrong string.
    let authority = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
    let default_port = if scheme == "https" { 443 } else { 80 };
    let (host, port) = match authority.rfind(':') {
        // Not a port separator if it is inside a bracketed IPv6 literal.
        Some(i) if !authority[i..].contains(']') => (
            &authority[..i],
            authority[i + 1..].parse().unwrap_or(default_port),
        ),
        _ => (authority, default_port),
    };
    let host = host.trim_start_matches('[').trim_end_matches(']');
    if host.is_empty() {
        return Err("URL has no host".to_string());
    }
    Ok((scheme, host, port))
}

/// Fetch `url` and return it as a `data:` URI the preprocessor can decode.
///
/// `Err` carries an operator-readable reason; the caller turns it into a 400.
pub fn fetch_as_data_uri(url: &str, policy: &RemoteImagePolicy) -> Result<String, String> {
    if !policy.enabled {
        return Err("remote image fetching is disabled".to_string());
    }

    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_secs(policy.timeout_secs)))
        // Redirects are followed MANUALLY below so each hop's address can be
        // checked. Letting the client follow them would check only hop zero.
        .max_redirects(0)
        .build()
        .into();

    let mut current = url.to_string();
    for _ in 0..=MAX_REDIRECTS {
        let (_scheme, host, port) = split_url(&current)?;
        check_host(host, port, policy)?;

        let resp = agent
            .get(&current)
            .call()
            .map_err(|e| format!("fetch failed: {e}"))?;
        let status = resp.status().as_u16();

        if (300..400).contains(&status) {
            let loc = resp
                .headers()
                .get("location")
                .and_then(|v| v.to_str().ok())
                .ok_or_else(|| format!("HTTP {status} with no Location header"))?;
            // Relative redirects are not resolved: doing it correctly needs a
            // URL joiner, and an image host that only emits relative hops is
            // rare enough to be worth an honest error over a hand-rolled one.
            if !loc.starts_with("http://") && !loc.starts_with("https://") {
                return Err(format!("relative redirect to {loc:?} is not followed"));
            }
            current = loc.to_string();
            continue;
        }
        if status != 200 {
            return Err(format!("HTTP {status}"));
        }

        let ctype = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let mime = ctype.split(';').next().unwrap_or("").trim().to_lowercase();
        if !mime.starts_with("image/") {
            return Err(format!("content-type {mime:?} is not an image"));
        }

        // Read one byte past the cap so the overrun is DETECTED rather than
        // silently truncated into a corrupt image. Content-Length is not
        // consulted: the remote controls it and may understate it.
        let mut buf = Vec::new();
        let cap = policy.max_bytes;
        resp.into_body()
            .into_reader()
            .take(cap as u64 + 1)
            .read_to_end(&mut buf)
            .map_err(|e| format!("read failed: {e}"))?;
        if buf.len() > cap {
            return Err(format!(
                "image exceeds the {cap}-byte cap (--vision-remote-image-max-mb)"
            ));
        }
        if buf.is_empty() {
            return Err("fetched an empty body".to_string());
        }

        use base64::Engine as _;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&buf);
        return Ok(format!("data:{mime};base64,{b64}"));
    }
    Err(format!("more than {MAX_REDIRECTS} redirects"))
}

#[cfg(test)]
#[path = "remote_image_tests.rs"]
mod tests;
