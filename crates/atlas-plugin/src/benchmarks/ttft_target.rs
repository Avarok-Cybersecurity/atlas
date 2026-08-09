// SPDX-License-Identifier: AGPL-3.0-only

//! Which endpoints count as the SAME BOX for a box-local baseline.
//!
//! Split out of `ttft.rs` to keep that file under the repo's 500-line cap.

/// Do two endpoint URLs name the same BOX?
///
/// Host only — the port is deliberately ignored. TTFT is box-local, so a
/// baseline from another machine must never be compared against; but a
/// self-started gate run binds an EPHEMERAL port by design, so keying
/// comparability on the full URL meant it could never match the baseline it
/// had just recorded on that same box. The gate then degraded to "reporting
/// only" every single time — silently, and in the safe-looking direction,
/// which is the worst kind: a guard that always abstains never fails.
///
/// Loopback spellings are one box: `localhost`, `127.0.0.1` and `[::1]` all
/// name this machine.
pub(super) fn same_box(a: &str, b: &str) -> bool {
    fn host(url: &str) -> String {
        let rest = url.split("://").nth(1).unwrap_or(url);
        let hostport = rest.split('/').next().unwrap_or(rest);
        // Strip the port, taking care not to cut an unbracketed IPv6 literal.
        let h = match hostport.rfind(':') {
            Some(i) if !hostport[i + 1..].contains(':') => &hostport[..i],
            _ => hostport,
        };
        let h = h.trim_start_matches('[').trim_end_matches(']');
        match h {
            "localhost" | "127.0.0.1" | "::1" => "localhost".to_string(),
            other => other.to_ascii_lowercase(),
        }
    }
    host(a) == host(b)
}

#[cfg(test)]
mod tests {
    use super::same_box;

    #[test]
    fn an_ephemeral_port_is_still_the_same_box() {
        // The regression: a self-started gate run binds a random port, so a
        // full-URL comparison never matched and the guard abstained forever.
        assert!(same_box("http://127.0.0.1:8888", "http://127.0.0.1:33033"));
        assert!(same_box("http://localhost:8888", "http://127.0.0.1:41999"));
    }

    #[test]
    fn another_machine_is_never_the_same_box() {
        // TTFT is box-local; comparing across boxes manufactures wins.
        assert!(!same_box("http://10.10.10.3:8888", "http://127.0.0.1:8888"));
        assert!(!same_box(
            "http://10.10.10.1:8888",
            "http://10.10.10.2:8888"
        ));
    }

    #[test]
    fn an_ipv6_literal_keeps_its_address() {
        assert!(same_box("http://[::1]:8888", "http://localhost:9"));
        assert!(!same_box("http://[fe80::1]:8888", "http://[fe80::2]:8888"));
    }
}
