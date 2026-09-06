// SPDX-License-Identifier: AGPL-3.0-only
//! Who a request is, for rate-limiting purposes.
//!
//! Split out of `rate_limiter.rs`: identity resolution answers "who is this",
//! the parent answers "how fast may they go", and this stack pushed the file
//! from 457 to 507 lines against a 500 cap. A real separation rather than a
//! line-count trick — byte-exact move.

/// Resolve a stable identity from the request headers + peer addr. Used by
/// the axum middleware; exposed here so tests can reuse it.
pub fn extract_identity(
    headers: &axum::http::HeaderMap,
    peer: Option<std::net::SocketAddr>,
) -> String {
    use axum::http::header;
    // 1. Bearer token.
    if let Some(v) = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        && let Some(tok) = v.strip_prefix("Bearer ")
    {
        let tok = tok.trim();
        if !tok.is_empty() {
            // Hash the token so we don't retain sensitive data in the map
            // keys or Prometheus labels (if ever exposed).
            return format!("bearer:{}", hash_token(tok));
        }
    }
    // 2. X-Forwarded-For.
    if let Some(xff) = headers.get("x-forwarded-for").and_then(|v| v.to_str().ok())
        && let Some(first) = xff.split(',').next()
    {
        let first = first.trim();
        if !first.is_empty() {
            return format!("xff:{first}");
        }
    }
    // 3. Peer socket.
    match peer {
        Some(addr) => format!("peer:{}", addr.ip()),
        None => "peer:unknown".to_string(),
    }
}

/// FNV-1a 64-bit hash. Avoids pulling a crypto dep; unnecessary here since
/// we only need a stable opaque label, not collision resistance.
fn hash_token(tok: &str) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in tok.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}
