// SPDX-License-Identifier: AGPL-3.0-only
//
// Process-local ID and timestamp helpers shared by every API surface and
// the internal stores. Provider-neutral: wire prefixes (`chatcmpl-`,
// `msg_`, `resp_`, `conv_`, `req_`) are applied by the callers that own
// the respective wire formats.

/// Read random bytes from the OS (Linux: getrandom syscall).
pub(crate) fn getrandom(buf: &mut [u8]) -> Result<(), ()> {
    use std::fs::File;
    use std::io::Read;
    File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(buf))
        .map_err(|_| ())
}

/// UUID v4 generation using OS randomness (no external crate needed).
pub(crate) fn uuid_v4() -> String {
    let mut bytes = [0u8; 16];
    if let Ok(()) = getrandom(&mut bytes) {
        bytes[6] = (bytes[6] & 0x0F) | 0x40; // version 4
        bytes[8] = (bytes[8] & 0x3F) | 0x80; // variant 1
    } else {
        // Fallback: nanosecond timestamp (unique but not random).
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        bytes = t.to_le_bytes();
    }
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0],
        bytes[1],
        bytes[2],
        bytes[3],
        bytes[4],
        bytes[5],
        bytes[6],
        bytes[7],
        bytes[8],
        bytes[9],
        bytes[10],
        bytes[11],
        bytes[12],
        bytes[13],
        bytes[14],
        bytes[15],
    )
}

/// Seconds since the Unix epoch (`created` fields on wire responses).
pub(crate) fn unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
