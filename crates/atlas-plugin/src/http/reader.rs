// SPDX-License-Identifier: AGPL-3.0-only

//! The chunked/content-length response reader used by the HTTP client.

use anyhow::{Result, anyhow, bail};

use super::{
    MAX_ERROR_BODY, content_length, find, is_chunked, message_from_body, status_is_success,
};

/// Incremental HTTP response reader: status/header parse, chunked decode, then
/// complete lines out. Owns the only place framing can go wrong.
#[derive(Default)]
pub(super) struct Reader {
    raw: Vec<u8>,
    header_end: Option<usize>,
    chunked: bool,
    pub(super) body: Vec<u8>,
    consumed: usize,
    /// Set once a non-200 status line is seen. The reader then switches to
    /// collecting the body rather than bailing on the spot — see `push`.
    pub(super) error_status: Option<String>,
    error_len: Option<usize>,
    /// The terminal zero-length chunk has been consumed.
    chunks_done: bool,
}

impl Reader {
    /// Feed socket bytes, get back the complete body lines they completed.
    pub(super) fn push(&mut self, bytes: &[u8]) -> Result<Vec<String>> {
        self.raw.extend_from_slice(bytes);
        // Already known to be an error: keep collecting the explanation.
        if self.error_status.is_some() {
            self.collect_error()?;
            return Ok(Vec::new());
        }
        if self.header_end.is_none() {
            let Some(pos) = find(&self.raw, b"\r\n\r\n") else {
                return Ok(Vec::new());
            };
            let head = String::from_utf8_lossy(&self.raw[..pos]).into_owned();
            let status = head.lines().next().unwrap_or_default().to_string();
            if !status_is_success(&status) {
                // Do NOT bail here. The body carries the server's own
                // explanation — including the actionable hint — and at this
                // point it may not have arrived yet: headers can land in a read
                // of their own. Bailing on the status line alone is why every
                // benchmark failure against a modelless server read
                // `endpoint returned "HTTP/1.1 503 Service Unavailable"` and
                // nothing about how to fix it.
                self.error_status = Some(status);
                self.error_len = content_length(&head);
                // Errors are framed like anything else — this server sends
                // them chunked — so the framing must be decoded before the
                // body is JSON. Reading it raw yields `14A\r\n{...}\r\n0` and
                // a parse failure that looks exactly like "no body".
                self.chunked = is_chunked(&head);
                self.header_end = Some(pos + 4);
                self.consumed = pos + 4;
                self.collect_error()?;
                return Ok(Vec::new());
            }
            self.chunked = is_chunked(&head);
            self.header_end = Some(pos + 4);
            self.consumed = pos + 4;
        }
        if self.chunked {
            self.decode_chunks()?;
        } else {
            self.body.extend_from_slice(&self.raw[self.consumed..]);
            self.consumed = self.raw.len();
        }
        Ok(self.take_lines())
    }

    /// Accumulate an error body, decoding its framing, and bail once complete.
    ///
    /// "Complete" is the terminal chunk when chunked, the declared
    /// `Content-Length` when there is one, and the cap otherwise. A body with
    /// none of those is reported at EOF by [`Reader::finish`], so a server that
    /// never says how much it is sending and never closes cannot wedge the run.
    fn collect_error(&mut self) -> Result<()> {
        if self.chunked {
            // A malformed chunk header in an error body is not worth failing
            // twice over: report the status we already have.
            if self.decode_chunks().is_err() {
                return self.fail();
            }
        } else {
            let incoming = &self.raw[self.consumed..];
            let keep = (MAX_ERROR_BODY - self.body.len().min(MAX_ERROR_BODY)).min(incoming.len());
            self.body.extend_from_slice(&incoming[..keep]);
            self.consumed = self.raw.len();
        }
        let have = self.body.len();
        let done =
            self.chunks_done || self.error_len.is_some_and(|n| have >= n) || have >= MAX_ERROR_BODY;
        if done { self.fail() } else { Ok(()) }
    }

    /// Report a pending error, whatever arrived. Called at EOF.
    pub(super) fn finish(&self) -> Result<()> {
        if self.error_status.is_some() {
            self.fail()?;
        }
        Ok(())
    }

    fn fail(&self) -> Result<()> {
        let status = self.error_status.clone().unwrap_or_default();
        let text = String::from_utf8_lossy(&self.body);
        match message_from_body(text.trim()) {
            Some(m) => bail!("endpoint returned {status:?}: {m}"),
            None => bail!("endpoint returned {status:?}"),
        }
    }

    /// Pull every whole chunk currently buffered into `body`. A partial chunk
    /// stays in `raw` until the rest arrives — which is exactly the case naive
    /// line-splitting gets wrong.
    fn decode_chunks(&mut self) -> Result<()> {
        loop {
            let rest = &self.raw[self.consumed..];
            let Some(nl) = find(rest, b"\r\n") else {
                return Ok(());
            };
            let header = std::str::from_utf8(&rest[..nl]).unwrap_or("");
            // A chunk-extension (`;name=value`) is legal after the size.
            let size_hex = header.split(';').next().unwrap_or("").trim();
            let size = usize::from_str_radix(size_hex, 16)
                .map_err(|_| anyhow!("malformed chunk size {size_hex:?}"))?;
            let start = nl + 2;
            let end = start + size;
            // +2 for the CRLF that terminates the chunk data.
            if rest.len() < end + 2 {
                return Ok(());
            }
            let data = &rest[start..end];
            let keep = if self.error_status.is_some() {
                (MAX_ERROR_BODY - self.body.len().min(MAX_ERROR_BODY)).min(data.len())
            } else {
                data.len()
            };
            self.body.extend_from_slice(&data[..keep]);
            self.consumed += end + 2;
            if size == 0 {
                self.chunks_done = true;
                return Ok(());
            }
        }
    }

    fn take_lines(&mut self) -> Vec<String> {
        let mut out = Vec::new();
        let mut start = 0;
        while let Some(nl) = find(&self.body[start..], b"\n") {
            let end = start + nl;
            out.push(
                String::from_utf8_lossy(&self.body[start..end])
                    .trim()
                    .to_string(),
            );
            start = end + 1;
        }
        self.body.drain(..start);
        out
    }
}
