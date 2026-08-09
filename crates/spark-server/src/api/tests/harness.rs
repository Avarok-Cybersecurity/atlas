// SPDX-License-Identifier: AGPL-3.0-only

//! A streaming harness for the content sanitizer.
//!
//! `sanitize_content_chunk` holds back up to `tag_max - 1` trailing bytes
//! so a tag split across a chunk boundary can still fuse. That makes
//! per-call output an implementation detail: the same legitimate byte can
//! land in this call's return value or the next one's, depending on how
//! long the longest marker happens to be.
//!
//! The tests here therefore assert on the WHOLE stream — every chunk plus
//! the end-of-stream flush — which is what the client actually receives.
//! Asserting per-call output would pin the tail-retention arithmetic
//! rather than the suppression behaviour.

use crate::api::sanitizer::sanitize_content_chunk;
use crate::api::stream_guards::flush_content_sanitizer;
use crate::tool_parser::LeakMarkers;

pub(super) struct Stream<'a> {
    markers: &'a LeakMarkers,
    tag_scan_buf: String,
    suppressing: bool,
    inside_envelope: bool,
    out: String,
}

impl<'a> Stream<'a> {
    pub(super) fn new(markers: &'a LeakMarkers) -> Self {
        Self {
            markers,
            tag_scan_buf: String::new(),
            suppressing: false,
            inside_envelope: false,
            out: String::new(),
        }
    }

    /// Feed one chunk; returns what this call alone emitted.
    pub(super) fn feed(&mut self, chunk: &str) -> String {
        let emitted = sanitize_content_chunk(
            chunk,
            &mut self.tag_scan_buf,
            &mut self.suppressing,
            &mut self.inside_envelope,
            self.markers,
        );
        self.out.push_str(&emitted);
        emitted
    }

    /// Feed the whole text in `size`-byte slices, mimicking token-sized
    /// SSE deltas. Splits on char boundaries.
    pub(super) fn feed_chunked(&mut self, text: &str, size: usize) {
        let mut start = 0;
        while start < text.len() {
            let mut end = (start + size).min(text.len());
            while !text.is_char_boundary(end) {
                end += 1;
            }
            self.feed(&text[start..end]);
            start = end;
        }
    }

    /// Close the stream and return everything the client received.
    pub(super) fn finish(mut self) -> String {
        let tail =
            flush_content_sanitizer(&mut self.tag_scan_buf, &mut self.suppressing, self.markers);
        self.out.push_str(&tail);
        self.out
    }

    pub(super) fn suppressing(&self) -> bool {
        self.suppressing
    }

    pub(super) fn inside_envelope(&self) -> bool {
        self.inside_envelope
    }

    pub(super) fn buffered(&self) -> &str {
        &self.tag_scan_buf
    }
}
