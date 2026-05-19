// SPDX-License-Identifier: AGPL-3.0-only
//
// `return_token_ids` drain helper. Kept beside the stream state so the
// drain-on-emit logic lives in one place (SSOT): every client-visible
// chunk pulls the IDs accumulated since the previous emit, guaranteeing
// the per-stream sum equals `usage.completion_tokens`.

use super::state::StreamState;

impl StreamState {
    /// Take the IDs buffered since the last emit when the request opted
    /// into `return_token_ids`; otherwise return empty (the builder
    /// no-ops on empty, so the wire format is unchanged for the default
    /// path). Draining — never cloning — keeps each sampled token
    /// counted exactly once across the stream.
    pub(super) fn take_ids_if(&mut self, on: bool) -> Vec<u32> {
        if on {
            std::mem::take(&mut self.pending_token_ids)
        } else {
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::StreamState;

    #[test]
    fn take_ids_if_drains_only_when_on() {
        let mut st = StreamState::new(false, false);
        st.pending_token_ids = vec![7, 8, 9];

        // Opted out: returns empty, buffer untouched (no token loss).
        assert!(st.take_ids_if(false).is_empty());
        assert_eq!(st.pending_token_ids, vec![7, 8, 9]);

        // Opted in: drains exactly once.
        assert_eq!(st.take_ids_if(true), vec![7, 8, 9]);
        assert!(st.pending_token_ids.is_empty());
        // Second drain is empty — each ID counted at most once.
        assert!(st.take_ids_if(true).is_empty());
    }
}
