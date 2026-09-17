// SPDX-License-Identifier: AGPL-3.0-only

//! HTTP byte accounting for the TUI Server Stats panel.
//!
//! Request side: `Content-Length` when the client declares it (all real Atlas
//! clients do — JSON bodies). Response side: a wrapping `http_body::Body`
//! counts frames as they are actually written, so streaming/SSE responses —
//! where `Content-Length` does not exist — are counted correctly.

use std::pin::Pin;
use std::task::{Context, Poll};

use axum::body::Body;
use axum::http::Request;
use axum::middleware::Next;
use axum::response::Response;

use crate::metrics::{HTTP_BYTES_IN, HTTP_BYTES_OUT};

/// Counting wrapper over the response body.
struct CountedBody {
    inner: Body,
}

impl http_body::Body for CountedBody {
    type Data = axum::body::Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
        // axum::body::Body is Unpin, so structural pinning is unnecessary.
        let this = self.get_mut();
        let polled = Pin::new(&mut this.inner).poll_frame(cx);
        if let Poll::Ready(Some(Ok(frame))) = &polled
            && let Some(data) = frame.data_ref()
        {
            HTTP_BYTES_OUT.inc_by(data.len() as u64);
        }
        polled
    }
}

/// Axum middleware: count request bytes in, wrap the response to count out.
pub(crate) async fn byte_count_middleware(req: Request<Body>, next: Next) -> Response {
    if let Some(len) = req
        .headers()
        .get(axum::http::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
    {
        HTTP_BYTES_IN.inc_by(len);
    }
    let resp = next.run(req).await;
    let (parts, body) = resp.into_parts();
    Response::from_parts(parts, Body::new(CountedBody { inner: body }))
}
