// SPDX-License-Identifier: AGPL-3.0-only

//! Read-side data plane for the dashboard: pure pollers over process-global
//! state (prometheus counters, scheduler snapshot, kernel audit, HF cache).
//! Nothing here touches the scheduler thread's locals.

pub mod kernels;
pub mod library;
pub mod metrics_poll;
