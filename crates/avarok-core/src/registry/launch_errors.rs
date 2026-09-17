// SPDX-License-Identifier: AGPL-3.0-only

//! What a failed kernel launch is allowed to say.
//!
//! One home for naming a raw CUfunction handle and for the wording of the two
//! driver failures that carry such a handle. [`KernelNames`] is the reverse map
//! the registry fills while the names are still in scope; the free functions
//! are pure, so every message here is unit-testable on a host with no GPU.

use std::collections::HashMap;
use std::sync::Mutex;

/// Raw CUfunction handle -> `"module::kernel"`, for every function the
/// registry has resolved.
///
/// A `RawCudaFunc` is a bare driver pointer: once it reaches
/// [`crate::registry::AvarokRegistry::launch_on_stream`] there is nothing left
/// in it that says what the kernel was, which is why a failed launch used to
/// report only a grid and a CUresult. This is the reverse map, and it is
/// written exactly once per kernel in
/// [`crate::registry::AvarokRegistry::raw_function_cached`] (model init) and
/// read ONLY from a failure path, so a launch pays nothing for it: no lookup,
/// no lock, no branch on the hot path.
pub(super) struct KernelNames(Mutex<HashMap<u64, String>>);

impl KernelNames {
    pub(super) fn new() -> Self {
        KernelNames(Mutex::new(HashMap::new()))
    }

    /// Remember what a freshly minted handle IS.
    pub(super) fn record(&self, handle: u64, label: String) {
        self.lock().insert(handle, label);
    }

    /// Forget every handle. The names describe pointers that are about to
    /// become stale, so they go with them.
    pub(super) fn clear(&self) {
        self.lock().clear();
    }

    /// How a handle should be named in a diagnostic.
    pub(super) fn label(&self, handle: u64) -> String {
        let guard = self.lock();
        kernel_label(guard.get(&handle).map(String::as_str), handle)
    }

    /// A poisoned lock here means a previous caller panicked while holding a
    /// map of diagnostic strings. Nothing about it is unsafe to keep using.
    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<u64, String>> {
        self.0.lock().unwrap_or_else(|poison| poison.into_inner())
    }
}

/// Name a kernel handle for a diagnostic.
///
/// `name` is `None` for a handle the registry never minted (nothing in Avarok
/// produces one today, but `RawCudaFunc` is a public tuple struct anyone can
/// construct), in which case the pointer is all there is to report.
///
/// Pure: no driver call, so the wording is unit-testable on a host with no GPU.
pub fn kernel_label(name: Option<&str>, handle: u64) -> String {
    match name {
        Some(name) => format!("{name} (fn@{handle:#x})"),
        None => format!("<unregistered kernel> (fn@{handle:#x})"),
    }
}

/// The text of a `cuLaunchKernel` failure. `label` comes from
/// [`crate::registry::AvarokRegistry::func_label`] and `err_text` from
/// [`crate::registry::cuda_error_text`], both of which are resolved by the
/// caller so this stays pure and testable.
pub fn launch_failure_message(
    label: &str,
    err_text: &str,
    grid: [u32; 3],
    block: [u32; 3],
    shared_mem: u32,
) -> String {
    format!(
        "cuLaunchKernel failed for {label}: {err_text} \
         (grid=[{},{},{}], block=[{},{},{}], shared_mem={shared_mem})",
        grid[0], grid[1], grid[2], block[0], block[1], block[2]
    )
}

/// The text of a `cuFuncSetAttribute(MAX_DYNAMIC_SHARED=..)` failure. Same
/// division of labour as [`launch_failure_message`].
pub fn func_attribute_failure_message(label: &str, shared_mem: u32, err_text: &str) -> String {
    format!("cuFuncSetAttribute(MAX_DYNAMIC_SHARED={shared_mem}) failed for {label}: {err_text}")
}
