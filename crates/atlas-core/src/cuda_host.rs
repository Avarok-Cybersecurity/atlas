// SPDX-License-Identifier: AGPL-3.0-only

//! The **process-scoped** half of the CUDA runtime: the context, the stream,
//! and the teardown check that guards a model swap.
//!
//! Split from [`crate::registry`] because the two have different lifetimes and
//! conflating them is what made in-process model swapping impossible. The host
//! is derived from the *device* and lives for the process; the module registry
//! is derived from the *checkpoint* and lives for one model.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaStream};

use crate::error::{AtlasError, Result};
use crate::registry::AtlasRegistry;

/// The CUDA context and stream — **process-scoped, on purpose**.
///
/// Creating and destroying a CUDA context is precisely what in-process model
/// swapping exists to avoid: it is slow, it invalidates every handle in the
/// process, and on GB10 UVM it is the operation with the worst failure modes.
/// The context is derived from the *device*, not from the checkpoint, so one
/// model's context serves the next one unchanged.
pub struct CudaHost {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    ordinal: usize,
}

impl CudaHost {
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }
}

// SAFETY: as for `AtlasRegistry` below — these are handles into a context that
// outlives every reader, and CUDA serialises the work itself via streams.
unsafe impl Send for CudaHost {}
unsafe impl Sync for CudaHost {}

/// The process's CUDA host.
///
/// **STATIC, AND THIS IS THE CASE WHERE IT MUST BE.** The argument, in full,
/// because the standing rule is that every surviving static earns one:
///
/// 1. **It is derived from the process and the device, not from any model.** A
///    CUDA context belongs to a (process, GPU) pair. Nothing about a checkpoint
///    reaches it, so it cannot go stale when the model changes — the property
///    that makes every other static here a hazard simply does not apply.
/// 2. **The driver enforces the singleton anyway.** There is one primary
///    context per device per process. Holding two handles would not give two
///    contexts; it would give two names for the same one, with the ownership
///    question moved from the type system to a convention.
/// 3. **Its whole purpose is to outlive every model.** Not destroying and
///    recreating the context across a swap IS the feature — that operation is
///    slow, invalidates every handle in the process, and has the worst failure
///    modes on GB10 UVM. A context propagated per-model would have to be
///    recreated per-model, which is precisely what is being avoided.
/// 4. **Propagating it changes nothing it does not already guarantee.** Every
///    consumer reaches it through the `AtlasRegistry` it already holds
///    (`registry.host()`), so the *usage* is propagated. This static is only
///    the place the one context is created and found.
///
/// Rebinding to a different ordinal is an error rather than a silent no-op —
/// that was the bug in the previous `get_or_init(ordinal, kernel_blobs)`.
static HOST: OnceLock<std::result::Result<Arc<CudaHost>, String>> = OnceLock::new();

/// Get (or create) the process CUDA host on `ordinal`.
///
/// The ordinal is fixed by the first call — a process serves one GPU, and a
/// later call asking for a different one is a bug worth reporting rather than
/// silently ignoring, which is exactly what the old `get_or_init` did with its
/// `kernel_blobs` argument.
pub fn host(ordinal: usize) -> Result<Arc<CudaHost>> {
    let result = HOST.get_or_init(|| {
        let ctx = CudaContext::new(ordinal).map_err(|e| format!("{e}"))?;
        let stream = ctx.new_stream().map_err(|e| format!("{e}"))?;
        Ok(Arc::new(CudaHost {
            ctx,
            stream,
            ordinal,
        }))
    });
    match result {
        Ok(h) if h.ordinal == ordinal => Ok(h.clone()),
        Ok(h) => Err(AtlasError::ModuleLoad(format!(
            "CUDA host already bound to GPU {} — cannot rebind to {ordinal}",
            h.ordinal
        ))),
        Err(msg) => Err(AtlasError::ModuleLoad(msg.clone())),
    }
}

/// Release a registry, refusing to pretend if anything still holds a handle.
///
/// This is the **missed-propagation detector**. The whole hazard of in-process
/// swapping is a reference to the old model surviving somewhere nobody
/// remembered to scope; `Arc::strong_count` turns that from silent wrong output
/// into a named error at teardown, before the next model loads.
pub fn release(registry: Arc<AtlasRegistry>) -> Result<()> {
    let outstanding = Arc::strong_count(&registry) - 1;
    if outstanding > 0 {
        return Err(AtlasError::ModuleLoad(format!(
            "cannot release the kernel modules: {outstanding} handle(s) are still live. \
             Something is holding the previous model's registry — find it before swapping, \
             or the next model will run against unloaded modules."
        )));
    }
    let mut owned = Arc::try_unwrap(registry).map_err(|_| {
        AtlasError::ModuleLoad("registry handle count changed during release".to_string())
    })?;
    let failures = owned.unload_raw();
    if !failures.is_empty() {
        return Err(AtlasError::ModuleLoad(format!(
            "{} module(s) failed to unload: {}",
            failures.len(),
            failures.join("; ")
        )));
    }
    Ok(())
}
