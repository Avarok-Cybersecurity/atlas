// SPDX-License-Identifier: AGPL-3.0-only
// Portions derived from Cinference (https://github.com/satellitedown/cinference, commit 383e5db), itself derived from NInfer by Neroued and contributors (https://github.com/Neroued/ninfer), Apache License 2.0. Source: src/core/decode_graph.cpp:62-80 (`DecodeGraphDefinition::topology_signature`). Changes: translated from C++ over the CUDA runtime API to Rust over the driver API (`cuGraphGetNodes`, `cuGraphNodeGetType`, `cuGraphKernelNodeGetParams` resolved through `cuGetProcAddress`), a library kernel's handle stands in when a node has no function, and the hash itself lives in `gpu/port_ext.rs`, by Atlas contributors.

//! Responsibility: CUDA graph TEMPLATES for [`AvarokCudaBackend`] (port sprint,
//! contract 3.7.1): ending a capture without discarding the graph, instantiating a
//! kept graph, its topology signature, in-place exec update (`cuGraphExecUpdate`)
//! and graph upload.
//!
//! `end_capture_cu` (in `gpu_impl_graph.rs`) instantiates and destroys the graph at
//! once, and it is unchanged: every existing capture site keeps that path. These
//! bodies are reached only through the template methods, which only port code with
//! its row on calls.
//!
//! Exec update and the signature need four entry points that are resolved at run
//! time (`port_ffi`); under SCALE (`avarok_scale`) none is, so `supports` is false,
//! an update answers `Ok(false)` and the signature refuses. Capture, instantiate and
//! destroy use the link-time imports `end_capture_cu` already uses, so templates
//! themselves work on every target.
//!
//! The safety contract at the top of `gpu_impl.rs` applies to every driver call here.

use anyhow::{Result, bail};
use avarok_core::registry::cuda_error_text;

use super::port_ffi::{self, ExecUpdateInfo, KernelNodeParams};
use super::{AvarokCudaBackend, cuGraphDestroy, cuStreamEndCapture};
use crate::gpu::port_ext::{self, NODE_TYPE_KERNEL};
use crate::gpu::{GraphHandle, GraphTemplate};

/// `CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE`: the update was refused (topology or a node
/// change the driver cannot apply in place); the exec must be treated as stale.
const CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE: i32 = 910;

impl AvarokCudaBackend {
    /// End the capture on `stream` and keep the graph.
    pub(super) fn end_capture_graph_cu(&self, stream: u64) -> Result<GraphTemplate> {
        let mut graph: u64 = 0;
        let status = unsafe { cuStreamEndCapture(stream, &mut graph) };
        if status != 0 {
            // An invalidated capture can still hand back a partial graph; never leak it.
            if graph != 0 {
                unsafe { cuGraphDestroy(graph) };
            }
            bail!("cuStreamEndCapture failed: status {status}");
        }
        if graph == 0 {
            bail!("cuStreamEndCapture returned no graph");
        }
        Ok(GraphTemplate(graph))
    }

    /// Instantiate `t` into a new exec. `t` stays the caller's (destroyed or not) on both
    /// outcomes; this is the only difference from the instantiate inside `end_capture_cu`.
    pub(super) fn graph_instantiate_cu(&self, t: &GraphTemplate) -> Result<GraphHandle> {
        if t.0 == 0 {
            bail!("graph_instantiate: null template");
        }
        let mut exec: u64 = 0;
        // Same symbol choice as `end_capture_cu`: SCALE exports only `cuGraphInstantiate`.
        #[cfg(not(avarok_scale))]
        let status = unsafe { super::cuGraphInstantiateWithFlags(&mut exec, t.0, 0) };
        #[cfg(avarok_scale)]
        let status = unsafe { super::cuGraphInstantiate(&mut exec, t.0, 0) };
        if status != 0 {
            bail!("cuGraphInstantiate failed: status {status}");
        }
        Ok(GraphHandle(exec))
    }

    /// End the capture, instantiate, and keep the graph as a template.
    pub(super) fn end_capture_template_cu(
        &self,
        stream: u64,
    ) -> Result<(GraphHandle, GraphTemplate)> {
        let t = self.end_capture_graph_cu(stream)?;
        match self.graph_instantiate_cu(&t) {
            Ok(exec) => Ok((exec, t)),
            Err(e) => {
                unsafe { cuGraphDestroy(t.0) };
                Err(e)
            }
        }
    }

    pub(super) fn graph_template_destroy_cu(&self, t: GraphTemplate) -> Result<()> {
        if t.0 == 0 {
            return Ok(());
        }
        let status = unsafe { cuGraphDestroy(t.0) };
        if status != 0 && !avarok_core::registry::is_teardown_noop(status) {
            bail!("cuGraphDestroy failed: {}", cuda_error_text(status));
        }
        Ok(())
    }

    /// Node count plus FNV-1a over `(node type, kernel function)` in node order.
    pub(super) fn graph_topology_signature_cu(&self, t: &GraphTemplate) -> Result<u64> {
        let d = port_ffi::driver();
        let (Some(get_nodes), Some(get_type), Some(get_params)) = (
            d.graph_get_nodes,
            d.graph_node_get_type,
            d.kernel_node_get_params,
        ) else {
            bail!(
                "graph_topology_signature: graph node queries are not available from this driver"
            );
        };
        if t.0 == 0 {
            bail!("graph_topology_signature: null template");
        }
        let mut count: usize = 0;
        let status = unsafe { get_nodes(t.0, std::ptr::null_mut(), &mut count) };
        if status != 0 {
            bail!(
                "cuGraphGetNodes (count) failed: {}",
                cuda_error_text(status)
            );
        }
        let mut nodes = vec![0u64; count];
        if count > 0 {
            let mut filled = count;
            let status = unsafe { get_nodes(t.0, nodes.as_mut_ptr(), &mut filled) };
            if status != 0 {
                bail!("cuGraphGetNodes failed: {}", cuda_error_text(status));
            }
            nodes.truncate(filled.min(count));
        }
        let mut typed = Vec::with_capacity(nodes.len());
        for &node in &nodes {
            let mut ty: u32 = u32::MAX;
            let status = unsafe { get_type(node, &mut ty) };
            if status != 0 {
                bail!("cuGraphNodeGetType failed: {}", cuda_error_text(status));
            }
            let func = if ty == NODE_TYPE_KERNEL {
                let mut p = KernelNodeParams::EMPTY;
                let status = unsafe { get_params(node, &mut p) };
                if status != 0 {
                    bail!(
                        "cuGraphKernelNodeGetParams failed: {}",
                        cuda_error_text(status)
                    );
                }
                // A node launched from a library kernel (`CUkernel`) may report no
                // function; its kernel handle identifies it just as well.
                if p.func != 0 { p.func } else { p.kern }
            } else {
                0
            };
            typed.push((ty, func));
        }
        Ok(port_ext::topology_signature(typed))
    }

    /// `cuGraphExecUpdate`. `Ok(false)` on refusal or when the entry point is missing.
    pub(super) fn graph_exec_update_cu(
        &self,
        exec: GraphHandle,
        t: &GraphTemplate,
    ) -> Result<bool> {
        let Some(update) = port_ffi::driver().graph_exec_update else {
            return Ok(false);
        };
        if exec.0 == 0 || t.0 == 0 {
            bail!("graph_exec_update: null exec or template");
        }
        let mut info = ExecUpdateInfo::EMPTY;
        match unsafe { update(exec.0, t.0, &mut info) } {
            0 => Ok(true),
            CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE => {
                tracing::debug!(
                    "cuGraphExecUpdate refused (CUgraphExecUpdateResult {})",
                    info.result
                );
                Ok(false)
            }
            s => bail!("cuGraphExecUpdate failed: {}", cuda_error_text(s)),
        }
    }

    /// `cuGraphUpload`; a no-op when the entry point is missing (upload only moves work
    /// off the first launch, it never changes what a launch does).
    pub(super) fn graph_upload_cu(&self, exec: GraphHandle, stream: u64) -> Result<()> {
        let Some(upload) = port_ffi::driver().graph_upload else {
            return Ok(());
        };
        let status = unsafe { upload(exec.0, stream) };
        if status != 0 {
            bail!("cuGraphUpload failed: {}", cuda_error_text(status));
        }
        Ok(())
    }

    pub(super) fn supports_graph_exec_update_cu(&self) -> bool {
        port_ffi::driver().graph_update_ready()
    }
}
