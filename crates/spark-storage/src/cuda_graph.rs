// SPDX-License-Identifier: AGPL-3.0-only
//
// CUDA graph capture + replay primitive (Phase 4). Captures a sequence of
// stream operations into a single graph and replays them via one
// `cuGraphLaunch` call (~2 µs vs ~5–10 µs per kernel in eager mode).

use anyhow::{Context, Result, bail};

unsafe extern "C" {
    fn cuStreamBeginCapture_v2(stream: u64, mode: u32) -> i32;
    fn cuStreamEndCapture(stream: u64, graph_out: *mut u64) -> i32;
    fn cuGraphInstantiateWithFlags(exec_out: *mut u64, graph: u64, flags: u64) -> i32;
    fn cuGraphLaunch(exec: u64, stream: u64) -> i32;
    fn cuGraphExecDestroy(exec: u64) -> i32;
    fn cuGraphDestroy(graph: u64) -> i32;
}

const CU_STREAM_CAPTURE_MODE_GLOBAL: u32 = 0;

pub struct CapturedStep {
    graph: u64,
    graph_exec: u64,
}

impl CapturedStep {
    /// Capture a sequence of stream operations issued by `body`. Device
    /// pointers passed to launches inside `body` are baked in at capture
    /// time; subsequent `launch` calls reread the same memory locations.
    pub fn capture<F>(stream: u64, body: F) -> Result<Self>
    where
        F: FnOnce() -> Result<()>,
    {
        let s = unsafe { cuStreamBeginCapture_v2(stream, CU_STREAM_CAPTURE_MODE_GLOBAL) };
        if s != 0 {
            bail!("cuStreamBeginCapture_v2 failed: {s}");
        }
        let body_result = body();
        let mut graph = 0u64;
        let s2 = unsafe { cuStreamEndCapture(stream, &mut graph) };
        // Surface body errors *after* end_capture so the stream isn't left
        // mid-capture (which makes any further use of it fail).
        body_result.context("CUDA graph body")?;
        if s2 != 0 {
            bail!("cuStreamEndCapture failed: {s2}");
        }
        let mut exec = 0u64;
        let s3 = unsafe { cuGraphInstantiateWithFlags(&mut exec, graph, 0) };
        if s3 != 0 {
            unsafe { cuGraphDestroy(graph) };
            bail!("cuGraphInstantiateWithFlags failed: {s3}");
        }
        Ok(Self {
            graph,
            graph_exec: exec,
        })
    }

    pub fn launch(&self, stream: u64) -> Result<()> {
        let s = unsafe { cuGraphLaunch(self.graph_exec, stream) };
        if s != 0 {
            bail!("cuGraphLaunch failed: {s}");
        }
        Ok(())
    }
}

impl Drop for CapturedStep {
    fn drop(&mut self) {
        unsafe {
            let _ = cuGraphExecDestroy(self.graph_exec);
            let _ = cuGraphDestroy(self.graph);
        }
    }
}
