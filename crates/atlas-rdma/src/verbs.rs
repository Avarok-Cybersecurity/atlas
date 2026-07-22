// SPDX-License-Identifier: AGPL-3.0-only
//
// Safe-ish Rust wrapper over the one-sided RDMA READ C-shim (`rdma_shim.c`).
// Compiled only where the shim is (`cfg(atlas_rdma_verbs)`, emitted by build.rs
// on Linux with rdma-core). Deliberately CUDA-free: the peer server (non-cuda)
// registers its store here, the client tier (cuda) registers its pinned arena.
//
// One `Verbs` == one RC QP + its device context. The QP-identity exchange
// (qpn/psn/gid) rides the TCP control channel; `connect` drives INIT->RTR->RTS.
// The client posts `IBV_WR_RDMA_READ`s with `post_read` and reaps them with
// `poll` (blocking busy-poll); the server QP only reaches RTS to respond, its
// CPU idle (the point of one-sided reads).

use anyhow::{Result, bail};
use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};

/// A 16-byte RoCEv2 GID, exchanged verbatim over the control channel.
pub type Gid = [u8; 16];

#[repr(C)]
struct RsConn {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn rs_create(dev_name: *const c_char, gid_idx: c_int) -> *mut RsConn;
    fn rs_destroy(c: *mut RsConn);
    fn rs_qpn(c: *mut RsConn) -> u32;
    fn rs_gid(c: *mut RsConn, out: *mut u8);
    fn rs_reg_mr(
        c: *mut RsConn,
        addr: *mut c_void,
        len: usize,
        flags: c_int,
        lkey: *mut u32,
        rkey: *mut u32,
    ) -> c_int;
    fn rs_connect(
        c: *mut RsConn,
        remote_qpn: u32,
        remote_psn: u32,
        local_psn: u32,
        remote_gid: *const u8,
    ) -> c_int;
    fn rs_post_read(
        c: *mut RsConn,
        local_addr: *mut c_void,
        lkey: u32,
        remote_addr: u64,
        rkey: u32,
        len: u32,
        wr_id: u64,
    ) -> c_int;
    fn rs_post_write(
        c: *mut RsConn,
        local_addr: *mut c_void,
        lkey: u32,
        remote_addr: u64,
        rkey: u32,
        len: u32,
        wr_id: u64,
    ) -> c_int;
    fn rs_poll(c: *mut RsConn, out_wr_id: *mut u64) -> c_int;
}

/// Registration keys returned when a memory region is pinned to the QP's PD.
#[derive(Clone, Copy, Debug)]
pub struct MrKeys {
    pub lkey: u32,
    pub rkey: u32,
}

/// One RC QP over a RoCEv2 device. Not `Clone`; owns the C connection.
pub struct Verbs {
    conn: *mut RsConn,
    /// This QP's send PSN — sent to the peer as its rq_psn.
    local_psn: u32,
}

// The C connection is used single-threaded (owned by the prefetch worker on the
// client, by one accept thread on the server). We move it across the thread that
// owns it; never share it. Marking Send lets the client's tier (which the worker
// thread owns) satisfy `ExpertTier: Send`.
unsafe impl Send for Verbs {}

impl Verbs {
    /// Open `dev_name` (e.g. `roceP2p1s0f1`) at RoCEv2 `gid_idx`, create the RC
    /// QP (INIT). `local_psn` seeds our send sequence — any 24-bit value the two
    /// ends agree on; we exchange it so a fixed constant is unnecessary.
    pub fn create(dev_name: &str, gid_idx: u32, local_psn: u32) -> Result<Self> {
        let dev = CString::new(dev_name).unwrap_or_default();
        // SAFETY: dev is a valid NUL-terminated string for the call's duration.
        let conn = unsafe { rs_create(dev.as_ptr(), gid_idx as c_int) };
        if conn.is_null() {
            bail!(
                "rs_create failed for device '{dev_name}' gid_idx {gid_idx} \
                 (check `ibv_devinfo -d {dev_name}` link state / GID)"
            );
        }
        Ok(Self { conn, local_psn })
    }

    pub fn qpn(&self) -> u32 {
        // SAFETY: conn is a live rs_conn for self's lifetime.
        unsafe { rs_qpn(self.conn) }
    }

    pub fn psn(&self) -> u32 {
        self.local_psn
    }

    pub fn gid(&self) -> Gid {
        let mut g = [0u8; 16];
        // SAFETY: conn live; out buffer is 16 bytes as the shim expects.
        unsafe { rs_gid(self.conn, g.as_mut_ptr()) };
        g
    }

    /// Register `[addr, addr+len)`. `remote_read` = REMOTE_READ only (the
    /// server's read-only store MR); otherwise LOCAL_WRITE (the client's arena,
    /// the HCA's READ landing buffer).
    ///
    /// # Safety
    /// `addr` must point at `len` bytes that outlive this `Verbs` (the MR is
    /// dereg'd on drop, but the NIC may DMA into/out of it until then).
    pub unsafe fn reg_mr(
        &mut self,
        addr: *mut c_void,
        len: usize,
        remote_read: bool,
    ) -> Result<MrKeys> {
        let mut lkey = 0u32;
        let mut rkey = 0u32;
        // SAFETY: conn live; lkey/rkey out params valid; caller upholds addr/len.
        let rc = unsafe {
            rs_reg_mr(
                self.conn,
                addr,
                len,
                remote_read as c_int,
                &mut lkey,
                &mut rkey,
            )
        };
        if rc != 0 {
            bail!("ibv_reg_mr failed (addr {addr:p} len {len} remote_read {remote_read})");
        }
        Ok(MrKeys { lkey, rkey })
    }

    /// Register `[addr, addr+len)` as a READ+WRITE remote region (REMOTE_READ |
    /// REMOTE_WRITE | LOCAL_WRITE) — the KV overflow blade's arena, which peers
    /// both write groups into (offload) and read back (restore).
    ///
    /// # Safety
    /// Same as [`Verbs::reg_mr`]: `addr` must back `len` live bytes outliving self.
    pub unsafe fn reg_mr_rw(&mut self, addr: *mut c_void, len: usize) -> Result<MrKeys> {
        let mut lkey = 0u32;
        let mut rkey = 0u32;
        // flags=3 → REMOTE_READ|REMOTE_WRITE|LOCAL_WRITE (see rdma_shim.c).
        // SAFETY: conn live; out params valid; caller upholds addr/len.
        let rc = unsafe { rs_reg_mr(self.conn, addr, len, 3, &mut lkey, &mut rkey) };
        if rc != 0 {
            bail!("ibv_reg_mr(RW) failed (addr {addr:p} len {len})");
        }
        Ok(MrKeys { lkey, rkey })
    }

    /// Drive INIT->RTR->RTS with the remote QP's identity.
    pub fn connect(&mut self, remote_qpn: u32, remote_psn: u32, remote_gid: &Gid) -> Result<()> {
        // SAFETY: conn live; remote_gid is 16 bytes.
        let rc = unsafe {
            rs_connect(
                self.conn,
                remote_qpn,
                remote_psn,
                self.local_psn,
                remote_gid.as_ptr(),
            )
        };
        match rc {
            0 => Ok(()),
            -1 => bail!("rs_connect: ibv_query_port failed"),
            -2 => bail!("rs_connect: modify_qp -> RTR failed (check MTU/GID/dest_qpn)"),
            -3 => bail!("rs_connect: modify_qp -> RTS failed"),
            other => bail!("rs_connect: unexpected code {other}"),
        }
    }

    /// Post a one-sided READ of `len` bytes from `remote_addr` (`rkey`) into the
    /// local `local_addr` (`lkey`), tagged `wr_id`. Non-blocking; reap with
    /// `poll`.
    ///
    /// # Safety
    /// `local_addr..+len` must lie inside a live MR registered under `lkey`.
    /// The op is asynchronous: that buffer and its MR must stay live and
    /// untouched by the CPU until the matching `wr_id` is reaped by `poll`
    /// (dropping the QP or freeing the buffer before then risks a NIC DMA into
    /// freed memory).
    pub unsafe fn post_read(
        &mut self,
        local_addr: *mut c_void,
        lkey: u32,
        remote_addr: u64,
        rkey: u32,
        len: u32,
        wr_id: u64,
    ) -> Result<()> {
        // SAFETY: conn live; caller upholds local_addr/lkey validity.
        let rc =
            unsafe { rs_post_read(self.conn, local_addr, lkey, remote_addr, rkey, len, wr_id) };
        if rc != 0 {
            bail!("ibv_post_send(RDMA_READ) failed: {rc}");
        }
        Ok(())
    }

    /// Post a one-sided WRITE of `len` bytes from local `local_addr` (`lkey`) to
    /// the remote `remote_addr` (`rkey`), tagged `wr_id`. Non-blocking; reap with
    /// `poll`. Used to offload a K/V group into the peer's RW blade.
    ///
    /// # Safety
    /// `local_addr..+len` must lie inside a live MR registered under `lkey`.
    /// The op is asynchronous: that buffer and its MR must stay live and
    /// unmodified until the matching `wr_id` is reaped by `poll` (the NIC DMAs
    /// from it after this returns).
    pub unsafe fn post_write(
        &mut self,
        local_addr: *mut c_void,
        lkey: u32,
        remote_addr: u64,
        rkey: u32,
        len: u32,
        wr_id: u64,
    ) -> Result<()> {
        // SAFETY: conn live; caller upholds local_addr/lkey validity.
        let rc =
            unsafe { rs_post_write(self.conn, local_addr, lkey, remote_addr, rkey, len, wr_id) };
        if rc != 0 {
            bail!("ibv_post_send(RDMA_WRITE) failed: {rc}");
        }
        Ok(())
    }

    /// Block until one completion arrives; return its `wr_id`. Errors on a
    /// non-success completion status (the NIC's `ibv_wc_status`).
    pub fn poll(&mut self) -> Result<u64> {
        let mut wr_id = 0u64;
        // SAFETY: conn live; wr_id out param valid.
        let rc = unsafe { rs_poll(self.conn, &mut wr_id) };
        if rc == 0 {
            Ok(wr_id)
        } else if rc < 0 {
            bail!("ibv_poll_cq error");
        } else {
            bail!("RDMA completion error: ibv_wc_status {rc}");
        }
    }
}

impl Drop for Verbs {
    fn drop(&mut self) {
        // SAFETY: conn was created by rs_create and not yet destroyed.
        unsafe { rs_destroy(self.conn) };
    }
}
